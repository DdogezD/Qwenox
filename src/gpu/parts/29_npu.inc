// 29_npu.inc — XDNA2 NPU block-float GEMM offload（仅 Windows；Linux 不定义
// QWENOX_NPU，本文件整体编译为空）。
//
// 编译开关：QWENOX_NPU=1（build_win.sh 检测到 $NPUASSETS 的 XRT 资产时加）。
// 运行开关：QWENOX_NPU=1 环境变量，默认关——同一份二进制可 A/B。exe 对
// xrt_coreutil.dll 走 delay-load，没装 NPU 驱动的机器照常运行（卸载不启用）。
//
// NPU kernel 固定形状 M=4096（token 行）x K=2560 x N=1024，操作数是
// v8bfp16ebs8 块浮点（9 字节 = 1 共享指数 + 8 个 int8 尾数，value =
// q * 2^(e-127) / 64）。缓冲全部是 hipHostMalloc(Mapped) 固定页，以 userptr
// 导入 XRT：GPU 原地编解码、NPU 直接 DMA，全程零拷贝。
//
// 提交模型照抄 gufo 的 NpuGemm：Submit/Wait 用 hipLaunchHostFunc 挂到 g_str
// 上，真正的 XRT 调用都在专用 worker 线程里做，NPU 计算期间 GPU 继续跑
// 不依赖其结果的后续 kernel。prefill 不进 CUDA graph，host func 安全。
//
// 参考实现：gufo（github.com/mikealanni/gufo）的 npu_gemm.cpp 与
// kernels.hip.cpp 的 Npu* 系列 kernel；Windows 侧的 XRT 用法在本仓库
// tools/npuprobe/npuio.hip 里验证过（必须用 System32 的 xrt_coreutil.dll；
// instr BO 用 group_id(1)、数据 BO 用 group_id(3)）。

#ifdef QWENOX_NPU
#ifndef _WIN32
#error "QWENOX_NPU is Windows-only"
#endif

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "npu_bins.inc"  // kXclbin/kXclbinLen + kInsts/kInstsLen（build_win.sh 生成）

// ---- GPU 侧编解码 kernel -----------------------------------------------------
// tile 几何：A 128x64 行主（2x1 super block of 8x8 行主块），B 列主
// （1x2 super block of 8x8 列主块），C 512x128 行主（2x2 super block）。
constexpr int kNpuTileM = 128, kNpuTileK = 64, kNpuTileN = 128;
constexpr int kNpuTileRowsC = 4 * kNpuTileM;  // 512

__device__ inline void npu_enc_group(const float* v, uint8_t* out) {
  unsigned me = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) me = max(me, (__float_as_uint(v[i]) >> 23) & 0xFFu);
  // 2^(133 - max_exp)：组内尾数单位的精确倒数
  const float inv = __uint_as_float(me >= 7 ? (260u - me) << 23 : 0u);
  out[0] = (uint8_t)me;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[1 + i] = (uint8_t)(int8_t)fminf(127.f, fmaxf(-128.f, rintf(v[i] * inv)));
}

// A 编码：f32 x（行距 ldx）-> A 流。先按 bf16 舍入（对齐 GPU GEMM 的输入精
// 度），再块浮点量化。一个 block 处理一个 128x64 tile；行 >= m_valid 编零。
__global__ void __launch_bounds__(256)
    k_npu_enc_a(const float* __restrict__ x, uint8_t* __restrict__ out,
                unsigned m_valid, size_t ldx) {
  __shared__ __align__(16) float tile[kNpuTileM * kNpuTileK];
  __shared__ __align__(16) uint8_t staged[kNpuTileM * kNpuTileK / 8 * 9];
  const unsigned tiles_k = 2560 / kNpuTileK;
  const unsigned tile_row = blockIdx.x / tiles_k, tile_col = blockIdx.x % tiles_k;
  const unsigned t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned row = (t >> 3) + 32 * j, vec = t & 7;
    const unsigned grow = tile_row * kNpuTileM + row;
    float4 lo = {0, 0, 0, 0}, hi = {0, 0, 0, 0};
    if (grow < m_valid) {
      const float* src = x + (size_t)grow * ldx + tile_col * kNpuTileK + vec * 8;
      lo = *(const float4*)src;
      hi = *(const float4*)(src + 4);
    }
    *(float4*)&tile[row * kNpuTileK + vec * 8] = lo;
    *(float4*)&tile[row * kNpuTileK + vec * 8 + 4] = hi;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned gi = t + 256 * j;
    const unsigned rib = gi & 7, block = (gi >> 3) & 1, sc = (gi >> 4) & 7, sr = gi >> 7;
    const unsigned row = sr * 16 + block * 8 + rib;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = __bfloat162float(__float2bfloat16(tile[row * kNpuTileK + sc * 8 + i]));
    npu_enc_group(v, staged + gi * 9);
  }
  __syncthreads();
  const uint4* s = (const uint4*)staged;
  uint4* d = (uint4*)(out + (size_t)blockIdx.x * (kNpuTileM * kNpuTileK / 8 * 9));
  for (unsigned i = t; i < kNpuTileM * kNpuTileK / 8 * 9 / 16; i += 256) d[i] = s[i];
}

// B 编码：bf16 权重 [n_valid][2560]（行 n = 输出列 n）-> B 流，行补零到 1024。
__global__ void k_npu_enc_w(const uint16_t* __restrict__ w, uint8_t* __restrict__ out,
                            unsigned n_valid) {
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (size_t)2560 * 1024 / 8) return;
  constexpr unsigned kGpt = kNpuTileK * kNpuTileN / 8;  // 1024
  const size_t tile = g / kGpt;
  const unsigned gi = (unsigned)(g % kGpt);
  const unsigned tiles_k = 2560 / kNpuTileK;
  const unsigned tile_col = (unsigned)(tile / tiles_k), tile_row = (unsigned)(tile % tiles_k);
  const unsigned cib = gi & 7, block = (gi >> 3) & 1, sr = (gi >> 4) & 7, sc2 = gi >> 7;
  const unsigned kk = tile_row * kNpuTileK + sr * 8;
  const unsigned nn = tile_col * kNpuTileN + (sc2 * 2 + block) * 8 + cib;
  float v[8] = {};
  if (nn < n_valid) {
    const uint16_t* src = w + (size_t)nn * 2560 + kk;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __uint_as_float((unsigned)src[i] << 16);
  }
  npu_enc_group(v, out + g * 9);
}

// C 解码：C 流（4096x1024）-> f32 行（行距 out_pitch，写到 col 起始处）。
// 只写 row < m_valid 且 col < n_valid 的元素。out_pitch 与 col 都是 8 的倍
// 数、out 16B 对齐，直接走 float4。
__global__ void k_npu_dec_c(const uint8_t* __restrict__ in, float* __restrict__ out,
                            unsigned m_valid, unsigned n_valid, size_t out_pitch) {
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (size_t)4096 * 1024 / 8) return;
  constexpr unsigned kGpt = kNpuTileRowsC * kNpuTileN / 8;  // 8192
  const size_t tile = g / kGpt;
  const unsigned gi = (unsigned)(g % kGpt);
  const unsigned tiles_n = 1024 / kNpuTileN;
  const unsigned tile_row = (unsigned)(tile / tiles_n), tile_col = (unsigned)(tile % tiles_n);
  const unsigned rib = gi & 7, bc = (gi >> 3) & 1, br = (gi >> 4) & 1, sc = (gi >> 5) & 7,
                 sr = gi >> 8;
  const size_t row = (size_t)tile_row * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
  const unsigned col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
  if (row >= m_valid || col >= n_valid) return;
  const uint8_t* p = in + g * 9;
  const float mult = ldexpf(1.f, (int)p[0] - 127) / 64.f;
  float* dst = out + row * out_pitch + col;
  float4 r[2];
  float* rf = (float*)r;
#pragma unroll
  for (int i = 0; i < 8; ++i) rf[i] = (float)(int8_t)p[1 + i] * mult;
  *(float4*)dst = r[0];
  *(float4*)(dst + 4) = r[1];
}

// ---- host 侧 NpuGemm --------------------------------------------------------
// 与 gufo npu_gemm.cpp 同构：worker 线程独占 XRT 调用，Submit/Wait 是
// hipLaunchHostFunc；至多 4 个 run 在飞（驱动规格队列深度）；run 超时
// （默认 8000ms，QWENOX_NPU_TIMEOUT_MS 覆盖）置 failed，此后结果无效；
// keepalive 线程每 1s 一个小任务防止 NPU 挂起（唤醒要 ~35ms）。
class NpuGemm {
public:
  static constexpr unsigned kM = 4096, kK = 2560, kN = 1024;
  static constexpr size_t kABytes = (size_t)kM * kK / 8 * 9;
  static constexpr size_t kBBytes = (size_t)kK * kN / 8 * 9;
  static constexpr size_t kCBytes = (size_t)kM * kN / 8 * 9;

  struct Buf {
    void* host = nullptr;
    void* dev = nullptr;  // 同一块内存的 GPU 指针
    size_t bytes = 0;
    xrt::bo bo;           // host 页的 userptr 导入
  };

  static NpuGemm* Create(std::string& err) {
    if (!LoadLibraryW(L"xrt_coreutil.dll")) {  // delay-load 保护
      err = "xrt_coreutil.dll 加载失败（NPU 驱动未安装）";
      return nullptr;
    }
    std::unique_ptr<NpuGemm> g(new NpuGemm);
    try {
      const std::string xp = write_xclbin();
      if (xp.empty()) {
        err = "写出内嵌 xclbin 到 %TEMP% 失败";
        return nullptr;
      }
      g->dev_ = xrt::device(0u);
      const xrt::xclbin xb{xp};
      g->dev_.register_xclbin(xb);
      g->ctx_ = xrt::hw_context(g->dev_, xb.get_uuid());
      g->kern_ = xrt::kernel(g->ctx_, xb.get_kernels().at(0).get_name());
      g->words_ = kInstsLen / 4;
      g->instr_ = xrt::bo(g->dev_, kInstsLen, xrt::bo::flags::cacheable, g->kern_.group_id(1));
      memcpy(g->instr_.map<void*>(), kInsts, kInstsLen);
      g->instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      g->grp_ = g->kern_.group_id(3);
    } catch (const std::exception& e) {
      err = std::string("NPU setup: ") + e.what();
      return nullptr;
    }
    g->hb_[0] = g->NewBuffer(kABytes);
    g->hb_[1] = g->NewBuffer(kBBytes);
    g->hb_[2] = g->NewBuffer(kCBytes);
    if (g->hb_[0] < 0 || g->hb_[1] < 0 || g->hb_[2] < 0) {
      err = "NPU heartbeat 缓冲分配失败";
      return nullptr;
    }
    memset(g->bufs_[g->hb_[0]].host, 0, kABytes);
    memset(g->bufs_[g->hb_[1]].host, 0, kBBytes);
    if (const char* t = getenv("QWENOX_NPU_TIMEOUT_MS"); t && *t)
      g->timeout_ = std::chrono::milliseconds(std::max(1, atoi(t)));
    if (!g->heartbeat()) {  // 趁 GPU 空闲先唤醒 NPU
      err = "NPU 首个任务未完成: " + g->last_error_;
      return nullptr;
    }
    NpuGemm* gp = g.get();
    g->worker_ = std::thread([gp] { gp->worker_loop(); });
    const char* ka = getenv("QWENOX_NPU_KEEPALIVE");
    if (!ka || ka[0] != '0') {  // 默认开，QWENOX_NPU_KEEPALIVE=0 关
      g->keepalive_ = std::thread([gp] {
        std::unique_lock lock(gp->ka_mutex_);
        while (!gp->ka_stop_) {
          gp->ka_wake_.wait_for(lock, std::chrono::milliseconds(1000));
          if (gp->ka_stop_) break;
          lock.unlock();
          gp->heartbeat();
          lock.lock();
        }
      });
    }
    return g.release();
  }

  ~NpuGemm() {
    {
      const std::lock_guard<std::mutex> lock(ka_mutex_);
      ka_stop_ = true;
    }
    ka_wake_.notify_all();
    if (keepalive_.joinable()) keepalive_.join();
    {
      const std::lock_guard<std::mutex> lock(q_mutex_);
      w_stop_ = true;
    }
    q_wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (failed_) {
      // 超时的 run 可能仍在 DMA：abort 之后绝不释放它用到的页和 run 对象
      // （泄漏到进程退出），否则可能带着在飞的 DMA 释放内存。
      for (auto& r : abandoned_) {
        try { r.abort(); } catch (const std::exception&) {}
      }
      new std::vector<xrt::run>(std::move(abandoned_));
      new std::vector<Buf>(std::move(bufs_));
      new std::deque<std::pair<xrt::run, int>>(std::move(pending_));
      return;
    }
    pending_.clear();
    for (auto& b : bufs_) {
      b.bo = xrt::bo();
      if (b.host) (void)hipHostFree(b.host);
    }
  }

  // 分配 GPU 可见的固定页并导入 XRT；返回 id，失败 -1。
  int NewBuffer(size_t bytes) {
    void* host = nullptr;
    if (hipHostMalloc(&host, bytes, hipHostMallocMapped) != hipSuccess) return -1;
    try {
      Buf b;
      b.host = host;
      b.bytes = bytes;
      if (hipHostGetDevicePointer(&b.dev, host, 0) != hipSuccess)
        throw std::runtime_error("hipHostGetDevicePointer failed");
      b.bo = xrt::bo(dev_, host, bytes, XRT_BO_FLAGS_HOST_ONLY, grp_);
      bufs_.push_back(std::move(b));
    } catch (const std::exception&) {
      (void)hipHostFree(host);
      return -1;
    }
    return (int)bufs_.size() - 1;
  }
  void* Dev(int id) const { return bufs_[id].dev; }
  void* Host(int id) const { return bufs_[id].host; }

  // 权重缓冲被 GPU 写完（调用方已同步流）后调用一次。
  void FinalizeWeights(int id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    bufs_[id].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // 把 count 个 run（读 A=a，B=b_ids[i]，写 C=c_ids[i]）排进 stream：到达该
  // 流位置时由 host func 转交 worker 线程，NPU 在 GPU 继续跑后面的 kernel 时
  // 并行执行。
  bool Submit(hipStream_t stream, int a, const int* b_ids, const int* c_ids, int count) {
    auto* call = new Call{this, a, {b_ids, b_ids + count}, {c_ids, c_ids + count}};
    if (hipLaunchHostFunc(
            stream,
            [](void* data) {
              std::unique_ptr<Call> c((Call*)data);
              NpuGemm* s = c->self;
              {
                const std::lock_guard<std::mutex> lock(s->q_mutex_);
                s->queue_.push_back(std::move(*c));
                ++s->enqueued_;
              }
              s->q_wake_.notify_one();
            },
            call) != hipSuccess) {
      delete call;
      return false;
    }
    return true;
  }

  // 阻塞 stream 直到所有已提交的 run 完成（C 缓冲之后即可被 GPU 读）。
  bool Wait(hipStream_t stream) {
    return hipLaunchHostFunc(
               stream, [](void* data) { ((NpuGemm*)data)->wait_all(); }, this) == hipSuccess;
  }

  // CPU 侧直接等（Wait 的 host func 排不上流时的兜底）。
  void WaitHost() { wait_all(); }

  bool Healthy() const { return !failed_; }

private:
  NpuGemm() = default;

  static std::string write_xclbin() {
    wchar_t tmp[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, tmp)) return "";
    std::wstring dir = std::wstring(tmp) + L"qwenox_npu\\";
    CreateDirectoryW(dir.c_str(), nullptr);
    const std::wstring xp = dir + L"final_4096x2560x1024_128x64x128.xclbin";
    // 同名同大小视为已写过（引擎每次启动都写一遍 ~900KB 没必要）
    if (FILE* f = _wfopen(xp.c_str(), L"rb")) {
      _fseeki64(f, 0, SEEK_END);
      const __int64 sz = _ftelli64(f);
      fclose(f);
      if (sz == (__int64)kXclbinLen) return w2u(xp);
    }
    if (FILE* f = _wfopen(xp.c_str(), L"wb")) {
      fwrite(kXclbin, 1, kXclbinLen, f);
      fclose(f);
      return w2u(xp);
    }
    return "";
  }
  static std::string w2u(const std::wstring& w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0,
                                      nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
  }

  bool wait_run(xrt::run& run) {
    const ert_cmd_state st = run.wait(timeout_);
    if (st == ERT_CMD_STATE_COMPLETED) return true;
    failed_ = true;
    last_error_ = st == ERT_CMD_STATE_TIMEOUT
                      ? "NPU run 超时（" + std::to_string(timeout_.count()) + " ms）"
                      : "NPU run 未完成";
    fprintf(stderr, "%s NPU: %s——本次请求的结果无效，后续层回退 GPU\n", log_ts(),
            last_error_.c_str());
    return false;
  }

  // 收割最老的未完成 run（调用方持 mutex_）。
  void reap_oldest() {
    auto& front = pending_.front();
    const bool ok = wait_run(front.first);
    xrt::run run = std::move(front.first);
    const int c = front.second;
    pending_.pop_front();
    if (!ok) {
      abandoned_.push_back(std::move(run));
      return;
    }
    bufs_[c].bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  bool heartbeat() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ || hb_[0] < 0) return false;
    if (!pending_.empty()) return true;  // 有真任务在飞：设备醒着
    const auto now = std::chrono::steady_clock::now();
    if (now - last_submit_ < std::chrono::milliseconds(2000)) return true;
    try {
      auto run = kern_(3, instr_, words_, bufs_[hb_[0]].bo, bufs_[hb_[1]].bo,
                       bufs_[hb_[2]].bo);
      if (wait_run(run)) return true;
      abandoned_.push_back(std::move(run));
      return false;
    } catch (const std::exception& e) {
      failed_ = true;
      last_error_ = e.what();
      return false;
    }
  }

  struct Call {
    NpuGemm* self;
    int a;
    std::vector<int> b, c;
  };

  void process(const Call& call) {
    const std::lock_guard<std::mutex> lock(mutex_);
    last_submit_ = std::chrono::steady_clock::now();
    if (failed_) return;
    try {
      bufs_[call.a].bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      for (size_t i = 0; i < call.b.size() && !failed_; ++i) {
        while (pending_.size() >= 4) reap_oldest();
        pending_.emplace_back(kern_(3, instr_, words_, bufs_[call.a].bo,
                                    bufs_[call.b[i]].bo, bufs_[call.c[i]].bo),
                              call.c[i]);
      }
      while (!pending_.empty()) reap_oldest();
    } catch (const std::exception& e) {
      failed_ = true;
      last_error_ = e.what();
      pending_.clear();
    }
  }

  void worker_loop() {
    std::unique_lock lock(q_mutex_);
    for (;;) {
      q_wake_.wait(lock, [this] { return w_stop_ || !queue_.empty(); });
      if (queue_.empty()) return;
      Call call = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      process(call);
      lock.lock();
      ++completed_;
      q_idle_.notify_all();
    }
  }

  void wait_all() {
    std::unique_lock lock(q_mutex_);
    q_idle_.wait(lock, [this] { return completed_ == enqueued_; });
  }

  xrt::device dev_;
  xrt::hw_context ctx_;
  xrt::kernel kern_;
  xrt::bo instr_;
  size_t words_ = 0;
  int grp_ = 0;
  std::vector<Buf> bufs_;
  int hb_[3] = {-1, -1, -1};
  std::chrono::milliseconds timeout_{8000};
  std::string last_error_;

  std::mutex mutex_;  // 保护 XRT 调用与 pending_
  std::deque<std::pair<xrt::run, int>> pending_;
  std::vector<xrt::run> abandoned_;
  std::atomic<bool> failed_{false};
  std::chrono::steady_clock::time_point last_submit_{};

  std::thread worker_;
  std::mutex q_mutex_;
  std::condition_variable q_wake_, q_idle_;
  std::deque<Call> queue_;
  uint64_t enqueued_ = 0, completed_ = 0;
  bool w_stop_ = false;

  std::thread keepalive_;
  std::mutex ka_mutex_;
  std::condition_variable ka_wake_;
  bool ka_stop_ = false;
};

#endif  // QWENOX_NPU
