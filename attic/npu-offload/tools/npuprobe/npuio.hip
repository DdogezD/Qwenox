// npuio.hip -- XDNA2 NPU I/O probe for Windows (and Linux) with gufo-npu's
// block-float GEMM (4096 x 2560 x 1024, final_*.xclbin + insts_*.txt).
//
// The GPU and the NPU never work at the same time here: every GPU kernel is
// followed by hipDeviceSynchronize before the NPU is started, and the GPU is
// idle while NPU runs are timed (Strix Halo AGESA < 1.0.0.2b can hang under
// concurrent GPU+NPU load, ROCm/TheRock#8709).
//
// Stages (each prints PASS/FAIL/SKIP):
//   T0 open device, load xclbin, create hw_context + kernel
//   T1 XRT-owned BO: alloc, CPU write/read through map(), sync cost
//   T2 GEMM on XRT BOs vs CPU reference (sampled rows)
//   T3 NPU timing: serial latency and 4-deep pipelined throughput
//   T4 userptr import of VirtualAlloc/aligned pages
//   T5 userptr import of hipHostMalloc(Mapped) pages; GPU->NPU and NPU->GPU
//   T6 reverse: XRT BO pages registered into HIP (hipHostRegister)
//   T7 GPU bandwidth over the shared pages (NPU idle)
//
// usage: npuio <dir with xclbin+insts> [reps]
//
// Windows: let the loader pick System32\xrt_coreutil.dll (the one xrt-smi uses).
// A copy of the DriverStore pair next to the exe opens the device but its
// shim rejects register_xclbin ("load_axlf: not supported").

#include <hip/hip_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#endif

#include <algorithm>
#include <cstdarg>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t M = 4096, K = 2560, N = 1024;
constexpr std::size_t kA = std::size_t{M} * K / 8 * 9;
constexpr std::size_t kB = std::size_t{K} * N / 8 * 9;
constexpr std::size_t kC = std::size_t{M} * N / 8 * 9;
constexpr unsigned kOpcode = 3;

int g_fail = 0;
void Result(const char* t, const char* st, const std::string& msg) {
  std::printf("[%s] %s  %s\n", t, st, msg.c_str());
  std::fflush(stdout);
  if (std::strcmp(st, "FAIL") == 0) ++g_fail;
}
std::string F(const char* fmt, ...) {
  char b[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  return b;
}
double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
#define HCK(x)                                                         \
  do {                                                                 \
    hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                            \
      std::printf("  HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                  __FILE__, __LINE__);                                 \
      throw std::runtime_error("hip");                                 \
    }                                                                  \
  } while (0)

// ---- block-float v8bfp16ebs8, same as gufo's NpuEncodeGroup -------------
std::uint32_t Bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float FromBits(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
void EncodeGroup(const float* v, std::uint8_t* out) {
  unsigned me = 0;
  for (int i = 0; i < 8; ++i) me = std::max(me, (Bits(v[i]) >> 23) & 0xFFu);
  const float inv = FromBits(me >= 7 ? (260u - me) << 23 : 0u);
  out[0] = static_cast<std::uint8_t>(me);
  for (int i = 0; i < 8; ++i) {
    const float q = std::fmin(127.f, std::fmax(-128.f, std::nearbyint(v[i] * inv)));
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  }
}
float GroupValue(const std::uint8_t* p, int i) {
  return static_cast<float>(static_cast<std::int8_t>(p[1 + i])) *
         (std::ldexp(1.0f, static_cast<int>(p[0]) - 127) / 64.0f);
}
// Group g of A: (row, first col) of 8 consecutive K values.
void APos(std::size_t g, std::uint32_t& r, std::uint32_t& c) {
  const std::uint32_t tiles_k = K / 64;
  const std::size_t t = g / 1024;
  const std::uint32_t gi = g % 1024;
  const std::uint32_t tr = t / tiles_k, tc = t % tiles_k;
  const std::uint32_t rib = gi & 7, block = (gi >> 3) & 1, sc = (gi >> 4) & 7, sr = gi >> 7;
  r = tr * 128 + sr * 16 + block * 8 + rib;
  c = tc * 64 + sc * 8;
}
// Group g of B: (first k, n) of 8 consecutive K values in column n.
void BPos(std::size_t g, std::uint32_t& k, std::uint32_t& n) {
  const std::uint32_t tiles_k = K / 64;
  const std::size_t t = g / 1024;
  const std::uint32_t gi = g % 1024;
  const std::uint32_t tc = t / tiles_k, tr = t % tiles_k;
  const std::uint32_t cib = gi & 7, block = (gi >> 3) & 1, sr = (gi >> 4) & 7, sc2 = gi >> 7;
  k = tr * 64 + sr * 8;
  n = tc * 128 + (sc2 * 2 + block) * 8 + cib;
}
// Group g of C: (row, first col) of 8 consecutive N values.
void CPos(std::size_t g, std::uint32_t& r, std::uint32_t& c) {
  const std::uint32_t tiles_n = N / 128;
  const std::size_t t = g / 8192;
  const std::uint32_t gi = g % 8192;
  const std::uint32_t tr = t / tiles_n, tc = t % tiles_n;
  const std::uint32_t rib = gi & 7, bc = (gi >> 3) & 1, br = (gi >> 4) & 1, sc = (gi >> 5) & 7, sr = gi >> 8;
  r = tr * 512 + (sr * 2 + br) * 8 + rib;
  c = tc * 128 + (sc * 2 + bc) * 8;
}

struct Data {
  std::vector<std::uint8_t> a, b;   // encoded streams
  std::vector<float> aq, bq;        // dequantized A [M][K], B [K][N]
};
Data MakeData() {
  Data d;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> ua(-1.f, 1.f), ub(-0.05f, 0.05f);
  std::vector<float> a(std::size_t{M} * K), b(std::size_t{K} * N);
  for (auto& x : a) x = ua(rng);
  for (auto& x : b) x = ub(rng);
  d.a.resize(kA); d.b.resize(kB);
  d.aq.resize(a.size()); d.bq.resize(b.size());
  float v[8];
  for (std::size_t g = 0; g < kA / 9; ++g) {
    std::uint32_t r, c; APos(g, r, c);
    for (int i = 0; i < 8; ++i) v[i] = a[std::size_t{r} * K + c + i];
    EncodeGroup(v, &d.a[g * 9]);
    for (int i = 0; i < 8; ++i) d.aq[std::size_t{r} * K + c + i] = GroupValue(&d.a[g * 9], i);
  }
  for (std::size_t g = 0; g < kB / 9; ++g) {
    std::uint32_t k, n; BPos(g, k, n);
    for (int i = 0; i < 8; ++i) v[i] = b[std::size_t{k + i} * N + n];
    EncodeGroup(v, &d.b[g * 9]);
    for (int i = 0; i < 8; ++i) d.bq[std::size_t{k + i} * N + n] = GroupValue(&d.b[g * 9], i);
  }
  return d;
}
std::vector<std::uint32_t> SampleRows() {
  std::vector<std::uint32_t> rows;
  for (std::uint32_t i = 0; i < 64; ++i) rows.push_back(i * 64 + (i * 37) % 64);
  return rows;
}
std::vector<double> Reference(const Data& d, const std::vector<std::uint32_t>& rows) {
  std::vector<double> ref(rows.size() * N, 0.0);
  for (std::size_t ri = 0; ri < rows.size(); ++ri) {
    const float* ar = &d.aq[std::size_t{rows[ri]} * K];
    double* out = &ref[ri * N];
    for (std::uint32_t k = 0; k < K; ++k) {
      const double av = ar[k];
      const float* br = &d.bq[std::size_t{k} * N];
      for (std::uint32_t n = 0; n < N; ++n) out[n] += av * br[n];
    }
  }
  return ref;
}
// Relative L2 error of the sampled rows of C against the reference.
double CheckC(const std::uint8_t* c, const std::vector<std::uint32_t>& rows,
              const std::vector<double>& ref, double* max_rel) {
  std::vector<float> got(rows.size() * N, NAN);
  std::vector<int> idx(M, -1);
  for (std::size_t i = 0; i < rows.size(); ++i) idx[rows[i]] = static_cast<int>(i);
  for (std::size_t g = 0; g < kC / 9; ++g) {
    std::uint32_t r, col; CPos(g, r, col);
    if (idx[r] < 0) continue;
    for (int i = 0; i < 8; ++i) got[idx[r] * N + col + i] = GroupValue(&c[g * 9], i);
  }
  double num = 0, den = 0, mx = 0;
  for (std::size_t ri = 0; ri < rows.size(); ++ri) {
    double rr = 0;
    for (std::uint32_t n = 0; n < N; ++n) rr += ref[ri * N + n] * ref[ri * N + n];
    rr = std::sqrt(rr / N) + 1e-30;
    for (std::uint32_t n = 0; n < N; ++n) {
      const double e = got[ri * N + n] - ref[ri * N + n];
      if (std::isnan(e)) return 1e9;
      num += e * e; den += ref[ri * N + n] * ref[ri * N + n];
      mx = std::max(mx, std::fabs(e) / rr);
    }
  }
  if (max_rel) *max_rel = mx;
  return std::sqrt(num / (den + 1e-30));
}

void* PageAlloc(std::size_t bytes) {
#ifdef _WIN32
  return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  void* p = nullptr;
  return posix_memalign(&p, 4096, bytes) == 0 ? p : nullptr;
#endif
}
void PageFree(void* p) {
#ifdef _WIN32
  VirtualFree(p, 0, MEM_RELEASE);
#else
  std::free(p);
#endif
}

__global__ void CopyKernel(const uint4* __restrict__ src, uint4* __restrict__ dst, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
       i += std::size_t{gridDim.x} * blockDim.x)
    dst[i] = src[i];
}
__global__ void SumKernel(const uint4* __restrict__ src, std::size_t n, unsigned long long* out) {
  unsigned long long s = 0;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
       i += std::size_t{gridDim.x} * blockDim.x) {
    const uint4 v = src[i];
    s += (unsigned long long)v.x + v.y + v.z + v.w;
  }
  atomicAdd(out, s);
}
unsigned long long CpuSum(const std::uint8_t* p, std::size_t bytes) {
  unsigned long long s = 0;
  const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(p);
  for (std::size_t i = 0; i < bytes / 4; ++i) s += w[i];
  return s;
}

struct Npu {
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel kern;
  xrt::bo instr;
  std::size_t words = 0;
  int grp = 0;
  std::chrono::milliseconds timeout{8000};
  // Runs one GEMM; returns ms or <0 on timeout/failure.
  double Run(xrt::bo& a, xrt::bo& b, xrt::bo& c) {
    const double t0 = Now();
    auto run = kern(kOpcode, instr, words, a, b, c);
    const auto st = run.wait(timeout);
    const double ms = (Now() - t0) * 1e3;
    if (st != ERT_CMD_STATE_COMPLETED) {
      std::printf("  run state %d after %.0f ms\n", static_cast<int>(st), ms);
      return -1;
    }
    return ms;
  }
  // First run on fresh BOs, serial median, 4-deep pipelined ms/run, and a
  // bitwise check of every C buffer against `want` afterwards.
  std::string Bench(xrt::bo& a, xrt::bo& b, xrt::bo* c, const std::vector<std::uint8_t>& want,
                    int reps, bool* ok) {
    for (int i = 0; i < 4; ++i) {
      std::memset(c[i].map<void*>(), 0xAB, kC);
      c[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    const double first = Run(a, b, c[0]);
    Run(a, b, c[1]);
    std::vector<double> lat;
    for (int r = 0; r < 10; ++r) lat.push_back(Run(a, b, c[r & 3]));
    std::sort(lat.begin(), lat.end());
    const int total = reps * 2;
    const double t0 = Now();
    std::vector<xrt::run> q;
    for (int r = 0; r < total; ++r) {
      if (q.size() == 4) {
        q.front().wait(timeout);
        q.erase(q.begin());
      }
      q.push_back(kern(kOpcode, instr, words, a, b, c[r & 3]));
    }
    for (auto& r : q) r.wait(timeout);
    const double per = (Now() - t0) * 1e3 / total;
    *ok = first >= 0 && lat.front() >= 0;
    for (int i = 0; i < 4; ++i) {
      c[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      *ok = *ok && std::memcmp(c[i].map<void*>(), want.data(), kC) == 0;
    }
    return F("first %.2f ms, serial median %.3f ms, pipelined %.3f ms/run (%.1f TOPS), all C bit-identical: %s",
             first, lat[lat.size() / 2], per, 2.0 * M * K * N / per / 1e9, *ok ? "yes" : "NO");
  }
};

// Log line straight to disk (survives a hard hang up to the last flush).
struct DiskLog {
  FILE* f = nullptr;
  explicit DiskLog(const std::string& p) { f = std::fopen(p.c_str(), "a"); }
  ~DiskLog() { if (f) std::fclose(f); }
  void Line(const std::string& s) {
    std::printf("%s\n", s.c_str());
    if (!f) return;
    std::fprintf(f, "%s\n", s.c_str());
    std::fflush(f);
#ifdef _WIN32
    _commit(_fileno(f));
#endif
  }
};

bool Exists(const std::string& p) {
  if (p.empty()) return false;
  FILE* f = std::fopen(p.c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}

// Continuous 4-deep NPU GEMM loop; every C is compared bitwise with the
// reference. One log line per second: runs, ms/run, worst completion wait,
// mismatches, timeouts. Exit 0 clean, 2 mismatches, 3 timeout, 1 setup.
int Stress(Npu& npu, const Data& data, const std::vector<std::uint32_t>& rows,
           const std::vector<double>& ref, double secs, const std::string& log_path,
           const std::string& stop_path) {
  DiskLog log(log_path);
  xrt::bo ba, bb, bc[4];
  std::vector<std::uint8_t> want;
  try {
    ba = xrt::bo(npu.dev, kA, xrt::bo::flags::host_only, npu.grp);
    bb = xrt::bo(npu.dev, kB, xrt::bo::flags::host_only, npu.grp);
    for (auto& c : bc) c = xrt::bo(npu.dev, kC, xrt::bo::flags::host_only, npu.grp);
    std::memcpy(ba.map<void*>(), data.a.data(), kA);
    std::memcpy(bb.map<void*>(), data.b.data(), kB);
    ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (npu.Run(ba, bb, bc[0]) < 0) throw std::runtime_error("reference run failed");
    bc[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    want.assign(bc[0].map<std::uint8_t*>(), bc[0].map<std::uint8_t*>() + kC);
    double mx = 0;
    const double rel = CheckC(want.data(), rows, ref, &mx);
    if (rel > 0.03) throw std::runtime_error(F("reference wrong: rel %.4f", rel));
  } catch (const std::exception& e) {
    log.Line(std::string("SETUP_FAIL ") + e.what());
    return 1;
  }
  npu.timeout = std::chrono::milliseconds(3000);
  log.Line(F("READY secs=%.0f", secs));
  const double t_start = Now();
  double t_sec = t_start;
  std::uint64_t runs = 0, runs_sec = 0, bad = 0;
  double worst = 0;
  xrt::run q[4];
  for (int i = 0; i < 4; ++i) q[i] = npu.kern(kOpcode, npu.instr, npu.words, ba, bb, bc[i]);
  int rc = 0;
  for (std::uint64_t it = 0;; ++it) {
    const int i = static_cast<int>(it & 3);
    const double tw = Now();
    const auto st = q[i].wait(npu.timeout);
    worst = std::max(worst, (Now() - tw) * 1e3);
    if (st != ERT_CMD_STATE_COMPLETED) {
      log.Line(F("TIMEOUT t=%.1f state=%d runs=%llu", Now() - t_start, static_cast<int>(st),
                 static_cast<unsigned long long>(runs)));
      rc = 3;
      break;
    }
    bc[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (std::memcmp(bc[i].map<void*>(), want.data(), kC) != 0) {
      if (bad < 5) {
        const auto* p = bc[i].map<const std::uint8_t*>();
        std::size_t diff = 0;
        for (std::size_t b = 0; b < kC; ++b) diff += p[b] != want[b];
        log.Line(F("MISMATCH t=%.2f run=%llu bytes_differ=%zu", Now() - t_start,
                   static_cast<unsigned long long>(runs), diff));
      }
      ++bad;
    }
    ++runs;
    ++runs_sec;
    const double now = Now();
    const bool done = now - t_start >= secs || (it % 64 == 0 && Exists(stop_path));
    if (now - t_sec >= 1.0 || done) {
      log.Line(F("t=%6.1f runs=%5llu ms/run=%.3f worst_wait_ms=%.2f mismatches=%llu",
                 now - t_start, static_cast<unsigned long long>(runs_sec),
                 (now - t_sec) * 1e3 / std::max<std::uint64_t>(runs_sec, 1), worst,
                 static_cast<unsigned long long>(bad)));
      t_sec = now;
      runs_sec = 0;
      worst = 0;
    }
    if (done) break;
    q[i] = npu.kern(kOpcode, npu.instr, npu.words, ba, bb, bc[i]);
  }
  if (rc == 0) {
    for (int k = 0; k < 4; ++k) q[k].wait(npu.timeout);
    rc = bad ? 2 : 0;
  }
  log.Line(F("END runs=%llu mismatches=%llu rc=%d", static_cast<unsigned long long>(runs),
             static_cast<unsigned long long>(bad), rc));
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string dir = argc > 1 ? argv[1] : ".";
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  const std::string xclbin_path = dir + "/final_4096x2560x1024_128x64x128.xclbin";
  const std::string insts_path = dir + "/insts_4096x2560x1024_128x64x128.txt";
  std::printf("npuio: M=%u K=%u N=%u  A %.2f MB  B %.2f MB  C %.2f MB\n", M, K, N,
              kA / 1e6, kB / 1e6, kC / 1e6);

  // ---- T0 ---------------------------------------------------------------
  Npu npu;
  try {
    const double t0 = Now();
    std::printf("  . xrt::device(0)\n");
    npu.dev = xrt::device(0u);
    std::printf("  . xclbin(file)\n");
    const xrt::xclbin xb{xclbin_path};
    std::printf("  . register_xclbin\n");
    npu.dev.register_xclbin(xb);
    std::printf("  . hw_context\n");
    npu.ctx = xrt::hw_context(npu.dev, xb.get_uuid());
    std::printf("  . kernel\n");
    npu.kern = xrt::kernel(npu.ctx, xb.get_kernels().at(0).get_name());
    std::printf("  . instr bo\n");
    std::ifstream f(insts_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + insts_path);
    const auto sz = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> w(sz);
    f.read(w.data(), static_cast<std::streamsize>(sz));
    npu.words = sz / 4;
    npu.instr = xrt::bo(npu.dev, sz, xrt::bo::flags::cacheable, npu.kern.group_id(1));
    std::memcpy(npu.instr.map<void*>(), w.data(), sz);
    npu.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    npu.grp = npu.kern.group_id(3);
    Result("T0", "PASS", F("kernel '%s', %zu instr words, groups instr=%d a=%d b=%d c=%d, %.0f ms",
                           xb.get_kernels().at(0).get_name().c_str(), npu.words,
                           npu.kern.group_id(1), npu.kern.group_id(3), npu.kern.group_id(4),
                           npu.kern.group_id(5), (Now() - t0) * 1e3));
  } catch (const std::exception& e) {
    Result("T0", "FAIL", e.what());
    std::printf("RESULT: FAIL (NPU setup)\n");
    return 1;
  }

  std::printf("  encoding random A/B on CPU ...\n");
  const Data data = MakeData();
  const auto rows = SampleRows();
  const auto ref = Reference(data, rows);
  std::vector<std::uint8_t> c_xrt;  // NPU result on XRT BOs, the bitwise reference

  if (argc > 2 && std::strcmp(argv[2], "stress") == 0) {
    const double secs = argc > 3 ? std::atof(argv[3]) : 60;
    const std::string log_path = argc > 4 ? argv[4] : "npustress.log";
    const std::string stop_path = argc > 5 ? argv[5] : "";
    return Stress(npu, data, rows, ref, secs, log_path, stop_path);
  }

  // ---- T1 ---------------------------------------------------------------
  xrt::bo ba, bb, bc[4];
  try {
    double t0 = Now();
    ba = xrt::bo(npu.dev, kA, xrt::bo::flags::host_only, npu.grp);
    bb = xrt::bo(npu.dev, kB, xrt::bo::flags::host_only, npu.grp);
    for (auto& c : bc) c = xrt::bo(npu.dev, kC, xrt::bo::flags::host_only, npu.grp);
    const double alloc_ms = (Now() - t0) * 1e3;
    auto* pa = ba.map<std::uint8_t*>();
    t0 = Now();
    for (int r = 0; r < 5; ++r) std::memcpy(pa, data.a.data(), kA);
    const double wr = 5 * kA / (Now() - t0) / 1e9;
    std::vector<std::uint8_t> tmp(kA);
    t0 = Now();
    for (int r = 0; r < 5; ++r) std::memcpy(tmp.data(), pa, kA);
    const double rd = 5 * kA / (Now() - t0) / 1e9;
    std::memcpy(pa, data.a.data(), kA);
    std::memcpy(bb.map<void*>(), data.b.data(), kB);
    t0 = Now();
    for (int r = 0; r < 10; ++r) ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const double s_to = (Now() - t0) * 1e3 / 10;
    t0 = Now();
    for (int r = 0; r < 10; ++r) bc[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const double s_from = (Now() - t0) * 1e3 / 10;
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    Result("T1", "PASS", F("alloc 6 BOs %.1f ms; CPU memcpy into map %.1f GB/s, out of map %.1f GB/s; "
                           "sync A to_dev %.3f ms, C from_dev %.3f ms",
                           alloc_ms, wr, rd, s_to, s_from));
  } catch (const std::exception& e) {
    Result("T1", "FAIL", e.what());
    std::printf("RESULT: FAIL\n");
    return 1;
  }

  // ---- T2 ---------------------------------------------------------------
  try {
    std::memset(bc[0].map<void*>(), 0xAB, kC);
    bc[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const double ms = npu.Run(ba, bb, bc[0]);
    if (ms < 0) throw std::runtime_error("first run did not complete");
    bc[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c_xrt.assign(bc[0].map<std::uint8_t*>(), bc[0].map<std::uint8_t*>() + kC);
    double mx = 0;
    const double rel = CheckC(c_xrt.data(), rows, ref, &mx);
    // Second run must be bit-identical (determinism).
    std::memset(bc[1].map<void*>(), 0x5A, kC);
    bc[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    npu.Run(ba, bb, bc[1]);
    bc[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const bool same = std::memcmp(bc[1].map<void*>(), c_xrt.data(), kC) == 0;
    Result("T2", rel < 0.03 && same ? "PASS" : "FAIL",
           F("first run %.1f ms; rel L2 %.4f, max |err|/row_rms %.3f (64 rows x 1024); "
             "rerun bit-identical: %s", ms, rel, mx, same ? "yes" : "NO"));
  } catch (const std::exception& e) {
    Result("T2", "FAIL", e.what());
  }

  // ---- T3 ---------------------------------------------------------------
  try {
    std::vector<double> lat;
    for (int r = 0; r < reps; ++r) lat.push_back(npu.Run(ba, bb, bc[r & 3]));
    std::sort(lat.begin(), lat.end());
    const double med = lat[lat.size() / 2];
    // Pipelined: 4 in flight.
    const int total = reps * 2;
    const double t0 = Now();
    std::vector<xrt::run> inflight;
    for (int r = 0; r < total; ++r) {
      if (inflight.size() == 4) {
        inflight.front().wait(npu.timeout);
        inflight.erase(inflight.begin());
      }
      inflight.push_back(npu.kern(kOpcode, npu.instr, npu.words, ba, bb, bc[r & 3]));
    }
    for (auto& r : inflight) r.wait(npu.timeout);
    const double per = (Now() - t0) * 1e3 / total;
    const double flop = 2.0 * M * K * N;
    Result("T3", "PASS",
           F("serial median %.2f ms (min %.2f max %.2f); pipelined %.2f ms/run = %.2f TOPS, "
             "stream A+B+C %.1f GB/s", med, lat.front(), lat.back(), per, flop / per / 1e9,
             (kA + kB + kC) / per / 1e6));
  } catch (const std::exception& e) {
    Result("T3", "FAIL", e.what());
  }

  // ---- T4 ---------------------------------------------------------------
  {
    void *ha = PageAlloc(kA), *hb = PageAlloc(kB), *hc = PageAlloc(kC);
    try {
      std::memcpy(ha, data.a.data(), kA);
      std::memcpy(hb, data.b.data(), kB);
      std::memset(hc, 0xAB, kC);
      xrt::bo ua(npu.dev, ha, kA, xrt::bo::flags::host_only, npu.grp);
      xrt::bo ub(npu.dev, hb, kB, xrt::bo::flags::host_only, npu.grp);
      xrt::bo uc(npu.dev, hc, kC, xrt::bo::flags::host_only, npu.grp);
      const bool alias = ua.map<void*>() == ha;
      ua.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      ub.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      uc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      const double ms = npu.Run(ua, ub, uc);
      uc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const bool same = ms >= 0 && std::memcmp(hc, c_xrt.data(), kC) == 0;
      Result("T4", same ? "PASS" : "FAIL",
             F("userptr VirtualAlloc pages: map()==user ptr %s; run %.1f ms; C bit-identical to T2: %s",
               alias ? "yes" : "no", ms, same ? "yes" : "NO"));
    } catch (const std::exception& e) {
      Result("T4", "FAIL", std::string("userptr import of plain pages: ") + e.what());
    }
    PageFree(ha); PageFree(hb); PageFree(hc);
  }

  // ---- HIP ----------------------------------------------------------------
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    Result("T5", "SKIP", "no HIP device");
  } else {
    hipDeviceProp_t prop{};
    (void)hipGetDeviceProperties(&prop, 0);
    std::printf("  HIP device: %s (%s), canMapHostMemory=%d\n", prop.name, prop.gcnArchName,
                prop.canMapHostMemory);
    unsigned long long* dsum = nullptr;
    std::uint8_t* d_a = nullptr;
    try {
      HCK(hipMalloc(&dsum, sizeof(unsigned long long)));
      HCK(hipMalloc(&d_a, kA));
      HCK(hipMemcpy(d_a, data.a.data(), kA, hipMemcpyHostToDevice));
    } catch (...) {
    }
    const unsigned long long c_sum = CpuSum(c_xrt.data(), kC);

    // ---- T5: hipHostMalloc pages imported into XRT ----------------------
    for (const unsigned flags : {unsigned(hipHostMallocMapped),
                                 unsigned(hipHostMallocMapped | hipHostMallocNonCoherent)}) {
      const char* fname = (flags & hipHostMallocNonCoherent) ? "Mapped|NonCoherent" : "Mapped";
      void *ha = nullptr, *hb = nullptr, *hc = nullptr;
      try {
        HCK(hipHostMalloc(&ha, kA, flags));
        HCK(hipHostMalloc(&hb, kB, flags));
        HCK(hipHostMalloc(&hc, kC, flags));
        void *da = nullptr, *dc = nullptr;
        HCK(hipHostGetDevicePointer(&da, ha, 0));
        HCK(hipHostGetDevicePointer(&dc, hc, 0));
        std::printf("  T5 %s: host %p dev %p (4K aligned: %s)\n", fname, ha, da,
                    (reinterpret_cast<std::uintptr_t>(ha) & 4095) == 0 ? "yes" : "no");
        std::memset(ha, 0, kA);
        std::memcpy(hb, data.b.data(), kB);
        std::memset(hc, 0xAB, kC);
        xrt::bo ua, ub, uc;
        try {
          ua = xrt::bo(npu.dev, ha, kA, xrt::bo::flags::host_only, npu.grp);
          ub = xrt::bo(npu.dev, hb, kB, xrt::bo::flags::host_only, npu.grp);
          uc = xrt::bo(npu.dev, hc, kC, xrt::bo::flags::host_only, npu.grp);
        } catch (const std::exception& e) {
          Result("T5", "FAIL", F("%s: XRT userptr import of hipHostMalloc pages: %s", fname, e.what()));
          throw std::runtime_error("skip");
        }
        // GPU writes A into the shared pages, then (GPU idle) NPU reads it.
        CopyKernel<<<256, 256>>>(reinterpret_cast<const uint4*>(d_a), reinterpret_cast<uint4*>(da), kA / 16);
        HCK(hipGetLastError());
        HCK(hipDeviceSynchronize());
        ua.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ub.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        uc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const double ms = npu.Run(ua, ub, uc);
        uc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const bool cpu_same = ms >= 0 && std::memcmp(hc, c_xrt.data(), kC) == 0;
        // NPU result read by the GPU through its own pointer (NPU idle).
        HCK(hipMemset(dsum, 0, 8));
        SumKernel<<<256, 256>>>(reinterpret_cast<const uint4*>(dc), kC / 16, dsum);
        HCK(hipDeviceSynchronize());
        unsigned long long g_sum = 0;
        HCK(hipMemcpy(&g_sum, dsum, 8, hipMemcpyDeviceToHost));
        const bool ok = cpu_same && g_sum == c_sum;
        Result("T5", ok ? "PASS" : "FAIL",
               F("%s: GPU wrote A -> NPU run %.1f ms -> C bit-identical (CPU view) %s, "
                 "GPU checksum of C %s", fname, ms, cpu_same ? "yes" : "NO",
                 g_sum == c_sum ? "matches" : "MISMATCH"));
      } catch (const std::exception& e) {
        if (std::string(e.what()) != "skip") Result("T5", "FAIL", F("%s: %s", fname, e.what()));
      }
      if (ha) (void)hipHostFree(ha);
      if (hb) (void)hipHostFree(hb);
      if (hc) (void)hipHostFree(hc);
    }

    // ---- T6: XRT BO pages registered into HIP ---------------------------
    try {
      xrt::bo xa(npu.dev, kA, xrt::bo::flags::host_only, npu.grp);
      xrt::bo xc(npu.dev, kC, xrt::bo::flags::host_only, npu.grp);
      void* pa = xa.map<void*>();
      void* pc = xc.map<void*>();
      std::memset(pa, 0, kA);
      std::memset(pc, 0xAB, kC);
      hipError_t e1 = hipHostRegister(pa, kA, hipHostRegisterMapped);
      hipError_t e2 = hipHostRegister(pc, kC, hipHostRegisterMapped);
      if (e1 != hipSuccess || e2 != hipSuccess) {
        Result("T6", "FAIL", F("hipHostRegister of XRT BO map(): %s / %s", hipGetErrorString(e1),
                               hipGetErrorString(e2)));
        (void)hipGetLastError();
      } else {
        void *da = nullptr, *dc = nullptr;
        HCK(hipHostGetDevicePointer(&da, pa, 0));
        HCK(hipHostGetDevicePointer(&dc, pc, 0));
        CopyKernel<<<256, 256>>>(reinterpret_cast<const uint4*>(d_a), reinterpret_cast<uint4*>(da), kA / 16);
        HCK(hipDeviceSynchronize());
        xa.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const double ms = npu.Run(xa, bb, xc);
        xc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const bool cpu_same = ms >= 0 && std::memcmp(pc, c_xrt.data(), kC) == 0;
        HCK(hipMemset(dsum, 0, 8));
        SumKernel<<<256, 256>>>(reinterpret_cast<const uint4*>(dc), kC / 16, dsum);
        HCK(hipDeviceSynchronize());
        unsigned long long g_sum = 0;
        HCK(hipMemcpy(&g_sum, dsum, 8, hipMemcpyDeviceToHost));
        Result("T6", cpu_same && g_sum == c_sum ? "PASS" : "FAIL",
               F("XRT BO -> hipHostRegister: GPU wrote A -> NPU %.1f ms -> C identical %s, GPU checksum %s",
                 ms, cpu_same ? "yes" : "NO", g_sum == c_sum ? "matches" : "MISMATCH"));
        (void)hipHostUnregister(pa);
        (void)hipHostUnregister(pc);
      }
    } catch (const std::exception& e) {
      Result("T6", "FAIL", e.what());
    }

    // ---- T7: GPU bandwidth over host pages (NPU idle) -------------------
    try {
      const std::size_t bytes = std::size_t{256} << 20;
      void* hp = nullptr;
      HCK(hipHostMalloc(&hp, bytes, hipHostMallocMapped));
      void* dp = nullptr;
      HCK(hipHostGetDevicePointer(&dp, hp, 0));
      void* dv = nullptr;
      HCK(hipMalloc(&dv, bytes));
      auto bw = [&](const void* s, void* d) {
        CopyKernel<<<1024, 256>>>(reinterpret_cast<const uint4*>(s), reinterpret_cast<uint4*>(d), bytes / 16);
        HCK(hipDeviceSynchronize());
        const double t0 = Now();
        for (int r = 0; r < 5; ++r)
          CopyKernel<<<1024, 256>>>(reinterpret_cast<const uint4*>(s), reinterpret_cast<uint4*>(d), bytes / 16);
        HCK(hipDeviceSynchronize());
        return 5.0 * bytes / (Now() - t0) / 1e9;
      };
      const double rd = bw(dp, dv), wr = bw(dv, dp), dd = bw(dv, static_cast<char*>(dv));
      Result("T7", "PASS", F("GPU copy kernel: pinned->vram %.1f GB/s, vram->pinned %.1f GB/s, "
                             "vram->vram(in place) %.1f GB/s", rd, wr, dd));
      (void)hipFree(dv);
      (void)hipHostFree(hp);
    } catch (const std::exception& e) {
      Result("T7", "FAIL", e.what());
    }
    if (dsum) (void)hipFree(dsum);
    if (d_a) (void)hipFree(d_a);
  }

  // ---- T8: NPU speed per memory kind (GPU idle) ----------------------------
  {
    bool ok = false;
    Result("T8", "INFO", "XRT host_only BOs:      " + npu.Bench(ba, bb, bc, c_xrt, reps, &ok));
    if (!ok) ++g_fail;
    std::vector<void*> pages;
    try {
      void *ha = PageAlloc(kA), *hb = PageAlloc(kB);
      pages = {ha, hb};
      std::memcpy(ha, data.a.data(), kA);
      std::memcpy(hb, data.b.data(), kB);
      xrt::bo ua(npu.dev, ha, kA, xrt::bo::flags::host_only, npu.grp);
      xrt::bo ub(npu.dev, hb, kB, xrt::bo::flags::host_only, npu.grp);
      ua.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      ub.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      xrt::bo uc[4];
      for (auto& c : uc) {
        void* h = PageAlloc(kC);
        pages.push_back(h);
        c = xrt::bo(npu.dev, h, kC, xrt::bo::flags::host_only, npu.grp);
      }
      Result("T8", "INFO", "VirtualAlloc userptr:   " + npu.Bench(ua, ub, uc, c_xrt, reps, &ok));
      if (!ok) ++g_fail;
    } catch (const std::exception& e) {
      Result("T8", "FAIL", std::string("VirtualAlloc userptr: ") + e.what());
    }
    for (void* p : pages) PageFree(p);
    for (const unsigned flags : {unsigned(hipHostMallocMapped),
                                 unsigned(hipHostMallocMapped | hipHostMallocNonCoherent)}) {
      std::vector<void*> hp;
      try {
        auto alloc = [&](std::size_t bytes) {
          void* p = nullptr;
          HCK(hipHostMalloc(&p, bytes, flags));
          hp.push_back(p);
          return p;
        };
        void* ha = alloc(kA);
        void* hb = alloc(kB);
        std::memcpy(ha, data.a.data(), kA);
        std::memcpy(hb, data.b.data(), kB);
        xrt::bo ua(npu.dev, ha, kA, xrt::bo::flags::host_only, npu.grp);
        xrt::bo ub(npu.dev, hb, kB, xrt::bo::flags::host_only, npu.grp);
        ua.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ub.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::bo uc[4];
        for (auto& c : uc) c = xrt::bo(npu.dev, alloc(kC), kC, xrt::bo::flags::host_only, npu.grp);
        const std::string name = (flags & hipHostMallocNonCoherent) ? "hipHostMalloc NonCoh:   "
                                                                    : "hipHostMalloc Mapped:   ";
        Result("T8", "INFO", name + npu.Bench(ua, ub, uc, c_xrt, reps, &ok));
        if (!ok) ++g_fail;
      } catch (const std::exception& e) {
        Result("T8", "FAIL", std::string("hipHostMalloc userptr: ") + e.what());
      }
      for (void* p : hp) (void)hipHostFree(p);
    }
  }

  // ---- T9: wake-up after idle -------------------------------------------
  try {
    std::string s;
    for (const int idle_ms : {200, 2000, 6000, 12000}) {
      std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
      const double ms = npu.Run(ba, bb, bc[0]);
      s += F("idle %d ms -> run %.2f ms; ", idle_ms, ms);
    }
    Result("T9", "INFO", s);
  } catch (const std::exception& e) {
    Result("T9", "FAIL", e.what());
  }

  std::printf("RESULT: %s (%d failing stage%s)\n", g_fail ? "FAIL" : "PASS", g_fail,
              g_fail == 1 ? "" : "s");
  return g_fail ? 1 : 0;
}
