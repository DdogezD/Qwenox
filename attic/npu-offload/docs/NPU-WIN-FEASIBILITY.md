# gufo-npu 的 GPU+NPU 异构方案：Windows 复刻可行性调研

日期：2026-10-13　　对象：`Workspace/CProject/gufo-npu`（Mike Alani 的 gufo fork，HEAD ef79fb2）
结论先行：**驱动和工具链这条路在 Windows 上是通的；真正的拦路虎是一个与操作系统无关的硬件/固件死机问题（AGESA），以及 Windows 上零拷贝能否成立这个只能实测的未知数。**

---

## 1. gufo-npu 到底让 NPU 干了什么

| 项目 | 内容 |
| --- | --- |
| 阶段 | 只在 **prefill**，decode 不用 NPU |
| NPU 算的东西 | 三类"GPU 暂时还用不到"的稠密 GEMM：① 线性注意力 `ssm_in` 的 z 行（36 层，默认 6×1024 行）② 16 个注意力头的 q/gate 行（12 层，`attn_qkv` 前 8×1024 行）③ 共享专家 gate/up（48 层，640 行） |
| 数据格式 | AIE2P 原生 **bfp16ebs8**（8 元素共享指数的块浮点，9 字节/8 值）；权重源必须是 Q8_0（UD-IQ4_XS 里这几类张量正是 Q8_0） |
| NPU kernel | 单一固定形状 4096×2560×1024，8 列×4 行 = 32 个 AIE 核整阵列；mlir-aie / IRON 生成，产物 `final_*.xclbin` + `insts_*.txt`（已随仓库提供） |
| 重叠方式 | GPU 流上 `hipLaunchHostFunc` 回调把 NPU 任务丢给工作线程；GPU 继续跑别的；第一次真正要用结果的 kernel 前再插一个回调等 NPU 完成；GPU 直接从 NPU 输出流里就地解码（不再拷贝） |
| 零拷贝 | `hipHostMalloc(Mapped)` 分配 pinned 页 → `xrt::bo(device, host_ptr, size, HOST_ONLY, group)` userptr 导入，两边共用同一批物理页 |
| 收益 | pp16384：1698 → **1821 tok/s（+7.2%）**；服务端 40K 冷启 1637 tok/s。贪心输出与纯 GPU 完全一致 |
| 额外内存 | pinned 权重约 1.1 GiB（408 块 × 2.95 MB）+ A/C 激活缓冲，合计 ~1.5 GiB 量级 |
| 代码量 | 9 个文件 +1730 行（`npu_gemm.cpp/hpp` 724 行是 XRT 封装，其余是 GPU 侧编码/解码 kernel 和 executor 接入） |

背景：上游 gufo 8 月做过一轮 XDNA2 研究并全部否决（35618a8：NPU 实测流带宽 47 GB/s，Q8_0 逐组反量化让算力掉到 0.94 TOPS，各种切分净收益 ≤ +2%）。这个 fork 能翻盘，靠的是换成 NPU 原生的块浮点格式、用 4096 行的大 tile 摊薄 dispatch 开销、只挑 GPU 不在关键路径上的计算。

---

## 2. 逐项对照：Linux 依赖 → Windows 对应物

| # | gufo-npu 依赖（Linux） | Windows 对应物 | 判断 |
| --- | --- | --- | --- |
| 1 | amdxdna 内核驱动 + NPU 固件 | AMD NPU MCDM 驱动（32.0.203.xxx；Strix Halo 在支持列表内，1.8 的生产驱动 32.0.203.376） | ✅ 有，官方支持 |
| 2 | XRT 用户态（`libxrt_coreutil.so` / `libxrt_core.so`） | 驱动自带 `C:\Windows\System32\xrt_coreutil.dll`；官方 **Windows XRT SDK**（XRT 2.21.75 `xrt_windows_sdk.zip`）提供头文件和导入库 | ✅ 有 |
| 3 | `xrt::device / register_xclbin / hw_context / kernel(opcode 3, insts BO, …)` | 这正是 mlir-aie 示例 host 代码的标准写法，mlir-aie 的 Windows CI 就在 NPU2 上跑这套 | ✅ 高概率可用 |
| 4 | xclbin + insts 设备二进制 | 设备二进制不依赖操作系统；同一颗 NPU2，形状不变即可直接复用 gufo-npu 提供的文件 | ✅ 高概率可用（需实测一次 GEMM 对拍） |
| 5 | 若要改 kernel 形状：mlir-aie/IRON 工具链 | 现已有**原生 Windows** 安装路线（VS2022/2026 + Python 3.13 + XRT SDK，`iron_setup.py`），不必 WSL | ✅ 有 |
| 6 | **userptr 导入 HIP pinned 页**（零拷贝核心） | 无任何公开资料。Linux 上能成立是因为 ROCm 的 hipHostMalloc 页本质是普通匿名页、amdxdna 可以 pin；Windows 上这些页由 GPU 的 WDDM 内核驱动分配，NPU 驱动能否再锁一次未知 | ❓ **必须实测** |
| 7 | 退路：`GUFO_NPU_ISOLATE=1`（XRT 自有 BO + CPU memcpy） | XRT 自分配 BO + `map()` 在 Windows 上是标准用法 | ✅ 可用，但有代价（见 §4） |
| 8 | `hipLaunchHostFunc`、`hipHostMalloc(Mapped)`、`hipHostGetDevicePointer` | HIP SDK for Windows 都有。已知问题只有 gfx1151 上**图捕获里**的 host 回调看不到前序写入（rocm-systems #13096）；本引擎的 HIP Graph 只用在 decode，NPU 只在 prefill，碰不到 | ✅ 可用（NPU 回调不能进 graph） |
| 9 | `xrt-smi configure --pmode performance` | Windows 版 `C:\Windows\System32\AMD\xrt-smi.exe` 同样支持 `--pmode` | ✅ 有 |
| 10 | 关 NPU runtime suspend（sysfs `power/control=on`） | 无直接对应；gufo 自带 1 s 心跳 keepalive 可替代大部分作用 | ⚠️ 基本可替代 |
| 11 | **锁 GPU fabric clock（sysfs `pp_dpm_fclk`）** | **Windows 没有公开可用的办法**：Ryzen Master 不支持这颗芯片，SMUDebug/ThrottleStop 无效；理论上 BIOS 隐藏菜单 APBDIS=1 + 固定 SOC P-state=P0 可以锁，但 Strix Halo 上没人验证过，且 GMKtec BIOS 通常不开放这些选项 | ❌ 见 §3 |
| 12 | 链接/构建（CMake、`/dev/accel`、`O_DSYNC` trace） | 小改动：trace 换 Win32 文件 API；链接 `xrt_coreutil.lib`；MSVC 需 `/Zc:__cplusplus`（避免 XRT 走旧 Boost 分支），hipcc/clang 下视情况 | ✅ 工作量小 |

**驱动这一项本身的结论：可行。** 驱动、XRT、编译工具链在 Windows 上都是 AMD 官方支持的路线，而且 gufo-npu 的 host 代码就是 mlir-aie 的标准用法。

---

## 3. 最大风险：GPU+NPU 并发死机（与操作系统无关）

gufo-npu README 要求"必须锁 FCLK，否则 GPU 和 NPU 一起跑会把机器卡死"。查下来这不是 Linux 驱动 bug，而是 SoC 固件问题：

- **ROCm/TheRock #8709**（2026-10-02 起，已标 `status: assessed`）：GPU 跑 ROCm 推理、NPU 同时推理时，Data Fabric 在 2–26 s 内报 deferred MCE，继续跑主机挂死需断电。报告者机器是 **GMKtec EVO-X2，BIOS 1.13，AGESA StrixHaloPI-FP11 1.0.0.2**。锁顶档 FCLK 后 30 分钟 0 错误。
- 同帖 Beelink GTR9 Pro（AGESA 1.0.0.0c）：**锁了 FCLK 仍然死**，下次开机原因是 "data fabric sync flood"；GPU 频率/功耗越高死得越快（Performance 140 W 下 ~3 s）。
- **halogen-flash-server #139**：另一台 Strix Halo 锁 FCLK 后仍硬死 3 次；只有"GPU 空闲时才发 NPU 任务"的严格串行能稳定。
- **AMD 工程师 2026-10-06 回复**：已复现，根因在 AGESA，**升级到 1.0.0.2b 可解决**；AGESA 随厂商 BIOS 发布，AMD 无法直接推送。

对 Windows 的含义：

1. 这是芯片固件层的问题，换 Windows 不会自动消失。AMD 自己的 Windows "hybrid" LLM 流程是 prefill 上 NPU、decode 上 GPU，两者基本是**分阶段**而非同时满载，不能当作"Windows 上并发安全"的证据。
2. Windows 上连"锁 FCLK"这个（并不彻底的）权宜手段都没有。
3. 所以**能否在 Windows 上稳定复刻，取决于你的机器 BIOS 里的 AGESA 是否 ≥ 1.0.0.2b**。若你的 NucBox 就是 EVO-X2 且仍是 BIOS 1.13，大概率正是 #8709 那台的同款受影响配置。截至今天没搜到 GMKtec 放出含 1.0.0.2b 的 EVO-X2 BIOS。
4. 在固件未修复的机器上做并发压测，后果是硬死机/断电重启，可能伴随 NPU 结果静默损坏（#8709 报告了"错误附近的 NPU 结果被破坏"）。

---

## 4. 零拷贝不成立时的退路代价（粗估）

若 Windows 上 userptr 导入失败，只能走隔离模式：NPU 用 XRT 自己的 BO，CPU 在每次运行前后 memcpy。

按 8192 token 的 prefill tile 粗算：
- 每层 A（输入，4096×2560 块浮点 ≈ 11.8 MB/4096 tok）进、C（输出，每 1024 列 ≈ 4.7 MB/4096 tok）出；
- z 36 层 ≈ 2.9 GB、q/gate 12 层 ≈ 1.2 GB、shexp 48 层 ≈ 2.0 GB，**合计约 6 GB/tile（≈ 0.75 MB/token）**；
- 多线程 memcpy 按 20–40 GB/s 计，每个 tile 多花 0.15–0.3 s，而一个 tile 本身约 4.5 s；单层拷贝延迟 2–4 ms，落在 NPU 的关键路径上。

也就是说拷贝开销与 +7% 的总收益同一量级，可能吃掉一半甚至全部收益，也可能因为能和 GPU 重叠而影响不大——作者只说了"更慢、调试用"，没给数字，必须实测。

另外还有两条备选零拷贝路线，同样都没有公开资料，可列入实测：
- 反向：XRT 分配 BO → `map()` 得到地址 → `hipHostRegister` 让 GPU 认这批页；
- 让 NPU 驱动 userptr 导入一块普通 `VirtualAlloc` 页，再对同一块页 `hipHostRegister`（两边都"借用"第三方的普通页，而不是互相借对方驱动的页）。

---

## 5. 收益预期（对本引擎）

- 只加速 prefill。gufo 在 pp16K 上省下约 0.65 s / 16K token（+7.2%）。NPU 那部分吞吐基本是固定的，本引擎 prefill 绝对速度与 gufo 同量级，预期也是 **+5~8%** 的区间，前提是零拷贝成立且 GPU 侧的接入同样做到"就地读 NPU 输出流"。
- 并发服务时要注意：NPU 跑的时候会抢共享 LPDDR 带宽（上游实测 NPU 满载时 GPU decode −12%~−18%）。本引擎有 prefill/decode 混合调度，一条请求 prefill 时 NPU 开着，会拖慢同时在 decode 的其他请求。
- 前提：本引擎用的 GGUF 里 `ssm_in` / `attn_qkv` / 共享专家必须是 Q8_0（IQ4_XS UD 版满足；hgn 格式需另写编码入口）。

---

## 6. 建议的推进顺序（每步都是一次性 PASS/FAIL，可随时止损）

| 步骤 | 内容 | 风险 | 决定什么 |
| --- | --- | --- | --- |
| P0 | 查 BIOS/AGESA 版本（HWiNFO64 摘要页或 BIOS 设置首页）；`xrt-smi examine` 看驱动/XRT 版本、NPU 是否在线 | 无 | AGESA < 1.0.0.2b 则后续只能做功能验证，不能做并发长跑 |
| P1 | 独立小程序：XRT SDK + gufo 的 xclbin/insts，XRT 自分配 BO，跑一次 4096×2560×1024 GEMM，与 CPU 参考对拍 | 低（NPU 单独跑） | 驱动+二进制在 Windows 上是否可用 |
| P2 | 同一程序换成 `hipHostMalloc(Mapped)` 指针做 userptr 导入；再试 §4 两条备选 | 低 | 零拷贝能否成立 → 决定走正常模式还是隔离模式 |
| P3 | GPU 空闲时 NPU 连续跑 + GPU 只读校验 NPU 输出（不并发满载） | 低 | 同步/缓存一致性是否正确 |
| P4 | 短时 GPU+NPU 并发（几十秒），观察事件查看器 WHEA 记录 | **高：固件未修复时可能硬死机** | 仅在 P0 确认 AGESA 已修复后才做 |
| P5 | 移植进引擎 prefill（NpuGemm 封装 + 权重编码 + 三处就地解码），KLD/贪心一致性回归 + pp 基准 | 中 | 实际收益 |

P1–P3 不涉及并发，可以在当前 BIOS 上安全完成，能先把"驱动问题"彻底定性。

---

## 6c. 实测：并发死机确证 + AGESA 直接检测（2026-10-13/16，NucBox_EVO-X2）

**AGESA 不用 HWiNFO 也能读**：原始 SMBIOS 表（WMI `root\wmi MSSmBios_RawSMBiosTables`）里直接带版本字符串，本机 BIOS 1.05 读出 `AGESA!V9 StrixHaloPI-FP11 1.0.0.1b` —— 比 #8709 出事机器的 1.0.0.2 还旧一档，属未修复。

**死机已实证**（P4，日志 `~/npuprobe/logs/ov32k_1`）：

- 20:44 NPU 压测循环（0.728 ms/run）+ 引擎 32K prefill 并发；GPU 一开跑 NPU 掉到 0.816→0.923 ms/run（**-13%~-21%**，与上游"NPU 满载拖 GPU decode −12%~−18%"同向）；
- 并发约 **34 秒**后 npu.log 心跳中断（20:45:15），整机硬死，断电重启后系统日志有 Kernel-Power 41（BugcheckCode=0，20:51:44）——与 #8709 症状一致；
- **Windows 下死机没有任何 WHEA 预警**（Linux 上 #8709 有 deferred MCE 前兆），所以 WHEA 看门狗只能算尽力而为，不能指望它救命。

**检测工具已入库**：`tools/npucheck/`——GUI 单文件版（`bash build_win.sh npu` 构建出 `build/npucheck-win.exe`，xclbin 内嵌，按钮式：系统检查/NPU 基线/GPU 基线/并发测试/死机分析，双击即用；也支持 `--selftest` 控制台模式）和脚本版 `npucheck.sh info|stress|postmortem|whea`。两者都带 MARKER 崩溃取证 + WHEA 看门狗 + NPU 结果逐位校验。用法见 `tools/npucheck/README.md`。

---

## 6b. 实测：Windows 上的 NPU IO（2026-10-13，NucBox_EVO-X2）

环境：BIOS EVO-X2 **1.05**（AGESA 旧，未修 #8709）；NPU 驱动 32.0.203.329，XRT **2.19.0**，NPU 固件 1.0.0.196；NPU pmode Performance；HIP = TheRock 10.1 hipcc。
探针：`tools/npuprobe/npuio.hip` + `build_run.sh`。全程 GPU 与 NPU 不同时干活。结果 **RESULT: PASS**。

| 项 | 结果 |
| --- | --- |
| 加载 gufo 的 xclbin/insts（mlir-aie 新工具链产物） | ✅ 旧驱动/XRT 2.19 直接能用（`register_xclbin` → `hw_context` → kernel `MLIR_AIE`），37–72 ms |
| GEMM 正确性（64 行抽样 × 1024 列，对 CPU 参考） | ✅ 相对 L2 误差 1.0%；重跑逐位一致 |
| NPU 单次 GEMM 4096×2560×1024 | 串行 0.79 ms，4 深流水 **0.728 ms/次 ≈ 29.5 TOPS**（A+B+C 合 26.7 GB/s，是算力瓶颈，不是带宽瓶颈） |
| 不同页类型给 NPU 读写（XRT 自有 BO / VirtualAlloc userptr / hipHostMalloc Mapped / Mapped+NonCoherent） | ✅ 全部 0.728 ms/次，速度无差别；每种 4 个 C 缓冲共 ~100 次运行输出全部逐位正确 |
| **零拷贝**：hipHostMalloc 页 userptr 导入 XRT | ✅ 成立。GPU kernel 写 A → NPU 读；NPU 写 C → GPU kernel 读校验和，均正确（中间有 hipDeviceSynchronize + bo.sync） |
| 反方向：XRT BO 的 map() 页 hipHostRegister 给 GPU | ✅ 也成立 |
| hipHostMalloc 在 Windows 上 | host 指针 == device 指针，4K 对齐 |
| bo.sync 开销 | A(11.8 MB) to_device 0.093 ms，C from_device 0.037 ms（缓存刷新，可忽略） |
| GPU 访问 pinned 主机页 | 拷贝 kernel ~100 GB/s（单向计数），与 VRAM↔VRAM 原地拷贝 104 GB/s 持平 → pinned 页对 GPU 就是全速内存 |
| CPU memcpy 进/出 XRT BO 映射 | 34–42 / 44–51 GB/s（可缓存内存；隔离模式退路也不算太贵） |
| NPU 空闲唤醒 | 空闲 0.2 s→1.04 ms，2 s→1.55 ms，**6 s→35.7 ms，12 s→68.6 ms** → 必须有心跳保活（gufo 每 1 s 一次） |
| 驱动自检 `xrt-smi validate` | latency 48.2 µs，throughput PASS |

踩坑：
1. **不要**把 DriverStore 里的 `xrt_coreutil.dll`/`xrt_core.dll` 拷到 exe 旁边：能开设备，但 shim 拒绝 `register_xclbin`（"load_axlf: not supported"）。用 System32 的 `xrt_coreutil.dll`（xrt-smi 用的那份，090e3fa）就正常。
2. `device.get_info<name>()` 在 2.19 头文件 + 这份 DLL 下段错误（ABI/枚举不对齐），别调。
3. XRT 头文件要 2.19 版（github 77c7088），`xrt/detail/version.h` 需手写；导入库从 DLL 导出表生成 .def → `llvm-lib`；hipcc 不能把 .lib 当输入（会被当 HIP 源），分两步编译/链接（`-Xlinker xxx.lib`）。
4. 主机直连 GitHub 超时，依赖在 VM 下载后 scp。

对方案的含义：
- §2 里唯一的 ❓（零拷贝）已解决，gufo 的正常模式在 Windows 可原样复刻；"驱动问题"可以划掉。
- 按 gufo 的分工（每 4096 token：z 216 次 + q/gate 96 次 + shexp 96 次 = 408 次），NPU 纯计算约 **0.30 s / 4096 token**（空载数字；并发时内存争用会让它变慢，未测）。
- 剩下的未知只有：GPU+NPU 并发时双方各掉多少速，以及这台 AGESA 下并发会不会死机——都属于 P4，需要用户拍板再做。

---

## 6d. 实测：引擎接入 —— 共享专家卸载（M1 串行 / M2 重叠，2026-10-17）

前提：BIOS 更新（AGESA 修复 #8709）后 GPU+NPU 并发不再死机，P4 通过，进入 P5。

- **接入点**：每层共享专家的 gate/up GEMM（4096×2560×1024 bf16）卸载给 NPU。权重在启动时一次性编码进 304 MB 固定页；P≥4097 的 prefill chunk 把前 4096 行给 NPU。开关 `QWENOX_NPU=1`（默认关），`QWENOX_NPU_MINP` 调行数门槛。实现：`src/gpu/parts/29_npu.inc`，挂钩在 `40_model.inc` 的 `moe_b_v2` / `moe_b_shared_ey`。
- **M1（串行）**：NPU 算完 GPU 才继续 → pp 739.1 tok/s，比 off（746.0）还慢——NPU 的 ~1.5 ms 没有被任何东西盖住。
- **M2（重叠）**：submit/finish 拆分——`moe:shared` 处提交 NPU 后立即跑路由专家（MoE 时间大头），在 `moe:reduce` 入队前 WaitHost 收尾 + down_proj + sgate，NPU 的 ~1.5 ms 被整个路由阶段盖住。输出与 M1 逐 bit 一致；pend 不匹配等异常自动丢弃在飞结果并永久回退 GPU。
- **pp（6000 tok prompt，gen 64）**：off 680.4–746.0（三次，方差大）/ M1 739.1 / M2 750.3–753.0 tok/s。这组数字没带生产 prefill 开关（基线只有生产配置的一半），不能代表实际收益，以下面的生产配置复测为准。
- **生产配置复测（`tools/npuprobe/m2ab.sh`）**：pp_win 那串 QWENOX_* 开关全开、chunk 6144、32K prompt，off/on 交替各 3 次，NPU 独占、pmode Default。32K 总耗时 off 24.14 / 24.24 / 24.26 s，on 24.57 / 24.33 / 24.50 s，三对都是开 NPU 更慢，平均 **−1.1%**（≈1353 → 1339 tok/s）。`QWENOX_KPROF_SYNC=1` 下 moe:shared（gate/up+down）只占每个 6144 chunk 的 3.0%（131 ms / 4.3 s），NPU 能接走的 gate/up 前 4096 行折合上限约 1.3%，不够抵 GPU 侧 A 编码 + C 解码两个 kernel 和 NPU DMA 的带宽争抢。全程无死机、无 WHEA。
- **正确性**：标准尺度 KLD（bf16_c512，16320 tok；`QWENOX_NPU_MINP=128` 让 512 窗口也走 NPU）：base 0.106686 vs NPU **0.106492**（噪声内，过 +0.002 门）；小规模逐 token KLD 0.0123。
- **待办**：并发服务（PARALLEL>1）下 NPU 抢 LPDDR 带宽对 decode 的影响未测（§5 第 2 条）；z/q/gate 卸载已做完，见 §6e/§6f（均默认关）。

## 6e. M3：GDN z 门控卸载 —— 微正收益默认关；Windows AI 组件抢 NPU 事件（2026-10-17）

M3 技术结论：

- **卸载点**：GDN 层 `in_proj_z`（6144×2560 bf16 = 6 片 N=1024），36 层 = 216 次/4096tok。`gdn_proj_in_b` 提交，收尾挂在 zb 首个消费点（gated norm）之前，让 NPU 的 ~4.4ms 被 qkv GEMM + conv + scan（~42ms）盖住。B 缓冲按 gi=l-(l+1)/4 索引（共 970 MB 固定页）。
- **教训**：收尾若和提交放在同一函数内（窗口只有 qkv/a/b 三个 GEMM ~15ms），gdn:proj 1011.5ms vs z 关 698.1ms，**负收益**；后移到 conv+scan 之后（~42ms 窗口）才转正。
- **稳态实测**（NPU 独占，P=6000 chunk）：gdn:proj 528.7 vs 624.1ms（-95ms），但整 chunk 只 -28.6ms（**+0.36%**）——NPU 的 A/C DMA（每层 ~40MB）与 GPU 抢 LPDDR 带宽，抵消大半。首 chunk 还有一次性权重编码 +260ms。
- **结论**：默认关（`QWENOX_NPU_Z=1` 开），代码保留在 `40_model.inc`（`npu_z_*`）。收益太薄，不值得高占空 NPU 的风险。

Windows AI 组件抢 NPU 事件（后续检测工具的依据）：

- **现象**：BIOS 1.09（AGESA 1.0.0.2b）开启 NPU 后，Windows 11 自动开始用 NPU 跑后台 AI 索引；M3 高占空测试两次整机硬死机（Kernel-Power 41，1:40 / 1:58，**均无 WHEA**，SoC 级挂死，死法同 #8709）。
- **排查**：`xrt-smi examine --report aie-partitions` 显示 5 个进程持 7 个 HW 上下文、提交计数 ~110/s 持续增长；全部是 `WindowsWorkload.Manager.1` 的 `WorkloadsSessionHost.exe`（ONNX Runtime + VitisAI EP）；模块级证据 `ImageSearchSession.dll`（图像语义搜索索引）、`ImageTextSearchSession.dll`（图片 OCR 索引）。父进程是 svchost（COM 激活），杀掉秒复活。
- **处置（全部可逆）**：组策略 `HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsAI` 设 `DisableAIDataAnalysis=1`、`DisableClickToDo=1`；`Set-Service WSAIFabricSvc -StartupType Disabled` + 停止；禁用 `\Microsoft\Windows\WindowsAI\` 下全部 6 个计划任务；kill 所有 WorkloadsSessionHost。之后 aie-partitions 无上下文且不再复活。
- **控制变量结论**：NPU 独占后 M3 连跑正常——**Windows 后台 NPU 租户是死机的必要帮凶**；AGESA 1.0.0.2b 的 #8709 修复不覆盖"引擎高占空 NPU + Windows 后台租户"的多租户场景。M1/M2 低占空（36 次/prefill）在多租户下也从未死机，触发需要高占空。
- **检测工具备忘**（用户计划做）：检测点 = `xrt-smi examine --report aie-partitions` 是否非空、WorkloadsSessionHost 进程、WSAIFabricSvc 服务状态、WindowsAI 计划任务状态、上述策略键值。

## 6f. M4：QSA q/gate 卸载 —— 方向封死（2026-10-17）

设计与实测：

- **卸载点**：QSA 层 `q_proj`（12288×2560 = q 6 片 + gate 6 片），12 层。提交在 `qsa_idx_b` 入口（xb 此刻就绪）；q 半在 qsplit 前收尾（窗口 = qsa:idx ~5ms + k/v GEMM ~2ms）；gate 半压到 qsa:flash（~200ms/层）之后、sigmoid gate 前收尾（依赖 gate_bf16 路径，`QWENOX_QSA_GATE_BF16=0` 时不卸载）。GPU 只补 P>4096 的尾部行。
- **卸载逻辑本身是对的**：chunk1 kprof `qsa:proj 182.4 → 88.3ms`（12 片版）/ 稳态 79.6ms（分半版），GEMM 确实被接走。
- **但收益再次被带宽吃掉**：稳态整 chunk segsum 7966 vs 7989ms，**仅 +0.29%**。NPU 每层 A 编码 + C 解码 ~68MB DMA 与 qsa:flash 抢 LPDDR。至此三个卸载点（shexp/z/qg）全部验证：GPU GEMM 全是显存带宽瓶颈，NPU 让出的算力换不来时间。
- **死机（关键）**：12 片单次提交（~9ms 连续 NPU）在**独占**下 chunk2 硬挂死（KP41 无 WHEA）；拆成 2×6 片（~4.4ms/段）后第一次 kprof 通过，紧接着 KLD 跑又死——这次先报 `unspecified launch failure`（GPU 侧 fault）再整机挂死。结论：**AGESA 1.0.0.2b 修复后，高占空 NPU + 满血 GPU 并发仍会硬死机**，多租户只是加速因子不是必要条件；安全边界在 M2（48×2 片/chunk、单片 ~1.5ms、几十次零死机）和 M4（12×6 片）之间，没人值得去精确定位。
- **定案**：**NPU 路线到此为止**。M2 稳定但在生产配置下是负收益（−1.1%，见 §6d 复测），M3/M4 收益 +0.3% 量级且会硬死机。全部代码移出主线，存档在 `attic/npu-offload/`（复活方法见该目录 README）。

死机记录（全部 Kernel-Power 41 断电式挂死，均无 WHEA）：1:40 M3-z 多租户 / 1:58 M3-z 多租户 / 2:34 M4-12片 独占 / 2:47 M4-分半 独占（KLD 跑，先 unspecified launch failure）。

---

## 7. 参考

- gufo-npu：`README.md`、`docs/NPU.md`、`src/models/qwen38_flash_next/kernels/rocm/npu_gemm.{hpp,cpp}`、`tools/npu/`；上游否决记录 `git show 35618a8`
- mlir-aie 原生 Windows 指南：https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md
- mlir-aie WSL 指南（gendef 生成 xrt_coreutil.lib）：https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md
- mlir-aie PR #3885（Windows 下 test_utils 的 XRT 辅助、`/Zc:__cplusplus`）：https://github.com/Xilinx/mlir-aie/pull/3885
- Ryzen AI 安装/驱动版本：https://ryzenai.docs.amd.com/en/latest/inst.html ；1.6 发行说明：https://ryzenai.docs.amd.com/en/1.6/relnotes.html
- xrt-smi（Windows pmode）：https://ryzenai.docs.amd.com/en/latest/xrt_smi.html
- XRT userptr API：https://xilinx.github.io/XRT/master/html/xrt_native.main.html
- 死机根因：https://github.com/ROCm/TheRock/issues/8709 ；https://github.com/peonist-ai/halogen-flash-server/issues/139 ；https://github.com/amd/xdna-driver/issues/1605
- HIP Windows gfx1151 host 回调图捕获问题：https://github.com/ROCm/rocm-systems/issues/13096
- Strix Halo 隐藏 BIOS 菜单讨论：https://winraid.level1techs.com/t/request-unlock-bios-uefi-for-amd-395-strix-halo/111395
- 作者另一个项目（Linux 上 NPU+GPU 流水线）：https://github.com/mikealanni/strix-halo-pipeline
