# 遗弃：XDNA2 NPU 卸载 prefill GEMM（Windows）

prefill 时把一部分稠密 GEMM 交给 Ryzen AI NPU（XDNA2）算，GPU 同时做别的。思路和 NPU kernel 来自 gufo-npu（Mike Alani 的 gufo fork，作者同意复刻），它在 Linux 上锁住 FCLK 后 prefill +8~10%。

这套代码在本引擎的 Windows 版上已经完整接通、结果正确，但**没有可用的收益，高占空时还会让整机硬死机**，所以移出主线、存档在这里。主线引擎和 `build_win.sh` 里已经没有任何 NPU 代码。

## 为什么放弃

| 卸载点 | 开关 | 生产配置实测 | 稳定性 |
| --- | --- | --- | --- |
| M2 共享专家 gate/up（48 层，每 chunk 前 4096 行） | `QWENOX_NPU=1` | 32K prefill **−1.1%**（24.21 → 24.47 s，3 对全慢） | 稳定，从未死机 |
| M3 GDN z 门控（36 层 × 6 片） | `QWENOX_NPU_Z=1` | 整 chunk +0.36% | 硬死机 2 次 |
| M4 QSA q/gate（12 层 × 12 片，分两半提交） | `QWENOX_NPU_QG=1` | 整 chunk +0.29% | 硬死机 2 次 |

- **收益被带宽吃掉**：本引擎 prefill 的 GPU GEMM 是显存带宽瓶颈，不是算力瓶颈。NPU 让出来的算力换不回时间，它的 A/C DMA 还要和 GPU 抢同一条 LPDDR，GPU 侧又多了编码/解码两个 kernel。gufo 的收益大头在 z 和 q/gate（共享专家只占它 NPU 工作量的 96/408），恰好就是这里会死机、不死机时也只有 +0.3% 的 M3/M4。
- **固件问题没修干净**：BIOS 已升到 Bosgame M5 1.09（AGESA StrixHaloPI-FP11 1.0.0.2b，AMD 说修了 ROCm/TheRock#8709）。低占空（M2，单次 NPU 运行约 1.5 ms）确实稳定；单次连续 NPU 负载到 4–9 ms、同时 GPU 满载时，仍然会断电式挂死（Kernel-Power 41，事前没有任何 WHEA 记录）。Windows 后台 AI 索引（`WorkloadsSessionHost.exe`）也在用 NPU，会加速死机，但不是必要条件。
- **Windows 没有退路**：gufo 在 Linux 上靠 `pp_dpm_fclk` 锁 FCLK 规避死机、减轻带宽抖动。Windows 上没有对应手段，ryzenadj 不支持这颗芯片的 FCLK 选项，BIOS 里也没有这个设置。

完整的调研、每一步的实测和死机记录见 `docs/NPU-WIN-FEASIBILITY.md`（§6b IO/零拷贝，§6c 死机，§6d–§6f M2–M4）。`docs/HANDOFF-NPU-WIN.md` 是 P4 之前的交接快照，机器信息和踩坑记录仍然有用。

## 什么情况下值得复活

满足下面任意一条，再回来看：

- AMD/主板厂商发布了新 AGESA，`tools/npucheck` 的高占空并发压测能长时间不死机；
- Windows 上出现了锁 FCLK 的办法；
- 引擎的 prefill GEMM 变成算力瓶颈（例如权重格式或 kernel 大改后 GEMM 不再吃满带宽），NPU 让出的算力才能换成时间；
- 改用 f16 激活直接编码，并在消费 kernel 里就地读 NPU 输出（像 gufo 那样），省掉单独的编码/解码 kernel 和一半内存流量。

## 目录

| 路径 | 内容 |
| --- | --- |
| `revive.patch` | 主线代码的全部 NPU 改动：`src/gpu/parts/29_npu.inc`（XRT 封装 + GPU 侧块浮点编解码 kernel，新文件）、`40_model.inc` 里 M2/M3/M4 的挂钩（全部在 `#ifdef QWENOX_NPU` 里）、`qwenox.cpp` 的 include、`build_win.sh` 的 `npu` 目标与 XRT 资产检测、`build_win.bat` 的用法说明 |
| `tools/npucheck/` | NPU/BIOS 检测工具：GUI 版 `npucheck-win.exe` 与脚本版 `npucheck.sh`（info / stress / postmortem / whea），带崩溃取证和 NPU 结果逐位校验 |
| `tools/npuprobe/` | 探针：`npuio.hip`（NPU 加载/正确性/速度/零拷贝/唤醒延迟，含 `stress` 连续压测）、`overlap.sh`（真实引擎 prefill + NPU 压测并发，带 WHEA 看门狗）、`m2ab.sh`（M2 生产配置 A/B）、`gpuload.hip`、`fclkfind.sh`、`whea.ps1`、`build_run.sh` |
| `docs/` | 调研报告与交接文档 |

功耗采样工具 `pmlog.cpp` 与 NPU 无关（`docs/POWER.md` 引用），仍在主线的 `tools/npuprobe/`。

## 复活步骤

在仓库根目录：

```bash
git mv attic/npu-offload/tools/npucheck tools/npucheck
git mv attic/npu-offload/tools/npuprobe/* tools/npuprobe/
git apply attic/npu-offload/revive.patch
```

补丁是对着存档那一刻的主线生成的，之后 `40_model.inc` 改动较多时可能要手工合并（挂钩点都在 `moe_b_v2` / `moe_b_shared_ey` / `gdn_proj_in_b` / `qsa_idx_b` 附近，用 `#ifdef QWENOX_NPU` 搜索）。`.gitignore` 里 `npucheck` 的 `out/` 规则也要改回 `tools/npucheck/out/`。

然后在 Windows 上：

1. 准备 XRT 资产目录（默认 `~/npuprobe`，`NPUASSETS=` 覆盖）：`xrtinc/`（XRT 2.19 头文件，`xrt/detail/version.h` 要手写）、`xrt_coreutil.def`（从 `System32\xrt_coreutil.dll` 导出表生成）、`npu/final_4096x2560x1024_128x64x128.xclbin` 与 `npu/insts_4096x2560x1024_128x64x128.txt`（gufo-npu 的 `tools/npu/` 里有）。
2. `bash build_win.sh engine`：检测到资产就编入 NPU（`-DQWENOX_NPU=1`），`xrt_coreutil.dll` 走 delay-load，没装 NPU 驱动的机器照常运行；`QWENOX_BUILD_NPU=0` 强制不编入。`bash build_win.sh npu` 编检测工具。
3. 运行时 `QWENOX_NPU=1` 开 M2，`QWENOX_NPU_Z=1` / `QWENOX_NPU_QG=1` 加开 M3/M4；`QWENOX_NPU_MINP` 调 NPU 生效的最小行数（默认 4097），`QWENOX_NPU_TIMEOUT_MS`（默认 8000）、`QWENOX_NPU_KEEPALIVE=0` 关 1 s 心跳。
4. 跑并发之前先用 `tools/npucheck/npucheck.sh info` 确认固件，并确认 `xrt-smi examine --report aie-partitions` 显示没有别的进程占着 NPU。**并发压测可能硬死机，要有人在机器旁边。**

## 踩过的坑（Windows XRT）

- 必须用 `System32\xrt_coreutil.dll`。把 DriverStore 里那对 DLL 拷到 exe 旁边能打开设备，但 `register_xclbin` 报 "load_axlf: not supported"。
- `device.get_info<name>()` 在 2.19 头文件 + 这份 DLL 下段错误，别调。
- hipcc 会把 `.lib` 当 HIP 源，导入库要用 `-Xlinker xxx.lib`，编译和链接分两步；Git Bash 会改写 `/` 开头的链接器参数，用 `-` 前缀。
- NPU 空闲 6 s 后首次运行要 36 ms、12 s 后 69 ms，必须有 1 s 心跳保活。
- instr BO 用 `group_id(1)`，数据 BO 用 `group_id(3)`；`hipHostMalloc(Mapped)` 的页可以直接 userptr 导入 XRT，零拷贝成立。
- 不要把 NPU 回调放进 HIP Graph（gfx1151 Windows 上 graph 里的 host 回调看不到前序写入，rocm-systems#13096）；NPU 只用在 prefill 的非 graph 路径。
