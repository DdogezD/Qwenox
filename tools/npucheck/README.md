# npucheck — Strix Halo NPU 检测工具

回答两个问题：

1. **当前 BIOS 的 AGESA 是否修复了 GPU+NPU 并发死机缺陷？**（阈值：AGESA ≥ 1.0.0.2b，依据 ROCm/TheRock#8709，AMD 2026-10-06 答复；详见 `dev-docs/NPU-WIN-FEASIBILITY.md` §3）
2. **（可选，有硬死机风险）实测 GPU+NPU 并发到底会不会死机。**

## 用法（Git Bash）

**构建**（仅 Windows，产物进 `build/`，与其他产物并列）：

```bash
bash build_win.sh npu     # 或双击 build_win.bat 带参数 npu
```

**GUI 版（推荐）**：`build/npucheck-win.exe` —— 双击即用，单文件 Win32 窗口
（xclbin/insts 已内嵌；XRT DLL 为 delay-load，没装 NPU 驱动也能打开做 BIOS 检查）。
按钮：[系统检查] [NPU 基线] [GPU 基线] [并发测试]（AGESA 未修复时双重确认）[死机分析]。
带秒级心跳落盘、WHEA 看门狗、[中止] 按钮。命令行自测：
`npucheck-win.exe --selftest[-npu N | -gpu N]`（不弹窗，输出到控制台，供脚本用）。

**脚本版**（自动使用 `build/` 里的 npuio.exe/gpuload.exe）：

```bash
bash tools/npucheck/npucheck.sh info            # 只读检查：BIOS/AGESA/NPU 驱动 → SAFE / UNSAFE / UNKNOWN
bash tools/npucheck/npucheck.sh stress --npu-only 60   # NPU 独占基线（安全）
bash tools/npucheck/npucheck.sh stress --gpu-only 30   # GPU 显存拷贝基线（安全）
bash tools/npucheck/npucheck.sh stress --force         # GPU+NPU 并发 60s —— 未修复固件上会死机！
bash tools/npucheck/npucheck.sh stress --force --secs 120 --engine 32768   # GPU 负载换成真实引擎 prefill
bash tools/npucheck/npucheck.sh postmortem      # 死机重启后：分析上次压测死在哪
bash tools/npucheck/npucheck.sh whea            # 最近 1 小时的 WHEA-Logger 事件
```

`info` 的退出码：0=SAFE，1=UNSAFE/UNKNOWN（可供脚本判断）。

## 原理

- **AGESA 读取**：原始 SMBIOS 表（WMI `root\wmi MSSmBios_RawSMBiosTables`）里直接带
  `AGESA!V9 StrixHaloPI-FP11 x.y.z.w` 字符串，无需 HWiNFO、无需进 BIOS 设置页。
- **崩溃取证**：stress 每个阶段前把 `logs/<tag>/MARKER`（时间、阶段）写盘；机器硬死后
  文件还在，重启后 `postmortem` 结合 Kernel-Power 41（非正常关机）事件和 npu.log
  最后心跳（秒级 flush + `_commit`）判定死机时刻与阶段。
- **提前中止**：并发阶段每秒查 WHEA-Logger，出现新事件立即停两个负载（#8709 在 Linux
  上死机前几秒有 deferred MCE 预警；注意本机 Windows 实测死机时 **没有** WHEA 预警）。
- **结果校验**：NPU 每次 GEMM 输出都与参考逐位比较，防"没死机但结果静默损坏"。

## 已知事实表（更新 BIOS 后先跑 info 核对）

| AGESA | 状态 |
| --- | --- |
| 1.0.0.0c | 未修复（Beelink GTR9 Pro，锁 FCLK 仍死） |
| 1.0.0.1b | 未修复（GMKtec EVO-X2 BIOS 1.05，本机 2026-10-13 实测死机） |
| 1.0.0.2 | 未修复（GMKtec EVO-X2 BIOS 1.13，#8709 报告机） |
| ≥ 1.0.0.2b | 已修复（AMD 官方说法；仍需实测确认） |

## 依赖

- NPU 驱动（带 `C:\Windows\System32\AMD\xrt-smi.exe`）、TheRock hipcc（`THEROCK` 环境变量）。
- XRT 2.19 头、`xrt_coreutil.def`、gufo-npu 的 xclbin/insts：默认取自 `~/npuprobe`，
  可用 `NPUASSETS=<目录>` 覆盖（该目录需含 `xrtinc/`、`xrt_coreutil.def`、`npu/`）。
- `--engine` 模式需要引擎仓库（`ENGROOT`，默认 `~/Workspace/gfx1151-engine`）里有
  `build/qwenox-engine-win.exe`、`models/qwen38-flash-next-{v2,ngram}.hgn` 和
  `data/qsa-oracle/<len>.tokens`。

## 警告

`stress --force` 在 AGESA 未修复的机器上**预期会硬死机**，只能断电重启。跑之前保存好
所有工作。本机的死机实录见 `dev-docs/NPU-WIN-FEASIBILITY.md` §6c。
