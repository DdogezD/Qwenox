# 交接：Windows 上复刻 gufo-npu 的 GPU+NPU 异构 prefill

> 存档说明：这是 P4 之前的交接快照，§2/§7 的"未测/下一步"已过时；P4–P5（M1–M4）的结果和最终结论见同目录 `NPU-WIN-FEASIBILITY.md` §6c–§6f 与上级 `README.md`。机器信息与踩坑记录仍然有效。

更新：2026-10-17。上一轮的调研和实测细节在 `dev-docs/NPU-WIN-FEASIBILITY.md`（§6b NPU IO、§6c 并发死机、§6d BIOS）。这份只写接手需要的东西。

## 1. 目标

把 gufo-npu（`Workspace/CProject/gufo-npu`，Mike Alani 的 gufo fork，作者已同意复刻）的做法搬进本引擎的 Windows 版：prefill 时把 GPU 暂时用不到的几块稠密 GEMM 交给 XDNA2 NPU 做，GPU 同时做别的。gufo 实测 pp16K +7.2%（1698 → 1821 tok/s），输出与纯 GPU 逐位一致。decode 不用 NPU。

## 2. 当前状态

| 项 | 状态 |
| --- | --- |
| 驱动 / XRT / 编译链 | ✅ 通。gufo 的 xclbin+insts 原样能跑 |
| NPU 单独跑（GPU 空闲） | ✅ 4096×2560×1024 GEMM 0.728 ms（29.5 TOPS），误差 1%，逐位可复现 |
| 零拷贝 | ✅ hipHostMalloc 页 userptr 导入 XRT 成立；反向 hipHostRegister XRT 页也成立 |
| 旧 BIOS 下并发 | ❌ 10-13 GPU prefill 开始约 2 s 硬死机，无 WHEA 前兆 |
| **BIOS 已换** | ✅ 10-16 用户刷了 Bosgame M5 SW1.09（同板 AXB35-02），10-17 核实：AGESA **1.0.0.2b**（AMD 确认的修复版） |
| 新 BIOS 下并发 | ⏳ **未测**。这是下一步 |
| 移植进引擎 | ⏳ 未开始 |

## 3. 机器（NucBox_EVO-X2，Windows 11）10-17 核实

| 项 | 值 |
| --- | --- |
| BIOS | 1.09（2026-05-08，Bosgame）；EC 仍是 GMK 1.06（没刷 EC） |
| AGESA / 微码 | StrixHaloPI-FP11 1.0.0.2b / 0x0B700037；SMU 固件 100.7（由镜像解包得知） |
| 内存 | Micron MT62F4G32D8DV-023 ×8，8000 MT/s；显存 96G 档（hipInfo 107.87 GB），主机内存 31.6 GB |
| 功耗墙（ryzenadj --info） | STAPM 120 W / **快 160 W / 慢 140 W**（GMK 1.05 时是 140/120，新 BIOS 默认更高；prefill 基线要重测） |
| NPU | 驱动 32.0.203.329，XRT 2.19.0，固件 1.0.0.196；**pmode 被 BIOS 重置回 Default**（之前测时是 Performance） |
| Secure Boot | 开（刷 BIOS 后变的）；BitLocker 关；Windows 已激活（OEM:DM，固件无 OA3 密钥） |
| WHEA | 近 2 天 0 条 |

访问：SSH `Mark@192.168.5.12`（PowerShell 5.1），密码见用户；连接脚本 `outputs/devtools/s.sh '<powershell>'`、`cp.sh`（scp，目标写成 `Mark@192.168.5.12:相对家目录路径`）。每次先核 `hostname` = NucBox_EVO-X2。**bash 工具必须给 timeout_ms**，默认 120 s 到点会把 SSH 一起掐断。

## 4. 工具（都没入库）

仓库 `tools/npuprobe/`（未跟踪），主机副本 `~/npuprobe/`（`out/` 里有编好的 exe，`npu/` 里是 gufo 的 xclbin+insts，`xrtinc/` 是 XRT 2.19 头文件）。

| 文件 | 用途 |
| --- | --- |
| `npuio.hip` + `build_run.sh` | NPU 探针 T0–T9（加载、正确性、速度、各种页的 userptr、零拷贝、唤醒延迟）。`npuio.exe ../npu stress <秒> <log> <stopfile>`：4 深流水连续跑并逐位比对，日志每秒落盘 |
| `overlap.sh` | `gpu <len>` / `overlap <len> [tag]` / `loop <len> <n> [tag]`：真实引擎 prefill（v2 hgn，带 pp_win.sh 那串 QWENOX_* 开关）+ NPU stress 同时跑，WHEA 看门狗，日志在 `~/npuprobe/logs/<tag>/` |
| `whea.ps1` | 数 WHEA 事件（须 `-ExecutionPolicy Bypass`） |
| `pmlog.cpp` | ADL PMLog 采样（GFXCLK/MEMCLK/功耗）；`awake` 保持亮屏 |
| `gpuload.hip`、`fclkfind.sh` | 纯 GPU 带宽负载；读 ryzenadj PM table（实时 FCLK 在 0x0578） |

`outputs/devtools/`：`s.sh`/`cp.sh`/`ap.sh`（SSH）、BIOS 解包分析脚本。`outputs/bios_ref/`：wiki 存档的 GMK 1.05/1.12、Bosgame 1.09、EC 1.08 原包。

### 踩过的坑

- XRT 用 **System32\xrt_coreutil.dll**（xrt-smi 那份）。把 DriverStore 那对 DLL 拷到 exe 旁边会开得了设备但 `register_xclbin` 报 "load_axlf: not supported"。
- `device.get_info<name>()` 在这套头文件 + DLL 下段错误，别调。
- 导入库：DLL 导出表 → .def → `llvm-lib`；`xrt/detail/version.h` 要手写；hipcc 编译和链接分两步（`-Xlinker xrt_coreutil.lib`），否则 .lib 被当成源文件。
- 直接跑 `qwenox-engine-win.exe` 不带 pp_win.sh 的开关，prefill 只有 ~800 tok/s（带上 1610–1633）。
- 引擎/pmlog 日志没强制落盘，死机时会丢；只有 npuio 的日志能留下（`_commit`）。
- 主机连不上 GitHub，依赖在 VM 下载后 scp。
- ryzenadj v0.19 对 Strix Halo 不支持 `--min/max-fclk-frequency`（报 not supported 后崩溃）。Windows 没有锁 FCLK 的办法，BIOS 里也没有对应选项。现在靠新 AGESA，不再需要锁。
- VM 的 NAT 会把 ping 回包 TTL 统一改成 64：ping 通不代表主机活着。

## 5. gufo-npu 的做法（要搬的东西）

关键文件（gufo-npu 仓库）：`src/models/qwen38_flash_next/kernels/rocm/npu_gemm.{hpp,cpp}`（XRT 封装，724 行）、`kernels.hip.cpp` 7405–7670（编码/解码 kernel）和 1941–2011（就地读 NPU 输出流）、`device_model.cpp` 474–740（加载时编码权重）、`docs/NPU.md`（开关说明）。

- NPU kernel 固定形状：A 4096 token × K 2560，B K 2560 × N 1024，C 4096 × 1024；全是 v8bfp16ebs8 块浮点（9 字节/8 值：1 字节指数 + 8 个 int8，值 = q·2^(e−127)/64）；tile 128×64×128，A/B/C 各自的分组顺序见 `npuio.hip` 的 `APos/BPos/CPos`（已对拍正确）。
- 三处卸载：
  1. 线性注意力 `ssm_in` 的 z 行：36 层，最后 6×1024 行（`GUFO_NPU_SLICES`）
  2. 全注意力 q/gate 行：12 层，`attn_qkv` 前 8×1024 行，16 个头（`GUFO_NPU_GATE_SLICES`）
  3. 共享专家 gate/up（640 行，补到 1024）：48 层
- 权重在加载时编码进 pinned 页（约 408 块 × 2.95 MB ≈ 1.1 GiB）。
- 调度：GPU 流上 `hipLaunchHostFunc` 把 NPU 任务交给工作线程（`Submit`），GPU 继续跑别的，第一个要用结果的 kernel 前再插一个回调等（`Wait`）；最多 4 个 run 在飞；运行超时 8 s 后标记失败、整个 NPU 路径停用。
- 必须的保活：1 s 一次心跳小任务（实测空闲 6 s 后首次运行 36 ms，12 s 后 69 ms）。
- `GUFO_NPU_WARMUP_MS`：让 NPU 躲开每次 prefill 开头 GPU 从空闲到满载的那段（旧 BIOS 死机正是发生在这一刻）。

## 6. 本引擎的对应位置（初步，未细看）

`src/gpu/parts/40_model.inc`：
- `prefill_chunk()` 6686 / `prefill_batch()` 6620 → `enq_layer_b()` 6016（每层的 prefill 流程）
- GDN 输入投影：`gdn_proj_in_b()` 3428
- QSA 投影：`qsa_proj_in_b()` 4594。我们的 q_proj / k_proj / v_proj 是**分开的**权重（gufo 是合并的 `attn_qkv` 13312 行），q/gate 行要在 `q_proj` 里找对应切片
- 共享专家：`moe_b_shared()` 5859 → `se_gate_up()` 5873
- 激活是 fp32（`d_xb` 等），gufo 是从 f16 编码，编码 kernel 要改输入类型
- 主流 `g_str`；decode 用 HIP Graph（gfx1151 Windows 上 graph 里的 host 回调有 bug，rocm-systems #13096），NPU 只能放在 prefill 的非 graph 路径
- 生产模型 v2 hgn 的这几类张量是什么 dtype 要查（gufo 要求 Q8_0；我们在 GPU 上反量化成 float 再编码即可，任何格式都行）

## 7. 下一步

**P4：新 BIOS 下的并发验证（必须先问用户，可能要断电重启）**
1. `xrt-smi configure --pmode performance`（已被 BIOS 重置为 Default）
2. `bash ~/npuprobe/overlap.sh gpu 32768 base_bg109`：新 BIOS 的纯 GPU 基线（功耗墙变了）
3. `bash ~/npuprobe/overlap.sh overlap 32768 ov_bg109`：和 10-13 死机时完全相同的负载；再用 `loop 32768 5` 跑几分钟
4. 用 `bg.ps1` 起成后台任务，SSH 只读日志；看 NPU ms/run 被挤到多少、GPU prefill 慢了多少
5. 稳定后再进 P5；若仍死机，复刻在 Windows 上到此为止

**P5：移植**（顺序建议）
1. `NpuGemm` 封装照搬（Windows 改：trace 用 `_commit`，路径分隔，DLL 用 System32 那份）
2. 编码 kernel（fp32 激活 → bfp16 A 流；权重加载时一次性编码）、解码 kernel（C 流 → fp32，或在 epilogue 里就地读）
3. 先只接共享专家 gate/up（形状最干净，不涉及切片），开关默认关，做 KLD 与贪心一致性对比
4. 再接 GDN z 行、QSA q/gate 行；测 pp8K/32K/128K
5. 收益口径：gufo 在 pp16K 上 +7.2%；本引擎 prefill 绝对速度与它同量级，预期 +5~8%；并发服务时 NPU 会抢内存带宽，decode 中的其他请求可能变慢，要测

## 8. 用户偏好（这个项目相关）

- 回复用中文；给一次性跑完、以 PASS/FAIL 结尾的脚本，不让用户手动多步操作
- 超过 1 分钟的任务先说明做什么、为什么、要多久；不轮询进度，后台跑完等用户说
- 并发测试有死机风险，跑之前一定先问
- 刷 BIOS 这类需要人在场的操作，只给步骤，不远程执行
