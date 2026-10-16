# attic：遗弃代码库

这里放被取代或因收益、稳定性没进主线的代码。都不参与任何构建；代码依赖的接口日后可能漂移，不保证还能直接编译。

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| `launcher/` | 旧 Windows 托盘启动器 | 已被取代，仅备份 |
| `npu-offload/` | XDNA2 NPU 卸载 prefill GEMM（Windows，gufo-npu 复刻） | 生产配置下无收益（M2 −1.1%），高占空并发在 AGESA 1.0.0.2b 下仍硬死机；自带 README 写明复活条件与步骤 |

## launcher — 旧 Windows 托盘启动器（备份，不再构建）

`start_win.exe` 的旧实现：原生 Win32 托盘程序 + 启动配置面板（launch_panel.inc）。
2026-10 起被取代：托盘能力移入 `gdec-api-win.exe`（`src/api/tray_win.cpp`），
配置面板由网页控制台（`#/engine` 页）取代，根目录 `start_win.exe` 变为最小
转发 stub（`src/start_stub_win.cpp`）。

此处仅为备份，不参与任何构建；代码依赖的 `svcconf` 等接口日后可能漂移，
不保证还能编译。

## npu-offload

见 `npu-offload/README.md`。
