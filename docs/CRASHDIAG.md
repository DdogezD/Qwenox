# 崩溃诊断

引擎或 API 在别人的机器上崩溃时，本机通常没法复现，只能靠对方发回来的材料定位。Qwenox 为此做了三件事：
进程崩溃时自己写一份**崩溃报告**（Windows 附带一个小 **minidump**）；运行中持续记录**飞行记录**，崩溃时一并写出；
API 把排查需要的东西打成一个**诊断包**（zip），对方只要把这个文件发回来。

## 用户怎么做

**Windows（托盘程序）**

- 引擎意外退出时，API 会自动生成诊断包，并弹出提示"引擎意外退出，已生成诊断包"。点这个提示会在资源管理器里
  选中那个 zip，把它发给开发者即可。
- 没有崩溃但觉得不对劲（卡住、变慢、输出异常）时，右键托盘图标 →"导出诊断包…"。几秒后资源管理器会打开并选中
  生成的 zip。引擎在运行时，包里还会多一份它当前的在线快照。
- 诊断包都在 `logs\diag\`，只保留最近 10 个。崩溃报告在 `logs\crash\`。

**Linux（start_hgn.sh / start_gguf.sh）**

- 引擎意外退出时，启动脚本会自动生成诊断包，并在终端打印路径。
- 手动生成：`build/qwenox-api --collect-diag`。不监听端口，打完包后打印 zip 路径，引擎正在跑也可以用。

**HTTP**：`POST /admin/diag` 生成诊断包，返回 `{path, name, bytes, url}`；`GET /admin/diag/file?name=…`
下载。两个接口都要管理密钥（`QWENOX_API_ADMIN_KEY`，没设密钥时不校验，与其它 `/admin` 接口一样）。

## 诊断包里有什么

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 打包原因、引擎退出码及其含义、各份崩溃报告的要点、包内文件清单 |
| `engine.log` / `api.log` | 引擎 / API 日志（引擎取开头 512 KB + 结尾 8 MB，API 取开头 256 KB + 结尾 4 MB） |
| `crash/*.txt` / `crash/*.dmp` | 本次引擎进程的崩溃报告与 minidump，外加时间窗内的其它报告（最多 6 份） |
| `crash/*.sym.txt` | Linux：用 `addr2line` 把报告里的栈帧还原成函数名与源码行 |
| `engine-live.txt` | 打包时引擎还在运行：经 `DIAG` 动词取回的飞行记录与最近 HIP 调用点 |
| `engine-live.dmp` | Windows，引擎由本 API 拉起且在运行：在线 minidump（看卡死时各线程停在哪） |
| `system.txt` | 系统版本、CPU、内存、显卡与驱动版本、TDR 设置、电源计划、GPU 相关环境变量 |
| `events.txt` | Windows 事件日志：显卡驱动复位（Display 4101）、意外重启/睡眠唤醒（Kernel-Power）、蓝屏、低内存、WHEA 硬件错误、程序崩溃/无响应（Application Error / WER / LiveKernelEvent） |
| `kernel.txt` | Linux：内核日志里与 amdgpu/kfd/drm/OOM/段错误/挂起相关的行 |
| `service.conf` | 配置；名字含 KEY / TOKEN / SECRET / PASSWORD / AUTH 的值打码为 `***` |
| `logs-index.txt` | `logs/`、`logs/crash/`、`logs/diag/` 的文件清单 |

事件日志与内核日志的时间窗覆盖引擎整个运行期，至少 72 小时，最多 7 天。

**隐私**：包里没有请求或回复的正文、模型文件、KV 缓存。minidump 只含线程栈、模块列表和栈上直接引用的少量内存，
其中可能夹带正在处理的文本片段；介意的话删掉 `crash/*.dmp` 与 `engine-live.dmp` 再发，`.txt` 报告已包含主要信息。

## 崩溃报告怎么读

文件名 `crash-<程序>-<pid>-<时间>.txt`，程序是 `engine` 或 `api`。从上到下：

- `kind` / `code` / `signal`：崩溃类别。Windows 是异常码（`0xC0000005` 访问冲突、`EXCEPTION_STACK_OVERFLOW` 栈溢出…），
  Linux 是信号与 `si_code`（`SEGV_MAPERR` = 地址没映射，`SEGV_ACCERR` = 有映射但权限不对）。`abort` / `terminate` /
  `fatal` 是程序主动结束（未捕获的 C++ 异常、HIP 调用失败、KV 页池耗尽等），`message` 写明原因。
- `last C++ exception thrown`（Windows）：最近一次抛出的 C++ 异常，abort / terminate 类报告靠它看出原因。
- `thread`：出错线程，标出是否主线程；Windows 另给出线程入口所在模块（例如 `amdhip64_7.dll+…` 就是 HIP 运行时自己的工作线程）。
- `fault`：访问冲突的读 / 写 / 执行，以及被访问的地址。
- `fault-region`：该地址当时的内存状态，这是区分问题类型的关键：
  - `FREE` / `UNMAPPED`：地址根本没映射，或者映射已经释放。如果前后相邻区里有一块大小合适的映射，多半是越界或释放后使用；
    如果地址完全不像正常指针（很小、或高位是乱码），多半是指针本身被写坏了。
  - `COMMIT … protect=READONLY`（或 `r--p`）：映射还在但只读，写它就会出错。
  - 显存 / 共享内存映射失效（例如驱动复位之后）时，常见的现象是地址落在一段刚好不存在的区间里，同时 `events.txt` 里有 Display 4101。
- `pc` / `pc-region`：出错指令所在的模块与偏移。
- `registers`、`code bytes`、`stack`：寄存器、出错指令附近的机器码、栈回溯。Windows 用 `.pdata` 展开；Linux 用 `backtrace()`。
  `pc` 不在可执行映射里时（跳到了坏地址），Linux 跳过回溯，只给 `stack scan`。
- `stack scan`：栈上指向代码区的值，栈展开失败时可以从这里找线索。
- `first-chance …`（Windows）：最近出现过的致命类异常，包括被接住的。
- `last HIP calls`：引擎最近的 HIP 调用点（`CK(...)` 宏所在的文件:行与表达式），可以看出崩溃前在做什么拷贝或同步。
- `flight recorder`：最近 512 条事件，包括请求进槽、命中缓存、prefill 进度、解码进度、kvsnap / rckpt 存取、图捕获、连接进出。
- `environment`：启动时的系统、驱动、电源、TDR、HIP 设备与 GPU 栈 DLL 版本、`QWENOX_*`/`HIP_*` 等环境变量。
- `loaded modules`（Windows）/ `/proc/self/maps`（Linux）：模块地址表，用来把地址换成"模块+偏移"。

引擎日志里同时会有两行摘要：`[crashdiag] CRASH …` 与 `[crashdiag] report: <路径>`。

## 开发者：把 minidump 还原成源码行（Windows）

`build_win.sh` 默认给引擎和 API 生成 PDB（`build/qwenox-engine-win.pdb`、`build/qwenox-win.pdb`）。PDB 只对应**同一次编译**
的 exe，用户分发包里不带，所以每次发版后都要把它另存下来：

```bash
bash tools/pack_symbols_win.sh   # → dist/qwenox-symbols-<版本>-<git>.zip（PDB + exe + GPU 运行库 DLL + BUILD-INFO.txt）
```

收到 `.dmp` 后，解开对应版本的符号包，用 WinDbg 或 cdb 打开：

```
cdb -z crash-engine-1234-20261022-103433.dmp -y "<符号包目录>;srv*%LOCALAPPDATA%\symbols*https://msdl.microsoft.com/download/symbols" -c ".ecxr; kv; ~*k; q"
```

`.ecxr` 切到出错现场，`kv` 是出错线程的栈，`~*k` 是所有线程的栈。崩溃报告也已经嵌在 dump 的注释流里（WinDbg 的 `.dumpdebug` 可以看到）。
AMD 的 DLL 没有公开符号，栈里落在 `amdhip64_7.dll` 内的帧只能看到"模块+偏移"；用 `BUILD-INFO.txt` 里的 PE 时间戳核对是同一版本的 DLL，
再反汇编该偏移所在的函数（`dumpbin /disasm` 或 `objdump -d --start-address=…`），结合导出表和内嵌的源码路径字符串判断是哪段运行时代码。

Linux 下 `build.sh` 给引擎主机代码加 `-g1`（只有行号表，GPU kernel 与代码生成不变，`QWENOX_HOST_DEBUG=0` 关闭），诊断包里的
`crash/*.sym.txt` 已经是还原好的函数与源码行。

## 自测

分发前，或者在一台新机器上第一次运行时，用一条命令验证整条链路：

- Windows（Git Bash，先 `bash build_win.sh all`）：`bash tools/crashdiag_test_win.sh`。不加载模型，约 1–2 分钟。会逐种制造崩溃，
  检查报告和 minidump；再打一个诊断包检查内容；然后让 API 拉起一个启动即崩溃的引擎，检查自动打包。最后一段需要引擎端口
  8730 空闲，所以测试前先退出托盘里的 Qwenox，否则跳过该段。最后一行是 `CRASHDIAG WIN TEST: PASS / FAIL`。
- Linux：`bash tools/crashdiag_test.sh`。前两段不需要 GPU（编一个独立测试程序逐种崩溃，再打一个诊断包），有 `build/` 产物时
  再测引擎和 API。最后一行是 `CRASHDIAG LINUX TEST: PASS / FAIL`。

单独触发一次崩溃：`qwenox-engine --crash-test <kind>`，kind 为 `write read exec abort throw stack thread fatal hipfatal`。
也可以用环境变量 `QWENOX_CRASH_TEST=<kind>` 配合 `QWENOX_CRASH_TEST_AT=start|ready|gen`，分别在启动后、模型就绪后、第一个请求进槽时崩溃。

## 开关

| 设置 | 作用 |
| --- | --- |
| `QWENOX_CRASH_DIAG=0` | 不安装崩溃处理（报告、minidump、飞行记录都没有） |
| `QWENOX_CRASH_DUMP=0` / service.conf `CRASH_DUMP=0` | 只写文本报告，不写 `.dmp` |
| `QWENOX_CRASH_DIR` | 崩溃报告目录（默认 `<项目根>/logs/crash`） |
| `QWENOX_PDB=0`（编译时） | `build_win.sh` 不生成 PDB |
| `QWENOX_HOST_DEBUG=0`（编译时） | `build.sh` 不给引擎主机代码加行号表 |

## 实现要点与取舍

- 代码在 `src/crashdiag.h`（引擎与 API 共用，header-only）与 `src/api/diagpack.cpp`。
- **Windows 的报告由未处理异常过滤器写**：只有没人接住的异常才会到这里，运行库自己用 `__try` 接住的探测性访问冲突不会产生误报。
  amd_comgr 里的 LLVM 在 HIP 初始化时会把顶层过滤器换成它自己的，而且不往下传，所以引擎在 HIP 初始化之后、模型就绪时、
  每个请求进槽时，以及每 1024 次 HIP 调用，都会把我们的过滤器装回顶层；写完报告再调用 LLVM 的过滤器，它的栈打印仍会出现在日志里。
  栈溢出和堆损坏不可能被正常接住，由向量化异常处理当场写。
- **未捕获的 C++ 异常**：MSVC 的 terminate 处理器是每线程的，工作线程里没接住的异常会直接 abort，报告里看不到原因。
  所以向量化异常处理在每次 C++ 抛出时记下类型名与 `what()`，报告里给出 `last C++ exception thrown`。
- **写报告的工作交给预先创建的线程**：栈溢出时出错线程没有余下的栈可用，minidump 也应该由别的线程来写。报告分三段落盘：
  先写核心信息（不取任何锁），再写模块表，最后写 minidump，所以即使某一步卡住，前面的内容也已经保存。
- **Linux 信号处理只用异步信号安全的调用**（`open`/`write`/`read`，数字格式化是自己写的），在备用栈上运行。写完后返回，让出错指令重新执行，
  进程按默认动作结束，退出码和 core 不受影响。
- **minidump 保持很小**：只含线程信息、模块表、栈与栈上直接引用的内存，没有进程里那 90 多 GB 的权重和 KV。
- **飞行记录是无锁环形缓冲**：每条 120 字节，记一条约几百纳秒，只在请求级和秒级的位置埋点。HIP 调用点每次只记一个编译期字符串指针和时间戳。
- **后台进程不弹"已停止工作"对话框**：否则崩溃进程会停在对话框上不退出，监管方拿不到退出码。Windows 错误报告（WER）照常记事件日志。
- **`__fastfail`（`0xC0000409`）不经过任何进程内处理器**，这种情况只有退出码，以及 `events.txt` 里 Application Error 1000 给出的出错模块与偏移。
