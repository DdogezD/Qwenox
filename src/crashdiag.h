// crashdiag.h — 进程内崩溃诊断（引擎与 API 共用，header-only）。
//
// 进程崩溃时在 <QWENOX_CRASH_DIR>（默认 <工作目录>/logs/crash）写一份文本报告
//   crash-<程序>-<pid>-<YYYYmmdd-HHMMSS>.txt
// Windows 另写同名 .dmp（小 minidump：各线程栈 + 模块表 + 内存区信息，几 MB，
// 不含进程的大块内存；文本报告也作为注释流嵌在 .dmp 里）。
//
// 报告内容：异常码 / 信号、出错指令所在模块+偏移、读写标志与被访问地址、
// 该地址的内存区状态（未映射 / 已释放 / 只读 …）、寄存器、栈回溯、最近的
// HIP 调用点（CK 宏记录）、飞行记录（note() 环形缓冲，最近 512 条）、
// 启动时的环境快照（系统、驱动、电源、QWENOX_* 环境变量），以及模块列表。
//
// Windows：报告由"未处理异常过滤器"写——只有没人接住的异常才到这里（运行库
//   用 __try 自己接住的探测性访问冲突不会误报）。amd_comgr 里的 LLVM 加载时会
//   把顶层过滤器换成它自己的且不往下传，所以 rearm() 定期把我们装回顶层（CK
//   调用点每 1024 次、模型就绪时），写完报告再调用它（它的栈打印照常保留）。
//   现场拷到静态区后交给预先创建的写报告线程（栈溢出时出错线程的栈不够用；
//   minidump 也应由别的线程写）。向量化异常处理（VEH）只做两件事：栈溢出 /
//   堆损坏当场写报告（这两种不会被正常接住）；其余致命码记进"first-chance"
//   小环（报告里列出，帮助判断）。另接 SIGABRT、std::terminate、CRT 无效参数。
//   __fastfail（0xC0000409）不经过任何进程内处理器，只能靠退出码判断。
// Linux：sigaction(SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT)，SA_ONSTACK（主线程
//   备用栈）+ SA_RESETHAND；处理函数只用异步信号安全的调用（open/write/read，
//   自带数字格式化），写完返回，故障指令重新执行后按默认动作结束（core）。
//   SIGTERM 只在 stderr 记一行发送者 pid。
//
// 开关：QWENOX_CRASH_DIAG=0 整体不装；QWENOX_CRASH_DUMP=0 不写 .dmp；
//       QWENOX_CRASH_DIR 改目录。每个进程只写第一份报告。
#pragma once

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <cwctype>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>  // 须先于 windows.h（与 engine_net.h / os_win32.h 共存）
#include <windows.h>
#include <dbghelp.h>
#include <csignal>
// 注册表（驱动版本 / TDR / 系统版本）；dbghelp / version / powrprof 运行时动态加载
#ifdef _MSC_VER  // TheRock clang（MSVC 目标）与 MSVC；mingw 目标由链接命令自带
#pragma comment(lib, "advapi32")
#endif
#else
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
extern char** environ;
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CRASHDIAG_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define CRASHDIAG_PRINTF(a, b)
#endif

namespace crashdiag {

// ============================================================================
// 时钟 / 线程 id（热路径可用：note()/site() 每次各调一次）
// ============================================================================

// UTC 微秒（墙钟）。
inline uint64_t wall_us() {
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  const uint64_t t = (uint64_t(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
  return (t - 116444736000000000ULL) / 10;
#else
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return uint64_t(ts.tv_sec) * 1000000ULL + uint64_t(ts.tv_nsec) / 1000;
#endif
}

#ifndef _WIN32
inline thread_local uint32_t t_tid = 0;
#endif
inline uint32_t tid() {
#ifdef _WIN32
  return GetCurrentThreadId();
#else
  if (!t_tid) t_tid = uint32_t(syscall(SYS_gettid));
  return t_tid;
#endif
}

// ============================================================================
// 飞行记录：最近 kRing 条短消息（带时间与线程 id），崩溃时整段写进报告。
// 无锁：并发写同一槽（绕圈）时以 seq 判定残缺条目并跳过。
// ============================================================================
constexpr int kRing = 512;
constexpr int kMsg = 120;
struct Entry {
  std::atomic<uint64_t> seq;
  uint64_t t_us;
  uint32_t tid;
  char msg[kMsg];
};
inline Entry g_ring[kRing];
inline std::atomic<uint64_t> g_ring_n{0};

inline void vnote(const char* fmt, va_list ap) {
  const uint64_t i = g_ring_n.fetch_add(1, std::memory_order_relaxed);
  Entry& e = g_ring[i % kRing];
  e.seq.store(2 * i + 1, std::memory_order_relaxed);
  std::atomic_signal_fence(std::memory_order_seq_cst);
  e.t_us = wall_us();
  e.tid = tid();
  vsnprintf(e.msg, kMsg, fmt, ap);
  e.seq.store(2 * i + 2, std::memory_order_release);
}
inline void note(const char* fmt, ...) CRASHDIAG_PRINTF(1, 2);
inline void note(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vnote(fmt, ap);
  va_end(ap);
}

// ============================================================================
// 最近的 HIP 调用点（引擎 CK 宏每次调用前记一条；字符串是编译期字面量）。
// ============================================================================
inline void rearm();  // Windows：把未处理异常过滤器装回顶层（见文件头）
constexpr int kSites = 64;
struct Site {
  std::atomic<const char*> s;
  std::atomic<uint64_t> t_us;
  std::atomic<uint32_t> tid;
};
inline Site g_sites[kSites];
inline std::atomic<uint32_t> g_sites_n{0};
inline void site(const char* s) {
  const uint32_t n = g_sites_n.fetch_add(1, std::memory_order_relaxed);
  if ((n & 1023) == 0) rearm();
  const uint32_t i = n % kSites;
  g_sites[i].s.store(s, std::memory_order_relaxed);
  g_sites[i].t_us.store(wall_us(), std::memory_order_relaxed);
  g_sites[i].tid.store(tid(), std::memory_order_relaxed);
}

// ============================================================================
// 进程状态（install 时填好；崩溃路径只读）
// ============================================================================
struct State {
  bool installed;
  bool dump;
  char program[32];
  char version[96];
  char dir[1024];  // UTF-8，报告里显示用
  char exe[1024];  // UTF-8
  int pid;
  uint32_t main_tid;
  uint64_t t0_us;
  int64_t tzoff_ms;  // 本地时间 - UTC
  char env[48 * 1024];
  size_t env_len;
#ifdef _WIN32
  wchar_t wdir[1024];
  const unsigned char* exe_base;
#endif
};
inline State g;

// 启动环境快照（崩溃报告末尾原样附上）。append 只在启动阶段调用（单线程）。
inline void append_env(const std::string& s) {
  const size_t room = sizeof g.env - 1 - g.env_len;
  const size_t n = s.size() < room ? s.size() : room;
  memcpy(g.env + g.env_len, s.data(), n);
  g.env_len += n;
  g.env[g.env_len] = 0;
}

inline bool installed() { return g.installed; }
inline const char* dir() { return g.dir; }

// ============================================================================
// 报告输出：静态缓冲 + 自带格式化（不分配内存，Linux 信号处理里可用）
// ============================================================================
#ifdef _WIN32
using fd_t = HANDLE;
inline const fd_t kNoFd = INVALID_HANDLE_VALUE;
#else
using fd_t = int;
inline const fd_t kNoFd = -1;
#endif

inline void raw_write(fd_t fd, const char* p, size_t n) {
  if (fd == kNoFd) return;
  while (n) {
#ifdef _WIN32
    DWORD w = 0;
    if (!WriteFile(fd, p, DWORD(n > (1u << 30) ? (1u << 30) : n), &w, nullptr) || !w) return;
#else
    const ssize_t w = ::write(fd, p, n);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return;
#endif
    p += w;
    n -= size_t(w);
  }
}

inline fd_t stderr_fd() {
#ifdef _WIN32
  return GetStdHandle(STD_ERROR_HANDLE);
#else
  return 2;
#endif
}

class Out {
 public:
  Out(char* buf, size_t cap, fd_t fd, char* mirror = nullptr, size_t mcap = 0)
      : buf_(buf), cap_(cap), fd_(fd), mirror_(mirror), mcap_(mcap) {}
  ~Out() { flush(); }
  Out& c(char ch) {
    if (n_ == cap_) flush();
    buf_[n_++] = ch;
    if (mirror_ && mn_ + 1 < mcap_) mirror_[mn_++] = ch;
    return *this;
  }
  Out& s(const char* p) {
    while (p && *p) c(*p++);
    return *this;
  }
  Out& sn(const char* p, size_t n) {
    for (size_t k = 0; k < n && p[k]; k++) c(p[k]);
    return *this;
  }
  Out& u(uint64_t v) {
    char t[24];
    int k = 0;
    do {
      t[k++] = char('0' + v % 10);
      v /= 10;
    } while (v);
    while (k) c(t[--k]);
    return *this;
  }
  Out& i(int64_t v) {
    if (v < 0) {
      c('-');
      return u(uint64_t(0) - uint64_t(v));
    }
    return u(uint64_t(v));
  }
  Out& x(uint64_t v, int width = 0) {
    s("0x");
    char t[16];
    int k = 0;
    do {
      t[k++] = "0123456789ABCDEF"[v & 15];
      v >>= 4;
    } while (v);
    while (k < width && k < 16) t[k++] = '0';
    while (k) c(t[--k]);
    return *this;
  }
  Out& pad2(unsigned v) { return c(char('0' + v / 10 % 10)).c(char('0' + v % 10)); }
  Out& nl() { return c('\n'); }
  void flush() {
    raw_write(fd_, buf_, n_);
    n_ = 0;
  }
  size_t mirrored() const { return mn_; }

 private:
  char* buf_;
  size_t cap_, n_ = 0;
  fd_t fd_;
  char* mirror_;
  size_t mcap_, mn_ = 0;
};

// days since 1970-01-01 → 公历（H. Hinnant 算法）
inline void civil(int64_t z, int& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = unsigned(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int64_t yy = int64_t(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp < 10 ? mp + 3 : mp - 9;
  y = int(yy + (m <= 2));
}

// 本地时间 "YYYY-MM-DD HH:MM:SS.mmm"；compact=true 时 "YYYYmmdd-HHMMSS"
inline void put_time(Out& o, uint64_t utc_us, bool compact = false) {
  const int64_t ms = int64_t(utc_us / 1000) + g.tzoff_ms;
  const int64_t days = ms >= 0 ? ms / 86400000 : (ms - 86399999) / 86400000;
  const int64_t r = ms - days * 86400000;
  int y;
  unsigned m, d;
  civil(days, y, m, d);
  const unsigned hh = unsigned(r / 3600000), mi = unsigned(r / 60000 % 60),
                 ss = unsigned(r / 1000 % 60), mss = unsigned(r % 1000);
  o.u(uint64_t(y));
  if (!compact) o.c('-');
  o.pad2(m);
  if (!compact) o.c('-');
  o.pad2(d);
  o.c(compact ? '-' : ' ');
  o.pad2(hh);
  if (!compact) o.c(':');
  o.pad2(mi);
  if (!compact) o.c(':');
  o.pad2(ss);
  if (!compact) o.c('.').c(char('0' + mss / 100)).c(char('0' + mss / 10 % 10)).c(char('0' + mss % 10));
}

// 报告公共尾部：HIP 调用点、飞行记录、环境快照
inline void put_sites(Out& o) {
  o.s("---- last HIP calls (newest first; CK sites, any thread) ----\n");
  const uint32_t n = g_sites_n.load(std::memory_order_relaxed);
  const uint32_t cnt = n < uint32_t(kSites) ? n : uint32_t(kSites);
  for (uint32_t k = 0; k < cnt; k++) {
    const Site& st = g_sites[(n - 1 - k) % kSites];
    const char* s = st.s.load(std::memory_order_relaxed);
    if (!s) continue;
    o.s("  ");
    put_time(o, st.t_us.load(std::memory_order_relaxed));
    o.s(" t").u(st.tid.load(std::memory_order_relaxed)).s("  ").s(s).nl();
  }
  if (!cnt) o.s("  (none)\n");
}
inline void put_ring(Out& o) {
  o.s("---- flight recorder (oldest first) ----\n");
  const uint64_t n = g_ring_n.load(std::memory_order_acquire);
  const uint64_t first = n > uint64_t(kRing) ? n - kRing : 0;
  for (uint64_t i = first; i < n; i++) {
    const Entry& e = g_ring[i % kRing];
    if (e.seq.load(std::memory_order_acquire) != 2 * i + 2) continue;  // 残缺/已被覆盖
    o.s("  ");
    put_time(o, e.t_us);
    o.s(" t").u(e.tid).s("  ").sn(e.msg, kMsg).nl();
  }
  if (!n) o.s("  (empty)\n");
}
inline void put_env(Out& o) {
  o.s("---- environment (at startup) ----\n");
  o.sn(g.env, g.env_len);
  if (g.env_len && g.env[g.env_len - 1] != '\n') o.nl();
}
inline void put_header(Out& o, uint64_t now, const char* kind) {
  o.s("==== Qwenox crash report ====\n");
  o.s("program: ").s(g.program).s(" ").s(g.version).nl();
  o.s("exe: ").s(g.exe).nl();
  o.s("pid: ").i(g.pid).s("  time: ");
  put_time(o, now);
  o.s("  uptime: ").u((now - g.t0_us) / 1000000).s(" s\n");
  o.s("kind: ").s(kind).nl();
}

// ============================================================================
// 敏感环境变量判定（启动快照与 API 诊断包共用）
// ============================================================================
inline bool secret_name(const std::string& n) {
  std::string u;
  for (char ch : n) u += char(ch >= 'a' && ch <= 'z' ? ch - 32 : ch);
  for (const char* k : {"KEY", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL"})
    if (u.find(k) != std::string::npos) return true;
  return u.size() >= 6 && u.compare(u.size() - 6, 6, "_TOKEN") == 0;
}
inline bool diag_env_name(const std::string& n) {
  for (const char* p : {"QWENOX_", "HIP_", "HSA_", "AMD_", "ROCBLAS_", "HIPBLASLT_", "GPU_",
                        "ROCR_", "ROCM_", "MIOPEN_"})
    if (n.compare(0, strlen(p), p) == 0) return true;
  return false;
}

// ============================================================================
//                                  Windows
// ============================================================================
#ifdef _WIN32

inline std::string w2u(const wchar_t* w, int n = -1) {
  if (!w) return {};
  if (n < 0) n = int(wcslen(w));
  if (!n) return {};
  const int k = WideCharToMultiByte(CP_UTF8, 0, w, n, nullptr, 0, nullptr, nullptr);
  std::string s(size_t(k > 0 ? k : 0), '\0');
  if (k > 0) WideCharToMultiByte(CP_UTF8, 0, w, n, &s[0], k, nullptr, nullptr);
  return s;
}
// 不分配内存的 UTF-16 → UTF-8（崩溃路径用）
inline void w2u_buf(const wchar_t* w, char* out, int cap) {
  out[0] = 0;
  const int k = WideCharToMultiByte(CP_UTF8, 0, w, -1, out, cap, nullptr, nullptr);
  if (k <= 0) out[0] = 0;
  out[cap - 1] = 0;
}

template <class T>
inline T win_proc(HMODULE m, const char* name) {
  return m ? reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(m, name))) : nullptr;
}

using MiniDumpWriteDump_t = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                         PMINIDUMP_EXCEPTION_INFORMATION,
                                         PMINIDUMP_USER_STREAM_INFORMATION,
                                         PMINIDUMP_CALLBACK_INFORMATION);
using NtQueryInformationThread_t = LONG(WINAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
using K32EnumProcessModules_t = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
using GetThreadDescription_t = HRESULT(WINAPI*)(HANDLE, PWSTR*);

struct WinFns {
  MiniDumpWriteDump_t mini_dump;
  NtQueryInformationThread_t nt_qit;
  K32EnumProcessModules_t enum_mods;
  GetThreadDescription_t thread_desc;
};
inline WinFns g_fn;

inline void win_load_fns() {
  if (!g_fn.mini_dump) {
    HMODULE dh = LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    g_fn.mini_dump = win_proc<MiniDumpWriteDump_t>(dh, "MiniDumpWriteDump");
  }
  HMODULE nt = GetModuleHandleW(L"ntdll.dll");
  HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
  g_fn.nt_qit = win_proc<NtQueryInformationThread_t>(nt, "NtQueryInformationThread");
  g_fn.enum_mods = win_proc<K32EnumProcessModules_t>(k32, "K32EnumProcessModules");
  g_fn.thread_desc = win_proc<GetThreadDescription_t>(k32, "GetThreadDescription");
}

// 合成的异常码：非异常路径（abort / terminate / fatal）的报告与 minidump 用
constexpr DWORD kSyntheticCode = 0xE0514E01;  // 'Q','N'

struct WinCrash {
  EXCEPTION_RECORD er;
  CONTEXT ctx;
  EXCEPTION_POINTERS ep;
  DWORD tid;
  const char* kind;
  char why[1024];
  bool first_chance;
};
inline WinCrash g_wc;
inline volatile LONG g_once = 0;
inline HANDLE g_ev_go = nullptr, g_ev_done = nullptr;
inline DWORD g_writer_tid = 0;
inline char g_buf[64 * 1024];
inline char g_mirror[256 * 1024];
inline void put_first_chance(Out& o);  // 定义在 veh 旁
inline void put_cxx(Out& o);
inline char g_stamp[32];

inline const char* exc_name(DWORD c) {
  switch (c) {
    case EXCEPTION_ACCESS_VIOLATION: return "EXCEPTION_ACCESS_VIOLATION";
    case EXCEPTION_IN_PAGE_ERROR: return "EXCEPTION_IN_PAGE_ERROR";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "EXCEPTION_ILLEGAL_INSTRUCTION";
    case EXCEPTION_PRIV_INSTRUCTION: return "EXCEPTION_PRIV_INSTRUCTION";
    case EXCEPTION_STACK_OVERFLOW: return "EXCEPTION_STACK_OVERFLOW";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "EXCEPTION_INT_DIVIDE_BY_ZERO";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "EXCEPTION_ARRAY_BOUNDS_EXCEEDED";
    case EXCEPTION_DATATYPE_MISALIGNMENT: return "EXCEPTION_DATATYPE_MISALIGNMENT";
    case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "EXCEPTION_NONCONTINUABLE_EXCEPTION";
    case EXCEPTION_INVALID_DISPOSITION: return "EXCEPTION_INVALID_DISPOSITION";
    case 0xC0000374: return "STATUS_HEAP_CORRUPTION";
    case 0xC0000409: return "STATUS_STACK_BUFFER_OVERRUN (fail-fast)";
    case kSyntheticCode: return "QWENOX_SYNTHETIC (no hardware exception)";
    default: return "?";
  }
}
inline bool is_fatal(DWORD c) {
  switch (c) {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_IN_PAGE_ERROR:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
    case EXCEPTION_INVALID_DISPOSITION:
    case 0xC0000374:
    case 0xC0000409: return true;
    default: return false;
  }
}

inline bool rd(const void* p, void* out, size_t n) {
  SIZE_T got = 0;
  return ReadProcessMemory(GetCurrentProcess(), p, out, n, &got) && got == n;
}
inline bool readable(uint64_t a, size_t n) {
  MEMORY_BASIC_INFORMATION m{};
  if (!VirtualQuery(reinterpret_cast<void*>(a), &m, sizeof m)) return false;
  if (m.State != MEM_COMMIT || (m.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
  return a + n <= uint64_t(reinterpret_cast<uintptr_t>(m.BaseAddress)) + m.RegionSize;
}

// 内存中 PE 头 → 模块名（导出表 Name，exe 用安装时记下的文件名）、时间戳、映像大小。
// 只读内存、不取加载器锁。
inline bool pe_info(const unsigned char* base, char* name, size_t cap, DWORD* ts,
                    DWORD* size) {
  name[0] = 0;
  IMAGE_DOS_HEADER dos;
  if (!rd(base, &dos, sizeof dos) || dos.e_magic != IMAGE_DOS_SIGNATURE) return false;
  IMAGE_NT_HEADERS64 nt;
  if (!rd(base + dos.e_lfanew, &nt, sizeof nt) || nt.Signature != IMAGE_NT_SIGNATURE)
    return false;
  if (ts) *ts = nt.FileHeader.TimeDateStamp;
  if (size) *size = nt.OptionalHeader.SizeOfImage;
  const IMAGE_DATA_DIRECTORY& d = nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
  if (d.VirtualAddress && d.Size) {
    IMAGE_EXPORT_DIRECTORY ex;
    if (rd(base + d.VirtualAddress, &ex, sizeof ex) && ex.Name) {
      size_t k = 0;
      for (; k + 1 < cap; k++) {
        char ch;
        if (!rd(base + ex.Name + k, &ch, 1) || !ch) break;
        name[k] = ch;
      }
      name[k] = 0;
    }
  }
  if (!name[0] && base == g.exe_base) {
    const char* b = strrchr(g.exe, '\\');
    b = b ? b + 1 : g.exe;
    strncpy(name, b, cap - 1);
    name[cap - 1] = 0;
  }
  if (!name[0]) strncpy(name, "?module", cap - 1);
  return true;
}

// 地址 → "0x… module+0xoff"
inline void put_addr(Out& o, uint64_t a) {
  o.x(a, 16);
  MEMORY_BASIC_INFORMATION m{};
  if (a && VirtualQuery(reinterpret_cast<void*>(a), &m, sizeof m) && m.Type == MEM_IMAGE &&
      m.AllocationBase) {
    char name[128];
    const unsigned char* base = static_cast<const unsigned char*>(m.AllocationBase);
    if (pe_info(base, name, sizeof name, nullptr, nullptr))
      o.s(" ").s(name).s("+").x(a - uint64_t(reinterpret_cast<uintptr_t>(base)));
  }
}

inline const char* prot_name(DWORD p) {
  switch (p & 0xFF) {
    case PAGE_NOACCESS: return "NOACCESS";
    case PAGE_READONLY: return "READONLY";
    case PAGE_READWRITE: return "READWRITE";
    case PAGE_WRITECOPY: return "WRITECOPY";
    case PAGE_EXECUTE: return "EXECUTE";
    case PAGE_EXECUTE_READ: return "EXECUTE_READ";
    case PAGE_EXECUTE_READWRITE: return "EXECUTE_READWRITE";
    case PAGE_EXECUTE_WRITECOPY: return "EXECUTE_WRITECOPY";
    case 0: return "-";
    default: return "?";
  }
}
inline void put_region1(Out& o, const MEMORY_BASIC_INFORMATION& m) {
  o.s(m.State == MEM_COMMIT    ? "COMMIT"
      : m.State == MEM_RESERVE ? "RESERVE"
      : m.State == MEM_FREE    ? "FREE"
                               : "?");
  if (m.State != MEM_FREE) {
    o.s(" ").s(m.Type == MEM_IMAGE ? "IMAGE" : m.Type == MEM_MAPPED ? "MAPPED" : m.Type == MEM_PRIVATE ? "PRIVATE" : "?");
    o.s(" protect=").s(prot_name(m.Protect));
    if (m.Protect & PAGE_GUARD) o.s("+GUARD");
    if (m.Protect & PAGE_NOCACHE) o.s("+NOCACHE");
    if (m.Protect & PAGE_WRITECOMBINE) o.s("+WRITECOMBINE");
    o.s(" alloc_base=").x(uint64_t(reinterpret_cast<uintptr_t>(m.AllocationBase)));
    o.s(" alloc_protect=").s(prot_name(m.AllocationProtect));
  }
  o.s(" region=").x(uint64_t(reinterpret_cast<uintptr_t>(m.BaseAddress))).s("+").x(m.RegionSize);
}
// 地址所在内存区；未映射时再给出前后相邻区（判断"刚好越过某块分配的末尾"）
inline void put_region(Out& o, const char* label, uint64_t a) {
  MEMORY_BASIC_INFORMATION m{};
  o.s(label);
  if (!VirtualQuery(reinterpret_cast<void*>(a), &m, sizeof m)) {
    o.s("VirtualQuery failed (kernel address or beyond user space)\n");
    return;
  }
  put_region1(o, m);
  o.nl();
  if (m.State == MEM_FREE || m.State == MEM_RESERVE) {
    const uint64_t b = uint64_t(reinterpret_cast<uintptr_t>(m.BaseAddress));
    MEMORY_BASIC_INFORMATION p{}, n{};
    if (b && VirtualQuery(reinterpret_cast<void*>(b - 1), &p, sizeof p)) {
      o.s("    prev: ");
      put_region1(o, p);
      o.nl();
    }
    if (VirtualQuery(reinterpret_cast<void*>(b + m.RegionSize), &n, sizeof n)) {
      o.s("    next: ");
      put_region1(o, n);
      o.nl();
    }
  }
}

inline void put_bytes(Out& o, const char* label, uint64_t a, size_t n) {
  o.s(label).x(a, 16).s(":");
  for (size_t k = 0; k < n; k++) {
    unsigned char b;
    if (k % 32 == 0) o.s("\n   ");
    if (!rd(reinterpret_cast<void*>(a + k), &b, 1)) {
      o.s(" ??");
      continue;
    }
    o.c(' ').c("0123456789abcdef"[b >> 4]).c("0123456789abcdef"[b & 15]);
  }
  o.nl();
}

inline void put_regs(Out& o, const CONTEXT& c) {
  o.s("registers:\n");
  const struct {
    const char* n;
    DWORD64 v;
  } r[] = {{"rax", c.Rax}, {"rbx", c.Rbx}, {"rcx", c.Rcx}, {"rdx", c.Rdx}, {"rsi", c.Rsi},
           {"rdi", c.Rdi}, {"rbp", c.Rbp}, {"rsp", c.Rsp}, {"r8 ", c.R8},  {"r9 ", c.R9},
           {"r10", c.R10}, {"r11", c.R11}, {"r12", c.R12}, {"r13", c.R13}, {"r14", c.R14},
           {"r15", c.R15}, {"rip", c.Rip}, {"efl", c.EFlags}};
  for (size_t k = 0; k < sizeof r / sizeof r[0]; k++) {
    o.s(k % 4 ? "  " : "  ").s(r[k].n).s("=").x(r[k].v, 16);
    if (k % 4 == 3) o.nl();
  }
  o.nl();
}

// x64 展开：RtlLookupFunctionEntry + RtlVirtualUnwind（异常分发本身用的同一套，
// 读 .pdata，不分配内存）。每步先确认 rsp 可读，栈坏了就停。
inline void put_stack(Out& o, CONTEXT c) {
  o.s("stack (unwound from the faulting context):\n");
  for (int k = 0; k < 64 && c.Rip; k++) {
    o.s("  #").u(uint64_t(k)).s(k < 10 ? "  " : " ");
    put_addr(o, c.Rip);
    o.nl();
    if (!readable(c.Rsp, 64)) {
      o.s("  (rsp not readable, stop)\n");
      break;
    }
    const DWORD64 prev_rip = c.Rip, prev_rsp = c.Rsp;
    DWORD64 img = 0;
    PRUNTIME_FUNCTION f = RtlLookupFunctionEntry(c.Rip, &img, nullptr);
    if (f) {
      PVOID hd = nullptr;
      DWORD64 est = 0;
      RtlVirtualUnwind(UNW_FLAG_NHANDLER, img, c.Rip, f, &c, &hd, &est, nullptr);
    } else {  // 叶函数：返回地址就在栈顶
      DWORD64 ra = 0;
      if (!rd(reinterpret_cast<void*>(c.Rsp), &ra, 8)) break;
      c.Rip = ra;
      c.Rsp += 8;
    }
    if (c.Rip == prev_rip && c.Rsp == prev_rsp) break;
  }
}

// 栈上"像返回地址"的值（展开失败时的兜底线索）
inline void put_stack_scan(Out& o, uint64_t rsp) {
  o.s("stack scan (values on the stack pointing into loaded images):\n");
  int found = 0;
  for (uint64_t a = rsp; a < rsp + 16384 && found < 40; a += 8) {
    uint64_t v = 0;
    if (!rd(reinterpret_cast<void*>(a), &v, 8)) break;
    MEMORY_BASIC_INFORMATION m{};
    if (v < 0x10000 || !VirtualQuery(reinterpret_cast<void*>(v), &m, sizeof m)) continue;
    if (m.Type != MEM_IMAGE || !(m.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                                              PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
      continue;
    o.s("  [rsp+").x(a - rsp).s("] ");
    put_addr(o, v);
    o.nl();
    found++;
  }
}

inline void put_thread(Out& o, DWORD t) {
  o.s("thread: ").u(t);
  if (t == g.main_tid) o.s(" (main thread)");
  if (t == g_writer_tid) o.s(" (crash writer thread)");
  HANDLE h = OpenThread(THREAD_QUERY_INFORMATION, FALSE, t);
  if (h && g_fn.nt_qit) {
    PVOID start = nullptr;
    if (g_fn.nt_qit(h, 9 /*ThreadQuerySetWin32StartAddress*/, &start, sizeof start, nullptr) == 0) {
      o.s("  start: ");
      put_addr(o, uint64_t(reinterpret_cast<uintptr_t>(start)));
    }
  }
  if (h) CloseHandle(h);
  o.nl();
}

inline void put_modules(Out& o) {
  o.s("---- loaded modules (base, size, PE timestamp, path) ----\n");
  static HMODULE mods[1024];
  DWORD need = 0;
  if (!g_fn.enum_mods || !g_fn.enum_mods(GetCurrentProcess(), mods, sizeof mods, &need)) {
    o.s("  (EnumProcessModules failed)\n");
    return;
  }
  const DWORD n = need / sizeof(HMODULE) < 1024 ? need / sizeof(HMODULE) : 1024;
  for (DWORD k = 0; k < n; k++) {
    static wchar_t wp[1024];
    static char up[2048];
    char name[128];
    DWORD ts = 0, sz = 0;
    pe_info(reinterpret_cast<const unsigned char*>(mods[k]), name, sizeof name, &ts, &sz);
    wp[0] = 0;
    GetModuleFileNameW(mods[k], wp, 1024);
    wp[1023] = 0;
    w2u_buf(wp, up, sizeof up);
    o.s("  ").x(uint64_t(reinterpret_cast<uintptr_t>(mods[k])), 16).s(" ").x(sz, 8).s(" ts=").x(ts, 8);
    o.s(" ").s(up).nl();
  }
}

inline void make_path(wchar_t* out, size_t cap, const wchar_t* ext) {
  size_t k = 0;
  auto put = [&](const wchar_t* s) {
    while (*s && k + 1 < cap) out[k++] = *s++;
  };
  auto puta = [&](const char* s) {
    while (*s && k + 1 < cap) out[k++] = wchar_t(static_cast<unsigned char>(*s++));
  };
  char pid[16];
  snprintf(pid, sizeof pid, "%d", g.pid);
  put(g.wdir);
  put(L"\\crash-");
  puta(g.program);
  put(L"-");
  puta(pid);
  put(L"-");
  puta(g_stamp);
  put(ext);
  out[k] = 0;
}

inline void stamp_now() {
  char b[32];
  Out o(b, sizeof b - 1, kNoFd, g_stamp, sizeof g_stamp);
  put_time(o, wall_us(), true);
  g_stamp[o.mirrored()] = 0;
}

// 写报告线程：等 g_ev_go，按阶段写（先写不需要任何锁的核心信息并落盘，再写
// 模块表，最后 minidump），任何一步卡住也不会丢前面的内容。
inline void win_write_report() {
  static wchar_t path[1300], dpath[1300];
  static char upath[2600];
  stamp_now();
  make_path(path, 1300, L".txt");
  make_path(dpath, 1300, L".dmp");
  w2u_buf(path, upath, sizeof upath);
  HANDLE h = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  const uint64_t now = wall_us();
  {
    Out o(g_buf, sizeof g_buf, h, g_mirror, sizeof g_mirror);
    put_header(o, now, g_wc.kind);
    const EXCEPTION_RECORD& er = g_wc.er;
    const uint64_t pc = g_wc.ctx.Rip;
    o.s("code: ").x(er.ExceptionCode, 8).s(" ").s(exc_name(er.ExceptionCode));
    if (g_wc.first_chance) o.s("  (written at first chance: this code cannot be handled)");
    o.nl();
    if (g_wc.why[0]) o.s("message: ").s(g_wc.why).nl();
    put_cxx(o);
    put_thread(o, g_wc.tid);
    if ((er.ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
         er.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        er.NumberParameters >= 2) {
      const ULONG_PTR op = er.ExceptionInformation[0];
      const uint64_t target = er.ExceptionInformation[1];
      o.s("fault: ").s(op == 0 ? "READ" : op == 1 ? "WRITE" : op == 8 ? "EXECUTE (DEP)" : "?");
      o.s(" at ").x(target, 16);
      if (er.ExceptionCode == EXCEPTION_IN_PAGE_ERROR && er.NumberParameters >= 3)
        o.s("  ntstatus=").x(er.ExceptionInformation[2], 8);
      o.nl();
      put_region(o, "fault-region: ", target);
    }
    o.s("pc: ");
    put_addr(o, pc);
    o.nl();
    put_region(o, "pc-region: ", pc);
    put_regs(o, g_wc.ctx);
    put_bytes(o, "code bytes @", pc >= 32 ? pc - 32 : pc, 64);
    put_stack(o, g_wc.ctx);
    o.flush();
    put_bytes(o, "stack memory @", g_wc.ctx.Rsp, 256);
    put_stack_scan(o, g_wc.ctx.Rsp);
    put_first_chance(o);
    put_sites(o);
    put_ring(o);
    put_env(o);
    o.flush();
    // 第 2 阶段：模块表（GetModuleFileName 可能取加载器相关锁）
    put_modules(o);
    if (g_fn.thread_desc) {
      HANDLE th = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, g_wc.tid);
      PWSTR d = nullptr;
      if (th && SUCCEEDED(g_fn.thread_desc(th, &d)) && d && d[0]) {
        static char ud[512];
        w2u_buf(d, ud, sizeof ud);
        o.s("thread-description: ").s(ud).nl();
      }
      if (d) LocalFree(d);
      if (th) CloseHandle(th);
    }
    o.flush();
    // 第 3 阶段：minidump（报告全文作为注释流嵌入）
    o.s("---- minidump ----\n");
    if (!g.dump) {
      o.s("disabled (QWENOX_CRASH_DUMP=0)\n");
    } else if (!g_fn.mini_dump) {
      o.s("dbghelp.dll unavailable\n");
    } else {
      HANDLE dh = CreateFileW(dpath, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
      if (dh == INVALID_HANDLE_VALUE) {
        o.s("cannot create dump file, error ").u(GetLastError()).nl();
      } else {
        MINIDUMP_EXCEPTION_INFORMATION mei;
        mei.ThreadId = g_wc.tid;
        mei.ExceptionPointers = &g_wc.ep;
        mei.ClientPointers = FALSE;
        MINIDUMP_USER_STREAM us;
        us.Type = CommentStreamA;
        us.BufferSize = ULONG(o.mirrored());
        us.Buffer = g_mirror;
        MINIDUMP_USER_STREAM_INFORMATION usi;
        usi.UserStreamCount = 1;
        usi.UserStreamArray = &us;
        const MINIDUMP_TYPE t = MINIDUMP_TYPE(
            MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules | MiniDumpWithHandleData |
            MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithProcessThreadData |
            MiniDumpWithFullMemoryInfo);
        const BOOL ok = g_fn.mini_dump(GetCurrentProcess(), GetCurrentProcessId(), dh, t, &mei,
                                       &usi, nullptr);
        const DWORD e = ok ? 0 : GetLastError();
        LARGE_INTEGER sz{};
        GetFileSizeEx(dh, &sz);
        CloseHandle(dh);
        static char ud[2600];
        w2u_buf(dpath, ud, sizeof ud);
        if (ok)
          o.s("written: ").s(ud).s(" (").u(uint64_t(sz.QuadPart)).s(" bytes)\n");
        else
          o.s("MiniDumpWriteDump failed, error ").x(e, 8).nl();
      }
    }
    o.s("==== end ====\n");
  }
  if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
  // 引擎日志里留一行摘要（stderr 经 API 的 tee 落到 engine-api-*.log）
  char b[4096];
  Out e(b, sizeof b, stderr_fd());
  e.s("\n[crashdiag] CRASH ").s(g_wc.kind).s(" ").x(g_wc.er.ExceptionCode, 8).s(" ");
  e.s(exc_name(g_wc.er.ExceptionCode));
  if (g_wc.er.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && g_wc.er.NumberParameters >= 2)
    e.s(g_wc.er.ExceptionInformation[0] == 1 ? " WRITE " : g_wc.er.ExceptionInformation[0] == 8 ? " EXEC " : " READ ")
        .x(g_wc.er.ExceptionInformation[1], 16);
  e.s(" pc ");
  put_addr(e, g_wc.ctx.Rip);
  e.s(" thread ").u(g_wc.tid).s("\n[crashdiag] report: ").s(upath).nl();
}

inline DWORD WINAPI writer_main(LPVOID) {
  for (;;) {
    WaitForSingleObject(g_ev_go, INFINITE);
    win_write_report();
    SetEvent(g_ev_done);
  }
}

inline void hand_to_writer() {
  g_wc.er.ExceptionRecord = nullptr;
  g_wc.ep.ExceptionRecord = &g_wc.er;
  g_wc.ep.ContextRecord = &g_wc.ctx;
  if (g_ev_go && GetCurrentThreadId() != g_writer_tid) {
    SetEvent(g_ev_go);
    WaitForSingleObject(g_ev_done, 120000);
  } else {
    win_write_report();  // 写报告线程不可用：就地写
  }
}

// first-chance 小环：致命类异常码第一次出现时记一条（可能随后被接住）
struct FirstChance {
  uint64_t t_us;
  uint32_t tid;
  DWORD code;
  uint64_t pc, target;
  ULONG_PTR op;
};
inline FirstChance g_fc[16];
inline volatile LONG g_fc_n = 0;

inline void report_from(EXCEPTION_POINTERS* ep, const char* kind, bool first_chance) {
  g_wc.er = *ep->ExceptionRecord;
  g_wc.ctx = *ep->ContextRecord;
  g_wc.tid = GetCurrentThreadId();
  g_wc.kind = kind;
  g_wc.first_chance = first_chance;
  g_wc.why[0] = 0;
  hand_to_writer();
}

// 最近一次抛出的 C++ 异常（MSVC ABI 异常码 0xE06D7363）：类型名 + what()。
// MSVC 的 terminate 处理器是每线程的，工作线程里没接住的异常直接 abort，
// 报告靠这里给出是哪个异常。
struct CxxThrow {
  uint64_t t_us;
  uint32_t tid;
  char type[96];
  char what[256];
};
inline CxxThrow g_cxx;
inline volatile LONG g_cxx_n = 0;
inline void note_cxx_throw(const EXCEPTION_RECORD& er) {
  // ExceptionInformation: [0] 魔数 0x19930520..22  [1] 异常对象  [2] ThrowInfo  [3] 映像基址
  if (er.NumberParameters < 4 || (er.ExceptionInformation[0] & ~ULONG_PTR(3)) != 0x19930520)
    return;
  const unsigned char* base = reinterpret_cast<const unsigned char*>(er.ExceptionInformation[3]);
  const unsigned char* obj = reinterpret_cast<const unsigned char*>(er.ExceptionInformation[1]);
  const int32_t* ti = reinterpret_cast<const int32_t*>(er.ExceptionInformation[2]);
  if (!base || !ti) return;
  CxxThrow c{};
  c.t_us = wall_us();
  c.tid = GetCurrentThreadId();
  int32_t cta_rva = 0, n = 0;
  if (!rd(ti + 3, &cta_rva, 4) || !cta_rva || !rd(base + cta_rva, &n, 4)) return;
  for (int32_t k = 0; k < n && k < 16; k++) {
    int32_t ct_rva = 0;
    if (!rd(base + cta_rva + 4 + 4 * k, &ct_rva, 4) || !ct_rva) break;
    int32_t ct[4] = {};  // properties, pType, thisDisplacement.mdisp, ...
    if (!rd(base + ct_rva, ct, sizeof ct) || !ct[1]) break;
    char name[96] = {};
    for (size_t i = 0; i + 1 < sizeof name; i++)  // TypeDescriptor: vftable, spare, name[]
      if (!rd(base + ct[1] + 16 + i, &name[i], 1) || !name[i]) break;
    if (k == 0) strncpy(c.type, name, sizeof c.type - 1);  // 最派生的类型
    if (obj && !strcmp(name, ".?AVexception@std@@")) {
      const std::exception* e = reinterpret_cast<const std::exception*>(obj + ct[2]);
      const char* w = e->what();
      size_t i = 0;
      for (; w && i + 1 < sizeof c.what; i++)
        if (!rd(w + i, &c.what[i], 1) || !c.what[i]) break;
      c.what[i] = 0;
      break;
    }
  }
  g_cxx = c;
  InterlockedIncrement(&g_cxx_n);
}

inline LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
  const DWORD code = ep->ExceptionRecord->ExceptionCode;
  if (code == 0xE06D7363 && g.installed) {
    note_cxx_throw(*ep->ExceptionRecord);
    return EXCEPTION_CONTINUE_SEARCH;
  }
  if (!is_fatal(code) || !g.installed) return EXCEPTION_CONTINUE_SEARCH;
  if (GetCurrentThreadId() == g_writer_tid) return EXCEPTION_CONTINUE_SEARCH;
  const EXCEPTION_RECORD& er = *ep->ExceptionRecord;
  FirstChance& f = g_fc[(InterlockedIncrement(&g_fc_n) - 1) & 15];
  f.t_us = wall_us();
  f.tid = GetCurrentThreadId();
  f.code = code;
  f.pc = ep->ContextRecord->Rip;
  f.op = er.NumberParameters >= 1 ? er.ExceptionInformation[0] : 0;
  f.target = er.NumberParameters >= 2 ? er.ExceptionInformation[1] : 0;
  if ((code == EXCEPTION_STACK_OVERFLOW || code == 0xC0000374 || code == 0xC0000409) &&
      InterlockedCompareExchange(&g_once, 1, 0) == 0)
    report_from(ep, "exception", true);
  return EXCEPTION_CONTINUE_SEARCH;
}

inline LPTOP_LEVEL_EXCEPTION_FILTER g_prev_uef = nullptr;
inline volatile LONG g_uef_depth = 0;
inline LONG WINAPI uef(EXCEPTION_POINTERS* ep) {
  const LONG depth = InterlockedIncrement(&g_uef_depth);
  if (depth == 1 && g.installed && GetCurrentThreadId() != g_writer_tid &&
      InterlockedCompareExchange(&g_once, 1, 0) == 0)
    report_from(ep, "unhandled exception", false);
  LONG r = EXCEPTION_CONTINUE_SEARCH;  // → WER / 默认终止，退出码 = 异常码
  // 串到被我们顶下去的过滤器（LLVM 的会打印栈后终止进程）；它若又回调我们，
  // depth > 1 时直接放行，不会循环
  if (depth == 1 && g_prev_uef) r = g_prev_uef(ep);
  InterlockedDecrement(&g_uef_depth);
  return r;
}

inline void rearm() {
  if (!g.installed) return;
  const LPTOP_LEVEL_EXCEPTION_FILTER p = SetUnhandledExceptionFilter(uef);
  if (p && p != uef) g_prev_uef = p;  // 别人后装的（如 LLVM 的）：串在我们后面
}

inline void put_cxx(Out& o) {
  if (!g_cxx_n) return;
  o.s("last C++ exception thrown: ");
  put_time(o, g_cxx.t_us);
  o.s(" t").u(g_cxx.tid).s("  ").s(g_cxx.type[0] ? g_cxx.type : "?");
  if (g_cxx.what[0]) o.s(": ").s(g_cxx.what);
  o.nl();
}

inline void put_first_chance(Out& o) {
  const LONG n = g_fc_n;
  o.s("---- first-chance fatal-type exceptions (newest first; may have been handled) ----\n");
  if (!n) o.s("  (none)\n");
  for (LONG k = 0; k < n && k < 16; k++) {
    const FirstChance& f = g_fc[(n - 1 - k) & 15];
    o.s("  ");
    put_time(o, f.t_us);
    o.s(" t").u(f.tid).s("  ").x(f.code, 8).s(" pc ");
    put_addr(o, f.pc);
    if (f.code == EXCEPTION_ACCESS_VIOLATION)
      o.s(f.op == 1 ? "  WRITE " : f.op == 8 ? "  EXEC " : "  READ ").x(f.target, 16);
    o.nl();
  }
}

// 非异常路径（abort / terminate / fatal）：抓当前线程现场，合成异常记录
inline bool synthetic(const char* kind, const char* why) {
  if (!g.installed) return false;
  if (InterlockedCompareExchange(&g_once, 1, 0) != 0) return false;
  memset(&g_wc.er, 0, sizeof g_wc.er);
  RtlCaptureContext(&g_wc.ctx);
  g_wc.er.ExceptionCode = kSyntheticCode;
  g_wc.er.ExceptionAddress = reinterpret_cast<PVOID>(g_wc.ctx.Rip);
  g_wc.tid = GetCurrentThreadId();
  g_wc.kind = kind;
  g_wc.first_chance = false;
  strncpy(g_wc.why, why ? why : "", sizeof g_wc.why - 1);
  g_wc.why[sizeof g_wc.why - 1] = 0;
  hand_to_writer();
  return true;
}

inline void __cdecl on_abort(int) { synthetic("abort", "abort() called (SIGABRT)"); }
inline void __cdecl on_invalid_param(const wchar_t* expr, const wchar_t* fn, const wchar_t* file,
                                     unsigned line, uintptr_t) {
  char b[1024];
  snprintf(b, sizeof b, "CRT invalid parameter: %s in %s (%s:%u)", w2u(expr).c_str(),
           w2u(fn).c_str(), w2u(file).c_str(), line);
  synthetic("invalid-parameter", b);
  abort();
}

#else  // ======================== Linux / POSIX ===============================

inline void rearm() {}  // POSIX：信号处理不会被别的库"顶掉"成需要重装的样子

inline char g_buf[64 * 1024];
inline char g_maps[1 << 20];
inline size_t g_maps_len = 0;
inline char g_path[1400];
inline volatile int g_lx_once = 0;
inline char g_altstack[64 * 1024];

inline void lx_mkdirs(const char* p) {
  char b[1024];
  strncpy(b, p, sizeof b - 1);
  b[sizeof b - 1] = 0;
  for (char* q = b + 1; *q; q++)
    if (*q == '/') {
      *q = 0;
      mkdir(b, 0755);
      *q = '/';
    }
  mkdir(b, 0755);
}

inline void lx_load_maps() {
  g_maps_len = 0;
  const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;
  for (;;) {
    const ssize_t r = read(fd, g_maps + g_maps_len, sizeof g_maps - 1 - g_maps_len);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    g_maps_len += size_t(r);
    if (g_maps_len >= sizeof g_maps - 1) break;
  }
  close(fd);
  g_maps[g_maps_len] = 0;
}
inline uint64_t lx_hex(const char*& p) {
  uint64_t v = 0;
  for (;; p++) {
    const char ch = *p;
    if (ch >= '0' && ch <= '9') v = v * 16 + uint64_t(ch - '0');
    else if (ch >= 'a' && ch <= 'f') v = v * 16 + uint64_t(ch - 'a' + 10);
    else if (ch >= 'A' && ch <= 'F') v = v * 16 + uint64_t(ch - 'A' + 10);
    else break;
  }
  return v;
}
// 在 maps 里找包含 a 的行；out_prev/out_next 给相邻行（a 未映射时有用）
struct MapHit {
  const char* line = nullptr;
  size_t len = 0;
  uint64_t start = 0, end = 0, off = 0;
  const char* path = nullptr;
  size_t path_len = 0;
  bool r = false, x = false;
};
inline bool lx_parse(const char* line, size_t len, MapHit& h) {
  const char* p = line;
  h.line = line;
  h.len = len;
  h.start = lx_hex(p);
  if (*p != '-') return false;
  p++;
  h.end = lx_hex(p);
  if (p + 4 < line + len) {
    h.r = p[1] == 'r';
    h.x = p[3] == 'x';
  }
  for (int f = 0; f < 2 && p < line + len; f++) {  // 跳过 perms，到 offset
    while (p < line + len && *p != ' ') p++;
    while (p < line + len && *p == ' ') p++;
  }
  h.off = lx_hex(p);
  for (int f = 0; f < 2 && p < line + len; f++) {  // 跳过 dev inode
    while (p < line + len && *p == ' ') p++;
    while (p < line + len && *p != ' ') p++;
  }
  while (p < line + len && *p == ' ') p++;
  h.path = p;
  h.path_len = size_t(line + len - p);
  return true;
}
inline bool lx_find(uint64_t a, MapHit& hit, MapHit* prev = nullptr, MapHit* next = nullptr) {
  const char* p = g_maps;
  const char* end = g_maps + g_maps_len;
  MapHit last;
  bool have_last = false;
  while (p < end) {
    const char* nl = static_cast<const char*>(memchr(p, '\n', size_t(end - p)));
    const size_t len = size_t((nl ? nl : end) - p);
    MapHit h;
    if (lx_parse(p, len, h)) {
      if (a >= h.start && a < h.end) {
        hit = h;
        return true;
      }
      if (h.start > a) {
        if (prev && have_last) *prev = last;
        if (next) *next = h;
        return false;
      }
      last = h;
      have_last = true;
    }
    p = nl ? nl + 1 : end;
  }
  if (prev && have_last) *prev = last;
  return false;
}
// "0x… /完整/路径+0x文件偏移"（API 打诊断包时用 addr2line 按这个符号化）
inline void lx_put_addr(Out& o, uint64_t a) {
  o.x(a, 16);
  MapHit h;
  if (a && lx_find(a, h) && h.path_len) o.s(" ").sn(h.path, h.path_len).s("+").x(a - h.start + h.off);
}
inline void lx_put_region(Out& o, const char* label, uint64_t a) {
  MapHit h, pv, nx;
  o.s(label);
  if (lx_find(a, h, &pv, &nx)) {
    o.s("mapped: ").sn(h.line, h.len).nl();
    return;
  }
  o.s("UNMAPPED\n");
  if (pv.line) o.s("    prev: ").sn(pv.line, pv.len).nl();
  if (nx.line) o.s("    next: ").sn(nx.line, nx.len).nl();
}

inline const char* sig_name(int s) {
  switch (s) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGFPE: return "SIGFPE";
    case SIGABRT: return "SIGABRT";
    case SIGTRAP: return "SIGTRAP";
    default: return "?";
  }
}
inline const char* code_name(int s, int c) {
  if (c <= 0) return "sent by kill()/raise()/tgkill()";
  if (s == SIGSEGV) return c == SEGV_MAPERR ? "SEGV_MAPERR (address not mapped)"
                         : c == SEGV_ACCERR ? "SEGV_ACCERR (permission denied)"
                                            : "?";
  if (s == SIGBUS) return c == BUS_ADRALN ? "BUS_ADRALN (misaligned)"
                        : c == BUS_ADRERR ? "BUS_ADRERR (no backing page: truncated file / device memory)"
                        : c == BUS_OBJERR ? "BUS_OBJERR (hardware error)"
                                          : "?";
  if (s == SIGILL) return c == ILL_ILLOPC ? "ILL_ILLOPC" : c == ILL_PRVOPC ? "ILL_PRVOPC" : "?";
  if (s == SIGFPE) return c == FPE_INTDIV ? "FPE_INTDIV" : "?";
  return "?";
}

// 写报告（信号处理里调用：只用 open/write/read/close/backtrace）
inline void lx_report(int sig, siginfo_t* si, ucontext_t* uc, const char* kind, const char* why) {
  const uint64_t now = wall_us();
  {  // 路径
    Out p(g_buf, sizeof g_buf, kNoFd, g_path, sizeof g_path);
    p.s(g.dir).s("/crash-").s(g.program).s("-").i(g.pid).s("-");
    put_time(p, now, true);
    p.s(".txt");
    g_path[p.mirrored()] = 0;
  }
  const int fd = open(g_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  lx_load_maps();
  uint64_t pc = 0, sp = 0;
#if defined(__x86_64__)
  if (uc) {
    pc = uint64_t(uc->uc_mcontext.gregs[REG_RIP]);
    sp = uint64_t(uc->uc_mcontext.gregs[REG_RSP]);
  }
#endif
  {
    Out o(g_buf, sizeof g_buf, fd);
    put_header(o, now, kind);
    if (sig) {
      o.s("signal: ").i(sig).s(" ").s(sig_name(sig)).s("  si_code=").i(si ? si->si_code : 0);
      if (si) o.s(" ").s(code_name(sig, si->si_code));
      o.nl();
      if (si && si->si_code <= 0)
        o.s("sender: pid ").i(si->si_pid).s(" uid ").i(si->si_uid).nl();
    }
    if (why && *why) o.s("message: ").s(why).nl();
    o.s("thread: ").u(tid());
    if (tid() == g.main_tid) o.s(" (main thread)");
    o.nl();
    if (si && (sig == SIGSEGV || sig == SIGBUS) && si->si_code > 0) {
      const uint64_t target = uint64_t(reinterpret_cast<uintptr_t>(si->si_addr));
      o.s("fault: ");
#if defined(__x86_64__)
      if (uc) {
        const uint64_t err = uint64_t(uc->uc_mcontext.gregs[REG_ERR]);
        o.s(err & 0x10 ? "EXECUTE" : err & 2 ? "WRITE" : "READ");
        o.s(" (page-fault error code ").x(err).s(")");
      }
#endif
      o.s(" at ").x(target, 16).nl();
      lx_put_region(o, "fault-region: ", target);
    }
    if (pc) {
      o.s("pc: ");
      lx_put_addr(o, pc);
      o.nl();
      lx_put_region(o, "pc-region: ", pc);
    }
#if defined(__x86_64__)
    if (uc) {
      const auto& r = uc->uc_mcontext.gregs;
      const struct {
        const char* n;
        int i;
      } regs[] = {{"rax", REG_RAX}, {"rbx", REG_RBX}, {"rcx", REG_RCX}, {"rdx", REG_RDX},
                  {"rsi", REG_RSI}, {"rdi", REG_RDI}, {"rbp", REG_RBP}, {"rsp", REG_RSP},
                  {"r8 ", REG_R8},  {"r9 ", REG_R9},  {"r10", REG_R10}, {"r11", REG_R11},
                  {"r12", REG_R12}, {"r13", REG_R13}, {"r14", REG_R14}, {"r15", REG_R15},
                  {"rip", REG_RIP}, {"efl", REG_EFL}};
      o.s("registers:\n");
      for (size_t k = 0; k < sizeof regs / sizeof regs[0]; k++) {
        o.s("  ").s(regs[k].n).s("=").x(uint64_t(r[regs[k].i]), 16);
        if (k % 4 == 3) o.nl();
      }
      o.nl();
    }
#endif
    o.flush();  // 先落盘：后面的栈回溯在坏现场下可能再次出错
    // 栈上"像返回地址"的值（指向可执行映射）：pc 本身是坏地址时这是主要线索
    MapHit sh;
    if (sp && lx_find(sp, sh) && sh.r) {
      o.s("stack scan (values on the stack pointing into executable mappings):\n");
      int found = 0;
      for (uint64_t a = sp; a + 8 <= sh.end && a < sp + 16384 && found < 40; a += 8) {
        uint64_t v;
        memcpy(&v, reinterpret_cast<const void*>(uintptr_t(a)), 8);
        MapHit vh;
        if (v < 0x10000 || !lx_find(v, vh) || !vh.x) continue;
        o.s("  [sp+").x(a - sp).s("] ");
        lx_put_addr(o, v);
        o.nl();
        found++;
      }
      o.flush();
    }
    put_sites(o);
    put_ring(o);
    put_env(o);
    o.flush();
    // 回溯放最后：glibc 的展开器从信号帧往上走，pc 不在可执行映射里时可能
    // 在这里再崩一次（SA_RESETHAND 已复位，进程按默认动作结束，前面的内容已落盘）
    MapHit ph;
    const bool pc_ok = !pc || (lx_find(pc, ph) && ph.x);
    if (pc_ok) {
      void* fr[64];
      const int n = backtrace(fr, 64);
      o.s(sig ? "stack (backtrace from the signal handler; the faulting pc is the pc line above):\n"
              : "stack (backtrace of the reporting thread):\n");
      for (int k = 0; k < n; k++) {
        o.s("  #").i(k).s(k < 10 ? "  " : " ");
        lx_put_addr(o, uint64_t(reinterpret_cast<uintptr_t>(fr[k])));
        o.nl();
      }
      o.s("symbols:\n");
      o.flush();
      if (fd >= 0) backtrace_symbols_fd(fr, n, fd);
    } else {
      o.s("stack: pc is outside executable mappings; backtrace skipped (see stack scan)\n");
    }
    o.s("---- /proc/self/maps ----\n");
    o.sn(g_maps, g_maps_len);
    o.s("==== end ====\n");
  }
  if (fd >= 0) close(fd);
  char b[2048];
  Out e(b, sizeof b, 2);
  e.s("\n[crashdiag] CRASH ").s(kind);
  if (sig) e.s(" ").s(sig_name(sig));
  if (si && (sig == SIGSEGV || sig == SIGBUS)) e.s(" at ").x(uint64_t(reinterpret_cast<uintptr_t>(si->si_addr)), 16);
  if (pc) {
    e.s(" pc ");
    lx_put_addr(e, pc);
  }
  e.s(" thread ").u(tid()).s("\n[crashdiag] report: ").s(g_path).nl();
}

inline void lx_on_signal(int sig, siginfo_t* si, void* uc) {
  const int saved = errno;
  if (!__atomic_exchange_n(&g_lx_once, 1, __ATOMIC_SEQ_CST))
    lx_report(sig, si, static_cast<ucontext_t*>(uc), "signal", nullptr);
  errno = saved;
  // SA_RESETHAND 已恢复默认动作：同步故障返回后重新触发并按默认结束（core）；
  // kill() 送来的与 SIGABRT 显式再发一次。
  if (sig == SIGABRT || !si || si->si_code <= 0) raise(sig);
}
inline void lx_on_term(int sig, siginfo_t* si, void*) {
  char b[256];
  Out e(b, sizeof b, 2);
  e.s("\n[crashdiag] got ").s(sig == SIGTERM ? "SIGTERM" : "SIGINT").s(" from pid ").i(si ? si->si_pid : 0).nl();
  e.flush();
  signal(sig, SIG_DFL);
  raise(sig);
}

#endif  // _WIN32

// ============================================================================
// 公共：非异常致命路径
// ============================================================================

// 写一份 kind=fatal 的报告（不退出）。返回是否写了（每进程只写第一份）。
inline bool report_fatal(const char* kind, const char* why) {
  if (!g.installed) return false;
#ifdef _WIN32
  return synthetic(kind, why);
#else
  if (__atomic_exchange_n(&g_lx_once, 1, __ATOMIC_SEQ_CST)) return false;
  lx_report(0, nullptr, nullptr, kind, why);
  return true;
#endif
}

// 致命错误：stderr 打一行原因 → 写报告 → exit(1)
[[noreturn]] inline void fatal(const char* fmt, ...) CRASHDIAG_PRINTF(1, 2);
[[noreturn]] inline void fatal(const char* fmt, ...) {
  char b[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s\n", b);
  fflush(stderr);
  note("FATAL: %s", b);
  report_fatal("fatal", b);
  std::exit(1);
}

inline void on_terminate() {
  char b[1024] = "std::terminate (no active exception)";
  if (std::exception_ptr ep = std::current_exception()) {
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      snprintf(b, sizeof b, "uncaught C++ exception: %s", e.what());
    } catch (...) {
      snprintf(b, sizeof b, "uncaught C++ exception (not std::exception)");
    }
  }
  fprintf(stderr, "%s\n", b);
  fflush(stderr);
  report_fatal("terminate", b);
  std::abort();
}

// ============================================================================
// 环境快照
// ============================================================================
#ifdef _WIN32
inline std::string reg_value(HKEY root, const wchar_t* key, const wchar_t* name) {
  HKEY k = nullptr;
  if (RegOpenKeyExW(root, key, 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return {};
  DWORD type = 0, n = 0;
  std::string out;
  if (RegQueryValueExW(k, name, nullptr, &type, nullptr, &n) == ERROR_SUCCESS && n) {
    std::string raw(n + 4, '\0');
    if (RegQueryValueExW(k, name, nullptr, &type, reinterpret_cast<BYTE*>(&raw[0]), &n) ==
        ERROR_SUCCESS) {
      if (type == REG_SZ || type == REG_EXPAND_SZ) {
        out = w2u(reinterpret_cast<const wchar_t*>(raw.data()));
      } else if (type == REG_MULTI_SZ) {
        const wchar_t* w = reinterpret_cast<const wchar_t*>(raw.data());
        while (*w) {
          if (!out.empty()) out += "; ";
          out += w2u(w);
          w += wcslen(w) + 1;
        }
      } else if (type == REG_DWORD && n >= 4) {
        DWORD v;
        memcpy(&v, raw.data(), 4);
        out = std::to_string(v);
      } else if (type == REG_QWORD && n >= 8) {
        uint64_t v;
        memcpy(&v, raw.data(), 8);
        out = std::to_string(v);
      } else if (type == REG_BINARY && n >= 8) {
        uint64_t v;
        memcpy(&v, raw.data(), 8);
        out = std::to_string(v);
      }
    }
  }
  RegCloseKey(k);
  return out;
}

inline std::string file_version(const wchar_t* path) {
  using Size_t = DWORD(WINAPI*)(LPCWSTR, LPDWORD);
  using Info_t = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPVOID);
  using Query_t = BOOL(WINAPI*)(LPCVOID, LPCWSTR, LPVOID*, PUINT);
  static HMODULE v = LoadLibraryExW(L"version.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  auto fsize = win_proc<Size_t>(v, "GetFileVersionInfoSizeW");
  auto finfo = win_proc<Info_t>(v, "GetFileVersionInfoW");
  auto fq = win_proc<Query_t>(v, "VerQueryValueW");
  if (!fsize || !finfo || !fq) return {};
  DWORD h = 0;
  const DWORD n = fsize(path, &h);
  if (!n) return {};
  std::string buf(n, '\0');
  if (!finfo(path, 0, n, &buf[0])) return {};
  VS_FIXEDFILEINFO* fi = nullptr;
  UINT len = 0;
  if (!fq(buf.data(), L"\\", reinterpret_cast<LPVOID*>(&fi), &len) || !fi) return {};
  char b[64];
  snprintf(b, sizeof b, "%u.%u.%u.%u", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS),
           HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
  return b;
}

// 已加载模块的版本（文件版本 + PE 时间戳 + 路径）；names 为空时列出全部模块里
// 与 GPU 栈相关的（amd*/hip*/roc*/hsa*）以及 exe。
inline std::string module_versions() {
  std::string s = "gpu-stack modules:\n";
  if (!g_fn.enum_mods) win_load_fns();
  static HMODULE mods[1024];
  DWORD need = 0;
  if (!g_fn.enum_mods || !g_fn.enum_mods(GetCurrentProcess(), mods, sizeof mods, &need))
    return s + "  (EnumProcessModules failed)\n";
  const DWORD n = need / sizeof(HMODULE) < 1024 ? need / sizeof(HMODULE) : 1024;
  for (DWORD k = 0; k < n; k++) {
    wchar_t wp[1024] = {};
    GetModuleFileNameW(mods[k], wp, 1024);
    const wchar_t* b = wcsrchr(wp, L'\\');
    b = b ? b + 1 : wp;
    std::wstring lb(b);
    for (auto& ch : lb) ch = wchar_t(towlower(ch));
    const bool want = k == 0 || lb.rfind(L"amd", 0) == 0 || lb.rfind(L"hip", 0) == 0 ||
                      lb.rfind(L"libhip", 0) == 0 || lb.rfind(L"roc", 0) == 0 ||
                      lb.rfind(L"hsa", 0) == 0 || lb.rfind(L"origami", 0) == 0 ||
                      lb.rfind(L"amdxc", 0) == 0 || lb.rfind(L"atiad", 0) == 0;
    if (!want) continue;
    char name[128];
    DWORD ts = 0, sz = 0;
    pe_info(reinterpret_cast<const unsigned char*>(mods[k]), name, sizeof name, &ts, &sz);
    char line[256];
    snprintf(line, sizeof line, "  %-22s ver %-16s ts=0x%08lX size=0x%08lX  ", w2u(b).c_str(),
             file_version(wp).c_str(), static_cast<unsigned long>(ts),
             static_cast<unsigned long>(sz));
    s += line + w2u(wp) + "\n";
  }
  return s;
}

inline std::string power_info() {
  using GetActive_t = DWORD(WINAPI*)(HKEY, GUID**);
  using ReadName_t = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, PUCHAR, LPDWORD);
  using ReadVal_t = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, LPDWORD);
  static HMODULE pp = LoadLibraryExW(L"powrprof.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  auto get_active = win_proc<GetActive_t>(pp, "PowerGetActiveScheme");
  auto read_name = win_proc<ReadName_t>(pp, "PowerReadFriendlyName");
  auto read_ac = win_proc<ReadVal_t>(pp, "PowerReadACValueIndex");
  auto read_dc = win_proc<ReadVal_t>(pp, "PowerReadDCValueIndex");
  std::string s = "power: ";
  SYSTEM_POWER_STATUS ps{};
  if (GetSystemPowerStatus(&ps))
    s += ps.ACLineStatus == 1 ? "on AC" : ps.ACLineStatus == 0 ? "on battery" : "AC unknown";
  GUID* scheme = nullptr;
  if (!get_active || get_active(nullptr, &scheme) != ERROR_SUCCESS || !scheme)
    return s + "; active scheme unknown\n";
  wchar_t nm[256] = {};
  DWORD nb = sizeof nm;
  if (read_name && read_name(nullptr, scheme, nullptr, nullptr, reinterpret_cast<PUCHAR>(nm), &nb) == ERROR_SUCCESS)
    s += "; scheme \"" + w2u(nm) + "\"";
  // 子组/设置 GUID（winnt.h 里的 GUID_VIDEO_* 等需要 INITGUID，这里自带一份）
  static const GUID kVideoSub = {0x7516b95f, 0xf776, 0x4464, {0x8c, 0x53, 0x06, 0x16, 0x7f, 0x40, 0xcc, 0x99}};
  static const GUID kVideoOff = {0x3c0bc021, 0xc8a8, 0x4e07, {0xa9, 0x73, 0x6b, 0x14, 0xcb, 0xcb, 0x2b, 0x7e}};
  static const GUID kSleepSub = {0x238c9fa8, 0x0aad, 0x41ed, {0x83, 0xf4, 0x97, 0xbe, 0x24, 0x2c, 0x8f, 0x20}};
  static const GUID kSleepAfter = {0x29f6c1db, 0x86da, 0x48c5, {0x9f, 0xdb, 0xf2, 0xb6, 0x7b, 0x1f, 0x44, 0xda}};
  static const GUID kHibAfter = {0x9d7815a6, 0x7ee4, 0x497e, {0x88, 0x88, 0x51, 0x5a, 0x05, 0xf0, 0x23, 0x64}};
  auto both = [&](const char* label, const GUID& sub, const GUID& set) {
    DWORD ac = 0, dc = 0;
    const bool a = read_ac && read_ac(nullptr, scheme, &sub, &set, &ac) == ERROR_SUCCESS;
    const bool d = read_dc && read_dc(nullptr, scheme, &sub, &set, &dc) == ERROR_SUCCESS;
    s += std::string("; ") + label + " AC " + (a ? (ac ? std::to_string(ac) + "s" : "never") : "?") +
         " / DC " + (d ? (dc ? std::to_string(dc) + "s" : "never") : "?");
  };
  both("display-off", kVideoSub, kVideoOff);
  both("sleep", kSleepSub, kSleepAfter);
  both("hibernate", kSleepSub, kHibAfter);
  LocalFree(scheme);
  return s + "\n";
}

inline std::string env_report() {
  if (!g_fn.enum_mods) win_load_fns();
  std::string s;
  auto line = [&](const std::string& k, const std::string& v) { s += k + ": " + v + "\n"; };
  const wchar_t* cv = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
  line("os", reg_value(HKEY_LOCAL_MACHINE, cv, L"ProductName") + " " +
                 reg_value(HKEY_LOCAL_MACHINE, cv, L"DisplayVersion") + " build " +
                 reg_value(HKEY_LOCAL_MACHINE, cv, L"CurrentBuildNumber") + "." +
                 reg_value(HKEY_LOCAL_MACHINE, cv, L"UBR") + " (" +
                 reg_value(HKEY_LOCAL_MACHINE, cv, L"EditionID") +
                 "; build >= 22000 is Windows 11)");
  line("cpu", reg_value(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                        L"ProcessorNameString") +
                  ", " + std::to_string(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)) +
                  " logical");
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof ms;
  if (GlobalMemoryStatusEx(&ms)) {
    char b[160];
    snprintf(b, sizeof b, "total %.1f GiB, available %.1f GiB, commit limit %.1f GiB",
             ms.ullTotalPhys / 1073741824.0, ms.ullAvailPhys / 1073741824.0,
             ms.ullTotalPageFile / 1073741824.0);
    line("memory", b);
  }
  line("system-uptime", std::to_string(GetTickCount64() / 1000) + " s");
  // 显卡：显示适配器类键（无需管理员）
  const wchar_t* dc = L"SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
  for (int k = 0; k < 16; k++) {
    wchar_t sub[256];
    swprintf(sub, 256, L"%ls\\%04d", dc, k);
    const std::string desc = reg_value(HKEY_LOCAL_MACHINE, sub, L"DriverDesc");
    if (desc.empty()) continue;
    std::string v = desc + " | driver " + reg_value(HKEY_LOCAL_MACHINE, sub, L"DriverVersion") +
                    " (" + reg_value(HKEY_LOCAL_MACHINE, sub, L"DriverDate") + ", " +
                    reg_value(HKEY_LOCAL_MACHINE, sub, L"ProviderName") + ")";
    const std::string rs = reg_value(HKEY_LOCAL_MACHINE, sub, L"RadeonSoftwareVersion");
    if (!rs.empty()) v += " | Adrenalin " + rs;
    const std::string mem = reg_value(HKEY_LOCAL_MACHINE, sub, L"HardwareInformation.qwMemorySize");
    if (!mem.empty())
      v += " | dedicated " + std::to_string(strtoull(mem.c_str(), nullptr, 10) >> 20) + " MiB";
    line("gpu", v);
  }
  const wchar_t* gd = L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers";
  auto dflt = [](const std::string& v, const char* d) { return v.empty() ? std::string(d) : v; };
  line("tdr", "TdrLevel=" + dflt(reg_value(HKEY_LOCAL_MACHINE, gd, L"TdrLevel"), "default(3)") +
                  " TdrDelay=" + dflt(reg_value(HKEY_LOCAL_MACHINE, gd, L"TdrDelay"), "default(2s)") +
                  " TdrDdiDelay=" + dflt(reg_value(HKEY_LOCAL_MACHINE, gd, L"TdrDdiDelay"), "default(5s)"));
  s += power_info();
  {
    HKEY k = nullptr;
    const bool ld = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                                  L"SOFTWARE\\Microsoft\\Windows\\Windows Error Reporting\\LocalDumps",
                                  0, KEY_READ | KEY_WOW64_64KEY, &k) == ERROR_SUCCESS;
    if (k) RegCloseKey(k);
    line("wer-localdumps", ld ? "configured" : "not configured");
  }
  line("crash-dir", g.dir);
  line("minidump", g.dump ? "on" : "off (QWENOX_CRASH_DUMP=0)");
  s += module_versions();
  s += "environment (diagnostic prefixes; secrets masked):\n";
  if (wchar_t* blk = GetEnvironmentStringsW()) {
    for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
      const std::string kv = w2u(p);
      const size_t eq = kv.find('=');
      if (eq == std::string::npos || eq == 0) continue;
      const std::string n = kv.substr(0, eq);
      if (!diag_env_name(n)) continue;
      s += "  " + n + "=" + (secret_name(n) ? std::string("***") : kv.substr(eq + 1)) + "\n";
    }
    FreeEnvironmentStringsW(blk);
  }
  return s;
}
#else
inline std::string read_small(const char* path, size_t cap = 4096) {
  FILE* f = fopen(path, "rb");
  if (!f) return {};
  std::string s(cap, '\0');
  const size_t n = fread(&s[0], 1, cap, f);
  fclose(f);
  s.resize(n);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}
inline std::string grep_first(const char* path, const char* key) {
  FILE* f = fopen(path, "r");
  if (!f) return {};
  char line[1024];
  std::string out;
  while (fgets(line, sizeof line, f))
    if (strncmp(line, key, strlen(key)) == 0) {
      out = line + strlen(key);
      break;
    }
  fclose(f);
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  size_t a = out.find_first_not_of(" \t:=\"");
  out = a == std::string::npos ? "" : out.substr(a);
  if (!out.empty() && out.back() == '"') out.pop_back();
  return out;
}
inline std::string module_versions() {
  std::string s = "gpu-stack libraries:\n";
  dl_iterate_phdr(
      [](dl_phdr_info* i, size_t, void* d) {
        const char* n = i->dlpi_name ? i->dlpi_name : "";
        if (strstr(n, "amdhip64") || strstr(n, "rocblas") || strstr(n, "hipblaslt") ||
            strstr(n, "hsa-runtime") || strstr(n, "amd_comgr") || strstr(n, "rocprofiler"))
          *static_cast<std::string*>(d) += std::string("  ") + n + "\n";
        return 0;
      },
      &s);
  return s;
}
inline std::string env_report() {
  std::string s;
  auto line = [&](const std::string& k, const std::string& v) { s += k + ": " + v + "\n"; };
  utsname u{};
  if (uname(&u) == 0) line("kernel", std::string(u.sysname) + " " + u.release + " " + u.version + " " + u.machine);
  line("os", grep_first("/etc/os-release", "PRETTY_NAME"));
  line("cpu", grep_first("/proc/cpuinfo", "model name") + ", " +
                  std::to_string(std::thread::hardware_concurrency()) + " logical");
  line("memory", "total " + grep_first("/proc/meminfo", "MemTotal") + ", available " +
                     grep_first("/proc/meminfo", "MemAvailable"));
  line("cmdline", read_small("/proc/cmdline"));
  const char* rp = getenv("ROCM_PATH");
  line("rocm", read_small((std::string(rp ? rp : "/opt/rocm") + "/.info/version").c_str()));
  const std::string amdgpu = read_small("/sys/module/amdgpu/version");
  line("amdgpu-module", amdgpu.empty() ? "in-tree (no version file)" : amdgpu);
  line("ttm-pages-limit", read_small("/sys/module/ttm/parameters/pages_limit"));
  for (int k = 0; k < 4; k++) {
    const std::string base = "/sys/class/drm/card" + std::to_string(k) + "/device/";
    const std::string vram = read_small((base + "mem_info_vram_total").c_str());
    if (vram.empty()) continue;
    line("gpu" + std::to_string(k), "vendor " + read_small((base + "vendor").c_str()) + " device " +
                                         read_small((base + "device").c_str()) + " vram " + vram +
                                         " gtt " + read_small((base + "mem_info_gtt_total").c_str()));
  }
  line("crash-dir", g.dir);
  s += module_versions();
  s += "environment (diagnostic prefixes; secrets masked):\n";
  for (char** e = environ; e && *e; e++) {
    const std::string kv = *e;
    const size_t eq = kv.find('=');
    if (eq == std::string::npos || eq == 0) continue;
    const std::string n = kv.substr(0, eq);
    if (!diag_env_name(n)) continue;
    s += "  " + n + "=" + (secret_name(n) ? std::string("***") : kv.substr(eq + 1)) + "\n";
  }
  return s;
}
#endif

// ============================================================================
// 安装
// ============================================================================
inline void install(const char* program, const char* version) {
  if (g.installed) return;
  if (const char* e = getenv("QWENOX_CRASH_DIAG"); e && strcmp(e, "0") == 0) return;
  strncpy(g.program, program, sizeof g.program - 1);
  strncpy(g.version, version, sizeof g.version - 1);
  g.t0_us = wall_us();
  g.main_tid = tid();
  const char* dump = getenv("QWENOX_CRASH_DUMP");
  g.dump = !(dump && strcmp(dump, "0") == 0);
#ifdef _WIN32
  g.pid = int(GetCurrentProcessId());
  {
    SYSTEMTIME su, sl;
    GetSystemTime(&su);
    GetLocalTime(&sl);
    FILETIME fu, fl;
    SystemTimeToFileTime(&su, &fu);
    SystemTimeToFileTime(&sl, &fl);
    const int64_t du = int64_t((uint64_t(fu.dwHighDateTime) << 32) | fu.dwLowDateTime);
    const int64_t dl = int64_t((uint64_t(fl.dwHighDateTime) << 32) | fl.dwLowDateTime);
    g.tzoff_ms = (dl - du) / 10000 / 60000 * 60000;  // 取整到分钟
  }
  wchar_t w[1024] = {};
  const DWORD n = GetEnvironmentVariableW(L"QWENOX_CRASH_DIR", w, 1024);
  if (!n || n >= 1024) {
    GetCurrentDirectoryW(1000, w);
    wcscat(w, L"\\logs\\crash");
  }
  // 逐级创建
  for (wchar_t* q = w + 3; *q; q++)
    if (*q == L'\\' || *q == L'/') {
      const wchar_t ch = *q;
      *q = 0;
      CreateDirectoryW(w, nullptr);
      *q = ch;
    }
  CreateDirectoryW(w, nullptr);
  wcsncpy(g.wdir, w, 1023);
  w2u_buf(g.wdir, g.dir, sizeof g.dir);
  wchar_t ex[1024] = {};
  GetModuleFileNameW(nullptr, ex, 1024);
  w2u_buf(ex, g.exe, sizeof g.exe);
  g.exe_base = reinterpret_cast<const unsigned char*>(GetModuleHandleW(nullptr));
  win_load_fns();
  g_ev_go = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  g_ev_done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  HANDLE t = CreateThread(nullptr, 512 * 1024, writer_main, nullptr, 0, &g_writer_tid);
  if (t) CloseHandle(t);
  if (!t) g_ev_go = nullptr;
  ULONG guarantee = 64 * 1024;  // 主线程栈溢出时留给 VEH 的余量
  SetThreadStackGuarantee(&guarantee);
  AddVectoredExceptionHandler(1, veh);
  g.installed = true;  // rearm() 需要
  rearm();
  signal(SIGABRT, on_abort);
  _set_invalid_parameter_handler(on_invalid_param);
  // 后台进程不弹"已停止工作"对话框（否则崩溃进程会挂在对话框上不退出，
  // 监管方看不到退出码）；WER 仍照常记事件日志 / LocalDumps。abort() 不走
  // fail-fast 上报，退出码保持 3。
  SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#else
  g.pid = int(getpid());
  {
    time_t tt = time(nullptr);
    tm lt{};
    localtime_r(&tt, &lt);
    g.tzoff_ms = int64_t(lt.tm_gmtoff) * 1000;
  }
  if (const char* d = getenv("QWENOX_CRASH_DIR"); d && *d) {
    strncpy(g.dir, d, sizeof g.dir - 1);
  } else {
    char cwd[900] = ".";
    if (!getcwd(cwd, sizeof cwd)) strcpy(cwd, ".");
    snprintf(g.dir, sizeof g.dir, "%s/logs/crash", cwd);
  }
  lx_mkdirs(g.dir);
  const ssize_t r = readlink("/proc/self/exe", g.exe, sizeof g.exe - 1);
  g.exe[r > 0 ? r : 0] = 0;
  void* warm[4];
  backtrace(warm, 4);  // 首次调用会 dlopen libgcc：别留到信号处理里
  stack_t ss{};
  ss.ss_sp = g_altstack;
  ss.ss_size = sizeof g_altstack;
  sigaltstack(&ss, nullptr);
  struct sigaction sa {};
  sa.sa_sigaction = lx_on_signal;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(s, &sa, nullptr);
  struct sigaction st {};
  st.sa_sigaction = lx_on_term;
  st.sa_flags = SA_SIGINFO;
  sigemptyset(&st.sa_mask);
  struct sigaction old {};
  // 只接管仍是默认动作的 SIGTERM（启动脚本/父进程设了别的就不动）
  if (sigaction(SIGTERM, nullptr, &old) == 0 && old.sa_handler == SIG_DFL)
    sigaction(SIGTERM, &st, nullptr);
#endif
  std::set_terminate(on_terminate);
  g.installed = true;
  note("crashdiag installed (%s %s)", program, version);
}

// 在线快照（不崩溃时用：引擎 DIAG 动词 → API 诊断包里的 engine-live.txt）：
// 当前时间 + 最近 HIP 调用点 + 飞行记录 + 启动环境。
inline std::string live_text() {
  std::string s(512 * 1024, '\0');
  char tmp[4096];
  size_t n = 0;
  {
    Out o(tmp, sizeof tmp, kNoFd, &s[0], s.size());
    o.s("==== Qwenox live diagnostics ====\n");
    o.s("program: ").s(g.program).s(" ").s(g.version).nl();
    o.s("pid: ").i(g.pid).s("  time: ");
    const uint64_t now = wall_us();
    put_time(o, now);
    o.s("  uptime: ").u((now - g.t0_us) / 1000000).s(" s\n");
    put_sites(o);
    put_ring(o);
    put_env(o);
    n = o.mirrored();
  }
  s.resize(n);
  return s;
}

// ============================================================================
// 自测：故意制造一次崩溃（分发后第一次运行时验证整条链路）
// ============================================================================
inline volatile int g_never = -1;
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline int recurse(int n) {
  volatile char pad[4096];
  pad[0] = char(n);
  if (n == g_never) return 0;
  return recurse(n + 1) + pad[0];
}
inline const char* self_test_kinds() { return "write read exec abort throw stack thread fatal"; }
// 返回 false = 未知 kind（其余情况不返回）
inline bool self_test(const char* kind) {
  note("crashdiag self-test: %s", kind);
  fprintf(stderr, "[crashdiag] self-test '%s': crashing on purpose\n", kind);
  fflush(stderr);
  volatile uintptr_t bad = 0x10;
  if (!strcmp(kind, "write")) {
    *reinterpret_cast<volatile int*>(bad) = 1;
  } else if (!strcmp(kind, "read")) {
    volatile int v = *reinterpret_cast<volatile int*>(bad + 8);
    (void)v;
  } else if (!strcmp(kind, "exec")) {
    reinterpret_cast<void (*)()>(bad + 0x20)();
  } else if (!strcmp(kind, "abort")) {
    std::abort();
  } else if (!strcmp(kind, "throw")) {
    std::thread([] { throw std::runtime_error("crashdiag self-test: uncaught exception in a worker thread"); }).join();
  } else if (!strcmp(kind, "stack")) {
    fprintf(stderr, "%d\n", recurse(0));
  } else if (!strcmp(kind, "thread")) {
    std::thread([bad] {
      note("crashdiag self-test: worker thread about to write to %p", reinterpret_cast<void*>(bad));
      *reinterpret_cast<volatile int*>(bad) = 2;
    }).join();
  } else if (!strcmp(kind, "fatal")) {
    fatal("crashdiag self-test: fatal() path");
  } else {
    return false;
  }
  std::abort();  // 不该到这里
}

}  // namespace crashdiag
