// npucheck_win.hip -- Strix Halo (gfx1151) NPU 检测工具，Win32 GUI 单文件版。
//
//   [系统检查]  BIOS/AGESA（原始 SMBIOS 直读）/NPU 驱动 → SAFE/UNSAFE 结论
//   [NPU 基线]  NPU 独占 GEMM 循环（安全）
//   [GPU 基线]  GPU 显存拷贝循环（安全）
//   [并发测试]  GPU+NPU 同时满载 —— AGESA < 1.0.0.2b 会硬死机（ROCm/TheRock#8709）！
//   [死机分析]  死机重启后读取 MARKER + Kernel-Power 41 + WHEA 定性
//
// 防护：并发前双重确认；每阶段前写 npucheck_MARKER.txt 并落盘；每秒查
// WHEA-Logger，出现新事件立即中止（注意：本机实测死机时 WHEA 无预警）；
// NPU 每次 GEMM 输出与参考逐位比较（防静默损坏）。
//
// 控制台自测（不弹窗，供脚本/验证用）：
//   npucheck_win.exe --selftest           只跑系统检查
//   npucheck_win.exe --selftest-npu 10    检查 + NPU 基线 10 秒
//   npucheck_win.exe --selftest-gpu 5     检查 + GPU 基线 5 秒
//
// 构建见 build.sh（需要 out/npu_bins.inc，由 build.sh 从 $NPUASSETS/npu 生成）。

#define NOMINMAX
#include <windows.h>
#include <winevt.h>
#include <io.h>

#include <hip/hip_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "npu_bins.inc"  // kXclbin[]/kXclbinLen + kInsts[]/kInstsLen（build.sh 生成）

namespace {

// ---------------------------------------------------------------- 常量

constexpr std::uint32_t M = 4096, K = 2560, N = 1024;
constexpr std::size_t kA = std::size_t{M} * K / 8 * 9;
constexpr std::size_t kB = std::size_t{K} * N / 8 * 9;
constexpr std::size_t kC = std::size_t{M} * N / 8 * 9;
constexpr unsigned kOpcode = 3;
// AGESA >= 1.0.0.2b 修复 GPU+NPU 并发死机（AMD 2026-10-06，ROCm/TheRock#8709）
constexpr char kFixedAgesa[] = "1.0.0.2b";

const COLORREF kGreen = RGB(0, 130, 0), kRed = RGB(200, 0, 0), kOrange = RGB(190, 110, 0);

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---------------------------------------------------------------- 输出

bool g_console = false;
HWND g_main = nullptr, g_log = nullptr, g_verdict = nullptr, g_phase = nullptr;
HFONT g_font_big = nullptr, g_font_log = nullptr;
COLORREF g_verdict_color = RGB(32, 32, 32);
std::string g_dir;       // exe 所在目录（UTF-8）
std::string g_logfile;   // 本次压测日志文件（空 = 不写）
std::mutex g_file_mu;
std::atomic<bool> g_abort{false};
std::atomic<bool> g_busy{false};

enum { WM_APPENDLOG = WM_APP + 1, WM_SETVERDICT, WM_SETPHASE, WM_TESTDONE };

std::wstring U2W(const std::string& s) {
  if (s.empty()) return L"";
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (n <= 0) return L"?";
  std::wstring w(n - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}
std::string W2U(const std::wstring& w) {
  if (w.empty()) return "";
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (n <= 0) return "?";
  std::string s(n - 1, 0);
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
  return s;
}

void LogLine(const std::string& s) {
  if (g_console) {
    std::printf("%s\n", s.c_str());
    std::fflush(stdout);
    return;
  }
  auto* p = new std::wstring(U2W(s));
  if (!g_main || !PostMessageW(g_main, WM_APPENDLOG, 0, (LPARAM)p)) delete p;
}
void LogFile(const std::string& s) {  // 压测日志落盘（flush+commit，死机也能保住）
  if (g_logfile.empty()) return;
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (FILE* f = std::fopen(g_logfile.c_str(), "ab")) {
    std::fprintf(f, "%s\n", s.c_str());
    std::fflush(f);
    _commit(_fileno(f));
    std::fclose(f);
  }
}
void Log(const std::string& s) { LogLine(s); }
void LogBoth(const std::string& s) { LogLine(s); LogFile(s); }

struct VerdictMsg { std::wstring text; COLORREF color; };
void SetVerdict(const std::string& s, COLORREF c) {
  if (g_console) { LogLine("[结论] " + s); return; }
  auto* p = new VerdictMsg{U2W(s), c};
  if (!g_main || !PostMessageW(g_main, WM_SETVERDICT, 0, (LPARAM)p)) delete p;
}
void SetPhase(const std::string& s) {
  if (g_console) return;
  auto* p = new std::wstring(U2W(s));
  if (!g_main || !PostMessageW(g_main, WM_SETPHASE, 0, (LPARAM)p)) delete p;
}

std::string NowLocal() {
  SYSTEMTIME st; GetLocalTime(&st);
  char b[64];
  std::snprintf(b, sizeof b, "%04d-%02d-%02d %02d:%02d:%02d",
                st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
  return b;
}

std::string ExeDir() {
  wchar_t buf[MAX_PATH];
  DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  std::wstring p(buf, n);
  auto pos = p.find_last_of(L"\\/");
  return W2U(p.substr(0, pos));
}

// ---------------------------------------------------------------- 系统检查

std::string RegStr(const wchar_t* value) {
  wchar_t buf[256] = {};
  DWORD n = sizeof buf;
  if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\BIOS",
                   value, RRF_RT_REG_SZ, nullptr, buf, &n) != ERROR_SUCCESS)
    return "";
  return W2U(buf);
}

// 原始 SMBIOS 里直接带 "AGESA!V9 StrixHaloPI-FP11 1.0.0.1b" 这样的字符串。
bool ReadAgesa(std::string& family, std::string& version) {
  const DWORD sig = ((DWORD)'R' << 24) | ((DWORD)'S' << 16) | ((DWORD)'M' << 8) | (DWORD)'B';  // 'RSMB'
  UINT sz = GetSystemFirmwareTable(sig, 0, nullptr, 0);
  if (!sz) return false;
  std::vector<BYTE> buf(sz);
  if (!GetSystemFirmwareTable(sig, 0, buf.data(), sz)) return false;
  const char* d = reinterpret_cast<const char*>(buf.data());
  for (UINT i = 0; i + 5 < sz; ++i) {
    if (std::memcmp(d + i, "AGESA", 5) != 0) continue;
    char line[128] = {};
    UINT j = 0;
    while (i + j < sz && d[i + j] >= 32 && d[i + j] < 127 && j < 120) { line[j] = d[i + j]; ++j; }
    char fam[64] = {}, ver[64] = {};
    if (std::sscanf(line, "AGESA!V%*d %63s %31s", fam, ver) == 2 ||
        std::sscanf(line, "AGESA %63s %31s", fam, ver) == 2) {
      unsigned a, b, c, e; char suf[8] = "";
      if (std::sscanf(ver, "%u.%u.%u.%u%7[a-z]", &a, &b, &c, &e, suf) >= 4) {
        family = fam; version = ver; return true;
      }
    }
  }
  return false;
}

// 版本比较：先比数字段再比后缀字母（1.0.0.2b > 1.0.0.2 > 1.0.0.1b）
bool AgesaBelow(const std::string& v, const std::string& fixed) {
  auto parse = [](const std::string& s, unsigned n[4], char& suf) {
    suf = 0;
    char sf[8] = "";
    int k = std::sscanf(s.c_str(), "%u.%u.%u.%u%7[a-z]", &n[0], &n[1], &n[2], &n[3], sf);
    if (k >= 5) suf = sf[0];
    return k >= 4;
  };
  unsigned a[4] = {}, b[4] = {}; char sa = 0, sb = 0;
  if (!parse(v, a, sa) || !parse(fixed, b, sb)) return true;  // 解析不了按未修复对待
  for (int i = 0; i < 4; ++i)
    if (a[i] != b[i]) return a[i] < b[i];
  return sa < sb;
}

std::string RunCapture(const std::wstring& cmdline) {
  SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) return "";
  STARTUPINFOW si{}; si.cb = sizeof si;
  si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
  si.hStdOutput = wr; si.hStdError = wr; si.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION pi{};
  std::wstring cmd = cmdline;
  std::string out;
  if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
    CloseHandle(wr); wr = nullptr;
    char buf[4096]; DWORD got = 0;
    while (ReadFile(rd, buf, sizeof buf, &got, nullptr) && got) out.append(buf, got);
    WaitForSingleObject(pi.hProcess, 10000);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
  }
  if (wr) CloseHandle(wr);
  CloseHandle(rd);
  return out;
}

std::string XrtSmi(const char* args) {
  const wchar_t* exe = L"C:\\Windows\\System32\\AMD\\xrt-smi.exe";
  if (GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES) return "";
  return RunCapture(std::wstring(L"\"") + exe + L"\" " + U2W(args));
}
std::string Field(const std::string& text, const char* key) {
  auto p = text.find(key);
  if (p == std::string::npos) return "";
  p = text.find(':', p);
  if (p == std::string::npos) return "";
  ++p;
  while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) ++p;
  auto e = text.find_first_of("\r\n", p);
  return text.substr(p, e == std::string::npos ? e : e - p);
}

std::string MarkerPath() { return g_dir + "\\npucheck_MARKER.txt"; }

// verdict: 0=SAFE 1=UNSAFE 2=UNKNOWN
int RunInfoCheck() {
  Log("===== BIOS / AGESA =====");
  Log("vendor  : " + RegStr(L"BIOSVendor"));
  Log("bios    : " + RegStr(L"BIOSVersion") + "  (" + RegStr(L"BIOSReleaseDate") + ")");
  Log("machine : " + RegStr(L"SystemManufacturer") + " " + RegStr(L"SystemProductName"));
  std::string fam, ver;
  int verdict;
  if (ReadAgesa(fam, ver)) {
    Log("agesa   : " + fam + " " + ver);
    verdict = AgesaBelow(ver, kFixedAgesa) ? 1 : 0;
  } else {
    Log("agesa   : (SMBIOS 里没读到，请用 BIOS 设置页或 HWiNFO64 确认)");
    verdict = 2;
  }
  Log(std::string("修复阈值: AGESA >= ") + kFixedAgesa +
      " 才不会 GPU+NPU 并发死机（ROCm/TheRock#8709，AMD 2026-10-06）");
  if (verdict == 0) SetVerdict("SAFE — AGESA " + ver + " 已修复，可以做并发测试", kGreen);
  if (verdict == 1) SetVerdict("UNSAFE — AGESA " + ver + " 未修复，GPU+NPU 并发会硬死机！", kRed);
  if (verdict == 2) SetVerdict("UNKNOWN — 读不到 AGESA，未确认修复前不要并发", kOrange);

  Log("");
  Log("===== NPU =====");
  const std::string ex = XrtSmi("examine");
  if (ex.empty()) {
    Log("xrt-smi 不存在 → NPU 驱动未安装或未启用");
  } else {
    Log("XRT Version        : " + Field(ex, "\n  Version"));
    Log("NPU Driver Version : " + Field(ex, "NPU Driver Version"));
    Log("NPU Firmware       : " + Field(ex, "NPU Firmware Version"));
    Log(ex.find("|NPU") != std::string::npos ? "NPU 设备: 在线" : "NPU 设备: 未找到！");
    Log("Power Mode         : " + Field(XrtSmi("examine -r platform"), "Power Mode"));
  }
  if (GetFileAttributesW(U2W(MarkerPath()).c_str()) != INVALID_FILE_ATTRIBUTES) {
    Log("");
    Log("!! 发现上次压测的 MARKER（测试中断，可能死机）→ 点 [死机分析]");
  }
  return verdict;
}

// ---------------------------------------------------------------- 事件日志

std::wstring UtcStr(const SYSTEMTIME& st) {
  wchar_t b[64];
  swprintf(b, 64, L"%04d-%02d-%02dT%02d:%02d:%02d.000Z",
           st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
  return b;
}

std::vector<std::wstring> QueryEvents(const wchar_t* xpath) {
  std::vector<std::wstring> out;
  EVT_HANDLE q = EvtQuery(nullptr, L"System", xpath, EvtQueryForwardDirection);
  if (!q) return out;
  for (;;) {
    EVT_HANDLE ev[16]; DWORD got = 0;
    if (!EvtNext(q, 16, ev, 0, 0, &got)) break;
    for (DWORD i = 0; i < got; ++i) {
      DWORD need = 0;
      EvtRender(nullptr, ev[i], EvtRenderEventXml, 0, nullptr, &need, nullptr);
      if (need) {
        std::wstring xml(need / 2 + 1, 0);
        if (EvtRender(nullptr, ev[i], EvtRenderEventXml, (DWORD)(xml.size() * 2),
                      xml.data(), &need, nullptr)) {
          xml.resize(need / 2 ? need / 2 - 1 : 0);
          out.push_back(xml);
        }
      }
      EvtClose(ev[i]);
    }
  }
  EvtClose(q);
  return out;
}

int WheaCountSince(const SYSTEMTIME& utc) {
  const std::wstring xp = L"*[System[Provider[@Name='Microsoft-Windows-WHEA-Logger'] "
                          L"and TimeCreated[@SystemTime>='" + UtcStr(utc) + L"']]]";
  return (int)QueryEvents(xp.c_str()).size();
}

std::string XmlField(const std::wstring& xml, const wchar_t* open, const wchar_t* close) {
  auto a = xml.find(open);
  if (a == std::wstring::npos) return "";
  a += wcslen(open);
  auto b = xml.find(close, a);
  if (b == std::wstring::npos) return "";
  return W2U(xml.substr(a, b - a));
}

// ---------------------------------------------------------------- NPU GEMM
// （编码/布局与 tools/npuprobe/npuio.hip 相同：bfp16ebs8，4096x2560x1024）

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
void APos(std::size_t g, std::uint32_t& r, std::uint32_t& c) {
  const std::uint32_t tiles_k = K / 64;
  const std::size_t t = g / 1024;
  const std::uint32_t gi = g % 1024;
  const std::uint32_t tr = t / tiles_k, tc = t % tiles_k;
  const std::uint32_t rib = gi & 7, block = (gi >> 3) & 1, sc = (gi >> 4) & 7, sr = gi >> 7;
  r = tr * 128 + sr * 16 + block * 8 + rib;
  c = tc * 64 + sc * 8;
}
void BPos(std::size_t g, std::uint32_t& k, std::uint32_t& n) {
  const std::uint32_t tiles_k = K / 64;
  const std::size_t t = g / 1024;
  const std::uint32_t gi = g % 1024;
  const std::uint32_t tc = t / tiles_k, tr = t % tiles_k;
  const std::uint32_t cib = gi & 7, block = (gi >> 3) & 1, sr = (gi >> 4) & 7, sc2 = gi >> 7;
  k = tr * 64 + sr * 8;
  n = tc * 128 + (sc2 * 2 + block) * 8 + cib;
}
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
  std::vector<std::uint8_t> a, b;   // 编码后
  std::vector<float> aq, bq;        // 反量化参考
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
double CheckC(const std::uint8_t* c, const std::vector<std::uint32_t>& rows,
              const std::vector<double>& ref) {
  std::vector<float> got(rows.size() * N, NAN);
  std::vector<int> idx(M, -1);
  for (std::size_t i = 0; i < rows.size(); ++i) idx[rows[i]] = static_cast<int>(i);
  for (std::size_t g = 0; g < kC / 9; ++g) {
    std::uint32_t r, col; CPos(g, r, col);
    if (idx[r] < 0) continue;
    for (int i = 0; i < 8; ++i) got[idx[r] * N + col + i] = GroupValue(&c[g * 9], i);
  }
  double num = 0, den = 0;
  for (std::size_t ri = 0; ri < rows.size(); ++ri)
    for (std::uint32_t n = 0; n < N; ++n) {
      const double e = got[ri * N + n] - ref[ri * N + n];
      if (std::isnan(e)) return 1e9;
      num += e * e; den += ref[ri * N + n] * ref[ri * N + n];
    }
  return std::sqrt(num / (den + 1e-30));
}

// 把内嵌的 xclbin/insts 写到 %TEMP%\npucheck\ 再按文件加载（XRT 2.19 只认文件路径最稳）
std::string WriteBins() {
  wchar_t tmp[MAX_PATH];
  GetTempPathW(MAX_PATH, tmp);
  std::wstring dir = std::wstring(tmp) + L"npucheck\\";
  CreateDirectoryW(dir.c_str(), nullptr);
  const std::wstring xp = dir + L"final_4096x2560x1024_128x64x128.xclbin";
  const std::wstring ip = dir + L"insts_4096x2560x1024_128x64x128.txt";
  auto put = [](const std::wstring& p, const unsigned char* d, unsigned n) {
    if (FILE* f = _wfopen(p.c_str(), L"wb")) {
      std::fwrite(d, 1, n, f);
      std::fclose(f);
      return true;
    }
    return false;
  };
  if (!put(xp, kXclbin, kXclbinLen) || !put(ip, kInsts, kInstsLen)) return "";
  return W2U(dir);
}

struct NpuCtx {
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel kern;
  xrt::bo instr;
  std::size_t words = 0;
  int grp = 0;
  std::chrono::milliseconds timeout{3000};
  double Run(xrt::bo& a, xrt::bo& b, xrt::bo& c) {
    const double t0 = Now();
    auto run = kern(kOpcode, instr, words, a, b, c);
    const auto st = run.wait(timeout);
    if (st != ERT_CMD_STATE_COMPLETED) return -1;
    return (Now() - t0) * 1e3;
  }
};

bool NpuSetup(NpuCtx& npu, std::string& err) {
  try {
    if (!LoadLibraryW(L"xrt_coreutil.dll")) {  // delay-load 保护：没装 NPU 驱动也能跑别的
      err = "xrt_coreutil.dll 加载失败（NPU 驱动未安装）";
      return false;
    }
    const std::string dir = WriteBins();
    if (dir.empty()) { err = "写出内嵌 xclbin 失败"; return false; }
    npu.dev = xrt::device(0u);
    const xrt::xclbin xb{dir + "final_4096x2560x1024_128x64x128.xclbin"};
    npu.dev.register_xclbin(xb);
    npu.ctx = xrt::hw_context(npu.dev, xb.get_uuid());
    npu.kern = xrt::kernel(npu.ctx, xb.get_kernels().at(0).get_name());
    std::string full = dir + "insts_4096x2560x1024_128x64x128.txt";
    if (FILE* f = std::fopen(full.c_str(), "rb")) {
      std::vector<char> w(kInstsLen);
      std::fread(w.data(), 1, kInstsLen, f);
      std::fclose(f);
      npu.words = kInstsLen / 4;
      npu.instr = xrt::bo(npu.dev, kInstsLen, xrt::bo::flags::cacheable, npu.kern.group_id(1));
      std::memcpy(npu.instr.map<void*>(), w.data(), kInstsLen);
      npu.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    } else { err = "读 insts 失败"; return false; }
    npu.grp = npu.kern.group_id(3);
    return true;
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
}

struct StressResult { std::uint64_t runs = 0, bad = 0; bool timeout = false; std::string abort; };

// 4 深流水 NPU GEMM 循环；每次输出与参考逐位比较；每秒一行日志（落盘）。
// check() 返回非空字符串 = 立即中止（如 WHEA 预警）。
StressResult NpuStress(NpuCtx& npu, const Data& data,
                       const std::vector<std::uint32_t>& rows,
                       const std::vector<double>& ref, double secs,
                       const std::function<std::string()>& check) {
  StressResult R;
  xrt::bo ba(npu.dev, kA, xrt::bo::flags::host_only, npu.grp);
  xrt::bo bb(npu.dev, kB, xrt::bo::flags::host_only, npu.grp);
  xrt::bo bc[4];
  for (auto& c : bc) c = xrt::bo(npu.dev, kC, xrt::bo::flags::host_only, npu.grp);
  std::memcpy(ba.map<void*>(), data.a.data(), kA);
  std::memcpy(bb.map<void*>(), data.b.data(), kB);
  ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (npu.Run(ba, bb, bc[0]) < 0) { R.timeout = true; LogBoth("TIMEOUT 参考运行未完成"); return R; }
  bc[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<std::uint8_t> want(bc[0].map<std::uint8_t*>(), bc[0].map<std::uint8_t*>() + kC);
  const double rel = CheckC(want.data(), rows, ref);
  if (rel > 0.03) {
    LogBoth("SETUP_FAIL 参考运行对不上 CPU 参考 rel=" + std::to_string(rel));
    R.timeout = true;
    return R;
  }
  LogBoth("READY secs=" + std::to_string((long long)secs));
  const double t_start = Now();
  double t_sec = t_start;
  std::uint64_t runs_sec = 0;
  double worst = 0;
  xrt::run q[4];
  for (int i = 0; i < 4; ++i) q[i] = npu.kern(kOpcode, npu.instr, npu.words, ba, bb, bc[i]);
  for (std::uint64_t it = 0;; ++it) {
    const int i = static_cast<int>(it & 3);
    const double tw = Now();
    const auto st = q[i].wait(npu.timeout);
    worst = std::max(worst, (Now() - tw) * 1e3);
    if (st != ERT_CMD_STATE_COMPLETED) {
      LogBoth("TIMEOUT t=" + std::to_string(Now() - t_start));
      R.timeout = true;
      break;
    }
    bc[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (std::memcmp(bc[i].map<void*>(), want.data(), kC) != 0) {
      ++R.bad;
      if (R.bad <= 5) LogBoth("MISMATCH t=" + std::to_string(Now() - t_start));
    }
    ++R.runs; ++runs_sec;
    const double now = Now();
    const bool done = now - t_start >= secs || g_abort.load();
    if (now - t_sec >= 1.0 || done) {
      char line[160];
      std::snprintf(line, sizeof line, "t=%6.1f runs=%5llu ms/run=%.3f worst_wait_ms=%.2f mismatches=%llu",
                    now - t_start, (unsigned long long)runs_sec,
                    (now - t_sec) * 1e3 / std::max<std::uint64_t>(runs_sec, 1), worst,
                    (unsigned long long)R.bad);
      LogBoth(line);
      SetPhase(line);
      t_sec = now; runs_sec = 0; worst = 0;
      if (check) {
        std::string why = check();
        if (!why.empty()) { R.abort = why; break; }
      }
    }
    if (done) break;
    q[i] = npu.kern(kOpcode, npu.instr, npu.words, ba, bb, bc[i]);
  }
  for (auto& r : q) r.wait(npu.timeout);
  char line[128];
  std::snprintf(line, sizeof line, "END runs=%llu mismatches=%llu",
                (unsigned long long)R.runs, (unsigned long long)R.bad);
  LogBoth(line);
  return R;
}

// ---------------------------------------------------------------- GPU 负载

__global__ void CopyKernel(const uint4* __restrict__ s, uint4* __restrict__ d, std::size_t n) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
       i += std::size_t{gridDim.x} * blockDim.x)
    d[i] = s[i];
}

struct GpuLoad {
  void *a = nullptr, *b = nullptr;
  std::size_t bytes = std::size_t{1} << 30;
  bool Setup(std::string& err) {
    hipError_t e1 = hipMalloc(&a, bytes), e2 = hipMalloc(&b, bytes);
    if (e1 != hipSuccess || e2 != hipSuccess) {
      err = std::string("hipMalloc: ") + hipGetErrorString(e1) + " / " + hipGetErrorString(e2);
      return false;
    }
    hipMemset(a, 1, bytes);
    return true;
  }
  // delay 秒后开始，跑 secs 秒或直到 g_abort；返回 GB/s
  double RunLoop(double delay, double secs) {
    const double t0 = Now();
    while (Now() - t0 < delay && !g_abort.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::uint64_t it = 0;
    const double t1 = Now();
    for (;; ++it) {
      CopyKernel<<<2048, 256>>>(reinterpret_cast<const uint4*>(a), reinterpret_cast<uint4*>(b), bytes / 16);
      if (it % 8 == 7) {
        if (hipDeviceSynchronize() != hipSuccess) { g_abort = true; return 0; }
        if (g_abort.load() || Now() - t1 >= secs) break;
      }
    }
    hipDeviceSynchronize();
    const double el = Now() - t1;
    return 2.0 * bytes * (it + 1) / el / 1e9;
  }
  ~GpuLoad() {
    if (a) hipFree(a);
    if (b) hipFree(b);
  }
};

// ---------------------------------------------------------------- MARKER

void WriteMarker(const char* phase, const std::string& start) {
  if (FILE* f = std::fopen(MarkerPath().c_str(), "wb")) {
    std::fprintf(f, "start=%s\nphase=%s\nphase_start=%s\nlogfile=%s\n",
                 start.c_str(), phase, NowLocal().c_str(), g_logfile.c_str());
    std::fflush(f);
    _commit(_fileno(f));
    std::fclose(f);
  }
}
void DeleteMarker() { DeleteFileW(U2W(MarkerPath()).c_str()); }

// ---------------------------------------------------------------- 压测编排

enum TestMode { kNpu = 1, kGpu = 2, kBoth = 3 };

void RunTest(TestMode mode, int secs) {
  SYSTEMTIME utc0; GetSystemTime(&utc0);
  const std::string start = NowLocal();
  const int whea0 = WheaCountSince(utc0);
  g_logfile = g_dir + "\\npucheck_stress_" + start.substr(0, 10) + "_" +
              start.substr(11, 2) + start.substr(14, 2) + start.substr(17, 2) + ".log";
  LogBoth("== 测试开始 " + start + "  mode=" + std::to_string(mode) +
          "  secs=" + std::to_string(secs) + "  WHEA 基线=" + std::to_string(whea0));

  NpuCtx npu;
  Data data;
  std::vector<std::uint32_t> rows;
  std::vector<double> ref;
  bool npu_ok = false;
  if (mode & kNpu) {
    SetPhase("初始化 NPU ...");
    Log("初始化 NPU（加载 xclbin、生成测试数据、CPU 参考）...");
    std::string err;
    if (!NpuSetup(npu, err)) {
      Log("NPU 初始化失败: " + err);
      SetVerdict("FAIL — NPU 初始化失败", kRed);
      return;
    }
    data = MakeData();
    rows = SampleRows();
    ref = Reference(data, rows);
    npu_ok = true;
  }
  GpuLoad gpu;
  if (mode & kGpu) {
    SetPhase("初始化 GPU ...");
    std::string err;
    if (!gpu.Setup(err)) {
      Log("GPU 初始化失败: " + err);
      SetVerdict("FAIL — GPU 初始化失败", kRed);
      return;
    }
  }

  StressResult nres;
  std::string gpu_line;
  std::atomic<bool> gpu_err{false};

  if (mode == kNpu || mode == kGpu) {
    // 独占基线
    WriteMarker(mode == kNpu ? "npu-only" : "gpu-only", start);
    SetPhase(mode == kNpu ? "NPU 基线运行中" : "GPU 基线运行中");
    if (mode == kNpu) {
      nres = NpuStress(npu, data, rows, ref, secs, nullptr);
    } else {
      const double bw = gpu.RunLoop(0, secs);
      gpu_line = "GPU 显存拷贝: " + std::to_string(bw).substr(0, 6) + " GB/s";
      LogBoth(gpu_line);
    }
  } else {
    // 并发：NPU 先跑 3 秒拿独占基线，然后 GPU 加入 secs 秒
    WriteMarker("concurrent", start);
    SetPhase("!! GPU+NPU 并发运行中 —— 未修复固件上可能死机 !!");
    std::thread gt([&] {
      try {
        const double bw = gpu.RunLoop(3.0, secs);
        gpu_line = "GPU 显存拷贝(并发段): " + std::to_string(bw).substr(0, 6) + " GB/s";
      } catch (...) { gpu_err = true; }
    });
    nres = NpuStress(npu, data, rows, ref, secs + 3.0, [&]() -> std::string {
      const int w = WheaCountSince(utc0);
      if (w > whea0) return "WHEA 硬件错误事件（" + std::to_string(w) + " 条），紧急中止";
      if (gpu_err.load()) return "GPU 线程出错";
      return "";
    });
    g_abort = true;
    gt.join();
    if (!gpu_line.empty()) LogBoth(gpu_line);
  }

  const int whea1 = WheaCountSince(utc0);
  LogBoth("WHEA 新增事件: " + std::to_string(whea1 - whea0));

  // 判定（机器还活着才会走到这；死了就等 postmortem）
  DeleteMarker();
  if (!nres.abort.empty()) {
    LogBoth("RESULT: ABORTED - " + nres.abort);
    SetVerdict("ABORTED — " + nres.abort, kOrange);
  } else if (nres.timeout || nres.bad > 0) {
    LogBoth("RESULT: FAIL - NPU 结果错误/超时");
    SetVerdict("FAIL — NPU 结果出现错误或超时！", kRed);
  } else {
    LogBoth("RESULT: PASS - 存活，NPU 结果全部逐位一致，无新增 WHEA");
    SetVerdict("PASS — 测试存活，无硬件错误事件", kGreen);
  }
  g_logfile.clear();
}

struct TestArgs { TestMode mode; int secs; };
DWORD WINAPI TestThread(void* p) {
  std::unique_ptr<TestArgs> a(static_cast<TestArgs*>(p));
  g_abort = false;
  try {
    RunTest(a->mode, a->secs);
  } catch (const std::exception& e) {
    Log(std::string("异常: ") + e.what());
    SetVerdict(std::string("FAIL — 异常: ") + e.what(), kRed);
  }
  g_busy = false;
  PostMessageW(g_main, WM_TESTDONE, 0, 0);
  return 0;
}

// ---------------------------------------------------------------- 死机分析

bool ParseLocalTime(const std::string& s, SYSTEMTIME& st) {
  int y, mo, d, h, mi, se;
  if (std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &mi, &se) != 6) return false;
  std::memset(&st, 0, sizeof st);
  st.wYear = y; st.wMonth = mo; st.wDay = d; st.wHour = h; st.wMinute = mi; st.wSecond = se;
  return true;
}

void RunPostmortem() {
  const std::string mp = MarkerPath();
  FILE* f = std::fopen(mp.c_str(), "rb");
  if (!f) {
    Log("没有找到 MARKER —— 上次压测正常结束（或没跑过）。");
    SetVerdict("无中断的测试记录", kGreen);
    return;
  }
  std::string start, phase, pstart, logfile;
  char line[512];
  while (std::fgets(line, sizeof line, f)) {
    const std::string l = line;
    auto eq = l.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = l.substr(0, eq);
    std::string v = l.substr(eq + 1);
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
    if (k == "start") start = v;
    if (k == "phase") phase = v;
    if (k == "phase_start") pstart = v;
    if (k == "logfile") logfile = v;
  }
  std::fclose(f);
  Log("发现中断的压测: " + mp);
  Log("  开始=" + start + "  中断时阶段=" + phase + "（阶段开始=" + pstart + "）");
  Log("");
  if (!logfile.empty()) {
    Log("---- 最后的心跳（" + logfile + " 末尾 3 行）----");
    if (FILE* lf = std::fopen(logfile.c_str(), "rb")) {
      std::fseek(lf, 0, SEEK_END);
      long sz = std::ftell(lf);
      std::fseek(lf, std::max(0L, sz - 4000), SEEK_SET);
      std::string tail(sz > 4000 ? 4000 : sz, 0);
      std::fread(tail.data(), 1, tail.size(), lf);
      std::fclose(lf);
      std::vector<std::string> lines;
      std::string cur;
      for (char ch : tail) {
        if (ch == '\n') { lines.push_back(cur); cur.clear(); } else if (ch != '\r') cur += ch;
      }
      const std::size_t from = lines.size() > 3 ? lines.size() - 3 : 0;
      for (std::size_t i = from; i < lines.size(); ++i) Log("  " + lines[i]);
      WIN32_FILE_ATTRIBUTE_DATA fa;
      if (GetFileAttributesExW(U2W(logfile).c_str(), GetFileExInfoStandard, &fa)) {
        SYSTEMTIME utc, loc;
        FileTimeToSystemTime(&fa.ftLastWriteTime, &utc);
        SystemTimeToTzSpecificLocalTime(nullptr, &utc, &loc);
        char b[64];
        std::snprintf(b, sizeof b, "  日志最后写入: %04d-%02d-%02d %02d:%02d:%02d（≈ 死机时刻）",
                      loc.wYear, loc.wMonth, loc.wDay, loc.wHour, loc.wMinute, loc.wSecond);
        Log(b);
      }
    } else {
      Log("  (日志文件打不开)");
    }
  }
  Log("");
  SYSTEMTIME st_local{}, st_utc{};
  const bool have_t = ParseLocalTime(start, st_local) &&
                      TzSpecificLocalTimeToSystemTime(nullptr, &st_local, &st_utc);
  if (have_t) {
    Log("---- Kernel-Power 41（非正常关机）事件（自 " + start + " 起）----");
    const std::wstring xp = L"*[System[(EventID=41) and TimeCreated[@SystemTime>='" +
                            UtcStr(st_utc) + L"']]]";
    auto evs = QueryEvents(xp.c_str());
    if (evs.empty()) Log("  (无)");
    std::string first_kp41;
    for (auto& x : evs) {
      const std::string tc = XmlField(x, L"SystemTime='", L"'");
      const std::string bc = XmlField(x, L"Name=\"BugcheckCode\">", L"<");
      Log("  " + tc + "  BugcheckCode=" + bc);
      if (first_kp41.empty()) first_kp41 = tc;
    }
    Log("");
    Log("---- WHEA 事件（自 " + start + " 起）: " + std::to_string(WheaCountSince(st_utc)) + " 条");
    Log("");
    if (!first_kp41.empty()) {
      // UTC ISO → SYSTEMTIME → 本地，算与阶段开始的间隔
      SYSTEMTIME kut{}, kloc{}, pst{};
      unsigned yy, mo, dd, hh, mm, ss;
      if (std::sscanf(first_kp41.c_str(), "%u-%u-%uT%u:%u:%u", &yy, &mo, &dd, &hh, &mm, &ss) == 6) {
        kut.wYear = yy; kut.wMonth = mo; kut.wDay = dd; kut.wHour = hh; kut.wMinute = mm; kut.wSecond = ss;
        SystemTimeToTzSpecificLocalTime(nullptr, &kut, &kloc);
        char b[128];
        std::snprintf(b, sizeof b, "确认硬死机/断电（Kernel-Power 41 @ 本地 %02d:%02d:%02d）",
                      kloc.wHour, kloc.wMinute, kloc.wSecond);
        Log("结论: " + std::string(b));
        if (ParseLocalTime(pstart, pst)) {
          FILETIME ft1, ft2;
          SystemTimeToFileTime(&pst, &ft1);
          SystemTimeToFileTime(&kloc, &ft2);
          const long long ds = ((*(long long*)&ft2 - *(long long*)&ft1) / 10000000LL);
          Log("      距 '" + phase + "' 阶段开始约 " + std::to_string(ds) + " 秒。");
        }
        SetVerdict("上次并发测试硬死机（阶段: " + phase + "）", kRed);
      }
    } else {
      Log("结论: 压测中断但没有 Kernel-Power 41 记录 —— 可能是手动结束。");
      SetVerdict("上次测试中断（未见死机记录）", kOrange);
    }
  }
  Log("（确认后可删除标记: " + mp + "）");
}

// ---------------------------------------------------------------- GUI

HWND g_secs_edit = nullptr;
HWND g_buttons[7] = {};
const wchar_t* kBtnText[7] = {L"系统检查", L"NPU 基线", L"GPU 基线", L"并发测试", L"死机分析", L"中止", L"清空日志"};
const int kBtnId[7] = {101, 102, 103, 104, 105, 106, 107};

int GetSecs() {
  wchar_t b[16] = {};
  GetWindowTextW(g_secs_edit, b, 15);
  int s = _wtoi(b);
  return s < 5 ? 5 : (s > 7200 ? 7200 : s);
}

void StartTest(TestMode mode) {
  if (g_busy.load()) return;
  if (mode == kBoth) {
    std::string fam, ver;
    const bool have = ReadAgesa(fam, ver);
    if (!have || AgesaBelow(ver, kFixedAgesa)) {
      const std::wstring msg1 =
          L"警告：当前 AGESA " + U2W(have ? ver : "?") +
          L" 未修复 GPU+NPU 并发死机缺陷（需要 >= " + U2W(kFixedAgesa) + L"）。\n\n"
          L"继续测试【预期会硬死机】，只能断电重启。\n"
          L"请先保存所有未保存的工作。仍要继续吗？";
      if (MessageBoxW(g_main, msg1.c_str(), L"并发死机测试", MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2) != IDYES)
        return;
      if (MessageBoxW(g_main, L"最后确认：机器可能完全卡死（鼠标键盘无响应）。\n真的开始吗？",
                      L"并发死机测试", MB_YESNO | MB_ICONSTOP | MB_DEFBUTTON2) != IDYES)
        return;
    } else {
      if (MessageBoxW(g_main, L"AGESA 已修复，理论上安全；仍属压力测试。\n开始 GPU+NPU 并发测试吗？",
                      L"并发测试", MB_YESNO | MB_ICONQUESTION) != IDYES)
        return;
    }
  }
  g_busy = true;
  for (int i = 0; i < 7; ++i) EnableWindow(g_buttons[i], kBtnId[i] == 106);
  auto* a = new TestArgs{mode, GetSecs()};
  CloseHandle(CreateThread(nullptr, 0, TestThread, a, 0, nullptr));
}

void AppendLog(const std::wstring& s) {
  const int len = GetWindowTextLengthW(g_log);
  SendMessageW(g_log, EM_SETSEL, len, len);
  SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)(s + L"\r\n").c_str());
}

LRESULT CALLBACK WndProc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_COMMAND:
      switch (LOWORD(wp)) {
        case 101: RunInfoCheck(); break;
        case 102: StartTest(kNpu); break;
        case 103: StartTest(kGpu); break;
        case 104: StartTest(kBoth); break;
        case 105: RunPostmortem(); break;
        case 106: g_abort = true; SetPhase("正在中止..."); break;
        case 107: SetWindowTextW(g_log, L""); break;
      }
      return 0;
    case WM_APPENDLOG: {
      std::unique_ptr<std::wstring> p(reinterpret_cast<std::wstring*>(lp));
      AppendLog(*p);
      return 0;
    }
    case WM_SETVERDICT: {
      std::unique_ptr<VerdictMsg> p(reinterpret_cast<VerdictMsg*>(lp));
      g_verdict_color = p->color;
      SetWindowTextW(g_verdict, p->text.c_str());
      InvalidateRect(g_verdict, nullptr, TRUE);
      return 0;
    }
    case WM_SETPHASE: {
      std::unique_ptr<std::wstring> p(reinterpret_cast<std::wstring*>(lp));
      SetWindowTextW(g_phase, p->c_str());
      return 0;
    }
    case WM_TESTDONE:
      for (int i = 0; i < 7; ++i) EnableWindow(g_buttons[i], kBtnId[i] != 106);
      SetPhase("");
      return 0;
    case WM_CTLCOLORSTATIC:
      if ((HWND)lp == g_verdict) {
        SetTextColor((HDC)wp, g_verdict_color);
        SetBkMode((HDC)wp, TRANSPARENT);
        return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);
      }
      break;
    case WM_SIZE: {
      const int cw = LOWORD(lp), ch = HIWORD(lp);
      MoveWindow(g_log, 8, 116, cw - 16, ch - 124, TRUE);
      return 0;
    }
    case WM_CLOSE:
      if (g_busy.load()) {
        MessageBoxW(w, L"测试进行中 —— 请先点 [中止] 等待测试收尾。", L"NPU 检测", MB_ICONINFORMATION);
        return 0;
      }
      DestroyWindow(w);
      return 0;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(w, msg, wp, lp);
}

int GuiMain() {
  g_dir = ExeDir();
  g_font_big = CreateFontW(-20, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET,
                           0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
  g_font_log = CreateFontW(-15, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
                           0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
  WNDCLASSW wc{};
  wc.lpfnWndProc = WndProc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
  wc.lpszClassName = L"NpuCheckWin";
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  RegisterClassW(&wc);
  g_main = CreateWindowExW(0, wc.lpszClassName, L"NPU 检测工具 — Strix Halo / gfx1151",
                           WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_THICKFRAME,
                           CW_USEDEFAULT, CW_USEDEFAULT, 960, 660,
                           nullptr, nullptr, wc.hInstance, nullptr);
  const int widths[7] = {96, 96, 96, 104, 96, 72, 88};
  int x = 8;
  HFONT font_btn = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
  for (int i = 0; i < 7; ++i) {
    g_buttons[i] = CreateWindowExW(0, L"BUTTON", kBtnText[i], WS_CHILD | WS_VISIBLE,
                                   x, 8, widths[i], 30, g_main, (HMENU)(INT_PTR)kBtnId[i],
                                   wc.hInstance, nullptr);
    SendMessageW(g_buttons[i], WM_SETFONT, (WPARAM)font_btn, TRUE);
    x += widths[i] + 8;
  }
  EnableWindow(g_buttons[5], FALSE);  // 中止
  HWND lab = CreateWindowExW(0, L"STATIC", L"时长(秒):", WS_CHILD | WS_VISIBLE,
                             x + 4, 13, 62, 20, g_main, nullptr, wc.hInstance, nullptr);
  SendMessageW(lab, WM_SETFONT, (WPARAM)font_btn, TRUE);
  g_secs_edit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"60",
                                WS_CHILD | WS_VISIBLE | ES_NUMBER,
                                x + 68, 10, 56, 24, g_main, nullptr, wc.hInstance, nullptr);
  SendMessageW(g_secs_edit, WM_SETFONT, (WPARAM)font_btn, TRUE);
  g_verdict = CreateWindowExW(0, L"STATIC", L"点击 [系统检查] 开始",
                              WS_CHILD | WS_VISIBLE | SS_CENTER | SS_CENTERIMAGE,
                              8, 46, 928, 30, g_main, nullptr, wc.hInstance, nullptr);
  SendMessageW(g_verdict, WM_SETFONT, (WPARAM)g_font_big, TRUE);
  g_phase = CreateWindowExW(0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_CENTER,
                            8, 82, 928, 20, g_main, nullptr, wc.hInstance, nullptr);
  SendMessageW(g_phase, WM_SETFONT, (WPARAM)g_font_log, TRUE);
  g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                          WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
                          8, 116, 928, 500, g_main, nullptr, wc.hInstance, nullptr);
  SendMessageW(g_log, WM_SETFONT, (WPARAM)g_font_log, TRUE);
  ShowWindow(g_main, SW_SHOW);

  Log("NPU 检测工具  " + NowLocal());
  Log("先点 [系统检查] 看 BIOS/AGESA 结论。[并发测试] 在未修复固件上会硬死机，务必先保存工作。");
  if (GetFileAttributesW(U2W(MarkerPath()).c_str()) != INVALID_FILE_ATTRIBUTES) {
    Log("");
    Log("!! 发现上次压测的 MARKER（测试中断，可能死机过）→ 点 [死机分析]");
    SetVerdict("上次测试中断 — 可能死机，请点 [死机分析]", kOrange);
  }

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  return 0;
}

// ---------------------------------------------------------------- 控制台自测

int ConsoleMain(int argc, char** argv) {
  g_console = true;
  g_dir = ExeDir();
  // stdout 被捕获（管道/文件）时直接用；否则附加父控制台
  DWORD cm;
  HANDLE ho = GetStdHandle(STD_OUTPUT_HANDLE);
  if (!ho || GetConsoleMode(ho, &cm)) {
    AttachConsole(ATTACH_PARENT_PROCESS);
    FILE* f = nullptr;
    freopen_s(&f, "CONOUT$", "w", stdout);
  }
  SetConsoleOutputCP(CP_UTF8);
  const int verdict = RunInfoCheck();
  Log("");
  int rc = verdict == 0 ? 0 : 1;
  double secs = 0;
  if (argc > 2) secs = std::atof(argv[2]);
  if (!std::strcmp(argv[1], "--selftest-npu") && secs > 0) {
    Log("== NPU 基线 " + std::to_string((int)secs) + "s");
    NpuCtx npu; std::string err;
    if (!NpuSetup(npu, err)) { Log("NPU 初始化失败: " + err); return 1; }
    const Data data = MakeData();
    const auto rows = SampleRows();
    const auto ref = Reference(data, rows);
    const auto r = NpuStress(npu, data, rows, ref, secs, nullptr);
    if (r.timeout || r.bad) { Log("RESULT: FAIL"); return 1; }
    Log("RESULT: PASS");
  } else if (!std::strcmp(argv[1], "--selftest-gpu") && secs > 0) {
    Log("== GPU 基线 " + std::to_string((int)secs) + "s");
    GpuLoad gpu; std::string err;
    if (!gpu.Setup(err)) { Log("GPU 初始化失败: " + err); return 1; }
    const double bw = gpu.RunLoop(0, secs);
    Log("GPU 显存拷贝: " + std::to_string(bw).substr(0, 6) + " GB/s");
    Log("RESULT: PASS");
  }
  Log(std::string("verdict=") + (verdict == 0 ? "SAFE" : verdict == 1 ? "UNSAFE" : "UNKNOWN"));
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && !std::strncmp(argv[1], "--selftest", 10)) return ConsoleMain(argc, argv);
  return GuiMain();
}
