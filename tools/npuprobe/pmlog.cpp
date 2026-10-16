// pmlog.cpp -- AMD ADL PMLog sensor reader for Windows (atiadlxx.dll).
//   pmlog dump                         all supported sensors, index=value
//   pmlog watch <secs> <log> [awake]   sample clocks/activity/power every 250 ms
//                                      awake: hold ES_DISPLAY_REQUIRED and nudge the
//                                      mouse every 20 s (Strix Halo pins the GPU near
//                                      600 MHz while the console display is off)
//   pmlog wake                         one 1-px mouse nudge + display-required pulse
// Sensor numbers follow ADL_PMLOG_SENSORS (adl_defines.h).
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {
constexpr int kMax = 256;
struct Single { int supported; int value; };
struct Output { int size; Single sensors[kMax]; };
typedef void*(__stdcall* MallocCb)(int);
typedef int(__stdcall* Create2)(MallocCb, int, void**);
typedef int(__stdcall* Query)(void*, int, Output*);
void* __stdcall Alloc(int n) { return std::malloc(static_cast<size_t>(n)); }

const char* Name(int i) {
  switch (i) {
    case 1: return "GFXCLK"; case 2: return "MEMCLK"; case 3: return "SOCCLK";
    case 4: return "UVDCLK1"; case 5: return "UVDCLK2"; case 6: return "VCECLK"; case 7: return "VCNCLK";
    case 8: return "T_EDGE"; case 9: return "T_MEM"; case 16: return "SOC_V"; case 17: return "SOC_P";
    case 18: return "SOC_I"; case 19: return "ACT_GFX"; case 20: return "ACT_MEM"; case 21: return "GFX_V";
    case 22: return "MEM_V"; case 23: return "ASIC_P"; case 27: return "T_HOTSPOT"; case 28: return "T_GFX";
    case 29: return "T_SOC"; case 30: return "GFX_P"; case 31: return "GFX_I"; case 32: return "T_CPU";
    case 33: return "CPU_P"; case 34: return "CPUCLK"; case 35: return "THROTTLER"; case 36: return "VCN1CLK1";
    case 37: return "VCN1CLK2"; case 38: return "SPS_CPU"; case 39: return "SPS_DGPU"; case 40: return "BUS_SPEED";
    case 41: return "BUS_LANES"; case 44: return "FCLK?"; case 46: return "SSPAIRED_ASICP";
    default: return "";
  }
}

Query g_q = nullptr;
void* g_ctx = nullptr;
int g_ad = -1;

bool Init() {
  HMODULE dll = LoadLibraryA("atiadlxx.dll");
  if (!dll) dll = LoadLibraryA("atiadlxy.dll");
  if (!dll) { std::printf("no atiadlxx.dll\n"); return false; }
  auto c2 = reinterpret_cast<Create2>(GetProcAddress(dll, "ADL2_Main_Control_Create"));
  g_q = reinterpret_cast<Query>(GetProcAddress(dll, "ADL2_New_QueryPMLogData_Get"));
  if (!c2 || !g_q || c2(Alloc, 1, &g_ctx) != 0) { std::printf("ADL init failed\n"); return false; }
  for (int i = 0; i < 8; ++i) {
    Output o; std::memset(&o, 0, sizeof o); o.size = sizeof o;
    if (g_q(g_ctx, i, &o) == 0 && (o.sensors[1].supported || o.sensors[23].supported)) { g_ad = i; return true; }
  }
  std::printf("no adapter with PMLog\n");
  return false;
}
bool Read(Output& o) { std::memset(&o, 0, sizeof o); o.size = sizeof o; return g_q(g_ctx, g_ad, &o) == 0; }
int V(const Output& o, int i) { return o.sensors[i].supported ? o.sensors[i].value : -1; }

void Nudge() {
  INPUT in[2] = {};
  in[0].type = INPUT_MOUSE; in[0].mi.dx = 1; in[0].mi.dwFlags = MOUSEEVENTF_MOVE;
  in[1].type = INPUT_MOUSE; in[1].mi.dx = -1; in[1].mi.dwFlags = MOUSEEVENTF_MOVE;
  SendInput(2, in, sizeof(INPUT));
}
}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "dump";
  if (mode == "wake") {
    SetThreadExecutionState(ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    Nudge();
    std::printf("woke\n");
    return 0;
  }
  if (!Init()) return 1;
  Output o;
  if (mode == "dump") {
    if (!Read(o)) return 1;
    for (int i = 0; i < kMax; ++i)
      if (o.sensors[i].supported) std::printf("%3d %-14s %d\n", i, Name(i), o.sensors[i].value);
    return 0;
  }
  const double secs = argc > 2 ? std::atof(argv[2]) : 60;
  FILE* f = std::fopen(argc > 3 ? argv[3] : "pmlog.log", "a");
  const bool awake = argc > 4 && std::strcmp(argv[4], "awake") == 0;
  if (awake) { SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED); Nudge(); }
  const auto t0 = std::chrono::steady_clock::now();
  double last_nudge = 0;
  std::fprintf(f, "# t gfxclk memclk socclk fclk44 act_gfx act_mem asic_w gfx_w cpu_w throttler\n");
  for (;;) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (t >= secs) break;
    if (awake && t - last_nudge >= 20) { Nudge(); last_nudge = t; }
    if (Read(o)) {
      SYSTEMTIME st; GetLocalTime(&st);
      std::fprintf(f, "%02d:%02d:%02d.%03d %.2f %d %d %d %d %d %d %d %d %d %d\n", st.wHour, st.wMinute,
                   st.wSecond, st.wMilliseconds, t, V(o, 1), V(o, 2), V(o, 3), V(o, 44), V(o, 19),
                   V(o, 20), V(o, 23) >= 0 ? V(o, 23) : V(o, 46), V(o, 30), V(o, 33), V(o, 35));
      std::fflush(f);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  std::fclose(f);
  if (awake) SetThreadExecutionState(ES_CONTINUOUS);
  return 0;
}
