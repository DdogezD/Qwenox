// win_dpi.h — Windows 高 DPI 公共小工具（头文件内联，零新增 TU / 库依赖）。
//
// 进程的 PerMonitorV2 声明优先由 manifest 提供；rc 工具不可用时 build_win.sh
// 不会嵌入 manifest，此时 enable_per_monitor_v2() 在第一个窗口创建前兜底。
// 必须在任何窗口创建之前调用：窗口建好后再声明 DPI 感知会失败，进程只能
// 停留在未声明状态（被系统位图拉伸）。
//
// 这里的 GetDpiForWindow / AdjustWindowRectExForDpi 都走 GetProcAddress：
// 项目用 TheRock 自带 clang + Windows SDK 头编译，SDK 头版本不一定声明这些
// Win10 1607+ API；动态取地址同时保留 Win7/8 的降级路径。
#pragma once

#include <windows.h>

namespace windpi {

// DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 的固定值（Win10 1703+）。
constexpr INT_PTR kPerMonitorAwareV2 = -4;

inline void enable_per_monitor_v2() {
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32) return;
    using SetContextFn = BOOL(WINAPI*)(HANDLE);
    auto set_context = reinterpret_cast<SetContextFn>(
        GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
    if (set_context) {
        set_context(reinterpret_cast<HANDLE>(kPerMonitorAwareV2));
        return;
    }
    SetProcessDPIAware();  // Win7/8：至少按系统 DPI 渲染
}

inline UINT dpi_for_window(HWND hwnd) {
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    using GetDpiFn = UINT(WINAPI*)(HWND);
    auto get_dpi = user32
                       ? reinterpret_cast<GetDpiFn>(GetProcAddress(user32, "GetDpiForWindow"))
                       : nullptr;
    const UINT dpi = get_dpi && hwnd ? get_dpi(hwnd) : 0;
    if (dpi) return dpi;
    HDC dc = GetDC(nullptr);
    const UINT fallback = dc ? static_cast<UINT>(GetDeviceCaps(dc, LOGPIXELSX)) : 0;
    if (dc) ReleaseDC(nullptr, dc);
    return fallback ? fallback : 96;
}

inline BOOL adjust_window_rect_ex_for_dpi(RECT* rect, DWORD style, BOOL menu, DWORD exstyle,
                                          UINT dpi) {
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    using AdjustFn = BOOL(WINAPI*)(LPRECT, DWORD, BOOL, DWORD, UINT);
    auto adjust = user32 ? reinterpret_cast<AdjustFn>(
                               GetProcAddress(user32, "AdjustWindowRectExForDpi"))
                         : nullptr;
    if (adjust) return adjust(rect, style, menu, exstyle, dpi);
    return AdjustWindowRectEx(rect, style, menu, exstyle);
}

inline int scale(int px, UINT dpi) {
    return MulDiv(px, static_cast<int>(dpi), 96);
}

}  // namespace windpi
