// version.h — 版本号字符串。唯一来源是仓库根目录的 VERSION 文件：
// build.sh / build_win.sh 读出后以 -DQWENOX_VERSION_RAW=0.0.8 -DQWENOX_GIT_RAW=<短哈希>
// 传入（不带引号：hipcc.exe 在 Windows 上转发参数会吃掉引号），这里再字符串化。
// 不经编译脚本直接编译时显示 "dev"。
#pragma once

#define QWENOX_STR2_(x) #x
#define QWENOX_STR_(x) QWENOX_STR2_(x)

#ifdef QWENOX_VERSION_RAW
#define QWENOX_VERSION QWENOX_STR_(QWENOX_VERSION_RAW)
#else
#define QWENOX_VERSION "dev"
#endif

#ifdef QWENOX_GIT_RAW
#define QWENOX_GIT QWENOX_STR_(QWENOX_GIT_RAW)
#else
#define QWENOX_GIT "unknown"
#endif

// 形如 "0.0.8 (2754824)"，用于日志与界面
#define QWENOX_VERSION_FULL QWENOX_VERSION " (" QWENOX_GIT ")"
