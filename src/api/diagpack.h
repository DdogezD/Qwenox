// diagpack.h — 诊断包：把定位崩溃 / 卡死需要的材料打成一个 zip，用户直接发给开发者。
//
// 输出 <root>/logs/diag/qwenox-diag-<YYYYmmdd-HHMMSS>.zip（只保留最近 10 个），包含：
//   README.txt        包里有什么、隐私说明（中英）
//   summary.txt       打包原因、退出码含义、崩溃报告摘要、文件清单
//   engine.log        引擎日志（开头 512 KB + 结尾 8 MB）
//   api.log           API 日志（开头 256 KB + 结尾 4 MB）
//   crash/…           logs/crash 里本次引擎 pid 的报告与 .dmp，以及时间窗内的其它报告
//   engine-live.txt   引擎还在运行时：DIAG 动词取回的飞行记录 / 最近 HIP 调用
//   engine-live.dmp   Windows 上引擎由本进程拉起且在运行时：在线 minidump（线程栈）
//   system.txt        系统 / 驱动 / 电源 / TDR / 环境变量（API 进程视角）
//   events.txt        Windows 事件日志（系统：显卡驱动、电源、意外关机、低内存…；
//                     应用：程序崩溃 / 无响应 / WER），默认最近 72 小时
//   kernel.txt        Linux：journalctl -k 中与 GPU / 内存相关的行
//   service.conf      配置（KEY/TOKEN/SECRET 类的值打码）
//   logs-index.txt    logs/ 目录清单
// 不包含：请求/回复正文、模型文件、KV 缓存。minidump 里的线程栈内存可能带有少量
// 正在处理的文本片段（README 里写明）。
#pragma once

#include <cstdint>
#include <string>

namespace diagpack {

struct Opts {
    std::string root = ".";      // 项目根（logs/ 所在目录）
    std::string reason;          // 打包原因，写进 summary.txt
    std::string engine_log;      // 空 = 自动找 logs/ 下最新的引擎日志
    std::string api_log;         // 空 = 自动找最新的 API 日志
    std::string conf_path;       // service.conf（空 = <root>/service.conf）
    int engine_pid = 0;          // 崩溃报告按 pid 匹配（0 = 只按时间窗）
    bool has_exit = false;       // 引擎退出码已知
    unsigned long exit_code = 0;
    int64_t since_unix = 0;      // 崩溃报告 / 事件日志时间窗起点（0 = 72 小时前）
    std::string engine_host;     // 非空且 engine_port>0：在线取引擎 DIAG 快照
    int engine_port = 0;
    std::string live_dump;       // 调用方已写好的引擎在线 minidump：打进包后删除
};

// 生成诊断包，返回 zip 路径；失败返回空串并写 err。不抛异常。
std::string build(const Opts& o, std::string* err);

// 退出码的含义（Windows NTSTATUS / Linux 128+信号 / 引擎约定）
std::string exit_code_text(unsigned long code);

// <root>/logs/diag
std::string out_dir(const std::string& root);

// 名字是否像本模块生成的包（HTTP 下载接口的白名单校验）
bool valid_name(const std::string& name);

}  // namespace diagpack
