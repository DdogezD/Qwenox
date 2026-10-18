// reqlog: one ZIP per API request, written after the response is complete.
//
// Opt-in via REQLOG=1 in service.conf, or QWENOX_REQLOG=1 in the environment
// (the latter wins, for standalone/test runs). Default off. REQLOG is
// deliberately NOT a managed service.conf key: the settings panel and web UI
// neither show nor accept it, so it can only be enabled by hand-editing the
// file. main() reads the effective values and calls configure() once.
//
// Only POST /v1/* requests are logged. Each archive holds:
//   meta.json       request metadata (method, path, headers, status, timing)
//   request.body    the raw request body, byte for byte
//   response.body   the raw response body, or the full SSE byte stream
//
// Layout: <REQLOG_DIR>/YYYYMMDD/HHMMSS-mmm-<pid>-<seq>-<kind>.zip,
// default dir logs/reqlog. Files are written to .tmp, fsynced and renamed.
// REQLOG_LEVEL selects the deflate level (1..10, default 6).
//
// Writing never throws and never fails a request: errors go to stderr only.
#pragma once

#include <map>
#include <string>
#include <string_view>

namespace reqlog {

struct Record {
    std::string method;
    std::string path;
    std::string query;
    std::string remote;
    std::string version;
    std::map<std::string, std::string> headers;

    int status = 0;
    bool sse = false;
    bool delivered = true;  // false: the client connection died mid-response
    double duration_ms = 0.0;

    std::string_view request_body;
    std::string_view response_body;
};

// Applies the effective service.conf/env values. Call once at startup, before
// enabled()/write(); the values are then read-only in practice.
void configure(bool on, const std::string& dir, int level);

// Reads QWENOX_REQLOG* when configure() was never called. Safe from any thread.
bool enabled();
std::string directory();  // REQLOG_DIR, default "logs/reqlog"
int level();              // REQLOG_LEVEL, default 6

// True for the requests this module records (POST /v1/*).
bool should_log(const std::string& method, const std::string& path);

// Writes one record. Returns false (and logs to stderr) on I/O or codec error.
bool write(const Record& r);

// Test helper: extract one entry from a ZIP written by write().
bool read_entry(const std::string& zip_path, const std::string& name,
                std::string* out, std::string* err);

}  // namespace reqlog
