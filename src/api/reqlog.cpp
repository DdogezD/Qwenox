// reqlog.cpp — see reqlog.h.
#include "reqlog.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <system_error>

#include <nlohmann/json.hpp>

#include "miniz.h"

#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace reqlog {
namespace {

using json = nlohmann::ordered_json;

struct Config {
    bool on = false;
    int level = 6;
    std::filesystem::path dir;
};

Config make_env_config() {
    Config out;
    const char* e = std::getenv("QWENOX_REQLOG");
    out.on = e != nullptr && std::strcmp(e, "0") != 0;
    out.dir = "logs/reqlog";
    if (const char* d = std::getenv("QWENOX_REQLOG_DIR"); d != nullptr && *d != '\0')
        out.dir = d;
    if (const char* l = std::getenv("QWENOX_REQLOG_LEVEL")) {
        const int v = std::atoi(l);
        if (v >= 1 && v <= 10) out.level = v;
    }
    return out;
}

Config& mutable_config() {
    static Config c = make_env_config();
    return c;
}

const Config& config() { return mutable_config(); }

std::atomic<unsigned long long> g_seq{0};

bool safe_header(const std::string& name) {
    return name != "authorization" && name != "proxy-authorization" &&
           name != "cookie" && name != "set-cookie" && name != "x-api-key";
}

std::string path_kind(const std::string& path) {
    if (path == "/v1/chat/completions") return "chat";
    if (path == "/v1/completions") return "completions";
    if (path == "/v1/responses") return "responses";
    if (path == "/v1/messages") return "messages";
    if (path == "/v1/messages/count_tokens") return "count_tokens";
    if (path == "/v1/tokenize") return "tokenize";
    if (path == "/v1/detokenize") return "detokenize";
    return "other";
}

std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

int process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

bool write_atomic(const std::filesystem::path& dst, const void* data, size_t n) {
    std::filesystem::path tmp = dst;
    tmp += ".tmp";
    FILE* f = nullptr;
#ifdef _WIN32
    f = _wfopen(tmp.c_str(), L"wb");
#else
    f = std::fopen(tmp.c_str(), "wb");
#endif
    if (f == nullptr) return false;
    bool ok = n == 0 || std::fwrite(data, 1, n, f) == n;
    if (std::fflush(f) != 0) ok = false;
#ifdef _WIN32
    if (_commit(_fileno(f)) != 0) ok = false;
#else
    if (fsync(fileno(f)) != 0) ok = false;
#endif
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        return false;
    }
    return true;
}

bool make_zip(std::string* out, const Record& r, const std::string& meta) {
    static const char kEmpty = '\0';
    const void* request_data = r.request_body.empty() ? &kEmpty : r.request_body.data();
    const void* response_data = r.response_body.empty() ? &kEmpty : r.response_body.data();
    const mz_uint level = static_cast<mz_uint>(config().level);

    mz_zip_archive z{};
    if (!mz_zip_writer_init_heap(&z, 0, 0)) return false;
    bool ok = mz_zip_writer_add_mem(&z, "meta.json", meta.data(), meta.size(), level);
    ok = ok && mz_zip_writer_add_mem(&z, "request.body", request_data,
                                     r.request_body.size(), level);
    ok = ok && mz_zip_writer_add_mem(&z, "response.body", response_data,
                                     r.response_body.size(), level);
    void* buf = nullptr;
    size_t n = 0;
    if (ok) ok = mz_zip_writer_finalize_heap_archive(&z, &buf, &n);
    mz_zip_writer_end(&z);
    if (!ok) {
        if (buf != nullptr) mz_free(buf);
        return false;
    }
    out->assign(static_cast<const char*>(buf), n);
    mz_free(buf);
    return true;
}

std::string meta_json(const Record& r, const std::filesystem::path& file) {
    const auto now = std::chrono::system_clock::now();
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const std::tm tm = local_tm(seconds);
    char stamp[40];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);

    json j;
    j["format"] = 1;
    j["ts_ms"] = now_ms;
    j["time"] = std::string(stamp) + "." +
                std::to_string(static_cast<long long>(now_ms % 1000));
    j["file"] = file.u8string();
    j["method"] = r.method;
    j["path"] = r.path;
    j["query"] = r.query;
    j["remote"] = r.remote;
    j["version"] = r.version;
    j["status"] = r.status;
    j["sse"] = r.sse;
    j["delivered"] = r.delivered;
    j["duration_ms"] = r.duration_ms;
    j["request_bytes"] = r.request_body.size();
    j["response_bytes"] = r.response_body.size();
    j["headers"] = json::object();
    for (const auto& [name, value] : r.headers)
        if (safe_header(name)) j["headers"][name] = value;
    return j.dump();
}

}  // namespace

void configure(bool on, const std::string& dir, int level) {
    Config& c = mutable_config();
    c.on = on;
    if (!dir.empty()) c.dir = dir;
    c.level = (level >= 1 && level <= 10) ? level : 6;
}

bool enabled() { return config().on; }

std::string directory() { return config().dir.u8string(); }

int level() { return config().level; }

bool should_log(const std::string& method, const std::string& path) {
    return method == "POST" && path.size() >= 4 && path.compare(0, 4, "/v1/") == 0;
}

bool write(const Record& r) {
    if (!config().on || !should_log(r.method, r.path)) return true;

    const auto now = std::chrono::system_clock::now();
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const std::tm tm = local_tm(seconds);
    char date[16];
    char clock[16];
    std::strftime(date, sizeof date, "%Y%m%d", &tm);
    std::strftime(clock, sizeof clock, "%H%M%S", &tm);
    const int msec = static_cast<int>(now_ms % 1000);
    const unsigned long long seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    char name[160];
    std::snprintf(name, sizeof name, "%s-%03d-%d-%llu-%s.zip", clock, msec,
                  process_id(), seq, path_kind(r.path).c_str());

    const std::filesystem::path dir = config().dir / date;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "reqlog: create %s: %s\n", dir.u8string().c_str(),
                     ec.message().c_str());
        return false;
    }
    const std::filesystem::path file = dir / name;

    std::string zip;
    if (!make_zip(&zip, r, meta_json(r, file))) {
        std::fprintf(stderr, "reqlog: zip encode failed for %s\n",
                     file.u8string().c_str());
        return false;
    }
    if (!write_atomic(file, zip.data(), zip.size())) {
        std::fprintf(stderr, "reqlog: write %s failed\n", file.u8string().c_str());
        return false;
    }
    return true;
}

bool read_entry(const std::string& zip_path, const std::string& name,
                std::string* out, std::string* err) {
    mz_zip_archive z{};
    if (!mz_zip_reader_init_file(&z, zip_path.c_str(), 0)) {
        if (err != nullptr) *err = "cannot open zip: " + zip_path;
        return false;
    }
    size_t size = 0;
    void* data = mz_zip_reader_extract_file_to_heap(&z, name.c_str(), &size, 0);
    mz_zip_reader_end(&z);
    if (data == nullptr) {
        if (err != nullptr) *err = "missing entry " + name + " in " + zip_path;
        return false;
    }
    out->assign(static_cast<const char*>(data), size);
    mz_free(data);
    return true;
}

}  // namespace reqlog
