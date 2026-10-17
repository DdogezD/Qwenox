// A4 gate (HANDOFF-AGENT-SHORT-PREFILL §A4): the backward early-stop
// compute_snap_cuts must return vectors identical to compute_snap_cuts_ref
// (the forward all-prefixes implementation) on the api_regression fixtures
// and on random synthetic conversations (consecutive tool messages, images,
// reasoning).
//
// Usage: snap_cuts_test [tokenizer-dir] [rounds]   defaults: models/tokenizer 2000
#include "snap_cuts.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "tokenizer.h"
#include "vision.h"

using chat_template::json;

namespace {

qwenox::Tokenizer g_tok;
int g_failures = 0;
int g_skipped = 0;
long long g_cases = 0;

void fail(const std::string& name, const std::vector<long long>& got,
          const std::vector<long long>& want) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: got [", name.c_str());
    for (long long v : got) std::fprintf(stderr, "%lld ", v);
    std::fprintf(stderr, "] want [");
    for (long long v : want) std::fprintf(stderr, "%lld ", v);
    std::fprintf(stderr, "]\n");
}

// Encodes like main.cpp's encode_spliced (no tcache): text path excludes the
// pad added-ids; vision path expands pad spans via the frames' grids.
bool encode(const std::string& text, const std::vector<vision::Frame>& frames,
            std::vector<int>* ids, std::vector<size_t>* starts) {
    ids->clear();
    starts->clear();
    if (!frames.empty()) {
        std::string err;
        if (!vision::encode_prompt(text, g_tok, frames, ids, &err, starts)) {
            ++g_skipped;
            return false;
        }
        return true;
    }
    static const std::unordered_set<int> kNoPads = {248056, 248057};
    for (const auto& s : g_tok.encode_with_offsets(text, false, &kNoPads)) {
        ids->push_back(s.id);
        starts->push_back(s.start);
    }
    return true;
}

chat_template::Options base_opts() {
    static const json kTrue = true;
    chat_template::Options opts;
    opts.add_generation_prompt = &kTrue;
    return opts;
}

void run_case(const char* name, const json& messages, chat_template::Options opts,
              const std::vector<vision::Frame>& frames = {}) {
    chat_template::RenderResult rr = chat_template::render_chat_template(&messages, opts);
    if (!rr.ok) {
        ++g_skipped;
        return;
    }
    std::vector<int> ids;
    std::vector<size_t> starts;
    if (!encode(rr.text, frames, &ids, &starts)) return;
    ++g_cases;
    const std::vector<long long> got = compute_snap_cuts(messages, opts, rr.text, ids, starts);
    const std::vector<long long> want =
        compute_snap_cuts_ref(messages, opts, rr.text, ids, starts);
    if (got != want) fail(name, got, want);
}

json msg(const char* role, const std::string& content) {
    return json{{"role", role}, {"content", content}};
}

json tool_call_msg(const std::string& name, const std::string& args) {
    return json{{"role", "assistant"},
                {"content", ""},
                {"tool_calls", json::array({json{{"id", "call_" + name},
                                                 {"type", "function"},
                                                 {"function", {{"name", name},
                                                               {"arguments", args}}}}})}};
}

json image_msg(const std::string& text) {
    return json{{"role", "user"},
                {"content", json::array({json{{"type", "image_url"},
                                              {"image_url", {{"url", "data:image/png;base64,x"}}}},
                                         json{{"type", "text"}, {"text", text}}})}};
}

vision::Frame small_frame() {
    vision::Frame f;
    f.grid = {1, 8, 8};  // 16 pad tokens
    return f;
}

const json& tools_json() {
    static const json t = [] {
        json fn;
        fn["name"] = "get_weather";
        fn["description"] = "Get weather";
        fn["parameters"] = {
            {"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}},
        };
        return json::array({json{{"type", "function"}, {"function", fn}}});
    }();
    return t;
}

void fixtures() {
    // api_regression_test.py bodies (post-normalization equivalents).
    run_case("regr-hi", json::array({msg("user", "hi")}), base_opts());
    {
        chat_template::Options opts = base_opts();
        static const json kFalse = false;
        opts.enable_thinking = &kFalse;
        run_case("regr-hi-nothink", json::array({msg("user", "hi")}), opts);
        run_case("regr-vision", json::array({image_msg("describe")}), opts,
                 {small_frame()});
    }
    // Tool chains, incl. consecutive tool messages (rendering depends on the
    // next message — the case the prefix check exists for).
    run_case("tool-chain",
             json::array({msg("system", "You are a helpful assistant."),
                          msg("user", "weather in bj and sh?"),
                          tool_call_msg("get_weather", "{\"city\": \"bj\"}"),
                          msg("tool", "{\"temp\": 20}"),
                          tool_call_msg("get_weather", "{\"city\": \"sh\"}"),
                          msg("tool", "{\"temp\": 23}"),
                          msg("assistant", "bj 20, sh 23"),
                          msg("user", "thanks")}),
             base_opts());
    {
        chat_template::Options opts = base_opts();
        opts.tools = &tools_json();
        run_case("tool-chain-tools",
                 json::array({msg("user", "check three cities"),
                              tool_call_msg("get_weather", "{\"city\": \"bj\"}"),
                              tool_call_msg("get_weather", "{\"city\": \"sh\"}"),
                              msg("tool", "{\"temp\": 20}"),
                              msg("tool", "{\"temp\": 23}"),
                              msg("tool", "error: timeout"),
                              msg("assistant", "two ok, one failed")}),
                 opts);
        static const json kJson = "json";
        opts.tool_call_format = &kJson;
        run_case("tool-chain-jsonfmt",
                 json::array({msg("user", "again"),
                              tool_call_msg("get_weather", "{\"city\": \"gz\"}"),
                              msg("tool", "{\"temp\": 26}")}),
                 opts);
    }
    // Reasoning / thinking variants.
    {
        chat_template::Options opts = base_opts();
        static const json kTrue = true;
        opts.preserve_thinking = &kTrue;
        run_case("reasoning",
                 json::array({msg("user", "think"),
                              json{{"role", "assistant"},
                                   {"content", "answer"},
                                   {"reasoning_content", "let me think step by step"}},
                              msg("user", "and?")}),
                 opts);
    }
    // Adversarial contents: template markers typed as text, empty content,
    // image mid-conversation (cuts must stop at the first pad id).
    run_case("marker-text",
             json::array({msg("user", "what does <|im_end|> mean?"),
                          msg("assistant", "it ends a turn; </think> ends thinking"),
                          msg("user", "and <tool_call>?"),
                          msg("assistant", "")}),
             base_opts());
    run_case("image-mid",
             json::array({msg("system", "sys"),
                          msg("user", "look"),
                          msg("assistant", "ok"),
                          image_msg("what is this"),
                          msg("assistant", "a dot"),
                          msg("user", "thanks")}),
             base_opts(), {small_frame()});
    // Long conversations at the bench sizes.
    for (int n : {9, 50, 200, 400}) {
        json msgs = json::array();
        for (int i = 0; i < n; i++) {
            if (i % 4 == 2)
                msgs.push_back(tool_call_msg("get_weather",
                                             "{\"city\": \"c" + std::to_string(i) + "\"}"));
            else if (i % 4 == 3)
                msgs.push_back(msg("tool", "{\"temp\": " + std::to_string(i) + "}"));
            else if (i % 4 == 1)
                msgs.push_back(msg("assistant", "reply " + std::to_string(i)));
            else
                msgs.push_back(msg("user", "question " + std::to_string(i)));
        }
        const std::string name = "long-" + std::to_string(n);
        run_case(name.c_str(), msgs, base_opts());
    }
}

// ------------------------------------------------------------------ random --

std::mt19937_64 g_rng(20261017);

std::string rand_text(size_t max_len) {
    static const char* pieces[] = {
        "hello ", "world", "12345 ", "\n", "。", "你好，", "🙂", "<|im_end|>",
        "</think>", "<tool_call>", "{\"k\": 1}", "a", "",  "lorem ipsum ",
        "```cpp\nint main(){}\n```", "|", "\t", "\"quoted\""};
    std::string out;
    const size_t target = g_rng() % (max_len + 1);
    while (out.size() < target) out += pieces[g_rng() % (sizeof(pieces) / sizeof(*pieces))];
    return out;
}

json rand_conversation(std::vector<vision::Frame>* frames) {
    frames->clear();
    const int roll = static_cast<int>(g_rng() % 100);
    const int n = roll < 50 ? 1 + (int)(g_rng() % 8)
                            : roll < 85 ? 9 + (int)(g_rng() % 52)
                                        : 61 + (int)(g_rng() % 390);
    json msgs = json::array();
    if (g_rng() % 3 == 0) msgs.push_back(msg("system", rand_text(80)));
    const char* roles[] = {"user", "assistant"};
    int turn = 0;
    while ((int)msgs.size() < n) {
        const int kind = static_cast<int>(g_rng() % 10);
        if (kind <= 4) {
            json m = msg(roles[turn++ % 2], rand_text(300));
            if (frames->size() < 2 && g_rng() % 16 == 0) {
                frames->push_back(small_frame());
                m = image_msg(rand_text(60));
            }
            msgs.push_back(std::move(m));
        } else if (kind <= 6) {
            // assistant tool_calls followed by 1-3 consecutive tool messages
            msgs.push_back(tool_call_msg("fn" + std::to_string(g_rng() % 100),
                                         rand_text(40)));
            const int tools = 1 + (int)(g_rng() % 3);
            for (int i = 0; i < tools && (int)msgs.size() < n; i++)
                msgs.push_back(msg("tool", rand_text(200)));
        } else if (kind == 7) {
            msgs.push_back(json{{"role", "assistant"},
                                {"content", rand_text(120)},
                                {"reasoning_content", rand_text(200)}});
        } else {
            msgs.push_back(msg(roles[turn++ % 2], ""));  // empty content
        }
    }
    return msgs;
}

void random_cases(int rounds) {
    static const json kTrue = true;
    static const json kFalse = false;
    static const json kJson = "json";
    for (int i = 0; i < rounds; i++) {
        std::vector<vision::Frame> frames;
        const json msgs = rand_conversation(&frames);
        chat_template::Options opts = base_opts();
        if (g_rng() % 2) opts.enable_thinking = (g_rng() % 2) ? &kTrue : &kFalse;
        if (g_rng() % 3 == 0) opts.tools = &tools_json();
        if (g_rng() % 4 == 0) opts.tool_call_format = &kJson;
        if (g_rng() % 4 == 0) opts.preserve_thinking = &kTrue;
        const std::string name = "random-" + std::to_string(i);
        run_case(name.c_str(), msgs, opts, frames);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "models/tokenizer";
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 2000;
    std::string err;
    if (!g_tok.load(dir, &err)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", err.c_str());
        return 1;
    }
    fixtures();
    random_cases(rounds);
    std::printf("cases=%lld skipped=%d failures=%d\n", g_cases, g_skipped, g_failures);
    if (g_failures) return 1;
    std::printf("RESULT PASS\n");
    return 0;
}
