#include "snap_cuts.h"

#include <algorithm>
#include <unordered_set>

using chat_template::json;

namespace {

// Same added-token ids as main.cpp: <|image_pad|> / <|video_pad|>.
constexpr int kImagePadId = 248056, kVideoPadId = 248057;

bool is_pad_id(int id) { return id == kImagePadId || id == kVideoPadId; }

size_t text_token_count(const std::vector<int>& ids) {
    return static_cast<size_t>(std::find_if(ids.begin(), ids.end(), is_pad_id) -
                               ids.begin());
}

// The token index whose byte offset is exactly cc, or -1. starts is ascending
// inside the text-only prefix [0, n_text) (expanded image pads beyond it share
// their span's offset).
long long token_at_char(const std::vector<size_t>& starts, size_t n_text, size_t cc) {
    const auto it =
        std::lower_bound(starts.begin(), starts.begin() + (ptrdiff_t)n_text, cc);
    const size_t ti = static_cast<size_t>(it - starts.begin());
    if (ti < n_text && ti > 0 && *it == cc) return static_cast<long long>(ti);
    return -1;
}

// A prefix render is a stable boundary only when it is an exact char prefix
// of the full render (tool-role rendering depends on the NEXT message, so not
// every message end qualifies).
bool exact_prefix(const chat_template::RenderResult& pr, const std::string& text) {
    return pr.ok && pr.text.size() < text.size() &&
           text.compare(0, pr.text.size(), pr.text) == 0;
}

// End of the last message's content = just before its closing <|im_end|>.
void tail_cut_char(const chat_template::json& messages, const chat_template::Options& opts,
                   const std::string& text, std::vector<size_t>* cut_chars) {
    chat_template::RenderResult ng = chat_template::render_chat_template(&messages, opts);
    if (ng.ok && ng.text.size() <= text.size() &&
        text.compare(0, ng.text.size(), ng.text) == 0) {
        const size_t tail = ng.text.rfind("<|im_end|>");
        if (tail != std::string::npos) cut_chars->push_back(tail);
    }
}

}  // namespace

std::vector<long long> compute_snap_cuts(const chat_template::json& messages,
                                         chat_template::Options opts,
                                         const std::string& text,
                                         const std::vector<int>& ids,
                                         const std::vector<size_t>& starts) {
    const size_t n_text = text_token_count(ids);
    opts.add_generation_prompt = nullptr;  // prefix renders stop at message ends
    std::vector<long long> cuts;
    std::unordered_set<size_t> seen;
    const auto consider = [&](size_t cc) {
        if (!seen.insert(cc).second) return;
        const long long ti = token_at_char(starts, n_text, cc);
        if (ti >= 0) cuts.push_back(ti);
    };
    {
        std::vector<size_t> tail;
        tail_cut_char(messages, opts, text, &tail);
        for (size_t cc : tail) consider(cc);
    }
    // The kept cuts are the 8 largest token boundaries; walking message ends
    // from the tail reaches them after ~8 renders instead of n-1.
    for (size_t k = messages.size(); k-- > 1 && cuts.size() < 8;) {
        json prefix = json::array();
        for (size_t i = 0; i < k; i++) prefix.push_back(messages[i]);
        const chat_template::RenderResult pr =
            chat_template::render_chat_template(&prefix, opts);
        if (exact_prefix(pr, text)) consider(pr.text.size());
    }
    std::sort(cuts.begin(), cuts.end());
    return cuts;
}

std::vector<long long> compute_snap_cuts_ref(const chat_template::json& messages,
                                             chat_template::Options opts,
                                             const std::string& text,
                                             const std::vector<int>& ids,
                                             const std::vector<size_t>& starts) {
    std::vector<size_t> cut_chars;
    opts.add_generation_prompt = nullptr;  // prefix renders stop at message ends
    for (size_t k = 1; k < messages.size(); k++) {
        json prefix = json::array();
        for (size_t i = 0; i < k; i++) prefix.push_back(messages[i]);
        chat_template::RenderResult pr =
            chat_template::render_chat_template(&prefix, opts);
        if (exact_prefix(pr, text)) cut_chars.push_back(pr.text.size());
    }
    tail_cut_char(messages, opts, text, &cut_chars);
    std::sort(cut_chars.begin(), cut_chars.end());
    cut_chars.erase(std::unique(cut_chars.begin(), cut_chars.end()),
                    cut_chars.end());
    std::vector<long long> cuts;
    if (cut_chars.empty()) return cuts;
    const size_t n_text = text_token_count(ids);
    size_t ti = 0;
    for (size_t cc : cut_chars) {
        while (ti < n_text && starts[ti] < cc) ti++;
        if (ti < n_text && starts[ti] == cc && ti > 0) cuts.push_back((long long)ti);
    }
    if (cuts.size() > 8) cuts.erase(cuts.begin(), cuts.end() - 8);
    return cuts;
}
