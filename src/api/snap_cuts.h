#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "chat_template.h"

// Semantic boundary hints (SNAPS): token positions where a future
// re-rendered prompt is likely to share a prefix with this one — the end of
// every earlier message and the end of the last message's content (just
// before its closing tag). The engine keeps cheap RAM checkpoints at these
// cuts, which is what makes edit-and-resend and multi-turn retokenization
// wobble hit the cache instead of re-prefilling. A cut is only sent when its
// character boundary is an exact token start in the final ids (`starts` =
// each id's byte offset in `text`), so every hint is a real token boundary of
// this prompt. Cuts stop at the first image: a prefix without image tokens is
// the same KV for text and vision requests, one past it is not.
//
// Only the last 8 cuts are used, so prefixes render from the tail backwards
// and stop once 8 cuts are confirmed (~9 renders instead of n-1; T2b measured
// 184 ms per request at 401 messages with the forward version).
std::vector<long long> compute_snap_cuts(const chat_template::json& messages,
                                         chat_template::Options opts,
                                         const std::string& text,
                                         const std::vector<int>& ids,
                                         const std::vector<size_t>& starts);

// The pre-2026-10-17 forward all-prefixes implementation, kept as the oracle
// for snap_cuts_test's output-equality gate. The server never calls it.
std::vector<long long> compute_snap_cuts_ref(const chat_template::json& messages,
                                             chat_template::Options opts,
                                             const std::string& text,
                                             const std::vector<int>& ids,
                                             const std::vector<size_t>& starts);
