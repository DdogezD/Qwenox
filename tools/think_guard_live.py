#!/usr/bin/env python3
"""Think-guard check against a running API on the real model.

The request field think_guard_test_cut makes the first run stop after N
tokens, inside the reasoning, as if the model had ended its turn there; the
API must close the think block and resume to a real answer. Covers
/v1/chat/completions (non-stream sampled, stream greedy), /v1/responses and
/v1/messages (stream, with a tool), plus one ordinary request that the guard
must leave alone. About 1-3 minutes. Ends with RESULT PASS or RESULT FAIL.

    python tools/think_guard_live.py [--base http://127.0.0.1:8731]

The API must be built from a tree that has the think guard (rebuild and
restart it first); an old API ignores the test field and the first check
fails with a note saying so.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SENTENCE = ("Considering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.")
CUT = 16


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def sse_frames(raw):
    for frame in raw.decode("utf-8").split("\n\n"):
        data = [line[6:] for line in frame.split("\n") if line.startswith("data: ")]
        if data and data[0] != "[DONE]":
            yield json.loads(data[0])


def chat(base, body, stream):
    body = dict(body, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    status, raw = post(base, "/v1/chat/completions", body)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
    if not stream:
        j = json.loads(raw)
        msg = j["choices"][0]["message"]
        return {"reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
                "calls": [c["function"]["name"] for c in msg.get("tool_calls") or []],
                "finish": j["choices"][0]["finish_reason"], "usage": j["usage"]}
    out = {"reasoning": "", "content": "", "calls": [], "finish": None, "usage": {}}
    for f in sse_frames(raw):
        if f.get("usage"):
            out["usage"] = f["usage"]
        for item in f.get("choices") or []:
            out["finish"] = item.get("finish_reason") or out["finish"]
            d = item.get("delta", {})
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            out["calls"] += [tc["function"]["name"] for tc in d.get("tool_calls", [])
                             if tc.get("function", {}).get("name")]
    return out


def responses(base, body):
    status, raw = post(base, "/v1/responses", dict(body, stream=False))
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
    r = json.loads(raw)
    text = "".join(p["text"] for it in r["output"] if it["type"] == "message" for p in it["content"])
    calls = [it["name"] for it in r["output"] if it["type"] == "function_call"]
    return {"reasoning": None, "content": text, "calls": calls, "finish": r["status"],
            "usage": r["usage"]}


def messages(base, body):
    status, raw = post(base, "/v1/messages", dict(body, stream=True))
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
    out = {"reasoning": "", "content": "", "calls": [], "finish": None, "usage": {}}
    for e in sse_frames(raw):
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
            out["calls"].append(e["content_block"]["name"])
        elif e["type"] == "content_block_delta":
            d = e["delta"]
            out["reasoning"] += d.get("thinking", "")
            out["content"] += d.get("text", "")
        elif e["type"] == "message_delta":
            out["finish"], out["usage"] = e["delta"]["stop_reason"], e["usage"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8731")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as resp:
            json.loads(resp.read())
    except Exception as error:  # noqa: BLE001
        print(f"API not reachable at {base}: {error}")
        print("RESULT FAIL")
        return 1

    question = "In two short sentences: why is the sky blue?"
    weather_tool = {"name": "get_weather", "description": "Current weather for a city",
                    "input_schema": {"type": "object", "properties": {
                        "city": {"type": "string"}}, "required": ["city"]}}
    cases = [
        ("chat non-stream, sampled", lambda: chat(base, {
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 4096, "think_guard_test_cut": CUT}, False), True),
        ("chat stream, greedy", lambda: chat(base, {
            "messages": [{"role": "user", "content": question}], "temperature": 0,
            "max_tokens": 4096, "think_guard_test_cut": CUT}, True), True),
        ("responses non-stream", lambda: responses(base, {
            "input": question, "max_output_tokens": 4096, "think_guard_test_cut": CUT}), True),
        ("messages stream + tool", lambda: messages(base, {
            "messages": [{"role": "user", "content":
                          "What is the weather in Beijing right now? Use the tool."}],
            "tools": [weather_tool], "max_tokens": 4096, "think_guard_test_cut": CUT}), True),
        ("ordinary request (guard must stay out)", lambda: chat(base, {
            "messages": [{"role": "user", "content": question}], "temperature": 0,
            "max_tokens": 4096}, False), False),
    ]
    failures = 0
    for name, run, expect_guard in cases:
        t0 = time.monotonic()
        try:
            r = run()
        except Exception as error:  # noqa: BLE001
            print(f"FAIL {name}: {error}")
            failures += 1
            continue
        dt = time.monotonic() - t0
        guard = (r["usage"] or {}).get("think_guard")
        answered = bool(r["content"].strip()) or bool(r["calls"])
        problems = []
        if expect_guard:
            if guard is None:
                problems.append("usage.think_guard missing (API not rebuilt with the think "
                                "guard, or the model closed its reasoning within "
                                f"{CUT} tokens)")
            elif guard != "resumed_unclosed":
                problems.append(f"think_guard={guard}, expected resumed_unclosed")
            if r["reasoning"] is not None and not r["reasoning"].rstrip().endswith(SENTENCE):
                problems.append("reasoning does not end with the closing sentence")
        elif guard is not None:
            problems.append(f"think_guard={guard} on an ordinary request")
        if not answered:
            problems.append("no answer and no tool call")
        if "</think>" in r["content"]:
            problems.append("</think> leaked into the answer")
        status = "FAIL" if problems else "PASS"
        failures += bool(problems)
        shown = r["content"].strip().replace("\n", " ")[:120] or f"(tool calls: {r['calls']})"
        print(f"{status} {name}  [{dt:.1f}s, finish={r['finish']}, guard={guard}]")
        print(f"     answer: {shown}")
        for p in problems:
            print(f"     -> {p}")
    print("RESULT PASS" if failures == 0 else f"RESULT FAIL ({failures} failed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
