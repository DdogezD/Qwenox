#!/usr/bin/env python3
"""Smoke test for the Anthropic Messages endpoints (/v1/messages, /count_tokens).

Runs against a live API (engine loaded). Standard library only; if the
official `anthropic` SDK is importable it is also used to parse a streamed and
a non-streamed reply, which is the strongest client-compatibility check.

    python tools/anthropic_messages_smoke.py [--base http://127.0.0.1:8731]

Prints one line per check and ends with RESULT PASS / RESULT FAIL.
Requests are sequential (no concurrency).
"""
import argparse
import base64
import http.client
import json
import struct
import sys
import time
import urllib.parse
import zlib

BASE = "http://127.0.0.1:8731"
results = []  # (name, ok, detail)
NO_THINK = {"type": "disabled"}


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  -- " + str(detail)[:600]),
          flush=True)
    return ok


def conn():
    u = urllib.parse.urlparse(BASE)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)


def post(path, body):
    c = conn()
    c.request("POST", path, json.dumps(body),
              {"Content-Type": "application/json", "x-api-key": "dummy",
               "anthropic-version": "2023-06-01"})
    r = c.getresponse()
    raw = r.read().decode("utf-8", "replace")
    try:
        data = json.loads(raw)
    except Exception:
        data = {"_raw": raw}
    return r.status, data


def stream(body):
    """Returns (status, [(event, data)], content_type)."""
    body = dict(body, stream=True)
    c = conn()
    c.request("POST", "/v1/messages", json.dumps(body),
              {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    r = c.getresponse()
    ctype = r.getheader("Content-Type", "")
    if r.status != 200:
        return r.status, [("http_error", r.read().decode("utf-8", "replace"))], ctype
    events, ev, data_lines = [], None, []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if ev is not None or data_lines:
                payload = "\n".join(data_lines)
                try:
                    payload = json.loads(payload)
                except Exception:
                    pass
                events.append((ev, payload))
            ev, data_lines = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    return r.status, events, ctype


def validate_stream(events):
    """Checks the Anthropic event grammar; returns (ok, why, assembled message)."""
    names = [e for e, _ in events]
    if not names or names[0] != "message_start":
        return False, "first event is %r" % (names[:1],), None
    if names[-1] != "message_stop":
        return False, "last event is %r" % (names[-1:],), None
    msg = None
    blocks = {}
    open_idx = None
    next_idx = 0
    saw_delta = False
    for ev, d in events:
        if not isinstance(d, dict) or d.get("type") != ev:
            return False, "event %s data.type mismatch: %r" % (ev, d), None
        if ev == "message_start":
            msg = d["message"]
            if msg.get("type") != "message" or msg.get("role") != "assistant":
                return False, "bad message_start %r" % msg, None
            if msg.get("content") != []:
                return False, "message_start content not empty", None
        elif ev == "content_block_start":
            if open_idx is not None:
                return False, "block %s started while %s open" % (d["index"], open_idx), None
            if d["index"] != next_idx:
                return False, "block index %s, expected %s" % (d["index"], next_idx), None
            open_idx = d["index"]
            next_idx += 1
            blocks[open_idx] = dict(d["content_block"])
            blocks[open_idx]["_json"] = ""
        elif ev == "content_block_delta":
            if d["index"] != open_idx:
                return False, "delta for closed block %s" % d["index"], None
            b, dt = blocks[open_idx], d["delta"]
            t = dt.get("type")
            if t == "text_delta" and b["type"] == "text":
                b["text"] += dt["text"]
            elif t == "thinking_delta" and b["type"] == "thinking":
                b["thinking"] += dt["thinking"]
            elif t == "input_json_delta" and b["type"] == "tool_use":
                b["_json"] += dt["partial_json"]
            else:
                return False, "delta %s inside %s block" % (t, b["type"]), None
        elif ev == "content_block_stop":
            if d["index"] != open_idx:
                return False, "stop for block %s, open %s" % (d["index"], open_idx), None
            b = blocks[open_idx]
            if b["type"] == "tool_use":
                try:
                    b["input"] = json.loads(b["_json"]) if b["_json"] else {}
                except Exception as e:
                    return False, "tool input JSON invalid: %r (%s)" % (b["_json"], e), None
            open_idx = None
        elif ev == "message_delta":
            if open_idx is not None:
                return False, "message_delta while block open", None
            saw_delta = True
            msg["stop_reason"] = d["delta"]["stop_reason"]
            msg["stop_sequence"] = d["delta"].get("stop_sequence")
            msg["usage"] = d["usage"]
        elif ev in ("ping", "message_stop"):
            pass
        else:
            return False, "unexpected event %s" % ev, None
    if not saw_delta:
        return False, "no message_delta", None
    content = []
    for i in range(next_idx):
        b = blocks[i]
        b.pop("_json", None)
        content.append(b)
    msg["content"] = content
    return True, "", msg


def blocks_of(msg, t):
    return [b for b in msg.get("content", []) if b.get("type") == t]


def text_of(msg):
    return "".join(b["text"] for b in blocks_of(msg, "text"))


def tiny_png(rgb, size=64):
    row = b"\x00" + bytes(rgb) * size
    raw = row * size

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--no-sdk", action="store_true")
    BASE = ap.parse_args().base.rstrip("/")
    sdk = ap.parse_args().no_sdk is False

    # wait for the API + engine
    t0 = time.time()
    while True:
        try:
            c = conn()
            c.request("GET", "/health")
            r = c.getresponse()
            h = json.loads(r.read())
            if r.status == 200 and h.get("tokenizer"):
                break
        except Exception:
            pass
        if time.time() - t0 > 1800:
            print("RESULT FAIL (API not reachable at %s)" % BASE)
            return 1
        time.sleep(5)
    check("health lists /v1/messages", "/v1/messages" in h.get("endpoints", []), h.get("endpoints"))

    # 1. count_tokens
    st, d = post("/v1/messages/count_tokens",
                 {"model": "x", "system": "You are terse.",
                  "messages": [{"role": "user", "content": "Hello"}]})
    check("count_tokens", st == 200 and isinstance(d.get("input_tokens"), int)
          and d["input_tokens"] > 3, (st, d))
    st2, d2 = post("/v1/messages/count_tokens",
                   {"model": "x", "system": "You are terse.", "tools": [WEATHER_TOOL],
                    "messages": [{"role": "user", "content": "Hello"}]})
    check("count_tokens grows with tools", st2 == 200 and d2.get("input_tokens", 0) >
          d.get("input_tokens", 0), (d, d2))

    # 2. non-stream basic, greedy
    basic = {"model": "x", "max_tokens": 64, "temperature": 0, "thinking": NO_THINK,
             "system": [{"type": "text", "text": "Answer with one short sentence."}],
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "What is the capital of France?"}]}]}
    st, m = post("/v1/messages", basic)
    ok = (st == 200 and m.get("type") == "message" and m.get("role") == "assistant"
          and str(m.get("id", "")).startswith("msg_") and m.get("stop_reason") == "end_turn"
          and "paris" in text_of(m).lower() and not blocks_of(m, "thinking"))
    check("non-stream basic", ok, (st, m))
    u = m.get("usage", {})
    check("usage fields", all(isinstance(u.get(k), int) for k in
                              ("input_tokens", "output_tokens", "cache_read_input_tokens",
                               "cache_creation_input_tokens")) and u.get("output_tokens", 0) > 0, u)

    # 3. stream == non-stream (greedy)
    st, ev, ctype = stream(basic)
    ok, why, sm = validate_stream(ev) if st == 200 else (False, ev, None)
    check("stream grammar (basic)", ok and "text/event-stream" in ctype, why or ctype)
    if sm:
        check("stream text == non-stream text", text_of(sm) == text_of(m),
              (text_of(sm), text_of(m)))
        check("stream stop_reason end_turn", sm.get("stop_reason") == "end_turn", sm.get("stop_reason"))
    # Multi-turn prefix reuse, shaped like an agent client: a long system
    # prompt, then a follow-up turn. The template drops the empty think block
    # of earlier assistant turns, so the follow-up diverges from the live KV
    # right after the previous user message and is served from the RAM
    # checkpoint at that boundary -- which needs >= 4096 tokens
    # (QWENOX_RCKPT_MIN), hence the long system prompt.
    long_sys = "You are a geography assistant. Answer in one short sentence.\n" + "\n".join(
        "Reference note %d: entry %d of the style guide says to keep answers short, "
        "factual and polite, and to avoid lists unless asked." % (i, i) for i in range(260))
    turn1 = {"model": "x", "max_tokens": 64, "temperature": 0, "thinking": NO_THINK,
             "system": long_sys,
             "messages": [{"role": "user", "content": "What is the capital of France?"}]}
    st, ct = post("/v1/messages/count_tokens", turn1)
    st, m1 = post("/v1/messages", turn1)
    if st == 200:
        turn2 = dict(turn1, messages=turn1["messages"] + [
            {"role": "assistant", "content": m1["content"]},
            {"role": "user", "content": "And of Italy?"}])
        st, m2 = post("/v1/messages", turn2)
        fu = m2.get("usage", {}) if st == 200 else {}
        check("multi-turn cache_read_input_tokens >= 4096 (system %s tok)" % ct.get("input_tokens"),
              st == 200 and fu.get("cache_read_input_tokens", 0) >= 4096
              and fu["input_tokens"] + fu["cache_read_input_tokens"] > ct.get("input_tokens", 0)
              and "rome" in text_of(m2).lower(), (st, fu, text_of(m2) if st == 200 else m2))
    else:
        check("multi-turn turn 1", False, (st, m1))

    # 4. thinking enabled, streamed
    think = {"model": "x", "max_tokens": 3000, "temperature": 0,
             "thinking": {"type": "enabled", "budget_tokens": 1024},
             "messages": [{"role": "user", "content": "What is 17*23? Reply with only the number."}]}
    st, ev, _ = stream(think)
    ok, why, tm = validate_stream(ev) if st == 200 else (False, ev, None)
    check("stream grammar (thinking)", ok, why)
    if tm:
        types = [b["type"] for b in tm["content"]]
        check("thinking block precedes text",
              types[:1] == ["thinking"] and (tm["stop_reason"] == "max_tokens" or "text" in types),
              types)
        if "text" in types:
            check("thinking answer 391", "391" in text_of(tm), text_of(tm))
    st, nm = post("/v1/messages", think)
    if st == 200 and tm:
        check("non-stream thinking == stream thinking",
              [b.get("thinking") for b in blocks_of(nm, "thinking")] ==
              [b.get("thinking") for b in blocks_of(tm, "thinking")] and text_of(nm) == text_of(tm),
              (nm.get("content"), tm.get("content")))
    else:
        check("non-stream thinking", False, (st, nm))

    # 5. tool use (auto), non-stream and stream
    tool_req = {"model": "x", "max_tokens": 512, "temperature": 0, "thinking": NO_THINK,
                "tools": [WEATHER_TOOL],
                "messages": [{"role": "user", "content":
                              "What's the weather in Paris right now? Use the tool."}]}
    st, tmsg = post("/v1/messages", tool_req)
    tu = blocks_of(tmsg, "tool_use") if st == 200 else []
    ok = (st == 200 and tmsg.get("stop_reason") == "tool_use" and len(tu) >= 1
          and tu[0]["name"] == "get_weather" and isinstance(tu[0].get("input"), dict)
          and "paris" in str(tu[0]["input"].get("city", "")).lower()
          and str(tu[0].get("id", "")).startswith("toolu_"))
    check("non-stream tool_use", ok, (st, tmsg))
    st, ev, _ = stream(tool_req)
    ok, why, stm = validate_stream(ev) if st == 200 else (False, ev, None)
    check("stream grammar (tool_use)", ok, why)
    if stm:
        stu = blocks_of(stm, "tool_use")
        check("stream tool_use input == non-stream",
              stm["stop_reason"] == "tool_use" and stu and tu and stu[0]["input"] == tu[0]["input"]
              and stu[0]["name"] == tu[0]["name"], (stm.get("content"), tu))
        check("stream has no whitespace-only text block",
              all(b["text"].strip() for b in blocks_of(stm, "text")), stm["content"])

    # 6. tool_result round trip
    if tu:
        hist = dict(tool_req)
        hist["messages"] = tool_req["messages"] + [
            {"role": "assistant", "content": tmsg["content"]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tu[0]["id"],
                 "content": [{"type": "text", "text": "Sunny, 23 degrees Celsius."}]}]}]
        st, rm = post("/v1/messages", hist)
        check("tool_result round trip", st == 200 and rm.get("stop_reason") == "end_turn"
              and "23" in text_of(rm), (st, rm))
    else:
        check("tool_result round trip", False, "skipped: no tool_use")

    # 7. forced tool_choice
    forced = dict(tool_req, tool_choice={"type": "tool", "name": "get_weather"},
                  messages=[{"role": "user", "content": "Tell me about Berlin."}])
    st, fm = post("/v1/messages", forced)
    check("tool_choice=tool forces a call", st == 200 and fm.get("stop_reason") == "tool_use"
          and blocks_of(fm, "tool_use") and blocks_of(fm, "tool_use")[0]["name"] == "get_weather",
          (st, fm))
    st, nm = post("/v1/messages", dict(tool_req, tool_choice={"type": "none"}))
    check("tool_choice=none gives text", st == 200 and not blocks_of(nm, "tool_use")
          and nm.get("stop_reason") == "end_turn", (st, nm))

    # 8. stop_sequences
    st, sm2 = post("/v1/messages", {
        "model": "x", "max_tokens": 64, "temperature": 0, "thinking": NO_THINK,
        "stop_sequences": ["FIVE"],
        "messages": [{"role": "user", "content":
                      "Write the words ONE TWO THREE FOUR FIVE SIX separated by spaces, nothing else."}]})
    check("stop_sequences", st == 200 and sm2.get("stop_reason") == "stop_sequence"
          and sm2.get("stop_sequence") == "FIVE" and "FIVE" not in text_of(sm2)
          and "FOUR" in text_of(sm2), (st, sm2))

    # 9. max_tokens
    st, mm = post("/v1/messages", {"model": "x", "max_tokens": 5, "temperature": 0,
                                   "thinking": NO_THINK,
                                   "messages": [{"role": "user", "content": "Write a long story."}]})
    check("max_tokens stop", st == 200 and mm.get("stop_reason") == "max_tokens"
          and mm.get("usage", {}).get("output_tokens") == 5, (st, mm))

    # 10. assistant prefill
    st, pm = post("/v1/messages", {
        "model": "x", "max_tokens": 64, "temperature": 0,
        "messages": [{"role": "user", "content": "Give the capital of Japan as JSON {\"capital\": ...}."},
                     {"role": "assistant", "content": "{\"capital\": \""}]})
    pt = text_of(pm) if st == 200 else ""
    check("assistant prefill continues", st == 200 and not blocks_of(pm, "thinking")
          and not pt.startswith("{\"capital\"") and "tokyo" in pt.lower(), (st, pm))

    # 11. image input (base64)
    img = base64.b64encode(tiny_png((220, 20, 20))).decode()
    st, im = post("/v1/messages", {
        "model": "x", "max_tokens": 32, "temperature": 0, "thinking": NO_THINK,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": "What single color fills this image? One word."}]}]})
    if st == 200:
        check("image input accepted", bool(text_of(im)), im)
        if "red" not in text_of(im).lower():
            print("WARN image answer did not say red: %r" % text_of(im))
    else:
        # no vision weights loaded is a deployment choice, not a protocol failure
        vis = "vision" in json.dumps(im).lower()
        check("image input (vision unavailable is ok)", vis, (st, im))

    # 12. errors use the Anthropic envelope
    st, e = post("/v1/messages", {"model": "x", "max_tokens": 10})
    check("error: missing messages", st == 400 and e.get("type") == "error"
          and e.get("error", {}).get("type") == "invalid_request_error", (st, e))
    st, e = post("/v1/messages", {"model": "x", "max_tokens": 10, "messages": [
        {"role": "user", "content": [{"type": "bogus"}]}]})
    check("error: unknown block type", st == 400 and e.get("type") == "error", (st, e))
    st, e = post("/v1/messages", {"model": "x", "max_tokens": 10, "temperature": 9,
                                  "messages": [{"role": "user", "content": "hi"}]})
    check("error: bad temperature", st == 400 and e.get("type") == "error", (st, e))
    st, ev, _ = stream({"model": "x", "max_tokens": 10, "temperature": 9,
                        "messages": [{"role": "user", "content": "hi"}]})
    check("error before stream starts is plain HTTP 400", st == 400, (st, ev))

    # 13. official SDK, if available
    if sdk:
        try:
            import anthropic
        except ImportError:
            anthropic = None
            print("SKIP anthropic SDK not installed (pip install anthropic)")
        if anthropic is not None:
            try:
                cl = anthropic.Anthropic(base_url=BASE, api_key="dummy", max_retries=0)
                # temperature / thinking go through extra_body: newer SDK
                # releases dropped some of these keyword arguments.
                r = cl.messages.create(model="x", max_tokens=64, tools=[WEATHER_TOOL],
                                       extra_body={"temperature": 0,
                                                   "thinking": {"type": "disabled"}},
                                       messages=[{"role": "user", "content":
                                                  "What's the weather in Rome? Use the tool."}])
                check("SDK non-stream tool_use", r.stop_reason == "tool_use" and any(
                    b.type == "tool_use" and b.name == "get_weather" for b in r.content), r)
                with cl.messages.stream(model="x", max_tokens=3000,
                                        extra_body={"temperature": 0, "thinking": {
                                            "type": "enabled", "budget_tokens": 1024}},
                                        messages=[{"role": "user", "content":
                                                   "What is 6*7? Only the number."}]) as s:
                    final = s.get_final_message()
                types = [b.type for b in final.content]
                check("SDK stream thinking+text", types[:1] == ["thinking"] and (
                    final.stop_reason == "max_tokens" or "42" in "".join(
                        b.text for b in final.content if b.type == "text")), (types, final))
                n = cl.messages.count_tokens(model="x", messages=[{"role": "user", "content": "hi"}])
                check("SDK count_tokens", n.input_tokens > 0, n)
            except Exception as ex:
                check("SDK calls", False, repr(ex))

    failed = [n for n, ok, _ in results if not ok]
    print("checks: %d, failed: %d%s" % (len(results), len(failed),
                                       (" (" + ", ".join(failed) + ")") if failed else ""))
    print("RESULT " + ("PASS" if not failed else "FAIL"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
