#!/usr/bin/env python3
"""api_prep_bench.py — T2b: API-side fixed cost vs message count
(HANDOFF-AGENT-SHORT-PREFILL.md §3.3).

Posts synthetic agent conversations (system + user + N alternating
assistant(tool_calls)/tool messages + final user) to the API with
max_tokens=1, twice per size (cold then tcache-warm). The API's own
`qwenox-api: prep msgs=.. render=.. encode=.. cuts=..` line (stderr) carries
the server-side breakdown; this script records client wall + prompt/cached
tokens per request and prints a summary table.

  python tools/api_prep_bench.py [--port 8733] [--sizes 50,200,400] [--tokens 80000]
"""
import argparse, json, sys, time, urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "log_temperatures",
        "description": "把一组温度读数写入监控系统",
        "parameters": {
            "type": "object",
            "properties": {
                "readings": {"type": "array", "items": {"type": "number"}},
                "unit": {"type": "string"},
            },
            "required": ["readings", "unit"],
        },
    },
}]


def post(port, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_secs"] = time.time() - t0
    return out


def filler(i, n, tag):
    # ~4 chars/token English prose, deterministic per (i, tag)
    words = f"{tag} record {i} reading value sample batch metric observation "
    reps = max(1, n // len(words) + 1)
    return (words * reps)[:n]


def build(n_msgs, total_chars):
    pairs = (n_msgs - 2) // 2  # minus system + final user
    per_pair = total_chars // max(1, pairs)
    a_len = per_pair * 3 // 5
    t_len = per_pair - a_len
    msgs = [{"role": "system", "content": "You are a careful operations assistant."},
            {"role": "user", "content": "Start logging the following readings."}]
    for i in range(pairs):
        msgs.append({
            "role": "assistant",
            "content": filler(i, a_len, "analysis"),
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "log_temperatures",
                             "arguments": json.dumps(
                                 {"readings": [i + 0.5, i + 1.5], "unit": "celsius"})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": filler(i, t_len, "result")})
    msgs.append({"role": "user", "content": "Done? Answer briefly."})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8733)
    ap.add_argument('--sizes', default='50,200,400')
    ap.add_argument('--tokens', type=int, default=80000,
                    help='approx total prompt tokens per conversation')
    a = ap.parse_args()
    total_chars = a.tokens * 4
    print(f'{"msgs":>5} {"run":>5} {"prompt_tok":>10} {"cached":>8} {"wall_s":>8}', flush=True)
    for n in [int(x) for x in a.sizes.split(',')]:
        msgs = build(n, total_chars)
        for run in ('cold', 'warm'):
            o = post(a.port, {"messages": msgs, "tools": TOOLS,
                              "max_tokens": 1, "temperature": 0})
            u = o.get("usage", {})
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(f'{n:>5} {run:>5} {u.get("prompt_tokens", 0):>10} '
                  f'{cached:>8} {o["_secs"]:>8.2f}', flush=True)


if __name__ == '__main__':
    main()
