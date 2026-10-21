#!/usr/bin/env python3
# spec_dump_collect.py — 串行发代码 / 散文采样请求，给 QWENOX_SPEC_DUMP 打请求边界。
#
# 引擎需以 QWENOX_SPEC_DUMP=<dump> 启动（建议 QWENOX_SPEC_CONF=0 + MTP_GAMMA=7，
# 每轮都起草满 7 步，任何更短的停稿规则都可离线截断回放：tools/conf_alloc_sim.py）。
# 本脚本与引擎同机运行，每个请求前后读 dump 行数，写清单 <dump>.manifest.json：
#   [{"kind": "code"|"prose", "id": ..., "l0": 起始行, "l1": 结束行, "tokens": n}, ...]
#
# 用法：python3 tools/spec_dump_collect.py --dump /path/spec.jsonl [--base-url ...]
#       [--n 12] [--max-tokens 768]
# 采样参数不传，走 API 默认（与客户端不传 temperature 时相同）。纯 stdlib。

import argparse
import json
import os
import sys
import time
import urllib.request

CODE = [
    "Write a complete quicksort implementation in Python with comments, edge-case handling and a small test harness. Output only the code.",
    "Implement a thread-safe LRU cache in C++17 with get/put, a doubly linked list and an unordered_map. Include a main() with tests.",
    "Write a Rust function that parses a CSV line with quoted fields and escaped quotes, plus unit tests.",
    "Implement Dijkstra's shortest path in Java using a PriorityQueue, with a sample graph in main.",
    "Write a Go HTTP server with routes /health, /items (GET, POST) storing items in memory with a mutex. Include handlers and main.",
    "Implement a binary search tree in TypeScript with insert, delete, in-order traversal and height. Include example usage.",
    "Write a C program that reads a text file and prints the 10 most frequent words using a hash table. Handle errors.",
    "Implement merge sort and heap sort in JavaScript and benchmark them on random arrays of 100k numbers.",
    "Write a Python asyncio crawler that fetches a list of URLs with a concurrency limit of 5 and retries on failure.",
    "Implement a trie in C# supporting insert, search, startsWith and delete, with xunit-style tests.",
    "Write a SQL schema for a library system (books, members, loans) and five example queries with explanations in comments.",
    "Implement a ring buffer in C with push, pop, peek and full/empty checks, plus a test main().",
    "Write a Python class implementing a matrix with addition, multiplication, transpose and determinant, with doctests.",
    "Implement the KMP string search algorithm in C++ with a function to build the prefix table and tests.",
    "Write a Bash script that backs up a directory into dated tar.gz archives and keeps only the last 7.",
    "Implement a simple tokenizer and recursive-descent parser for arithmetic expressions in Python, evaluating + - * / and parentheses.",
]

PROSE = [
    "Write a short story (about 500 words) about a lighthouse keeper who finds a message in a bottle.",
    "用中文写一篇约 600 字的散文，描写秋天傍晚的老街和街边的小吃摊。",
    "Explain to a high-school student why the sky is blue and why sunsets are red, in plain language.",
    "Write a persuasive essay arguing that cities should invest more in public libraries.",
    "用中文讲讲你对“慢生活”的理解，可以举几个日常的例子。",
    "Describe a day in the life of a medieval blacksmith, in vivid narrative prose.",
    "Write a letter from a grandmother to her grandson who just started university.",
    "用中文写一个童话故事：一只害怕黑夜的小猫头鹰。",
    "Compare the leadership styles of a ship's captain and an orchestra conductor in an essay.",
    "Write a travel diary entry about arriving in a small mountain village in the rain.",
    "用中文写一篇关于第一次学骑自行车的回忆。",
    "Explain the history of the printing press and how it changed Europe, for a general audience.",
    "Write a dialogue between two old friends meeting again after twenty years.",
    "用中文介绍一下茶在中国人生活中的位置，写成一篇随笔。",
    "Write a reflective essay on what makes a good teacher.",
    "Write a short mystery scene where a detective notices something odd about a dinner party.",
]


def nlines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def chat(base, prompt, max_tokens, timeout):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8731")
    ap.add_argument("--n", type=int, default=12, help="每类请求数")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    man, fails = [], 0
    jobs = [("code", i, p) for i, p in enumerate(CODE[:a.n])]
    jobs += [("prose", i, p) for i, p in enumerate(PROSE[:a.n])]
    t0 = time.time()
    for kind, i, p in jobs:
        l0 = nlines(a.dump)
        try:
            r = chat(a.base_url, p, a.max_tokens, a.timeout)
            ntok = (r.get("usage") or {}).get("completion_tokens", 0)
        except Exception as e:  # noqa: BLE001
            print(f"{kind}#{i} FAIL {e}", flush=True)
            fails += 1
            continue
        time.sleep(0.2)  # engine flushes the round line before replying; margin
        l1 = nlines(a.dump)
        man.append({"kind": kind, "id": i, "l0": l0, "l1": l1, "tokens": ntok})
        print(f"{kind}#{i} tokens {ntok} rounds {l1 - l0} ({time.time() - t0:.0f}s)",
              flush=True)
    with open(a.dump + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    rounds = {k: sum(m["l1"] - m["l0"] for m in man if m["kind"] == k)
              for k in ("code", "prose")}
    ok = fails == 0 and rounds["code"] > 0 and rounds["prose"] > 0
    print(f"rounds code {rounds['code']} prose {rounds['prose']}, fails {fails}")
    print("SPEC_DUMP_COLLECT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
