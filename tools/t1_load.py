#!/usr/bin/env python3
# t1_load.py — 采样解码负载（T1 每轮开销拆分用）：C 路并发，每路依次发代码 / 散文题。
# 采样参数不传，走 API 默认。引擎需以 QWENOX_SYNC_PROF=1 启动，[roundprof] /
# [syncprof] 行写在引擎日志里，用 tools/t1_parse.py 汇总。纯 stdlib。
#
# 用法：python3 tools/t1_load.py --conc 4 --rpw 2 [--max-tokens 512] [--base-url ...]
#       [--seed 1000 --out res.json]   带 seed（每题 seed+题号）并把各题输出写成 JSON，
#                                       供不同并发 / 开关下逐字对比（tools/c7_verify.sh）

import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a complete quicksort implementation in Python with comments, edge-case handling and a small test harness. Output only the code.",
    "Write a short story (about 500 words) about a lighthouse keeper who finds a message in a bottle.",
    "Implement a thread-safe LRU cache in C++17 with get/put and a main() with tests.",
    "用中文写一篇约 600 字的散文，描写秋天傍晚的老街和街边的小吃摊。",
    "Implement Dijkstra's shortest path in Java using a PriorityQueue, with a sample graph in main.",
    "Explain to a high-school student why the sky is blue and why sunsets are red, in plain language.",
    "Write a Go HTTP server with routes /health and /items storing items in memory with a mutex.",
    "Write a letter from a grandmother to her grandson who just started university.",
]


def chat(base, prompt, max_tokens, timeout, seed=None):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8731")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--rpw", type=int, default=2, help="每路请求数")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, lock = [], threading.Lock()
    outs = {}
    bar = threading.Barrier(a.conc)

    def worker(w):
        bar.wait()
        for r in range(a.rpw):
            pi = (w + r * a.conc) % len(PROMPTS)
            p = PROMPTS[pi]
            t0 = time.time()
            try:
                j = chat(a.base_url, p, a.max_tokens, a.timeout,
                         None if a.seed is None else a.seed + pi)
                n = (j.get("usage") or {}).get("completion_tokens", 0)
                with lock:
                    outs[str(pi)] = (j.get("choices") or [{}])[0].get("message")
                err = None
            except Exception as e:  # noqa: BLE001
                n, err = 0, str(e)
            with lock:
                res.append((w, n, time.time() - t0, err))

    t0 = time.time()
    th = [threading.Thread(target=worker, args=(w,)) for w in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    tok = sum(r[1] for r in res)
    fails = [r for r in res if r[3]]
    for f in fails:
        print("FAIL", f)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(outs, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"C{a.conc}: {len(res)} req, {tok} tokens in {wall:.1f} s = {tok / wall:.1f} tok/s")
    print("T1_LOAD", "PASS" if not fails and tok > 0 else "FAIL")
    sys.exit(0 if not fails and tok > 0 else 1)


if __name__ == "__main__":
    main()
