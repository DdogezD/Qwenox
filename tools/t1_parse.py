#!/usr/bin/env python3
# t1_parse.py — 汇总 QWENOX_SYNC_PROF 引擎日志：[roundprof] 按轮数加权平均，
# [syncprof] 调用位置跨请求累加（每轮次数 / 阻塞 ms）。
# 用法：python3 tools/t1_parse.py engine.log [起始行 结束行]

import re
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    if len(sys.argv) >= 4:
        lines = lines[int(sys.argv[2]):int(sys.argv[3])]
    seg, rounds_rp = defaultdict(float), 0
    gam, cpr = 0.0, 0.0
    sites = defaultdict(lambda: [0, 0.0])
    kinds = defaultdict(lambda: [0, 0.0])
    sp_rounds, sp_ms, sp_calls, sp_win = 0, 0.0, 0, 0.0
    for l in lines:
        m = re.search(r"\[roundprof\] chain-sample mtp (\d+) rounds, ms/round:(.*)", l)
        if m:
            n = int(m.group(1))
            rounds_rp += n
            body = m.group(2)
            for k, v in re.findall(r"(\w+) (-?[\d.]+)", body.split("|")[0]):
                seg[k] += float(v) * n
            o = re.search(r"other (-?[\d.]+)", body)
            if o:
                seg["other"] += float(o.group(1)) * n
            g = re.search(r"γ avg ([\d.]+), commit/round ([\d.]+)", body)
            if g:
                gam += float(g.group(1)) * n
                cpr += float(g.group(2)) * n
            continue
        m = re.search(r"\[syncprof\] chain-sample: ([\d.]+) ms window, (\d+) rounds \| (\d+) calls"
                      r".*blocked ([\d.]+) ms", l)
        if m:
            sp_win += float(m.group(1))
            sp_rounds += int(m.group(2))
            sp_calls += int(m.group(3))
            sp_ms += float(m.group(4))
            continue
        m = re.search(r"\[syncprof\]\s+by kind:(.*)", l)
        if m:
            for name, n, ms in re.findall(r"([a-zA-Z0-9 ]+?) (\d+)/([\d.]+)ms", m.group(1)):
                kinds[name.strip()][0] += int(n)
                kinds[name.strip()][1] += float(ms)
            continue
        m = re.search(r"\[syncprof\]\s+(\S+)\s*:(\d+)\s+(.+?)\s+n=(\d+).*?\s([\d.]+) ms \(", l)
        if m:
            key = f"{m.group(1)}:{m.group(2)} {m.group(3).strip()}"
            sites[key][0] += int(m.group(4))
            sites[key][1] += float(m.group(5))
    if not rounds_rp:
        print("no [roundprof] lines")
        print("T1_PARSE FAIL")
        sys.exit(1)
    r = rounds_rp
    print(f"MTP rounds {r}, γ avg {gam / r:.2f}, commit/round {cpr / r:.2f}")
    print("ms/round: " + "  ".join(f"{k} {v / r:.2f}" for k, v in seg.items()))
    sr = max(1, sp_rounds)
    print(f"sync calls {sp_calls / sr:.1f}/round, blocked {sp_ms / sr:.2f} ms/round "
          f"({100 * sp_ms / max(1e-9, sp_win):.1f}% of window; windows overlap under concurrency)")
    print("by kind (per round): " + "  ".join(
        f"{k} {n / sr:.1f}x/{ms / sr:.2f}ms" for k, (n, ms) in
        sorted(kinds.items(), key=lambda x: -x[1][1])))
    print("top sites (per round):")
    for k, (n, ms) in sorted(sites.items(), key=lambda x: -x[1][1])[:20]:
        print(f"  {k:50s} {n / sr:6.2f}x {ms / sr:7.3f} ms  ({ms / max(1, n):.3f} ms/call)")
    print("T1_PARSE PASS")


if __name__ == "__main__":
    main()
