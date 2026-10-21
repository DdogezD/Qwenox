#!/usr/bin/env python3
# kprof_verify_sum.py — 汇总 QWENOX_KPROF=1 QWENOX_KPROF_V=1 引擎日志里 spec verify 的
# "kprof base=.. P=.. segsum=.." 行：按 P 分组，各分段 / 子区间的平均 ms。
# 用法：python3 tools/kprof_verify_sum.py engine.log [起始行 结束行] [--maxp 16]

import re
import sys
from collections import defaultdict


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    maxp = 16
    if "--maxp" in sys.argv:
        maxp = int(sys.argv[sys.argv.index("--maxp") + 1])
        args = [a for a in args if a != str(maxp)]
    lines = open(args[0], encoding="utf-8", errors="replace").read().splitlines()
    if len(args) >= 3:
        lines = lines[int(args[1]):int(args[2])]
    byp = defaultdict(lambda: [0, 0.0, defaultdict(float)])
    for l in lines:
        m = re.match(r".*kprof base=\d+ P=(\d+) segsum=([\d.]+) ms(.*)", l)
        if not m:
            continue
        P = int(m.group(1))
        if P > maxp:
            continue  # prompt prefill chunks
        e = byp[P]
        e[0] += 1
        e[1] += float(m.group(2))
        for part in m.group(3).split(" | ")[1:]:
            mm = re.match(r"\[?(.+?) ([\d.]+) \(", part.strip())
            if mm:
                key = ("  ↳ " if part.strip().startswith("[") else "") + mm.group(1)
                e[2][key] += float(mm.group(2))
    if not byp:
        print("no verify kprof lines")
        print("KPROF_SUM FAIL")
        sys.exit(1)
    for P in sorted(byp):
        n, tot, seg = byp[P]
        print(f"P={P}: {n} verifies, segsum {tot / n:.2f} ms")
        for k, v in sorted(seg.items(), key=lambda x: (x[0].startswith("  ↳"), -x[1])):
            print(f"    {k:28s} {v / n:7.2f} ms  {100 * v / tot:5.1f}%")
    print("KPROF_SUM PASS")


if __name__ == "__main__":
    main()
