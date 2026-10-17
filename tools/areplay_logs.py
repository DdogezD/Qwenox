#!/usr/bin/env python3
"""areplay_logs.py — parse the engine logs of the agent-replay T1 runs.

  python tools/areplay_logs.py logs/areplay_engine_r1_ltmiss_d40960.log [...]

Handles, per log:
  lt_miss N=.. K=.. P=.. X ms        -> totals, by (N,K) shape, by P
  prof: gemm_host=.. ple_wait=..     -> sums per prof-chunk segment (P>=64)
  prof-chunk: P=.. wall=..ms         -> per-P table
  cutprof: <what> <ms> ms            -> count/sum by what
  kprof base=.. P=.. segsum=.. ms | name v (p%) [| [sub v (p%, ...)]] ...
                                     -> per-P bucket: top segments by ms
"""
import re, sys
from collections import defaultdict
from pathlib import Path

RE_LT = re.compile(r'lt_miss N=(\d+) K=(\d+) P=(\d+) ([\d.]+) ms')
RE_PROF = re.compile(r'prof: gemm_host=([\d.]+)ms topk_wait=([\d.]+)ms '
                     r'bucket\+h2d=([\d.]+)ms ple_host=([\d.]+)ms ple_wait=([\d.]+)ms')
RE_PCHUNK = re.compile(r'prof-chunk: P=(\d+) wall=([\d.]+)ms')
RE_CUT = re.compile(r'cutprof: (\S+)\s+([\d.]+) ms')
RE_KPROF = re.compile(r'kprof base=(\d+) P=(\d+) segsum=([\d.]+) ms(.*)')
RE_KSEG = re.compile(r'\| (?:\[([^\]]+) ([\d.]+) \([\d.]+%[^)]*\)\]|([^|\[]+?) ([\d.]+) \([\d.]+%\))')


def rep_lt(lines):
    rows = [RE_LT.search(l) for l in lines]
    rows = [m for m in rows if m]
    if not rows:
        return
    tot = sum(float(m.group(4)) for m in rows)
    print(f'  lt_miss: {len(rows)} misses, total {tot:.0f} ms, '
          f'mean {tot / len(rows):.1f} ms')
    by_shape = defaultdict(lambda: [0, 0.0, set()])
    by_p = defaultdict(lambda: [0, 0.0])
    for m in rows:
        N, K, P, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        s = by_shape[(N, K)]
        s[0] += 1; s[1] += ms; s[2].add(P)
        by_p[P][0] += 1; by_p[P][1] += ms
    print('  by shape (N, K): count, total ms, distinct P')
    for (N, K), (c, ms, ps) in sorted(by_shape.items(), key=lambda kv: -kv[1][1]):
        print(f'    ({N:>6}, {K:>6}): {c:4d}  {ms:9.0f} ms  P={sorted(ps)[:12]}'
              f'{"..." if len(ps) > 12 else ""} ({len(ps)} distinct)')
    print('  by P: count, total ms  (top 15 by total)')
    for P, (c, ms) in sorted(by_p.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f'    P={P:>5}: {c:4d}  {ms:9.0f} ms')


def rep_prof(lines):
    chunks = []   # (P, wall_ms, gemm_host, topk_wait, bucket, ple_host, ple_wait)
    pend = None
    for l in lines:
        m = RE_PROF.search(l)
        if m:
            pend = tuple(float(x) for x in m.groups())
        m = RE_PCHUNK.search(l)
        if m:
            P, wall = int(m.group(1)), float(m.group(2))
            chunks.append((P, wall) + (pend or (0, 0, 0, 0, 0)))
            pend = None
    if not chunks:
        return
    print(f'  prof-chunk: {len(chunks)} chunks')
    print('      P   wall_ms  gemm_host  topk_wait  bucket+h2d  ple_host  ple_wait')
    for P, wall, gh, tw, bk, ph, pw in chunks:
        print(f'  {P:>6} {wall:9.1f} {gh:9.0f} {tw:10.0f} {bk:10.0f} {ph:9.0f} {pw:9.0f}')


def rep_cut(lines):
    by = defaultdict(lambda: [0, 0.0])
    for l in lines:
        m = RE_CUT.search(l)
        if m:
            by[m.group(1)][0] += 1
            by[m.group(1)][1] += float(m.group(2))
    if not by:
        return
    print('  cutprof: what, count, total ms')
    for w, (c, ms) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f'    {w:<16} {c:5d}  {ms:9.1f} ms')


def rep_kprof(lines):
    by_pb = defaultdict(lambda: defaultdict(float))   # P bucket -> seg -> ms
    segsum = defaultdict(float)
    sub = defaultdict(lambda: defaultdict(float))
    for l in lines:
        m = RE_KPROF.search(l)
        if not m:
            continue
        P = int(m.group(2))
        segsum[P] += float(m.group(3))
        for sm in RE_KSEG.finditer(m.group(4)):
            if sm.group(1) is not None:  # bracketed sub-item
                sub[P][sm.group(1).strip()] += float(sm.group(2))
            else:
                by_pb[P][sm.group(3).strip()] += float(sm.group(4))
    if not by_pb:
        return
    for P in sorted(by_pb):
        tot = segsum[P]
        print(f'  kprof P={P} (n/a chunks counted together): segsum total {tot:.0f} ms')
        for name, ms in sorted(by_pb[P].items(), key=lambda kv: -kv[1])[:14]:
            print(f'    {name:<28} {ms:9.1f} ms  {100 * ms / tot:5.1f}%')
        for name, ms in sorted(sub[P].items(), key=lambda kv: -kv[1])[:6]:
            print(f'      [sub] {name:<24} {ms:9.1f} ms  {100 * ms / tot:5.1f}%')


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        lines = p.read_text(encoding='utf-8', errors='replace').replace('\r', '\n').split('\n')
        print(f'== {p.name} ({len(lines)} lines)')
        rep_lt(lines)
        rep_prof(lines)
        rep_cut(lines)
        rep_kprof(lines)
        print()


if __name__ == '__main__':
    main()
