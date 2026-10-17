#!/usr/bin/env python3
"""areplay_report.py — aggregate logs/areplay/*.json from agent_replay.py runs.

  python tools/areplay_report.py [tag_substr ...]   (default: all)

Per file: two-segment fit (<=300 / >=1000 new tokens, k=1 rows), mean k0-k1
delta (Lt miss cost), and the full row table. With several files, ends with a
side-by-side k=1 table (pre_ms per L per run).
"""
import json, sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent / 'logs' / 'areplay'


def fit(pts):
    n = len(pts)
    if n < 2:
        return float('nan'), float('nan')
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def summarize(rows):
    small = [(x['n_new'], x['pre_ms']) for x in rows if x['k'] == 1 and x['n_new'] <= 300]
    big = [(x['n_new'], x['pre_ms']) for x in rows if x['k'] == 1 and x['n_new'] >= 1000]
    lt = [a0['pre_ms'] - a1['pre_ms'] for a0, a1 in zip(rows[0::2], rows[1::2])]
    i0, s0 = fit(small)
    i1, s1 = fit(big)
    return (i0, s0, i1, s1, (sum(lt) / len(lt), max(lt)) if lt else (float('nan'),) * 2)


def main():
    pats = sys.argv[1:]
    files = sorted(D.glob('*.json'))
    if pats:
        files = [f for f in files if any(p in f.name for p in pats)]
    if not files:
        print('no logs/areplay/*.json', file=sys.stderr)
        sys.exit(1)
    runs = {}
    for f in files:
        rows = json.loads(f.read_text())
        runs[f.stem] = rows
        i0, s0, i1, s1, (lm, lx) = summarize(rows)
        print(f'== {f.stem}  ({len(rows)} rows)')
        print(f'   fit <=300 : {i0:6.0f} ms + {s0:.3f} ms/tok')
        print(f'   fit >=1000: {i1:6.0f} ms + {s1:.3f} ms/tok')
        print(f'   Lt miss (k0-k1): mean {lm:6.1f} ms  max {lx:6.1f} ms')
    if len(runs) > 1:
        names = list(runs)
        print()
        hdr = '      L   k |' + '|'.join(f'{n:>18}' for n in names)
        print(hdr)
        Ls = sorted({x['L'] for r in runs.values() for x in r})
        for L in Ls:
            for k in (0, 1):
                cells = []
                for n in names:
                    hit = [x for x in runs[n] if x['L'] == L and x['k'] == k]
                    cells.append(f'{hit[0]["pre_ms"]:12.1f} ms' if hit else ' ' * 17)
                print(f'{L:8d} {k:3d} |' + '|'.join(f'{c:>18}' for c in cells))


if __name__ == '__main__':
    main()
