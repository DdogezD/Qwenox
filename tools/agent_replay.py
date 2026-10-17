#!/usr/bin/env python3
"""agent_replay.py — T0 agent-replay benchmark (HANDOFF-AGENT-SHORT-PREFILL.md §3.1).

Engine must already be serving on 127.0.0.1:8732 (single slot, v2 weights,
QWENOX_KVSNAP=0, rckpt default on). Uses ONE long connection: only the same
connection reuses the live prefix. Builds a depth-D context, then for each
increment length L runs twice (k=0 fresh length -> Lt miss, k=1 same length new
content -> Lt hit), chaining prompt += generated tokens + new tokens each round.

  python tools/agent_replay.py --depth 40960 --tag r0

Per-round line: L, k, actual new tokens, pre_ms, ttft-pre_ms, cached.
Fits two linear segments (<=300 and >=1000 new tokens, k=1 only) and prints the
mean k0-k1 delta (= Lt per-P miss cost). Last line AGENT_REPLAY PASS|FAIL.
Rows -> logs/areplay/<tag>_d<depth>.json
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefill_sched import Conn, load_tokens  # noqa: E402

LS = [16, 40, 64, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]


def fit(pts):  # [(x, y)] -> (intercept, slope)
    n = len(pts)
    if n < 2:
        return float('nan'), float('nan')
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=40960)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gen', type=int, default=8)
    a = ap.parse_args()
    T = load_tokens()
    need = a.depth + 2 * sum(LS) + 64
    assert len(T) >= need, (len(T), need)
    c = Conn()
    t_start = time.time()
    ctx = T[:a.depth]
    cur = a.depth
    print(f'[{time.strftime("%H:%M:%S")}] building depth-{a.depth} context '
          f'({a.depth} tokens prefill, not timed)', flush=True)
    r = c.gen(ctx, a.gen, 4)
    if r.get('reason') is None:
        print('initial GEN failed', flush=True)
        sys.exit(1)
    ctx = ctx + r['tokens']
    print(f'[{time.strftime("%H:%M:%S")}] context ready: pre {r["pre_ms"]:.0f} ms, '
          f'now {len(ctx)} tokens', flush=True)
    rows, ok = [], True
    for L in LS:
        for k in (0, 1):
            new = T[cur:cur + L]
            cur += L
            prompt = ctx + new
            r = c.gen(prompt, a.gen, 4)
            n_new = len(prompt) - (r['cached'] or 0)
            if r['cached'] is None or r['cached'] < len(ctx) - 8:
                ok = False
            rows.append(dict(L=L, k=k, n_new=n_new, pre_ms=r['pre_ms'],
                             ttft_ms=r['ttft'] * 1e3, cached=r['cached'],
                             gen_tok=r['tokens']))
            print(f"L={L:5d} k={k} new={n_new:5d} pre={r['pre_ms']:8.1f} ms "
                  f"ttft-pre={r['ttft'] * 1e3 - r['pre_ms']:6.1f} ms cached={r['cached']}",
                  flush=True)
            ctx = prompt + r['tokens']
    small = [(x['n_new'], x['pre_ms']) for x in rows if x['k'] == 1 and x['n_new'] <= 300]
    big = [(x['n_new'], x['pre_ms']) for x in rows if x['k'] == 1 and x['n_new'] >= 1000]
    lt = [a0['pre_ms'] - a1['pre_ms'] for a0, a1 in zip(rows[0::2], rows[1::2])]
    print('fit <=300 : %.0f ms + %.3f ms/tok' % fit(small))
    print('fit >=1000: %.0f ms + %.3f ms/tok' % fit(big))
    print('Lt miss (k0-k1) mean %.1f ms, max %.1f ms' % (sum(lt) / len(lt), max(lt)))
    out = Path(__file__).resolve().parent.parent / 'logs' / 'areplay'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.tag}_d{a.depth}.json').write_text(json.dumps(rows, indent=1))
    print('wall %.1f min' % ((time.time() - t_start) / 60))
    print('AGENT_REPLAY', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
