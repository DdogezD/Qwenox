#!/usr/bin/env python3
"""codeneedle 运行与对比（直连裸引擎 127.0.0.1:8732 的 qwenox 协议）。

  python bench/codeneedle/run.py run --tag A   [--set S] [--seeds 4] [--only L16K]
  python bench/codeneedle/run.py cmp --a A --b B

run：每篇长文先单独 prefill 一次（SNAPS 在长文末尾留 RAM 检查点），之后每道题只 prefill
     问题部分。每题跑 1 次贪心 + seeds 次全分布采样（temp 1、top_k 0、top_p 1、min_p 0，
     SEEDED + LOGPROBS，seed 由题号决定，两轮相同）。结果写 logs/codeneedle/<tag>.json。
cmp：逐档（16K/32K/64K）对比两轮：贪心准确率与翻转（精确 McNemar）、采样准确率与翻转、
     同一采样答案的 logprob 变化；最后一行 CODENEEDLE PASS|FAIL。
     FAIL = 运行出错，或 B 相对 A 的退化在统计上显著（p<0.05）。
"""
import argparse
import hashlib
import json
import math
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.ROOT / 'logs' / 'codeneedle'
EOS = [C.IM_END, C.ENDOFTEXT]


class Engine:
    def __init__(self, port, timeout=1800):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=timeout)
        self.f = self.sock.makefile('rb')
        self.req = 0

    def gen(self, ids, n, sample=None, snap=None):
        """sample = (temp, top_k, top_p, min_p, seed) 或 None（贪心）。返回 dict。"""
        self.req += 1
        f = ['GEN', self.req, n, len(EOS), *EOS, len(ids), *ids, 0]   # drafter 0：不投机
        if sample:
            t, k, p, mp, seed = sample
            f += ['SAMPLE', t, k, p, mp, seed, 'SEEDED', 'LOGPROBS']
        if snap:
            f += ['SNAPS', 1, snap]
        self.sock.sendall((' '.join(map(str, f)) + '\n').encode())
        toks, lps = [], []
        while True:
            v = self.f.readline().decode().split()
            if not v:
                raise RuntimeError('engine disconnected')
            if v[0] == 'T' and int(v[1]) == self.req:
                toks.append(int(v[2]))
                lps.append(float(v[3]) if len(v) > 3 else None)
            elif v[0] == 'D' and int(v[1]) == self.req:
                num = lambda i: float(v[i]) if len(v) > i else 0.0
                return dict(tokens=toks, lps=lps, reason=v[2], n_prompt=int(num(3)),
                            prefill_ms=num(5), n_cached=int(num(10)))


def seed_of(qid, k):
    return int.from_bytes(hashlib.sha256(f'{qid}#{k}'.encode()).digest()[:6], 'little')


def load_set(path):
    S = json.load(open(path, encoding='utf-8'))
    paths = sorted({f[0] for h in S['haystacks'] for f in h['frags']})
    blobs = C.read_blobs(S['rev'], paths)
    return S, blobs


def cmd_run(a):
    S, blobs = load_set(a.set)
    tk = C.tokenizer()
    special = C.special_ids(tk)
    eng = Engine(a.port)
    byhay = defaultdict(list)
    for q in S['questions']:
        byhay[q['hay']].append(q)
    hays = [h for h in S['haystacks'] if not a.only or h['id'].startswith(a.only)]
    res, errors, t0 = {}, 0, time.time()
    stats = dict(warm_tokens=0, warm_ms=0.0, q_cached_hits=0, q_total=0)
    for hi, h in enumerate(hays):
        frags = [tuple(f) for f in h['frags']]
        text = C.haystack_text(blobs, frags)
        if C.sha(text) != h['sha256']:
            raise SystemExit(f'{h["id"]}: haystack sha mismatch (rev {S["rev"][:10]} not available?)')
        pre = C.prefix_ids(tk, text)
        nsp = sum(t in special for t in pre)
        if nsp != 1:     # 只允许开头的 <|im_start|>
            raise SystemExit(f'{h["id"]}: {nsp} special tokens in the prompt prefix (want 1)')
        w = eng.gen(pre, 1, snap=len(pre))
        if w['reason'] == 'error':
            errors += 1
            print(f'  {h["id"]}: warm-up error'); continue
        stats['warm_tokens'] += len(pre)
        stats['warm_ms'] += w['prefill_ms']
        basenames = defaultdict(int)
        for p in {f[0] for f in frags}:
            basenames[p.rsplit('/', 1)[-1]] += 1
        nok = 0
        for q in byhay[h['id']]:
            ids = pre + C.suffix_ids(tk, q['q'])
            r = dict(type=q['type'], hay=h['id'], L=h['L'], depth=q['depth'], answer=q['answer'])
            g = eng.gen(ids, a.max_tokens, snap=len(pre))
            stats['q_total'] += 1
            stats['q_cached_hits'] += g['n_cached'] >= len(pre)
            out = tk.decode([t for t in g['tokens'] if t not in EOS])
            ok, got = C.judge(q, out, basenames)
            r.update(greedy=dict(out=out, ok=ok, got=got, reason=g['reason']))
            errors += g['reason'] == 'error'
            nok += ok
            r['samples'] = []
            for k in range(a.seeds):
                s = eng.gen(ids, a.max_tokens, sample=(1, 0, 1, 0, seed_of(q['id'], k)), snap=len(pre))
                errors += s['reason'] == 'error'
                toks = [t for t in s['tokens'] if t not in EOS]
                o = tk.decode(toks)
                sok, sgot = C.judge(q, o, basenames)
                lp = [x for t, x in zip(s['tokens'], s['lps']) if t not in EOS and x is not None]
                r['samples'].append(dict(out=o, ok=sok, got=sgot, toks=toks, lp=sum(lp),
                                         lp0=lp[0] if lp else None, reason=s['reason']))
            res[q['id']] = r
        print(f'  [{hi + 1}/{len(hays)}] {h["id"]}: {len(pre)} tok prefill {w["prefill_ms"] / 1000:.1f}s, '
              f'greedy {nok}/{len(byhay[h["id"]])}, {time.time() - t0:.0f}s elapsed', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = dict(tag=a.tag, set=Path(a.set).name, set_sha=hashlib.sha256(Path(a.set).read_bytes()).hexdigest(),
               rev=S['rev'], seeds=a.seeds, errors=errors, stats=stats, wall_s=time.time() - t0, res=res)
    (OUT / f'{a.tag}.json').write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    tps = stats['warm_tokens'] / max(1e-9, stats['warm_ms'] / 1000)
    print(f'{a.tag}: {len(res)} questions, errors {errors}, cache hits {stats["q_cached_hits"]}/{stats["q_total"]}, '
          f'haystack prefill {tps:.0f} tok/s, wall {doc["wall_s"] / 60:.1f} min -> {OUT / (a.tag + ".json")}')
    return 0 if errors == 0 else 1


# ---------------- 统计 ----------------
def binom_two_sided(k, n):
    """精确符号检验：n 次翻转里有 k 次朝某方向，双侧 p。"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_ci(b, c, n):
    """配对准确率差 (B-A) 的 95% 置信区间（单位：百分点）。b=A对B错, c=A错B对。"""
    if n == 0:
        return 0.0, 0.0, 0.0
    d = (c - b) / n
    var = max(0.0, (b + c) / n - d * d) / n
    h = 1.96 * math.sqrt(var)
    return 100 * d, 100 * (d - h), 100 * (d + h)


def cmd_cmp(a):
    A = json.loads((OUT / f'{a.a}.json').read_text(encoding='utf-8'))
    B = json.loads((OUT / f'{a.b}.json').read_text(encoding='utf-8'))
    fails = []
    if A['set_sha'] != B['set_sha']:
        fails.append('two runs used different question sets')
    for d in (A, B):
        if d['errors']:
            fails.append(f'{d["tag"]}: {d["errors"]} engine errors')
    ids = [k for k in A['res'] if k in B['res']]
    if len(ids) != len(A['res']) or len(ids) != len(B['res']):
        fails.append(f'question sets differ ({len(A["res"])} vs {len(B["res"])})')
    groups = defaultdict(list)
    for k in ids:
        groups[f'{A["res"][k]["L"] // 1024}K'].append(k)
    groups = dict(sorted(groups.items(), key=lambda x: int(x[0][:-1])))
    groups['all'] = ids
    print(f'A = {A["tag"]}   B = {B["tag"]}   {len(ids)} questions x (1 greedy + {A["seeds"]} sampled)')
    print(f'{"":5} {"greedy A":>9} {"greedy B":>9} {"A>B":>4} {"B>A":>4} {"p":>6} {"B-A 95% CI (pt)":>20} '
          f'{"same ans":>8} | {"sample A":>8} {"sample B":>8} {"A>B":>4} {"B>A":>4} {"p":>6} | {"dlogp":>7} {"n":>4}')
    verdict_bad = []
    if not ids:
        fails.append('no results to compare')
        groups = {}
    for gname, ks in groups.items():
        n = len(ks)
        ga = sum(A['res'][k]['greedy']['ok'] for k in ks)
        gb = sum(B['res'][k]['greedy']['ok'] for k in ks)
        b = sum(A['res'][k]['greedy']['ok'] and not B['res'][k]['greedy']['ok'] for k in ks)
        c = sum(B['res'][k]['greedy']['ok'] and not A['res'][k]['greedy']['ok'] for k in ks)
        same = sum(A['res'][k]['greedy']['out'] == B['res'][k]['greedy']['out'] for k in ks)
        p = binom_two_sided(b, b + c)
        d, lo, hi = paired_ci(b, c, n)
        sa = sb = sb_ = sc_ = 0
        dl = []
        for k in ks:
            for x, y in zip(A['res'][k]['samples'], B['res'][k]['samples']):
                sa += x['ok']; sb += y['ok']
                sb_ += x['ok'] and not y['ok']; sc_ += y['ok'] and not x['ok']
                if x['toks'] == y['toks'] and x['toks']:
                    dl.append(y['lp'] - x['lp'])
        ns = n * A['seeds']
        sp = binom_two_sided(sb_, sb_ + sc_)
        mdl = sum(dl) / len(dl) if dl else 0.0
        print(f'{gname:5} {100 * ga / n:8.1f}% {100 * gb / n:8.1f}% {b:4d} {c:4d} {p:6.3f} '
              f'{d:+6.1f} [{lo:+5.1f},{hi:+5.1f}] {100 * same / n:7.0f}% | '
              f'{100 * sa / ns:7.1f}% {100 * sb / ns:7.1f}% {sb_:4d} {sc_:4d} {sp:6.3f} | {mdl:+7.3f} {len(dl):4d}')
        if gname != 'all' and (ga / n > 0.97 or ga / n < 0.2):
            print(f'      note: {gname} baseline {100 * ga / n:.0f}% is near the edge; flips here carry little signal')
        if (p < 0.05 and b > c) or (sp < 0.05 and sb_ > sc_):
            verdict_bad.append(gname)
    # 按题型
    print('by type (greedy A -> B):  ' + '  '.join(
        f'{t} {sum(A["res"][k]["greedy"]["ok"] for k in ids if A["res"][k]["type"] == t)}->'
        f'{sum(B["res"][k]["greedy"]["ok"] for k in ids if A["res"][k]["type"] == t)}'
        f'/{sum(A["res"][k]["type"] == t for k in ids)}'
        for t in ('env_file', 'func_file', 'const_val', 'env_def')))
    print('A>B: A right, B wrong.  same ans: greedy outputs identical.  dlogp: mean logP(B)-logP(A) of the '
          'same sampled answer (negative = B less sure).')
    flips = [k for k in ids if A['res'][k]['greedy']['ok'] != B['res'][k]['greedy']['ok']]
    for k in flips[:12]:
        x, y = A['res'][k]['greedy'], B['res'][k]['greedy']
        print(f'  flip {k}: want {A["res"][k]["answer"]!r}  A {x["got"]!r}  B {y["got"]!r}')
    for f in fails:
        print('  FAIL: ' + f)
    if verdict_bad:
        print(f'  B is measurably worse in: {", ".join(verdict_bad)} (p < 0.05)')
    if fails or verdict_bad:
        print('CODENEEDLE FAIL')
        return 1
    print('CODENEEDLE PASS (no measurable degradation)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--tag', required=True)
    r.add_argument('--set', default=str(C.HERE / 'set.json'))
    r.add_argument('--port', type=int, default=8732)
    r.add_argument('--seeds', type=int, default=4)
    r.add_argument('--max-tokens', type=int, default=40)
    r.add_argument('--only', default='', help='只跑 id 以此开头的长文，如 L16K')
    c = sub.add_parser('cmp')
    c.add_argument('--a', required=True)
    c.add_argument('--b', required=True)
    a = ap.parse_args()
    sys.exit(cmd_run(a) if a.cmd == 'run' else cmd_cmp(a))


if __name__ == '__main__':
    main()
