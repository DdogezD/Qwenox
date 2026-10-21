#!/usr/bin/env python3
# conf_alloc_sim.py — 离线回放：并发采样 MTP 轮的验证行分配策略（DEEPINFER-NOTES C1）。
#
# 输入：QWENOX_SPEC_DUMP（QWENOX_SPEC_CONF=0 + MTP_GAMMA=7 采的，每轮 g=7）+
#       spec_dump_collect.py 写的 <dump>.manifest.json（请求边界与 code/prose）。
# 截断一致性：同一位置的草稿与接受随机数按位置派生，γ' < 7 时接受数 = min(a, γ')。
#
# 一轮（B 路同时）的耗时，推广现行 OccGamma 的成本式到各路 γ 不等：
#   T = A + B·SEQ + ROW·Σ(γi+1) + DRAFT1·Σmax(γi,1)      （采样草稿逐路串行）
# 策略：
#   cur   现行：OccGamma.pick（每路自己的接受历史，夹到 48/B−1）后 θ=0.9 停稿
#   thr   新：草稿 j 的校准接受率累乘 P_j < τ 时停（草稿 j 照常验证），τ 全局统一；
#         超 48 行按 P 截尾。报告扫描最优 τ 与不动点 τ = (ROW+DRAFT1)·λ（λ = 全局 token/ms）
#   orac  上界：每路按事后真实接受数挑 γ（不可实现，只看天花板）
# 用法：python3 tools/conf_alloc_sim.py spec.jsonl [--levels 1,2,4,8] [--mix code,prose]

import argparse
import json
import math
import random

KMAX = 7


def load(path):
    rounds = [json.loads(l) for l in open(path, encoding="utf-8")]
    man = json.load(open(path + ".manifest.json", encoding="utf-8"))
    reqs = []
    for m in man:
        rs = [(r["qt"], r["a"]) for r in rounds[m["l0"]:m["l1"]]
              if r.get("g") == KMAX and len(r.get("qt", [])) == KMAX]
        if rs:
            reqs.append({"kind": m["kind"], "rounds": rs})
    return reqs


class Calib:
    """c(qt) = P(草稿 j 被接受 | 前面都被接受, 草稿 j 的 top-1 落在该桶)。"""
    EDGES = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 1.01]

    def __init__(self, reqs):
        nb = len(self.EDGES) - 1
        hit, tot = [1.0] * nb, [2.0] * nb  # 先验 0.5，权重 2
        for r in reqs:
            for qt, a in r["rounds"]:
                for j in range(min(a + 1, KMAX)):
                    b = self.bin(qt[j])
                    tot[b] += 1
                    hit[b] += 1 if j < a else 0
        self.c = [h / t for h, t in zip(hit, tot)]
        self.n = tot

    def bin(self, x):
        for i in range(len(self.EDGES) - 1):
            if x < self.EDGES[i + 1]:
                return i
        return len(self.EDGES) - 2

    def __call__(self, x):
        return self.c[self.bin(x)]


class Occ:
    """现行 OccGamma（采样：draft_batched=false），逐路实例。"""

    def __init__(self, cm):
        self.cm = cm
        self.p = [0.85] * 8
        self.n = [4.0] * 8
        self.last = -1
        self.rounds = 0

    def cost(self, B, G):
        cm = self.cm
        return cm.A + B * (cm.SEQ + cm.ROW * (G + 1)) + cm.DRAFT1 * B * max(G, 1)

    def gain(self, G):
        e, pr = 1.0, 1.0
        for k in range(G):
            pr *= self.p[k]
            e += pr
        return e

    def pick(self, g, B, rows=48):
        if B > 1:
            g = min(g, max(1, rows // B - 1))
        if B <= 1:
            return g
        best, sb = g, self.gain(g) / self.cost(B, g)
        for G in range(g):
            s = self.gain(G) / self.cost(B, G)
            if s > sb:
                best, sb = G, s
        if 0 <= self.last <= g and best != self.last and \
                sb <= 1.01 * self.gain(self.last) / self.cost(B, self.last):
            best = self.last
        self.last = best
        self.rounds += 1
        return best + 1 if best < g and (self.rounds & 7) == 0 else best

    def update(self, g, a):
        for j in range(min(g, a + 1)):
            self.n[j] += 1
            self.p[j] += max(0.02, 1 / self.n[j]) * ((1 if j < a else 0) - self.p[j])


def theta_len(qt, g, th=0.9):
    """现行 spec_conf_stop：草稿 j 的 top-1 < θ 时停，验证到 j。"""
    for j in range(g):
        if j + 1 >= g or qt[j] < th:
            return j + 1
    return g


def thr_len(qt, cal, tau, k=1.0, la=1.0):
    """草稿 j 起草后：P_j·la < τ 就停（la = 下一步的期望接受率，前瞻）。"""
    P = 1.0
    for j in range(KMAX):
        P *= min(0.99, cal(qt[j]) * k)
        if P * la < tau or j + 1 >= KMAX:
            return j + 1, P
    return KMAX, P


def simulate(streams, B, policy, cm, cal=None, tau=0.0, nround=4000, seed=0):
    rng = random.Random(seed)
    pos = [rng.randrange(len(s)) for s in streams[:B]]
    occs = [Occ(cm) for _ in range(B)]
    # thra：每路"实际/预测"接受率修正（逐行 EMA，按本路已验证的行更新）
    ko, kp = [12.0] * B, [12.0] * B  # 先验逐行接受率 0.6
    tok = 0.0
    ms = 0.0
    rows_tot = 0
    for _ in range(nround):
        lens, Ps, rs = [], [], []
        for i in range(B):
            s = streams[i]
            qt, a = s[pos[i] % len(s)]
            pos[i] += 1
            rs.append((qt, a))
            if policy == "cur":
                g = occs[i].pick(KMAX, B)
                g = theta_len(qt, max(1, g)) if g > 0 else 0
                occs[i].update(g, min(a, g))
                lens.append(g)
                Ps.append(1.0)
            elif policy in ("thr", "thra"):
                # thra：不做比例修正，改为前瞻——下一步期望接受率 = 本路近期逐行实际接受率
                la = ko[i] / 20 if policy == "thra" else 1.0
                g, P = thr_len(qt, cal, tau, 1.0, la)
                lens.append(g)
                Ps.append(P)
            else:  # oracle：事后最优（各路独立，按边际 token/行）
                lens.append(min(a + 1, KMAX))
                Ps.append(1.0)
        # 48 行硬预算：按 P 从低到高逐行截掉（每路至少留 1 个草稿位 → γ ≥ 0）
        while sum(g + 1 for g in lens) > cm.ROWS:
            i = min((k for k in range(B) if lens[k] > 0), key=lambda k: Ps[k])
            lens[i] -= 1
            Ps[i] = Ps[i] / max(1e-6, cal(rs[i][0][lens[i]])) if cal else Ps[i]
        for i in range(B):
            tok += min(rs[i][1], lens[i]) + 1
            if policy == "thra":
                qt, a = rs[i]
                for j in range(min(a + 1, lens[i])):
                    ko[i] += 0.05 * ((1.0 if j < a else 0.0) - ko[i] / 20)
                    kp[i] += 0.05 * (cal(qt[j]) - kp[i] / 20)
        rows = sum(g + 1 for g in lens)
        rows_tot += rows
        ms += cm.A + B * cm.SEQ + cm.ROW * rows + cm.DRAFT1 * sum(max(g, 1) for g in lens)
    return tok / ms * 1000.0, rows_tot / nround


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--A", type=float, default=40.0)
    ap.add_argument("--SEQ", type=float, default=2.0)
    ap.add_argument("--ROW", type=float, default=5.0)
    ap.add_argument("--DRAFT1", type=float, default=3.0)
    ap.add_argument("--rows", type=int, default=48)
    a = ap.parse_args()
    cm = argparse.Namespace(A=a.A, SEQ=a.SEQ, ROW=a.ROW, DRAFT1=a.DRAFT1, ROWS=a.rows)
    reqs = load(a.dump)
    by = {k: [r["rounds"] for r in reqs if r["kind"] == k] for k in ("code", "prose")}
    for k, v in by.items():
        n = sum(len(x) for x in v)
        acc = sum(min(r[1], KMAX) for x in v for r in x) / max(1, n)
        print(f"{k}: {len(v)} req, {n} rounds, mean accepted @γ7 {acc:.2f}")
    # 校准：奇偶请求交叉（用一半拟合、另一半回放），避免同数据自评偏乐观
    halves = [reqs[0::2], reqs[1::2]]
    cals = [Calib(halves[1]), Calib(halves[0])]
    print("calib (all):", " ".join(f"{e:.2f}:{c:.2f}" for e, c in
                                  zip(Calib.EDGES, Calib(reqs).c)))
    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"cost A={cm.A} SEQ={cm.SEQ} ROW={cm.ROW} DRAFT1={cm.DRAFT1} rows={cm.ROWS}")
    for mix in ("code", "prose", "mix"):
        print(f"\n== {mix}")
        print(f"{'B':>2} {'cur':>7} {'thr*':>7} {'τ*':>5} {'thr_fp':>7} {'τfp':>5} "
              f"{'orac':>7} {'thra':>7}  rows cur/thr")
        for B in [int(x) for x in a.levels.split(",")]:
            res = {}
            for h in (0, 1):
                hr = halves[h]
                pool = {k: [r["rounds"] for r in hr if r["kind"] == k]
                        for k in ("code", "prose")}
                # 第 i 路 = 该类全部请求轮转 i 位后首尾相接；mix 时偶数路代码、奇数路散文
                def stream(kind, i):
                    p = pool[kind]
                    k = i % len(p)
                    return [r for x in p[k:] + p[:k] for r in x]
                streams = [stream(("code" if i % 2 == 0 else "prose") if mix == "mix" else mix, i)
                           for i in range(B)]
                cur = simulate(streams, B, "cur", cm)
                best = max(((simulate(streams, B, "thr", cm, cals[h], t), t) for t in taus),
                           key=lambda x: x[0][0])
                # 不动点：τ = (ROW+DRAFT1)·λ（多一行 = 多起草一步 + 多验证一行；λ token/ms）
                t = 0.1
                for _ in range(8):
                    r = simulate(streams, B, "thr", cm, cals[h], t)
                    t = max(0.01, min(0.95, (cm.ROW + cm.DRAFT1) * r[0] / 1000.0))
                fp = simulate(streams, B, "thr", cm, cals[h], t)
                ta = 0.1
                for _ in range(8):
                    r = simulate(streams, B, "thra", cm, cals[h], ta)
                    ta = max(0.01, min(0.95, (cm.ROW + cm.DRAFT1) * r[0] / 1000.0))
                fpa = simulate(streams, B, "thra", cm, cals[h], ta)
                orac = simulate(streams, B, "orac", cm)
                for key, val in (("cur", cur[0]), ("best", best[0][0]), ("tau", best[1]),
                                 ("fp", fp[0]), ("tfp", t), ("orac", orac[0]), ("fpa", fpa[0]),
                                 ("rc", cur[1]), ("rt", fp[1])):
                    res[key] = res.get(key, 0) + val / 2
            print(f"{B:>2} {res['cur']:7.1f} {res['best']:7.1f} {res['tau']:5.2f} "
                  f"{res['fp']:7.1f} {res['tfp']:5.2f} {res['orac']:7.1f} {res['fpa']:7.1f}  "
                  f"{res['rc']:.1f}/{res['rt']:.1f}   fp {100 * (res['fp'] / res['cur'] - 1):+.1f}%"
                  f"  thra {100 * (res['fpa'] / res['cur'] - 1):+.1f}%")


if __name__ == "__main__":
    main()
