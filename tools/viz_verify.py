#!/usr/bin/env python3
"""viz_verify.py -- one-shot check of the QWENOX_VIZ runtime taps (19_viz.inc).

Starts the engine once per configuration, talks the engine line protocol
directly (GEN / VIZ), stops it again. Configurations:
  base   optional reference engine built without the taps (--base)
  off    the engine under test, QWENOX_VIZ unset
  on     QWENOX_VIZ=1 (QWENOX_PARALLEL from the environment)
  on1    QWENOX_VIZ=1, QWENOX_PARALLEL=1 (single-slot graph paths), compared
         with off1 / base1 (same QWENOX_PARALLEL=1: one and several slots take
         different verify paths, so their outputs are compared separately)
Checks:
  exact   greedy tokens identical across every configuration and repeat
  speed   decode tok/s of "on" within --max-slow percent of "off"
          (and "off" within the same bound of "base" when given)
  data    every exposed position carries 48 layers of routing with 10
          distinct expert ids in [0, experts) and weights summing to 1;
          the top-1 candidate at position p equals the generated token at
          p+1 (greedy), for single and concurrent requests and across the
          ring wrap (prompt longer than the 2048-position ring)
Ends with PASS or FAIL. Usage (normally via tools/viz_verify_win.sh):
  python tools/viz_verify.py --engine EXE [--base EXE] [--port 8750] -- <engine args>
"""
import argparse, json, os, socket, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_TOKENS = 320


def log(*a):
    print(*a, flush=True)


class Eng:
    def __init__(self, port):
        self.port = port

    def conn(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=900)
        return s, s.makefile("r", encoding="utf-8", newline="\n")

    def gen(self, req, ids, n):
        s, f = self.conn()
        try:
            line = "GEN %d %d 0 %d %s\n" % (req, n, len(ids), " ".join(map(str, ids)))
            s.sendall(line.encode())
            toks = []
            while True:
                l = f.readline()
                if not l:
                    raise RuntimeError("engine closed the connection")
                p = l.split()
                if p[0] == "T":
                    toks.append(int(p[2]))
                elif p[0] == "D":
                    if p[2] in ("error", "cancel"):
                        raise RuntimeError("GEN failed: " + l.strip())
                    return dict(toks=toks, n_gen=int(p[4]), prefill_ms=float(p[5]),
                                decode_ms=float(p[6]), rounds=int(p[8]) if len(p) > 8 else 0)
        finally:
            s.close()

    def viz(self, args=""):
        s, f = self.conn()
        try:
            s.sendall(("VIZ %s\n" % args if args else "VIZ\n").encode())
            while True:
                l = f.readline()
                if not l:
                    raise RuntimeError("engine closed the connection")
                if l.startswith("V "):
                    return json.loads(l[2:])
        finally:
            s.close()


def start(exe, args, env_extra, port, logpath):
    env = dict(os.environ)
    for k, v in env_extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    lf = open(logpath, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([exe] + args, stdout=lf, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 1200:
        if p.poll() is not None:
            raise RuntimeError("engine exited early, see " + logpath)
        try:
            with open(logpath, encoding="utf-8", errors="replace") as r:
                if "serve: listening" in r.read():
                    return p, time.time() - t0
        except OSError:
            pass
        time.sleep(2)
    p.kill()
    raise RuntimeError("engine did not start in 20 min, see " + logpath)


def stop(p):
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(30)
    time.sleep(5)  # let the driver release device memory


def check_slot(e, slot, gen_toks, req):
    """Validate one slot's exposed records against the request's tokens."""
    summ = e.viz()
    meta = L = None
    for s in summ["slots"]:
        if s["slot"] == slot:
            meta = s
    if not meta or meta["req"] != req:
        return dict(err="slot %d does not hold req %d" % (slot, req))
    L, E = summ["layers"], summ["experts"]
    n_prompt, n_tok = meta["n_prompt"], meta["n_tok"]
    frm = max(0, n_tok - 2000)  # stays inside the 2048-position ring
    items = []
    while frm < n_tok:
        d = e.viz("%d %d 128" % (slot, frm))
        if not d["items"]:
            break
        items += d["items"]
        frm = d["items"][-1]["p"] + 1
    fe = d["final_end"]
    tok = {it["p"]: it["t"] for it in items}
    r = dict(positions=0, routed=0, bad=0, nolayer=0, leaked=0, top=0, align=0, misalign=0)
    for i, t in enumerate(gen_toks):  # the log must hold exactly the generated tokens
        if tok.get(n_prompt + i, t) != t:
            r["bad"] += 1
    for it in items:
        p = it["p"]
        r["positions"] += 1
        if p >= fe:
            if "e" in it or "k" in it:
                r["leaked"] += 1  # the not-yet-final last token must stay hidden
            continue
        if "e" not in it:
            r["nolayer"] += L
            continue
        r["routed"] += 1
        for l in range(L):
            ids = it["e"][l * 10:l * 10 + 10]
            w = it["w"][l * 10:l * 10 + 10]
            if -1 in ids:
                r["nolayer"] += 1
                continue
            if len(set(ids)) != 10 or min(ids) < 0 or max(ids) >= E or abs(sum(w) - 255) > 12:
                r["bad"] += 1
        if "k" in it and p >= n_prompt - 1 and p + 1 in tok:
            r["top"] += 1
            if it["k"][0] == tok[p + 1]:
                r["align"] += 1
            else:
                r["misalign"] += 1
    return r


def data_ok(r, need_top):
    return ("err" not in r and r["bad"] == 0 and r["nolayer"] == 0 and r["leaked"] == 0
            and r["misalign"] == 0 and r["routed"] > 0 and (r["top"] >= need_top))


def run_cfg(name, exe, args, env, port, prompts, logdir, viz, conc):
    log("== %s: starting %s" % (name, os.path.basename(exe)))
    p, secs = start(exe, args, env, port, os.path.join(logdir, "viz_verify_%s.log" % name))
    log("   listening after %.0f s" % secs)
    e = Eng(port)
    out = dict(toks=[], tps=[], data=[], ok=True)
    try:
        e.gen(1, prompts[0], 16)  # warm-up
        req = 10
        for rep in range(2):
            ng = ms = 0
            toks = []
            for i, ids in enumerate(prompts):
                req += 1
                g = e.gen(req, ids, GEN_TOKENS)
                toks.append(g["toks"])
                ng += g["n_gen"]
                ms += g["decode_ms"]
                if viz and rep == 0:
                    best = None
                    for s in e.viz()["slots"]:
                        if s["req"] == req:
                            best = s["slot"]
                    r = check_slot(e, best if best is not None else 0, g["toks"], req)
                    r["prompt"] = i
                    out["data"].append(r)
                    if not data_ok(r, GEN_TOKENS - 2):
                        out["ok"] = False
            out["toks"].append(toks)
            out["tps"].append(1000.0 * ng / ms if ms else 0.0)
            log("   rep %d: decode %.2f tok/s (%d tokens)" % (rep + 1, out["tps"][-1], ng))
        if viz and conc:
            res = {}

            def one(k, rq, ids):
                res[k] = (rq, e.gen(rq, ids, GEN_TOKENS))

            th = [threading.Thread(target=one, args=(0, 901, prompts[1])),
                  threading.Thread(target=one, args=(1, 902, prompts[2]))]
            for t in th:
                t.start()
                time.sleep(0.02)
            for t in th:
                t.join()
            slots = {s["req"]: s["slot"] for s in e.viz()["slots"]}
            for k in (0, 1):
                rq, g = res[k]
                if rq not in slots:
                    r = dict(err="concurrent req %d not in any slot" % rq)
                else:
                    r = check_slot(e, slots[rq], g["toks"], rq)
                r["prompt"] = "conc%d" % k
                out["data"].append(r)
                if not data_ok(r, GEN_TOKENS - 2):
                    out["ok"] = False
                # concurrent decode must stay bit-identical to the serial run
                ref = out["toks"][0][1 + k]
                if g["toks"] != ref:
                    r["conc_diverged"] = True
            log("   concurrent pair checked on slots %s" % sorted(slots.values()))
    finally:
        stop(p)
    return out


def speed_ab(a, eargs, prompts, common, cfgs):
    """Alternate the configurations N times; each load runs 4 reps, the first
    is warm-up (graph capture / descriptor warm-up). Median of the rest."""
    import statistics
    cfgs = [c for c in cfgs if not c[0].endswith("1")]
    if os.environ.get("VIZ_PARTS_AB"):  # split the cost: routing only / top-5 only
        on = next(c for c in cfgs if c[0] == "on")
        cfgs = [c for c in cfgs if c[0] != "base"]
        cfgs.append(("onR", on[1], dict(on[2], QWENOX_VIZ_PARTS="1"), True, False))
        cfgs.append(("onT", on[1], dict(on[2], QWENOX_VIZ_PARTS="2"), True, False))
    tps = {c[0]: [] for c in cfgs}
    for rnd in range(a.speed_rounds):
        for name, exe, env, _, _ in cfgs:
            log("== round %d %s" % (rnd + 1, name))
            p, _ = start(exe, eargs, env, a.port, os.path.join(a.logdir, "viz_speed_%s.log" % name))
            e = Eng(a.port)
            try:
                e.gen(1, prompts[0], 16)
                req = 10
                for rep in range(4):
                    ng = ms = 0
                    for ids in prompts:
                        req += 1
                        g = e.gen(req, ids, GEN_TOKENS)
                        ng += g["n_gen"]
                        ms += g["decode_ms"]
                    v = 1000.0 * ng / ms
                    log("   rep %d %.2f%s" % (rep + 1, v, " (warm-up)" if rep == 0 else ""))
                    if rep:
                        tps[name].append(v)
            finally:
                stop(p)
    med = {n: statistics.median(v) for n, v in tps.items()}
    log("")
    for n, v in tps.items():
        log("%-4s median %.2f  min %.2f  max %.2f  (n=%d)" % (n, med[n], min(v), max(v), len(v)))
    slow = 100.0 * (med["off"] - med["on"]) / med["off"]
    log("viz on vs off: %+.2f%% slower (median)" % slow)
    if "base" in med:
        log("viz off vs base: %+.2f%% slower (median)" % (100.0 * (med["base"] - med["off"]) / med["base"]))
    ok = slow <= a.max_slow
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--port", type=int, default=8750)
    ap.add_argument("--prompts", default=os.path.join(HERE, "viz_verify_prompts.json"))
    ap.add_argument("--max-slow", type=float, default=2.0)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--speed-rounds", type=int, default=0,
                    help="speed A/B only: cycle base/off/on N times, 4 reps per load")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    eargs = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    eargs = [x.replace("@PORT@", str(a.port)) for x in eargs]
    prompts = json.load(open(a.prompts))
    os.makedirs(a.logdir, exist_ok=True)
    par = os.environ.get("QWENOX_PARALLEL", "1")
    common = {"QWENOX_KVSNAP": "0", "QWENOX_RCKPT_MAX": "0"}  # every turn computes
    one = dict(common, QWENOX_PARALLEL="1")
    cfgs = []
    if a.base:
        cfgs.append(("base", a.base, dict(common, QWENOX_VIZ=None), False, False))
    cfgs += [("off", a.engine, dict(common, QWENOX_VIZ=None), False, False),
             ("on", a.engine, dict(common, QWENOX_VIZ="1"), True, int(par) > 1)]
    if par != "1":
        if a.base and not a.speed_rounds:
            cfgs.append(("base1", a.base, dict(one, QWENOX_VIZ=None), False, False))
        cfgs.append(("off1", a.engine, dict(one, QWENOX_VIZ=None), False, False))
        cfgs.append(("on1", a.engine, dict(one, QWENOX_VIZ="1"), True, False))
    if a.speed_rounds:
        return speed_ab(a, eargs, prompts, common, cfgs)
    log("prompts: %s tokens; %d generated each; QWENOX_PARALLEL=%s" %
        ([len(p) for p in prompts], GEN_TOKENS, par))
    R = {}
    for name, exe, env, viz, conc in cfgs:
        R[name] = run_cfg(name, exe, eargs, env, a.port, prompts, a.logdir, viz, conc)

    fails = []
    for name, r in R.items():
        ref = R["off1" if name.endswith("1") else "off"]["toks"][0]
        for rep, t in enumerate(r["toks"]):
            for i in range(len(prompts)):
                if t[i] != ref[i]:
                    k = next((j for j in range(min(len(t[i]), len(ref[i]))) if t[i][j] != ref[i][j]),
                             min(len(t[i]), len(ref[i])))
                    fails.append("exact: %s rep %d prompt %d differs from %s at token %d" %
                                 (name, rep + 1, i, "off1" if name.endswith("1") else "off", k))
    if "off1" in R:
        same = R["off1"]["toks"][0] == R["off"]["toks"][0]
        log("note: PARALLEL=1 and PARALLEL=%s outputs %s (engine property, not a viz check)" %
            (par, "identical" if same else "differ"))
    best = {n: max(r["tps"]) for n, r in R.items()}
    slow = 100.0 * (best["off"] - best["on"]) / best["off"]
    log("")
    log("decode tok/s (best of 2): " + ", ".join("%s %.2f" % (n, v) for n, v in best.items()))
    log("viz on vs off: %+.2f%% slower (target <= 1%%, fail > %.1f%%)" % (slow, a.max_slow))
    if slow > a.max_slow:
        fails.append("speed: viz on is %.2f%% slower than off" % slow)
    if "base" in best:
        sb = 100.0 * (best["base"] - best["off"]) / best["base"]
        log("viz off vs base: %+.2f%% slower" % sb)
        if sb > a.max_slow:
            fails.append("speed: viz off is %.2f%% slower than base" % sb)
    for name in ("on", "on1"):
        for r in R.get(name, {"data": []})["data"]:
            log("data %-4s prompt %-6s %s" % (name, r.get("prompt"), json.dumps(
                {k: v for k, v in r.items() if k != "prompt"})))
            if not data_ok(r, GEN_TOKENS - 2):
                fails.append("data: %s prompt %s %s" % (name, r.get("prompt"), r))
            if r.get("conc_diverged"):
                fails.append("exact: %s concurrent prompt %s differs from serial" %
                             (name, r.get("prompt")))
    log("")
    for f in fails:
        log("FAIL " + f)
    log("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
