#!/usr/bin/env python3
"""bench_i4r_patch.py — prefill/decode sweep for the GGUF body + i4r patch.

Configs:
  gguf   : pure GGUF (original IQ3_S experts, GGUF LUT path)
  a0     : i4r patch, MOE_A4=0   (conversion only)
  a2     : i4r patch, MOE_A4=2   (down-only int4)
  a3     : i4r patch, MOE_A4=3   (gate/up + down int4)

Usage:
  python3 tools/bench_i4r_patch.py --mode prefill --reps 2
  python3 tools/bench_i4r_patch.py --mode decode --lengths 8k --reps 2

The production env is read from --env-file (the `ENV ...` lines printed by
`bash start_gguf.sh --check`). QWENOX_KVSNAP / PLE_URING / PARALLEL are
overridden for a single-stream, no-snapshot benchmark.
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

LEN_LABEL = {"1k": 1024, "4k": 4096, "8k": 8192, "16k": 16384,
             "32k": 32768, "64k": 65536, "128k": 131072}

PREFILL_RE = re.compile(r"prefill: (\d+) tokens in ([\d.]+) s")
DECODE_RE = re.compile(r"decode: (\d+) tokens in ([\d.]+) ms = ([\d.]+) tok/s")


def read_env(path):
    env = {}
    if not path or not os.path.exists(path):
        return env
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("ENV "):
            kv = line[4:].strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
    return env


def parse_log(text):
    toks = 0
    secs = 0.0
    for m in PREFILL_RE.finditer(text):
        toks += int(m.group(1))
        secs += float(m.group(2))
    prefill = toks / secs if secs > 0 else None
    dec = None
    m = DECODE_RE.search(text)
    if m:
        dec = float(m.group(3))
    return prefill, dec, toks, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--gguf", default=os.path.join(REPO, "models",
                    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"))
    ap.add_argument("--patch", default="/home/mark/Workspace/Work/devtools/"
                    "qwen38-swift-i4r-expert-patch.hgn")
    ap.add_argument("--bin", default="/tmp/qwenox-engine-iq3-mt")
    ap.add_argument("--env-file", default="/tmp/gguf_check.txt")
    ap.add_argument("--tokens-dir", default="/tmp/i4r_bench_tokens")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_i4r_results.tsv"))
    ap.add_argument("--logdir", default="/tmp/i4r_bench_logs")
    ap.add_argument("--lengths", default="1k,4k,8k,16k,32k,64k,128k")
    ap.add_argument("--configs", default="gguf,a0,a2,a3")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gen", type=int, default=1)
    ap.add_argument("--maxctx", type=int, default=0,
                    help="0 = per-length (N + gen + 2K); >0 = fixed")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.mode == "decode":
        if args.gen == 1:
            args.gen = 64
        if args.lengths == "1k,4k,8k,16k,32k,64k,128k":
            args.lengths = "8k"
        if args.configs == "gguf,a0,a2,a3":
            args.configs = "gguf,a0"

    os.makedirs(args.logdir, exist_ok=True)
    base_env = read_env(args.env_file)
    base_env.update({
        "QWENOX_GGUF": args.gguf,
        "QWENOX_GGUF_DENSE": "1",
        "QWENOX_GGUF_MTP": "",
        "QWENOX_KVSNAP": "0",
        "QWENOX_PLE_URING": "0",
        "QWENOX_PARALLEL": "1",
        "QWENOX_PREFILL_CHUNK": "16384",
        "QWENOX_NOWARMUP": "1",
        "QWENOX_RCKPT_MAX": "0",
    })

    lengths = [s.strip() for s in args.lengths.split(",") if s.strip()]
    configs = [s.strip() for s in args.configs.split(",") if s.strip()]
    if not all(l in LEN_LABEL for l in lengths):
        sys.exit(f"bad length label(s): {lengths}")
    if not all(c in ("gguf", "a0", "a2", "a3") for c in configs):
        sys.exit(f"bad config(s): {configs}")

    rows = []
    t_start = time.time()
    for cfg in configs:
        for lab in lengths:
            tokfile = os.path.join(args.tokens_dir, f"{lab}.tokens")
            if not os.path.exists(tokfile):
                sys.exit(f"missing {tokfile}")
            for rep in range(1, args.reps + 1):
                env = dict(base_env)
                cmd = [args.bin, args.gguf]
                if cfg == "gguf":
                    env.pop("QWENOX_GGUF_EXPERTS", None)
                    env["QWENOX_MOE_A4"] = "0"
                else:
                    env["QWENOX_GGUF_EXPERTS"] = "0"
                    env["QWENOX_MOE_A4"] = cfg[1]
                    cmd.append(args.patch)
                N = LEN_LABEL[lab]
                ctx = args.maxctx or max(4096, N + max(args.gen, 1) + 2048)
                cmd += ["--tokens-file", tokfile, "--gen", str(args.gen),
                        "--maxctx", str(ctx)]
                log = os.path.join(args.logdir, f"{args.mode}_{cfg}_{lab}_r{rep}.log")
                t0 = time.time()
                with open(log, "w") as f:
                    p = subprocess.run(cmd, env=env, stdout=f,
                                       stderr=subprocess.STDOUT,
                                       timeout=args.timeout)
                wall = time.time() - t0
                text = open(log, errors="replace").read()
                pre, dec, toks, secs = parse_log(text)
                row = dict(mode=args.mode, cfg=cfg, len=lab, rep=rep,
                           rc=p.returncode, tokens=toks, secs=secs,
                           prefill=pre, decode=dec, wall=wall, log=log, ctx=ctx)
                rows.append(row)
                val = pre if args.mode == "prefill" else dec
                print(f"[{cfg:4s} {lab:4s} r{rep}] rc={p.returncode} "
                      f"prefill={pre and round(pre,1)} decode={dec and round(dec,1)} "
                      f"tokens={toks} wall={wall:.0f}s", flush=True)

    with open(args.out, "w") as f:
        f.write("mode\tcfg\tlen\trep\trc\ttokens\tsecs\tprefill_tok_s\tdecode_tok_s\twall_s\tctx\tlog\n")
        for r in rows:
            f.write(f"{r['mode']}\t{r['cfg']}\t{r['len']}\t{r['rep']}\t{r['rc']}\t"
                    f"{r['tokens']}\t{r['secs']:.6f}\t"
                    f"{r['prefill'] if r['prefill'] is not None else ''}\t"
                    f"{r['decode'] if r['decode'] is not None else ''}\t"
                    f"{r['wall']:.1f}\t{r['ctx']}\t{r['log']}\n")
    print(f"\nwrote {args.out}  total {time.time()-t_start:.0f}s")

    # summary: best (max) per config/length
    print("\nSUMMARY (best of reps)")
    hdr = f"{'len':>5s}" + "".join(f"{c:>12s}" for c in configs)
    print(hdr)
    for lab in lengths:
        line = f"{lab:>5s}"
        for cfg in configs:
            vals = [(r["prefill"] if args.mode == "prefill" else r["decode"])
                    for r in rows if r["cfg"] == cfg and r["len"] == lab
                    and r["rc"] == 0]
            vals = [v for v in vals if v]
            line += f"{max(vals):12.1f}" if vals else f"{'FAIL':>12s}"
        print(line)


if __name__ == "__main__":
    main()
