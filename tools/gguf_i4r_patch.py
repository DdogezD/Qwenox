#!/usr/bin/env python3
"""gguf_i4r_patch.py — build an i4r routed-expert patch from a GGUF file.

Keeps the GGUF "text body" untouched: the patch contains only the 96 routed
expert tensors (fused gate_up_proj + down_proj) and their 192 su/svh sidecars,
in hgn v2 format. Load it as an overlay with

  QWENOX_GGUF=<shard1>.gguf QWENOX_GGUF_DENSE=1 QWENOX_GGUF_EXPERTS=0 \
  QWENOX_MOE_A4=3 qwenox <shard1>.gguf patch.hgn ...

Pipeline per 64-expert chunk:
  GGUF IQ blocks --(tools/gguf_expert_rows, engine's gguf::dequant_row)-->
  f32 [n,O,K] --(H128 rotation, su/svh from a reference hgn)-->
  i4r codes + per-128 f16 scales --(pwrite)--> [codes][scales] payload.

The sidecars are copied from a reference hgn (an existing v2 hgn or the
expert patch extracted from it). For this model's experts they are +/-1;
any consistent set would work, but reusing the official ones keeps the
rotation basis identical to the hgn v2 runtime.

Usage:
  PYTHONPATH=/home/mark/Workspace/pylib python3 tools/gguf_i4r_patch.py \
    --gguf models/<shard1>.gguf \
    --sidecar-src /path/to/qwen38-flash-next-v2.hgn \
    --out /path/to/i4r-expert-patch.hgn --workers 8

  # smoke test: one layer, still produces a valid (but non-runnable) patch
  ... --layers 0 --out /tmp/layer0.hgn

Requires: numpy, g++ (the helper is built on demand), and about 2-4 GiB RAM
per worker for the gate_up chunks.
"""
import argparse, json, os, struct, subprocess, sys, time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "v2repro"))
import hgnc  # noqa: E402
import v2enc  # noqa: E402

HDR, ENT = hgnc.HDR, hgnc.ENT
CHUNK_DEFAULT = 64

# per-role (O, K) for this architecture
ROLE_SHAPE = {"gate_up": (1280, 2560), "down": (2560, 640)}

SIDECAR_SUFFIX = {
    "gate_up": ("su", "gate_up_proj.svh"),
    "down": ("down_proj.suh", "down_proj.svh"),
}

_G = {}


def i4r_sizes(E, O, K):
    codes = E * O * K // 2
    scales = E * O * (K // 128) * 2
    return codes, scales


def parse_layers(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def build_entries(layers, E, sidecar_src):
    """Return ordered Entry list for the patch, with sidecar payload locations."""
    _, _, _, src_ents = hgnc.read(sidecar_src)
    src = {e.name: e for e in src_ents}
    out = []
    side = []  # (entry, source_entry)

    def add_side(name):
        if name not in src:
            raise KeyError(f"sidecar {name} not in {sidecar_src}")
        e = hgnc.Entry()
        s = src[name]
        e.name, e.store, e.shape, e.size = s.name, s.store, s.shape, s.size
        e.checksum = 0
        e.variant = 0
        out.append(e)
        side.append((e, s))

    for l in layers:
        for role in ("gate_up", "down"):
            O, K = ROLE_SHAPE[role]
            codes, scales = i4r_sizes(E, O, K)
            e = hgnc.Entry()
            e.name = f"layers.{l}.mlp.experts.{role}_proj.weight"
            e.store = 23
            e.shape = (E, O, K)
            e.size = codes + scales
            e.checksum = 0
            e.variant = 0
            out.append(e)
            add_side(f"layers.{l}.mlp.experts.{SIDECAR_SUFFIX[role][0]}")
            add_side(f"layers.{l}.mlp.experts.{SIDECAR_SUFFIX[role][1]}")
    out.sort(key=lambda e: e.name)
    return out, side


def load_sidecars(sidecar_src, layers):
    """f32 su/sv per (layer, role), exactly like v2_encode.py's sym_f32."""
    _, _, _, ents = hgnc.read(sidecar_src)
    by = {e.name: e for e in ents}
    res = {}
    for l in layers:
        for role in ("gate_up", "down"):
            su, sv = SIDECAR_SUFFIX[role]
            a = by[f"layers.{l}.mlp.experts.{su}"]
            b = by[f"layers.{l}.mlp.experts.{sv}"]
            res[(l, role)] = (
                np.frombuffer(bytes(hgnc.payload(sidecar_src, a)), dtype="<f2").astype(np.float32),
                np.frombuffer(bytes(hgnc.payload(sidecar_src, b)), dtype="<f2").astype(np.float32),
            )
    return res


def read_exact(pipe, n):
    buf = bytearray(n)
    mv = memoryview(buf)
    off = 0
    while off < n:
        k = pipe.readinto(mv[off:])
        if not k:
            raise RuntimeError(f"helper closed early: {off}/{n} bytes")
        off += k
    return buf


def worker(task):
    layer, role, e0, e1 = task
    O, K = ROLE_SHAPE[role]
    n = e1 - e0
    expected = n * O * K * 4
    p = subprocess.Popen([_G["helper"], _G["gguf"], str(layer), role,
                          str(e0), str(e1)], stdout=subprocess.PIPE)
    try:
        raw = read_exact(p.stdout, expected)
    finally:
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"helper failed on {task} rc={p.returncode}")
    W = np.frombuffer(raw, dtype="<f4").reshape(n, O, K)
    su, sv = _G["side"][(layer, role)]
    X = v2enc.rotate(W, 1.0 / su, 1.0 / sv)
    X[~np.isfinite(X)] = 0.0
    pk, sc = v2enc.i4r_encode_rot(X)

    # optional loss self-check on the first expert of the chunk
    if _G.get("rel_l2", True):
        Xh = v2enc.i4r_decode_rot(pk[:1], sc[:1])
        Wh = v2enc.unrotate(Xh, su, sv)
        rl = float(np.linalg.norm(Wh - W[:1]) / np.linalg.norm(W[:1]))
    else:
        rl = 0.0

    codes, _ = i4r_sizes(_G["E"], O, K)
    base = _G["payload_off"][f"layers.{layer}.mlp.experts.{role}_proj.weight"]
    fd = os.open(_G["out"], os.O_WRONLY)
    try:
        os.pwrite(fd, pk.tobytes(), base + e0 * O * K // 2)
        os.pwrite(fd, sc.astype("<f2").tobytes(),
                  base + codes + e0 * O * (K // 128) * 2)
    finally:
        os.close(fd)
    del W, X
    return (layer, role, e0, e1, rl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="first GGUF shard")
    ap.add_argument("--sidecar-src", required=True,
                    help="hgn file providing the 192 su/svh sidecars (v2 hgn or extracted patch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--chunk", type=int, default=CHUNK_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--helper", default=os.path.join(HERE, "gguf_expert_rows"))
    ap.add_argument("--no-rel-l2", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    E = args.experts
    if not os.path.exists(args.helper):
        print(f"building {args.helper}", flush=True)
        subprocess.run(["g++", "-O2", "-std=c++17", "-Isrc",
                        "tools/gguf_expert_rows.cpp", "-o", args.helper],
                       cwd=REPO, check=True)

    entries, side = build_entries(layers, E, args.sidecar_src)
    n = len(entries)
    data_off = hgnc.align64(HDR + n * ENT)
    off = data_off
    payload_off = {}
    for e in entries:
        e.offset = off
        payload_off[e.name] = off
        off += hgnc.align64(e.size)
    total = off
    print(f"patch: {len(layers)} layers, {n} tensors, {total/2**30:.3f} GiB", flush=True)
    if args.dry_run:
        for e in entries:
            print(f"  {e.name:55s} store={e.store:2d} size={e.size}")
        return

    # create header/table, sidecar payloads, then reserve the i4r regions
    with open(args.out, "wb") as f:
        hgnc.write_header(f, 2, n, data_off, total, "gguf-i4r-expert-patch")
        for e in entries:
            f.write(e.pack())
        if f.tell() < data_off:
            f.write(b"\0" * (data_off - f.tell()))
        for e, s in side:
            f.seek(e.offset)
            f.write(bytes(hgnc.payload(args.sidecar_src, s)))
        f.truncate(total)
    print(f"wrote table + sidecars -> {args.out}", flush=True)

    side_f32 = load_sidecars(args.sidecar_src, layers)
    tasks = []
    for l in layers:
        for role in ("gate_up", "down"):
            for e0 in range(0, E, args.chunk):
                tasks.append((l, role, e0, min(e0 + args.chunk, E)))
    tasks.sort()
    print(f"converting {len(tasks)} chunks with {args.workers} workers", flush=True)

    global _G
    _G = dict(helper=args.helper, gguf=args.gguf, out=args.out, E=E,
              side=side_f32, payload_off=payload_off, rel_l2=not args.no_rel_l2)
    t0 = time.time()
    rels = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, tasks)):
            rels.append(r[4])
            if (i + 1) % 16 == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] {el:.0f}s mean rel-L2 "
                      f"{np.mean(rels):.5f} max {max(rels):.5f}", flush=True)
    if not args.no_rel_l2:
        print(f"rel-L2: mean {np.mean(rels):.5f} max {max(rels):.5f}")

    # verify the table and size
    v, ident, _, ents = hgnc.read(args.out)
    assert v == 2 and len(ents) == n, (v, len(ents), n)
    assert os.path.getsize(args.out) == total, (os.path.getsize(args.out), total)
    print(f"done: {args.out} {total} B = {total/2**30:.3f} GiB, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
