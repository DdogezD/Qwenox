# -*- coding: utf-8 -*-
"""ple_ngram_bf16.py — repack the 128 BF16 PLE n-gram shards into a standalone
hgn file (dtype 0), the bf16 counterpart of qwen38-flash-next-ngram.hgn (fp8,
dtype 10, 47.68 GiB). Pure stdlib (no numpy): a byte-exact copy of the source
shards, no requantization.

Output layout (HGN1 v2, src/hgn.h): one tensor
  layers.1.ple.ngram_embedding.weight  dtype 0  dims (128, R, 160)
payload = the 128 shards' raw bf16 bytes stacked in shard order (no trailing
global scale — scale 1.0 is implicit). Size = 128 * R * 160 * 2 ≈ 95.4 GiB for
R = 2,500,012.

The engine reads it via NGRAM_FILE exactly like the fp8 file (row bytes 320,
dequant on GPU / host without a scale).

Usage: ple_ngram_bf16.py MODEL_DIR -o OUT [--dry-run]
"""
import argparse
import json
import os
import shutil
import struct
import time

HDR, ENT = 0x68, 0xA0
MAGIC = 0x314E4748  # "HGN1"
TABLE = "layers.1.ple.ngram_embedding.weight"
SHARD_PAT = ("model.language_model.layers.1.ple.ple_embedding."
             "ngram_embedding.shard_%d.weight")
BUF = 16 << 20
UNIT = {"BF16": 2, "F16": 2, "I64": 8, "F32": 4, "I32": 4}


def align64(x):
    return (x + 63) // 64 * 64


def locate(model_dir, name):
    """-> (shard_path, payload_offset, shape, dtype) for a safetensors tensor."""
    with open(os.path.join(model_dir, "model.safetensors.index.json"),
              encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    p = os.path.join(model_dir, wm[name])
    with open(p, "rb") as fh:
        hn = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hn))
    spec = hdr[name]
    shape = tuple(spec["shape"])
    start, end = spec["data_offsets"]
    n = 1
    for d in shape:
        n *= d
    assert end - start == n * UNIT[spec["dtype"]], name
    return p, 8 + hn + start, shape, spec["dtype"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="HF safetensors directory (bf16 shards)")
    ap.add_argument("-o", "--out", required=True, help="output .hgn path")
    ap.add_argument("--ident", default="qwen3.8-flash-next-ngram-bf16")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = None
    shards = []
    for s in range(128):
        name = SHARD_PAT % s
        p, off, shape, dt = locate(args.model_dir, name)
        if dt != "BF16" or len(shape) != 2 or shape[1] != 160:
            raise SystemExit(f"{name}: expected BF16 [R,160], got {dt} {shape}")
        if rows is None:
            rows = shape[0]
        elif shape[0] != rows:
            raise SystemExit(f"{name}: row count {shape[0]} != {rows}")
        shards.append((p, off))
    size = 128 * rows * 160 * 2
    doff = align64(HDR + ENT)
    fsize = align64(doff + size)
    print(f"PLE bf16 table: 128 x [{rows}, 160] = {size / 2**30:.2f} GiB -> "
          f"{args.out}", flush=True)
    if args.dry_run:
        print("DRY RUN — nothing written")
        return
    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    free = shutil.disk_usage(outdir).free
    if free < fsize + (1 << 30):
        raise SystemExit(f"insufficient space: free {free / 2**30:.0f} GiB")

    t0 = time.time()
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"\0" * doff)
        done = 0
        for s, (p, off) in enumerate(shards):
            left = rows * 160 * 2
            with open(p, "rb") as fs:
                fs.seek(off)
                while left:
                    data = fs.read(min(BUF, left))
                    if not data:
                        raise IOError(f"short read {p} (shard {s})")
                    f.write(data)
                    left -= len(data)
                    done += len(data)
            if (s + 1) % 8 == 0:
                print(f"  [{s + 1}/128] {done / 2**30:.1f} GiB "
                      f"{time.time() - t0:.0f}s", flush=True)
        pad = fsize - (doff + size)
        if pad:
            f.write(b"\0" * pad)
        # header + record (back-filled over the leading zeros)
        f.seek(0)
        idb = args.ident.encode()
        assert len(idb) <= 64
        f.write(struct.pack("<IIIIQQQ", MAGIC, 2, 1, 0, HDR, doff, fsize))
        f.write(idb + b"\0" * (64 - len(idb)))
        rec = bytearray(ENT)
        nb = TABLE.encode()
        rec[: len(nb)] = nb
        struct.pack_into("<II", rec, 0x60, 0, 3)  # dtype 0 = bf16, ndims 3
        struct.pack_into("<4q", rec, 0x68, 128, rows, 160, 0)
        struct.pack_into("<QQII", rec, 0x88, doff, size, 0, 0)
        f.write(rec)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.out)
    el = time.time() - t0
    print(f"done: {args.out} {os.path.getsize(args.out) / 2**30:.2f} GiB, "
          f"{el:.0f}s ({size / el / 2**30:.2f} GiB/s)", flush=True)


if __name__ == "__main__":
    main()
