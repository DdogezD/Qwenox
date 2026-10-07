#!/usr/bin/env python3
"""
build_windowsxp_quality_overlay.py

Build a Halogen-compatible quality overlay for
windowsxp811203/Qwen3.8-Flash-Next-Abliterated (lambda ~= 1.5)
WITHOUT downloading the stock BF16 checkpoint.

Inputs:
  1) the user's WindowsXP BF16 checkpoint directory
  2) Halogen's official QUALITY sidecar:
       qwen38-flash-next-w4b.overlay.hgn
     (NOT overlay-speed.hgn)

Core idea
---------
For the 12 full-attention o_proj tensors Halogen stores as q8g64:

    W1 = WindowsXP abliterated BF16
    Q0 = official Halogen calibrated stock overlay tensor

Since WindowsXP applies a rank-1 residual-writer edit

    W1 = (I - lambda * r r^T) W0,

the leading left singular direction of (W1 - Q0) is a high-SNR estimate of r:
Q0 is only an 8-bit approximation of stock W0, while the ablation signal is
rank-1 and shared across all 12 layers.

After recovering r, stock weights for edited OUT-DIM residual writers can be
reconstructed approximately from W1 alone:

    W0 = (I - lambda P)^-1 W1
       = W1 + lambda/(1-lambda) * P W1,  P = r r^T

For lambda=1.5:
    W0 = W1 - 3 * P W1

Then transfer Halogen's calibrated quantization residual:

    Ecal   = Q0 - W0_est
    target = W1 + Ecal
           = Q0 + (W1 - W0_est)

and project target back to the SAME HGN dtype as the official quality tensor.

Safety rules
------------
* Untouched official-quality tensors are copied byte-for-byte.
* Edited quality tensors supported by this method are transformed:
    - self_attn.o_proj
    - linear_attn.out_proj
    - mlp.shared_expert.down_proj
* Any edited tensor that cannot be safely transformed is OMITTED from the
  output overlay, so the abliterated base remains authoritative.
* mtp.* is omitted by default. Use the MTP sidecar generated from the
  abliterated BF16 model by gfx1151-engine's current flashnext2hgn.py.
* Fused mlp.experts.down_proj is not in Halogen's quality sidecar and is NOT
  fixed by this script. A direction-preserving expert Q4 pass is a separate
  second stage.

This is a calibration-residual transplant, not a reproduction of Halogen's
unpublished activation-calibration pipeline. It preserves official calibrated
blobs exactly where possible and transfers their residual to modified layers.

Put this file in:
    gfx1151-engine/tools/build_windowsxp_quality_overlay.py

Requires:
    Python 3 + numpy
    tools/flashnext2hgn.py from the current gfx1151-engine checkout
"""

import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flashnext2hgn import (
    HgnWriter,
    VIS_PREFIX,
    map_lm_name,
    quant_q8g64,
    q8g64_size,
    q4cp_size,
    scan_model_dir,
)

HDR = 104
REC = 160

DT_BF16 = 0
DT_Q4CP = 5
DT_Q8G64 = 7

RE_SELF_O = re.compile(r"^layers\.(\d+)\.self_attn\.o_proj\.weight$")
RE_LINEAR_O = re.compile(r"^layers\.(\d+)\.linear_attn\.out_proj\.weight$")
RE_SHARED_DOWN = re.compile(r"^layers\.(\d+)\.mlp\.shared_expert\.down_proj\.weight$")
RE_FUSED_DOWN = re.compile(r"^layers\.(\d+)\.mlp\.experts\.down_proj\.weight$")

SUPPORTED_EDITED_QUALITY = (RE_SELF_O, RE_LINEAR_O, RE_SHARED_DOWN)


@dataclass
class HgnTensor:
    name: str
    dtype: int
    dims: tuple
    data_offset: int
    data_size: int
    extra: int


class HgnReader:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)

        if len(self.mm) < HDR:
            raise SystemExit(f"{self.path}: truncated HGN")

        magic, version, count, _, records_off, _, file_size = \
            struct.unpack_from("<4sIIIQQQ", self.mm, 0)

        if magic != b"HGN1":
            raise SystemExit(f"{self.path}: bad HGN magic {magic!r}")
        if version not in (1, 2):
            raise SystemExit(f"{self.path}: unsupported HGN version {version}")
        if file_size != len(self.mm):
            raise SystemExit(
                f"{self.path}: header file_size={file_size}, actual={len(self.mm)}"
            )

        self.version = version
        self.count = count
        self.records_offset = records_off
        raw_name = self.mm[0x28:0x68].split(b"\0", 1)[0]
        self.model_name = raw_name.decode("utf-8", errors="replace")

        self.order = []
        self.tensors = {}
        for i in range(count):
            off = records_off + i * REC
            rec = self.mm[off:off + REC]
            name = rec[:96].split(b"\0", 1)[0].decode("utf-8")
            dtype, ndims = struct.unpack_from("<II", rec, 96)
            dims4 = struct.unpack_from("<QQQQ", rec, 104)
            dims = tuple(dims4[:ndims])
            data_offset, data_size, extra = struct.unpack_from("<QQQ", rec, 136)
            t = HgnTensor(name, dtype, dims, data_offset, data_size, extra)
            if name in self.tensors:
                raise SystemExit(f"{self.path}: duplicate tensor {name}")
            self.order.append(name)
            self.tensors[name] = t

    def blob_bytes(self, t):
        # Return bytes, not memoryview, so mmap can always be closed on Windows.
        a = t.data_offset
        b = a + t.data_size
        return self.mm[a:b]

    def close(self):
        self.mm.close()
        os.close(self.fd)


def human(n):
    return f"{n / 2**30:.3f} GiB"


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("zero/nonfinite vector")
    return (v / n).astype(np.float32)


def mapped_lm_table(model_dir):
    raw = scan_model_dir(model_dir)
    out = {}
    for src_name, (st, dt, shape) in raw.items():
        if src_name.startswith(VIS_PREFIX):
            continue
        m = map_lm_name(src_name)
        if m is None:
            continue
        if m.startswith("layers.1.ple.ple_embedding.ngram_embedding.shard_"):
            continue
        out[m] = (st, dt, tuple(shape), src_name)
    return out


def load_meta(model_dir):
    p = os.path.join(model_dir, "ABLIT_META.json")
    if not os.path.exists(p):
        return None, set()
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    edited = set()
    for s in meta.get("stats", []):
        n = s.get("tensor")
        if not n:
            continue
        m = map_lm_name(n)
        if m is None:
            # Some metadata already stores mapped/non-prefixed names.
            m = n
        edited.add(m)
    return meta, edited


def fallback_edited(name):
    # Only used if ABLIT_META.json is absent.
    if name == "embed_tokens.weight":
        return True
    if name == "layers.1.ple.value_proj.weight":
        return True
    if name.startswith("mtp."):
        # Metadata is strongly preferred for MTP, but all mtp entries are
        # skipped from this overlay anyway.
        return True
    if RE_SELF_O.match(name) or RE_LINEAR_O.match(name):
        return True
    if RE_SHARED_DOWN.match(name) or RE_FUSED_DOWN.match(name):
        return True
    return False


def is_supported_edited_quality(name):
    return any(rx.match(name) for rx in SUPPORTED_EDITED_QUALITY)


def source_f32(item):
    st, _, shape, src = item
    return st.f32(src).reshape(-1, shape[-1])


def parse_q4cp(blob, rows, cols):
    size, stride = q4cp_size(rows, cols)
    if len(blob) != size:
        raise ValueError(f"q4cp size mismatch {len(blob)} != {size}")

    b = memoryview(blob)
    cb = np.frombuffer(b[:64], dtype="<f4").copy()
    codes_n = rows * cols // 2
    packed = np.frombuffer(
        b[64:64 + codes_n], dtype=np.uint8
    ).reshape(rows, cols // 2)

    g = cols // 32
    sr = np.frombuffer(
        b[64 + codes_n:], dtype=np.uint8
    ).reshape(rows, stride)
    scales16 = sr[:, :g * 2].copy().view(np.float16).reshape(rows, g)
    return cb, packed, scales16, stride


def dequant_q4cp(blob, rows, cols):
    cb, packed, scales16, _ = parse_q4cp(blob, rows, cols)
    g = cols // 32
    out = np.empty((rows, cols), np.float32)
    row_chunk = max(1, (1 << 24) // cols)

    for r0 in range(0, rows, row_chunk):
        r1 = min(rows, r0 + row_chunk)
        p = packed[r0:r1]
        nib = np.empty((r1 - r0, cols), np.uint8)
        nib[:, 0::2] = p & 0x0F
        nib[:, 1::2] = p >> 4

        scale = scales16[r0:r1].astype(np.float32)
        qn = cb[nib.reshape(r1 - r0, g, 32)]
        out[r0:r1] = (qn * scale[:, :, None]).reshape(r1 - r0, cols)

    return out


def dequant_q8g64(blob, rows, cols):
    g = cols // 64
    stride = cols + g * 4
    if len(blob) != rows * stride:
        raise ValueError(f"q8g64 size mismatch {len(blob)} != {rows * stride}")

    a = np.frombuffer(blob, dtype=np.uint8).reshape(rows, stride)
    codes = a[:, :cols].reshape(rows, g, 64).astype(np.float32)
    sm = a[:, cols:].copy().view(np.float16).reshape(rows, g, 2)
    scale = sm[:, :, 0].astype(np.float32)
    mn = sm[:, :, 1].astype(np.float32)
    return (codes * scale[:, :, None] + mn[:, :, None]).reshape(rows, cols)


def dequant_overlay(reader, t):
    cols = t.dims[-1]
    rows = math.prod(t.dims) // cols
    blob = reader.blob_bytes(t)
    if t.dtype == DT_Q4CP:
        return dequant_q4cp(blob, rows, cols)
    if t.dtype == DT_Q8G64:
        return dequant_q8g64(blob, rows, cols)
    if t.dtype == DT_BF16:
        u = np.frombuffer(blob, dtype=np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
    raise ValueError(f"{t.name}: unsupported overlay dtype {t.dtype}")


def top2_left_subspace(a, iters=10, seed=0):
    """
    Approximate top two left singular vectors/values using subspace iteration.
    a is [rows, sampled_cols], usually 2560 x <=512.
    """
    a = np.asarray(a, dtype=np.float32)
    rows = a.shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((rows, 2)).astype(np.float32)
    u, _ = np.linalg.qr(u, mode="reduced")

    for _ in range(iters):
        v = a.T @ u
        v, _ = np.linalg.qr(v, mode="reduced")
        u = a @ v
        u, _ = np.linalg.qr(u, mode="reduced")

    b = u.T @ a
    ub, s, _ = np.linalg.svd(b, full_matrices=False)
    u = u @ ub
    return u[:, 0].astype(np.float32), s.astype(np.float64)


def sample_columns(a, n):
    c = a.shape[1]
    if n <= 0 or n >= c:
        return a
    # Evenly spaced deterministic sample avoids depending on RNG/shard order.
    idx = np.linspace(0, c - 1, n, dtype=np.int64)
    return a[:, idx]


def recover_direction(official, ablit_table, anchor_cols, power_iters):
    anchors = []
    samples = []
    q0_samples = []

    names = []
    for name in official.order:
        t = official.tensors[name]
        m = RE_SELF_O.match(name)
        if m and t.dtype == DT_Q8G64 and name in ablit_table:
            names.append(name)

    if len(names) != 12:
        raise SystemExit(
            f"expected 12 q8 full-attention o_proj anchors in quality overlay, "
            f"found {len(names)}"
        )

    print("\nrecovering refusal direction from 12 Q8 o_proj anchors ...", flush=True)

    for i, name in enumerate(names):
        t = official.tensors[name]
        w1 = source_f32(ablit_table[name]).astype(np.float32, copy=False)
        q0 = dequant_overlay(official, t).reshape(w1.shape)
        if w1.shape[0] != 2560:
            raise SystemExit(f"{name}: expected output dim 2560, got {w1.shape}")

        d = w1 - q0
        ds = sample_columns(d, anchor_cols).copy()
        qs = sample_columns(q0, anchor_cols).copy()
        u1, s = top2_left_subspace(ds, iters=power_iters, seed=0x5100 + i)
        ratio = float(s[0] / max(s[1], 1e-30))
        anchors.append({
            "name": name,
            "u": u1,
            "s1": float(s[0]),
            "s2": float(s[1]),
            "s12": ratio,
        })
        samples.append(ds)
        q0_samples.append(qs)
        print(
            f"  {name}: sampled={ds.shape[1]} s1/s2={ratio:.4f}",
            flush=True
        )
        del w1, q0, d

    # Global direction from all 12 anchors simultaneously.
    all_d = np.concatenate(samples, axis=1)
    r, s_global = top2_left_subspace(
        all_d, iters=power_iters + 2, seed=0xA811203
    )
    r = normalize(r)

    # Sign-align per-anchor estimates against global vector and report cosine.
    cosines = []
    for a in anchors:
        c = float(np.dot(a["u"], r))
        if c < 0:
            a["u"] = -a["u"]
            c = -c
        a["cos_global"] = c
        cosines.append(c)

    # Weighted consensus is slightly more robust than a single concatenated
    # solve when one layer has unusually high q8 calibration residual.
    acc = np.zeros_like(r, dtype=np.float64)
    for a in anchors:
        # Weight by signal strength but cap domination via sqrt.
        acc += math.sqrt(max(a["s1"], 1e-30)) * a["u"].astype(np.float64)
    r_cons = normalize(acc)

    if np.dot(r_cons, r) < 0:
        r_cons = -r_cons

    for a in anchors:
        a["cos_consensus"] = abs(float(np.dot(a["u"], r_cons)))

    min_cos = min(a["cos_consensus"] for a in anchors)
    mean_cos = float(np.mean([a["cos_consensus"] for a in anchors]))
    median_s12 = float(np.median([a["s12"] for a in anchors]))
    global_s12 = float(s_global[0] / max(s_global[1], 1e-30))

    # Lambda self-calibration.  Under the exact edit model,
    #   D = W1 - Q0 = -lambda * r (r^T W0) + q8 noise,   r^T Q0 ~= r^T W0,
    # so the least-squares estimate per anchor is
    #   lambda_hat = -<r^T D, r^T Q0> / ||r^T Q0||^2 .
    # Sign-invariant in r; q8 noise is mostly orthogonal to r.
    r64 = r_cons.astype(np.float64)
    lam_ests = []
    for a, ds, qs in zip(anchors, samples, q0_samples):
        rd = r64 @ ds.astype(np.float64)
        rq = r64 @ qs.astype(np.float64)
        lam_i = -float(rd @ rq) / max(float(rq @ rq), 1e-30)
        a["lambda_est"] = lam_i
        lam_ests.append(lam_i)
    all_q = np.concatenate(q0_samples, axis=1)
    rd_all = r64 @ all_d.astype(np.float64)
    rq_all = r64 @ all_q.astype(np.float64)
    lambda_global = -float(rd_all @ rq_all) / max(float(rq_all @ rq_all), 1e-30)
    del all_q, q0_samples

    summary = {
        "anchor_count": len(anchors),
        "anchor_cols": anchor_cols,
        "power_iters": power_iters,
        "min_anchor_cos": min_cos,
        "mean_anchor_cos": mean_cos,
        "median_anchor_s1_s2": median_s12,
        "global_s1_s2": global_s12,
        "lambda_est_global": lambda_global,
        "lambda_est_median": float(np.median(lam_ests)),
        "lambda_est_min": float(np.min(lam_ests)),
        "lambda_est_max": float(np.max(lam_ests)),
    }

    print("\ndirection diagnostics:")
    print(f"  global s1/s2  : {global_s12:.6f}")
    print(f"  median s1/s2  : {median_s12:.6f}")
    print(f"  mean cosine   : {mean_cos:.6f}")
    print(f"  min cosine    : {min_cos:.6f}")
    print(f"  lambda est    : {lambda_global:.6f} "
          f"(median {summary['lambda_est_median']:.6f}, "
          f"range [{summary['lambda_est_min']:.6f}, "
          f"{summary['lambda_est_max']:.6f}])")

    del all_d, samples
    return r_cons.astype(np.float32), anchors, summary


def reconstruct_stock_out_writer(w1, r, lam):
    """
    Invert W1=(I-lambda rr^T)W0 for [hidden,in] output writers.
    """
    if abs(1.0 - lam) < 1e-8:
        raise ValueError("lambda=1 is singular and cannot be inverted")
    if w1.shape[0] != r.shape[0]:
        raise ValueError(
            f"output dimension mismatch: W {w1.shape}, r {r.shape}"
        )
    coeff = lam / (1.0 - lam)
    rt_w1 = r.astype(np.float64) @ w1.astype(np.float64)
    w0 = w1.astype(np.float64) + coeff * (
        r.astype(np.float64)[:, None] * rt_w1[None, :]
    )
    return w0.astype(np.float32)


def nearest_cb_indices(x, cb):
    j = np.searchsorted(cb, x)
    j = np.clip(j, 1, len(cb) - 1)
    left = cb[j - 1]
    right = cb[j]
    return np.where(x - left <= right - x, j - 1, j).astype(np.uint8)


def project_q4cp_fixed_codebook(target, official_blob, rows, cols, iters=4):
    """
    Project target to q4cp while keeping Halogen's official 16-value codebook.
    Codes/scales are optimized by coordinate descent from official scales.
    """
    cb, _, old_scales16, stride = parse_q4cp(
        official_blob, rows, cols
    )
    g = cols // 32
    out_codes = np.empty((rows, cols // 2), np.uint8)
    out_scales = np.zeros((rows, stride), np.uint8)

    row_chunk = max(1, (1 << 24) // cols)
    for r0 in range(0, rows, row_chunk):
        r1 = min(rows, r0 + row_chunk)
        tg = target[r0:r1].reshape(-1, g, 32).astype(
            np.float32, copy=False
        )

        # Start exactly from the official calibrated per-group scales.
        s = old_scales16[r0:r1].astype(np.float32)
        a = np.max(np.abs(tg), axis=2)
        s = np.where(np.abs(s) > 0, s, np.where(a > 0, a, 1.0))

        for _ in range(iters):
            idx = nearest_cb_indices(tg / s[:, :, None], cb)
            qn = cb[idx]
            num = np.sum(tg * qn, axis=2, dtype=np.float64)
            den = np.sum(qn * qn, axis=2, dtype=np.float64)
            sn = np.divide(
                num, den,
                out=s.astype(np.float64, copy=True),
                where=den > 1e-20,
            ).astype(np.float32)
            s = np.where(np.isfinite(sn) & (sn > 0), sn, np.abs(s))

        # Quantize scale to what HGN actually stores, then choose final codes.
        s16 = s.astype(np.float16)
        sf = s16.astype(np.float32)
        sf = np.where(sf == 0, 1.0, sf)
        idx = nearest_cb_indices(tg / sf[:, :, None], cb)

        flat = idx.reshape(r1 - r0, cols)
        out_codes[r0:r1] = flat[:, 0::2] | (flat[:, 1::2] << 4)

        sb = s16.reshape(r1 - r0, g).view(np.uint8).reshape(
            r1 - r0, g * 2
        )
        out_scales[r0:r1, :g * 2] = sb

    blob = (
        cb.astype("<f4", copy=False).tobytes()
        + out_codes.tobytes()
        + out_scales.tobytes()
    )
    expect, _ = q4cp_size(rows, cols)
    if len(blob) != expect:
        raise RuntimeError(f"q4 output size mismatch {len(blob)} != {expect}")
    return blob


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def tensor_metrics(name, w0, w1, q0, q1):
    delta = w1 - w0
    e0 = q0 - w0
    e1 = q1 - w1
    drift = e1 - e0
    return {
        "name": name,
        "delta_rms": rms(delta),
        "official_error_rms": rms(e0),
        "new_error_rms": rms(e1),
        "cal_residual_drift_rms": rms(drift),
        "drift_over_delta": rms(drift) / max(rms(delta), 1e-30),
        "drift_over_official_error": rms(drift) / max(rms(e0), 1e-30),
        "new_error_over_weight": rms(e1) / max(rms(w1), 1e-30),
        "drift_maxabs": float(np.max(np.abs(drift))),
    }


def selftest():
    rng = np.random.default_rng(1234)
    h, c = 256, 384
    lam = 1.5
    r0 = normalize(rng.standard_normal(h))
    w0 = rng.standard_normal((h, c), dtype=np.float32) * 0.03
    w1 = w0 - lam * r0[:, None] * (r0 @ w0)[None, :]

    # Simulate small stock quant/calibration noise.
    q0 = w0 + rng.standard_normal((h, c), dtype=np.float32) * 2e-4
    d = w1 - q0

    r, s = top2_left_subspace(d, iters=15, seed=99)
    rn = normalize(r)
    cos = abs(float(np.dot(rn, r0)))
    w0r = reconstruct_stock_out_writer(w1, rn, lam)
    rel = rms(w0r - w0) / rms(w0)

    # lambda self-calibration: -<r^T d, r^T q0>/||r^T q0||^2, sign-invariant.
    rd = rn.astype(np.float64) @ d.astype(np.float64)
    rq = rn.astype(np.float64) @ q0.astype(np.float64)
    lam_hat = -float(rd @ rq) / float(rq @ rq)

    print(f"selftest direction cosine={cos:.8f}")
    print(f"selftest reconstructed-stock relative-rms={rel:.8g}")
    print(f"selftest lambda estimate={lam_hat:.6f} (true {lam:g})")
    if cos < 0.995 or rel > 0.01 or abs(lam_hat - lam) > 0.02:
        raise SystemExit("SELFTEST FAILED")
    print("SELFTEST PASS")


def build(args):
    start = time.time()
    official = HgnReader(args.official_overlay)
    ablit = mapped_lm_table(args.ablit_dir)
    meta, edited_meta = load_meta(args.ablit_dir)

    if meta is not None:
        meta_lambda = meta.get("lambda")
        if args.lam is None and meta_lambda is not None:
            lam = float(meta_lambda)
        else:
            lam = 1.5 if args.lam is None else float(args.lam)
        print(
            f"ABLIT_META.json: edited={len(edited_meta)} "
            f"lambda={meta_lambda!r}"
        )
    else:
        lam = 1.5 if args.lam is None else float(args.lam)
        print("WARNING: ABLIT_META.json not found; using architecture patterns.")

    if abs(1.0 - lam) < 1e-8:
        raise SystemExit("lambda=1 cannot be inverted without stock BF16")

    print(f"using lambda={lam:g}")
    print(
        f"official overlay: {official.path} "
        f"({official.count} tensors, {human(os.path.getsize(official.path))})"
    )

    r, anchors, dir_summary = recover_direction(
        official, ablit, args.anchor_cols, args.power_iters
    )

    weak = (
        dir_summary["min_anchor_cos"] < args.min_anchor_cos
        or dir_summary["median_anchor_s1_s2"] < args.min_median_s12
    )
    if weak and not args.force_direction:
        official.close()
        raise SystemExit(
            "\nDIRECTION VALIDATION FAILED. "
            f"Need min cosine >= {args.min_anchor_cos:g} and median s1/s2 >= "
            f"{args.min_median_s12:g}. Re-run with larger --anchor-cols first; "
            "--force-direction exists only for deliberate experiments."
        )
    if weak:
        print("WARNING: forcing a weak direction estimate.")

    # Lambda self-calibration gate: the data itself must agree with the
    # assumed edit strength before we invert (I - lam P) with it.
    lam_est = dir_summary["lambda_est_global"]
    lam_dev = abs(lam_est - lam)
    print(f"lambda check: estimated {lam_est:.4f} vs configured {lam:g} "
          f"(dev {lam_dev:.4f}, tol {args.lambda_tol:g})")
    if lam_dev > args.lambda_tol:
        if not args.force_direction:
            official.close()
            raise SystemExit(
                "\nLAMBDA VALIDATION FAILED. "
                f"Estimated lambda {lam_est:.4f} deviates from configured "
                f"{lam:g} by {lam_dev:.4f} > {args.lambda_tol:g}. "
                "The recovered edit strength contradicts the assumption; "
                "check ABLIT_META.json / --lambda rather than proceeding "
                "on a wrong inversion. --force-direction bypasses this only "
                "for deliberate experiments."
            )
        print("WARNING: forcing past lambda mismatch.")

    # Classify every official-overlay entry before writing anything.
    plan = []
    counts = {
        "copy": 0,
        "xfer": 0,
        "skip_mtp": 0,
        "skip_edited_unsupported": 0,
        "missing_ablit": 0,
    }

    for name in official.order:
        t = official.tensors[name]

        if name.startswith("mtp.") and not args.include_mtp:
            plan.append(("skip_mtp", name, t))
            counts["skip_mtp"] += 1
            continue

        edited = (
            name in edited_meta if meta is not None
            else fallback_edited(name)
        )

        if name not in ablit:
            # Untouched copies do not strictly require BF16 source, but if
            # metadata says edited we must never copy stock over the ablit base.
            if edited:
                plan.append(("missing_ablit", name, t))
                counts["missing_ablit"] += 1
            else:
                plan.append(("copy", name, t))
                counts["copy"] += 1
            continue

        if not edited:
            plan.append(("copy", name, t))
            counts["copy"] += 1
            continue

        if is_supported_edited_quality(name):
            if t.dtype not in (DT_Q4CP, DT_Q8G64):
                plan.append(("skip_edited_unsupported", name, t))
                counts["skip_edited_unsupported"] += 1
            else:
                plan.append(("xfer", name, t))
                counts["xfer"] += 1
        else:
            # Important: do not copy official stock versions of edited tensors
            # such as embed_tokens/PLE/expert/MTP into the abliterated model.
            plan.append(("skip_edited_unsupported", name, t))
            counts["skip_edited_unsupported"] += 1

    print("\noutput plan:")
    for k, v in counts.items():
        print(f"  {k:24s}: {v}")

    xfer_names = [n for a, n, _ in plan if a == "xfer"]
    print("\nmodified official-quality tensors to transfer:")
    for n in xfer_names:
        print(" ", n)

    if len(xfer_names) != 96:
        print(
            f"WARNING: expected ~96 WindowsXP-edited quality tensors, "
            f"found {len(xfer_names)}. The manifest will record the exact set."
        )

    out_path = os.path.abspath(args.out)
    direction_path = (
        os.path.abspath(args.direction_out)
        if args.direction_out
        else out_path + ".direction.npy"
    )
    manifest_path = (
        os.path.abspath(args.manifest)
        if args.manifest
        else out_path + ".manifest.json"
    )
    report_path = (
        os.path.abspath(args.report)
        if args.report
        else out_path + ".report.tsv"
    )

    # Save the direction even on dry-run: it is the expensive/reusable result.
    os.makedirs(os.path.dirname(direction_path), exist_ok=True)
    np.save(direction_path, r.astype(np.float32))
    print(f"\ndirection -> {direction_path}")

    manifest = {
        "method": "Halogen official quality residual transplant onto WindowsXP abliterated BF16",
        "official_overlay": official.path,
        "ablit_dir": os.path.abspath(args.ablit_dir),
        "lambda": lam,
        "direction_summary": dir_summary,
        "anchors": [
            {
                "name": a["name"],
                "s1": a["s1"],
                "s2": a["s2"],
                "s1_s2": a["s12"],
                "cos_consensus": a["cos_consensus"],
                "lambda_est": a["lambda_est"],
            }
            for a in anchors
        ],
        "plan_counts": counts,
        "xfer_tensors": xfer_names,
        "skipped_edited_tensors": [
            n for a, n, _ in plan
            if a in ("skip_edited_unsupported", "missing_ablit")
        ],
        "note": (
            "Untouched quality tensors are byte-identical copies. "
            "Supported edited quality tensors use reconstructed-stock "
            "calibration-residual transplant. Fused expert down_proj is not "
            "handled here and requires the separate direction-preserving Q4 stage."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest  -> {manifest_path}")

    if args.dry_run:
        official.close()
        print("\nDRY RUN — overlay not written.")
        return

    entries = [(a, n, t) for a, n, t in plan if a in ("copy", "xfer")]
    writer = HgnWriter(
        out_path,
        args.name or "qwen38-flash-next-windowsxp-quality"
    )
    for _, name, t in entries:
        writer.plan(name, t.dtype, t.dims, t.data_size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer.begin()

    reports = []
    for i, (action, name, t) in enumerate(entries, 1):
        ts = time.time()
        oblob = official.blob_bytes(t)

        if action == "copy":
            writer.write(oblob)
            print(
                f"[{i:3d}/{len(entries)}] COPY    dt{t.dtype} {name}",
                flush=True
            )
            continue

        w1 = source_f32(ablit[name]).astype(np.float32, copy=False)
        if w1.shape[0] != r.shape[0]:
            raise SystemExit(
                f"{name}: supported edited writer has shape {w1.shape}; "
                f"expected output dim {r.shape[0]}"
            )

        w0 = reconstruct_stock_out_writer(w1, r, lam)
        q0 = dequant_overlay(official, t).reshape(w1.shape)

        # Transfer official calibrated quantization residual:
        # target - W1 == Q0 - W0_est
        target = q0 + (w1 - w0)

        rows, cols = w1.shape
        if t.dtype == DT_Q4CP:
            blob = project_q4cp_fixed_codebook(
                target, oblob, rows, cols, args.q4_iters
            )
            q1 = dequant_q4cp(blob, rows, cols)
            tag = "XFER-Q4"
        elif t.dtype == DT_Q8G64:
            blob = quant_q8g64(target)
            if len(blob) != q8g64_size(rows, cols):
                raise RuntimeError(f"{name}: q8 output size mismatch")
            q1 = dequant_q8g64(blob, rows, cols)
            tag = "XFER-Q8"
        else:
            raise RuntimeError("unreachable dtype")

        if len(blob) != t.data_size:
            raise SystemExit(
                f"{name}: transformed blob size {len(blob)} != "
                f"official {t.data_size}"
            )

        writer.write(blob)
        m = tensor_metrics(name, w0, w1, q0, q1)
        reports.append(m)

        print(
            f"[{i:3d}/{len(entries)}] {tag:7s} {name} "
            f"drift/delta={m['drift_over_delta']:.5g} "
            f"drift/e0={m['drift_over_official_error']:.5g} "
            f"{time.time() - ts:.1f}s",
            flush=True,
        )

        del w0, q0, target, q1, blob

    writer.finish()
    official.close()

    keys = [
        "name",
        "delta_rms",
        "official_error_rms",
        "new_error_rms",
        "cal_residual_drift_rms",
        "drift_over_delta",
        "drift_over_official_error",
        "new_error_over_weight",
        "drift_maxabs",
    ]
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(keys) + "\n")
        for m in reports:
            f.write("\t".join(
                m[k] if isinstance(m[k], str) else f"{m[k]:.10g}"
                for k in keys
            ) + "\n")

    print(f"\nwrote     -> {out_path} ({human(os.path.getsize(out_path))})")
    print(f"report    -> {report_path}")
    print(f"elapsed   -> {(time.time() - start) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(
        description="Build a WindowsXP λ=1.5 quality overlay from the "
                    "official Halogen quality sidecar without stock BF16."
    )
    ap.add_argument("--ablit-dir",
                    help="WindowsXP BF16 checkpoint directory")
    ap.add_argument("--official-overlay",
                    help="qwen38-flash-next-w4b.overlay.hgn (QUALITY, not speed)")
    ap.add_argument("--out",
                    help="output .overlay.hgn")
    ap.add_argument("--lambda", dest="lam", type=float,
                    help="override lambda; default reads ABLIT_META.json, else 1.5")
    ap.add_argument("--anchor-cols", type=int, default=512,
                    help="columns sampled per Q8 o_proj for direction recovery")
    ap.add_argument("--power-iters", type=int, default=12,
                    help="subspace iterations for direction recovery")
    ap.add_argument("--min-anchor-cos", type=float, default=0.95,
                    help="fail if any anchor direction cosine is below this")
    ap.add_argument("--min-median-s12", type=float, default=2.0,
                    help="fail if median anchor s1/s2 is below this")
    ap.add_argument("--lambda-tol", type=float, default=0.05,
                    help="fail if the data-estimated lambda deviates from the "
                         "configured lambda by more than this")
    ap.add_argument("--force-direction", action="store_true",
                    help="continue despite weak direction diagnostics")
    ap.add_argument("--q4-iters", type=int, default=4,
                    help="fixed-codebook q4 code/scale iterations")
    ap.add_argument("--include-mtp", action="store_true",
                    help="include official mtp.*; normally DO NOT use this")
    ap.add_argument("--direction-out")
    ap.add_argument("--manifest")
    ap.add_argument("--report")
    ap.add_argument("--name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")

    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    missing = [
        k for k in ("ablit_dir", "official_overlay", "out")
        if getattr(args, k) is None
    ]
    if missing:
        ap.error("required unless --selftest: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    if args.q4_iters < 1:
        ap.error("--q4-iters must be >= 1")
    if args.anchor_cols < 32:
        ap.error("--anchor-cols must be >= 32")

    build(args)


if __name__ == "__main__":
    main()
