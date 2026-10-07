#!/usr/bin/env python3
"""
flashnext2hgn_windowsxp_dualccd_hq.py

WindowsXP Qwen3.8-Flash-Next-Abliterated -> current gfx1151-engine HGN-HQ,
while retaining the refusal-direction recovery / direction-preserving expert
repair from the earlier WindowsXP dual-CCD converter.

This wrapper intentionally delegates the actual conversion to the CURRENT
sibling tools/flashnext2hgn.py, then applies the WindowsXP-specific expert
constraint in-place.  This keeps the converter synchronized with upstream
HGN-HQ changes instead of carrying another fork of the quantizer.

Current-HQ behavior inherited from flashnext2hgn.py:
  * routed experts: weighted Q4CP scale search (imatrix when supplied,
    otherwise x^2 weighting)
  * default routed-expert codebook: UNIVERSAL_EXPERT_CB
  * dense overlay: q8g32 (+ the upstream few-row BF16 exceptions), including
    embed_tokens; this is the recommended main overlay
  * MTP sidecar: q8g64
  * PLE table: fp8 e4m3 + global scale
  * vision tower: BF16

WindowsXP-specific behavior added here:
  1. Build a Halogen-derived WindowsXP overlay with
     build_windowsxp_quality_overlay.py.  The official overlay is still used
     to recover the shared refusal direction r and to create an audit/legacy
     calibration-transfer sidecar.
  2. Run the current upstream HGN-HQ converter on the abliterated BF16 source.
  3. Post-process every fused *.mlp.experts.down_proj.weight Q4CP tensor
     (48 trunk + MTP) so r^T Q remains close to r^T W_ablit, preserving the
     lambda=1.5 edit instead of accidentally drifting toward lambda=1.
  4. On Windows, pin the upstream multiprocessing expert workers one process
     per physical core, split evenly across the two CCD/L3 domains.  --jobs is
     PER CCD for compatibility with the earlier script: --jobs 8 => 16 total.

Outputs:
  <name>.hgn                    current HGN-HQ base, expert-down repaired
  <name>.overlay.hgn            current upstream q8g32 dense overlay (default)
  <name>-mtp.hgn                current upstream q8g64 MTP sidecar
  <name>-vision.hgn             BF16 vision tower
  <name>.direction.npy          recovered refusal direction
  <name>.experts.json           post-quant direction/weighted-error report
  <name>.halogen.overlay.hgn    legacy/audit Halogen-calibration transplant
  <name>.halogen.overlay.hgn.*  its manifest/report from qov
  tokenizer/

Why the main overlay changed from the old script:
  current gfx1151-engine HGN-HQ uses 8-bit dense weights.  The old calibrated
  Halogen overlay is 4-bit for most dense tensors, so loading it as the main
  overlay would overwrite the new q8g32 tensors and lose the current HGN-HQ
  quality gain.  The Halogen-derived overlay is therefore retained as an
  auxiliary artifact and direction source.  Use --overlay-mode halogen only
  when deliberately reproducing the old overlay behavior.

Requirements in the same tools/ directory:
  flashnext2hgn.py              current gfx1151-engine version
  q4cp_imat.py                  current upstream helper
  build_windowsxp_quality_overlay.py  previous WindowsXP calibration-transfer helper

Recommended on Ryzen AI Max+ 395:
  --jobs 8                      8 physical cores/CCD, 16 worker processes total

Example:
  python tools/flashnext2hgn_windowsxp_dualccd_hq.py \
    C:/Users/dogez/.models/Qwen3.8-Flash-Next-Abliterated \
    --out C:/Users/dogez/.models/Qwen3.8-Flash-Next-Abliterated-HGN-HQ \
    --official-overlay C:/Users/dogez/.models/halogen-official/qwen38-flash-next-w4b.overlay.hgn \
    --jobs 8
"""

import argparse
import contextlib
import ctypes
import inspect
import json
import math
import multiprocessing as multiprocessing_module
import os
import shutil
import struct
import time
from pathlib import Path
from types import SimpleNamespace

# Avoid nested BLAS/OpenMP thread pools inside each pinned quantizer process.
# Set before importing numpy / flashnext2hgn.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

import flashnext2hgn as core

try:
    import build_windowsxp_quality_overlay as qov
except ImportError as e:
    raise SystemExit(
        "build_windowsxp_quality_overlay.py must be beside this script in tools/"
    ) from e


EXPERT_SUFFIX = "mlp.experts.down_proj.weight"
REQUIRED_CORE_API = (
    "convert",
    "EXP_RE",
    "Imatrix",
    "UNIVERSAL_EXPERT_CB",
    "q4cp_codebook",
    "expert_spot_check",
    "base_dtype",
    "scan_model_dir",
    "map_lm_name",
    "DT_Q4CP",
    "_qjob",
)


def require_current_core():
    missing = [n for n in REQUIRED_CORE_API if not hasattr(core, n)]
    if missing:
        raise SystemExit(
            "tools/flashnext2hgn.py is not the HGN-HQ converter this wrapper was "
            f"synced against; missing: {', '.join(missing)}\n"
            "Update gfx1151-engine first, then rerun."
        )
    sig = inspect.signature(core.convert)
    for n in ("imatrix", "jobs", "overlay", "overlay_skip", "per_job", "expert_cb"):
        if n not in sig.parameters:
            raise SystemExit(
                f"current flashnext2hgn.convert() has no '{n}' parameter; "
                "upstream changed its API and this wrapper must be re-synced."
            )


def human(n):
    return f"{n / 2**30:.3f} GiB"


def is_fused_expert_down(name, shape):
    return (
        name.endswith(EXPERT_SUFFIX)
        and len(shape) == 3
        and shape[-2] == 2560
        and shape[-1] % 32 == 0
    )


# ---------------------------------------------------------------------------
# Q4CP views / direction diagnostics and repair
# ---------------------------------------------------------------------------


def q4_views(blob, shape):
    """Mutable Q4CP views for fused [E,H,C]. Layout follows core.q4cp_size."""
    E, H, C = map(int, shape)
    rows = E * H
    size, stride = core.q4cp_size(rows, C)
    if len(blob) != size:
        raise ValueError(f"q4 size mismatch {len(blob)} != {size}")

    ba = bytearray(blob)
    mv = memoryview(ba)
    cb = np.frombuffer(mv, dtype="<f4", count=16, offset=0)
    codes_n = rows * C // 2
    packed = np.ndarray(
        shape=(E, H, C // 2), dtype=np.uint8, buffer=mv, offset=64
    )
    scales_off = 64 + codes_n
    scales = np.ndarray(
        shape=(rows, C // 32),
        dtype="<f2",
        buffer=mv,
        offset=scales_off,
        strides=(stride, 2),
    ).reshape(E, H, C // 32)
    return ba, cb, packed, scales


def unpack_codes(packed_chunk, cols):
    nib = np.empty(packed_chunk.shape[:-1] + (cols,), dtype=np.uint8)
    nib[..., 0::2] = packed_chunk & 0x0F
    nib[..., 1::2] = packed_chunk >> 4
    return nib


def pack_codes_into(dst, nib):
    dst[...] = nib[..., 0::2] | (nib[..., 1::2] << 4)


def q4_direction_projection(cb, packed, scales, r, chunk_h=64):
    E, H, half = packed.shape
    C = half * 2
    G = C // 32
    out = np.zeros((E, C), dtype=np.float64)
    r64 = np.asarray(r, dtype=np.float64)

    for h0 in range(0, H, chunk_h):
        h1 = min(H, h0 + chunk_h)
        nib = unpack_codes(packed[:, h0:h1, :], C)
        sc = scales[:, h0:h1, :].astype(np.float32)
        qn = cb[nib].reshape(E, h1 - h0, G, 32)
        q = qn * sc[:, :, :, None]
        out += np.einsum(
            "b,ebgc->egc",
            r64[h0:h1],
            q.astype(np.float64),
            optimize=True,
        ).reshape(E, C)
        del nib, sc, qn, q
    return out.astype(np.float32)


def _bf16_dir_projection(u16, shape, r, chunk_h=64):
    """r^T W for BF16 [E,H,C] given as a uint16 view, chunk over H."""
    E, H, C = map(int, shape)
    out = np.zeros((E, C), dtype=np.float64)
    r64 = np.asarray(r, dtype=np.float64)

    for h0 in range(0, H, chunk_h):
        h1 = min(H, h0 + chunk_h)
        w = (u16[:, h0:h1, :].astype(np.uint32) << 16).view(np.float32)
        out += np.einsum(
            "h,ehc->ec",
            r64[h0:h1],
            w,
            dtype=np.float64,
            optimize=True,
        )
        del w
    return out.astype(np.float32)


def source_direction_projection_bf16(st, src, shape, r, chunk_h=64):
    """Compute r^T W for BF16 [E,H,C] without materializing the whole tensor."""
    E, H, C = map(int, shape)
    sdt, sshape, raw = st.raw(src)
    if sdt != "BF16" or tuple(sshape) != tuple(shape):
        raise ValueError(f"{src}: expected BF16 {shape}, got {sdt} {sshape}")
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(E, H, C)
    return _bf16_dir_projection(u16, shape, r, chunk_h)


def preserve_expert_direction_bf16(
    st,
    src,
    shape,
    blob,
    r,
    max_rows=1024,
    goal_ratio=0.02,
    chunk_h=64,
    check_every=16,
    max_flip_frac=0.08,
    full_guard=True,
    full_guard_kappa=1.0,
):
    """Greedy +/-1 Q4 nibble repair preserving the target-model r^T W.

    The starting blob is the CURRENT upstream HGN-HQ weighted Q4CP result.
    We do not re-quantize it.  Only adjacent codebook entries are considered,
    and a change is accepted iff it reduces |r^T(W-Q)| for that expert/input
    column.  This preserves the current scale search / universal codebook while
    re-imposing the WindowsXP lambda=1.5 residual-writer direction.

    With full_guard=True (default) a flip is accepted only if its directional
    gain outweighs its full-space error cost:

        dfull = dq*(dq - 2*(W-Q))  <=  kappa * (err0^2 - err_new^2),
        dq = (cb[new]-cb[old])*scale

    With kappa=1 every accepted flip monotonically decreases the joint
    objective J = sum|r^T(W-Q)|^2 + sum|W-Q|^2.  kappa=0 forbids any
    full-error increase (too strict in practice: a near-nearest-code tensor
    has dfull >= 0 for almost every single-step move, so nothing can flip).
    Large kappa approaches the unguarded behavior.  The post-repair
    weighted-error check in patch_expert_downs remains as the final
    fail-closed gate.
    """
    E, H, C = map(int, shape)
    if H != len(r):
        raise ValueError(f"expert tensor shape {shape} incompatible with r={len(r)}")

    sdt, sshape, raw = st.raw(src)
    if sdt != "BF16" or tuple(sshape) != tuple(shape):
        raise ValueError(f"{src}: expected BF16 {shape}, got {sdt} {sshape}")
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(E, H, C)

    ba, cb, packed, scales = q4_views(blob, shape)
    target = _bf16_dir_projection(u16, shape, r, chunk_h)
    current = q4_direction_projection(cb, packed, scales, r, chunk_h)
    residual = target - current

    initial_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    target_rms = float(np.sqrt(np.mean(np.square(target, dtype=np.float64))))
    if initial_rms == 0.0:
        return bytes(ba), {
            "shape": list(shape),
            "initial_residual_rms": 0.0,
            "final_residual_rms": 0.0,
            "residual_ratio": 0.0,
            "target_projection_rms": target_rms,
            "rows_visited": 0,
            "flips": 0,
            "flip_fraction": 0.0,
            "guard_blocked": 0,
            "history": [],
        }

    order = np.argsort(-np.abs(r))
    if max_rows <= 0 or max_rows > H:
        max_rows = H

    total_nibbles = E * H * C
    flips = 0
    guard_blocked = 0
    history = []
    rows_visited = 0

    for k, h in enumerate(order[:max_rows], 1):
        old = unpack_codes(packed[:, h:h + 1, :], C)[:, 0, :]
        sc = np.repeat(scales[:, h, :].astype(np.float32), 32, axis=1)
        rh = float(r[h])

        down = np.maximum(old.astype(np.int16) - 1, 0).astype(np.uint8)
        up = np.minimum(old.astype(np.int16) + 1, 15).astype(np.uint8)
        impact_down = rh * (cb[down] - cb[old]) * sc
        impact_up = rh * (cb[up] - cb[old]) * sc

        err0 = np.abs(residual)
        err_down = np.abs(residual - impact_down)
        err_up = np.abs(residual - impact_up)

        if full_guard:
            # Full-space error of this row BEFORE any flip: E = W - Q.
            w_row = (u16[:, h, :].astype(np.uint32) << 16).view(np.float32)
            e_full = w_row - cb[old] * sc
            # dq = change in Q if the nibble moves one codebook step.
            dq_down = (cb[down] - cb[old]) * sc
            dq_up = (cb[up] - cb[old]) * sc
            # d||E||^2 = (E - dq)^2 - E^2 = dq*(dq - 2E).
            dfull_down = dq_down * (dq_down - 2.0 * e_full)
            dfull_up = dq_up * (dq_up - 2.0 * e_full)
            # Directional gain in squared units; >0 iff the flip improves.
            gain_down = err0 * err0 - err_down * err_down
            gain_up = err0 * err0 - err_up * err_up
            allow_down = (old > 0) & (gain_down > 0) & (
                dfull_down <= full_guard_kappa * gain_down
            )
            allow_up = (old < 15) & (gain_up > 0) & (
                dfull_up <= full_guard_kappa * gain_up
            )
            errd = np.where(allow_down, err_down, np.inf)
            erru = np.where(allow_up, err_up, np.inf)
            choose_down = allow_down & (err_down <= erru)
            choose_up = allow_up & (err_up < errd)
            guard_blocked += int(np.count_nonzero(
                (((gain_down > 0) & (old > 0)) | ((gain_up > 0) & (old < 15)))
                & ~(choose_down | choose_up)
            ))
        else:
            choose_down = (old > 0) & (err_down < err0) & (err_down <= err_up)
            choose_up = (old < 15) & (err_up < err0) & (err_up < err_down)

        new = old.copy()
        new[choose_down] = down[choose_down]
        new[choose_up] = up[choose_up]
        if np.any(choose_down):
            residual[choose_down] -= impact_down[choose_down]
        if np.any(choose_up):
            residual[choose_up] -= impact_up[choose_up]
        pack_codes_into(packed[:, h:h + 1, :], new[:, None, :])

        nflip = int(np.count_nonzero(choose_down) + np.count_nonzero(choose_up))
        flips += nflip
        rows_visited = k
        flip_fraction = flips / total_nibbles
        if flip_fraction > max_flip_frac:
            raise RuntimeError(
                f"expert correction exceeded max flip fraction {max_flip_frac:.2%}: "
                f"{flip_fraction:.2%}"
            )

        if k % check_every == 0 or k == max_rows:
            rrms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
            ratio = rrms / initial_rms
            history.append({
                "rows_visited": k,
                "residual_rms": rrms,
                "ratio_to_initial": ratio,
                "flips": flips,
                "flip_fraction": flip_fraction,
            })
            if ratio <= goal_ratio:
                break

    final_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    return bytes(ba), {
        "shape": list(shape),
        "initial_residual_rms": initial_rms,
        "final_residual_rms": final_rms,
        "residual_ratio": final_rms / max(initial_rms, 1e-30),
        "target_projection_rms": target_rms,
        "rows_visited": rows_visited,
        "flips": flips,
        "flip_fraction": flips / total_nibbles,
        "guard_blocked": guard_blocked,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Windows dual-CCD affinity for upstream multiprocessing.Pool
# ---------------------------------------------------------------------------


def _windows_cpu_sets():
    if os.name != "nt":
        return []
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetSystemCpuSetInformation
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    fn.restype = ctypes.c_int

    need = ctypes.c_ulong(0)
    fn(None, 0, ctypes.byref(need), None, 0)
    if need.value == 0:
        raise OSError(ctypes.get_last_error(),
                      "GetSystemCpuSetInformation size query failed")
    buf = ctypes.create_string_buffer(need.value)
    got = ctypes.c_ulong(need.value)
    if not fn(buf, need.value, ctypes.byref(got), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation failed")

    raw = memoryview(buf.raw[:got.value])
    out, off = [], 0
    while off + 8 <= len(raw):
        size, typ = struct.unpack_from("<II", raw, off)
        if size < 8 or off + size > len(raw):
            raise RuntimeError(f"invalid CPU-set record at {off}: size={size}")
        if typ == 0 and size >= 32:  # CpuSetInformation
            (cpu_id, group, logical, core_id, llc, numa, efficiency,
             flags, reserved, allocation_tag) = struct.unpack_from(
                "<IH6BIQ", raw, off + 8
            )
            out.append({
                "id": int(cpu_id),
                "group": int(group),
                "logical": int(logical),
                "core": int(core_id),
                "llc": int(llc),
                "numa": int(numa),
                "efficiency": int(efficiency),
                "flags": int(flags),
            })
        off += size
    return out


def dual_ccd_cpu_plan(jobs_per_ccd):
    jobs_per_ccd = int(jobs_per_ccd)
    if jobs_per_ccd < 1:
        raise ValueError("--jobs must be >= 1")

    if os.name != "nt":
        n = min(os.cpu_count() or 1, jobs_per_ccd * 2)
        return list(range(n)), ["unbound"], [list(range(n))]

    recs = [r for r in _windows_cpu_sets() if r["group"] == 0]
    if not recs:
        raise RuntimeError("Windows CPU-set query returned no group-0 CPUs")

    by_llc = {}
    for r in recs:
        by_llc.setdefault(r["llc"], []).append(r)

    domains = []
    for llc, rr in by_llc.items():
        # Pick exactly one logical CPU from each physical core: no SMT sibling.
        by_core = {}
        for r in sorted(rr, key=lambda x: x["logical"]):
            by_core.setdefault(r["core"], r["logical"])
        cpus = [by_core[c] for c in sorted(by_core)]
        domains.append((llc, cpus))

    domains.sort(key=lambda x: (-len(x[1]), x[0]))
    if len(domains) < 2:
        raise RuntimeError(f"could not identify two LLC/CCD domains: {domains!r}")

    (llc0, c0), (llc1, c1) = domains[:2]
    if len(c0) < jobs_per_ccd or len(c1) < jobs_per_ccd:
        raise RuntimeError(
            f"--jobs {jobs_per_ccd} requires that many physical cores per CCD; "
            f"detected LLC{llc0}={len(c0)}, LLC{llc1}={len(c1)}"
        )
    c0, c1 = c0[:jobs_per_ccd], c1[:jobs_per_ccd]
    plan = []
    for i in range(jobs_per_ccd):
        plan.extend((c0[i], c1[i]))
    return plan, [llc0, llc1], [c0, c1]


def _pin_worker_process(cpu_plan, counter, lock):
    """Pool initializer: pin the whole worker process to one physical core."""
    if os.name != "nt":
        return
    with lock:
        idx = int(counter.value)
        counter.value += 1
    if idx >= len(cpu_plan):
        idx %= len(cpu_plan)
    cpu = int(cpu_plan[idx])

    bit_width = ctypes.sizeof(ctypes.c_size_t) * 8
    if cpu >= bit_width:
        raise RuntimeError(
            f"logical CPU {cpu} exceeds process-affinity mask width {bit_width}"
        )
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_proc = k32.GetCurrentProcess
    get_proc.restype = ctypes.c_void_p
    set_mask = k32.SetProcessAffinityMask
    set_mask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    set_mask.restype = ctypes.c_int
    if not set_mask(get_proc(), ctypes.c_size_t(1 << cpu)):
        raise OSError(ctypes.get_last_error(),
                      f"SetProcessAffinityMask(cpu={cpu}) failed")


@contextlib.contextmanager
def upstream_pool_dual_ccd(jobs_per_ccd):
    """Patch multiprocessing.Pool only while core.convert() creates its pool."""
    mp = multiprocessing_module
    workers = jobs_per_ccd * 2

    if os.name != "nt":
        # Keep old --jobs-per-CCD semantics even off Windows, but no affinity.
        yield workers
        return

    cpu_plan, llcs, per_ccd = dual_ccd_cpu_plan(jobs_per_ccd)
    workers = len(cpu_plan)
    print(
        "q4-affinity: "
        f"CCD/L3 {llcs[0]} CPUs={per_ccd[0]} | "
        f"CCD/L3 {llcs[1]} CPUs={per_ccd[1]} | "
        f"{workers} processes",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()
    original_pool = mp.Pool

    def affinity_pool(processes=None, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            raise RuntimeError(
                "upstream flashnext2hgn.py started using a Pool initializer; "
                "re-sync this wrapper instead of stacking initializers blindly"
            )
        n = workers if processes is None else int(processes)
        if n != workers:
            raise RuntimeError(
                f"unexpected upstream Pool({n}); wrapper planned {workers} pinned workers"
            )
        return ctx.Pool(
            processes=n,
            initializer=_pin_worker_process,
            initargs=(cpu_plan, counter, lock),
            **kwargs,
        )

    mp.Pool = affinity_pool
    try:
        yield workers
    finally:
        mp.Pool = original_pool


# ---------------------------------------------------------------------------
# Halogen calibration-transfer sidecar + direction recovery
# ---------------------------------------------------------------------------


def run_quality(args, name, outdir):
    halogen_path = os.path.join(outdir, f"{name}.halogen.overlay.hgn")
    direction_path = os.path.join(outdir, f"{name}.direction.npy")
    qargs = SimpleNamespace(
        ablit_dir=args.model_dir,
        official_overlay=args.official_overlay,
        out=halogen_path,
        lam=args.lam,
        anchor_cols=args.anchor_cols,
        power_iters=args.power_iters,
        min_anchor_cos=args.min_anchor_cos,
        min_median_s12=args.min_median_s12,
        lambda_tol=args.lambda_tol,
        force_direction=args.force_direction,
        q4_iters=args.quality_q4_iters,
        include_mtp=False,
        direction_out=direction_path,
        manifest=halogen_path + ".manifest.json",
        report=halogen_path + ".report.tsv",
        name=name + "-halogen-xfer",
        dry_run=args.dry_run,
    )
    qov.build(qargs)
    if not os.path.exists(direction_path):
        raise SystemExit(f"quality builder did not create {direction_path}")
    r = np.load(direction_path).astype(np.float32)
    n = float(np.linalg.norm(r))
    if not np.isfinite(n) or n == 0:
        raise SystemExit("recovered direction is zero/non-finite")
    r /= n
    return halogen_path, direction_path, r


# ---------------------------------------------------------------------------
# Source mapping + post-process the current upstream HQ base
# ---------------------------------------------------------------------------


def mapped_source_table(model_dir):
    raw = core.scan_model_dir(model_dir)
    out = {}
    for src_name, (st, dt, shape) in raw.items():
        if src_name.startswith(core.VIS_PREFIX):
            continue
        m = core.map_lm_name(src_name)
        if m is None:
            continue
        if m.startswith("layers.1.ple.ple_embedding.ngram_embedding.shard_"):
            continue
        out[m] = (st, dt, tuple(shape), src_name)
    return out


def patch_expert_downs(args, name, outdir, r):
    base_path = os.path.join(outdir, f"{name}.hgn")
    if not os.path.exists(base_path):
        raise SystemExit(f"base not found after upstream conversion: {base_path}")

    source = mapped_source_table(args.model_dir)
    reader = qov.HgnReader(base_path)
    recs = []
    for n in reader.order:
        t = reader.tensors[n]
        if is_fused_expert_down(n, t.dims):
            recs.append((n, t.dtype, tuple(t.dims), t.data_offset, t.data_size))
    reader.close()

    if not recs:
        raise SystemExit("no fused expert down_proj tensors found in generated base")
    expected = core.EXPECTED_CONFIG["num_hidden_layers"] + 1  # 48 trunk + MTP
    if len(recs) != expected:
        raise SystemExit(
            f"expected {expected} fused expert down_proj tensors (48 trunk + MTP), "
            f"found {len(recs)}"
        )

    imat = core.Imatrix(args.imatrix) if args.imatrix else None
    reports = []
    print(f"\npatching {len(recs)} weighted-Q4 expert down_proj tensors in place ...",
          flush=True)

    with open(base_path, "r+b", buffering=0) as f:
        for idx, (m, dt, shape, off, size) in enumerate(recs, 1):
            if dt != core.DT_Q4CP:
                raise SystemExit(f"{m}: expected q4cp dtype {core.DT_Q4CP}, got {dt}")
            if m not in source:
                raise SystemExit(f"{m}: source tensor not found")
            st, sdt, sshape, src = source[m]
            if sdt != "BF16" or tuple(sshape) != tuple(shape):
                raise SystemExit(
                    f"{m}: expected BF16 source {shape}, got {sdt} {sshape}"
                )

            t0 = time.time()
            f.seek(off)
            original = f.read(size)
            if len(original) != size:
                raise SystemExit(f"{m}: short read {len(original)} != {size}")

            patched, met = preserve_expert_direction_bf16(
                st,
                src,
                shape,
                original,
                r,
                max_rows=args.expert_max_rows,
                goal_ratio=args.expert_goal_ratio,
                chunk_h=args.expert_chunk_h,
                check_every=args.expert_check_every,
                max_flip_frac=args.expert_max_flip_frac,
                full_guard=not args.no_expert_full_guard,
                full_guard_kappa=args.expert_full_guard_kappa,
            )
            if met["residual_ratio"] > args.expert_fail_ratio:
                raise RuntimeError(
                    f"{m}: direction-preserving residual ratio "
                    f"{met['residual_ratio']:.6f} exceeds hard limit "
                    f"{args.expert_fail_ratio:.6f}; base left unmodified for this tensor"
                )

            f.seek(off)
            f.write(patched)
            # unbuffered write; the following independent reader sees page-cache data

            # Current upstream's expert metric, re-run AFTER our nibble repair.
            mt = core.EXP_RE.match(m)
            if not mt:
                raise RuntimeError(f"{m}: no longer matches current core.EXP_RE")
            E, R, C = map(int, shape)
            layer = core.EXPECTED_CONFIG["num_hidden_layers"] if mt.group(1) else int(mt.group(2))
            iv = imat.experts(layer, "down", E, C) if imat is not None else None
            absmax_err, written_err = core.expert_spot_check(
                st, src, base_path, off, E, R, C, iv, ref_cb=None
            )
            met.update({
                "tensor": m,
                "dtype": int(dt),
                "weighted_absmax_err": float(absmax_err),
                "weighted_written_err": float(written_err),
                "weighted_vs_absmax_ratio": float(written_err / max(absmax_err, 1e-30)),
                "elapsed_s": float(time.time() - t0),
            })

            # Upstream HGN-HQ is fail-closed on the same-codebook weighted scale
            # search.  Keep that property after our WindowsXP correction.  If the
            # post-repair tensor loses it, restore the exact upstream blob.
            if not written_err < absmax_err:
                f.seek(off)
                f.write(original)
                raise RuntimeError(
                    f"{m}: direction repair regressed the current HGN-HQ expert "
                    f"below its absmax baseline ({written_err:.6e} >= {absmax_err:.6e}); "
                    "the original upstream tensor was restored"
                )

            reports.append(met)
            print(
                f"  [{idx:02d}/{len(recs)}] {m}: "
                f"dir={met['residual_ratio']:.5f} "
                f"flips={met['flip_fraction']:.3%} "
                f"blk={met['guard_blocked']} "
                f"werr={met['weighted_vs_absmax_ratio']:.4f} "
                f"rows={met['rows_visited']} {met['elapsed_s']:.1f}s",
                flush=True,
            )
            del original, patched

        # One durability flush for all in-place tensor edits; avoid 49 full fsyncs.
        os.fsync(f.fileno())

    report_path = os.path.join(outdir, f"{name}.experts.json")
    payload = {
        "method": (
            "current flashnext2hgn HGN-HQ weighted Q4CP + post-quant adjacent-nibble "
            "repair preserving WindowsXP target r^T W"
        ),
        "full_guard": not args.no_expert_full_guard,
        "full_guard_kappa": args.expert_full_guard_kappa,
        "model_dir": os.path.abspath(args.model_dir),
        "imatrix": os.path.abspath(args.imatrix) if args.imatrix else None,
        "expert_codebook": args.expert_codebook,
        "count": len(reports),
        "direction_file": os.path.join(outdir, f"{name}.direction.npy"),
        "tensors": reports,
    }
    tmp = report_path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as jf:
        json.dump(payload, jf, indent=2, ensure_ascii=False)
        jf.write("\n")
    os.replace(tmp, report_path)
    print(f"expert report -> {report_path}")

    bad = [x for x in reports if x["residual_ratio"] > args.expert_warn_ratio]
    if bad:
        print(
            f"WARNING: {len(bad)} expert tensors ended above direction residual "
            f"ratio {args.expert_warn_ratio:g}; inspect experts.json.",
            flush=True,
        )
    return report_path


# ---------------------------------------------------------------------------
# Delegate conversion to the current upstream script
# ---------------------------------------------------------------------------


def _windows_start_stub(outdir, name, has_mtp, has_vision, engine_dir,
                        has_overlay=False):
    """core.write_start_script is Linux-only; do not make conversion fail on Windows."""
    p = os.path.join(outdir, "start.sh")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "echo 'This conversion was produced by the WindowsXP wrapper. "
            "Use gfx1151-engine start_win.exe / your Windows PowerShell launcher.' >&2\n"
            "exit 1\n"
        )
    return p


def run_current_upstream(args, name, outdir):
    overlay_hq8 = (args.overlay_mode == "hq8") and not args.skip_overlay
    overlay_skip = tuple(s for s in args.overlay_skip.split(",") if s)

    print("\n=== phase 2: CURRENT upstream HGN-HQ conversion ===", flush=True)
    print(f"  expert codebook : {args.expert_codebook}")
    print(f"  imatrix         : {args.imatrix or 'none -> x^2 weighting'}")
    print(f"  main overlay    : {'q8g32 HGN-HQ' if overlay_hq8 else 'not generated by upstream'}")
    print(f"  q4 workers      : {args.jobs}/CCD x 2 = {args.jobs * 2} processes")

    old_start = core.write_start_script
    if os.name == "nt":
        core.write_start_script = _windows_start_stub
    try:
        with upstream_pool_dual_ccd(args.jobs) as total_workers:
            core.convert(
                args.model_dir,
                outdir,
                name,
                args.skip_vision,
                args.skip_ple,
                args.skip_mtp_sidecar,
                dry_run=args.dry_run,
                engine_dir=args.engine_bin,
                classic=False,
                imatrix=args.imatrix,
                jobs=total_workers,
                overlay=overlay_hq8,
                only_overlay=False,
                overlay_skip=overlay_skip,
                per_job=args.per_job,
                expert_cb=args.expert_codebook,
            )
    finally:
        core.write_start_script = old_start


def install_halogen_as_main(args, name, outdir, halogen_path):
    if args.skip_overlay or args.overlay_mode != "halogen" or args.dry_run:
        return
    if not os.path.exists(halogen_path):
        raise SystemExit(f"Halogen overlay missing: {halogen_path}")
    main_overlay = os.path.join(outdir, f"{name}.overlay.hgn")
    tmp = main_overlay + ".part"
    shutil.copyfile(halogen_path, tmp)
    os.replace(tmp, main_overlay)
    print(
        f"legacy overlay mode: {halogen_path} -> {main_overlay} "
        f"({human(os.path.getsize(main_overlay))})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    require_current_core()
    ap = argparse.ArgumentParser(
        description=(
            "Current gfx1151 HGN-HQ converter + WindowsXP Halogen direction "
            "recovery + dual-CCD pinned expert quantization + post-Q4 direction repair."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model_dir", help="WindowsXP abliterated BF16 safetensors directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--official-overlay", required=True,
                    help="official Halogen QUALITY overlay (not overlay-speed.hgn)")
    ap.add_argument("--name")

    # Current upstream HGN-HQ options.
    ap.add_argument("--imatrix",
                    help="current upstream llama.cpp/GGUF or legacy imatrix; "
                         "omit for x^2-weighted expert quantization")
    ap.add_argument("--expert-codebook", choices=("universal", "trained"),
                    default="universal")
    ap.add_argument("--overlay-mode", choices=("hq8", "halogen"), default="hq8",
                    help="hq8 (default): current q8g32 dense overlay; halogen: "
                         "old transplanted Halogen overlay as main .overlay.hgn")
    ap.add_argument("--overlay-skip", default="", metavar="SUB1,SUB2",
                    help="current hq8 overlay skip substrings, e.g. lm_head")
    ap.add_argument("--skip-overlay", action="store_true")
    ap.add_argument("--per-job", type=int, default=4,
                    help="experts per upstream worker task (current default: 4)")
    ap.add_argument("--engine-bin",
                    help="passed to current upstream converter; Windows start.sh "
                         "generation is replaced by a stub")

    # Dual-CCD semantics retained from the previous script.
    ap.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="expert worker PROCESSES PER CCD; Ryzen AI Max+ 395: 8 => 16 total",
    )

    # Halogen direction recovery / audit overlay.
    ap.add_argument("--lambda", dest="lam", type=float)
    ap.add_argument("--anchor-cols", type=int, default=512)
    ap.add_argument("--power-iters", type=int, default=12)
    ap.add_argument("--min-anchor-cos", type=float, default=0.95)
    ap.add_argument("--min-median-s12", type=float, default=2.0)
    ap.add_argument("--lambda-tol", type=float, default=0.05,
                    help="fail if data-estimated lambda deviates from the "
                         "configured lambda by more than this")
    ap.add_argument("--force-direction", action="store_true")
    ap.add_argument("--quality-q4-iters", type=int, default=4)

    # Post-HQ expert direction preservation.
    ap.add_argument("--expert-max-rows", type=int, default=1024)
    ap.add_argument("--expert-check-every", type=int, default=16)
    ap.add_argument("--expert-goal-ratio", type=float, default=0.02)
    ap.add_argument("--expert-warn-ratio", type=float, default=0.03)
    ap.add_argument("--expert-fail-ratio", type=float, default=0.05)
    ap.add_argument("--expert-max-flip-frac", type=float, default=0.08)
    ap.add_argument("--expert-chunk-h", type=int, default=64)
    ap.add_argument("--no-expert-full-guard", action="store_true",
                    help="allow direction-repair flips that increase the "
                         "full-space element error (not recommended)")
    ap.add_argument("--expert-full-guard-kappa", type=float, default=30.0,
                    help="full-error guard tradeoff: accept a flip iff "
                         "dfull <= kappa * (directional gain)^2. kappa=1 keeps "
                         "J = |r^T(W-Q)|^2 + |W-Q|^2 monotone but stalls far "
                         "above the residual goal on real weighted-Q4CP "
                         "tensors; 30 reached goal 0.02 at 1/3 of the "
                         "unguarded full-error cost on the L15 probe")
    ap.add_argument("--no-expert-preserve", action="store_true")

    # Output choices inherited from previous/current converters.
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-ple", action="store_true")
    ap.add_argument("--skip-mtp-sidecar", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    args.model_dir = os.path.abspath(args.model_dir)
    args.out = os.path.abspath(args.out)
    args.official_overlay = os.path.abspath(args.official_overlay)
    if args.imatrix:
        args.imatrix = os.path.abspath(args.imatrix)

    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if args.per_job < 1:
        ap.error("--per-job must be >= 1")
    if args.imatrix and not os.path.isfile(args.imatrix):
        ap.error(f"--imatrix {args.imatrix}: no such file")
    if not os.path.isfile(args.official_overlay):
        ap.error(f"--official-overlay {args.official_overlay}: no such file")
    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        ap.error(f"{args.model_dir}: config.json not found")

    name = args.name or os.path.basename(os.path.normpath(args.model_dir))
    os.makedirs(args.out, exist_ok=True)

    print("=== phase 1: Halogen calibration transfer + refusal direction ===",
          flush=True)
    halogen_path, direction_path, r = run_quality(args, name, args.out)

    run_current_upstream(args, name, args.out)

    if not args.dry_run and not args.no_expert_preserve:
        print("\n=== phase 3: WindowsXP direction-preserving post-Q4 repair ===",
              flush=True)
        patch_expert_downs(args, name, args.out, r)
    elif args.no_expert_preserve:
        print("\nWARNING: --no-expert-preserve selected; current upstream Q4 expert "
              "down_proj tensors are left unmodified.")

    install_halogen_as_main(args, name, args.out, halogen_path)

    print("\n=== outputs ===")
    print(f"base       : {os.path.join(args.out, name + '.hgn')}")
    if not args.skip_overlay:
        print(f"main ovl   : {os.path.join(args.out, name + '.overlay.hgn')} "
              f"({args.overlay_mode})")
    print(f"halogen ovl: {halogen_path}")
    print(f"direction  : {direction_path}")
    if not args.no_expert_preserve and not args.dry_run:
        print(f"experts    : {os.path.join(args.out, name + '.experts.json')}")
    if not args.skip_mtp_sidecar:
        print(f"MTP        : {os.path.join(args.out, name + '-mtp.hgn')}")
    if not args.skip_vision:
        print(f"vision     : {os.path.join(args.out, name + '-vision.hgn')}")

    if args.overlay_mode == "hq8" and not args.skip_overlay:
        print(
            "\nNOTE: .overlay.hgn is the CURRENT upstream q8g32 dense overlay. "
            "The Halogen-transplanted 4-bit overlay is kept separately as "
            f"{Path(halogen_path).name}; do not load it after the q8g32 overlay "
            "unless you intentionally want the legacy behavior."
        )


if __name__ == "__main__":
    multiprocessing_module.freeze_support()
    main()
