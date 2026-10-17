#!/usr/bin/env bash
# a1_kld_win.sh - A1 gate (HANDOFF-AGENT-SHORT-PREFILL §A1) on Windows:
#   c512  vs data/kld/bf16_c512.kld (llama.cpp BF16 reference), 64 chunks
#   c8192 vs an engine self-baseline written by the QWENOX_LT_BUCKET=0 run
#         (data/qsa-oracle/131072.tokens, 4 chunks x 8192)
# Each context runs twice: QWENOX_LT_BUCKET=0 (pre-A1 exact-P selection) and
# default (bucketed). Gate (tools/kld_aggr.sh): mean_kld(test) <=
# mean_kld(base) + 0.002 and same_top drop <= 0.5 pp.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

BIN=build/qwenox-engine-win.exe
M=models/qwen38-flash-next-v2.hgn
OVL="models/qwen38-flash-next-ngram.hgn models/qwen38-flash-next-mtp.hgn"
REF=data/kld/bf16_c512.kld
TOKS=data/qsa-oracle/131072.tokens
BASE8192=logs/kld_a1_base8192.kld

# production env (same block as tools/api_prep_bench_win.sh)
for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
export QWENOX_PARALLEL=1 QWENOX_PREFILL_CHUNK=8192 QWENOX_CONC_PREFILL_CHUNK=8192
export QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16

run() {  # run TAG EXTRA_ENV ARGS...
  local tag=$1 extra=$2; shift 2
  echo "== $tag ($extra) =="
  env $extra "$BIN" "$M" $OVL "$@" > "logs/kld_a1_${tag}.out" \
      2> "logs/kld_a1_${tag}.err" || { tail -5 "logs/kld_a1_${tag}.err"; return 1; }
  grep -E "kld_summary|kld_saved" "logs/kld_a1_${tag}.out" || true
}

if [[ -z "${GATE_ONLY:-}" ]]; then
# c512: both modes against the llama.cpp BF16 reference.
if [[ -z "${SKIP_C512:-}" ]]; then
  run c512_base "QWENOX_LT_BUCKET=0" --kld-base "$REF" || exit 1
  run c512_bucket ""                   --kld-base "$REF" || exit 1
fi

# c8192: LT_BUCKET=0 writes the self-baseline, bucketed run scores against it.
if [[ -z "${SKIP_C8192_BASE:-}" ]]; then
  run c8192_base "QWENOX_LT_BUCKET=0" --tokens-file "$TOKS" --kld-save "$BASE8192" \
      --maxctx 8192 --kld-ctx 8192 --kld-chunks "${CHUNKS8192:-4}" || exit 1
fi
run c8192_bucket "" --kld-base "$BASE8192" --maxctx 8192 || exit 1
fi

python - <<'EOF'
import re, sys
def summary(tag):
    for line in open(f"logs/kld_a1_{tag}.out", encoding="utf-8", errors="replace"):
        if line.startswith("kld_summary"):
            return {k: float(v) for k, v in
                    re.findall(r"(\w+)=(-?[\d.]+)", line)}
    sys.exit(f"no kld_summary in logs/kld_a1_{tag}.out")
ok = True
for ctx in ("c512", "c8192"):
    # c8192_base is the --kld-save self-baseline (no summary of its own);
    # being the reference, its deltas are zero by definition.
    b = summary(f"{ctx}_base") if ctx == "c512" else {"mean_kld": 0.0, "same_top": 100.0}
    h = summary(f"{ctx}_bucket")
    dk, dt = h["mean_kld"] - b["mean_kld"], h["same_top"] - b["same_top"]
    good = dk <= 0.002 and dt >= -0.5
    ok &= good
    print(f"{ctx}: bucket-base mean_kld {dk:+.6f} (gate +0.002), "
          f"same_top {dt:+.3f} pp (gate -0.5) -> {'PASS' if good else 'FAIL'}")
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
