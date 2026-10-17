#!/usr/bin/env bash
# GG_A4 KLD A/B 重测（qtab 同步修复后的引擎二进制，直接跑引擎绕过 flock 检查）
# 用法: bash tools/gga4fix_kld_sweep.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BIN=build/qwenox-engine-win.exe
# IQ4_XS 权重位置可用 GGUF_FILE 或 MODELS_ROOT 覆盖
MODELS_ROOT="${MODELS_ROOT:-C:/Users/Mark/Models}"
M="${GGUF_FILE:-$MODELS_ROOT/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
REF=data/kld/bf16_c512.kld

for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
export QWENOX_KVSNAP=0 QWENOX_GGUF_MTP=""

for a4 in 0 1 2 3; do
  tag=gga4fix_kld_$a4
  echo "== $tag (QWENOX_GG_A4=$a4) =="
  QWENOX_GG_A4=$a4 "$BIN" "$M" --kld-base "$REF" --maxctx 4096 \
      > "logs/${tag}.out" 2> "logs/${tag}.err" \
    || { tail -5 "logs/${tag}.err"; exit 1; }
  grep -E "kld_summary" "logs/${tag}.out" || true
done
