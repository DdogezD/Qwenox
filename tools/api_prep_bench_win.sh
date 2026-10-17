#!/usr/bin/env bash
# api_prep_bench_win.sh - T2b driver (HANDOFF-AGENT-SHORT-PREFILL.md §3.3):
# v2 engine :8732 + instrumented API (--console) :8733, then
# tools/api_prep_bench.py with 50/200/400-message synthetic agent talks.
# API prep breakdown lines land in logs/api_prep_api.log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
M=$ROOT/models
EPID=""; APID=""
cleanup() {
  kill $EPID $APID 2>/dev/null
  for p in $EPID $APID; do [ -n "$p" ] && taskkill //PID $p //F //T >/dev/null 2>&1; done
  if command -v powershell > /dev/null; then
    powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox*') -and (\$_.CommandLine -match '873[23]') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  fi
}
trap cleanup EXIT

if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "a qwenox process is running - stop it first"; exit 1
fi

# Stale logs would let the readiness greps below match a previous run.
rm -f logs/api_prep_engine.log logs/api_prep_api.log

# production env (v2 weights, 1 slot, rckpt default on)
for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
export QWENOX_PARALLEL=1 QWENOX_PREFILL_CHUNK=8192 QWENOX_CONC_PREFILL_CHUNK=8192
export QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16 QWENOX_PLE_MISS_LOG=1

build/qwenox-engine-win.exe "$(win $M/qwen38-flash-next-v2.hgn)" \
  "$(win $M/qwen38-flash-next-ngram.hgn)" "$(win $M/qwen38-flash-next-mtp.hgn)" \
  --serve --host 127.0.0.1 --port 8732 --maxctx 131072 \
  > logs/api_prep_engine.log 2>&1 &
EPID=$!
ok=0
for i in $(seq 1 150); do
  grep -q 'serve: listening' logs/api_prep_engine.log 2>/dev/null && { ok=1; break; }
  kill -0 $EPID 2>/dev/null || { tail -n 10 logs/api_prep_engine.log; exit 1; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "引擎启动超时"; exit 1; }
echo "engine listening"

build/qwenox-win.exe --console --tokenizer models/tokenizer --engine 127.0.0.1:8732 \
  --host 127.0.0.1 --port 8733 --context 131072 > logs/api_prep_api.log 2>&1 &
APID=$!
for i in $(seq 1 30); do
  netstat -an | grep -E "[:.]8733\s+.*LISTENING" >/dev/null 2>&1 && break
  sleep 1
done
echo "api listening"

python "$HERE/api_prep_bench.py" --port 8733 --sizes "${SIZES:-50,200,400}" \
  --tokens "${TOKENS:-80000}" 2>&1 | tee logs/api_prep_bench.txt
echo
echo "==== API prep 拆分（logs/api_prep_api.log）===="
grep "qwenox-api: prep" logs/api_prep_api.log | tail -n 10
echo "API_PREP_BENCH DONE"
