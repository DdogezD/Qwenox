#!/usr/bin/env bash
# agent_replay_win.sh - driver for HANDOFF-AGENT-SHORT-PREFILL.md T0/T1 (Windows).
# Starts the engine on 127.0.0.1:8732 with the production env block (v2 hgn,
# PARALLEL=1, KVSNAP=0, rckpt default ON) plus per-run extra env, waits for
# "serve: listening", runs tools/agent_replay.py, stops the engine, next run.
#
#   bash tools/agent_replay_win.sh            # R0-R3 at depth 40960, then R0 at 8192/98304
#   RUNS="r0|40960|" bash tools/agent_replay_win.sh   # custom: tag|depth|extra-env, space separated
#
# Engine logs: logs/areplay_engine_<tag>_d<depth>.log
# Replay rows: logs/areplay/<tag>_d<depth>.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p logs/areplay
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
BIN=$ROOT/build/qwenox-engine-win.exe
M=$ROOT/models
MODEL_ARGS="$(win $M/qwen38-flash-next-v2.hgn) $(win $M/qwen38-flash-next-ngram.hgn) $(win $M/qwen38-flash-next-mtp.hgn)"
MAXCTX=${MAXCTX:-262144}

[[ -f "$BIN" ]] || { echo "missing engine: $BIN"; exit 1; }
if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "a qwenox process is running - stop it first (the test needs the whole GPU)"; exit 1
fi
echo "root $ROOT  head $(git log --oneline -1 | cut -c1-60)"

prod_env() {  # prefill_sched_win.sh prod_env, but PARALLEL=1 and rckpt default ON
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
}

ENG_PID=""
serve() {  # serve TAG DEPTH EXTRA_ENV...
  local tag=$1 depth=$2; shift 2
  prod_env
  for kv in "$@"; do export "$kv"; done
  local log=$ROOT/logs/areplay_engine_${tag}_d${depth}.log
  rm -f "$log"  # a stale log would let the listening grep match a previous run
  local margs; read -r -a margs <<< "$MODEL_ARGS"
  (cd "$(dirname "$BIN")" && exec "$BIN" "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
      --maxctx "$MAXCTX" > "$log" 2>&1) &
  ENG_PID=$!
  local t0=$(date +%s)
  while :; do
    if tr '\r' '\n' < "$log" | grep -q "serve: listening"; then
      echo "  [$tag] listening after $(( $(date +%s) - t0 ))s"; return 0
    fi
    if ! kill -0 $ENG_PID 2>/dev/null; then
      echo "  [$tag] engine exited before listening"; tr '\r' '\n' < "$log" | grep -v "pload [0-9]" | tail -8; return 1
    fi
    (( $(date +%s) - t0 > 600 )) && { echo "  [$tag] engine start timeout"; return 1; }
    sleep 2
  done
}
stop_eng() {
  [[ -n "$ENG_PID" ]] && kill $ENG_PID 2>/dev/null
  if command -v powershell > /dev/null; then
    powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  fi
  sleep 5; ENG_PID=""
}
trap stop_eng EXIT

DEFAULT_RUNS=(
  "r0|40960|"
  "r1_ltmiss|40960|QWENOX_LT_MISS_LOG=1"
  "r2_prof|40960|QWENOX_PROF=1"
  "r3_kprof|40960|QWENOX_KPROF=1"
  "r0|8192|"
  "r0|98304|"
)
if [[ -n "${RUNS:-}" ]]; then
  mapfile -t RUNS <<< "$RUNS"   # newline-separated; extra env may contain spaces
else
  RUNS=("${DEFAULT_RUNS[@]}")
fi
FAILS=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r tag depth extra <<< "$spec"
  echo; echo "=== [$(date +%H:%M:%S)] run $tag depth=$depth extra='$extra'"
  if serve "$tag" "$depth" $extra; then
    timeout 3000 python "$HERE/agent_replay.py" --depth "$depth" --tag "${tag}" \
      2>&1 | tee "logs/areplay/${tag}_d${depth}.txt"
    [[ ${PIPESTATUS[0]} == 0 ]] || FAILS+=("$tag d$depth: replay exit")
    tr '\r' '\n' < "logs/areplay_engine_${tag}_d${depth}.log" \
      | grep -E "hipError|HIP error|Exception Code|FATAL|abort|KV pool exhausted|cannot fit" | head -3 \
      | while read -r ln; do echo "  engine log: $ln"; done
  else
    FAILS+=("$tag d$depth: engine start")
  fi
  stop_eng
done

echo
if (( ${#FAILS[@]} )); then
  printf '  - %s\n' "${FAILS[@]}"
  echo "AGENT_REPLAY_DRIVER FAIL"; exit 1
fi
echo "AGENT_REPLAY_DRIVER PASS"
