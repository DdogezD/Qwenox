#!/usr/bin/env bash
# bench/codeneedle/ab_win.sh - one-shot A/B: does QWENOX_MOE_A4 hurt long-context
# reading? (Windows Git Bash; Linux works too.) v2 hgn, serial, ~30-40 min total.
#
#   bash bench/codeneedle/ab_win.sh
#
#   A4=3        value for the B run (default 3; 1 = gate/up only, 2 = down only)
#   SEEDS=4     sampled answers per question (plus 1 greedy)
#   ONLY=L16K   run only haystacks whose id starts with this (quick look)
#   BIN=...     engine binary (default build/qwenox-engine-win.exe)
#   MODEL_ARGS  model files (default v2 hgn + ngram + mtp under models/)
#
# 1. engine with MOE_A4 unset -> bench/codeneedle/run.py run --tag cn_a4off
# 2. engine with MOE_A4=$A4   -> run --tag cn_a4on  (the A4 log line must appear)
# 3. run.py cmp: per-length accuracy, flips (exact McNemar), logprob shift.
# Both runs: QWENOX_KVSNAP=0 (no SSD snapshots, so neither run can reuse the
# other's prompt KV), RAM checkpoints on (one prefill per haystack). Needs the
# service stopped (whole-GPU arena). Last line CODENEEDLE PASS|FAIL.
# Logs: logs/codeneedle/eng_*.log, logs/codeneedle/cn_*.json, logs/codeneedle/ab.txt
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
L=logs/codeneedle; mkdir -p $L
exec > >(tee $L/ab.txt) 2>&1
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then EXE=qwenox-engine-win.exe; else EXE=qwenox-engine; fi
BIN=${BIN:-$ROOT/build/$EXE}
M=$ROOT/models
MODEL_ARGS="${MODEL_ARGS:-$(win $M/qwen38-flash-next-v2.hgn) $(win $M/qwen38-flash-next-ngram.hgn) $(win $M/qwen38-flash-next-mtp.hgn)}"
MAXCTX=${MAXCTX:-262144}
A4=${A4:-3}
SEEDS=${SEEDS:-4}
ONLY=${ONLY:-}
PY=${PY:-python}
command -v $PY > /dev/null || PY=python3
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

[[ -f "$BIN" ]] || { echo "missing engine: $BIN"; echo "CODENEEDLE FAIL"; exit 1; }
REV=$($PY -c "import json;print(json.load(open('bench/codeneedle/set.json',encoding='utf-8'))['rev'])")
git cat-file -e "$REV^{commit}" 2> /dev/null || {
  echo "question set needs commit ${REV:0:10} (git fetch first)"; echo "CODENEEDLE FAIL"; exit 1; }
if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "a qwenox process is running - stop the service first (the test needs the whole GPU)"
  echo "CODENEEDLE FAIL"; exit 1
fi
echo "root $ROOT  head $(git log --oneline -1 | cut -c1-60)"
echo "engine $BIN"
echo "set    bench/codeneedle/set.json (source rev ${REV:0:10}); B run: MOE_A4=$A4; $SEEDS sampled + 1 greedy per question"
echo "ETA    ~15-20 min per run (engine load + 576K prefill tokens + 720 short answers)"

prod_env() {  # start_win.sh production kernel set, 1 slot; $1 = MOE_A4 (0 = unset)
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_PARALLEL=1 QWENOX_PREFILL_CHUNK=${CHUNK:-6144}
  export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=8
  if (( $1 )); then export QWENOX_MOE_A4=$1; fi
  return 0
}

ENG_PID=""
serve() {  # serve TAG A4
  local tag=$1
  prod_env "$2"
  local log=$ROOT/$L/eng_$tag.log
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
    (( $(date +%s) - t0 > 600 )) && { echo "  [$tag] timeout"; return 1; }
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

engine_errors() {
  local e
  e=$(tr '\r' '\n' < $L/eng_$1.log | grep -E "hipError|HIP error|Exception Code|FATAL|abort|KV pool exhausted|cannot fit|rejected" | head -3)
  if [[ -n "$e" ]]; then echo "$e" | sed 's/^/  engine log: /'; fail "engine error in $1"; fi
}
run() {  # run TAG
  $PY bench/codeneedle/run.py run --tag "$1" --seeds "$SEEDS" ${ONLY:+--only "$ONLY"}
  [[ $? == 0 ]] || fail "$1 run had errors"
}

stamp "1/3 MOE_A4 unset (reference)"
if serve a4off 0; then
  run cn_a4off; engine_errors a4off
  tr '\r' '\n' < $L/eng_a4off.log | grep -q "W4A4 iu4 on" && fail "A4 path ran with the switch off"
else
  fail "engine (A4 off) did not start"
fi
stop_eng

stamp "2/3 MOE_A4=$A4"
if serve a4on "$A4"; then
  run cn_a4on; engine_errors a4on
  tr '\r' '\n' < $L/eng_a4on.log | grep -m1 "QWENOX_MOE_A4" | sed 's/^/  engine log: /'
  tr '\r' '\n' < $L/eng_a4on.log | grep -q "W4A4 iu4 on" || fail "A4 path never ran (no log line)"
else
  fail "engine (A4=$A4) did not start"
fi
stop_eng

stamp "3/3 compare"
if [[ -f $L/cn_a4off.json && -f $L/cn_a4on.json ]]; then
  $PY bench/codeneedle/run.py cmp --a cn_a4off --b cn_a4on | grep -v "^CODENEEDLE"
  RC=${PIPESTATUS[0]}
else
  RC=1; fail "missing run results"
fi
echo
if (( ${#FAILS[@]} )); then
  printf '  - %s\n' "${FAILS[@]}"
  echo "CODENEEDLE FAIL"; exit 1
fi
if (( RC )); then echo "CODENEEDLE FAIL (MOE_A4=$A4 measurably worse, see table)"; exit 1; fi
echo "CODENEEDLE PASS (MOE_A4=$A4: no measurable degradation)"
