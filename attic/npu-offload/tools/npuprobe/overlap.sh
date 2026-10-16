#!/usr/bin/env bash
# overlap.sh -- GPU (real engine prefill) + NPU (gufo GEMM loop) at the same time,
# with a WHEA watchdog. Git Bash on Windows.
#
#   bash overlap.sh gpu     <len>          engine prefill alone (baseline)
#   bash overlap.sh overlap <len> [tag]    NPU loop + engine prefill together
#   bash overlap.sh loop    <len> <n> [tag]  NPU loop + n back-to-back prefills
#
# <len>: 8192 / 32768 / 65536 / 131072 (data/qsa-oracle/<len>.tokens).
# Any new WHEA-Logger event stops both loads at once (on Linux the Data Fabric
# logged deferred errors seconds before the host hung, ROCm/TheRock#8709).
# Every NPU result is compared bitwise; the NPU log is flushed to disk each second.
set -u
P="$(cd "$(dirname "$0")" && pwd)"
ENGROOT="${ENGROOT:-$HOME/Workspace/gfx1151-engine}"
MODE="${1:?mode}"; LEN="${2:?len}"
N=1; TAG=""
if [[ "$MODE" == loop ]]; then N="${3:?n}"; TAG="${4:-}"; else TAG="${3:-}"; fi
TAG="${TAG:-${MODE}_${LEN}_$(date +%H%M%S)}"
L="$P/logs/$TAG"; mkdir -p "$L"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
STOP="$L/STOP"; rm -f "$STOP"
NPU_PMODE=$("/c/Windows/System32/AMD/xrt-smi.exe" examine -r platform 2>&1 | tr -d '\r' | sed -n 's/.*Power Mode *: *//p' | head -1)
PLAN=$(powershell -NoProfile -Command '(Get-CimInstance -Namespace root/cimv2/power -ClassName Win32_PowerPlan -Filter "IsActive=True").ElementName' 2>/dev/null | tr -d '\r')
echo "== $TAG  mode=$MODE len=$LEN n=$N  NPU pmode=$NPU_PMODE  power plan=$PLAN  since=$SINCE"

whea() { powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$P/whea.ps1")" -Since "$SINCE" 2>/dev/null | tr -d '\r'; }
W0="$(whea | head -1)"
[[ "$W0" =~ ^[0-9]+$ ]] || { echo "WHEA watchdog not working: '$W0'"; echo "RESULT: FAIL (watchdog)"; exit 1; }

run_engine() {  # $1 = index
  # same prefill switches as tools/pp_win.sh
  ( cd "$ENGROOT" && env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=8192 QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 QWENOX_INDEX_FUSED2=1 \
      QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      build/qwenox-engine-win.exe models/qwen38-flash-next-v2.hgn models/qwen38-flash-next-ngram.hgn \
      --tokens-file "data/qsa-oracle/$LEN.tokens" --gen 1 --maxctx $((LEN + 8192)) ) >"$L/engine_$1.log" 2>&1
}

# clocks/power every 250 ms for the whole test (AWAKE=1 also keeps the display on)
( cd "$P/out" && ./pmlog.exe watch 3000 "$(cygpath -w "$L/pm.log")" ${AWAKE:+awake} ) >/dev/null 2>&1 &
PMPID=$!

NPID=""
if [[ "$MODE" != gpu ]]; then
  ( cd "$P/out" && ./npuio.exe ../npu stress 100000 "$(cygpath -w "$L/npu.log")" "$(cygpath -w "$STOP")" ) >"$L/npu.stdout" 2>&1 &
  NPID=$!
  for i in $(seq 120); do grep -q -E 'READY|SETUP_FAIL' "$L/npu.log" 2>/dev/null && break; sleep 0.5; done
  grep -q READY "$L/npu.log" || { echo "NPU stress did not start"; cat "$L/npu.stdout" | tail -5; echo "RESULT: FAIL"; exit 1; }
  sleep 6   # NPU alone baseline
  echo "NPU alone: $(grep '^t=' "$L/npu.log" | tail -3 | sed -E 's/.*ms\/run=([0-9.]+).*/\1/' | tr '\n' ' ') ms/run"
fi

ABORT=""
T0=$(date +%s)
for k in $(seq "$N"); do
  run_engine "$k" & EPID=$!
  echo "engine $k started (t=$(( $(date +%s) - T0 )) s)"
  while kill -0 "$EPID" 2>/dev/null; do
    sleep 1
    w=$(whea | head -1)
    if [[ "$w" =~ ^[0-9]+$ ]] && (( w > W0 )); then ABORT="WHEA ($w events)"; break; fi
    if [[ -n "$NPID" ]] && ! kill -0 "$NPID" 2>/dev/null; then ABORT="NPU loop exited"; break; fi
    if grep -q -E 'MISMATCH|TIMEOUT' "$L/npu.log" 2>/dev/null; then ABORT="NPU result error"; break; fi
  done
  if [[ -n "$ABORT" ]]; then
    touch "$STOP"; taskkill //F //IM qwenox-engine-win.exe >/dev/null 2>&1; wait "$EPID" 2>/dev/null
    break
  fi
  wait "$EPID"; echo "engine $k rc=$? (t=$(( $(date +%s) - T0 )) s): $(grep -E 'prefill: [0-9]+ tokens in' "$L/engine_$k.log" | tail -1 | tr -d '\r')"
done

if [[ -n "$NPID" ]]; then
  [[ -n "$ABORT" ]] || sleep 5   # NPU alone again
  touch "$STOP"; for i in $(seq 40); do kill -0 "$NPID" 2>/dev/null || break; sleep 0.25; done
  kill "$NPID" 2>/dev/null
fi

kill "$PMPID" 2>/dev/null; taskkill //F //IM pmlog.exe >/dev/null 2>&1
echo "---- clocks while the GPU is busy (ACT_GFX>=50): median / min / max"
awk '!/^#/ && $7>=50 {n++; g[n]=$3; m[n]=$4; s[n]=$5; a2[n]=$7; w[n]=$10}
  function med(x, n,  i,j,t,y){for(i=1;i<=n;i++)y[i]=x[i]; for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(y[j]<y[i]){t=y[i];y[i]=y[j];y[j]=t}; return y[int((n+1)/2)] " / " y[1] " / " y[n]}
  END{ if(!n){print "no busy samples"; exit}; print "samples " n; print "GFXCLK " med(g,n); print "MEMCLK " med(m,n); print "SOCCLK " med(s,n); print "ACT_GFX " med(a2,n); print "GFX_W " med(w,n)}' "$L/pm.log"
echo "MEMCLK values seen (all samples): $(awk '!/^#/{c[$4]++} END{for(k in c) printf "%s:%d ", k, c[k]}' "$L/pm.log")"
echo "---- engine prefill lines"
for f in "$L"/engine_*.log; do echo "$(basename "$f"): $(grep -E 'prefill: [0-9]+ tokens in' "$f" | tr -d '\r' | sed -E 's/.*prefill: //' | tr '\n' ';')"; done
if [[ -n "$NPID" ]]; then
  echo "---- NPU (ms/run per second; GPU busy in the middle)"
  grep '^t=' "$L/npu.log" | awk '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="ms/run") v=a[2]; if(a[1]=="worst_wait_ms") w=a[2]}; n++; s=s sprintf("%.2f ", v); if(w>mw) mw=w} END {print s; print "seconds=" n " worst_wait_ms=" mw}' | fold -w 200
  grep -E 'MISMATCH|TIMEOUT|END|SETUP' "$L/npu.log"
fi
echo "---- WHEA since $SINCE: $(whea | head -1)"; whea | tail -n +2 | head -20
if [[ -n "$ABORT" ]]; then echo "RESULT: STOPPED - $ABORT"; exit 2; fi
if [[ -n "$NPID" ]] && ! grep -q 'END .*mismatches=0 rc=0' "$L/npu.log"; then echo "RESULT: FAIL (NPU)"; exit 1; fi
echo "RESULT: PASS (no WHEA events, NPU results all bit-identical)"
