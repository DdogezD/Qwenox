#!/usr/bin/env bash
# m2ab.sh -- M2 (shared-expert gate/up on NPU) A/B in the production prefill config.
# Git Bash on the Windows host, started via ~/aggrprobe/bg.ps1.
#   runs: kprof-sync GPU-only (ceiling) + off/on x3 alternating, 32K prompt, chunk 6144
# Summary: ~/npuprobe/logs/m2ab/summary.log (appended after every run, so a hang keeps it).
set -u
ENG="$HOME/Workspace/gfx1151-engine"
L="$HOME/npuprobe/logs/m2ab"; mkdir -p "$L"; S="$L/summary.log"
LEN=${LEN:-32768}; CHUNK=${CHUNK:-6144}; REPS=${REPS:-3}
say() { echo "$*" >>"$S"; }
: >"$S"
say "== m2ab start $(date '+%F %T')  len=$LEN chunk=$CHUNK reps=$REPS host=$(hostname)"
say "pmode: $(/c/Windows/System32/AMD/xrt-smi.exe examine -r platform 2>&1 | tr -d '\r' | sed -n 's/.*Power Mode *: *//p' | head -1)"
AIE=$(/c/Windows/System32/AMD/xrt-smi.exe examine --report aie-partitions 2>&1 | tr -d '\r')
if ! grep -q 'No hardware contexts' <<<"$AIE"; then
  say "NPU has other tenants (Windows AI?) -- refusing to run:"; say "$AIE"; say "M2AB RESULT: FAIL (NPU not exclusive)"; exit 1
fi
say "NPU exclusive: yes"

run() {  # $1=tag  $2=npu(0/1)  $3=extra env
  local tag=$1 npu=$2 extra=${3:-}
  local log="$L/$tag.log"
  local t0=$(date +%s)
  ( cd "$ENG" && env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=$CHUNK QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 QWENOX_INDEX_FUSED2=1 \
      QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 QWENOX_KPROF=1 \
      $( ((npu)) && echo QWENOX_NPU=1 ) $extra \
      timeout 900 build/qwenox-engine-win.exe models/qwen38-flash-next-v2.hgn models/qwen38-flash-next-ngram.hgn \
      --tokens-file "data/qsa-oracle/$LEN.tokens" --gen 1 --maxctx $((LEN + 8192)) ) >"$log" 2>&1
  local rc=$?
  local pf=$(grep -E 'prefill: [0-9]+ tokens in' "$log" | tail -1 | tr -d '\r' | sed -E 's/.*prefill: //')
  # steady-state = sum of kprof segsum over chunks 2..n (chunk 1 carries NPU weight encoding)
  local ss=$(grep -E '^kprof base=' "$log" | tr -d '\r' | awk '{for(i=1;i<=NF;i++) if($i ~ /^segsum=/){sub("segsum=","",$i); v[NR]=$i}} END{s=0; for(k=2;k<=NR;k++) s+=v[k]; printf "%.1f ms over %d chunks (chunk1 %.1f)", s, NR-1, v[1]}')
  local npul=$(grep -E 'NPU' "$log" | tr -d '\r' | head -3 | tr '\n' ' ')
  say "$tag rc=$rc $(( $(date +%s) - t0 ))s | prefill: $pf | steady: $ss"
  [[ -n "$npul" ]] && say "   npu log: $npul"
  return $rc
}

R=PASS
# 1) ceiling: per-segment GPU time with host-synced marks, NPU off
run kprof_sync 0 "QWENOX_KPROF_SYNC=1" || R="FAIL kprof_sync"
grep -E '^kprof base=' "$L/kprof_sync.log" | tr -d '\r' | sed -n 2p | tr '|' '\n' | grep -E 'moe:shared|segsum|moe:' | sed 's/^/   /' >>"$S"
# 2) A/B alternating
for i in $(seq "$REPS"); do
  [[ $R == PASS ]] || break
  run "off_$i" 0 || { R="FAIL off_$i"; break; }
  run "on_$i" 1 || { R="FAIL on_$i"; break; }
  grep -q 'NPU 卸载就绪' "$L/on_$i.log" || { R="FAIL on_$i (NPU not enabled)"; break; }
  grep -q -E 'NPU:|回退 GPU' "$L/on_$i.log" && { R="FAIL on_$i (NPU fell back)"; break; }
done

say ""
say "---- tok/s (from prefill lines)"
for m in off on; do
  say "$m: $(for f in "$L"/${m}_*.log; do grep -E 'prefill: [0-9]+ tokens in' "$f" | tail -1 | tr -d '\r' | grep -oE '[0-9.]+ tok/s' | head -1; done | tr '\n' ' ')"
done
say "---- steady-state segsum (chunks 2..n, ms)"
for m in off on; do
  say "$m: $(for f in "$L"/${m}_*.log; do grep -E '^kprof base=' "$f" | tr -d '\r' | awk '{for(i=1;i<=NF;i++) if($i ~ /^segsum=/){sub("segsum=","",$i); v[NR]=$i}} END{s=0; for(k=2;k<=NR;k++) s+=v[k]; printf "%.1f ", s}'; done)"
done
say "WHEA since start: $(powershell -NoProfile -Command "(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; StartTime=(Get-Date).AddHours(-1)} -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null | tr -d '\r')"
say "M2AB RESULT: $R   ($(date '+%F %T'))"
