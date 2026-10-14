#!/usr/bin/env bash
# vram_conc_win.sh — Windows：并发路数 × 页池大小的显存占用（hgn v2，托盘生产配置）。
#
#   bash tools/vram_conc_win.sh                         # 512K 页池，1/2/4 路，约 5 分钟/配置
#   PARS="1 2" POOL=0 SHORT=1 bash tools/vram_conc_win.sh   # 页池跟随 MAX_CONTEXT，只测启动+一条 8K
#
# 每个配置起一次引擎（端口 8732，需先停掉生产服务），阶段：就绪 → 单条 8K → 2×32K 同时 → 2×100K 同时
# （SHORT=1 只跑前两个）。tools/vram_conc_sampler.ps1 每约 2 s 记驱动计数器（任务管理器的专用/共享
# GPU 内存，整卡与引擎进程）和进程内存；tools/vram_conc.py 发请求并在每阶段后取引擎 MEM。
# 另记 arena 实际用量、就绪后的运行期分配与 hipMalloc 回退。汇总在 logs/vram_conc${SUF}/summary.txt。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="$(pwd)"; L=$A/logs/vram_conc${SUF:-}; mkdir -p $L
OUT=$L/summary.txt; : > $OUT
PARS="${PARS:-1 2 4}"
log() { echo "$*" | tee -a $OUT; }
log "head $(git log --oneline -1 | cut -c1-60)  $(date '+%F %T')  PARS=$PARS"
prod_env() {
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=64 QWENOX_RCKPT_MAX=8 QWENOX_KV_PAGED=1
  if (( ${POOL:-524288} )); then export QWENOX_KV_POOL_TOKENS=${POOL:-524288}; else unset QWENOX_KV_POOL_TOKENS; fi
  export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144 QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_API_MAX_IMAGES=32
  export QWENOX_ARENA_DEBUG=1
  export QWENOX_KVSNAP_DIR="$(cygpath -w $L/kvsnap)"
  unset QWENOX_SPEC_GAMMA QWENOX_PREFILL_CHUNK QWENOX_CONC_PREFILL_CHUNK QWENOX_VIZ
  unset QWENOX_GGUF QWENOX_GGUF_DENSE QWENOX_GGUF_PLE QWENOX_GGUF_EXPERTS QWENOX_GGUF_MTP
}
killeng() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  sleep 25
}
M=./models
for par in $PARS; do
  tag=p$par; lg=$L/engine_$tag.log; csv=$L/samples_$tag.csv; stop=$L/stop_$tag
  rm -rf $L/kvsnap; rm -f $stop
  log "=== PARALLEL=$par  (MAX_CONTEXT 262144, KV_POOL_TOKENS ${POOL:-524288}, v2+ngram+mtp+vision)"
  prod_env; export QWENOX_PARALLEL=$par
  powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w tools/vram_conc_sampler.ps1)" -Out "$(cygpath -w $csv)" -Stop "$(cygpath -w $stop)" &
  spid=$!
  sleep 3
  log "  $(date +%T) engine start"
  t0=$SECONDS
  build/qwenox-engine-win.exe $M/qwen38-flash-next-v2.hgn $M/qwen38-flash-next-ngram.hgn $M/qwen38-flash-next-mtp.hgn \
     --serve --host 127.0.0.1 --port 8732 --maxctx 262144 --vision-tower $M/qwen38-flash-next-vision.hgn > $lg 2>&1 &
  epid=$!; ok=0
  while (( SECONDS - t0 < 900 )); do
    grep -q 'serve: listening' $lg 2>/dev/null && { ok=1; break; }
    kill -0 $epid 2>/dev/null || break; sleep 2
  done
  if (( ! ok )); then log "  START FAIL"; tr '\r' '\n' < $lg | tail -8 | tee -a $OUT; touch $stop; killeng; continue; fi
  log "  $(date +%T) ready in $((SECONDS - t0)) s"
  nready=$(tr '\r' '\n' < $lg | wc -l)
  tr '\r' '\n' < $lg | grep -E "^devarena:|^arena: |^weight arena: [0-9]|^memory\[|^\[kvpage\]|sequence slots|borrow armed|row backup|prefill chunk" | sed 's/^/    /' | tee -a $OUT
  tr '\r' '\n' < $lg | awk '/^arena-debug:/ && /@ [0-9.]+ GiB/ { e = $6 + $3 / 1024; if (e > m) m = e }
      /hipMalloc fallback/ { fb++; fbm += $3 }
      END { printf "    STARTUP ARENA USED %.2f GiB; hipMalloc fallbacks %d (%.1f MiB)\n", m, fb, fbm }' | tee -a $OUT
  python -u tools/vram_conc.py data/qsa-oracle 2>&1 | sed 's/^/  /' | tee -a $OUT
  sleep 2; touch $stop; sleep 3
  log "  runtime allocations after ready (arena-debug / fallbacks / memory lines):"
  tr '\r' '\n' < $lg | tail -n +$((nready+1)) | grep -E "arena-debug|fallback|^memory\[|hipMalloc|out of memory|KvOverflow|evict" | head -60 | sed 's/^/    /' | tee -a $OUT
  tr '\r' '\n' < $lg | tail -n +$((nready+1)) | awk '/^arena-debug:/ { n++; s += $3 } /hipMalloc fallback/ { fb++; fbm += $3 }
      END { printf "    RUNTIME arena-debug allocs %d (%.1f MiB), fallbacks %d (%.1f MiB)\n", n, s, fb, fbm }' | tee -a $OUT
  # sampler peaks per phase window
  python - "$csv" <<'PY' | sed 's/^/  /' | tee -a $OUT
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))
k=['adapter_ded_gb','adapter_sh_gb','proc_ded_gb','proc_sh_gb','ws_gb','priv_gb']
if r:
  print('samples %d  %s..%s' % (len(r), r[0]['t'], r[-1]['t']))
  print('MAX   ' + '  '.join('%s %.2f' % (x, max(float(a[x]) for a in r)) for x in k))
  print('LAST  ' + '  '.join('%s %.2f' % (x, float(r[-1][x])) for x in k))
PY
  killeng
  wait $spid 2>/dev/null
done
log "MEMCONC DONE $(date '+%F %T')"
