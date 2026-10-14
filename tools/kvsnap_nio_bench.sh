#!/usr/bin/env bash
# kvsnap 普通 IO vs 直通 IO（O_DIRECT / NO_BUFFERING）一键对比，前台运行，约 15 分钟，
# 最后一行 NIO BENCH: PASS/FAIL。Windows Git Bash 与 Linux 都能跑。
#   bash tools/kvsnap_nio_bench.sh
#   MODES=direct NTOK=32768 bash tools/kvsnap_nio_bench.sh
#   MODEL_ARGS="主权重 ngram mtp" bash tools/kvsnap_nio_bench.sh    # 默认 models/ 下的 v2 三件套
#   NIO_BASE=X:/kvsnap NIO_TAG=nas bash tools/kvsnap_nio_bench.sh   # 缓存放别的盘/网络盘，日志 logs/nio-nas
#   NIO_EXTRA="QWENOX_KVSNAP_READERS=8 QWENOX_KVSNAP_WRITERS=8" bash tools/kvsnap_nio_bench.sh  # 额外引擎变量
# 前提：生产服务已停；引擎已编译（Windows: bash build_win.sh engine；Linux: ./build.sh）。
# 每种 IO（MODES，默认 "buffered direct"）：
#   存盘   空目录 data/kvsnap-nio-<IO>（预埋几个与缓存无关的文件/目录），生产内核组合 + 单槽，
#          prompt = oracle 前 NTOK（默认 65536）token，GEN 16 → 回合结束存盘（约 1.7 GB）；
#          live 续写 32 token 作 gold
#   恢复   重启引擎（RCKPT=0，只走 SSD），同一续写请求恢复 REPS（默认 3）次，每次 == gold
# 门槛：恢复全部 == gold；两种 IO 的 gold 相同、写出的缓存文件逐字节相同；日志里的 IO 方式
#       与模式一致；预埋的无关文件启动后仍在。耗时、读写速度、系统文件缓存变化只报告。
# 回报：从 "==== NIO 汇总 ====" 往下的内容。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) WIN=1 ;; *) WIN=0 ;; esac
if ((WIN)); then
  BIN="${BIN:-$ROOT/build/qwenox-engine-win.exe}"; PY="${PY:-python}"
else
  BIN="${BIN:-$ROOT/build/qwenox-engine}"; PY="${PY:-python3}"
fi
pth() { if ((WIN)); then cygpath -m "$1"; else echo "$1"; fi; }
MODES="${MODES:-buffered direct}"
NTOK="${NTOK:-65536}"
REPS="${REPS:-3}"
MAXCTX="${MAXCTX:-81920}"
NIO_BASE="${NIO_BASE:-$ROOT/data}"   # 缓存目录 = $NIO_BASE/kvsnap-nio-<IO>
L=logs/nio${NIO_TAG:+-$NIO_TAG}; mkdir -p $L
mkdir -p "$NIO_BASE" || { echo "无法创建 $NIO_BASE"; echo "NIO BENCH: FAIL"; exit 1; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
pass() { echo "  PASS: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

if [[ -z "${MODEL_ARGS:-}" ]]; then
  M="$ROOT/models"
  for f in qwen38-flash-next-v2.hgn qwen38-flash-next-ngram.hgn qwen38-flash-next-mtp.hgn; do
    [[ -f "$M/$f" ]] || { echo "缺 $M/$f；用 MODEL_ARGS=\"主权重 ngram mtp\" 指定"; echo "NIO BENCH: FAIL"; exit 1; }
  done
  MODEL_ARGS="$(pth "$M/qwen38-flash-next-v2.hgn") $(pth "$M/qwen38-flash-next-ngram.hgn") $(pth "$M/qwen38-flash-next-mtp.hgn")"
fi
[[ -f "$BIN" ]] || { echo "缺引擎：$BIN"; echo "NIO BENCH: FAIL"; exit 1; }
if ((WIN)); then
  tasklist 2>/dev/null | grep -qi qwenox && { echo "检测到 qwenox 进程在运行，请先停掉服务"; echo "NIO BENCH: FAIL"; exit 1; }
else
  pgrep -f 'qwenox-engine|qwenox-api' > /dev/null && { echo "检测到 qwenox 进程在运行，请先停掉服务"; echo "NIO BENCH: FAIL"; exit 1; }
fi
echo "head $(git log --oneline -1 | cut -c1-70)  dirty: $(git status --short | wc -l)"
echo "bin  $BIN ($(date -r "$BIN" +%F_%T 2>/dev/null))  MODES=$MODES NTOK=$NTOK REPS=$REPS"
echo "cache base $NIO_BASE  logs $L  extra env: ${NIO_EXTRA:-none}"

prod_env() {  # start_win.sh / start.sh 同款生产内核组合，单槽
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_PARALLEL=1 QWENOX_PREFILL_CHUNK=8192
  ((WIN)) || export QWENOX_PLE_URING=1
  export QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=40
}

ENG_PID=""
serve() {  # serve TAG ENV=V... : 起引擎到 listening（最多 10 分钟）
  local tag=$1; shift
  prod_env
  local e; for e in ${NIO_EXTRA:-} "$@"; do export "$e"; done
  local log=$ROOT/$L/$tag.log margs
  read -r -a margs <<< "$MODEL_ARGS"
  (cd build && exec "$BIN" "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
      --maxctx "$MAXCTX" > "$log" 2>&1) &
  ENG_PID=$!
  local t0; t0=$(date +%s)
  while :; do
    if tr '\r' '\n' < "$log" | grep -q "serve: listening"; then
      echo "  [$tag] listening after $(( $(date +%s) - t0 ))s"
      tr '\r' '\n' < "$log" | grep -E 'kvsnap: (dir|index|removed|direct)' | sed 's/^/    /'
      return 0
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
  if ((WIN)); then
    powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  fi
  [[ -n "$ENG_PID" ]] && wait $ENG_PID 2>/dev/null
  sleep 4; ENG_PID=""
}
trap stop_eng EXIT
crashed() { tr '\r' '\n' < "$1" | grep -E 'hipError|HIP error|Segmentation|Aborted|FATAL|terminate called' | head -2; }

for m in $MODES; do
  case $m in buffered) d=0 ;; direct) d=1 ;; *) echo "未知 IO 模式 $m"; exit 1 ;; esac
  DIR="$NIO_BASE/kvsnap-nio-$m"
  rm -rf "$DIR"; mkdir -p "$DIR/tmp_keep" "$DIR/s_keep"
  echo keep > "$DIR/notes.txt"; echo keep > "$DIR/tmp_notes.txt"; echo keep > "$DIR/s_keep/x"
  WDIR="$(pth "$DIR")"
  stamp "$m：存盘（prompt $NTOK token）"
  if serve save-$m QWENOX_KVSNAP_DIR="$WDIR" QWENOX_KVSNAP_DIRECT=$d; then
    $PY tools/kvsnap_nio_bench.py mem --dir "$DIR" > $L/mem_${m}_before.json
    $PY tools/kvsnap_nio_bench.py save --state $L/$m.json --ntok "$NTOK" || fail "$m 存盘（KVSYNC 计数不对）"
    $PY tools/kvsnap_nio_bench.py mem --dir "$DIR" > $L/mem_${m}_saved.json
    c="$(crashed $L/save-$m.log)"; [[ -z "$c" ]] || fail "$m 存盘引擎崩溃：$c"
    stop_eng
  else fail "$m 存盘引擎启动失败（$L/save-$m.log）"; stop_eng; continue; fi
  if [[ -f "$DIR/notes.txt" && -f "$DIR/tmp_notes.txt" && -f "$DIR/s_keep/x" && -d "$DIR/tmp_keep" ]]; then
    pass "$m 目录里与缓存无关的文件/目录启动后都还在"
  else fail "$m 启动清理删掉了无关文件：$(ls "$DIR" | grep -v -E '^(c_|m_|k_)' | tr '\n' ' ')"; fi
  stamp "$m：重启后恢复 ×$REPS"
  if serve restore-$m QWENOX_KVSNAP_DIR="$WDIR" QWENOX_KVSNAP_DIRECT=$d QWENOX_RCKPT=0; then
    $PY tools/kvsnap_nio_bench.py restore --state $L/$m.json --reps "$REPS" && pass "$m 恢复 ×$REPS == gold" || fail "$m 恢复输出 != gold"
    $PY tools/kvsnap_nio_bench.py mem --dir "$DIR" > $L/mem_${m}_restored.json
    c="$(crashed $L/restore-$m.log)"; [[ -z "$c" ]] || fail "$m 恢复引擎崩溃：$c"
    stop_eng
  else fail "$m 恢复引擎启动失败（$L/restore-$m.log）"; stop_eng; fi
done

echo; echo "==== NIO 汇总 ===="
$PY tools/kvsnap_nio_bench.py summary --logs $L --modes "$MODES" || fail "汇总检查（见上）"
if [[ "$MODES" == *buffered* && "$MODES" == *direct* ]]; then
  $PY tools/kvsnap_nio_bench.py cmpdir "$NIO_BASE/kvsnap-nio-buffered" "$NIO_BASE/kvsnap-nio-direct" \
    && pass "两种 IO 写出的缓存文件逐字节相同" || fail "两种 IO 写出的缓存文件不同"
fi
echo
if ((${#FAILS[@]})); then
  printf '  - %s\n' "${FAILS[@]}"; echo "NIO BENCH: FAIL"; exit 1
fi
echo "（测完可删 $NIO_BASE/kvsnap-nio-*，约 2×1.8 GB）"
echo "NIO BENCH: PASS"
