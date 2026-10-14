#!/usr/bin/env bash
# tools/viz_verify_win.sh [BASE_ENGINE_EXE] — QWENOX_VIZ 运行时可视化埋点一键验证（Windows Git Bash）。
# 用 service.conf 的权重与生产开关（同 start_win.sh）依次启动引擎：
#   [base]（可选，参数给出的不带埋点的引擎） / off（VIZ 关）/ on（VIZ 开）/ on1（VIZ 开 + 单路）
# 检查 greedy 输出逐 token 一致、VIZ 开的 decode 速度、路由与 top-5 数据的位置对齐，
# 最后打印 PASS / FAIL。约 6–10 分钟（每种配置加载一次权重）。生产服务须先停掉。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f service.conf ]] && source service.conf
BASE_EXE="${1:-}"
MODEL_DIR="${MODEL_DIR:-./models}"
WEIGHTS="${WEIGHTS:-$(sed -n 's/^# start_win: weights=\([a-z0-9]*\).*/\1/p' service.conf 2>/dev/null | head -1)}"
if [[ "$WEIGHTS" == gguf ]]; then
  MODEL_FILE="$GGUF_FILE"; NGRAM_FILE="$GGUF_FILE"; OVERLAY_FILE=; MTP_FILE=
  VISION_FILE="${GGUF_VISION_FILE-}"
  export QWENOX_GGUF_MTP="${GGUF_MTP_FILE:-none}"
else
  MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/heretic.hgn}"
  NGRAM_FILE="${NGRAM_FILE:-$MODEL_FILE}"
  MTP_FILE="${MTP_FILE-$MODEL_DIR/heretic-mtp.hgn}"
  VISION_FILE="${VISION_FILE-$MODEL_DIR/heretic-vision.hgn}"
  unset QWENOX_GGUF_MTP
fi
OVERLAY_FILE="${OVERLAY_FILE-}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
MTP_GAMMA="${MTP_GAMMA:-0}"
KV_PAGED="${KV_PAGED:-1}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-0}"
PARALLEL="${PARALLEL:-4}"
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
if (( MTP_GAMMA > 0 )); then export QWENOX_SPEC_GAMMA="$MTP_GAMMA"; else unset QWENOX_SPEC_GAMMA; fi
unset QWENOX_KV_PAGED QWENOX_KV_POOL_TOKENS QWENOX_VIZ
if (( KV_PAGED )); then
  export QWENOX_KV_PAGED=1
  if (( KV_POOL_TOKENS )); then export QWENOX_KV_POOL_TOKENS="$KV_POOL_TOKENS"; fi
fi
export QWENOX_PARALLEL="$PARALLEL"
export QWENOX_ROPE_FACTOR="${ROPE_FACTOR:-1}" QWENOX_ROPE_ORIGINAL_CTX="${ROPE_ORIGINAL_CTX:-262144}"
export QWENOX_ROPE_BETA_FAST="${ROPE_BETA_FAST:-32}" QWENOX_ROPE_BETA_SLOW="${ROPE_BETA_SLOW:-1}"
export QWENOX_ROPE_ATTN_SCALE="${ROPE_ATTN_SCALE:-0}"

args=("$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || args+=("$OVERLAY_FILE")
[[ "$NGRAM_FILE" -ef "$MODEL_FILE" ]] || args+=("$NGRAM_FILE")
[[ -z "$MTP_FILE" ]] || args+=("$MTP_FILE")
args+=(--serve --host 127.0.0.1 --port @PORT@ --maxctx "$MAX_CONTEXT")
[[ -z "$VISION_FILE" ]] || args+=(--vision-tower "$VISION_FILE")

PY=python; command -v python >/dev/null 2>&1 || PY=python3
base=(); [[ -z "$BASE_EXE" ]] || base=(--base "$BASE_EXE")
# SPEED_ROUNDS=N：只做测速 A/B（base/off/on 轮流加载 N 轮，每次 4 遍，丢第 1 遍取中位数）
[[ -z "${SPEED_ROUNDS:-}" ]] || base+=(--speed-rounds "$SPEED_ROUNDS")
exec "$PY" tools/viz_verify.py --engine build/qwenox-engine-win.exe "${base[@]}" -- "${args[@]}"
