#!/usr/bin/env bash
# hgn 权重启动器（Linux）：models/ 下的 qwen38-flash-next-*.hgn。llama.cpp GGUF 权重请用 start_gguf.sh。
# 前台运行，Ctrl+C 同时停止本次启动的 API 和引擎；--check 只检查配置、不启动。
# 路径与部署参数见 service.conf（"hgn 权重"一段），同名环境变量优先。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LAUNCHER=start_hgn.sh
source "$ROOT/tools/serve_common.sh"
serve_init "$@"

for f in "$MODEL_FILE" "$NGRAM_FILE" "$OVERLAY_FILE" "$MTP_FILE" "$VISION_FILE"; do
  [[ -z "$f" || "$f" == *.hgn ]] || fail "$LAUNCHER only accepts .hgn weights: $f (for GGUF use bash start_gguf.sh)"
done
need 'main model' "$MODEL_FILE"
NGRAM_FILE=${NGRAM_FILE:-$MODEL_FILE}
need 'n-gram table' "$NGRAM_FILE" 'w4b shares the main model file; v2 points to *-ngram.hgn'
# overlay is optional: warn and skip when missing (e.g. config left over after switching to v2 weights).
if [[ -n "$OVERLAY_FILE" && ! -r "$OVERLAY_FILE" ]]; then
  echo "warning: overlay not found: $OVERLAY_FILE, skipped (v2 weights use no overlay; set OVERLAY_FILE=\"\" to silence this)" >&2
  OVERLAY_FILE=
fi
[[ -z "$MTP_FILE" ]] || need 'MTP draft' "$MTP_FILE" 'set MTP_FILE="" to fall back to the overlay built-in draft head'
[[ -z "$VISION_FILE" ]] || need 'vision tower' "$VISION_FILE" 'set VISION_FILE="" for text-only mode'
MISSING_HINT='hgn weights are converted by tools/flashnext2hgn.py (see CONVERT.md); for llama.cpp GGUF weights use bash start_gguf.sh (see GGUF.md).'

FORMAT=hgn
MAIN_MODEL=$MODEL_FILE
MODEL_ARGS=("$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || MODEL_ARGS+=("$OVERLAY_FILE")
# 独立 n-gram 文件（v2）排在 overlay 之后、MTP 之前；与主模型同一文件（w4b）则不重复传。
[[ "$NGRAM_FILE" -ef "$MODEL_FILE" ]] || MODEL_ARGS+=("$NGRAM_FILE")
[[ -z "$MTP_FILE" ]] || MODEL_ARGS+=("$MTP_FILE")
VISION=$VISION_FILE
serve_run
