#!/usr/bin/env bash
# start.sh 已按权重格式拆成两个启动器（参数与 --check 相同）。
cat >&2 <<'EOF'
error: start.sh has been split; pick the launcher for your weight format:
  bash start_hgn.sh    hgn weights (models/qwen38-flash-next-*.hgn)
  bash start_gguf.sh   GGUF weights (models/Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf, same files as llama.cpp)
Both read service.conf and list any missing weight files. See QUICKSTART.md.
EOF
exit 1
