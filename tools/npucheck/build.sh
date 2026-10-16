#!/usr/bin/env bash
# 转发到根构建（产物在 build/）：bash build_win.sh npu
# XRT 头/def 与 xclbin/insts 默认取自 ~/npuprobe，可用 NPUASSETS 环境变量覆盖。
set -uo pipefail
exec bash "$(cd "$(dirname "$0")" && pwd)/../../build_win.sh" npu
