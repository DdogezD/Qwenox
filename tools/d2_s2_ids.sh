#!/usr/bin/env bash
# d2_s2_ids.sh — D2-S2a 验收④：会合调度（VerifyPool）下并发逐 bit 一致。
# QWENOX_PARALLEL=4 起 serve（仅引擎行协议），eng_cli 直连、drafter=1
# （MTP-only spec_loop，即 S2a 挂接的路径），固定 prompt、greedy、257 token：
#   base  单跑基线 ids md5
#   c2    2 并发同 prompt，两客户端 ids md5 均须 ≡ base
#   c4    4 并发同 prompt，四客户端 ids md5 均须 ≡ base
# 引擎带 QWENOX_VR_DBG=1，跑完核对引擎日志出现多段 [vr] batch（会合真实
# 生效的硬证据；D2_NOVR=1 对照模式下必须为零）。
# 可选：D2_GAMMA=8 强制 γ=8（active=4 时触发 24 行预算 clamp 路径）；
#       D2_NOVR=1 以 QWENOX_NO_VR=1 起引擎（分时旧路径对照，亦须 ≡ base）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
CLI=build/eng_cli.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOKDIR=./models/tokenizer
EPORT=18730

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=4
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
export QWENOX_VR_DBG=1
if [[ "${D2_GAMMA:-0}" -gt 0 ]]; then export QWENOX_SPEC_GAMMA="$D2_GAMMA"; fi
if [[ "${D2_NOVR:-0}" = 1 ]]; then export QWENOX_NO_VR=1; fi

# eng_cli：纯主机 C++，缺了就用 TheRock 自带 clang++ 现编（同 build.sh 的源列表）。
if [[ ! -x "$CLI" ]]; then
  TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
  CXX="$TR/lib/llvm/bin/clang++.exe"
  [[ -x "$CXX" ]] || { echo "FAIL: 缺 $CLI 且找不到 TheRock clang++: $CXX"; exit 1; }
  echo "[编译] $CLI"
  "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -Ithird_party \
    src/api/engine_client.cpp src/api/tokenizer.cpp src/api/eng_cli.cpp \
    -lws2_32 -o "$CLI" || { echo "FAIL: eng_cli 编译失败"; exit 1; }
fi

ELOG=logs/d2s2_engine.log
epid=''
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$epid" ]]; then
    kill "$epid" 2>/dev/null || true
    taskkill //PID "$epid" //F //T >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if netstat -an | grep -E "[:.]$EPORT\s+.*LISTENING" >/dev/null 2>&1; then
  echo "FAIL: 端口 $EPORT 已被占用"; exit 1
fi

echo "== d2_s2_ids $(date '+%F %T')  PARALLEL=4 NO_VR=${D2_NOVR:-0} GAMMA=${D2_GAMMA:-auto}"
"$BIN" "$M" "$NG" "$MTP" --serve --host 127.0.0.1 --port "$EPORT" \
    --maxctx 16384 >"$ELOG" 2>&1 &
epid=$!
begin=$SECONDS
until grep -q 'serve: listening' "$ELOG" 2>/dev/null; do
  kill -0 "$epid" 2>/dev/null || { tail -n 15 "$ELOG"; echo "FAIL: 引擎提前退出"; exit 1; }
  (( SECONDS - begin < 1800 )) || { echo "FAIL: 引擎启动超时"; exit 1; }
  sleep 2
done
echo "  .. engine ready ($((SECONDS - begin))s)"

REQ='{"prompt":"Write a complete, production-quality quicksort implementation in Python. Requirements: include detailed comments explaining each step, handle edge cases (empty input, single element, duplicates), and include a small test harness (main function or equivalent) that sorts several example arrays and prints the results. Output only the code, no explanations.","max_tokens":257,"drafter":1}'

run_client() {  # $1=标签
  "$CLI" --host "127.0.0.1:$EPORT" --tokenizer "$TOKDIR" "$REQ" \
      > "logs/d2s2_$1.json" 2> "logs/d2s2_$1.err"
}
ids_md5() { grep -o '"ids":\[[0-9,]*\]' "logs/d2s2_$1.json" | md5sum | cut -d' ' -f1; }

run_client base || { echo "FAIL: base 请求 rc=$?"; tail -3 logs/d2s2_base.err; exit 1; }
base=$(ids_md5 base)
[[ -n "$base" ]] || { echo "FAIL: base 无 ids"; cat logs/d2s2_base.json; exit 1; }
echo "base    md5=$base"

ok=1
for conc in 2 4; do
  pids=()
  for ((i = 0; i < conc; i++)); do run_client "c${conc}_$i" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
  for ((i = 0; i < conc; i++)); do
    h=$(ids_md5 "c${conc}_$i")
    if [[ -n "$h" && "$h" == "$base" ]]; then r=IDENTICAL; else r=DIFF; ok=0; fi
    printf 'c%d[%d]  md5=%s %s\n' "$conc" "$i" "$h" "$r"
  done
done

nb=$(grep -c '\[vr\] batch' "$ELOG" 2>/dev/null || true)
nm=$(grep -a '\[vr\] batch' "$ELOG" 2>/dev/null | grep -avc 'batch: 1 segs' || true)
echo "  .. [vr] batch 行: $nb（其中多段: $nm）"
if [[ "${D2_NOVR:-0}" != 1 && "$nm" -eq 0 ]]; then
  echo "FAIL: 会合从未形成多段 batch（vseg_pool 未生效？）"; ok=0
fi
echo "S2A IDS ACCEPT: $([ $ok = 1 ] && echo PASS || echo FAIL)"
[[ $ok = 1 ]]
