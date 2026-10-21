#!/bin/bash
# kvsnap K1/K2/K4 end-to-end (Linux, hgn v2 + MTP): three engine starts on a
# fresh kvsnap dir -- pause checkpoint SSD restore == RAM restore token for
# token, two-segment hot entry + kvsnap_hits.txt across a restart, and the
# RAM-vs-SSD tier choice when a subagent pushed the main agent out of RAM.
# No production service may run (build/start guard); ports 8740/8741.
#   [KV_E2E_DIR=/mnt/share/kvsnap-e2e] bash tools/kvsnap_e2e/run.sh
# KV_E2E_DIR (default data/kvsnap-e2e) holds the cache in qwenox-kvsnap/, wiped
# first. An environment variable, not an argument: a path segment starting
# with "qwenox" on this script's command line trips the start guard.
# Writes logs/kv_e2e.out; ends with KV E2E: PASS/FAIL.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT" || exit 2
mkdir -p logs
TESTDIR="${KV_E2E_DIR:-$ROOT/data/kvsnap-e2e}"
export ROOT
OUT=logs/kv_e2e.out
exec > >(tee "$OUT") 2>&1
export PARALLEL=1 RCKPT_MAX=3 MAX_CONTEXT=65536 KVSNAP_MAX_GB=50 \
       KVSNAP_DIR="$TESTDIR" ENGINE_PORT=8740 API_PORT=8741 VISION_FILE=""
# hgn v2 ships its n-gram table as *-ngram.hgn or *-ngram-bf16.hgn
if [[ -z ${NGRAM_FILE:-} && ! -f models/qwen38-flash-next-ngram.hgn ]]; then
  export NGRAM_FILE=./models/qwen38-flash-next-ngram-bf16.hgn
fi
KV="$TESTDIR/qwenox-kvsnap"
rm -rf "$KV"   # this test's cache only
fails=0
chk() {  # chk <ok 0/1> <text>
  if [[ $1 == 1 ]]; then echo "CHECK PASS  $2"; else echo "CHECK FAIL  $2"; fails=$((fails+1)); fi
}
SPID=""; ELOG=""
start() {
  local before; before=$(ls -d logs/*-*/ 2>/dev/null | sort)
  setsid bash start_hgn.sh >"logs/serve-$1.out" 2>&1 &
  SPID=$!
  local t=0
  while (( t < 900 )); do
    ELOG=$(comm -13 <(echo "$before") <(ls -d logs/*-*/ 2>/dev/null | sort) | tail -1)
    [[ -n $ELOG ]] && ELOG="${ELOG%/}/engine.log"
    if [[ -n $ELOG && -f $ELOG ]] && grep -q 'serve: listening' "$ELOG"; then
      echo "== $1: engine up after ${t}s ($ELOG)"; return 0
    fi
    kill -0 $SPID 2>/dev/null || { echo "== $1: start_hgn.sh exited"; tail -20 "logs/serve-$1.out"; return 1; }
    sleep 3; t=$((t+3))
  done
  echo "== $1: timeout"; return 1
}
stop() {
  # TERM (a background job ignores INT); start_hgn.sh's cleanup stops its scope
  local P="^$ROOT/build/qwenox-"
  kill -TERM -- -"$SPID" 2>/dev/null
  for _ in $(seq 20); do pgrep -f "$P" >/dev/null || break; sleep 1; done
  pkill -TERM -f "$P" 2>/dev/null
  for _ in $(seq 30); do pgrep -f "$P" >/dev/null || break; sleep 1; done
  pkill -KILL -f "$P" 2>/dev/null
  sleep 2
  pgrep -af "$P" && echo "== WARNING: still running"
  while ss -ltn | grep -qE ':874[01] '; do sleep 1; done
}
kvlog() { grep -aE 'kvsnap|rckpt|ckpt:|fork:|cut:|evicted|prompt eval' "$ELOG" | tr '\r' '\n' | grep -av '^\s*$' | tail -n "${1:-40}"; }

# ---- run 1: first turn + RAM restores --------------------------------------
start run1 || { echo "KV E2E: FAIL (start)"; exit 1; }
python3 "$HERE/kv_e2e.py" run1; r=$?
stop
echo "---- run1 engine log (kvsnap/rckpt) ----"; kvlog 40
NP=$(python3 -c "import json;print(json.load(open('logs/kv_e2e_state.json'))['n_pause'])" 2>/dev/null)
CT=$(python3 -c "import json;print(json.load(open('logs/kv_e2e_state.json'))['cut_tail'])" 2>/dev/null)
[[ $r == 0 && -n $NP ]] || { echo "KV E2E: FAIL (run1)"; exit 1; }
chk "$(grep -aq "kvsnap: saved $NP tokens" "$ELOG" && echo 1)" "K1：暂停点检查点（$NP token）已写入 SSD"
chk "$(grep -aq "rckpt: restored $NP tokens" "$ELOG" && echo 1)" "run1：暂停点从 RAM 恢复（基准）"
chk "$(grep -aq "rckpt: restored $CT tokens" "$ELOG" && echo 1)" "run1：切点从 RAM 恢复（对照基准）"
ls "$KV" | grep -c '^k_' | sed 's/^/SSD checkpoints after run1: /'

# ---- run 2: same requests from SSD, the pause one twice --------------------
start run2 || { echo "KV E2E: FAIL (start)"; exit 1; }
python3 "$HERE/kv_e2e.py" run2; r=$?
stop
echo "---- run2 engine log ----"; kvlog 30
[[ $r == 0 ]] || fails=$((fails+1))
chk "$(( $(grep -ac "kvsnap: restored $NP tokens" "$ELOG") == 2 ))" "run2：暂停点检查点从 SSD 恢复 2 次"
chk "$(grep -aq "kvsnap: restored $CT tokens" "$ELOG" && echo 1)" "run2：切点从 SSD 恢复（对照）"
chk "$(grep -aq "is hot (2 restores)" "$ELOG" && echo 1)" "K2：恢复两次进保护段"
echo "kvsnap_hits.txt:"; cat "$KV/kvsnap_hits.txt"
chk "$(grep -q "^k_${NP}_.* 2$" "$KV/kvsnap_hits.txt" && echo 1)" "K2：命中数写入 kvsnap_hits.txt"

# ---- run 3: hits read back; subagent pushes the main agent out of RAM ------
start run3 || { echo "KV E2E: FAIL (start)"; exit 1; }
grep -a 'kvsnap: index' "$ELOG"
chk "$(grep -aq 'checkpoints (1 hot)' "$ELOG" && echo 1)" "K2：重启后读回保护段（1 hot）"
python3 "$HERE/kv_e2e.py" run3; r=$?
stop
echo "---- run3 engine log ----"; kvlog 40
[[ $r == 0 ]] || fails=$((fails+1))
chk "$(grep -aq -- '-> SSD' "$ELOG" && echo 1)" "K4：RAM 只命中系统提示时改走 SSD"
chk "$(grep -aA3 -- '-> SSD' "$ELOG" | grep -aq 'kvsnap: restored' && echo 1)" "K4：随后从 SSD 恢复主代理检查点"

echo
nf=$(grep -c '^CHECK FAIL' "$OUT")
if (( nf == 0 && fails == 0 )); then echo "KV E2E: PASS"; else echo "KV E2E: FAIL ($nf check(s))"; fi
