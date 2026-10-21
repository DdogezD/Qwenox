#!/usr/bin/env bash
# c7_verify.sh — C7（采样草稿合批，QWENOX_DR_SAMPLE）一键验证 + 吞吐 A/B（Linux，hgn）。
# 在仓库根目录运行，需先 bash build.sh，且本机没有 qwenox 在跑（会自己起停两次服务）。
#
# 每个服务（QWENOX_DR_SAMPLE=1 与 =0 各一次，PARALLEL=8，kvsnap 关）：
#   1. 带 seed 的 8 题：C1 串行跑一遍、C4 并发跑一遍，逐字对比（C1 不会合批，C4 会）；
#   2. 不带 seed 的默认采样吞吐：C4（8 请求）、C8（8 请求），各 512 token。
# 判定：DR=1 的 C1/C4 一致题数不少于 DR=0 基线（基线通常 8/8），且日志里出现 ≥2 路的合批草稿。
# 约 7–8 分钟。末行 C7_VERIFY PASS / FAIL。结果在 logs/c7_verify/。
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT=logs/c7_verify
mkdir -p "$OUT"
pgrep -x qwenox-engine >/dev/null && { echo "已有 qwenox-engine 在运行，先停掉"; echo "C7_VERIFY FAIL"; exit 1; }

stop_all() {
  for p in $(pgrep -f '^bash start_hgn\.sh'); do kill -INT "$p" 2>/dev/null; done
  for _ in $(seq 1 120); do
    pgrep -x qwenox-engine >/dev/null || pgrep -x qwenox-api >/dev/null || return 0
    sleep 1
  done
}
trap stop_all EXIT

declare -A SAME DRB TPS4 TPS8
for DR in 1 0; do
  echo "== 服务 QWENOX_DR_SAMPLE=$DR"
  QWENOX_DR_SAMPLE=$DR QWENOX_VR_DBG=1 PARALLEL=8 KVSNAP_MAX_GB=0 \
    setsid -f bash start_hgn.sh >"$OUT/serve$DR.out" 2>&1 </dev/null
  for _ in $(seq 1 300); do grep -q '服务已就绪' "$OUT/serve$DR.out" && break; sleep 1; done
  grep -q '服务已就绪' "$OUT/serve$DR.out" || { tail -n 20 "$OUT/serve$DR.out"; echo "C7_VERIFY FAIL"; exit 1; }
  LOG="$(ls -td logs/2*/ | head -1)engine.log"
  python3 tools/t1_load.py --conc 1 --rpw 8 --max-tokens 256 --seed 1000 --out "$OUT/c1_dr$DR.json" || { echo "C7_VERIFY FAIL"; exit 1; }
  python3 tools/t1_load.py --conc 4 --rpw 2 --max-tokens 256 --seed 1000 --out "$OUT/c4_dr$DR.json" || { echo "C7_VERIFY FAIL"; exit 1; }
  r4=$(python3 tools/t1_load.py --conc 4 --rpw 2 --max-tokens 512 | tee -a "$OUT/perf$DR.txt" | sed -n 's/.*= \([0-9.]*\) tok\/s/\1/p')
  r8=$(python3 tools/t1_load.py --conc 8 --rpw 1 --max-tokens 512 | tee -a "$OUT/perf$DR.txt" | sed -n 's/.*= \([0-9.]*\) tok\/s/\1/p')
  TPS4[$DR]=$r4; TPS8[$DR]=$r8
  SAME[$DR]=$(python3 - "$OUT/c1_dr$DR.json" "$OUT/c4_dr$DR.json" <<'EOF'
import json, sys
a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
same = sum(1 for k in a if k in b and a[k] == b[k])
for k in sorted(a):
    if a.get(k) != b.get(k):
        x, y = json.dumps(a.get(k), ensure_ascii=False), json.dumps(b.get(k), ensure_ascii=False)
        i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        print(f"  题 {k} 不一致，首个差异在第 {i} 字符", file=sys.stderr)
print(same)
EOF
)
  DRB[$DR]=$(grep -c '\[dr\] batch: [2-8] segs' "$LOG")
  grep '\[dr\] batch:' "$LOG" | tail -n 1 | sed 's/^/  最后一批：/'
  cp "$LOG" "$OUT/engine_dr$DR.log"
  echo "  seeded C1/C4 一致 ${SAME[$DR]}/8；≥2 路合批草稿 ${DRB[$DR]} 次；C4 ${TPS4[$DR]} tok/s，C8 ${TPS8[$DR]} tok/s"
  stop_all
done

echo "== 汇总"
echo "seeded C1/C4 一致：DR=1 ${SAME[1]}/8，DR=0（基线）${SAME[0]}/8"
echo "≥2 路合批草稿批次：DR=1 ${DRB[1]}，DR=0 ${DRB[0]}"
python3 -c "import sys; a4,b4,a8,b8=map(float,sys.argv[1:]); print(f'吞吐 C4 {b4:.1f} -> {a4:.1f} ({100*(a4/b4-1):+.1f}%)，C8 {b8:.1f} -> {a8:.1f} ({100*(a8/b8-1):+.1f}%)')" \
  "${TPS4[1]}" "${TPS4[0]}" "${TPS8[1]}" "${TPS8[0]}"
if (( SAME[1] >= SAME[0] && DRB[1] > 0 )); then echo "C7_VERIFY PASS"; else echo "C7_VERIFY FAIL"; exit 1; fi
