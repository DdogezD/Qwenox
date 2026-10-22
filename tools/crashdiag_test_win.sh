#!/usr/bin/env bash
# crashdiag_test_win.sh — 崩溃诊断一键自测（Windows，Git Bash；先 bash build_win.sh all）。
#
# 不加载模型，约 1–2 分钟，最后一行 CRASHDIAG WIN TEST: PASS / FAIL。
#   A. 引擎 --crash-test 各种崩溃：每种都要生成崩溃报告（.txt）和 minidump（.dmp），
#      报告里要有异常信息、飞行记录、环境快照、模块表
#   B. qwenox-win.exe --collect-diag：诊断包 zip 内容齐全（摘要 / 系统 / 事件日志 /
#      崩溃报告 / 配置打码）
#   C. 整条链路：API（控制台模式，端口 18731）拉起引擎，引擎带 QWENOX_CRASH_TEST=thread
#      启动即在工作线程崩溃 → API 自动打诊断包 → /admin/engine/status 给出路径。
#      需要引擎端口 8730 空闲：托盘里正在运行的 Qwenox 请先退出，否则 C 段跳过（不算失败）。
#   D. PDB 已生成（build_win.sh 默认生成；QWENOX_PDB=0 时只提示）
# 测试产物在 build/.crashdiag-test/；C 段写进 logs/ 的测试文件（该次引擎日志、崩溃报告、
# 诊断包）在 PASS 后删除。
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
E=build/qwenox-engine-win.exe
A=build/qwenox-win.exe
T=build/.crashdiag-test
PY="$(command -v python || command -v python3 || true)"
[[ -x "$E" && -x "$A" ]] || { echo "先运行 bash build_win.sh all（缺 $E 或 $A）"; echo "CRASHDIAG WIN TEST: FAIL"; exit 1; }
[[ -n "$PY" ]] || { echo "需要 python（检查 zip 内容）"; echo "CRASHDIAG WIN TEST: FAIL"; exit 1; }
rm -rf "$T"
mkdir -p "$T"
fails=0
ok() { echo "  PASS  $*"; }
bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
has() {  # 描述 文件 正则
  if grep -a -q -E "$3" "$2" 2>/dev/null; then ok "$1"; else bad "$1（$2 里找不到 /$3/）"; fi
}

echo "== A. 引擎崩溃报告（--crash-test，每种约 2–5 秒）"
for kind in write thread stack abort throw fatal hipfatal; do
  d="$T/a-$kind"
  mkdir -p "$d"
  QWENOX_CRASH_DIR="$(cygpath -w "$ROOT/$d")" timeout 120 "$E" --crash-test "$kind" >"$d/engine.log" 2>&1
  rc=$?
  rep="$(ls "$d"/crash-engine-*.txt 2>/dev/null | head -1)"
  dmp="$(ls "$d"/crash-engine-*.dmp 2>/dev/null | head -1)"
  echo " [$kind] 退出码 $rc"
  (( rc != 0 && rc != 124 )) && ok "$kind: 非零退出" || bad "$kind: 退出码 $rc"
  if [[ -z "$rep" ]]; then
    bad "$kind: 没有崩溃报告（引擎输出见 $d/engine.log）"
    continue
  fi
  case "$kind" in
    write) has "$kind: 访问冲突 + 写 0x10" "$rep" 'fault: WRITE at 0x0000000000000010' ;;
    thread)
      has "$kind: 访问冲突" "$rep" 'code: 0xC0000005'
      if grep -a '^thread:' "$rep" | grep -q 'main thread'; then bad "$kind: 应记为工作线程"; else ok "$kind: 记为工作线程"; fi ;;
    stack) has "$kind: 栈溢出" "$rep" 'EXCEPTION_STACK_OVERFLOW' ;;
    abort) has "$kind: abort" "$rep" 'kind: abort' ;;
    throw) has "$kind: 未捕获异常" "$rep" 'uncaught C\+\+ exception' ;;
    fatal) has "$kind: fatal" "$rep" 'kind: fatal' ;;
    hipfatal)
      has "$kind: HIP 错误" "$rep" 'message: HIP error'
      has "$kind: HIP 环境" "$rep" '^hip: runtime' ;;
  esac
  has "$kind: 出错位置已解析到模块" "$rep" '^pc: 0x[0-9A-F]+ [^ ]+\+0x'
  has "$kind: 飞行记录" "$rep" 'crashdiag self-test'
  has "$kind: 环境快照（显卡驱动）" "$rep" '^gpu: '
  has "$kind: 模块表" "$rep" 'amdhip64'
  if [[ -n "$dmp" && "$(head -c 4 "$dmp")" == "MDMP" ]]; then
    ok "$kind: minidump（$(stat -c %s "$dmp") 字节）"
  else
    bad "$kind: minidump 缺失或格式不对"
  fi
  has "$kind: 引擎日志里有报告路径" "$d/engine.log" '\[crashdiag\] report:'
done

echo "== B. 诊断包（--collect-diag，读事件日志约 5–20 秒）"
R="$T/root"
mkdir -p "$R/logs/crash"
cp service.conf "$R/service.conf"
printf 'QWENOX_API_ADMIN_KEY=should-be-masked\n' >>"$R/service.conf"
cp "$T"/a-write/crash-engine-* "$R/logs/crash/" 2>/dev/null
printf 'test engine log line\n' >"$R/logs/engine-api-20990101-000000.log"
p="$(timeout 180 "$A" --collect-diag --service-conf "$(cygpath -w "$ROOT/$R/service.conf")" \
      --diag-reason crashdiag_test_win 2>"$T/b.err" | tr -d '\r' | tail -1)"
if [[ -n "$p" && -f "$(cygpath -u "$p" 2>/dev/null || echo "$p")" ]]; then
  ok "诊断包已生成：$p"
  "$PY" - "$(cygpath -w "$(cygpath -u "$p")")" <<'EOF' | while IFS= read -r l; do echo "$l"; [[ "$l" == *FAIL* ]] && echo x >>"$T/b.fail"; done
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = z.namelist()
def chk(desc, cond): print(("  PASS  " if cond else "  FAIL  ") + desc)
for n in ("README.txt", "summary.txt", "system.txt", "events.txt", "service.conf", "engine.log", "logs-index.txt"):
    chk("包内有 " + n, n in names)
chk("包内有崩溃报告 .txt", any(n.startswith("crash/") and n.endswith(".txt") for n in names))
chk("包内有 minidump .dmp", any(n.startswith("crash/") and n.endswith(".dmp") for n in names))
ev = z.read("events.txt").decode("utf-8", "replace") if "events.txt" in names else ""
chk("事件日志：系统段", "==== System ====" in ev)
chk("事件日志：应用段", "==== Application ====" in ev)
chk("事件日志查询未报错", "query failed" not in ev)
sysx = z.read("system.txt").decode("utf-8", "replace") if "system.txt" in names else ""
chk("系统信息含显卡驱动", "\ngpu: " in sysx)
chk("系统信息含电源计划", "\npower: " in sysx)
conf = z.read("service.conf").decode("utf-8", "replace") if "service.conf" in names else ""
chk("配置里的密钥已打码", "should-be-masked" not in conf and "QWENOX_API_ADMIN_KEY=***" in conf)
s = z.read("summary.txt").decode("utf-8", "replace") if "summary.txt" in names else ""
chk("摘要含崩溃要点", "fault: WRITE" in s)
EOF
  [[ -f "$T/b.fail" ]] && fails=$((fails + $(wc -l <"$T/b.fail")))
else
  bad "诊断包没生成（$T/b.err）：$(tail -3 "$T/b.err" 2>/dev/null)"
fi

echo "== C. 整条链路（API 拉起引擎 → 引擎崩溃 → 自动打包）"
if (exec 3<>/dev/tcp/127.0.0.1/8730) 2>/dev/null; then
  echo "  SKIP  引擎端口 8730 正在使用（托盘里的 Qwenox 在运行？先退出再测这一段）"
elif (exec 3<>/dev/tcp/127.0.0.1/18731) 2>/dev/null; then
  echo "  SKIP  测试端口 18731 被占用"
else
  QWENOX_CRASH_TEST=thread QWENOX_CRASH_TEST_AT=start QWENOX_API_ADMIN_KEY= \
    "$A" --console --port 18731 >"$T/c-api.log" 2>&1 &
  apipid=$!
  up=0
  for _ in $(seq 1 30); do
    curl -s -m 2 http://127.0.0.1:18731/health >/dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  if (( !up )); then
    bad "测试 API 没起来（$T/c-api.log）"
  else
    curl -s -m 10 -X POST http://127.0.0.1:18731/admin/engine/start >"$T/c-start.json"
    st=""
    for _ in $(seq 1 150); do
      curl -s -m 3 http://127.0.0.1:18731/admin/engine/status >"$T/c-status.json" 2>/dev/null
      st="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('state',''))" "$T/c-status.json" 2>/dev/null)"
      [[ "$st" == exited || "$st" == stopped ]] && break
      sleep 1
    done
    if [[ "$st" != exited ]]; then
      bad "引擎没有按预期崩溃退出（状态 '$st'；启动返回：$(head -c 300 "$T/c-start.json")）"
    else
      ok "引擎崩溃，API 记为 exited"
      out="$("$PY" - "$T/c-status.json" <<'EOF'
import json, sys, zipfile
j = json.load(open(sys.argv[1]))
d = j.get("diag", "")
print("LOG", j.get("log", ""))
print("DIAG", d)
def chk(desc, cond): print(("  PASS  " if cond else "  FAIL  ") + desc)
chk("退出码为访问冲突 0xC0000005", j.get("exit_code") == 0xC0000005)
chk("status 里给出诊断包路径", bool(d))
if d:
    try:
        z = zipfile.ZipFile(d); names = z.namelist(); s = z.read("summary.txt").decode("utf-8", "replace")
        pid = [l.split(":")[1].strip() for l in s.splitlines() if l.startswith("engine pid:")]
        chk("包内有该引擎 pid 的崩溃报告", bool(pid) and any(n.startswith("crash/crash-engine-%s-" % pid[0]) and n.endswith(".txt") for n in names))
        chk("包内有该引擎 pid 的 minidump", bool(pid) and any(n.startswith("crash/crash-engine-%s-" % pid[0]) and n.endswith(".dmp") for n in names))
        chk("摘要写明退出码含义", "STATUS_ACCESS_VIOLATION" in s)
        chk("包内有引擎日志", "engine.log" in names)
        print("PID", pid[0] if pid else "")
    except Exception as e:
        chk("诊断包可读（%s）" % e, False)
EOF
)"
      echo "$out" | grep -E '^  (PASS|FAIL)'
      fails=$((fails + $(echo "$out" | grep -c '^  FAIL')))
      c_log="$(echo "$out" | sed -n 's/^LOG //p')"
      c_diag="$(echo "$out" | sed -n 's/^DIAG //p')"
      c_pid="$(echo "$out" | sed -n 's/^PID //p')"
    fi
  fi
  wp="$(cat /proc/$apipid/winpid 2>/dev/null)"
  [[ -n "$wp" ]] && taskkill //F //PID "$wp" >/dev/null 2>&1
  kill "$apipid" 2>/dev/null
  wait "$apipid" 2>/dev/null
fi

echo "== D. 调试符号"
for f in build/qwenox-engine-win.pdb build/qwenox-win.pdb; do
  if [[ -f "$f" ]]; then ok "$f（$(stat -c %s "$f") 字节）"
  elif [[ "${QWENOX_PDB:-1}" == 0 ]]; then echo "  SKIP  $f（QWENOX_PDB=0）"
  else bad "$f 不存在（build_win.sh 应默认生成）"; fi
done

if (( fails == 0 )); then
  # C 段写进 logs/ 的测试文件
  [[ -n "${c_log:-}" ]] && rm -f "$(cygpath -u "$c_log")"
  [[ -n "${c_diag:-}" ]] && rm -f "$(cygpath -u "$c_diag")"
  [[ -n "${c_pid:-}" ]] && rm -f logs/crash/crash-engine-"$c_pid"-*
  echo "CRASHDIAG WIN TEST: PASS"
  exit 0
fi
echo "失败 $fails 项；测试产物在 $T/（C 段的引擎日志/崩溃报告/诊断包留在 logs/ 下便于查看）"
echo "CRASHDIAG WIN TEST: FAIL"
exit 1
