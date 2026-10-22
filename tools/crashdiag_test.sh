#!/usr/bin/env bash
# crashdiag_test.sh — 崩溃诊断一键自测（Linux）。最后一行 CRASHDIAG LINUX TEST: PASS / FAIL。
#
#   A. 不需要 GPU：g++ 编 tools/crashdiag_selftest.cpp，逐种制造崩溃（写/读/执行坏地址、
#      abort、未捕获异常、栈溢出、工作线程崩溃、fatal），检查报告内容
#   B. 不需要 GPU：g++ 编 diagpack + 一个小驱动，在临时目录打一次诊断包，检查 zip
#      （崩溃报告、addr2line 符号化、配置打码、DIAG 在线快照）
#   C. 有 build/qwenox-engine（bash build.sh 之后）：引擎 --crash-test write / hipfatal
#      （不加载模型，需要 GPU 运行库）
#   D. 有 build/qwenox-api：--collect-diag 打真实诊断包
# 测试产物在 ${TMPDIR:-/tmp}/crashdiag-test/。
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
T="${TMPDIR:-/tmp}/crashdiag-test"
CXX="${CXX:-g++}"
rm -rf "$T"
mkdir -p "$T"
fails=0
ok() { echo "  PASS  $*"; }
bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
has() { if grep -a -q -E "$3" "$2" 2>/dev/null; then ok "$1"; else bad "$1（$2 里找不到 /$3/）"; fi; }

echo "== A. 处理器（独立测试程序）"
if ! "$CXX" -O2 -g1 -std=c++17 -Wall -Wextra -Wpedantic -Werror -pthread \
     "$ROOT/tools/crashdiag_selftest.cpp" -o "$T/st" 2>"$T/st.build.log"; then
  bad "编译 crashdiag_selftest 失败：$(head -5 "$T/st.build.log")"
else
  for kind in write read exec abort throw stack thread fatal; do
    d="$T/a-$kind"
    mkdir -p "$d"
    (cd "$d" && QWENOX_CRASH_DIR="$d" timeout 60 "$T/st" "$kind" >out.log 2>&1)
    rc=$?
    rep="$(ls "$d"/crash-selftest-*.txt 2>/dev/null | head -1)"
    (( rc != 0 && rc != 124 )) && ok "$kind: 退出码 $rc" || bad "$kind: 退出码 $rc"
    [[ -n "$rep" ]] || { bad "$kind: 没有报告"; continue; }
    case "$kind" in
      write) has "$kind: 写 0x10" "$rep" 'fault: WRITE .* at 0x0000000000000010' ;;
      read) has "$kind: 读坏地址" "$rep" 'fault: READ .* at 0x0000000000000018' ;;
      exec) has "$kind: 执行坏地址" "$rep" 'fault: EXECUTE' ;;
      abort) has "$kind: SIGABRT" "$rep" 'signal: 6 SIGABRT' ;;
      throw) has "$kind: 未捕获异常" "$rep" 'message: uncaught C\+\+ exception' ;;
      stack) has "$kind: 栈溢出（落在 [stack] 下方）" "$rep" 'next: .*\[stack\]' ;;
      thread)
        has "$kind: SIGSEGV" "$rep" 'SIGSEGV'
        if grep -a '^thread:' "$rep" | grep -q 'main thread'; then bad "$kind: 应记为工作线程"; else ok "$kind: 工作线程"; fi ;;
      fatal) has "$kind: fatal" "$rep" 'kind: fatal' ;;
    esac
    has "$kind: 飞行记录" "$rep" 'about to crash'
    has "$kind: HIP 调用点" "$rep" 'hipMemcpyAsync\(d_pos'
    has "$kind: 环境快照" "$rep" '^kernel: Linux'
    has "$kind: maps" "$rep" '^---- /proc/self/maps'
    has "$kind: stderr 摘要" "$d/out.log" '\[crashdiag\] report:'
  done
fi

echo "== B. 诊断包（diagpack + 测试驱动）"
B="$T/b"
mkdir -p "$B/obj" "$B/root/logs/crash" "$B/root/logs/20990101-000000-1"
ok_build=1
for m in miniz miniz_tdef miniz_tinfl miniz_zip; do
  cc -O2 -x c -I"$ROOT/third_party/miniz" -c "$ROOT/third_party/miniz/$m.c" -o "$B/obj/$m.o" 2>>"$B/build.log" || ok_build=0
done
cat >"$B/drv.cpp" <<'EOF'
#include "diagpack.h"
#include <cstdio>
#include <cstdlib>
int main(int, char** argv) {
  diagpack::Opts o; o.root = argv[1]; o.reason = "crashdiag_test"; o.engine_pid = atoi(argv[2]);
  o.has_exit = true; o.exit_code = 139; o.engine_host = "127.0.0.1"; o.engine_port = atoi(argv[3]);
  std::string e; const std::string p = diagpack::build(o, &e);
  printf("%s\n", p.empty() ? ("ERR " + e).c_str() : p.c_str()); return p.empty();
}
EOF
"$CXX" -O1 -std=c++17 -Wall -Wextra -Wpedantic -Werror -DQWENOX_VERSION_RAW=test -DQWENOX_GIT_RAW=test \
  -I"$ROOT/src/api" -I"$ROOT/third_party/miniz" "$B/drv.cpp" "$ROOT/src/api/diagpack.cpp" "$B"/obj/*.o \
  -pthread -o "$B/drv" 2>>"$B/build.log" || ok_build=0
if (( !ok_build )); then
  bad "编译 diagpack 驱动失败：$(head -8 "$B/build.log")"
else
  src="$(ls "$T"/a-thread/crash-selftest-*.txt 2>/dev/null | head -1)"
  [[ -n "$src" ]] && cp "$src" "$B/root/logs/crash/crash-engine-4242-20990101-000000.txt"
  seq 1 900000 | sed 's/^/engine line /' >"$B/root/logs/20990101-000000-1/engine.log"
  echo "api line" >"$B/root/logs/20990101-000000-1/api.log"
  printf 'MODEL_DIR=models\nQWENOX_API_ADMIN_KEY=supersecret\n' >"$B/root/service.conf"
  port=$((20000 + RANDOM % 20000))
  python3 - "$port" >"$B/fake.log" 2>&1 <<'EOF' &
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); s.settimeout(30)
c, _ = s.accept(); assert c.recv(100) == b"DIAG\n"
c.sendall(b"==== Qwenox live diagnostics ====\nfake ring line\n\n.\n"); c.close()
EOF
  fpid=$!
  sleep 0.5
  zp="$(timeout 120 "$B/drv" "$B/root" 4242 "$port")"
  wait "$fpid" 2>/dev/null
  if [[ -f "$zp" ]]; then
    ok "诊断包：$zp"
    python3 - "$zp" <<'EOF' >"$B/check.log"
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); n = z.namelist()
def chk(d, c): print(("  PASS  " if c else "  FAIL  ") + d)
for f in ("README.txt", "summary.txt", "engine.log", "api.log", "system.txt", "kernel.txt", "service.conf", "engine-live.txt"):
    chk("包内有 " + f, f in n)
chk("崩溃报告", "crash/crash-engine-4242-20990101-000000.txt" in n)
sym = z.read("crash/crash-engine-4242-20990101-000000.sym.txt").decode() if "crash/crash-engine-4242-20990101-000000.sym.txt" in n else ""
chk("addr2line 符号化（找到 self_test）", "self_test" in sym or "addr2line not found" in sym)
chk("配置打码", "supersecret" not in z.read("service.conf").decode())
chk("引擎日志截头截尾", "bytes skipped" in z.read("engine.log").decode())
chk("在线快照", "fake ring line" in z.read("engine-live.txt").decode())
chk("摘要退出码含义", "SIGSEGV (128+11)" in z.read("summary.txt").decode())
EOF
    cat "$B/check.log"
    fails=$((fails + $(grep -c '^  FAIL' "$B/check.log")))
  else
    bad "诊断包没生成：$zp"
  fi
fi

echo "== C. 引擎 --crash-test（需要 bash build.sh 之后的 build/qwenox-engine）"
ENG="$(ls "$ROOT"/build/qwenox-engine "$ROOT"/build/gdec 2>/dev/null | head -1)"
if [[ -z "$ENG" ]]; then
  echo "  SKIP  没有 build/qwenox-engine"
else
  [[ -f "$ROOT/build/bundled-runtime.conf" ]] && export LD_LIBRARY_PATH="$ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  for kind in write hipfatal; do
    d="$T/c-$kind"
    mkdir -p "$d"
    (cd "$ROOT" && QWENOX_CRASH_DIR="$d" timeout 120 "$ENG" --crash-test "$kind" >"$d/out.log" 2>&1)
    rc=$?
    rep="$(ls "$d"/crash-engine-*.txt 2>/dev/null | head -1)"
    (( rc != 0 && rc != 124 )) && ok "engine $kind: 退出码 $rc" || bad "engine $kind: 退出码 $rc"
    [[ -n "$rep" ]] || { bad "engine $kind: 没有报告（$d/out.log）"; continue; }
    [[ "$kind" == write ]] && has "engine write: 写 0x10" "$rep" 'fault: WRITE .* at 0x0000000000000010'
    [[ "$kind" == hipfatal ]] && has "engine hipfatal: HIP 错误" "$rep" 'message: HIP error'
    has "engine $kind: HIP 环境" "$rep" '^hip: runtime'
    has "engine $kind: pc 在引擎里" "$rep" '^pc: 0x[0-9A-F]+ .*qwenox-engine'
  done
fi

echo "== D. API --collect-diag（需要 build/qwenox-api）"
if [[ ! -x "$ROOT/build/qwenox-api" ]]; then
  echo "  SKIP  没有 build/qwenox-api"
else
  R="$T/d-root"
  mkdir -p "$R/logs/crash"
  cp "$ROOT/service.conf" "$R/service.conf"
  ls "$T"/c-write/crash-engine-*.txt >/dev/null 2>&1 && cp "$T"/c-write/crash-engine-*.txt "$R/logs/crash/"
  zp="$(timeout 120 "$ROOT/build/qwenox-api" --collect-diag --service-conf "$R/service.conf" --diag-reason crashdiag_test 2>"$T/d.err" | tail -1)"
  if [[ -f "$zp" ]]; then
    ok "诊断包：$zp"
    python3 - "$zp" <<'EOF' >"$T/d.check"
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); n = z.namelist()
def chk(d, c): print(("  PASS  " if c else "  FAIL  ") + d)
chk("kernel.txt", "kernel.txt" in n)
chk("system.txt 含内核/ROCm", "kernel: Linux" in z.read("system.txt").decode())
syms = [x for x in n if x.endswith(".sym.txt")]
if syms:
    chk("引擎栈帧已符号化（-g1）", "crash_test_now" in z.read(syms[0]).decode() or "self_test" in z.read(syms[0]).decode())
EOF
    cat "$T/d.check"
    fails=$((fails + $(grep -c '^  FAIL' "$T/d.check")))
  else
    bad "诊断包没生成：$(tail -3 "$T/d.err")"
  fi
fi

if (( fails == 0 )); then
  echo "CRASHDIAG LINUX TEST: PASS"
  exit 0
fi
echo "失败 $fails 项；产物在 $T/"
echo "CRASHDIAG LINUX TEST: FAIL"
exit 1
