#!/usr/bin/env bash
# Windows (TheRock) 编译入口：与 Linux build.sh 并列，产物输出到 build/。
#
# 用法:
#   bash build_win.sh            # 全部产物：引擎、benchmark、API、根目录入口、设置工具
#   bash build_win.sh bench      # 性能测试工具 → build/qwenox-bench.exe
#   bash build_win.sh api        # OpenAI HTTP 前端 → build/qwenox-win.exe
#                                #   （GUI 托盘程序：API 服务 + 控制台网页 + 引擎启停）
#   bash build_win.sh launcher   # 根目录最小入口 → ./start_win.exe（双击拉起 API）
#   bash build_win.sh settings   # 根目录设置工具 → ./settings_win.exe（双击配置/检查，不启动服务）
#   bash build_win.sh test       # 编 build/ktest-win.exe 并运行 kernel 单测
#   bash build_win.sh npu        # NPU 检测工具（仅 Windows）→ build/npucheck-win.exe
#                                #   （GUI）+ build/npuio.exe + build/gpuload.exe；
#                                #   XRT 头/def 与 xclbin/insts 不进仓库，默认取自
#                                #   ~/npuprobe（NPUASSETS=/path 覆盖），缺资产时该
#                                #   目标报错，但不影响 all
# 同样检测到这些资产时，engine/bench/test 会编入 NPU 卸载（-DQWENOX_NPU=1，
# 运行时 QWENOX_NPU=1 启用，QWENOX_BUILD_NPU=0 不编入）。
#
# 前置：TheRock 多架构包（默认 C:\therock-dist-windows-multiarch-10.0.0\...，
# 可用 THEROCK=/path 覆盖）；GPU_ARCH 默认 gfx1151。
#
# 脚本同时准备好运行环境（只需做一次，之后重复运行为幂等跳过）：
# - winlibs 暂存：hipcc 的 -l<name> 解析找 <name>.lib；TheRock 的 hipBLASLt
#   只有 libhipblaslt.dll.a —— 改名暂存一份（lld-link 直接吃 .dll.a 内容）。
# - 运行依赖 DLL 复制到 exe 旁（屏蔽 System32 里 HIP SDK 7.2 的旧 DLL）;
#   origami.dll 是 libhipblaslt.dll 的静态依赖（lld 的 "?" 报错不点名它）。
# - rocBLAS/hipBLASLt kernel db junction：rocBLAS 按 <exe目录>/rocblas/library
#   找 db（缺主索引 TensileLibrary.dat 时回退扫描 gfx1151/ 分片目录，目录必须
#   能枚举），TheRock 的 db 实际在 bin/rocblas/library —— mklink /J 免管理员。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
HIPCC="$TR/bin/hipcc.exe"
# hipcc 会读 HIP_PATH 定位 clang（实测指向坏路径直接编译失败；指向 HIP SDK 7.2
# 时靠布局差异侥幸回退自定位，不该依赖）。强制 HIP_PATH 指向 TheRock，屏蔽机器
# 上其他 HIP/ROCm 安装（SDK 6.4/7.1/7.2 会设 HIP_PATH/HIP_PATH_64/HIP_PATH_72）。
# ROCM_PATH 则必须不设：clang 看到它会去 $ROCM_PATH/amdgcn/bitcode 找设备库，
# 而 TheRock 的设备库在 lib/llvm/amdgcn/bitcode，靠 clang 自定位才找得到。
TR_WIN="$(cygpath -w "$TR")"
export HIP_PATH="$TR_WIN"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
GPU_ARCH="${GPU_ARCH:-gfx1151}"
TARGET="${1:-all}"
[[ "$TARGET" == all || "$TARGET" == engine || "$TARGET" == bench || "$TARGET" == api || "$TARGET" == launcher || "$TARGET" == settings || "$TARGET" == test || "$TARGET" == npu ]] || { sed -n '2,17p' "$0" >&2; exit 2; }

# HIP 工具链/winlibs/kernel db 只服务 engine/bench/test；api 与 launcher 是纯主机 C++，
# 在没有本机 gfx 分片 kernel db 的机器上（如只部署 API 的机器）也应能编译。
NEED_HIP=0
[[ "$TARGET" == all || "$TARGET" == engine || "$TARGET" == bench || "$TARGET" == test || "$TARGET" == npu ]] && NEED_HIP=1
[[ "$NEED_HIP" == 0 || -x "$HIPCC" ]] || { echo "找不到 TheRock hipcc: $HIPCC（设 THEROCK=...）" >&2; exit 1; }

# 版本号：唯一来源是根目录 VERSION（见 src/version.h）。宏值不带引号，
# 由 version.h 字符串化——hipcc.exe 转发参数时会吃掉引号。
QWENOX_VER="$(tr -d ' \r\n' < VERSION 2>/dev/null || true)"
[[ "$QWENOX_VER" =~ ^[0-9A-Za-z.-]+$ ]] || { echo "VERSION 文件缺失或格式不对（应形如 0.0.8）" >&2; exit 1; }
QWENOX_GITREV="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
VER_FLAGS=(-DQWENOX_VERSION_RAW="$QWENOX_VER" -DQWENOX_GIT_RAW="$QWENOX_GITREV")
echo "[版本] $QWENOX_VER ($QWENOX_GITREV)"

mkdir -p build
# MSVC 运行时（微软官方可再分发），覆盖没装 VC++ Redistributable 的裸机
for d in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
  [[ -f "build/$d" ]] || cp "/c/windows/System32/$d" build/
done

if [[ "$NEED_HIP" == 1 ]]; then
mkdir -p build/winlibs
[[ -f build/winlibs/rocblas.lib ]]   || cp "$TR/lib/rocblas.lib" build/winlibs/
[[ -f build/winlibs/amdhip64.lib ]]  || cp "$TR/lib/amdhip64.lib" build/winlibs/
[[ -f build/winlibs/hipblaslt.lib ]] || cp "$TR/lib/libhipblaslt.dll.a" build/winlibs/hipblaslt.lib

for d in amdhip64_7.dll rocm_kpack.dll amd_comgr.dll rocblas.dll libhipblaslt.dll origami.dll; do
  [[ -f "build/$d" ]] || cp "$TR/bin/$d" build/
done

# kernel db 拷真身而非 junction：rocBLAS 按 <exe目录>/rocblas/library 找 db
# （缺主索引 TensileLibrary.dat 时回退枚举 gfx<arch>/ 分片目录，目录必须能
# 枚举）。只拷本机架构分片（rocblas 17M + hipblaslt 13M；全架构是 703M+530M）。
# 拷完 build/ 即自包含：目标机器无需安装 TheRock/ROCm 运行库，拷走即用。
for d in rocblas hipblaslt; do
  [[ ! -L "build/$d" ]] || rm "build/$d"  # 旧版 junction：删链接换真身
  if [[ ! -d "build/$d/library/$GPU_ARCH" ]]; then
    mkdir -p "build/$d/library"
    cp -r "$TR/bin/$d/library/$GPU_ARCH" "build/$d/library/" \
      || { echo "kernel db 拷贝失败：$TR/bin/$d/library/$GPU_ARCH" >&2; exit 1; }
    echo "[db] build/$d/library/$GPU_ARCH 已拷贝"
  fi
done
# rocBLAS 顶层另有按 arch 命名的 fallback 分片（KB 级），一并带上
for f in "$TR/bin/rocblas/library/"*"$GPU_ARCH"*; do
  [[ -f "$f" ]] || continue
  b="$(basename "$f")"
  [[ -f "build/rocblas/library/$b" ]] || cp "$f" build/rocblas/library/
done
fi  # NEED_HIP

# _CRT_NONSTDC_NO_DEPRECATE：屏蔽 MSVC 头文件对 strdup 等 POSIX 名的
# deprecated 标记（_CRT_NONSTDC_DEPRECATE → __declspec(deprecated)）。
FLAGS=(-O3 -std=c++17 --offload-arch="$GPU_ARCH" -D_CRT_SECURE_NO_WARNINGS
       -D_CRT_NONSTDC_NO_DEPRECATE "${VER_FLAGS[@]}"
       -I"$TR/include" -Lbuild/winlibs -lrocblas -lhipblaslt)

# ---- NPU 卸载（QWENOX_NPU，仅 Windows）-------------------------------------
# 检测到 XRT 资产（默认 ~/npuprobe，NPUASSETS 覆盖）就给 engine/bench/test
# 加编译期开关 -DQWENOX_NPU=1；运行时 QWENOX_NPU=1 才启用，默认关，同一份二
# 进制可 A/B。xrt_coreutil.dll 走 delay-load，没装 NPU 驱动的机器照常运行。
# QWENOX_BUILD_NPU=0 强制不编入。
NPU_FLAGS=()
NPU_LINK=()
npu_assets_ok() {  # $1=资产目录（XRT 头/def + xclbin/insts）
    [[ -d "$1/xrtinc" && -f "$1/xrt_coreutil.def" &&
       -f "$1/npu/final_4096x2560x1024_128x64x128.xclbin" &&
       -f "$1/npu/insts_4096x2560x1024_128x64x128.txt" ]]
}
ensure_xrt_lib() {  # $1=资产目录；由 xrt_coreutil.def 生成导入库
    [[ -f build/winlibs/xrt_coreutil.lib ]] && return 0
    local LLVMLIB="$TR/lib/llvm/bin/llvm-lib.exe"
    [[ -x "$LLVMLIB" ]] || LLVMLIB="/c/Program Files/AMD/ROCm/7.2/bin/llvm-lib.exe"
    "$LLVMLIB" /def:"$(cygpath -w "$1/xrt_coreutil.def")" /machine:x64 \
      /out:"$(cygpath -w "$ROOT/build/winlibs")\\xrt_coreutil.lib" >/dev/null \
      || { echo "llvm-lib 生成 xrt_coreutil.lib 失败" >&2; return 1; }
    echo "[生成] build/winlibs/xrt_coreutil.lib"
}
gen_npu_bins() {  # $1=资产目录；内嵌 NPU 设备二进制（~900KB 源文本，增量生成）
    local XCLBIN="$1/npu/final_4096x2560x1024_128x64x128.xclbin"
    local INSTS="$1/npu/insts_4096x2560x1024_128x64x128.txt"
    local INC=build/npu_bins.inc
    [[ -f "$INC" && ! "$XCLBIN" -nt "$INC" && ! "$INSTS" -nt "$INC" ]] && return 0
    echo "[生成] $INC（内嵌 xclbin+insts）"
    {
      echo "// generated by build_win.sh -- embedded NPU device binaries"
      echo "static const unsigned char kXclbin[] = {"
      od -An -v -tx1 "$XCLBIN" | awk '{for(i=1;i<=NF;i++)printf "0x%s,",$i}'
      echo "};"
      echo "static const unsigned int kXclbinLen = sizeof kXclbin;"
      echo "static const unsigned char kInsts[] = {"
      od -An -v -tx1 "$INSTS" | awk '{for(i=1;i<=NF;i++)printf "0x%s,",$i}'
      echo "};"
      echo "static const unsigned int kInstsLen = sizeof kInsts;"
    } > "$INC"
}
if [[ "$NEED_HIP" == 1 && "${QWENOX_BUILD_NPU:-1}" != 0 ]]; then
    NA="${NPUASSETS:-$HOME/npuprobe}"
    if npu_assets_ok "$NA" && ensure_xrt_lib "$NA" && gen_npu_bins "$NA"; then
        NPU_FLAGS=(-DQWENOX_NPU=1 -I"$(cygpath -w "$NA/xrtinc")"
                   -I"$(cygpath -w "$ROOT/build")")
        NPU_LINK=(-Xlinker build/winlibs/xrt_coreutil.lib
                  -Xlinker -delayload:xrt_coreutil.dll -Xlinker delayimp.lib)
        echo "[NPU] 编进 engine/bench/test（运行时 QWENOX_NPU=1 启用；QWENOX_BUILD_NPU=0 不编入）"
    fi
fi

# 并行执行若干 build_* 函数（仅用于相互无依赖的目标）：每个后台跑，输出先
# 收进 build/.log-<fn>.txt，结束后按序回放；失败目标回放大尾巴并汇总非零返回。
run_parallel() {
    local fns=("$@") pids=() i fail=0
    for fn in "${fns[@]}"; do
        "$fn" >"build/.log-$fn.txt" 2>&1 &
        pids+=($!)
    done
    for i in "${!fns[@]}"; do
        if wait "${pids[$i]}"; then
            cat "build/.log-${fns[$i]}.txt"
        else
            fail=1
            echo "[失败] ${fns[$i]}，日志末尾：" >&2
            tail -25 "build/.log-${fns[$i]}.txt" >&2
        fi
        rm -f "build/.log-${fns[$i]}.txt"
    done
    (( fail == 0 ))
}

build_engine() {
    echo "[编译] build/qwenox-engine-win.exe"
    # 先编到临时文件再原子替换，编译失败保留上次成功的二进制（对齐 Linux build.sh）
    "$HIPCC" "${FLAGS[@]}" ${NPU_FLAGS[@]+"${NPU_FLAGS[@]}"} src/gpu/qwenox.cpp \
      -o build/qwenox-engine-win.exe.tmp ${NPU_LINK[@]+"${NPU_LINK[@]}"} \
      && mv -f build/qwenox-engine-win.exe.tmp build/qwenox-engine-win.exe
}

build_bench() {
    echo "[编译] build/qwenox-bench.exe"
    "$HIPCC" "${FLAGS[@]}" ${NPU_FLAGS[@]+"${NPU_FLAGS[@]}"} -std=c++17 -Ithird_party \
      src/gpu/bench_main.cpp -o build/qwenox-bench.exe.tmp ${NPU_LINK[@]+"${NPU_LINK[@]}"} \
      && mv -f build/qwenox-bench.exe.tmp build/qwenox-bench.exe
}

# 静态页嵌入：编译生成器并重新生成 src/api/static_gen.inc（复用调用方设好的 CXX）。
# 生成器输出是确定性的，内容未变不会重写文件；static_gen.inc 是生成物，
# 不入库（.gitignore），此处按需重新生成。
gen_static_inc() {
    [[ -x build/gen_static_inc.exe && ! tools/gen_static_inc.cpp -nt build/gen_static_inc.exe ]] || {
        echo "[编译] build/gen_static_inc.exe"
        "$CXX" -O2 -std=c++17 tools/gen_static_inc.cpp -o build/gen_static_inc.exe || exit 1
    }
    echo "[生成] src/api/static_gen.inc"
    ./build/gen_static_inc.exe src/api/static src/api/static_gen.inc || exit 1
}

# GUI 子系统链接选项（托盘/入口程序，双击不出控制台）；入口仍是 main()。
# 链接器选项用 - 前缀：Git Bash 会把 / 开头的参数当路径改写。
GUI_LDFLAGS=()
gui_ldflags() {
    [[ ${#GUI_LDFLAGS[@]} -gt 0 ]] && return
    if "$CXX" -dumpmachine | grep -q msvc; then
      GUI_LDFLAGS=(-Xlinker -subsystem:windows -Xlinker -entry:mainCRTStartup)
    else
      GUI_LDFLAGS=(-mwindows)
    fi
}

build_api() {
    # OpenAI HTTP 前端：纯主机 C++，用 TheRock 自带 clang++（不拖 HIP 依赖）。
    # vision.cpp 的图片解码在 Windows 上走 stb_image（vendor 单头文件，
    # 编译进 exe，零新增 DLL），支持 PNG/JPEG；WebP 明确报错。
    # Windows 上是 GUI 子系统托盘程序（默认无控制台，--console 恢复），
    # 取代旧 start_win.exe 启动器的生态位：托盘菜单/网页 #/engine 启停引擎。
    # 15 个 TU 互相独立，多核并行编译后一次链接。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gen_static_inc
    gui_ldflags
    echo "[编译] build/qwenox-win.exe（15 个 TU 并行）"
    local srcs=(http engine_client tokenizer chat_template json_py toolparse vision
                reqstat reqstat_read power engine_sup tray_win snap_cuts reqlog main)
    local objdir=build/.api-obj
    mkdir -p "$objdir"
    # miniz 是 C：单独按 C 编译，别让 clang++ 按 C++ 语义处理。
    local m
    for m in miniz miniz_tdef miniz_tinfl miniz_zip; do
        "$CXX" -O2 -x c -D_CRT_SECURE_NO_WARNINGS -Ithird_party/miniz \
          -c "third_party/miniz/$m.c" -o "$objdir/$m.o" >"$objdir/$m.log" 2>&1 || {
            echo "  [失败] third_party/miniz/$m.c：" >&2
            tail -15 "$objdir/$m.log" >&2
            exit 1
        }
    done
    local pids=() s i fail=0
    for s in "${srcs[@]}"; do
        "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS "${VER_FLAGS[@]}" \
          -Isrc/api -Ithird_party -Ithird_party/miniz -I"$TR/include" \
          -c "src/api/$s.cpp" -o "$objdir/$s.o" >"$objdir/$s.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!srcs[@]}"; do
        wait "${pids[$i]}" || { fail=1; echo "  [失败] ${srcs[$i]}.cpp：" >&2; tail -15 "$objdir/${srcs[$i]}.log" >&2; }
    done
    [[ $fail == 0 ]] || exit 1
    "$CXX" "$objdir"/*.o -lws2_32 -lshell32 -luser32 -lgdi32 -ladvapi32 \
      "${GUI_LDFLAGS[@]}" -o build/qwenox-win.exe
}

# exe 文件图标（可选）：优先 TheRock 的 llvm-rc，没有则退回 Windows SDK 的
# rc.exe（取最新版本目录）；都没有时 RES 为空，调用方退化为无图标编译
# （程序内图标不受影响，那是内嵌 ICO）。结果写入 RES 数组。
mkiconres() {  # $1=rc 源文件  $2=res 输出  $3=exe 名（提示用）
    RES=()
    "$CXX" -dumpmachine | grep -q msvc || return 0
    local RC="$TR/lib/llvm/bin/llvm-rc.exe"
    local RCFLAGS=(-no-preprocess)
    if [[ ! -x "$RC" ]]; then
      RC="$(ls "/c/Program Files (x86)/Windows Kits/10/bin"/*/x64/rc.exe 2>/dev/null | sort -V | tail -1)"
      RCFLAGS=()
    fi
    if [[ -n "$RC" && -x "$RC" ]] && "$RC" "${RCFLAGS[@]}" -fo "$2" "$1"; then
      RES=("$2")
    else
      echo "提示：llvm-rc 不可用，$3 文件不带图标（程序内图标不受影响）" >&2
    fi
}

build_launcher() {
    # 根目录最小入口 start_win.exe：只负责以项目根为工作目录拉起
    # build\qwenox-win.exe（托盘程序）。旧启动器（配置面板/双进程看守）
    # 已归档到 attic/launcher/，不再编译。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gui_ldflags
    mkiconres src/start_stub.rc build/start_stub.res start_win.exe
    echo "[编译] start_win.exe"
    STUB_SRC=(-O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS src/start_stub_win.cpp
              -luser32 -lshell32 "${GUI_LDFLAGS[@]}" -o start_win.exe)
    if ! "$CXX" "${STUB_SRC[@]}" ${RES[@]+"${RES[@]}"}; then
      [[ ${#RES[@]} -gt 0 ]] || exit 1
      echo "提示：带图标资源链接失败，改为不带文件图标重试" >&2
      "$CXX" "${STUB_SRC[@]}"
    fi
}

build_settings() {
    # 根目录设置工具 settings_win.exe：双击打开配置面板（编辑参数 / 环境检查 /
    # 保存 service.conf），不启动服务。面板代码源自归档的旧启动器
    # （attic/launcher/），纯主机 C++，GUI 子系统。
    CXX="$TR/lib/llvm/bin/clang++.exe"
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gui_ldflags
    mkiconres src/settings_win.rc build/settings_win.res settings_win.exe
    echo "[编译] settings_win.exe"
    SETTINGS_SRC=(-O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS src/settings_win.cpp
                  -lws2_32 -luser32 -lshell32 -lgdi32 -lgdiplus -ldwmapi -luxtheme -lcomdlg32 -lole32
                  "${GUI_LDFLAGS[@]}" -o settings_win.exe)
    if ! "$CXX" "${SETTINGS_SRC[@]}" ${RES[@]+"${RES[@]}"}; then
      [[ ${#RES[@]} -gt 0 ]] || exit 1
      echo "提示：带图标资源链接失败，改为不带文件图标重试" >&2
      "$CXX" "${SETTINGS_SRC[@]}"
    fi
}

build_test() {
    python tools/kv_admission_test.py --cxx "$TR/lib/llvm/bin/clang++.exe"
    echo "[编译] build/ktest-win.exe"
    "$HIPCC" "${FLAGS[@]}" ${NPU_FLAGS[@]+"${NPU_FLAGS[@]}"} -I src/gpu tools/ktest.cu \
      -o build/ktest-win.exe ${NPU_LINK[@]+"${NPU_LINK[@]}"}
    echo "[运行] ktest-win（预期末尾 ALL PASS）"
    (cd build && HIP_PATH="$TR_WIN" ROCM_PATH="$TR_WIN" ./ktest-win.exe)
}

# NPU 检测工具（仅 Windows）：GUI 版 build/npucheck-win.exe（xclbin/insts 内嵌、
# xrt_coreutil.dll delay-load——没装 NPU 驱动也能打开做 BIOS 检查）+ 脚本版用的
# build/npuio.exe / build/gpuload.exe。资产（XRT 2.19 头、xrt_coreutil.def、
# xclbin/insts）不进仓库，默认 ~/npuprobe，NPUASSETS 覆盖。XRT DLL 必须用
# System32 那份（DriverStore 的 shim 拒绝 register_xclbin），故只链导入库。
# 注意 .lib 走 -Xlinker（hipcc 会把 .lib 输入当 HIP 源）；链接器选项用 -
# 前缀（Git Bash 会把 / 开头的参数当路径改写）。
build_npuio() {
    echo "[编译] build/npuio.exe"
    "$HIPCC" -O2 -std=c++17 --offload-arch="$GPU_ARCH" -D_CRT_SECURE_NO_WARNINGS \
      -I"$NPU_XRTINC" -Wno-everything -c tools/npuprobe/npuio.hip -o build/npuio.o \
      && "$HIPCC" build/npuio.o -o build/npuio.exe.tmp -Xlinker build/winlibs/xrt_coreutil.lib \
      && mv -f build/npuio.exe.tmp build/npuio.exe || exit 1
}
build_gpuload() {
    echo "[编译] build/gpuload.exe"
    "$HIPCC" -O2 -std=c++17 --offload-arch="$GPU_ARCH" -Wno-everything \
      tools/npuprobe/gpuload.hip -o build/gpuload.exe.tmp \
      && mv -f build/gpuload.exe.tmp build/gpuload.exe || exit 1
}
build_npucheck_gui() {
    CXX="$TR/lib/llvm/bin/clang++.exe"  # gui_ldflags 用它判断目标三元组
    [[ -x "$CXX" ]] || { echo "找不到 TheRock clang++: $CXX" >&2; exit 1; }
    gui_ldflags
    echo "[编译] build/npucheck-win.exe"
    "$HIPCC" -O2 -std=c++17 --offload-arch="$GPU_ARCH" -D_CRT_SECURE_NO_WARNINGS \
      -I"$NPU_XRTINC" -I"$NPU_INCDIR" -Wno-everything \
      -c tools/npucheck/npucheck_win.hip -o build/npucheck_win.o \
      && "$HIPCC" build/npucheck_win.o -o build/npucheck-win.exe.tmp \
         -Xlinker build/winlibs/xrt_coreutil.lib -Xlinker -delayload:xrt_coreutil.dll \
         -Xlinker delayimp.lib -lwevtapi -luser32 -lgdi32 -ladvapi32 "${GUI_LDFLAGS[@]}" \
      && mv -f build/npucheck-win.exe.tmp build/npucheck-win.exe || exit 1
}
build_npu() {
    local NA="${NPUASSETS:-$HOME/npuprobe}"
    npu_assets_ok "$NA" \
      || { echo "找不到 XRT 资产: $NA（需 xrtinc/、xrt_coreutil.def 与 npu/ 下的" \
                "xclbin/insts，设 NPUASSETS=...）" >&2; exit 1; }
    ensure_xrt_lib "$NA" || exit 1
    gen_npu_bins "$NA"

    NPU_XRTINC="$(cygpath -w "$NA/xrtinc")"
    NPU_INCDIR="$(cygpath -w "$ROOT/build")"
    run_parallel build_npuio build_gpuload build_npucheck_gui
}

case "$TARGET" in
  all)
    run_parallel build_engine build_bench build_api build_launcher build_settings
    ;;
  engine) build_engine ;;
  bench) build_bench ;;
  api) build_api ;;
  launcher) build_launcher ;;
  settings) build_settings ;;
  test) build_test ;;
  npu) build_npu ;;
esac
echo '[完成] 编译输出位于 build/；启动服务用 ./start_win.exe（或 bash start_win.sh），改配置用 ./settings_win.exe'
