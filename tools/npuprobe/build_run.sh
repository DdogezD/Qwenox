#!/usr/bin/env bash
# Git Bash on Windows: build npuio.exe against the NPU driver's XRT DLLs and run it.
# Expects next to this script: npuio.hip, xrtinc/ (XRT 2.19 headers), xrt_coreutil.def,
# npu/ (gufo-npu xclbin + insts).
set -uo pipefail
cd "$(dirname "$0")"
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
ENG="${ENG:-$HOME/Workspace/gfx1151-engine/build}"
export HIP_PATH="$(cygpath -w "$TR")"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
DS=$(ls -d /c/Windows/System32/DriverStore/FileRepository/kipudrv.inf_amd64_* | while read d; do
  [ -f "$d/xrt_core.dll" ] && echo "$(stat -c %Y "$d/xrt_coreutil.dll") $d"; done | sort -n | tail -1 | cut -d' ' -f2)
echo "driver dir: $DS"
mkdir -p out
# Do NOT copy the DriverStore XRT DLLs here: System32\xrt_coreutil.dll must be used.
rm -f out/xrt_coreutil.dll out/xrt_core.dll
for d in amdhip64_7.dll amd_comgr.dll rocm_kpack.dll msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
  [ -f "out/$d" ] || cp -f "$ENG/$d" out/ 2>/dev/null || cp -f "$TR/bin/$d" out/ 2>/dev/null || true
done
LLVMLIB="$TR/lib/llvm/bin/llvm-lib.exe"; [ -x "$LLVMLIB" ] || LLVMLIB="/c/Program Files/AMD/ROCm/7.2/bin/llvm-lib.exe"
"$LLVMLIB" /def:xrt_coreutil.def /machine:x64 /out:out/xrt_coreutil.lib >/dev/null || { echo "llvm-lib failed"; echo "RESULT: FAIL (build)"; exit 1; }
rm -f out/npuio.exe out/npuio.o
"$TR/bin/hipcc.exe" -O2 -std=c++17 --offload-arch=gfx1151 -Ixrtinc -Wno-everything \
  -c npuio.hip -o out/npuio.o 2>&1 | cut -c1-300 | tail -30
"$TR/bin/hipcc.exe" out/npuio.o -o out/npuio.exe -Xlinker out/xrt_coreutil.lib 2>&1 | cut -c1-300 | tail -30
[ -f out/npuio.exe ] || { echo "RESULT: FAIL (build)"; exit 1; }
echo "build OK"
cd out && ./npuio.exe ../npu "${1:-20}"
