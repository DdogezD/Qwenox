#!/usr/bin/env bash
# pack_symbols_win.sh — 发版时另存调试符号（Windows，Git Bash）。
#
# 用户分发包里不带 PDB；收到用户的崩溃 minidump（crash-engine-*.dmp）后，要用
# 同一次编译的 exe + PDB 才能还原函数名与源码行。每次发版在 build_win.sh all
# 之后运行一次，把 dist/qwenox-symbols-<版本>-<git>.zip 存好（上传到 GitHub
# Release 的附件，或留在本机存档）。
#
# 用法: bash tools/pack_symbols_win.sh      → dist/qwenox-symbols-<ver>-<git>.zip
# 包内：build/*.pdb、对应的 exe、GPU 运行库 DLL（minidump 里的模块按 PE 时间戳
# 匹配，同版本 DLL 才能对上）、BUILD-INFO.txt（版本、提交、各文件 PE 时间戳）。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VER="$(tr -d ' \r\n' < VERSION)"
GIT="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
mkdir -p dist
OUT="dist/qwenox-symbols-$VER-$GIT.zip"
ls build/*.pdb >/dev/null 2>&1 || { echo "build/ 里没有 PDB：先运行 bash build_win.sh all（QWENOX_PDB 不能为 0）" >&2; exit 1; }
PY="$(command -v python || command -v python3)"
"$PY" - "$OUT" "$VER" "$GIT" <<'EOF'
import glob, os, struct, sys, time, zipfile
out, ver, git = sys.argv[1:4]
files = sorted(glob.glob("build/*.pdb"))
for pdb in list(files):
    exe = pdb[:-4] + ".exe"
    if os.path.exists(exe): files.append(exe)
files += [f for f in ("build/amdhip64_7.dll", "build/amd_comgr.dll", "build/rocblas.dll",
                      "build/libhipblaslt.dll", "build/rocm_kpack.dll", "build/origami.dll")
          if os.path.exists(f)]
def pe_stamp(p):
    try:
        with open(p, "rb") as f:
            d = f.read(4096)
        off = struct.unpack_from("<I", d, 0x3C)[0]
        ts = struct.unpack_from("<I", d, off + 8)[0]
        size = struct.unpack_from("<I", d, off + 24 + 56)[0]
        return "ts=0x%08X size=0x%08X" % (ts, size)
    except Exception:
        return ""
info = ["Qwenox symbols %s (%s), packed %s" % (ver, git, time.strftime("%Y-%m-%d %H:%M:%S")), ""]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)
        info.append("%-34s %12d  %s" % (f, os.path.getsize(f), pe_stamp(f) if not f.endswith(".pdb") else ""))
    z.writestr("BUILD-INFO.txt", "\n".join(info) + "\n")
print(out, os.path.getsize(out), "bytes,", len(files), "files")
EOF
