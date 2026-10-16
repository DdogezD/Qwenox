#!/usr/bin/env bash
# npucheck.sh -- Strix Halo (gfx1151) NPU 检测工具
#
#   bash npucheck.sh info                 BIOS/AGESA/NPU 驱动检查（只读，无风险）
#   bash npucheck.sh stress [选项]         GPU+NPU 压测（含并发死机测试，见下）
#   bash npucheck.sh postmortem           死机重启后：分析上次压测发生了什么
#   bash npucheck.sh whea ["YYYY-MM-DD HH:MM:SS"]   查 WHEA-Logger 事件
#
# stress 选项：
#   --npu-only [秒=60]      只跑 NPU 基线（安全）
#   --gpu-only [秒=30]      只跑 GPU 显存拷贝基线（安全）
#   --secs N                并发阶段时长（默认 60）
#   --engine <len>          GPU 负载换成真实引擎 prefill（8192/32768/65536/131072）
#   --tag <名>              日志目录名（默认 stress_时间戳）
#   --force                 AGESA 未修复/未知时仍然执行（有硬死机风险！）
#
# 背景：Strix Halo AGESA < 1.0.0.2b 存在 Data Fabric 固件缺陷，GPU+NPU
# 并发会硬死机（ROCm/TheRock#8709，AMD 2026-10-06 确认）。本工具:
#   1) info 直接从原始 SMBIOS 读 AGESA 版本并给出结论（无需 HWiNFO）；
#   2) stress 每个阶段前写 MARKER 文件并每秒查 WHEA，出现新 WHEA 事件立即中止
#      （#8709 中 deferred MCE 比死机早几秒出现）；
#   3) 真死机了，重启后 postmortem 靠 MARKER + Kernel-Power 41 + npu.log
#      最后心跳定性。
set -uo pipefail
P="$(cd "$(dirname "$0")" && pwd)"
# 优先用根构建产物（bash build_win.sh npu → build/），退回本地 out/（旧 build.sh）
if [[ -x "$P/../../build/npuio.exe" && -x "$P/../../build/gpuload.exe" ]]; then
  OUT="$(cd "$P/../../build" && pwd)"
else
  OUT="$P/out"
fi
LOGS="$P/logs"
NPUASSETS="${NPUASSETS:-$HOME/npuprobe}"
NPU_DIR="$NPUASSETS/npu"
XRTSMI="/c/Windows/System32/AMD/xrt-smi.exe"
ENGROOT="${ENGROOT:-$HOME/Workspace/gfx1151-engine}"

# AGESA 修复阈值：>= 1.0.0.2b 视为已修复（AMD 在 ROCm/TheRock#8709 的答复）。
# 已知受影响：1.0.0.0c（Beelink GTR9 Pro，锁 FCLK 仍死）、1.0.0.1b（本机 BIOS 1.05，
# 2026-10-13 并发实测死机）、1.0.0.2（GMKtec BIOS 1.13，#8709 报告机）。
FIXED="1.0.0.2b"

psq() { # 跑一段 PowerShell，UTF-8 输出，去掉 \r
  powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $*" 2>/dev/null | tr -d '\r'
}

# ---------- AGESA ----------

read_agesa() { # 输出 "StrixHaloPI-FP11 1.0.0.1b"，读不到则为空
  psq 'try { $b=(Get-CimInstance -Namespace root/wmi -ClassName MSSmBios_RawSMBiosTables -ErrorAction Stop).SMBiosData; $s=[Text.Encoding]::ASCII.GetString($b); $m=[regex]::Match($s,"AGESA!V9\s+(\S+?)\s+(\d+\.\d+\.\d+\.\d+[a-z]?)"); if($m.Success){$m.Groups[1].Value+" "+$m.Groups[2].Value} } catch {}' | tail -1
}

agesa_parts() { sed -E 's/^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)([a-z]?)$/\1 \2 \3 \4 \5/' <<<"$1"; }

agesa_lt() { # $1 < $2 ?
  local a1 a2 a3 a4 as b1 b2 b3 b4 bs
  read -r a1 a2 a3 a4 as <<<"$(agesa_parts "$1")"
  read -r b1 b2 b3 b4 bs <<<"$(agesa_parts "$2")"
  local i x y
  for i in 1 2 3 4; do
    eval "x=\$a$i; y=\$b$i"
    ((10#$x < 10#$y)) && return 0
    ((10#$x > 10#$y)) && return 1
  done
  [[ "$as" < "$bs" ]]
}

agesa_verdict() { # $1 = 版本串；输出 SAFE/UNSAFE/UNKNOWN
  local v="$1"
  [[ -z "$v" ]] && { echo UNKNOWN; return; }
  if agesa_lt "$v" "$FIXED"; then echo UNSAFE; else echo SAFE; fi
}

# ---------- WHEA ----------

whea_count() { # $1 = since "YYYY-MM-DD HH:MM:SS"
  psq "\$t=[datetime]::Parse('$1'); @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; StartTime=\$t} -ErrorAction SilentlyContinue).Count" | grep -E '^[0-9]+$' | tail -1
}

cmd_whea() {
  local since="${1:-$(date -d '1 hour ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')}"
  echo "WHEA-Logger events since $since:"
  psq "\$t=[datetime]::Parse('$since'); \$ev=@(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; StartTime=\$t} -ErrorAction SilentlyContinue); 'count=' + \$ev.Count; foreach(\$e in \$ev){ \$m=(\$e.Message -replace '\s+',' '); if(\$m.Length -gt 200){\$m=\$m.Substring(0,200)}; '{0:HH:mm:ss} id={1} {2}' -f \$e.TimeCreated, \$e.Id, \$m }"
}

# ---------- info ----------

cmd_info() {
  echo "===== BIOS / AGESA ====="
  psq '$b=Get-CimInstance Win32_BIOS; $s=Get-ItemProperty "HKLM:\HARDWARE\DESCRIPTION\System\BIOS"; "vendor  : " + $b.Manufacturer; "bios    : " + $b.SMBIOSBIOSVersion + "  (" + $b.ReleaseDate.ToString("yyyy-MM-dd") + ")"; "machine : " + $s.SystemManufacturer + " " + $s.SystemProductName'
  local ag; ag="$(read_agesa)"
  local fam="" ver=""
  if [[ -n "$ag" ]]; then fam="${ag% *}"; ver="${ag##* }"; fi
  local vd; vd="$(agesa_verdict "$ver")"
  if [[ -n "$ver" ]]; then
    echo "agesa   : $fam $ver"
  else
    echo "agesa   : (SMBIOS 里没读到，请用 BIOS 设置页或 HWiNFO64 确认)"
  fi
  echo "修复阈值: AGESA >= $FIXED 才不会 GPU+NPU 并发死机（ROCm/TheRock#8709，AMD 2026-10-06）"
  case "$vd" in
    SAFE)    echo "结论    : SAFE - AGESA 已修复，可做并发测试" ;;
    UNSAFE)  echo "结论    : UNSAFE - 固件未修复，GPU+NPU 并发会硬死机；stress 需要 --force" ;;
    UNKNOWN) echo "结论    : UNKNOWN - 无法判定；未确认修复前 stress 需要 --force" ;;
  esac

  echo
  echo "===== NPU ====="
  if [[ -x "$XRTSMI" ]]; then
    "$XRTSMI" examine 2>/dev/null | tr -d '\r' | grep -E '^\s+(Version|NPU Driver|NPU Firmware)|\|NPU' | sed 's/^ *//'
    local pm; pm=$("$XRTSMI" examine -r platform 2>/dev/null | tr -d '\r' | sed -n 's/.*Power Mode *: *//p' | head -1)
    echo "power mode : ${pm:-?}"
  else
    echo "xrt-smi 不存在（$XRTSMI）→ NPU 驱动未安装或未启用"
  fi

  echo
  echo "===== 遗留压测标记 ====="
  local m; m=$(ls -t "$LOGS"/*/MARKER 2>/dev/null | head -1)
  if [[ -n "$m" ]]; then
    echo "发现 MARKER: $m —— 上次压测中断（可能死机），请运行: bash npucheck.sh postmortem"
  else
    echo "无（上次压测正常结束，或未运行过）"
  fi
  [[ "$vd" == SAFE ]]
}

# ---------- stress ----------

die() { echo "$*" >&2; exit 1; }

write_marker() { # $1 = phase
  mkdir -p "$L"
  { echo "tag=$TAG"
    echo "start=$SINCE"
    echo "phase=$1"
    echo "phase_start=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "logdir=$(cygpath -w "$L")"
  } >"$L/MARKER"
}

npu_start() { # $1 = secs；等待 READY，置 $NPID
  rm -f "$L/STOP"
  ( cd "$OUT" && ./npuio.exe "$(cygpath -w "$NPU_DIR")" stress "$1" \
      "$(cygpath -w "$L/npu.log")" "$(cygpath -w "$L/STOP")" ) >"$L/npu.stdout" 2>&1 &
  NPID=$!
  local i
  for i in $(seq 120); do
    grep -q -E 'READY|SETUP_FAIL' "$L/npu.log" 2>/dev/null && break
    kill -0 "$NPID" 2>/dev/null || break
    sleep 0.5
  done
  grep -q READY "$L/npu.log" 2>/dev/null && return 0
  echo "NPU stress 未能启动:"; tail -5 "$L/npu.stdout" 2>/dev/null
  return 1
}

npu_stop() {
  touch "$L/STOP"
  local i
  for i in $(seq 40); do kill -0 "$NPID" 2>/dev/null || break; sleep 0.25; done
  kill "$NPID" 2>/dev/null
  wait "$NPID" 2>/dev/null
}

npu_summary() { # $1 = 标题
  echo "---- NPU ($1): ms/run 每秒"
  grep '^t=' "$L/npu.log" 2>/dev/null | awk '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="ms/run") v=a[2]; if(a[1]=="worst_wait_ms") w=a[2]}; n++; s=s sprintf("%.2f ", v); if(w>mw) mw=w} END {print s; print "seconds=" n " worst_wait_ms=" mw}' | fold -w 200
  grep -E 'MISMATCH|TIMEOUT|END' "$L/npu.log" 2>/dev/null
}

run_engine() {
  ( cd "$ENGROOT" && env QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
      QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 \
      QWENOX_PREFILL_CHUNK=8192 QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 QWENOX_INDEX_FUSED2=1 \
      QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 \
      build/qwenox-engine-win.exe models/qwen38-flash-next-v2.hgn models/qwen38-flash-next-ngram.hgn \
      --tokens-file "data/qsa-oracle/$ENG_LEN.tokens" --gen 1 --maxctx $((ENG_LEN + 8192)) ) \
      >"$L/engine.log" 2>&1
}

cmd_stress() {
  local MODE=concurrent CON_SECS=60 NPU_SECS=60 GPU_SECS=30 FORCE=0 ENG_LEN="" TAG=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --npu-only) MODE=npu; [[ "${2:-}" =~ ^[0-9]+$ ]] && { NPU_SECS=$2; shift; } ;;
      --gpu-only) MODE=gpu; [[ "${2:-}" =~ ^[0-9]+$ ]] && { GPU_SECS=$2; shift; } ;;
      --secs)     CON_SECS="${2:?}"; shift ;;
      --engine)   ENG_LEN="${2:?}"; shift ;;
      --tag)      TAG="${2:?}"; shift ;;
      --force)    FORCE=1 ;;
      *) die "未知参数: $1" ;;
    esac
    shift
  done
  [[ -x "$OUT/npuio.exe" && -x "$OUT/gpuload.exe" ]] || die "out/ 里没有 npuio.exe/gpuload.exe，先运行 bash build.sh"
  [[ -f "$NPU_DIR/final_4096x2560x1024_128x64x128.xclbin" ]] || die "找不到 NPU 二进制: $NPU_DIR（设 NPUASSETS 环境变量）"
  [[ -z "$ENG_LEN" || -f "$ENGROOT/data/qsa-oracle/$ENG_LEN.tokens" ]] || die "没有 data/qsa-oracle/$ENG_LEN.tokens"

  # 预检
  local ag ver vd; ag="$(read_agesa)"; ver="${ag##* }"; [[ -n "$ag" ]] || ver=""
  vd="$(agesa_verdict "$ver")"
  TAG="${TAG:-stress_$(date +%Y%m%d_%H%M%S)}"
  L="$LOGS/$TAG"
  SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

  echo "== $TAG  mode=$MODE  agesa=${ver:-unknown} verdict=$vd  gpu_load=$([[ -n "$ENG_LEN" ]] && echo "engine:$ENG_LEN" || echo gpuload)"
  if [[ "$MODE" == concurrent && "$vd" != SAFE ]]; then
    if [[ "$FORCE" != 1 ]]; then
      echo
      echo "AGESA ${ver:-未知} 未修复并发死机缺陷（需要 >= $FIXED）。"
      echo "继续执行有硬死机风险（只能断电重启）。确认要测就加 --force。"
      exit 2
    fi
    echo
    echo "!!! 警告：AGESA ${ver:-未知} < $FIXED，GPU+NPU 并发在同类固件上实测硬死机 !!!"
    echo "!!! 10 秒后开始（Ctrl+C 取消）；重要文件先保存，日志/MARKER 已落盘 !!!"
    local i; for i in $(seq 10 -1 1); do echo -n "$i "; sleep 1; done; echo
  fi

  # WHEA 看门狗自检
  local W0; W0="$(whea_count "$SINCE")"
  [[ "$W0" =~ ^[0-9]+$ ]] || die "WHEA 看门狗不可用，拒绝继续（没有这个保护不能跑压测）"

  write_marker preflight
  local ABORT=""

  # ---- 阶段 1: NPU 独占 ----
  if [[ "$MODE" != gpu ]]; then
    write_marker npu-only
    echo "== 阶段 1: NPU 独占 ${NPU_SECS}s"
    npu_start "$NPU_SECS" || { echo "RESULT: FAIL (NPU setup)"; exit 1; }
    wait "$NPID"; local rc=$?
    npu_summary "独占"
    [[ $rc == 0 ]] || { echo "RESULT: FAIL (NPU rc=$rc)"; rm -f "$L/MARKER"; exit 1; }
  fi

  # ---- 阶段 2: GPU 独占 ----
  if [[ "$MODE" != npu && -z "$ENG_LEN" ]]; then
    write_marker gpu-only
    echo "== 阶段 2: GPU 独占 ${GPU_SECS}s（显存拷贝）"
    ( cd "$OUT" && ./gpuload.exe "$GPU_SECS" ) | tee "$L/gpu.log"
  fi

  # ---- 阶段 3: 并发 ----
  if [[ "$MODE" == concurrent ]]; then
    write_marker concurrent
    echo "== 阶段 3: GPU+NPU 并发 ${CON_SECS}s"
    local npu_cap=$((CON_SECS + 60)); [[ -n "$ENG_LEN" ]] && npu_cap=7200
    npu_start "$npu_cap" || { echo "RESULT: FAIL (NPU setup)"; exit 1; }
    sleep 3   # 先留几秒 NPU 独占基线
    if [[ -n "$ENG_LEN" ]]; then
      run_engine & GPID=$!
    else
      ( cd "$OUT" && ./gpuload.exe "$CON_SECS" ) >"$L/gpu.log" 2>&1 & GPID=$!
    fi
    local T0=$SECONDS
    while kill -0 "$GPID" 2>/dev/null; do
      sleep 1
      local w; w="$(whea_count "$SINCE")"
      if [[ "$w" =~ ^[0-9]+$ ]] && (( w > W0 )); then ABORT="WHEA ($w events)"; break; fi
      if ! kill -0 "$NPID" 2>/dev/null; then ABORT="NPU loop 退出"; break; fi
      if grep -q -E 'MISMATCH|TIMEOUT' "$L/npu.log" 2>/dev/null; then ABORT="NPU 结果错误"; break; fi
    done
    if [[ -n "$ABORT" ]]; then
      kill "$GPID" 2>/dev/null; taskkill //F //IM qwenox-engine-win.exe >/dev/null 2>&1
      wait "$GPID" 2>/dev/null
    else
      wait "$GPID"
      [[ -n "$ENG_LEN" ]] && grep -E 'prefill: [0-9]+ tokens in' "$L/engine.log" | tail -2 | tr -d '\r'
    fi
    npu_stop
    echo "并发耗时 $((SECONDS - T0))s"
    npu_summary "并发段（含前后几秒独占）"
    [[ -z "$ENG_LEN" ]] && cat "$L/gpu.log"
    if ! grep -q 'END .*mismatches=0 rc=0' "$L/npu.log"; then
      rm -f "$L/MARKER"
      echo "RESULT: FAIL（NPU 结果校验未通过；日志在 $L）"
      exit 1
    fi
  fi

  echo "---- WHEA since $SINCE: $(whea_count "$SINCE")"
  if [[ -n "$ABORT" ]]; then
    echo "RESULT: ABORTED - $ABORT（机器还活着；日志在 $L）"
    rm -f "$L/MARKER"
    exit 2
  fi
  rm -f "$L/MARKER"
  if [[ "$MODE" == gpu ]]; then
    echo "RESULT: PASS（存活，无新增 WHEA）"
  else
    echo "RESULT: PASS（存活，NPU 结果全部逐位一致，无新增 WHEA）"
  fi
}

# ---------- postmortem ----------

cmd_postmortem() {
  local m; m=$(ls -t "$LOGS"/*/MARKER 2>/dev/null | head -1)
  [[ -z "$m" ]] && { echo "没有找到遗留的 MARKER —— 上次压测正常结束（或没跑过）。"; return 0; }
  local d; d="$(dirname "$m")"
  local tag start phase pstart logdir
  tag=$(sed -n 's/^tag=//p' "$m"); start=$(sed -n 's/^start=//p' "$m")
  phase=$(sed -n 's/^phase=//p' "$m"); pstart=$(sed -n 's/^phase_start=//p' "$m")
  echo "发现中断的压测: $m"
  echo "  tag=$tag  开始=$start  中断时阶段=$phase（阶段开始=$pstart）"
  echo
  echo "---- 最后的心跳（npu.log 末尾 3 行）----"
  tail -3 "$d/npu.log" 2>/dev/null || echo "(无 npu.log)"
  local mt; mt=$(stat -c '%y' "$d/npu.log" 2>/dev/null | cut -d. -f1)
  [[ -n "$mt" ]] && echo "  日志最后写入: $mt（≈ 死机时刻）"
  echo
  echo "---- Kernel-Power 41（非正常关机）事件（自 $start 起）----"
  psq "\$t=[datetime]::Parse('$start'); \$ev=@(Get-WinEvent -FilterHashtable @{LogName='System'; Id=41; StartTime=\$t} -MaxEvents 5 -ErrorAction SilentlyContinue); if(\$ev.Count -eq 0){'(无)'}; foreach(\$e in \$ev){ '{0:yyyy-MM-dd HH:mm:ss} BugcheckCode={1}' -f \$e.TimeCreated, \$e.Properties[0].Value }"
  echo
  echo "---- WHEA 事件（自 $start 起）----"
  cmd_whea "$start"
  echo
  local kp; kp=$(psq "\$t=[datetime]::Parse('$start'); \$e=Get-WinEvent -FilterHashtable @{LogName='System'; Id=41; StartTime=\$t} -MaxEvents 1 -ErrorAction SilentlyContinue; if(\$e){\$e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss')}" | grep -E '^[0-9]' | head -1)
  if [[ -n "$kp" ]]; then
    local ps_s kp_s
    ps_s=$(date -d "$pstart" +%s 2>/dev/null); kp_s=$(date -d "$kp" +%s 2>/dev/null)
    echo "结论: 确认硬死机/断电（Kernel-Power 41 @ $kp）。"
    [[ -n "$ps_s" && -n "$kp_s" ]] && echo "      距 '$phase' 阶段开始约 $((kp_s - ps_s)) 秒。"
  else
    echo "结论: 压测中断但没有 Kernel-Power 41 记录 —— 可能是手动结束或掉电未记录，请结合上面的日志看。"
  fi
  echo "（确认无误后可删除标记: rm \"$m\"）"
}

# ---------- main ----------

case "${1:-info}" in
  info)       cmd_info ;;
  stress)     shift; cmd_stress "$@" ;;
  postmortem) cmd_postmortem ;;
  whea)       shift; cmd_whea "$@" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
