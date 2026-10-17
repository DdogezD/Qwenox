#!/bin/sh
# usage: fabric-clock.sh hold | release | status   （hold/release 需要 root）
#
# 把 GPU 的 FCLK（pp_dpm_fclk）钉在最高档。NPU（amdxdna）忙的时候 FCLK 变速
# 会挂死整机，启用 QWENOX_NPU 前必须先 hold（引擎启动时也会自检）。重启失效。
set -u
for dev in /sys/class/drm/card*/device; do
  [ -e "$dev/pp_dpm_fclk" ] && break
done
[ -e "$dev/pp_dpm_fclk" ] || { echo "fabric-clock: 没有 amdgpu pp_dpm_fclk" >&2; exit 1; }
top=$(grep -c . "$dev/pp_dpm_fclk")
top=$((top - 1))

held() {
  [ "$(cat "$dev/power_dpm_force_performance_level")" = manual ] &&
    [ "$(grep -c '\*' "$dev/pp_dpm_fclk")" = 1 ] &&
    grep '\*' "$dev/pp_dpm_fclk" | grep -q "^$top:"
}

case "${1:-status}" in
  hold)
    echo manual > "$dev/power_dpm_force_performance_level" &&
      echo "$top" > "$dev/pp_dpm_fclk" || exit 1
    i=0
    until held || [ "$i" -ge 50 ]; do sleep 0.1; i=$((i + 1)); done
    held && echo "FCLK 已锁定在最高档（$top）" || { echo "fabric-clock: 锁定失败" >&2; exit 1; } ;;
  release)
    echo auto > "$dev/power_dpm_force_performance_level" && echo 已解锁 ;;
  status)
    if held; then echo "held"; else echo "NOT held"; exit 1; fi ;;
  *) echo "usage: $0 hold | release | status" >&2; exit 2 ;;
esac
