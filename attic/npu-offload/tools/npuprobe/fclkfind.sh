#!/usr/bin/env bash
# fclkfind.sh -- read-only: dump PM table idle and under GPU-only load, to find
# the live FCLK entry. Optional $1 = label for output files.
set -u
P="$(cd "$(dirname "$0")" && pwd)"; RA="$HOME/Downloads/ryzenadj-win64"; TAG="${1:-find}"
cd "$RA"
./ryzenadj.exe --dump-table > "$P/pm_${TAG}_idle.txt" 2>&1
( cd "$P/out" && ./gpuload.exe 12 ) > "$P/pm_${TAG}_load.log" 2>&1 &
G=$!
sleep 6
./ryzenadj.exe --dump-table > "$P/pm_${TAG}_busy.txt" 2>&1
wait $G
cat "$P/pm_${TAG}_load.log"
sleep 4
./ryzenadj.exe --dump-table > "$P/pm_${TAG}_after.txt" 2>&1
echo done
