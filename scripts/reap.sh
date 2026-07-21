#!/bin/sh
# scripts/reap.sh -- find and kill orphaned bessemer test / benchmark / MPI
# processes left behind by an interrupted or killed run.
#
# WHY: mpirun/prterun children and the debug-device sweep can survive a killed
# ctest and keep spinning (a 100%-CPU deck_test, a 0%-CPU adaptive_tgv), or sit
# for DAYS. Run this after killing any run, and at the end of a session, so a
# stray process never pins a core in the background.
#
#   scripts/reap.sh          # list what would be killed, then kill it
#   scripts/reap.sh --dry    # list only, kill nothing

# Match this repo's test/app/bench binaries -- both absolute
# (.../bessemer/build/cpu/test/foo) and repo-relative (build/cpu/test/foo, as
# an mpirun/prterun argument) forms contain "build/<preset>/<test|apps|bench>/".
# ERE, not BRE: use (a|b|c), never \(a\|b\|c\).
_pat='build/[^/ ]+/(test|apps|bench)/'

_pids=$(ps -eo pid,cmd | grep -E "$_pat" | grep -v -e grep -e reap.sh | awk '{print $1}')

if [ -z "$_pids" ]; then
  echo "reap.sh: no orphaned bessemer test/mpi processes found."
  exit 0
fi

echo "reap.sh: orphaned bessemer processes:"
ps -o pid,etimes,%cpu,cmd -p $(echo "$_pids" | tr '\n' ',' | sed 's/,$//') 2>/dev/null

if [ "$1" = "--dry" ]; then
  echo "reap.sh: --dry, nothing killed."
  exit 0
fi

echo "reap.sh: killing..."
# shellcheck disable=SC2086
kill -9 $_pids 2>/dev/null
sleep 1
_left=$(ps -eo pid,cmd | grep -E "$_pat" | grep -v -e grep -e reap.sh | wc -l)
echo "reap.sh: done ($_left still alive)."
