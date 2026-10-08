#!/bin/bash
# bench/dfg3/run_queue.sh QUEUE OUTDIR [BALLAST_ARGS]
#
# Runs the DFG 2D-3 study queue on two slots of np 4, each rank pinned to its
# own physical core (slot 0: cores 0-3, slot 1: cores 4-7; the hyperthread
# siblings stay idle) -- so wall times are comparable between runs: every
# timed run shares the machine with exactly one other np-4 job.
#
# QUEUE: one run per line, `name|dfg_cylinder args` (blank lines and lines
# starting with # are skipped); `-c 3 -out OUTDIR/name` is added. Each run
# logs to OUTDIR/name.log and appends one line to OUTDIR/summary.txt:
# `name=... kind=run core0=... started=<epoch> RESULT ...` (status=crashed if
# the app printed no RESULT line). Resumable: names already in summary.txt are
# skipped.
#
# BALLAST_ARGS: when one slot has drained the queue while the other still runs
# a real job, the idle slot repeats this run (kind=ballast) to keep the load
# constant; the repeats double as a timing-noise sample.
#
# Launch detached:  setsid nohup bench/dfg3/run_queue.sh Q OUT "args" &
# Never above 4 ranks per run (CLAUDE.md guardrail).

set -u
QUEUE=$1
OUT=$2
BALLAST=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
APP=$REPO/build/cpu/apps/dfg_cylinder
PIN=$REPO/bench/dfg3/pin.sh
# shellcheck disable=SC1091
. "$REPO/scripts/env.sh" > /dev/null || exit 1
mkdir -p "$OUT"
LOCK=$OUT/.lock
CLAIMED=$OUT/.claimed
# Resume: only names with a summary line count as claimed.
if [ -f "$OUT/summary.txt" ]; then
   sed -n 's/^name=\([^ ]*\) kind=run .*/\1/p' "$OUT/summary.txt" > "$CLAIMED"
else
   : > "$CLAIMED"
fi

# Print the next unclaimed queue line and claim it (atomic across slots).
next_job() {
   (
      flock 9
      while IFS= read -r line; do
         case "$line" in ''|'#'*) continue ;; esac
         name=${line%%|*}
         grep -qx "$name" "$CLAIMED" && continue
         echo "$name" >> "$CLAIMED"
         echo "$line"
         break
      done < "$QUEUE"
   ) 9> "$LOCK"
}

# run_one CORE0 NAME ARGS KIND
run_one() {
   local core0=$1 name=$2 args=$3 kind=$4
   local d="$OUT/$name"
   mkdir -p "$d"
   local t0
   t0=$(date +%s)
   # shellcheck disable=SC2086
   mpirun -np 4 --bind-to none "$PIN" "$core0" "$APP" -c 3 $args -out "$d" \
      > "$d.log" 2>&1
   local rc=$?
   local res
   res=$(grep '^RESULT' "$d.log" | tail -1)
   [ -z "$res" ] && res="RESULT status=crashed rc=$rc"
   (
      flock 9
      echo "name=$name kind=$kind core0=$core0 started=$t0 $res" >> "$OUT/summary.txt"
   ) 9> "$LOCK"
}

worker() {
   local slot=$1 core0=$2 other=$((1 - $1))
   echo busy > "$OUT/.slot$slot"
   while true; do
      local job
      job=$(next_job)
      [ -z "$job" ] && break
      run_one "$core0" "${job%%|*}" "${job#*|}" run
   done
   echo idle > "$OUT/.slot$slot"
   if [ -n "$BALLAST" ]; then
      local k=0
      while [ "$(cat "$OUT/.slot$other" 2> /dev/null)" = busy ]; do
         k=$((k + 1))
         run_one "$core0" "ballast_s${slot}_$k" "$BALLAST" ballast
      done
   fi
}

echo "run_queue: $QUEUE -> $OUT (pid $$) $(date)"
worker 0 0 &
w0=$!
sleep 2 # stagger the two launches
worker 1 4 &
w1=$!
wait "$w0" "$w1"
echo "run_queue: done $(date)"
