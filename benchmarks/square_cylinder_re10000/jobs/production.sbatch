#!/bin/bash
#SBATCH -J bessemer-re10k
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00
#SBATCH -o re10k-%x-%j.out

# Phase C (docs/design/re10000_plan.md): one Re 10 000 level, continued from its
# checkpoint by every further job. Usage, from the repository root:
#
#   sbatch [-p RM-shared -n 64] benchmarks/square_cylinder_re10000/jobs/production.sbatch l1eq
#
# - The rank count is the job's (SLURM_NTASKS) and must stay the same for every
#   job of a level: a checkpoint restarts only at the rank count that wrote it
#   (checked here before the run). Pick it from the probe (jobs/probe.sbatch).
# - Each job continues from the level's rolling checkpoint (every 1,000 steps;
#   a job killed at its time limit loses at most that many steps) and exits at
#   once if the level is already done (summary status ok). It never submits
#   anything itself: a chain of jobs is submitted, and approved, as a whole
#   (benchmarks/square_cylinder_re10000/README.md, "Production").
# - EXTEND_T_FINAL=<t>: continue a finished level to a later t_final (more
#   averaging time when the error bar is too large).
#
# Cost: ranks x the time used. l1eq ~1.5-2 k core-hours in total, l2eq ~6-11 k
# (extrapolated; the probe measures it). An agent asks the human first, for the
# whole chain with its total cost (AGENTS.md rule 3).
#
# Overrides (and testing on a workstation without Slurm): LEVEL_DECK=<deck file>
# instead of benchmarks/square_cylinder_re10000/<level>.yaml; NP=<ranks>.

JOBS_DIR=${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/benchmarks/square_cylinder_re10000/jobs}
[ -f "$JOBS_DIR/common.sh" ] || JOBS_DIR=$(cd "$(dirname "$0")" && pwd)
. "$JOBS_DIR/common.sh" || exit 1

level=${1:?usage: production.sbatch <level: l1eq | l2eq>}
deck=${LEVEL_DECK:-$BENCH/$level.yaml}
[ -f "$deck" ] || { echo "FATAL: no deck $deck"; exit 1; }
np=${NP:-${SLURM_NTASKS:?no SLURM_NTASKS: set NP}}
dir=$RUN_ROOT/$level
mkdir -p "$dir"
cd "$dir" || exit 1

# The deck's own output and checkpoint names (relative to this directory).
out=$(sed -n 's/^  path: \(sqcyl[^ ]*\).*/\1/p' "$deck" | head -1)
[ -n "$out" ] || out=$(awk '/^output:/{f=1} f&&/^  path:/{print $2; exit}' "$deck")
chk=$(awk '/^checkpoint:/{f=1} f&&/^  path:/{print $2; exit}' "$deck")
summary=$dir/$out/$(awk '/^output:/{f=1} f&&/^  name:/{print $2; exit}' "$deck")_summary.json

status=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$summary" 2>/dev/null)
if [ "$status" = ok ] && [ -z "${EXTEND_T_FINAL:-}" ]; then
  echo "== $level is done (summary status ok): nothing to run"
  exit 0
fi

# The deck this job runs: the level's, plus a restart and a later t_final.
run_deck=$dir/run_${SLURM_JOB_ID:-local}.yaml
cp "$deck" "$run_deck"
if [ -n "$chk" ] && [ -f "$dir/$chk/meta.yaml" ]; then
  chk_np=$(awk '/^np:/{print $2; exit}' "$dir/$chk/meta.yaml")
  if [ "$chk_np" != "$np" ]; then
    echo "FATAL: the checkpoint $dir/$chk was written at $chk_np ranks, this job has $np:"
    echo "       a checkpoint restarts only at its own rank count. Submit with -n $chk_np."
    exit 1
  fi
  printf '\nrestart: %s\n' "$dir/$chk" >> "$run_deck"
  echo "== continuing $level from $dir/$chk ($(awk '/^t:/{print "t = " $2; exit}' "$dir/$chk/meta.yaml"))"
else
  echo "== starting $level"
fi
if [ -n "${EXTEND_T_FINAL:-}" ]; then
  sed -i "s/^  t_final: .*/  t_final: $EXTEND_T_FINAL/" "$run_deck"
  echo "== t_final -> $EXTEND_T_FINAL"
fi

mpirun -np "$np" "$RUN_CASE" "$run_deck" >> "$dir/$level.log" 2>&1
rc=$?
status=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$summary" 2>/dev/null)
echo "== run_case exit $rc, summary status ${status:-?}; log $dir/$level.log"
tail -n 2 "$dir/$level.log"
if [ "$status" = ok ]; then
  echo "== analyse: python3 $BENCH/../square_cylinder_re1000/analyze.py" \
       "$level=$dir/$out/$(basename "$summary" _summary.json)_history.csv"
fi
exit $rc
