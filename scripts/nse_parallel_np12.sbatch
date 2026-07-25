#!/bin/bash
#SBATCH -J incns-nse-par2
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -n 2
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o nse-par2-%j.out
#SBATCH -e nse-par2-%j.out
#
# scripts/nse_parallel_np12.sbatch -- the NSE path at np in {1,2}.
#
# >>> THE 2-GPU VARIANT of scripts/nse_parallel.sbatch. <<<
# Same three tests, same report format, but it stops at np2 so it needs only
# TWO cards. It exists because a 4-GPU-on-one-node request can sit behind a
# deep queue: measured 2026-07-24, all 80 h100 cards were allocated except two,
# with 136 h100 jobs pending. Two cards free up much sooner than four.
#
# np2 is where a shared-face assembly bug would FIRST appear (np1 cannot see a
# rank boundary at all), so this buys most of the parallel signal early.
# nse_parallel.sbatch still owns np4 -- run BOTH; they use different job names
# and write to DIFFERENT output dirs, so neither clobbers the other:
#   this one -> build/nse_parallel_np12/     nse_parallel.sbatch -> build/nse_parallel/
#
# -n 2 -c 8 = 16 cores: 2 MPI ranks x 8 cores. An h100-80 node is 104 cores /
# 8 GPUs (~13 per GPU), so 16 for 2 GPUs stays inside the proportionate share
# while leaving the build's -j16 real cores. Requested explicitly because
# GPU-shared shares the NODE -- the default core count is not what a 2-rank
# job wants.
#
# GPU-shared, not GPU: the `GPU` partition on Bridges-2 is WHOLE-NODE ONLY and
# an h100 node has 8 cards, so a partial --gres request there is rejected
# ("must request all resources, gpu count N is low"). GPU-shared runs on the
# SAME h100-80:8 nodes and hands out N DISTINCT cards -- it shares the node,
# not the GPUs -- which is what one-rank-per-card needs.
#
# SUBMIT (a human submits batch jobs -- CLAUDE.md guardrail):
#   cd /ocean/projects/mth260019p/schneier/bessemer
#   sbatch scripts/nse_parallel_np12.sbatch
#
# Everything lands in build/nse_parallel_np12/ as one machine-checkable report
# plus per-run logs; the summary greps the RESULT lines at the end.

set -u

echo "=== job $SLURM_JOB_ID  node=$(hostname)  $(date -Is) ==="

cd "$SLURM_SUBMIT_DIR" || exit 1

OUT="build/nse_parallel_np12"
mkdir -p "$OUT"
REPORT="$OUT/report.txt"
: > "$REPORT"

# --- env: REQUIRED before any cmake command (a bare cmake wipes the cache) ---
. scripts/env.sh || { echo "FATAL: env.sh failed"; exit 1; }

export INCNS_PRESET=cuda
export INCNS_DEVICE=cuda

{
  echo "job_id           $SLURM_JOB_ID"
  echo "node             $(hostname)"
  echo "date             $(date -Is)"
  echo "gpus_allocated   ${SLURM_GPUS_ON_NODE:-unknown} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "machine          ${INCNS_MACHINE:-unknown}"
  echo "mfem             ${MFEM_DIR:-unknown}"
} >> "$REPORT"

echo "--- nvidia-smi ---" | tee -a "$REPORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tee -a "$REPORT"

# --- build (incremental; never --preset, which would reconfigure) ------------
echo "=== build ===" | tee -a "$REPORT"
BUILD_LOG="$OUT/build.log"
if cmake --build build/cuda --target convection_test nse_mms_test tgv_nse_test \
     -j16 > "$BUILD_LOG" 2>&1; then
  echo "BUILD ok" | tee -a "$REPORT"
else
  echo "BUILD FAILED -- see $BUILD_LOG" | tee -a "$REPORT"
  echo "--- last 40 lines ---" | tee -a "$REPORT"
  tail -40 "$BUILD_LOG" | tee -a "$REPORT"
  exit 1
fi

# --- the sweep ---------------------------------------------------------------
# Each (test, np) pair gets its own log. RESULT lines are the machine-readable
# summary: one per run, greppable, with the gtest tallies parsed out.
run_one() {
  exe=$1; np=$2
  log="$OUT/${exe}_np${np}.log"
  echo "=== ${exe} np${np} ===" | tee -a "$REPORT"

  start=$(date +%s)
  mpirun -n "$np" "build/cuda/test/${exe}" > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  wall=$((end - start))

  # gtest prints "[==========] N tests from M test suites ran." and
  # "[  PASSED  ] N tests." / "[  FAILED  ] N tests, listed below:"
  # NB the banner is "[==========] Running N tests from M test suites." -- the
  # "Running" is load-bearing here; without it the [0-9]+ matches nothing and
  # every run silently reports ran=0. (Checked against captured gtest output.)
  ran=$(grep -oE '^\[=+\] Running [0-9]+ test' "$log" | tail -1 | grep -oE '[0-9]+' | head -1)
  passed=$(grep -oE '^\[  PASSED  \] [0-9]+ test' "$log" | tail -1 | grep -oE '[0-9]+' | head -1)
  failed=$(grep -oE '^\[  FAILED  \] [0-9]+ test' "$log" | tail -1 | grep -oE '[0-9]+' | head -1)
  : "${ran:=0}"; : "${passed:=0}"; : "${failed:=0}"

  if [ "$rc" -eq 0 ]; then verdict=PASS; else verdict=FAIL; fi
  echo "RESULT ${exe} np=${np} rc=${rc} verdict=${verdict} ran=${ran} passed=${passed} failed=${failed} wall_s=${wall}" \
    | tee -a "$REPORT"

  # Surface the numbers the tests print unconditionally, so the physics is
  # checkable from the report alone without opening each log.
  grep -E '^\[ (NSE MMS|TGV NSE) ' "$log" | sed 's/^/  /' | tee -a "$REPORT"

  if [ "$rc" -ne 0 ]; then
    echo "  --- failure excerpt ---" | tee -a "$REPORT"
    grep -E '^\[  FAILED  \]|Failure|error:|Assertion|CUDA error|out of memory|MPI_ABORT|Signal' \
      "$log" | head -25 | sed 's/^/  /' | tee -a "$REPORT"
  fi
}

# convection_test  -- the 2.1 operator itself; shared-face assembly lives here,
#                     so this is where a parallel assembly bug surfaces first.
# nse_mms_test     -- 2D+3D temporal order; the 2.2 sign-off evidence, now in
#                     parallel. Rates must still be ~2 at every rank count.
# tgv_nse_test     -- the unforced TGV oracle + analytic energy harness.
for np in 1 2; do
  for exe in convection_test nse_mms_test tgv_nse_test; do
    run_one "$exe" "$np"
  done
done

# --- summary -----------------------------------------------------------------
{
  echo
  echo "=== SUMMARY ==="
  grep '^RESULT ' "$REPORT" || echo "(no RESULT lines -- the sweep did not run)"
  echo
  # NB `grep -c` PRINTS 0 and EXITS 1 on no-match, so a `|| echo 0` fallback
  # appends a SECOND zero and yields the string "0\n0" -- which fails `-eq` and
  # would report RED on an all-green run. Count with grep|wc instead, which
  # always emits exactly one number. (This bug was caught by fixture-testing the
  # summary logic; do not "simplify" it back.)
  n_fail=$(grep 'verdict=FAIL' "$REPORT" 2>/dev/null | wc -l | tr -d ' ')
  n_pass=$(grep 'verdict=PASS' "$REPORT" 2>/dev/null | wc -l | tr -d ' ')
  echo "TOTALS pass_runs=${n_pass} fail_runs=${n_fail}"
  if [ "$n_fail" -eq 0 ] && [ "$n_pass" -gt 0 ]; then
    echo "OVERALL GREEN"
  else
    echo "OVERALL RED"
  fi
  echo "=== end $(date -Is) ==="
} | tee -a "$REPORT"

# Orphaned MPI/test processes after any interrupted run (CLAUDE.md).
./scripts/reap.sh >/dev/null 2>&1 || true

echo
echo "Report:    $SLURM_SUBMIT_DIR/$REPORT"
echo "Per-run:   $SLURM_SUBMIT_DIR/$OUT/<test>_np<N>.log"
