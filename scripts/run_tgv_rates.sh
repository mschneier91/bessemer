#!/usr/bin/env bash
# Build + run the rewritten tgv_nse_test (rate-based) and tee everything to a
# log Claude can read back. See [[feedback-relay-via-script-file]].
#
# Usage, INSIDE a GPU allocation (never a login node):
#   ./scripts/run_tgv_rates.sh          # np1 only, the fast iteration
#   INCNS_NP="1 2 4" ./scripts/run_tgv_rates.sh
#
# What to look for in the log: the "[ TGV NSE ]" lines. Every accuracy
# assertion in this test is a CONVERGENCE RATE, so the numbers that matter are
# the rate lines, not the error magnitudes:
#   spatial rates (u: r1 r2 | p: r1 r2)   -> p ~3; u is an OPEN ISSUE, see below
#   convection-off velocity rate          -> expect ~4 (clean, flat)
#   temporal u (e1, e2, e3, r1, r2)       -> expect r ~2
#   temporal p (e1, e2, e3, r1, r2)       -> expect r ~2 (asserted >1.5)
#   energy rates (ke, eps, div)           -> expect ~4, ~3, ~3
#
# Taylor-Hood Q3/Q2: velocity h^4, pressure h^3 -- the VELOCITY IS ONE ORDER
# BETTER. Both O(dt^2) in time. Convection OFF measures the h^4 cleanly.
#
# OPEN: with convection ON the velocity rate is still climbing at n=16
# (2.39/3.00/3.41) and its error nearly equals the PRESSURE error, which correct
# Taylor-Hood should not do. Either pre-asymptotic or pressure error leaking
# into the velocity through the coupling. Extending the ladder past n=16
# decides it -- not yet run. See the OPEN ISSUE block in test/tgv_nse_test.cpp.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
OUT=$REPO/build/tgv_rates
mkdir -p "$OUT"
LOG=$OUT/report.txt

# shellcheck disable=SC1091
. scripts/env.sh   # MANDATORY before any cmake -- see landmine 7

{
  echo "date  $(date -Is)"
  echo "node  $(hostname -f)"
  echo "gpus  ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git   $(git rev-parse --short HEAD) on $(git branch --show-current)"
  echo "=== build ==="
} > "$LOG"

# Incremental build of just this target -- never --preset (wipes the cache).
if cmake --build build/cuda --target tgv_nse_test -j16 >> "$LOG" 2>&1; then
  echo "BUILD ok" >> "$LOG"
else
  echo "BUILD FAILED -- see above" >> "$LOG"
  tail -40 "$LOG"; exit 1
fi

for np in ${INCNS_NP:-1}; do
  echo "=== tgv_nse_test np=$np ===" >> "$LOG"
  RUN=$OUT/tgv_nse_test_np${np}.log
  INCNS_DEVICE=cuda mpirun -n "$np" build/cuda/test/tgv_nse_test > "$RUN" 2>&1
  rc=$?
  ran=$(grep -c '^\[ RUN' "$RUN" 2>/dev/null || echo 0)
  passed=$(grep -c '^\[       OK \]' "$RUN" 2>/dev/null || echo 0)
  failed=$(grep -c '^\[  FAILED  \].*ms)' "$RUN" 2>/dev/null || echo 0)
  verdict=PASS; [ "$rc" -ne 0 ] && verdict=FAIL
  echo "RESULT tgv_nse_test np=$np rc=$rc verdict=$verdict ran=$ran passed=$passed failed=$failed" >> "$LOG"
  # The measured numbers behind every assertion -- the point of the run.
  grep '^\[ TGV NSE' "$RUN" | sed 's/^/  /' >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "  --- failure excerpt ---" >> "$LOG"
    grep -E 'Failure$|^\[  FAILED  \]|expected|rate ' "$RUN" | head -30 | sed 's/^/  /' >> "$LOG"
  fi
done

echo "=== end $(date -Is) ===" >> "$LOG"
./scripts/reap.sh >/dev/null 2>&1 || true
cat "$LOG"
echo
echo "full log: $LOG"
