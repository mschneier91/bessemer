#!/bin/sh
# scripts/test.sh [ctest args...]  -- run the test suite via ctest.
# Tests self-register at np in {1, 2, 4} (see test/CMakeLists.txt), so a plain
# ctest run exercises all rank counts. Pass e.g. `-L fast` to select a label.
#
# Tiers: `-L dev` is the seconds-long inner-loop pre-filter (np=1 only);
# `-L fast` is the real gate and is what "green" means. See test/CMakeLists.txt.
#
# Overrides:
#   INCNS_PRESET  -- which build/<preset> tree to test (default: cpu)
#   INCNS_JOBS    -- ctest parallelism (default 1 = serial, as before).
#                    Passed as `ctest -j N`. ctest schedules by each test's
#                    PROCESSORS property (= its MPI rank count), so N is a
#                    budget of RANKS in flight, not tests: at N=8 the np=1
#                    tests pack 8-wide while an np=4 test still reserves 4.
#                    CAUTION on GPU: concurrent ranks share the card, so a large
#                    N can OOM an 80 GB H100 the same way np=4 on 1 GPU does.
#                    Start at 4-8 for `-L dev` on one GPU; leave it at 1 for a
#                    timing run, where contention would corrupt the numbers.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "test.sh: environment setup failed" >&2; exit 1; }

set -e
PRESET=${INCNS_PRESET:-cpu}
BUILD_DIR="$INCNS_REPO_ROOT/build/$PRESET"
if [ ! -d "$BUILD_DIR" ]; then
  echo "test.sh: no build at $BUILD_DIR -- run scripts/build.sh first" >&2
  exit 1
fi
cd "$BUILD_DIR"
JOBS=${INCNS_JOBS:-1}
if [ "$JOBS" -gt 1 ] 2>/dev/null; then
  # -j alone is the rank budget: ctest sums each test's PROCESSORS property
  # across running tests and will not exceed N.
  #
  # Deliberately NOT --test-load: that throttles on the system LOAD AVERAGE,
  # not on ranks. One MPI+hypre+CUDA test pushes load past any sane N on a
  # 12-core allocation, so ctest then blocks every further launch and the run
  # degrades to strictly serial -- and because load average is a decaying
  # 1-minute mean, it keeps blocking long after the test that caused it ends.
  ctest --output-on-failure -j "$JOBS" "$@"
else
  ctest --output-on-failure "$@"
fi
