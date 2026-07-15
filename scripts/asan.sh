#!/bin/sh
# scripts/asan.sh [extra ctest args] -- build the C++ fast tier under
# AddressSanitizer + UndefinedBehaviorSanitizer and run it at np in {1, 2}
# (the H3 sanitizer gate). This is a diagnostic sweep, not part of the normal
# loop: it is slower and rebuilds into build/cpu-asan (a separate tree, so the
# warm cpu ccache/build is untouched).
#
# Python is deliberately NOT built here -- ASan against a non-instrumented
# Python interpreter needs LD_PRELOAD of the ASan runtime and is out of scope;
# this covers the compiled C++ suite only.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "asan.sh: environment setup failed" >&2; exit 1; }

set -e
cd "$INCNS_REPO_ROOT"
cmake --preset cpu-asan
cmake --build --preset cpu-asan

# Leak detection is off: MFEM / HYPRE / MPI are not instrumented and hold
# allocations to exit, which would drown out real findings. We are hunting
# heap-overflow / use-after-free / UB, and -fno-sanitize-recover already makes
# any UB a hard failure.
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:detect_stack_use_after_return=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

cd "$INCNS_REPO_ROOT/build/cpu-asan"
# np in {1, 2}: np=2 already exercises the partition-boundary / reduction paths;
# np=4 adds runtime without new sanitizer coverage for this sweep.
ctest --output-on-failure -L fast -E "_np4$" "$@"
