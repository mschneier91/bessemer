#!/bin/sh
# scripts/dev.sh [ctest args...] -- the INNER-LOOP check: build, then run the
# `dev` tier (every fast-tier test, np=1 only) in parallel. Seconds, not minutes.
#
# WHAT THIS IS NOT: a definition of green. CLAUDE.md requires np in {2, 4} to
# pass before anything counts as green, because np=1 cannot see partition-
# boundary or reduction bugs. Use this while editing; run the real gate
#
#   INCNS_DEVICE=cuda INCNS_PRESET=cuda ./scripts/test.sh -L 'smoke|fast'
#
# before committing (and note it wants >= 1 GPU per rank -- on a 1-GPU
# allocation the np=4 tests will OOM, which is an allocation shape issue, not a
# code failure).
#
# Overrides:
#   INCNS_PRESET  build tree to use          (default cuda)
#   INCNS_DEVICE  MFEM backend               (default cuda)
#   INCNS_JOBS    parallel rank budget       (default 8; see test.sh)
#   INCNS_NO_BUILD  set to skip the rebuild
#
# Extra args pass through to ctest, so a targeted loop is just:
#   ./scripts/dev.sh -R stokes_solver

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# Assign INTO the exported names -- `PRESET=${INCNS_PRESET:-cuda}` followed by a
# bare `export INCNS_PRESET` would export an UNSET variable, and test.sh would
# fall back to its own `cpu` default and look for build/cpu.
export INCNS_PRESET=${INCNS_PRESET:-cuda}
export INCNS_DEVICE=${INCNS_DEVICE:-cuda}
export INCNS_JOBS=${INCNS_JOBS:-8}

BUILD_DIR="$INCNS_REPO_ROOT/build/$INCNS_PRESET"
[ -d "$BUILD_DIR" ] || {
  echo "dev.sh: no build at $BUILD_DIR" >&2; exit 1; }

if [ -z "${INCNS_NO_BUILD:-}" ]; then
  # shellcheck disable=SC1091
  . "$_script_dir/env.sh" || {
    echo "dev.sh: environment setup failed" >&2; exit 1; }
  # Incremental, never --preset (re-stamps the ccache launcher; see the GPU plan).
  cmake --build "$BUILD_DIR" -j16 || {
    echo "dev.sh: BUILD FAILED -- tests skipped" >&2; exit 1; }
fi

exec "$_script_dir/test.sh" -L dev "$@"
