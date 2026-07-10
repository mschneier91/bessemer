#!/bin/sh
# scripts/build.sh [preset]  -- configure + build via spack toolchain.
# Default preset: cpu (the build that matters for Sprint 1).
# NOTE: never `rm -rf build/` -- it dumps the warm ccache. Ask before wiping.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "build.sh: environment setup failed" >&2; exit 1; }

set -e
PRESET=${1:-cpu}
cd "$INCNS_REPO_ROOT"
cmake --preset "$PRESET"
cmake --build --preset "$PRESET"
