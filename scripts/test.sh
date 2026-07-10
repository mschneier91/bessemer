#!/bin/sh
# scripts/test.sh [ctest args...]  -- run the test suite via ctest.
# Tests self-register at np in {1, 2, 4} (see test/CMakeLists.txt), so a plain
# ctest run exercises all rank counts. Pass e.g. `-L fast` to select a label.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "test.sh: environment setup failed" >&2; exit 1; }

set -e
PRESET=cpu
BUILD_DIR="$INCNS_REPO_ROOT/build/$PRESET"
if [ ! -d "$BUILD_DIR" ]; then
  echo "test.sh: no build at $BUILD_DIR -- run scripts/build.sh first" >&2
  exit 1
fi
cd "$BUILD_DIR"
ctest --output-on-failure "$@"
