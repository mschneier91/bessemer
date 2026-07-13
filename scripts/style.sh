#!/bin/sh
# scripts/style.sh [format|check]  -- astyle formatting (MFEM convention).
#   format (default) : reformat sources in place
#   check            : fail if any source is not already formatted (used by the hook)
#
# astyle is version-sensitive: a different version silently reformats the whole
# tree. We pin to the version in the desktop env lockfile and refuse to run on a
# mismatch rather than produce a misleading diff.

INCNS_ASTYLE_VERSION=3.4.11

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "style.sh: environment setup failed" >&2; exit 1; }

set -e
MODE=${1:-format}
RC="$INCNS_REPO_ROOT/config/style.astylerc"

_have=$(astyle --version 2>&1 | awk '{print $NF}')
if [ "$_have" != "$INCNS_ASTYLE_VERSION" ]; then
  echo "style.sh: astyle $_have != pinned $INCNS_ASTYLE_VERSION -- refusing to run." >&2
  echo "          (wrong astyle silently reformats the tree; fix the env, do not commit.)" >&2
  exit 1
fi

# Collect tracked C++ sources under the code directories.
_files=$(find "$INCNS_REPO_ROOT/src" "$INCNS_REPO_ROOT/apps" "$INCNS_REPO_ROOT/test" \
           "$INCNS_REPO_ROOT/python" \
           \( -name '*.hpp' -o -name '*.cpp' \) 2>/dev/null || true)
if [ -z "$_files" ]; then
  echo "style.sh: no C++ sources to format yet."
  exit 0
fi

case "$MODE" in
  format)
    # shellcheck disable=SC2086
    astyle --options="$RC" --suffix=none $_files
    ;;
  check)
    # --dry-run prints "Formatted  <file>" for files that WOULD change.
    # shellcheck disable=SC2086
    _out=$(astyle --options="$RC" --dry-run $_files)
    if echo "$_out" | grep -q '^Formatted'; then
      echo "style.sh: the following files are not astyle-clean:" >&2
      echo "$_out" | grep '^Formatted' >&2
      echo "Run: scripts/style.sh format" >&2
      exit 1
    fi
    echo "style.sh: all sources are astyle-clean."
    ;;
  *)
    echo "style.sh: unknown mode '$MODE' (use format|check)" >&2
    exit 1
    ;;
esac
