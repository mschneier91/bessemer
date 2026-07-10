#!/bin/sh
# scripts/docs.sh  -- generate the libincns API docs with Doxygen.
# Output: build/docs/html/index.html (git-ignored). Sources env.sh so a
# spack-provided doxygen is picked up when present.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "docs.sh: environment setup failed" >&2; exit 1; }

if ! command -v doxygen >/dev/null 2>&1; then
  echo "docs.sh: doxygen not found on PATH." >&2
  echo "  Install it, then re-run. Options:" >&2
  echo "    * spack (project-consistent): add 'doxygen' to" >&2
  echo "      environments/${INCNS_MACHINE}/spack.yaml, re-concretize, and spack install." >&2
  echo "    * system package, e.g. 'apt install doxygen' (graphviz for diagrams)." >&2
  exit 1
fi

set -e
cd "$INCNS_REPO_ROOT"
doxygen Doxyfile
echo "docs.sh: HTML written to build/docs/html/index.html"
