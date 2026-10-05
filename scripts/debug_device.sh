#!/bin/sh
# scripts/debug_device.sh [extra ctest args] -- run the fast tier on MFEM's
# "debug" device to catch silent host<->device fallbacks BEFORE a GPU exists.
#
# The debug device puts every UseDevice vector in the (host-emulated) device
# memory space and mprotects the host mirror, so any un-annotated host access of
# device data FAULTS (SIGSEGV / MFEM MmuError) with a stack trace at the exact
# line -- instead of silently copying, which on a real GPU is a performance bug
# invisible to the normal fast tier. This is the device analog of scripts/asan.sh.
#
# It is a RUNTIME backend (INCNS_DEVICE=debug, honored by ConfigureDevice), so no
# separate build is needed -- it reuses build/cpu. It is much slower (mprotect on
# every access), so this is a diagnostic sweep, not part of the agentic loop.
#
# STATUS (2026-10-05): GREEN -- 74/74 (every fast-tier test at np 1 and 2).
# The last failure was an intermittent page-protection fault from wrapping
# hypre-malloc'd vectors (see CLAUDE.md, "Never hand MFEM device paths a
# hypre-malloc'd buffer"), so a red sweep now means a REGRESSION. Green here is
# still not GPU-ready: the debug device cannot catch a host-compiled forall.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)

# shellcheck disable=SC1091
. "$_script_dir/env.sh" || { echo "debug_device.sh: env setup failed" >&2; exit 1; }

set -e
BUILD_DIR="$INCNS_REPO_ROOT/build/cpu"
if [ ! -d "$BUILD_DIR" ]; then
  echo "debug_device.sh: no build at $BUILD_DIR -- run scripts/build.sh first" >&2
  exit 1
fi

cd "$BUILD_DIR"
# np in {1, 2}: partition paths are exercised at 2; the mprotect overhead makes
# a wider sweep needlessly slow for a fallback hunt.
INCNS_DEVICE=debug ctest --output-on-failure -L fast -E "_np4$" "$@"
