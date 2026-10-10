#!/bin/sh
# scripts/machine.sh -- print the name of the machine environment this checkout
# uses: bessemer activates environments/<name>/ (docs/install/spack.md).
#
# In order:
#   1. $INCNS_MACHINE, if set;
#   2. environments/.machine, one line, written by scripts/setup.sh (not
#      committed: it belongs to this checkout on this machine);
#   3. hostname patterns of shared clusters with a committed environment.
# Exits 1 with a hint when none applies (a machine that hasn't been set up).

REPO=${INCNS_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}

if [ -n "${INCNS_MACHINE:-}" ]; then
  echo "$INCNS_MACHINE"
  exit 0
fi

if [ -s "$REPO/environments/.machine" ]; then
  name=$(sed -n '1{s/[[:space:]]//g;p;}' "$REPO/environments/.machine")
  if [ -n "$name" ]; then
    echo "$name"
    exit 0
  fi
fi

host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)
case "$host" in
  # Bridges-2: br0xx login nodes + w0xx H100 nodes. psc_gpu is pinned to
  # cuda_arch=90 (H100 only, see docs/install/bridges2.md) -- v0xx (V100) and
  # gl0xx (L40S) are deliberately NOT matched, since a cuda_arch=90 binary is
  # the wrong SM arch on them.
  br0* | w0* ) echo psc_gpu; exit 0 ;;
esac

echo "machine.sh: no environment selected for this machine (hostname '$host')." >&2
echo "  Set one up: scripts/setup.sh --plan (docs/install/spack.md)," >&2
echo "  or select an existing one: echo <name> > environments/.machine" >&2
exit 1
