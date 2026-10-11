#!/bin/sh
# scripts/machine.sh -- print the name of the machine environment this checkout
# uses: bessemer activates environments/<name>/ (docs/install/spack.md).
#
# In order:
#   1. $INCNS_MACHINE, if set;
#   2. environments/.machine, one line, written by scripts/setup.sh (not
#      committed: it belongs to this checkout on this machine);
#   3. the hostname patterns in environments/<name>/hosts (shell globs, one per
#      line, matched against the full hostname): shared clusters with a
#      committed environment.
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

# The full name where there is one: a short pattern such as "w0*" would also
# match somebody's laptop.
host=$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown)
for hosts in "$REPO"/environments/*/hosts; do
  [ -f "$hosts" ] || continue
  set -f # the patterns are matched against $host, never expanded as file names
  for pat in $(sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "$hosts"); do
    # shellcheck disable=SC2254
    case "$host" in
      $pat) basename "$(dirname "$hosts")"; exit 0 ;;
    esac
  done
  set +f
done

echo "machine.sh: no environment selected for this machine (hostname '$host')." >&2
echo "  Set one up: scripts/setup.sh --plan (docs/install/spack.md)," >&2
echo "  or select an existing one: echo <name> > environments/.machine" >&2
exit 1
