#!/usr/bin/env bash
# Build + run the 2D TGV NSE oracle (Sprint 2.2), teeing everything to a log so
# it can be read back after the fact (see the relay-via-script-file workflow).
#
# Run this INSIDE a GPU allocation, never on a login node.
#
#   ./scripts/run_tgv_oracle.sh          # np1 first, then np2 (the real check)
#   INCNS_NP="1"      ./scripts/run_tgv_oracle.sh
#   INCNS_NP="1 2 4"  ./scripts/run_tgv_oracle.sh   # needs >=1 GPU per rank
set -u -o pipefail

cd "$(dirname "$0")/.."
LOG="build/tgv_oracle.log"

# env.sh BEFORE any cmake command -- a cmake run without it wipes the cache.
. scripts/env.sh

: "${INCNS_PRESET:=cuda}"
: "${INCNS_DEVICE:=cuda}"
: "${INCNS_NP:=1 2}"
export INCNS_PRESET INCNS_DEVICE

mkdir -p build
{
  echo "=== $(date -Is)  host=$(hostname)  preset=$INCNS_PRESET device=$INCNS_DEVICE ==="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "(no nvidia-smi)"

  echo "=== incremental build (never --preset: that would reconfigure) ==="
  cmake --build "build/${INCNS_PRESET}" --target tgv_nse_test -j16 || exit 1

  for np in $INCNS_NP; do
    echo "=== tgv_nse_test np${np} ==="
    mpirun -n "$np" "build/${INCNS_PRESET}/test/tgv_nse_test" || echo "FAILED np${np}"
  done
  echo "=== done $(date -Is) ==="
} 2>&1 | tee "$LOG"

echo
echo "Log written to $LOG"
