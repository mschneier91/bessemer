#!/bin/sh
# scripts/time_mms3d.sh -- build + TIME the 3D unsteady-Stokes MMS driver
# (apps/unsteady_mms_3d.cpp) CPU vs GPU, sweeping resolution. This is STEP 4 of
# the GPU bring-up: the first CPU-vs-GPU performance measurement of the port.
#
# RUN INSIDE A GPU COMPUTE ALLOCATION, under tmux -- never on a login node, and
# never as a bare foreground command that dies with a dropped SSH session.
#
# It sources scripts/env.sh (REQUIRED: without the env active, a cmake auto-regen
# can wipe build/cuda's cache -- see landmine 7 in the GPU plan) and then builds
# ONE binary, build/cuda/apps/unsteady_mms_3d, running it with both `-d cpu` and
# `-d cuda`. A separate build/cpu is deliberately NOT required: the only
# nvcc-compiled TU is the grad-div integrator, which this MMS never turns on
# (grad_div defaults to 0), so the `-d cpu` path is the same host code a g++
# build would run. Set INCNS_CPU_BUILD=1 to force a real build/cpu for the CPU
# runs instead.
#
# The MMS reproduces its exact solution at every resolution, so ||u-u_exact||
# stays at ~solver tolerance as -n grows -- it is a correctness check, not the
# thing being measured. The wall time is the number that matters; scale -n up
# (INCNS_NS) until the GPU wins (small 8^3-class problems are too little work).
#
# Overrides (all optional):
#   INCNS_NS         -n sweep, space-separated       (default "16 24 32")
#   INCNS_DEVICES    backends to time                (default "cpu cuda")
#   INCNS_NP         MPI ranks                       (default 1)
#   INCNS_PREC       velocity preconditioner         (default amg)
#   INCNS_OU         velocity order k_u              (default 3)
#   INCNS_DT         time step                       (default 0.02)
#   INCNS_TF         end time                        (default 0.2)
#   INCNS_CPU_BUILD  if set, use build/cpu for the -d cpu runs
#
# Everything is teed to build/time_mms3d.<timestamp>.log (build/ is gitignored)
# so the full run can be read back later.

_script_dir=$(cd "$(dirname "$0")" && pwd)
export INCNS_REPO_ROOT=$(cd "$_script_dir/.." && pwd)
mkdir -p "$INCNS_REPO_ROOT/build"
LOG="$INCNS_REPO_ROOT/build/time_mms3d.$(date +%Y%m%d-%H%M%S).log"

run() {
  # shellcheck disable=SC1091
  . "$_script_dir/env.sh" || { echo "time_mms3d.sh: environment setup failed" >&2; return 1; }
  set -e

  NS=${INCNS_NS:-"16 24 32"}
  DEVICES=${INCNS_DEVICES:-"cpu cuda"}
  NP=${INCNS_NP:-1}
  PREC=${INCNS_PREC:-amg}
  OU=${INCNS_OU:-3}
  DT=${INCNS_DT:-0.02}
  TF=${INCNS_TF:-0.2}

  CUDA_DIR="$INCNS_REPO_ROOT/build/cuda"
  echo "== time_mms3d :: $(date) :: $(hostname) =="
  echo "NS='$NS' DEVICES='$DEVICES' NP=$NP PREC=$PREC OU=$OU DT=$DT TF=$TF"

  # --- build (incremental; NEVER --preset / build.sh -- they re-stamp ccache) --
  if [ ! -d "$CUDA_DIR" ]; then
    echo "time_mms3d.sh: no build/cuda -- reconfigure with the plan's recovery recipe first" >&2
    return 1
  fi
  echo "-- building unsteady_mms_3d (build/cuda) --"
  cmake --build "$CUDA_DIR" --target unsteady_mms_3d -j16

  CUDA_BIN="$CUDA_DIR/apps/unsteady_mms_3d"
  CPU_BIN="$CUDA_BIN"
  if [ -n "${INCNS_CPU_BUILD:-}" ]; then
    CPU_DIR="$INCNS_REPO_ROOT/build/cpu"
    [ -d "$CPU_DIR" ] || { echo "time_mms3d.sh: INCNS_CPU_BUILD set but no build/cpu" >&2; return 1; }
    echo "-- building unsteady_mms_3d (build/cpu) --"
    cmake --build "$CPU_DIR" --target unsteady_mms_3d -j16
    CPU_BIN="$CPU_DIR/apps/unsteady_mms_3d"
  fi

  # --- sweep -----------------------------------------------------------------
  echo
  printf '%-6s %-5s %-13s %-14s\n' device n wall_s u_err
  echo "----------------------------------------------------"
  for d in $DEVICES; do
    bin=$CUDA_BIN
    [ "$d" = cpu ] && bin=$CPU_BIN
    for n in $NS; do
      out=$(mpirun -n "$NP" "$bin" -d "$d" -n "$n" -ou "$OU" \
                   -dt "$DT" -tf "$TF" -prec "$PREC" 2>&1) || {
        echo ">> run FAILED: device=$d n=$n"; echo "$out"; continue; }
      wall=$(printf '%s\n' "$out" | sed -n 's/.*run_wall_s=\([0-9.eE+-]*\).*/\1/p')
      uerr=$(printf '%s\n' "$out" | sed -n 's/.*||u - u_exact||_L2=\([0-9.eE+-]*\).*/\1/p')
      printf '%-6s %-5s %-13s %-14s\n' "$d" "$n" "${wall:-?}" "${uerr:-?}"
    done
  done
}

run 2>&1 | tee "$LOG"
echo "== full log: $LOG =="
