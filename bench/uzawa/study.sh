#!/usr/bin/env bash
# The Schur (Uzawa) vs coupled study of docs/uzawa.md (the proposal's
# Phases 0, 2 and 3: docs/design/uzawa_schur_cg_spec.md s. 7), as
# bench/throughput/bench.py runs on ONE GPU, np 1.
#
#   bench/uzawa/study.sh phase0|phase2|phase3|inner|guess|dt|order|channel|capacity|all
#
# Run it INSIDE an allocation you already hold (AGENTS.md rule 3): it starts
# mpirun and nothing else, and submits no job. Needs build/cuda (the CUDA
# preset) built from this branch and, for `channel`, the frozen channel
# restart of docs/throughput.md ($BENCH_OUT/cases/chan180_t60/chk).
#
# Every run writes the per-step solver log (solver.log_interval 1), so that
# bench.py's result.json carries iterations, the work inside the solve, the
# final coupled residual against the tolerance and the device memory in
# use; bench/uzawa/compare.py reads two of them. Results go to
# $BENCH_OUT/uzawa/<label>/ (BENCH_OUT defaults to ../runs/bench).
#
# Cost (TGV 32^3 Q3/Q2 at ~0.05 s per step after the throughput work, 70
# steps a run): seconds to a minute per run; `phase3` is about 40 runs,
# `capacity` a few full-memory runs at 80^3 and above (minutes each).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BENCH="$REPO/bench/throughput/bench.py"
OUT="${BENCH_OUT:-$REPO/../runs/bench}/uzawa"
N="${N:-32}"               # TGV mesh (N^3 elements)
WARMUP="${WARMUP:-10}"
MEASURE="${MEASURE:-30}"
REPEATS="${REPEATS:-1}"
mkdir -p "$OUT"

# Every run: the solver log, fixed steps (the template's), CUDA, np 1.
run() {
   local label="$1"; shift
   python3 "$BENCH" run --label "$label" --outroot "$OUT" \
      --warmup "$WARMUP" --measure "$MEASURE" --repeats "$REPEATS" \
      --set solver.log_interval=1 "$@"
}
schur() { echo "--set solver.outer=schur"; }
cmp() { python3 "$HERE/compare.py" "$OUT/$1/result.json" "$OUT/$2/result.json"; }

# Phase 0: the coupled baseline, with the phase profile (Gram-Schmidt is its
# own region since this branch: bench.py's gram_schmidt phase).
phase0() {
   run coupled --sizes "$N"
   run coupled-phases --sizes "$N" --profile sync --repeats 1
}

# Phase 2: correctness. Both methods at a tight tolerance from the same
# start: velocity, energy, divergence must agree to the tolerance. Then the
# case defaults (rtol 1e-6, projected guess) for the default comparison.
phase2() {
   run tight-coupled --sizes "$N" --set solver.rtol=1e-10
   run tight-schur --sizes "$N" --set solver.rtol=1e-10 $(schur) \
      --set solver.uzawa.history=true
   cmp tight-coupled tight-schur
   run coupled --sizes "$N"
   run schur --sizes "$N" $(schur) --set solver.uzawa.history=true
   cmp coupled schur
}

# The inner (velocity) tolerance of the Schur operator, the relaxation
# constant, and the legacy Laplacian Schur PC (a fixed linear one).
inner() {
   for mode in tight loose relaxed; do
      run "schur-inner-$mode" --sizes "$N" $(schur) \
         --set solver.uzawa.inner_tolerance=$mode --set solver.uzawa.history=true
   done
   for c in 0.1 10; do
      run "schur-relax-c$c" --sizes "$N" $(schur) --set solver.uzawa.relax_c=$c
   done
   run coupled-legacy --sizes "$N" --set solver.schur=laplacian_legacy
   run schur-legacy --sizes "$N" $(schur) --set solver.schur=laplacian_legacy \
      --set solver.uzawa.history=true
   cmp coupled-legacy schur-legacy
}

# The initial guess: the coupled projection (2k system vectors) against the
# Schur method's pressure-only projection, and the previous solution.
guess() {
   for g in previous project; do
      run "coupled-guess-$g" --sizes "$N" --set solver.initial_guess=$g
      run "schur-guess-$g" --sizes "$N" $(schur) --set solver.initial_guess=$g
      cmp "coupled-guess-$g" "schur-guess-$g"
   done
}

# Iterations against the step: factors of 2 around the template's dt = 0.5/N
# (CFL ~0.29 at t = 0). t_final is set to keep the step count.
dt() {
   local steps=$((WARMUP + MEASURE))
   for f in 0.5 2; do
      local d tf
      d=$(python3 -c "print(0.5 / $N * $f)")
      tf=$(python3 -c "print(0.5 / $N * $f * $steps * (1 + 1e-9))")
      run "coupled-dt$f" --sizes "$N" --set time.dt="$d" --set time.t_final="$tf"
      run "schur-dt$f" --sizes "$N" $(schur) --set time.dt="$d" --set time.t_final="$tf"
      cmp "coupled-dt$f" "schur-dt$f"
   done
}

# Velocity orders 5 and 7 (the proposal's p = 3, 5, 7; the Q3 runs are the
# others above), on coarser meshes that keep the DOFs near the Q3 run's.
order() {
   for o in 5 7; do
      local n=$(( N * 3 / o ))
      for m in coupled schur; do
         local extra=""
         [ "$m" = schur ] && extra="$(schur)"
         run "$m-q$o" --sizes "$n" --set discretization.order_u=$o \
            --set discretization.order_p=$((o - 1)) $extra
      done
      cmp "coupled-q$o" "schur-q$o"
   done
}

# The channel at Re_tau 180 from its frozen restart (walls, OIFS BDF3).
channel() {
   local chk="${BENCH_OUT:-$REPO/../runs/bench}/cases/chan180_t60/chk"
   run chan-coupled --case channel_retau180 --sizes 24 --restart "$chk"
   run chan-schur --case channel_retau180 --sizes 24 --restart "$chk" $(schur)
   cmp chan-coupled chan-schur
}

# Capacity: the largest TGV mesh each method fits on one H100 (coupled: 88^3
# on main). A few steps each; an out-of-memory run shows as FAILED.
capacity() {
   for n in ${CAPACITY_SIZES:-88 92 96 100}; do
      WARMUP=2 MEASURE=3 run "cap-coupled-n$n" --sizes "$n" || true
      WARMUP=2 MEASURE=3 run "cap-schur-n$n" --sizes "$n" $(schur) || true
   done
}

phase3() { inner; guess; dt; order; channel; }

case "${1:-}" in
   phase0|phase2|phase3|inner|guess|dt|order|channel|capacity) "$1" ;;
   all) phase0; phase2; phase3; capacity ;;
   *) sed -n '2,20p' "$0"; exit 2 ;;
esac
