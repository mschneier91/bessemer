#!/bin/bash
# bench/dfg3/make_sweep.sh "IMEX_ARGS" "ROT_ARGS" > sweep.q
#
# The DFG 2D-3 work-precision sweep queue (run_queue.sh format): both schemes
# on mesh levels M1 (CFL targets 0.3 .. 1.5) and M0 (0.1 .. 1.5), BDF2/EXT3,
# grad-div gamma = h_K, Dirichlet outflow (the inflow profile), outer FGMRES to
# 1e-8. Longest runs first, each IMEX/rotational pair adjacent so the two
# usually run side by side. IMEX_ARGS / ROT_ARGS: the solver configuration
# chosen by the calibration (e.g. "-conv -apc jacobi_pcg", "-rot -pbj -schur cc").
set -u
IMEX=$1
ROT=$2
COMMON="-gd 1 -dout -rtol 1e-8 -ext 3"
for level in 1 0; do
   if [ "$level" = 1 ]; then targets="0.3 0.5 0.7 0.9 1.2 1.5"
   else targets="0.1 0.2 0.3 0.5 0.7 0.9 1.2 1.5"; fi
   for c in $targets; do
      echo "m${level}_imex_c$c|$IMEX $COMMON -mref $level -cflt $c"
      echo "m${level}_rot_c$c|$ROT $COMMON -mref $level -cflt $c"
   done
done
