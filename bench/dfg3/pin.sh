#!/bin/bash
# pin.sh FIRST_CORE cmd... : bind this MPI rank to core FIRST_CORE + local rank.
core=$(( $1 + ${OMPI_COMM_WORLD_LOCAL_RANK:-0} ))
shift
exec taskset -c "$core" "$@"
