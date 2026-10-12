#!/bin/bash
#SBATCH -J bessemer-build-test
#SBATCH -p RM-shared
#SBATCH -n 16
#SBATCH -t 1:00:00
#SBATCH -o build-test-%j.out

# Phase B1 (docs/design/re10000_plan.md): build the cpu preset with psc_cpu on an
# RM node and run the fast tier. Charges 16 cores x the time used (about 30-45
# min: ~10-12 core-hours).
#
#   cd <repo> && sbatch benchmarks/square_cylinder_re10000/jobs/build_test.sbatch
#
# Submitting spends the allocation: an agent asks the human first (AGENTS.md
# rule 3). With several allocations add -A <id>.

# Submit from the repository root; outside Slurm the script finds itself.
JOBS_DIR=${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/benchmarks/square_cylinder_re10000/jobs}
[ -f "$JOBS_DIR/common.sh" ] || JOBS_DIR=$(cd "$(dirname "$0")" && pwd)
. "$JOBS_DIR/common.sh" || exit 1
cd "$REPO" || exit 1
./scripts/build.sh cpu || { echo "FATAL: build failed"; exit 1; }
./scripts/test.sh cpu -L fast -j "${SLURM_NTASKS:-4}"
