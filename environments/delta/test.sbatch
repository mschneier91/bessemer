#!/bin/bash
#SBATCH -J bessemer-test-delta
#SBATCH -A bjdd-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH -t 3:00:00
#SBATCH -o bessemer-test-%j.out

# environments/delta/test.sbatch -- build bessemer's CUDA preset on an A100
# node, smoke-run a case, then the fast tier at one rank on the GPU
# (docs/install/spack.md, "An A100 cluster": the two tests that time out on
# GPUs are left out; np 2 wants one GPU per rank).
#
#   cd environments/delta && sbatch test.sbatch [ctest args...]
#
# Extra arguments go to ctest, e.g. `-I 1,30` for the first 30 tests: when the
# A100 queue is full, the 1-hour gpuA100x4-interactive partition starts at once
# (sbatch -p gpuA100x4-interactive -t 1:00:00 test.sbatch -I 1,30), and the
# build is incremental, so the tier can run as a few short jobs.
#
# Submitting it spends the allocation: an agent asks the human first (AGENTS.md
# rule 3). The tests launch through environments/delta/mpiexec (srun steps
# inside this job).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/../.."
echo "host $(hostname -f), $(date), commit $(git rev-parse --short HEAD) on $(git branch --show-current)"
export INCNS_MACHINE=delta
# Compile for the stack's target, as CI does (developing.md §7.6): MFEM is built
# for x86_64_v3 (FMA), and without the same -march persistent_fgmres_test's
# comparison with MFEM fails at roundoff (5e-11 against 1e-14, job 23250720).
# CXXFLAGS seeds only a fresh configure, so a cache made without it is dropped.
export CXXFLAGS=-march=x86-64-v3
grep -qx 'CMAKE_CXX_FLAGS:STRING=-march=x86-64-v3' build/cuda/CMakeCache.txt 2>/dev/null ||
  rm -f build/cuda/CMakeCache.txt
scripts/build.sh cuda || { echo "BUILD FAILED"; exit 1; }
echo "build done, $(date)"
scripts/doctor.sh --run
echo "doctor done, $(date)"
INCNS_DEVICE=cuda INCNS_JOBS=2 scripts/test.sh cuda -L "smoke|fast" -R "_np1$" \
  -E "^(nse_mms_test|tgv_nse_test)_np" --timeout 3600 "$@"
echo "tests done (exit $?), $(date)"
