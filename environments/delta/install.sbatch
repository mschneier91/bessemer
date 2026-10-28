#!/bin/bash
#SBATCH -J bessemer-spack-delta
#SBATCH -A bjdd-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH -t 4:00:00
#SBATCH -o spack-install-%j.out

# environments/delta/install.sbatch -- install this environment from its
# spack.lock on an A100 node.
#
#   cd environments/delta && sbatch install.sbatch
#
# Submitting it spends the allocation: an agent asks the human first (AGENTS.md
# rule 3). A quarter of a gpuA100x4 node (1 A100, 16 of 64 cores, 60 GB) is
# the smallest request with 16 cores, billed as one A100-hour per hour of use.
# Nothing here uses the GPU itself. Rerunning is safe: Spack resumes at the
# first package that is not installed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/../.."
echo "host $(hostname -f), $(date)"
module list 2>&1 | head -20
curl -sSI https://github.com | head -1 || echo "NO INTERNET on this node"

export SPACK_DISABLE_LOCAL_CONFIG=1
. "${SPACK_ROOT:-$HOME/spack}/share/spack/setup-env.sh"
scripts/envlock.sh check environments/delta
# --dirty keeps the job environment's LD_LIBRARY_PATH, which the default
# modules fill: Cray MPICH's libmpi_gnu.so needs libfabric.so.1 and
# libpmi*.so.0 from /opt/cray at link time, and the linker finds them only
# there (Spack's compiler wrapper passes -rpath, so ld ignores the library's
# own search path). Without it every MPI package failed to configure ("Could
# NOT find MPI", job 23241743). The stage goes to the project space, so the
# logs of a failed build outlive the node's /tmp.
spack -c "config:build_stage:[/projects/bjdd/$USER/spack-stage]" \
  -e environments/delta install --dirty -j "$SLURM_CPUS_PER_TASK" --show-log-on-error

# numba from pip's wheel (Spack's py-numba builds LLVM).
spack env activate environments/delta
python3 -m pip install --no-cache-dir numba
python3 -c 'import numba, numpy; print("numba", numba.__version__, "numpy", numpy.__version__)'
echo delta > environments/.machine
echo "install done, $(date)"
