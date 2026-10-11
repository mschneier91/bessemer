#!/bin/bash
#SBATCH -J bessemer-psc_cpu
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o spack-install-%j.out

# environments/psc_cpu/install.sbatch -- install this environment from its
# spack.lock on a compute node (docs/install/bridges2.md, Steps 2-4).
#
#   cd environments/psc_cpu && sbatch install.sbatch
#
# Submitting it spends the allocation: an agent asks the human first (AGENTS.md
# rule 3). It charges your default allocation; with several, add `-A <id>` to
# the sbatch command. Do the login-node steps first (Step 1): sources come from
# the local mirror, since compute nodes have no internet. Rerunning is safe:
# Spack resumes at the first package that is not installed.
#
# One H100 slice (12 cores): only its CPU cores are used; this
# environment has no CUDA at all.
# You are charged for the time used, not for the two-hour limit.

# The submitting shell may already have the login node build of gcc loaded
# (b2rm), and sbatch passes that environment on. Unload it while HOSTNAME still
# names the login node, then load this node's build (Gotchas #1 and #15).
type module >/dev/null 2>&1 || . /usr/share/lmod/lmod/init/bash
module unload gcc cuda 2>/dev/null
export HOSTNAME=$(hostname)
module load gcc/13.3.1-p20240614
case "$(command -v gcc)" in
  */b2gpu/*) ;;
  *) echo "FATAL: gcc is $(command -v gcc), not the b2gpu build"; exit 1 ;;
esac

. "${SPACK_ROOT:-$PROJECT/spack}/share/spack/setup-env.sh" || exit 1
# Only this environment's spack.yaml and the Spack checkout's own settings
# apply, as when scripts/env.sh activates it: nothing from ~/.spack.
export SPACK_DISABLE_LOCAL_CONFIG=1
cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f spack.lock ] || { echo "FATAL: no spack.lock in $PWD (submit from the environment directory)"; exit 1; }
spack env activate . || exit 1
echo "== node $(hostname), $(nproc) cores, gcc $(command -v gcc)"
module list 2>&1

# --dirty: perl's direct gcc calls need the module's LD_LIBRARY_PATH (Gotcha #4).
# --use-buildcache never: no probing of public binary caches from an offline node.
spack install -j "$(nproc)" --dirty --use-buildcache never || { echo "FATAL: spack install failed"; exit 1; }

# From here on a failed check does not stop the later ones, but the job ends
# with a non-zero status and a FAILED line naming each.
failed=""
check() { "$@" || failed="$failed [$*]"; }

echo "== Step 3: verify the stack"
spack env activate . || exit 1
check command -v mpicxx
check sh -c "! ompi_info | grep -qi cuda"
check ucx_info -v

echo "== Step 4: numba from the wheel mirror (numpy stays the Spack one)"
check command -v python3
# The installed numpy, pinned so that pip cannot bring a second one.
np=$(python3 -c "from importlib.metadata import version; print(version('numpy'))")
[ -n "$np" ] || failed="$failed [numpy is not installed in the view]"
check python3 -m pip install --no-index --find-links="$PROJECT/pip-wheels" numba "numpy==$np"
check python3 -c "from numba import cfunc; import numpy; print('cfunc OK, numpy', numpy.__version__)"

if [ -n "$failed" ]; then
  echo "== FAILED:$failed"
  exit 1
fi
echo "== done"
