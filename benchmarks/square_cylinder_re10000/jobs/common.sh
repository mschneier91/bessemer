# benchmarks/square_cylinder_re10000/jobs/common.sh -- sourced by the job scripts.
#
# Bridges-2 Regular Memory (RM) nodes, CPU only, the psc_cpu environment. On
# another machine (or to test the scripts on a workstation) set INCNS_MACHINE
# to its environment and INCNS_SKIP_MODULES=1.

# JOBS_DIR: set by the job script (under Slurm, $0 is a spooled copy).
REPO=$(cd "$JOBS_DIR/../../.." && pwd)
INCNS_REPO_ROOT=$REPO
export INCNS_REPO_ROOT
BENCH=$REPO/benchmarks/square_cylinder_re10000

if [ -z "${INCNS_SKIP_MODULES:-}" ]; then
  # PSC builds gcc per node class: an RM node needs the b2rm build (docs/install/
  # bridges2.md, Gotchas #1 and #15). The submitting shell may carry the login
  # node's module: unload it while HOSTNAME still names that node.
  type module >/dev/null 2>&1 || . /usr/share/lmod/lmod/init/bash
  module unload gcc cuda 2>/dev/null
  export HOSTNAME=$(hostname)
  module load gcc/13.3.1-p20240614
  case "$(command -v gcc)" in
    */b2rm/*) ;;
    *) echo "FATAL: gcc is $(command -v gcc), not the b2rm (RM node) build;" \
            "psc_cpu must be resolved for RM first (docs/design/re10000_plan.md, B0)"
       exit 1 ;;
  esac
  export INCNS_MACHINE=${INCNS_MACHINE:-psc_cpu}
fi

. "$REPO/scripts/env.sh" || { echo "FATAL: scripts/env.sh failed"; exit 1; }
RUN_CASE=$REPO/build/cpu/apps/run_case
[ -x "$RUN_CASE" ] || { echo "FATAL: no $RUN_CASE (run jobs/build_test.sbatch first)"; exit 1; }

# Where runs go: persistent, outside the checkout.
RUN_ROOT=${RUN_ROOT:-${PROJECT:-$REPO/runs}/bessemer-runs/re10000}
mkdir -p "$RUN_ROOT"

echo "== $(date '+%F %T')  node $(hostname)  commit $(git -C "$REPO" rev-parse --short HEAD)" \
     " ranks ${SLURM_NTASKS:-?}  run root $RUN_ROOT"
