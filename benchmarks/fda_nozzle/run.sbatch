#!/bin/bash
# benchmarks/fda_nozzle/run.sbatch -- one FDA nozzle deck on GPUs, one per rank.
#
#   cd benchmarks/fda_nozzle
#   sbatch -n 1 --gpus-per-node=1 -t 0:30:00 run.sbatch fda_nozzle_re500_smoke.yaml
#   sbatch -n 2 --gpus-per-node=2 -t 8:00:00 run.sbatch fda_nozzle_re500.yaml
#   sbatch ... run.sbatch fda_nozzle_re500.yaml runs/fda_nozzle/re500/chk   # restart
#
# Written for NCSA Delta (gpuA100x4: 40 GB A100s, Cray MPICH through srun;
# docs/install/delta.md). For Bridges-2 swap the header for the commented one
# below and the launcher for mpirun (docs/install/bridges2.md). Fill in the
# account, and check the partition and limits on the machine's own page.
#
# Submitting it spends the allocation: an agent states the resources, the
# walltime and the cost, and asks the human first (AGENTS.md rule 3), for each
# job. The 56k-hex mesh is above the rule's 32^3 elements, so every run of
# these decks needs that yes.
#
# The mesh file is not in git (126 MB). Generate it once, on a login node with
# gmsh's Python module (pip install --user gmsh):
#   python3 benchmarks/fda_nozzle/fda_nozzle_mesh.py --level 0 --order 3 \
#       -o runs/fda_nozzle/fda_nozzle_l0.msh
#
#SBATCH -J fda-nozzle
#SBATCH -A FILL_IN_ACCOUNT
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH -t 0:30:00
#SBATCH -o fda-nozzle-%j.out
#
# Bridges-2 (one H100 80 GB per rank):
#   #SBATCH -p GPU-shared
#   #SBATCH --gres=gpu:h100-80:1
#   #SBATCH -t 0:30:00
#   #SBATCH -o fda-nozzle-%j.out
set -uo pipefail
deck=${1:?usage: run.sbatch <deck.yaml in benchmarks/fda_nozzle> [restart chk dir]}
restart=${2:-}
cd "$SLURM_SUBMIT_DIR/../.."   # the checkout's root: the decks' paths are relative to it
echo "host $(hostname -f), $(date), commit $(git rev-parse --short HEAD) on $(git branch --show-current)"
echo "deck $deck, ranks ${SLURM_NTASKS:-1}${restart:+, restart from $restart}"

. scripts/env.sh                         # the machine's Spack environment (AGENTS.md rule 1)
mesh=runs/fda_nozzle/fda_nozzle_l0.msh
[ -s "$mesh" ] || { echo "missing $mesh: generate it first (header)"; exit 1; }
[ -x build/cuda/apps/run_case ] || { echo "build first: scripts/build.sh cuda"; exit 1; }

src=benchmarks/fda_nozzle/$deck
if [ -n "$restart" ]; then
  # A restart is the same deck plus `restart: <dir>` (same rank count). The
  # copy goes next to the run's output so the deck in git stays as it is.
  mkdir -p runs/fda_nozzle
  run_deck=runs/fda_nozzle/${deck%.yaml}_restart_$SLURM_JOB_ID.yaml
  { cat "$src"; echo; echo "restart: $restart"; } > "$run_deck"
else
  run_deck=$src
fi

export INCNS_DEVICE=cuda
# Delta: Cray MPICH through srun, one GPU per rank. Bridges-2: replace with
#   mpirun -np "${SLURM_NTASKS:-1}" build/cuda/apps/run_case "$run_deck"
srun -n "${SLURM_NTASKS:-1}" -c "${SLURM_CPUS_PER_TASK:-16}" --cpu-bind=cores \
  build/cuda/apps/run_case "$run_deck"
rc=$?
echo "run_case exit $rc, $(date)"
scripts/reap.sh 2>/dev/null || true
exit $rc
