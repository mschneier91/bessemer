#!/bin/bash
#SBATCH -J bessemer-gpu-aware-delta
#SBATCH -A bjdd-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=120g
#SBATCH -t 3:00:00
#SBATCH -o bessemer-gpu-aware-%j.out

# environments/delta/gpu_aware.sbatch -- check GPU-aware Cray MPICH on two
# A100s (half a gpuA100x4 node): Taylor-Green Q6 at 192^3 nodes (21M velocity
# DOFs) for a few steps at 1 rank, at 2 ranks with GPU-aware MPI and at 2 ranks
# staged through the host; then the fast tier at np 2, one GPU per rank.
# Needs build/cuda (test.sbatch).
#
#   cd environments/delta && sbatch gpu_aware.sbatch
#
# Submitting it spends the allocation (two A100-hours per hour): an agent asks
# the human first (AGENTS.md rule 3).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR/../.."
repo=$PWD
echo "host $(hostname -f), $(date), commit $(git rev-parse --short HEAD) on $(git branch --show-current)"
export INCNS_MACHINE=delta
. scripts/env.sh || exit 1
nvidia-smi -L

out=$repo/runs/delta_gpu_aware/$SLURM_JOB_ID
mkdir -p "$out"
deck=$out/tgv_n32_short.yaml
sed -e 's/elements: \[21, 21, 21\]/elements: [32, 32, 32]/' \
    -e 's/t_final: 20/t_final: 0.5/' \
    -e 's/^  enabled: true/  enabled: false/' \
    -e 's/^  progress: 0.5/  progress: 0.1/' \
    benchmarks/tgv_re1600/tgv_re1600_n21.yaml > "$deck"
grep -E "elements|t_final|enabled" "$deck"

run() # name ranks [env assignments...]
{
  name=$1; np=$2; shift 2
  mkdir -p "$out/$name" && cd "$out/$name" || return
  echo "=== $name: $np rank(s) $*, $(date)"
  srun --overlap -n "$np" -c 16 --cpu-bind=cores \
    env INCNS_DEVICE=cuda "$@" "$repo/build/cuda/apps/run_case" "$deck" > run.log 2>&1
  echo "exit $?"
  grep -E "GPU-aware MPI|rror" run.log | head -5
  tail -4 run.log
  python3 - tgv_n21/tgv_summary.json <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for k, v in s.items():
    if isinstance(v, (str, int, float, bool)) or v is None:
        print(f"  {k}: {v}")
    elif isinstance(v, dict) and k != "profile":
        print(f"  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items()
                                      if not isinstance(b, (dict, list)))[:400])
PY
  cd "$repo"
}

run np1 1
run np2_gpu_aware 2
run np2_host_staged 2 MPICH_GPU_SUPPORT_ENABLED=0 INCNS_GPU_AWARE_MPI=0
echo "=== diagnostics, last lines"
for d in np1 np2_gpu_aware np2_host_staged; do
  echo "$d: $(tail -1 "$out/$d/tgv_n21/tgv_diagnostics.csv" 2>/dev/null)"
done
for d in np1 np2_gpu_aware np2_host_staged; do
  echo "$d profile:"; head -12 "$out/$d/tgv_n21/tgv_profile.csv" 2>/dev/null
done

echo "=== fast tier at np 2, one GPU per rank, $(date)"
INCNS_DEVICE=cuda INCNS_JOBS=2 scripts/test.sh cuda -L "smoke|fast" -R "_np2$" \
  -E "^(nse_mms_test|tgv_nse_test)_np" --timeout 3600
echo "tests done (exit $?), $(date)"
