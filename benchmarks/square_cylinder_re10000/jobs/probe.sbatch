#!/bin/bash
#SBATCH -J bessemer-re10k-probe
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 1:30:00
#SBATCH -o re10k-probe-%j.out

# Phase B2 (docs/design/re10000_plan.md): the scaling probe. Each probe deck (a
# fixed Re 10 000 mesh of the two production sizes, 20-30 steps) at each rank
# count, on one RM node; then a table of seconds per step, parallel efficiency,
# iterations and memory per rank. These numbers replace the plan's assumed 60%
# efficiency.
#
#   cd <repo> && sbatch benchmarks/square_cylinder_re10000/jobs/probe.sbatch
#
# Cost: a full RM node charges 128 cores x the time used; estimated under 1 h,
# so ~100-130 core-hours, capped by the 1:30 limit at 192. An agent asks the
# human first (AGENTS.md rule 3).
#
# Overrides (also how to test the script on a workstation without Slurm):
#   PROBE_DECKS  deck names in benchmarks/square_cylinder_re10000 (default:
#                probe_142k probe_407k), or paths to deck files
#   PROBE_NPS    rank counts (default: 32 64 128)

JOBS_DIR=${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/benchmarks/square_cylinder_re10000/jobs}
[ -f "$JOBS_DIR/common.sh" ] || JOBS_DIR=$(cd "$(dirname "$0")" && pwd)
. "$JOBS_DIR/common.sh" || exit 1

DECKS=${PROBE_DECKS:-probe_142k probe_407k}
NPS=${PROBE_NPS:-32 64 128}
OUT=$RUN_ROOT/probe-${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
TIME=$(command -v /usr/bin/time || true)

for deck in $DECKS; do
  case "$deck" in
    */*) file=$deck ;;
    *) file=$BENCH/$deck.yaml ;;
  esac
  name=$(basename "$file" .yaml)
  for np in $NPS; do
    dir=$OUT/${name}_np$np
    mkdir -p "$dir"
    echo "== $name at $np ranks ($(date '+%T'))"
    # Per-rank peak memory with GNU time when the node has it.
    if [ -n "$TIME" ]; then
      wrap="$TIME -f %M -o $dir/rss.%RANK%"
    else
      wrap=""
    fi
    ( cd "$dir" && mpirun -np "$np" bash -c '
        w=$1; shift
        if [ -n "$w" ]; then
          exec ${w//%RANK%/${OMPI_COMM_WORLD_RANK:-0}} "$@"
        fi
        exec "$@"' _ "$wrap" "$RUN_CASE" "$file" --summary "$dir/summary.json" \
        > "$dir/run.log" 2>&1 ) || echo "   FAILED (see $dir/run.log)"
  done
done

echo "== the table (seconds per step: median over the steps after the third)"
python3 - "$OUT" <<'EOF'
import csv, glob, json, os, statistics, sys
out = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(out, "*_np*"))):
    name, np_ = os.path.basename(d).rsplit("_np", 1)
    hist = glob.glob(os.path.join(d, "*", "*_history.csv"))
    if not hist:
        rows.append((name, int(np_), None, None, None, None, None))
        continue
    with open(hist[0]) as f:
        r = list(csv.DictReader(f))
    steps = r[3:] if len(r) > 6 else r
    wall = statistics.median(float(x["step_wall"]) for x in steps)
    its = statistics.median(float(x["iterations"]) for x in steps)
    elements = int(float(r[-1]["elements"]))
    rss = [int(open(p).read().split()[-1]) for p in glob.glob(os.path.join(d, "rss.*"))]
    rows.append((name, int(np_), elements, len(r), wall, its,
                 max(rss) / 1024.0 if rss else None))
print("%-12s %5s %9s %6s %10s %8s %8s %13s %11s" % (
    "deck", "ranks", "elements", "steps", "s/step", "its", "eff", "core-ms/el", "MB/rank max"))
base = {}
for name, np_, el, n, w, its, rss in sorted(rows, key=lambda x: (x[0], x[1])):
    if w is None:
        print("%-12s %5d  (no history: the run failed)" % (name, np_))
        continue
    base.setdefault(name, (np_, w))
    b_np, b_w = base[name]
    eff = (b_w * b_np) / (w * np_)
    print("%-12s %5d %9d %6d %10.3f %8.1f %7.0f%% %13.3f %11s" % (
        name, np_, el, n, w, its, 100 * eff, 1e3 * w * np_ / el,
        "%.0f" % rss if rss is not None else "-"))
print("eff: relative to the fewest ranks of that deck. core-ms/el: core-milliseconds per")
print("element per step (the desktop's Re 1000 L2 at 4 ranks, AMR mesh, CFL 2: ~1.5). Cost:")
print("core-ms/el x elements x steps / 3.6e6 core-hours.")
EOF
