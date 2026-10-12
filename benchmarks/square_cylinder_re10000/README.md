# Square cylinder at Re = 10 000 (2D): resolution study on Bridges-2

The Re 1000 study (`../square_cylinder_re1000/`) continued to Re = 10 000: same domain,
decks, stepping and analysis, **in 2D on purpose** (a numerical study, not a model of the
3D physical flow). The plan, the costs and the literature context:
[docs/design/re10000_plan.md](../../docs/design/re10000_plan.md).

**Status (2026-10-12): not run.** The decks load and the job scripts were tested on the
desktop without Slurm (tiny decks). Every Bridges-2 job needs the human's yes, with its
resources and cost (AGENTS.md rule 3). The allocation is 5 000 core-hours, CPU only.

## Files

| File | What |
|---|---|
| `l1eq.yaml` | Re 10 000 at Re 1000 L1's tolerance (0.25), `min_size` 0.032: ~150 k elements, ~1.5-2 k core-hours (the pilot) |
| `l2eq.yaml` | at L2's tolerance (0.125), `min_size` 0.016: ~470 k elements, ~6-11 k core-hours (over the allocation) |
| `probe_142k.yaml`, `probe_407k.yaml` | fixed meshes of those sizes (no AMR), ~20-30 steps at CFL 2: timing only |
| `jobs/build_test.sbatch` | Phase B1: build and fast tier on an RM node (~10 core-hours) |
| `jobs/probe.sbatch` | Phase B2: the probes at 32, 64, 128 ranks on one RM node (~100-190 core-hours) |
| `jobs/production.sbatch` | Phase C: one level, continued from its checkpoint by each further job |

The element counts, steps and costs are extrapolated from the desktop's Re 1000 L2; the
probe measures them.

## Before anything: the RM environment (Phase B0, on the cluster)

`environments/psc_cpu` is resolved but not installed, and it pins the GPU nodes' compiler
build. For RM nodes: point its gcc external at
`/opt/packages/gcc/v13.3.1-p20240614/b2rm`, resolve on the login node, and install on an
RM node. The job scripts check for the `b2rm` gcc and stop otherwise.

## Running (from the repository root on Bridges-2)

```sh
sbatch benchmarks/square_cylinder_re10000/jobs/build_test.sbatch    # B1
sbatch benchmarks/square_cylinder_re10000/jobs/probe.sbatch         # B2: read its table
```

**Production.** Pick the rank count from the probe, then keep it for every job of the
level: a checkpoint restarts only at its own rank count (the script checks). Submit the
whole chain at once, as one approval with its total cost. Jobs after the level is done
exit at once and charge almost nothing.

```sh
# l1eq as three chained 48 h jobs on one RM node (128 ranks): at most 3 x 48 x 128 =
# 18 432 core-hours of limit, ~1.5-2 k expected; a job is charged for the time it runs.
jid=$(sbatch --parsable benchmarks/square_cylinder_re10000/jobs/production.sbatch l1eq)
for i in 2 3; do
  jid=$(sbatch --parsable --dependency=afterany:$jid \
        benchmarks/square_cylinder_re10000/jobs/production.sbatch l1eq)
done
# fewer ranks (often fewer core-hours, more wall time): add -p RM-shared -n 64 to each
```

Runs go to `$PROJECT/bessemer-runs/re10000/<level>/` (override with `RUN_ROOT`), with the
log `<level>.log`, the output directory and the checkpoint. To lengthen a finished level's
averaging window: `EXTEND_T_FINAL=200 sbatch ... production.sbatch l1eq` (it continues from
the checkpoint; the history file continues too).

## Analysis

As at Re 1000 (same script, same criterion):

```sh
python3 benchmarks/square_cylinder_re1000/analyze.py --t0 30 \
    l1eq=$PROJECT/bessemer-runs/re10000/l1eq/sqcyl_re10000_l1eq/sqcyl_re10000_l1eq_history.csv
```

It handles histories written across restarts: rows a later segment supersedes are dropped.
A level counts only if its final element count is below its cap (`max_elements`).

## Results

| Level | elements | steps | core-hours | C_D mean | C_L rms | spectral peaks | commit |
|---|---|---|---|---|---|---|---|
| l1eq | | | | | | | |
| l2eq | | | | | | | |
