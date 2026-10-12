# Square cylinder at Re = 1000: resolution convergence

Flow past a square cylinder (side D, zero incidence) at Re = U D / nu = 1000 in the large
domain of Joly, Etienne & Pelletier (J. Fluids Struct. 28 (2012) 232-243, Fig. 3). 2D.
The point of this benchmark is not to match a published value (there is none for this
configuration) but to find the AMR resolution **beyond which the flow statistics stop
changing**, so that higher Reynolds numbers can be scoped from a resolved Re = 1000.
The plan and its reasoning: [docs/design/re1000_convergence_plan.md](../../docs/design/re1000_convergence_plan.md).

## The ladder

One deck per level; only the AMR indicator tolerance, `min_size` and the element cap
differ. Same initial condition, same adaptive CFL-controlled stepping (OIFS CFL 2, BDF3).

| Level | tolerance | `min_size` | cap (a fuse) | elements (expected) | wall at 4 ranks (expected) |
|---|---|---|---|---|---|
| `l05.yaml` | 0.5 | 0.2 | 20,000 | 3-5 k | ~1 h on an 8 GB MacBook Air |
| `l1.yaml` | 0.25 | 0.1 | 40,000 | 12-18 k | ~8-12 h on the Air |
| `l2.yaml` | 0.125 | 0.05 | 100,000 | 35-50 k | ~2-3 days on the Air; desktop |
| `l3.yaml` | 0.0625 | 0.025 | 250,000 | 100-150 k | HPC |

The expected figures are extrapolated from short runs (docs/status.md); the measured ones
go in the results table below. **A level counts only if its final element count is below
its cap**: a count pinned at the cap means the mesh was cost-limited, not accuracy-limited.
Memory is about 44 KB per element (measured at 12 k elements), so a cap of 100 k is ~4.4 GB.

## Protocol

- `t_final` 130: spin-up to t ~ 30 (the first run was statistically stationary from
  t ~ 20-30), then 100 time units of averaging.
- AMR in simulated time (an event every time unit) until `amr.end_time` 60, then a fixed mesh.
- A rolling checkpoint every 1,000 steps. To extend a run (the plan: extend while the error
  bar is above the target), add `restart: <chk dir>` to the deck at the same rank count and
  raise `time.t_final` (a restart with a larger `t_final` has **not been tried yet**; the
  checkpoint is written, so try it before relying on it).
- No `reference:` block, and `forces.statistics` is off: at Re 1000 the shedding is
  modulated and broadband, so the deck's period finder reports meaningless "periods".
  Statistics come from the per-step history (`output.history`).
- More than 4 ranks, or a 3D mesh above 32^3, or a GPU, needs the human's approval
  (AGENTS.md rule 3).

## Running and analysing

```sh
mkdir -p runs/sqcyl && cd runs/sqcyl
mpirun -np 4 ../../build/cpu/apps/run_case ../../benchmarks/square_cylinder_re1000/l1.yaml
python3 ../../benchmarks/square_cylinder_re1000/analyze.py \
    l05=sqcyl_re1000_l05/sqcyl_re1000_l05_history.csv \
    l1=sqcyl_re1000_l1/sqcyl_re1000_l1_history.csv \
    --save ../../benchmarks/square_cylinder_re1000/results
```

`analyze.py` (needs numpy) prints, per level, the element count, steps, dt, wall time,
the C_D mean and C_L rms over t >= 30 with batch-means sampling errors, and the two
strongest spectral peaks of C_L; then, for consecutive levels, whether each difference
exceeds both twice the sampling error and 2 % of the value. `--save DIR` writes
`<level>_series.csv` (t, C_D, C_L at dt = 0.05) and `<level>_stats.json` (the numbers, the
element count, the wall time, the commit). Those files are what gets committed; the raw
histories and the field output stay in the git-ignored `runs/`.

## What to quote, and what not to

- **C_D mean, C_L rms and the Strouhal number** are the quantities (human 2026-10-11). C_L
  mean is ~0 by symmetry.
- **The Strouhal number is the position of the spectral peak of C_L**, reported with the
  second peak: the spectrum has two comparable peaks (~0.09-0.10 and ~0.19) in the first
  run, so one number is incomplete. The mean up-crossing rate (~0.15-0.16) falls between.
- **Do not compare trajectories** past the first few time units: the seeded instability
  amplifies whatever the discretisation does to the seed, so L1, L2 and L3 agree on the first
  C_D peak (value 2.10, 2.08, 2.14) but not on the lift swings after t ~ 5. Statistics only.
- **Sampling error is the cost.** At 70-130 time units it is 2-5 %. A difference between two
  levels smaller than that is not a result; lengthen the window before refining.

## Results

Fill in as levels finish (`results/`). The only number so far is from an exploratory run
that was cap-limited (9,003 elements from t = 40, so **not** a level of this ladder):

| Run | elements | window | C_D mean | C_L rms | spectral peaks |
|---|---|---|---|---|---|
| exploratory, tol 0.25 / `min_size` 0.1, cap 9,000, MacBook Air | 9,003 (capped) | t 30-160 | 2.09 +- 0.04 | 1.30 +- 0.04 | 0.092, 0.185 |

| Level | elements | steps | wall | C_D mean | C_L rms | peaks | machine, commit |
|---|---|---|---|---|---|---|---|
| l05 | | | | | | | |
| l1 | | | | | | | |
| l2 | | | | | | | |
| l3 | | | | | | | |
