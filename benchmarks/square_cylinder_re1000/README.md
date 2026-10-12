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
| `l1_q5.yaml` | 0.25 | 0.1 | 40,000 | 12-18 k (Q5/Q4: about l2's unknowns) | ~2-3 days at 4 ranks, desktop |

`l1_q5.yaml` is the element-order experiment (human 2026-10-12): l1's AMR with Q5/Q4
elements. Compared with l2 (about the same unknowns) and l1 (the same mesh), it says
whether higher order is cheaper per unit of accuracy here, which sets the order for Re 10 000.
**Started and stopped 2026-10-12 at t = 5** (human: "seems like it won't help us much"):
~2 ms per element per step against Q3's ~0.4 on the same desktop (sharing it with l2), and
21 solver iterations per step against 17, so about 2x Q3's cost per unknown; extrapolated to
~4 days. Not resumed.

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
  raise `time.t_final`, and run from the same directory. Tested on a tiny deck (2026-10-12),
  where it found two bugs, both fixed: the history file was replaced instead of continued,
  and a restart with forces aborted in its first summary. `analyze.py` drops the history
  rows a later segment supersedes. **A level run before that fix must not be restarted
  with the old binary.** On Bridges-2, `../square_cylinder_re10000/jobs/production.sbatch`
  does all of this.
- No `reference:` block. Keep `forces.statistics: true`: it also enables the per-step force
  evaluation, and with it off the history's C_D and C_L are `nan` (found the hard way:
  two runs were lost). At Re 1000 the shedding is modulated and broadband, so the summary's
  `forces.periodic` block (the period finder) is meaningless: ignore it. Statistics come from
  the per-step history (`output.history`) and `analyze.py`.
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

## Is the mesh freeze safe? (`apps/mesh_census`)

AMR here only refines (coarsening would break the time-stepping history), so behind the
shed vortices the mesh grows for as long as AMR runs. `amr.end_time: 60` freezes it. That is
safe for the forces if the region near the body is fully refined well before t = 60 and
the growth that is cut off is far downstream, where it does not reach the forces. Whether
it is, nobody had measured (human 2026-10-12: "how are we confident that having the mesh
stopping expanding at t=60 is the right call?").

`apps/mesh_census` answers the first half. It rebuilds a run's mesh from a checkpoint,
one refinement pass at a time exactly as a restart does, and counts the elements per band
of distance downstream of the square after each pass. It only reads; run it at the run's
rank count, on a copy of the checkpoint (a running job overwrites it in place):

```sh
cp -r chk_l2 /tmp/chk_l2_copy
mpirun -np 4 build/cpu/apps/mesh_census -d benchmarks/square_cylinder_re1000/l2.yaml \
    -c /tmp/chk_l2_copy -H sqcyl_re1000_l2/sqcyl_re1000_l2_history.csv \
    -b -2,2,5,10,20,40,60 -e 3 -o l2_census.csv
```

Reading it: the freeze is safe for the near field if the bands within ~10 D of the square
have stopped growing well before t = 60. The second half, whether the forces depend on the
far wake left coarse, needs a run: branch a level from its t = 60 checkpoint with AMR
continuing (`amr.end_time` 130) and compare the statistics over t = 60-130 with the frozen
run (cheapest at l1).

**First reading, L2 at t = 29 (checkpoint of 2026-10-12, 29 passes):** elements per band of
x (the square at 0):

| t | total | < -2 | -2..2 | 2..5 | 5..10 | 10..20 | >= 20 |
|---|---|---|---|---|---|---|---|
| 21 | 5,980 | 229 | 1,946 | 2,704 | 467 | 274 | 360 |
| 24 | 8,274 | 229 | 1,948 | 3,502 | 1,940 | 295 | 360 |
| 27 | 11,214 | 229 | 2,038 | 3,630 | 4,609 | 348 | 360 |
| 29 | 13,083 | 229 | 2,042 | 3,899 | 5,858 | 695 | 360 |

The band around the square stopped growing by t ~ 24 (its smallest cells, 0.058, are close
to the `min_size` floor of 0.05); 2-5 D is slowing (+7 % from t = 27 to 29); the growth is a front moving
downstream (5-10 D filling now, 10-20 D starting, nothing refined beyond 20 D yet). Refined
elements reach |y| = 7 D in 2-20 D: the wake is wide.

**Second reading, L2 at t = 42 (42 passes):**

| t | total | -2..2 | 2..5 | 5..10 | 10..20 | 20..40 |
|---|---|---|---|---|---|---|
| 30 | 14,091 | 2,042 | 3,997 | 6,304 | 1,159 | 264 |
| 36 | 20,715 | 2,061 | 4,001 | 8,142 | 5,882 | 304 |
| 39 | 24,569 | 2,062 | 4,005 | 8,340 | 9,485 | 352 |
| 42 | 28,735 | 2,067 | 4,005 | 8,387 | 13,158 | 793 |

Everything within 10 D of the square is refined by t ~ 39 (2-5 D since t ~ 30; 5-10 D
+0.6 % over t = 39-42), well before the freeze at 60. The front moves downstream at about
one diameter per time unit (the flow speed): 10-20 D fills now, 20-40 D starts, so at the
freeze it should be near 40 D and only the wake beyond leaves coarse. **The near field is
safe to freeze.** Whether the forces feel the coarse far wake is the A/B branch's question.
About half the elements end up 10-40 D downstream: if the forces don't need them, refining
only near the body would roughly halve the mesh (a cost lever for Re 10 000,
docs/design/re10000_plan.md §5).

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

Window t >= 30 (100 time units), batch means of 10 time units, np 4. `results/` holds the
resampled series and the stats JSON for each level. Read the errors as crude: 10 batches, and
the C_L spectrum at the finer level is broadband (the two peaks hold only 12-16 % of the power).
**l05 -> l1 differs well beyond the sampling error** (C_D +31 %, C_L rms +45 %, 2 sigma 9 % and 20 %),
so l05 is under-resolved. **l1 against l2 is not yet known** (l2 is running on the desktop). With
100-unit windows the sampling error of l1 is 3 % (C_D) and 6 % (C_L rms), so the 1-2 % criterion
cannot be tested yet: it needs a window several times longer.

Fill in as levels finish (`results/`). The only number so far is from an exploratory run
that was cap-limited (9,003 elements from t = 40, so **not** a level of this ladder):

| Run | elements | window | C_D mean | C_L rms | spectral peaks |
|---|---|---|---|---|---|
| exploratory, tol 0.25 / `min_size` 0.1, cap 9,000, MacBook Air | 9,003 (capped) | t 30-160 | 2.09 +- 0.04 | 1.30 +- 0.04 | 0.092, 0.185 |

| Level | elements | steps | wall | C_D mean | C_L rms | peaks | machine, commit |
|---|---|---|---|---|---|---|---|
| l05 | 3,797 | 2,276 | 0.8 h | 1.575 +- 0.022 | 0.869 +- 0.033 | 0.180 (59 % of power), 0.080 | MacBook Air M3 8 GB, Apple clang 17, src at 2870c38, 2026-10-12 |
| l1 | 19,429 | 5,291 | 9.5 h (incl. ~1.5 h of lid-closed sleep) | 2.063 +- 0.069 | 1.258 +- 0.081 | 0.110 (16 %), 0.150 (12 %) | same |
| l2 | | | | | | | |
| l3 | | | | | | | |
