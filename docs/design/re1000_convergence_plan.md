# Re = 1000 convergence study: plan

Status: agreed 2026-10-11 (human and agent); L0.5 and L1 are running on the MacBook Air.
Goal: find the AMR resolution beyond which the **statistics** of the Re = 1000 square-cylinder
flow stop changing meaningfully (1-2 %), so that Re = 10 000 can be scoped from a resolved
Re = 1000, not a guess. Domain, Joly, Etienne & Pelletier (2012) Fig. 3; U = D = 1.

## 1. Method

Same initial condition, same deck, same adaptive stepping (CFL-controlled, OIFS CFL 2,
BDF3); only the AMR criterion changes from level to level. Run each level out past the
spin-up, then compare **statistics** between consecutive levels:

- **C_D mean, C_L rms, and the Strouhal number** (human 2026-10-11: these three).
  C_L mean is ~0 by symmetry. The Strouhal number is the position of the spectral peak
  of C_L (Welch), reported with the next peak: at Re 1000 the spectrum has two comparable
  peaks (~0.10 and ~0.19), so a single St from the deck's period finder is not meaningful.
- **Error bars** from batch means of the per-step history (`output.history`), so that "the
  answer doesn't change very much" has a number: a difference between levels counts only
  if it exceeds the sampling error.
- **Stop refining** when two consecutive levels agree to 1-2 % (or within the error bar,
  whichever is larger).

The level is defined by the AMR indicator tolerance and `min_size` together (each
halves per level). `max_elements` is a **fuse**, not part of the definition: a level only
counts if its final element count is below the cap (a flat count equal to the cap means the
mesh was cost-limited; that happened to the first 9,000-element run).

## 2. Run protocol (decided 2026-10-11)

| Item | Value | Why |
|---|---|---|
| `t_final` | 130 | spin-up ~30 (the first level-1 run was stationary from t ~ 20-30) + 100 averaging; extend from the checkpoint while the error bar is above target |
| `amr.end_time` | 60 | the mesh is fixed for the averaging window, and the far wake is not refined for ever (to be checked: it assumes the forces do not depend on it; `apps/mesh_census` measures where the mesh grows: benchmarks/square_cylinder_re1000/README.md, "Is the mesh freeze safe?") |
| averaging window | t >= 30 | |
| `forces.statistics` | off | the period finder needs periodic shedding; it reports `found: true` with a period spread of 1.34 at Re 1000 and a Strouhal number that is just the crossing rate |
| `reference:` block | none | the paper values are for Re 200 |
| checkpoint | every 1,000 steps | same np to restart, to extend a run |
| cap | 20,000 (L0.5), 40,000 (L1), set per level | a fuse for an 8 GB machine |

Decks: `runs/re1000/series/{l05,l1}.yaml` (derived from `cases/square_cylinder_re200_oifs.yaml`
by Re, tolerance, `min_size`, `max_elements`, `t_final`, `amr.end_time`, checkpoint, output).

## 3. Levels and cost (this Mac, 8 GB, np 4; extrapolated except where marked)

Measured: 0.33-0.44 ms per element per step (higher under memory pressure); ~2 KB per dof;
dt ~ 0.025 at `min_size` 0.1 and ~0.012 at 0.05.

| Level | tolerance / `min_size` | elements | wall to t = 130 | where |
|---|---|---|---|---|
| L0.5 | 0.5 / 0.2 | 3-5k | ~1 h | here |
| L1 | 0.25 / 0.1 | 12-18k | ~8-12 h | here |
| L2 | 0.125 / 0.05 | 35-50k (13,083 at t = 30, measured) | ~2-3 days | desktop or HPC |
| L3 | 0.0625 / 0.025 | 100-150k | days; ~6 GB | HPC |

L3 was started and stopped at t ~ 16 (14,361 elements at t = 16, checkpoint at t ~ 14 in
`runs/re1000/study/chk_l3`): the start-up transient is not a statistic (section 5).

## 4. Analysis

`benchmarks/square_cylinder_re1000/analyze.py` (to be written; the prototype is
`runs/re1000/l1_stats.py`): reads each level's history, resamples to a uniform grid, and
prints per level C_D mean, C_L rms, the spectral peaks, each with batch-means error; then
the differences between consecutive levels against the error bars, and the element counts.

## 5. What was learned first (and why the plan looks like this)

- **Time step:** CFL 2 against CFL 1 on the same mesh: window means within 1-2 %; trajectories
  agree out to t ~ 12-16. CFL 2 is not the limiting error at level 1.
- **The start-up transient is not a convergence test.** L0.5, L1, L2 (and L3 to t = 16) agree
  on the first C_D peak to a few % in value (2.10, 2.08, 2.14) but the lift trajectories
  scatter non-monotonically after t ~ 5 (first excursion -0.73, -0.35, -0.80). The seeded
  instability amplifies whatever the discretisation does to the seed. Pointwise differences
  say nothing about resolution; statistics do.
- **Sampling error is the cost driver.** First level-1 run (cap-limited, 9,003 elements):
  C_D mean 2.07, C_L rms 1.28; 2-5 % sampling error at 70-100 time units by batch means,
  though the running mean from t = 20 settles to within ~1 % by t ~ 60. The averaging
  window, not the mesh, sets the cost.
- **The flow sheds**, but modulated and broadband (C_L rms 1.26 against 0.39 at Re 200;
  C_L up-crossing periods 3.3-9.2). It is not periodic.
- Estimate for Re 10 000 on this machine: not possible (1-3 M unknowns, ~16-48 s per step,
  ~200 k steps, memory); HPC work (docs/status.md).

## 6. Data in the repo

Raw histories stay in `runs/` (git-ignored; 50-400 KB each, more for fine levels).
Proposal (human to confirm), following `benchmarks/`: a directory
`benchmarks/square_cylinder_re1000/` with

- `README.md`: the case, the protocol of section 2, the results table, how to reproduce;
- the decks (`l05.yaml`, `l1.yaml`, ...), derived and commented as above;
- `analyze.py`, as section 4;
- `results/<level>_series.csv`: t, C_D, C_L resampled to dt = 0.05 (about 2,600 rows,
  ~100 KB, enough for means, rms and spectra to f = 10), and `results/<level>_stats.json`
  (the numbers, error bars, element count, wall time, commit hash, machine).

No reference data (there is none published for this configuration); the levels are
the reference for each other. Results are also summarised in docs/status.md.

## 7. Gates to Re = 10 000

Superseded by [re10000_plan.md](re10000_plan.md) (2026-10-12: measured costs, the
5 000-core-hour CPU allocation, the phases and job scripts). The gates below are kept there.

Resolved Re 1000 (section 1), the temporal check (section 5), error-barred statistics
(section 4), a CPU strong-scaling or GPU np 4 validation (docs/status.md, developing.md
7.5), and an HPC allocation (AGENTS.md rule 3: the human says yes to each job, with
resources, walltime and cost stated).
