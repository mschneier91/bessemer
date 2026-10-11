# Re = 1000 convergence campaign: plan

Draft, 2026-10-11, written while the first full run was at t = 100 of 160. Goal (human):
establish that Re = 1000 (square cylinder, Joly, Etienne & Pelletier 2012 domain, 2D) is
**converged before going to Re = 10 000**. "Converged" means a 1-2 % change between levels
in the quantities below (human 2026-10-11), judged against the statistical uncertainty
(section 1). Items marked **DECIDE** are the human's.

## 0. What is measured so far (this Mac, 8 GB RAM, 4 performance cores, np 4, OIFS CFL 2)

| Quantity | Value | Source |
|---|---|---|
| Cost per element per step | 0.33-0.40 ms | probe t <= 30 and level-1 run (9,003 el: 3.0 s/step) |
| dt at `min_size` 0.1 | 0.019-0.028 | level-1 run; Re 200 at `min_size` 0.2 was 0.072 |
| dofs per element (Q3/Q2, 2D) | ~22 | Re 200: 53,526 + 11,842 dofs at 3,000 el |
| Mesh at tolerance 0.25, `min_size` 0.1 | **> 9,000 el** | cap 9,000 bound from t = 40 to at least t = 100 |
| Flow | irregular: C_D 1.25-2.9, C_L +-2.4, no settled period | level-1 run to t = 100 |

Not measured: how many elements the tolerance alone asks for; the statistical uncertainty of
the force means; memory per unknown; whether a restart accepts changed AMR parameters.

## 1. What "converged" can mean for an irregular flow

The 2D flow at Re 1000 is not periodic here, so there is no period to average over and no
single "right" time series: two runs at the same resolution differ in detail. What can
converge are **statistical means**, to within their own sampling error. Therefore:

- Quantities: C_D mean, C_L rms, a Strouhal number from the spectral peak of C_L (Welch,
  not the deck's period finder, which needs periodic shedding), and the mean wake length if
  a probe or field statistic is added later (not available now).
- Sampling error: **batch means** on the per-step history (`output.history: true` writes
  `<name>_history.csv`: t, dt, C_D, C_L, iterations, cells), non-uniform dt handled by time
  weights. No solver change is needed; this is post-processing (Python is in the stack).
- Criterion (**DECIDE**): level k+1 vs level k differ by less than max(1-2 %, 2 sigma of
  the difference). If sigma alone exceeds 1-2 %, the averaging window is too short and
  the level has not been tested yet, whatever the means say.
- Error sources to keep apart: spatial (tolerance, `min_size`, cap), temporal (CFL),
  statistical (window length), start-up (spin-up before the window).

## 2. Free work first (no compute)

1. **Post-process the level-1 history** (t > 90, where AMR is off and the mesh is fixed):
   batch-means sigma versus window length. This fixes the averaging window T_avg every
   later run needs, and T_avg is what drives cost.
2. **Keep the level-1 final checkpoint** (`runs/re1000/l1/chk`, rolling, written at the end)
   before anything reuses the directory. It is a developed state at 9,003 elements.

## 3. Two cheap tests (<= 30 min each, np 4)

- **T1, restart with changed AMR parameters.** Restart from the level-1 checkpoint with
  `amr.tolerance` lowered or the cap raised, for 2-3 time units. Answers whether a refined
  level can *continue* a developed flow (skipping a second spin-up) or must start again.
  Unknown today; restart is documented only as "same number of MPI ranks".
- **T2, mesh census.** Same restart with the cap at 100,000 and tolerance 0.25 for ~10 time
  units: where does the accuracy-driven element count settle, and how fast? Watch memory
  (RSS) and s/step. This is the number every level's cost hangs on, and the one missing
  number from the probe.

## 4. The series

Levels are defined by halving `tolerance` and `min_size` together, with the cap set high
enough that it never binds (check at the end: final elements < cap, count levelling by
itself; a flat count equal to the cap means the level is cost-limited and does not count).

Cost model (assumptions): wall = steps x 0.35 ms x elements; steps = T / dt with
T = 30 (equilibrate after restart) + T_avg (taken as 100, to be set by section 2);
elements x2-3 and dt x0.4-0.5 per level (dt at `min_size` 0.1 -> 0.2 went 0.025 -> 0.072,
a factor 2.9, so the dt factor may be nearer 0.35).

| Level | tolerance / `min_size` | elements (assumed) | dt | wall on this Mac (estimate) | where |
|---|---|---|---|---|---|
| L1' (cap not binding) | 0.25 / 0.1 | 12-20k (T2 gives it) | ~0.025 | ~6-10 h | here, overnight |
| L2 | 0.125 / 0.05 | 30-50k | 0.011-0.013 | ~25-70 h | desktop (~2.5x faster than the Air; unmeasured) |
| L3 | 0.0625 / 0.025 | 75-125k | 0.005-0.006 | ~10 days or more | HPC only (Bridges-2) |

Memory: ~22 dofs per element, so L2 is ~0.7-1.1 M dofs and L3 ~1.7-2.8 M. Bytes per dof
are **not measured**; on 8 GB total, L2 is doubtful on the Air and L3 is out.

Decision rule: run L1' and L2, compare with the criterion of section 1. If they agree,
Re 1000 is converged at L2 resolution and L3 is not needed. If not, L3 is the HPC job
and Re 10 000 waits.

If T1 shows a restart can continue at a lower tolerance, L2 and L3 start from L1''s
developed state and need only ~30 time units to re-equilibrate before the window instead of
a ~90 spin-up (130 instead of ~190 time units, about a third less); the table above
already assumes that.

## 5. Checks that are not resolution

- **Temporal:** a branch at CFL 1 (OIFS) from the same checkpoint on the same mesh, 30 time
  units, against CFL 2. Re 200 showed CFL 4 loses lift amplitude (+21 % C_L,rms), so this is
  not a formality at Re 1000. Cost ~2x per time unit.
- **Independence of the start:** the shedding is seeded by a bump (`perturbation 0.1` at
  (2, 0)). For a chaotic flow, two or three windows from different seeds are a legitimate
  substitute for one very long window and give a direct sigma. Cost-neutral per time unit.
- **Domain, base mesh:** not varied (the paper's domain; base mesh refined by AMR only).

## 6. Gate to Re = 10 000

Not before: converged Re 1000 (section 4), the temporal check (section 5), error-barred
statistics (section 2), a CPU strong-scaling or GPU np 4 validation (status.md, developing.md
7.5), an element-order question (Q5-Q7 per unknown at high Re, status.md), and **HPC**:
this Mac cannot hold it (1-3 M unknowns, ~16-48 s/step, ~200 k steps). Any batch job needs
the human's yes with resources, walltime and cost stated (AGENTS.md rule 3).

## 7. Order

1. Finish the level-1 run; analyse its history (section 2). Free.
2. T1 and T2 (section 3). ~1 h.
3. Choose T_avg and the cap; run L1'. Overnight.
4. L2 on the desktop (**DECIDE**: machine, and whether a day or more is acceptable).
5. CFL 1 branch; seed independence.
6. Decide L3 / HPC / Re 10 000 from the comparison.

## Decisions for the human

1. The criterion in section 1 (1-2 % or 2 sigma, whichever is larger).
2. Where L2 runs (desktop, or Bridges-2 with an allocation).
3. Whether a Welch-spectrum Strouhal number replaces the deck's period-based one (the
   deck's `strouhal` needs periodic shedding).
4. Whether to add a field-based statistic (mean wake length) before L2: it is a solver
   feature, not a deck key.
