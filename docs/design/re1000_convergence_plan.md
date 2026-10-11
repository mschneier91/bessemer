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

The flow **does shed**: C_L swings +-2.4 and changes sign every ~6 time units on average.
But at Re 1000 the shedding is modulated and broadband, not one clean period (checked on
the level-1 history, t = 60-105, 45 time units: C_L rms 1.26 against 0.39 at Re 200; the
C_L up-crossing periods run 3.3-9.2 with mean 6.1 and std 2.2, against 6.39 +- 0.01 at
Re 200; only ~35 % of the spectral power lies within +-20 % of the peak, against all of it
at Re 200). The same script reproduces the Re 200 deck's St = 0.1565 exactly, so the
method is sound. Consequences: there is no single period to average over and two runs at
the same resolution differ in detail; what can converge are **statistical means**, to
within their own sampling error. Therefore:

- Quantities (human 2026-10-11: these three, nothing else): **C_D mean, C_L rms
  (the mean C_L is ~0 by symmetry), and the Strouhal number**. The Strouhal number is
  taken from the spectral peak of C_L (Welch; the deck's period finder needs near-periodic
  shedding and may report `found: false`). With a broadband signal the peak and the
  mean up-crossing period can differ (0.09 against 0.164 on that 45-unit window, whose
  frequency resolution is only 0.022), so the *definition* has to be fixed once and
  applied to every level, and the window has to be long enough to resolve the peak.
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

**These element counts are assumptions, not measurements** (human 2026-10-11 questioned
them, rightly): "halving tolerance and `min_size` gives x2-3 elements" is a rule of thumb
extrapolated from a mesh that the cap was holding at 9,003. 9,000 Q3 elements is already
~200 k dofs, which is a lot for 2D at Re 1000, and level 1 may be close to converged;
L2 and L3 may turn out smaller than tabulated, or unnecessary. The census (T2) replaces
the guesses with a number; until then read the L2 and L3 rows as upper-end planning
figures. The Re 10 000 figure (1-3 M unknowns) is from status.md, not from this plan.

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

## Decisions

1. Criterion: **decided** (human 2026-10-11): 1-2 % between levels, against the
   statistical error (the larger of the two).
2. Where L2 runs: **the human's** (desktop, or Bridges-2 with an allocation).
3. Strouhal from the spectral peak of C_L: **agreed in principle** (it reproduces the
   Re 200 value exactly); the definition for a broadband spectrum is to be fixed once
   the longer window is available (section 1).
4. A field-based statistic (mean wake length): **not wanted** (human 2026-10-11: drag,
   lift and Strouhal are the three quantities).
