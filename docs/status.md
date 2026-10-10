# Project status

The living handoff log: where the project stands, what was decided (dated, with who
decided), and what is open. The maintainer and their agents update it as work lands.
Rules and design live in [developing.md](developing.md); this file is the *state*.

## Where development left off (2026-10-10)

Keep this section short and current: whoever stops working updates it (AGENTS.md rule 8).
The dated log below keeps the history.

**On branch `oifs-mass` (not merged yet):** OIFS's substeps invert the BDF step's own mass
operator, so the two can no longer differ (human: "I literally want it to error out if
the two aren't consistent, using the diagonal for both of them as the default is fine").
Pointwise for a diagonal mass (collocated, conforming: the default, unchanged results),
Jacobi-CG per RK stage otherwise (consistent mass; collocated at hanging nodes, ~6
its/stage), always the FULL inverse (a constrained solve left 3.6e-4 vs IMEX). Setup
checks M^-1(Mz) = z and aborts otherwise. OIFS now matches IMEX with either mass (wall MMS
3.0e-6 both; was 1.6e-3 mismatched). AMR runs change slightly (exact P^T D P instead of
row-sum lumping). Benchmark plan (3D TGV, channel Re_tau 180/395, sphere, DFG 3D,
Ethier-Steinman) proposed; waiting on the human's answers (Re_tau pair, forcing,
directory layout, GPU-hours).

**Just landed (branch `license`, merged to `main` and pushed 2026-10-09, human: "merge and
push"):** BSD-3-Clause `LICENSE` (copyright Michael
Schneier; human: "it should be me"). README rewritten so the agent is the way in: what to
ask, what the agent asks first, what comes back. Human: the test cases are not the
standard CFD benchmarks to showcase (literature table removed; `cases/` decks are now
"example decks"), and the "production DNS/LES on DOE systems" line removed. AGENTS.md
gains "Working with the user" and rule 9 (licensing); user runs go under `runs/`.
Spack for anyone's machine (human: "someone points their agent at the repo ... it can do
everything from installing with spack (with users permissions of course)"):
`environments/stack.yaml` (portable stack, + py-matplotlib), `scripts/setup.sh` (plan,
then `--yes`; builds gcc@14.3.0 and declares it external), `scripts/machine.sh` (no default
machine any more; this checkout has `environments/.machine` = desktop),
`docs/install/spack.md` (how it all fits, for agents). Not yet run: `setup.sh --yes` on a
fresh machine. Next: that test (host terminal, clean VM or container); the TGV example
(deferred by the human).

**Just landed (branch `agent-first`, merged to `main` and pushed 2026-10-09, human:
"commit merge and push"):**
- Environment refresh: MFEM 4.10.1-dev with MUMPS, hypre 3.2, libCEED 1.0. MFEM pinned
  to `1c19aba` because of an upstream regression (PR #5297, reported 2026-10-09).
- Agent-first front door: `AGENTS.md`, short README, `docs/using.md`,
  `docs/install/desktop.md`, CLAUDE.md split into `developing.md` + this file.
- Generated deck reference (`docs/deck_reference.md`); unknown deck keys are errors.
- `scripts/doctor.sh`: setup check + smoke run (seconds).
- Benchmarks as decks: `mesh.geometry` (square cylinder, DFG channel), deck velocity
  BCs (constant / parabolic, ramp / sine), boundary coverage check, `uniform` initial
  flow, time-based AMR, force statistics, progress / history, pressure probe,
  reference values, and `summary.json` from every `run_case` run. Decks:
  `cases/square_cylinder_re200[_oifs].yaml`, `dfg_2d1.yaml`, `dfg_2d3.yaml`,
  `cavity.yaml`. The OIFS deck reproduced the validated result (C_D 1.4275, C_L,rms
  0.3940, St 0.1568).

**Validated:** `dfg_2d1.yaml` (relative errors C_D 1.05e-4, C_L 2.79e-3, dp 4.06e-3 = the
slow-tier values, 70 s), `dfg_2d3.yaml` (dt 0.001; C_D,max 0.08%, 172 s),
`square_cylinder_re200_oifs.yaml` (C_D 1.4265, C_L,rms 0.3923, St 0.1565 in 2,191 steps,
13 min: the driver's run to every digit). **IMEX deck run:** `square_cylinder_re200.yaml` validated 2026-10-09: C_D 1.4436,
C_L,rms 0.4034, St 0.1568 in 11,130 steps, 42 min at 4 ranks. (An earlier run that seemed
to stop at t = 90 had been killed by the agent itself, to restart it on the corrected deck;
the restart never launched.) Against the OIFS deck: C_D 1.2% and C_L,rms 2.8% higher, St
equal; the "within 1%" figure is IMEX with OIFS's collocated mass, not this deck.
Debug-device sweep: green, 114/114. The full sweep after the merge failed only
`run_monitor_test` R3 (AMR on a periodic Taylor-Green box), the known MFEM periodic-NC
debug-device false positive; R3 now skips there like `amr_flow_test.cpp:88`. Bridges-2: its lock must be
re-resolved on the cluster (MUMPS + pin edit).

**Next, in order:**
1. A cold-start test: a fresh agent with no context, given real tasks (install check,
   reproduce a benchmark from its deck, set up a new case); fix where it stumbles.
2. HPC preparation for Re 10k / 100k (2D): decide whether OIFS at CFL 2 becomes the
   default; GPU validation (developing.md §7.5) or a CPU strong-scaling check;
   statistics with error bars.
3. When MFEM fixes PR #5297's regression: drop the pin in both `spack.yaml` files.

**Open decisions for the human:** OIFS CFL 2 as the default for production runs.

**Open finding (2026-10-09):** DFG 2D-3 on the refreshed stack differs from the 10-07
study: C_D,max 2.94858 at dt 0.001 (study 2.94451, a 0.14% spatial shift) and dt 0.0015
now diverges at t = 3.91 (the study found it stable; threshold ~0.0017). Bitwise the same
with the pre-OIFS code (`530f93d`) on the new stack, so not this branch's code; the
square cylinder is bitwise unchanged across the refresh (driver OIFS CFL 2: 1.4265 /
0.3923 / 0.1565, 2,191 steps, both stacks). Suspects: MFEM 17d1afc → 1c19aba (curved
DFG mesh / quadrature?) or a code change between the 10-07 study and 10-08. The deck uses
dt 0.001.

## Log: state and decisions (last full update 2026-10-08)

**Branches.** `main` includes `rotational-schur` (merged and pushed 2026-10-07:
rotation-aware Schur PC, DFG 2D-3 driver, `jacobi_pcg` default + grad-div in LOR-AMG, step
control `fixed|error|cfl` with Nek's CFL and the BDF3/EXT3 estimator, case-level CFL control
at 0.5, the debug-device ess-list fix developing.md §6) and **`dfg-scheme-comparison`** (fast-forward
merged and pushed 2026-10-08, human: "merge and push everything"). New work goes on a new
branch; merge/push only when the human says so. dfg-scheme-comparison: DFG app `-mref` nested meshes, per-step
history CSV and `step_wall`; `bench/dfg3/` (pinned two-slot queue runner, sweep generator,
analysis, HTML report generator, raw results of 2026-10-08); the work-precision write-up
(`docs/imex_vs_semi_implicit.md` §7 + summary at its top); **case-level default back to
BDF2/EXT2** (human 2026-10-08; developing.md §5.2). The charts were published as a private claude.ai
artifact (regenerate with `bench/dfg3/report.py OUT.html
bench/dfg3/results_2026-10-08/findings.html bench/dfg3/results_2026-10-08/sweep.txt`).

**Scheme comparison concluded (human 2026-10-08): for DFG 2D-3 there is no benefit to the
semi-implicit rotational form** — same CFL limit as IMEX, 7–9% more per step, comparable
lift, 50–70× worse drag; both fail silently past the limit (developing.md §5.6, the doc §7). The finest
mesh (M2) was started and stopped unfinished at the human's call.

**`directional-do-nothing`** (fast-forward merged and pushed 2026-10-08, human): Braack &
Mucha's directional do-nothing outflow condition for the IMEX convective form, default on,
explicit (on the RHS with the convection; no operator or preconditioner change — a
semi-implicit LHS version is the fallback if outlet backflow ever destabilizes below the CFL
limit) (developing.md §4 BCs; `docs/outflow_conditions.md`); `directional_do_nothing_test`; DFG app
`-cdn` for the classical condition.

**Square cylinder (Joly, Etienne & Pelletier, J. Fluids Struct. 28 (2012) 232–243;
human 2026-10-08) — branch `square-cylinder`, fast-forward merged and pushed 2026-10-08
(human: "commit merge and push this stuff")**: `src/mesh/square_cylinder.{hpp,cpp}`,
`apps/square_cylinder.cpp`,
`ConvectiveCfl::RateAndLocation` + `Case::ConvectiveCflNumber(Vector* where)` (where the
CFL rate peaks), `cylinder_mesh_test` M3. Details developing.md §5.6. The paper PDF `big_domain.pdf` sits
untracked in the repo root (do not commit it; copyright).
- **Re = 200, α = 0 — DONE, all within the human's 5%** (IMEX, Q3/Q2, CFL 0.5, EXT2, AMR):
  C_D 1.4434 (paper 1.44, +0.2%), mean C_L 0.0003 (0.001), C_L,rms 0.4034 (0.42, −4.0%),
  St 0.1568 (0.151, +3.8%), 10 periods, period spread 0.03%. Fine AMR (tol 0.12, 5,003
  cells vs 3,001): C_D 1.4425, C_L −0.0003, C_L,rms 0.4025, St 0.1569 (11,518 steps, dt
  5.3e-3–2.5e-2) — every coefficient within 0.2% of the base run → wake-converged; corner resolution (smallest cell 0.125 in
  both) and dt were NOT varied. dt 6.2e-3–2.5e-2 (mean 1.4e-2, ~440 steps/period vs the
  paper's ~32 at UΔt/D = 0.2); 11,138 steps, 69 min at np 4. Remaining −4% / +3.8% may be the
  paper's large implicit step (untested). Literature scatter (their Table 1): C_L,rms
  0.32–0.55, St 0.142–0.170.
- **Re = 1000 — moved to HPC (human 2026-10-08: "that whole campaign will be left to the
  hpc")**. A desktop level-1 run (tol 0.25, min size 0.1, cap 6000) was stopped at t = 20:
  dt 1.5–4e-3 (5–9× below Re 200), 4,708 cells at t = 20 and still refining, CFL peak in the
  near wake ((2.4, −0.3) → (9.4, −0.4)), ~0.44 s/step at np 4 → ~5 h for t = 160 at that
  level. No public benchmark: plan = self-convergence over 2–3 AMR levels (halve
  tolerance AND min size; track cells/unknowns, dt range, C_D, C_L,rms, St; converged when
  < 1–2% change) + a CFL 0.5 vs 0.25 branch from the t = 90 checkpoint; check periodicity
  (2D at Re 1000 may be irregular → batch-means error bars). Consider `-amr-start` (spin up
  on the coarse mesh) to cut the startup cost.
- **dt sawtooth on AMR meshes (open):** once shedding starts, every ~0.13 time units the CFL
  rate jumps ~1.5× in ONE step → the controller cuts dt ~1.9e-2 → ~1.0e-2 and regrows ×1.2.
  Forces unaffected (C_D jitter ~1e-4); ~10–20% extra steps. Likely a fast mode at its
  stability limit at CFL 0.5 on the AMR (hanging-node) mesh — conforming DFG meshes were
  stable to 0.7. Diagnose with the CFL location now printed; a 0.4 target likely removes it.
- **Human's goal: Re = 10k and 100k on HPC** (2D on purpose; human knows it is not
  physical). Estimates (2D: unknowns ∝ Re, dt ∝ Re^-1/2, work ∝ Re^3/2, ~30× per decade):
  Re 10k ~1–3e6 unknowns, smallest cell ~0.02 (Q3), dt ~8e-4; Re 100k ~1–3e7, ~0.006,
  ~2.5e-4; ~1e4 core-hours at Re 100k if the solver scales. Prerequisites agreed so far:
  the dt-sawtooth diagnosis; GPU validation (developing.md §7.5) or a CPU strong-scaling check of the CC
  Schur / LOR-AMG; an element-order comparison at Re 1000 (Q5–Q7 likely better per unknown
  at high Re); OIFS (subcycled convection, CFL 2–4) before Re 100k; time-based AMR
  scheduling (start / interval / end) in Case/Parameters for decks (now only in the
  driver); statistics with error bars; same-np restart constrains job chaining. OIFS now
  exists (`oifs` branch: CFL 2 within 1% of IMEX at Re 200, see above). Batch
  scripts: drafted by Claude, submitted by the human (guardrail).
- **AMR stays refinement-only (human 2026-10-08):** derefinement breaks the multistep
  history (refinement transfers every BDF/EXT level exactly; coarsening must project them),
  and single-step schemes are order-bound. For a statistically stationary turbulent wake,
  refine-only converges to resolving the wake region anyway; bound the startup waste with
  `-amr-start` (spin up on the coarse mesh — spin-up only has to reach the state, not
  accurately) and `-amr-end`; a remesh-and-restart (multistep startup ramp) is the fallback.

**`oifs` (branch; human 2026-10-08 "give a shot at implementing the OIFS stuff", 2026-10-09
"you should be running everything with BDF3 with the OIFS stuff"; fast-forward merged and
pushed 2026-10-09, human: "merge and push"):** OIFS (developing.md §5.2), BDF3 by default under OIFS, collocated GLL mass always under
OIFS. Square cylinder Re 200 (same AMR as the base IMEX run, np 4; vs IMEX CFL 0.5 with
the SAME collocated mass: C_D 1.4381, C_L,rms 0.3957, St 0.1564, 9,103 steps):
```
OIFS BDF3, collocated   C_D            C_L,rms         St             steps
  CFL 1                 1.4346 -0.2%   0.3931 -0.7%    0.1565 +0.1%   4,379
  CFL 2                 1.4265 -0.8%   0.3923 -0.9%    0.1565 +0.1%   2,191
  CFL 4                 1.4772 +2.7%   0.4778 +21%     0.1507 -3.6%   1,120
```
**CFL 2 is the sweet spot: within 1% of IMEX in 4.2× fewer steps (~3× less wall time;
an OIFS step costs ~1.3–1.5× an IMEX step).** CFL 4 loses the lift amplitude. History of
the study (each step measured): BDF2 + consistent BDF mass converged in dt to the WRONG
answer (C_D 1.49, C_L,rms 0.45: +3% / +12% at CFL 4 → 0.5) — the mass mismatch (developing.md §5.2),
not the wall treatment (pointwise wall acceleration: no change, reverted), the force
evaluation (wall MMS: = IMEX to 1.5e-3), hanging nodes (`oifs_test` O3) or the outflow
(IMEX classical = directional BITWISE on this domain: no outlet backflow). BDF2's
trajectory time error at CFL 4 was ~4× BDF3's (C_D). OIFS also has no dt sawtooth (steady
dt at every CFL). The collocated mass itself moves IMEX by −0.4% C_D / −1.8% C_L,rms.
DDN under OIFS done (explicit, extrapolated in the BDF step: Braack & Mucha's Table 5.1
to IMEX's values at CFL 2, stable to CFL 8). **Error control under OIFS: not pursued
(human 2026-10-09: "we are always going to need to do some multiple of CFL basically,
so maybe we just skip the error estimator")** — rejected at validation. Open: OIFS cost
per step (substep CFL 0.5 — try 1.0); the fixed-step BDF3-OIFS startup (O(dt²) starter);
whether the case-level default for HPC runs should become OIFS CFL 2.

**Environment refresh (2026-10-09, human):** full desktop refresh with current spack:
MFEM 4.10.1-dev **+MUMPS**, hypre 3.2.0, libCEED 1.0.0, SuiteSparse 7.14 (developing.md
§3). MFEM's head broke 3D parallel anisotropic AMR (PR #5297; reported upstream), so MFEM
is **pinned to `1c19aba`** in both `spack.yaml` files until it is fixed. On the pinned
stack: fast tier 159/159, Python 12/12. The Bridges-2 lock still has to be re-resolved on
that machine (its `spack.yaml` carries the same MUMPS + pin edit).

**`agent-first` (branch, 2026-10-09, human: "an agent first repo where someone can just
point their agent at it"):** `AGENTS.md` front door, short README, `docs/using.md`,
`docs/install/desktop.md`, this file and `developing.md` split out of CLAUDE.md (which now
imports them), specs moved to `docs/design/`. Next on that track: `scripts/doctor.sh`,
validated decks for the benchmarks (needs the drivers' AMR scheduling / averaging in the
library), `summary.json` per run, a generated deck reference, a cold-start agent test.

**Open follow-ups** (none started):
- Outflow condition for the rotational form: do-nothing acts on the Bernoulli head
  ("notoriously bad" for this form, human); the DFG studies used a Dirichlet outflow instead.
- A velocity PC whose iteration count does not grow as σ = β₀/Δt shrinks (Jacobi-PCG's
  does, so CFL 0.5 → 0.7 saves 29% of steps but only ~10–15% of wall time) — e.g. the
  p-multigrid velocity block (spec Part D).
- Warn when the measured CFL creeps above `cfl_target` (the IMEX/EXT3 failure symptom on
  DFG); the rotational form's silent failure shows no such signal.
- If the rotational form is pursued: find the source of its DFG drag error (the form's
  spatial error vs the force evaluation from the Bernoulli head).
- `auto` Schur: compute μ every N steps (it costs ≤1% per step now; low priority).
- Fast tier ~177–197 s, near the 3-min budget; critical path `tgv_nse_test_np2`.
- LOR-AMG with strong grad-div in 3D is still 4–5× its no-grad-div count; BoomerAMG's
  elasticity mode is the standard fix but needs `Ordering::byVDIM` (velocity is byNODES).
- Rotation-aware Schur `auto` thresholds (spec's 20/10) uncalibrated in 3D; default `cc`.
- p-multigrid velocity block (spec Part D Level 1) not built (human: PBJ only for now).
- Skew-symmetric convective form: gated on upstream MFEM (developing.md §5.4 TRAP).
- Not pursued (human 2026-10-07: "not interested in implicit methods at the moment"): a
  CFL-free option (Oseen linearization (w*·∇)u^{n+1}, or converged Newton/Picard steps).
- PENDING GPU VALIDATION list (developing.md §7.5).
- Upstream candidates: the MFEM debug-device BlockVector false positive (developing.md §6).
- 2D NSE TGV pressure-rate oracle never built (human accepted the 3D MMS, 2026-07-24).
- `VectorRotationalConvectionComponentIntegrator::AssembleEA` is untested.
- Convective-form TGV at ν = 0.05 is pre-asymptotic on coarse Q3 meshes (4×4 → 8×8 raises
  the error 0.012 → 0.024 before 16×16 drops it to 0.0027); oracles run at ν = 1.

**Design specs** in `docs/design/` (authoritative design records; developing.md wins where they
disagree): `docs/design/SPEC_cahouet_chabard_mfem.md` (+ `docs/precond_cc.md`),
`docs/design/rotational_convection_pa_spec.md`, `docs/design/rotational_schur_velocity_mg_spec.md`,
`docs/design/amr_spec.md`, `docs/design/vecdivdiv_spec.md`, `docs/install/bridges2.md`.
