# CLAUDE.md

Project instructions for Claude Code. Read this fully before planning any task. Section 0 is
the handoff state — start there.

## 0. Current state and next steps (updated 2026-10-08)

**Branches.** `main` = `b64bd19`: all of `rotational-schur`, merged and pushed 2026-10-07
(rotation-aware Schur PC, DFG 2D-3 driver, `jacobi_pcg` default + grad-div in LOR-AMG, step
control `fixed|error|cfl` with Nek's CFL and the BDF3/EXT3 estimator, case-level CFL control
at 0.5, the debug-device ess-list fix §6). Working branch **`dfg-scheme-comparison`** (NOT
merged; merge/push only when the human says so): DFG app `-mref` nested meshes, per-step
history CSV and `step_wall`; `bench/dfg3/` (pinned two-slot queue runner, sweep generator,
analysis, HTML report generator, raw results of 2026-10-08); the work-precision write-up
(`docs/imex_vs_semi_implicit.md` §7 + summary at its top); **case-level default back to
BDF2/EXT2** (human 2026-10-08; §5.2).

**Scheme comparison concluded (human 2026-10-08): for DFG 2D-3 there is no benefit to the
semi-implicit rotational form** — same CFL limit as IMEX, 7–9% more per step, comparable
lift, 50–70× worse drag; both fail silently past the limit (§5.6, the doc §7). The finest
mesh (M2) was started and stopped unfinished at the human's call.

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
- Skew-symmetric convective form: gated on upstream MFEM (§5.4 TRAP).
- Not pursued (human 2026-10-07: "not interested in implicit methods at the moment"): a
  CFL-free option (Oseen linearization (w*·∇)u^{n+1}, or converged Newton/Picard steps).
- PENDING GPU VALIDATION list (§7.5). `README.md` is stale (still says NSE is gated).
- Upstream candidates: the MFEM debug-device BlockVector false positive (§6).
- 2D NSE TGV pressure-rate oracle never built (human accepted the 3D MMS, 2026-07-24).
- `VectorRotationalConvectionComponentIntegrator::AssembleEA` is untested.
- Convective-form TGV at ν = 0.05 is pre-asymptotic on coarse Q3 meshes (4×4 → 8×8 raises
  the error 0.012 → 0.024 before 16×16 drops it to 0.0027); oracles run at ν = 1.

**Specs in the repo root** (authoritative design records; this file wins where they
disagree): `SPEC_cahouet_chabard_mfem.md` (+ `docs/precond_cc.md`),
`rotational_convection_pa_spec.md`, `rotational_schur_velocity_mg_spec.md`,
`amr_spec.md`, `vecdivdiv_spec.md`, `spack_install_directions.md`.

## 1. Guardrails — hard rules, do not violate

- **NEVER submit to the batch scheduler** (`sbatch`, `srun` beyond the fast-tier rank
  count, `flux`, `bsub`). Draft the batch script and stop; a human submits it.
- **NEVER run above 4 MPI ranks or a 32³ mesh** without explicit approval, and do not
  launch GPU runs without approval. If verifying seems to need a big run, say so and stop.
- **Reap orphaned test/MPI processes.** `mpirun`/`prterun` children survive a killed
  `ctest` and can spin for days. After interrupting ANY run, and at the end of every
  session, run `scripts/reap.sh` (`--dry` to list). Never `pkill -f <pattern>` with a
  pattern that appears in your own command line — it kills your own shell (exit 144); use
  `pkill -x <exe>`.
- **Never build or run project code with the system toolchain** (gcc, cmake, MPI). If
  `scripts/env.sh`'s toolchain assert fails, fix the environment or stop.
- Do not edit `third_party/` or the spack MFEM install.
- **Never push, force-push, or touch `main`** without the human's explicit go-ahead. Work on
  a branch; the human says "merge and push".
- If a change alters numerical results (formulation, quadrature, time scheme,
  preconditioner, grad-div, defaults, null-space/mean handling), say so in the plan and flag
  which baselines need re-blessing. Never weaken a tolerance to make a test pass.
- **Never `rm -rf build/`** (ccache; a 30 s rebuild becomes 15 min). Ask before wiping.
- When unsure whether something is in scope, stop and ask — don't stub it.

## 2. Working protocol

- **Correctness over throughput.** One feature at a time; no speculative scaffolding.
  A feature is done when: (1) its module tests pass at np ∈ {1, 2, 4}; (2) the entire fast
  tier is green; (3) the commit message states what was verified, with what evidence, and
  what is not yet covered; (4) stop for human review before the next feature.
- **Definition of done:** CPU build clean under `-Werror`; full fast tier green; new code
  carries its unit test; baselines unchanged or explicitly re-blessed with a reason;
  `scripts/docs.sh` warning-free; `scripts/style.sh` clean. A CUDA build, when enabled, must
  also compile clean.
- **Standing directive (2.0, human 2026-07-17): every solver default is provisional until
  re-measured under real NSE stepping** (CFL-limited Δt, small ν). Re-decide defaults with
  measured evidence (slow tier / benches), never by carrying Stokes conclusions forward.
  Done so far: CC default (2026-07-17), `jacobi_pcg` velocity PC (2026-10-07).
- Git: `git status`/`git diff` before and after; commit when the fast tier is green;
  commit messages end with the attribution line the harness provides.

## 3. Environment & build

- **Everything comes from spack**, including GCC, cmake, ninja and MPI (on clusters the
  vendor MPI is the one declared external). Per-machine envs:
  `environments/<machine>/spack.yaml` + committed `spack.lock`. Machines: **`desktop`**
  (default) and **`psc_gpu`** (Bridges-2: `br0*`/`w0*` hostnames). `scripts/env.sh`
  resolves the machine (`INCNS_MACHINE` override), activates the env and asserts the
  toolchain resolves inside it. Build/test scripts source it — never activate by hand,
  never hardcode machine paths elsewhere. A new machine = a new `environments/<name>/`;
  existing envs and lockfiles are never edited to make another machine work. Spack builds
  are long — never kill them for being slow, and report failures verbatim instead of
  "fixing" them by pinning MFEM to a release, swapping versions or using a system tool.
- **MFEM tracks `dev`** (commit recorded in the lock). Verify API/capability claims against
  the installed headers (`~/spack/opt/spack/linux-skylake/mfem-develop-*/include/mfem`);
  if an API seems missing, suspect a commit mismatch and flag it.
- **Presets** (`CMakePresets.json`): `cpu` (RelWithDebInfo, ccache — the one that matters),
  `cuda` (optional, GPU machines), `cpu-python` (pybind11 module), `cpu-asan` (ASan+UBSan;
  `scripts/asan.sh`). Build: `scripts/build.sh cpu`. Warnings are errors.
- **Scripts:** `build.sh`, `test.sh`, `dev.sh` (np1 pre-filter), `debug_device.sh`,
  `asan.sh`, `style.sh` (astyle **3.4.11** pinned; refuses other versions — a huge diff
  means the wrong astyle: stop), `docs.sh` (Doxygen, must be warning-free), `reap.sh`,
  `install-hooks.sh` (git hooks call build/test), plus TGV/MMS timing helpers and drafted
  `.sbatch` files.
- **GPU: allocate ≥ 1 GPU per MPI rank.** `ConfigureDevice` binds rank % device count;
  ranks sharing a card serialize (11× slowdown at np2, OOM at np4 measured on one H100).
  A fast-tier GPU sweep wants 4 GPUs, e.g. `interact -p GPU-shared
  --gres=gpu:h100-80:4 -t 4:00:00`. Runtime backend: `Parameters.device` or env
  `INCNS_DEVICE=cpu|cuda|debug` (env wins).

## 4. What the code is

A time-dependent high-order FEM solver for incompressible Navier–Stokes (and unsteady
Stokes) on MFEM (parallel `ParMesh`/`ParGridFunction`), **coupled and monolithic**:
velocity and pressure are solved together as one saddle-point system per step — no
splitting, no projection. It builds as a **library** (`libincns`); a case is a thin driver
(C++, Python, or a YAML deck). **The solver core is never edited to run a new case** — a
case that needs a source change exposes a missing library feature. Production DNS/LES runs
target DOE systems, not this desktop.

```
src/
  mesh/        periodic_box (box factory: periodic flags, per-direction stretching),
               case_mesh (THE factory: MakeCaseMesh/PartitionMesh; NC-ready with AMR),
               cylinder_channel (DFG geometry), mesh_size_coefficient (gamma = c h_K)
  amr/         amr_parameters, gradient_indicator (nvcc), refinement_marker, mesh_adapter
               (refine/rebalance + exact transfer, RefinementRecord), history_projection
  spaces/      mixed_spaces (velocity/pressure ParFESpaces, block offsets)
  operators/   stokes_operator (blocks, PA; grad-div; LOR source; rotation term N),
               convection (dealiased (u.grad)u), rotational_convection (N(w*), nvcc),
               grad_div_integrator (sum-factorized, nvcc, upstream-bound), pressure_schur
               (mass Schur), block_preconditioner (mass-path block PC)
  precond/     cahouet_chabard (CC Schur PC + config), mixed_poisson_op (B M_v^-1 B^T,
               never assembled), mass_inverse, block_stokes_pc (Diag/Tri shapes),
               nullspace, lp_surrogate (LOR-AMG L_p), point_block_jacobi (nvcc),
               rotational_schur (CC/tensor/auto + RotationNumber, nvcc), viscous_ratio
  solver/      stokes_solver (one saddle-point solve; SolveStats, MonitoredSolver),
               velocity_preconditioner (enums), case (THE driver-facing class)
  time/        multistep_coeffs, time_integrator (in-repo BDF; NOT mfem::ODESolver),
               adaptive_controller, integrator_state, cfl (nvcc)
  post/        pressure_mean, kinetic_head (nvcc), body_force (lift/drag), diagnostics,
               output (ParaView), checkpoint
  quadrature/  rule_book (OWNS all rules; order + 1D family per operator)
  bc/          boundary_conditions (Dirichlet/no-slip/outflow per attribute; null-space detection)
  config/      parameters (+ YAML), nondimensionalization, initial_conditions
  util/        profiler (INCNS_PROFILE), device
  exact/       tgv2d.hpp
apps/          run_case (YAML driver), taylor_green, dfg_cylinder (DFG 2D-1/2D-2/2D-3),
               unsteady_mms_3d, hello_mpi
bench/         bench_graddiv, bench_rotation_pc, bench_velocity_pc (manual, not ctest);
               dfg3/ (DFG 2D-3 scheme-comparison harness + raw results)
cases/         tgv2d_stokes.yaml, stokes_mms.yaml, channel_noslip.yaml
python/        bindings.cpp (module incns), incns/__init__.py (@incns.field)
examples/python/stokes_ex/   run.py, run_yaml.py, run_channel.py, run_amr.py (also py tests)
docs/          precond_cc.md (CC conventions), imex_vs_semi_implicit.md (DFG 2D-3 study)
test/          gtest MPI tests (np 1/2/4) + baselines.yaml
```

**Public surface.** `incns::Parameters` (fields or `LoadYAML(path)`; `Normalize()` before
building the mesh), `MakeCaseMesh(p)`, one `incns::Case(ParMesh&, const Parameters&)` that
dispatches on `equation: stokes|navier_stokes` (never a class per physics),
`BoundaryConditions` (`AddVelocityDirichlet(attr, coeff)`, `AddNoSlip(attr)`,
`AddOutflow(attr)`; periodicity is mesh-level; an empty set on a periodic mesh = fully
periodic), `SetInitialVelocity`, `SetForcing`, `Run()` or `Step()/Time()/Done()`,
`Velocity()/Pressure()`, forces (§5.6). NSE is not a separate solver: convection is an
additive term inside `StokesTimeIntegrator`.
- **Nondimensionalization** (`nondim:`): dimensionless (records Re = 1/ν) or dimensional
  (L_ref, U_ref; `Normalize()` rescales lengths, times, ν, tolerances, force references).
  Named ICs (`initial_velocity: zero|taylor_green_2d`) are nondimensional only; dimensional
  analytic data goes through `WrapDimensionalVelocity/Forcing`.
- **Diagnostics** (`output.diagnostics`): kinetic energy, dissipation rate, ‖∇·u‖ to a CSV;
  also exposed as scalars to Python.
- **Checkpoint/restart** (`checkpoint:`, `restart:`): rolling checkpoints of the marching
  state (`IntegratorState`); same-np restart, bitwise (adapted meshes via the refinement log).

## 5. Numerical design — decisions that constrain the code

### 5.1 Discretization

- **Quads/hexes only**, continuous H1 `Q_k` spaces; reject simplices at load time.
- **Orders are runtime options**; default Taylor–Hood **Q3/Q2**. Equal orders need a
  pressure stabilization that does not exist — warn if `k_u == k_p` (grad-div is not one).
- **BCs:** velocity Dirichlet (eliminated), outflow/do-nothing (natural), periodic
  (`MakePeriodic`). **Dirichlet data may be time-dependent:** `SetTime(t)` + re-elimination
  every step (stale elimination silently drops temporal order; only the unsteady MMS sees it).
- **Pressure null space** exists iff no boundary is outflow (fully periodic or fully
  Dirichlet); detect it from the BC set. Remove it by **orthogonalizing against the constant
  mode — never pin a pressure dof**, ever. Output pressure is shifted to zero
  **mass-weighted** mean (never a nodal average).
- **Partial assembly** everywhere; GPU path kept open. Build kernels from PA integrators and
  device-aware `Vector` ops (see §6 on `forall`).
- **Quadrature via the RuleBook only.** Gauss–Legendre by default; the RuleBook owns
  per-family `IntegrationRules` and hands out rules by (order, 1D family). Option: collocated
  **GLL mass** (k+1 GLL points, diagonal mass; requires the GLL nodal basis — verify it).
  Never hardcode a rule at a call site or pass a temporary rule's address.
- **Dealiasing is mandatory** for the nonlinear term (both forms): Gauss–Legendre order 3k
  from the RuleBook, never GLL-collocated or default rules (MFEM's `miniapps/fluids/navier`
  collocates at 2k−1 — rejected; use it for structure only).
- **Grad-div** γ(∇·u, ∇·v) on the velocity block: **γ = c_gd·h_K per element** (human
  2026-07-11; `grad_div_scale: OrderH`), or γ = c_gd·ν (`OrderNu`, 2026-07-17). Integrator:
  the in-repo sum-factorized `GradDivIntegrator` (elmat-identical to
  `ElasticityIntegrator(λ=γ, μ=0)`). **γ never enters the Schur block.** With `OrderNu`,
  γ is not negligible there — set `cc.nu_pc = ν(1+c_gd)` explicitly if wanted.
  **γ = h is NOT small compared with ν at low viscosity** (γ/ν ≈ 60 at h = 1/16, ν = 1e-3):
  it dominates the velocity stiffness — see the LOR-AMG fix in §5.3.
- **Output:** ParaView with `SetHighOrderOutput(true)` and LOD ≥ k_u, or high-order fields
  render as mush that looks like a solver bug.

### 5.2 Time integration (in-repo; never `mfem::ODESolver`)

- **BDF2** for the implicit part, variable-step coefficients recomputed every step from the
  actual step ratios (reusing uniform coefficients under varying Δt is the #1 adaptive bug).
  First step: trapezoidal starter (viscous term split half/half).
- **Step control** (`time.step_control` → `Parameters::step_control`, enum `StepControl`;
  human request 2026-10-07 to mimic Nek). **Case-level default `cfl`** (human 2026-10-07);
  the low-level `TimeIntegratorOptions` keep fixed steps (integrator unit tests pin what
  they test — the `schur` precedent). The older `time.adaptive: true|false` = error|fixed
  (giving both keys aborts).
  - `cfl` (default): **dt = `cfl_target` / c(uⁿ) before every step** (one solve per step):
    shrinks at once, grows ≤ ×1.2 per step and only once the target allows 5% more (the
    solver refreshes rarely in quasi-steady flow), `time.dt_max` cap, lands exactly on
    t_final; `time.dt` is the first step (capped by the target). **Stokes has no
    convective CFL and steps at `time.dt` under `cfl`** (`Parameters::CflSteps()`).
    `cfl_target` default **0.5** = Nek's `targetCFL`: the step as a multiple of the CFL = 1
    step. Explicit convection bounds it below ~1 (DFG 2D-3: unstable at Nek-CFL ~0.8–1.1);
    Nek's multiples of 2–4 come from OIFS (sub-stepped characteristics), which this code
    does not have. `cfl_test` C3 pins the controller; restart (`checkpoint_test`) and AMR
    no-op events (`amr_event_test` E2) reproduce CFL-mode runs exactly.
  - `fixed`: dt constant; with `cfl_max > 0` an abort if exceeded. NSE tests that are not
    about step control pin it (`p.step_control = StepControl::Fixed`).
  - `error`: BDF2−BDF3 LTE control, below. Two solves per step.
- **Extrapolation order `time.ext_order: 2|3`** of the nonlinear term (IMEX convection and
  the rotational form's w*), ramped EXT1 → EXT2 → EXT3 from the available history (three
  levels kept for EXT3). Accuracy: on the NSE MMS (quadratic in t, so BDF2 is exact) EXT3
  shows order 2.94 → 2.98 vs EXT2's 1.98 → 1.99, error 70× lower (`nse_mms_test`
  Ext3TemporalOrder2D). **Case-level default EXT2 (human 2026-10-08, after one day at EXT3)**:
  on DFG 2D-3 EXT3 fails *silently* at CFL 0.7 in both schemes where EXT2 is accurate, and
  buys no accuracy there (the error is spatial) — `docs/imex_vs_semi_implicit.md` §7.
  `TimeIntegratorOptions::ext_order` is 2 too. Stability is regime-dependent.
  Stable radius |λΔt| of the explicit term along rays at an angle from the negative real
  axis (90° = imaginary axis = undamped advection):
  ```
              90°    85°   80°   70°   45°   0°
  BDF2/EXT2   ~0    0.48  0.60  0.78  1.09  1.33
  BDF2/EXT3   0.63  0.62  0.62  0.60  0.58  0.57
  BDF3/EXT3   0.63  0.66  0.68  0.72  0.83  0.95
  ```
  EXT3 wins only within ~10° of the imaginary axis (nearly undamped advection, high-Re DNS —
  the reason NekRS uses it); for damped spectra BDF2/EXT2's region is larger and BDF2/EXT3's
  is the smallest. DFG 2D-3 (Re ≤ 100, grad-div) confirms it: IMEX at Δt 0.002 blew up at
  t = 2.36 with EXT3 vs 3.28 with EXT2; 0.0025: 2.02 vs 2.80; the rotational scheme was
  also worse. BDF3/EXT3 dominates BDF2/EXT3 everywhere, but BDF3 is test-/estimator-only
  as an advancing scheme (a human decision to change).
- **IMEX convective form:** N(u) = (u·∇)u evaluated on the history and extrapolated (EXT2,
  variable-step weights; EXT1 on the first step), on the RHS. **Landmine:** the trapezoidal
  starter builds its RHS inline, not via `AssembleBdfRhs`, so it needs its own
  `SubtractConvection` — without it step 1 silently solves Stokes.
- **BDF3** exists only for the adaptive error estimate (and is marched in tests to prove it
  is 3rd order); never the advancing solution.
- **Error-controlled stepping** (`step_control: error`): an EMBEDDED pair — the step
  advances with BDF2 (+ EXT of `ext_order`), and the auxiliary candidate is a genuine
  **BDF3/EXT3** solution of the same step (its own EXT3 convection / its own w* for the
  rotational form). LTE = their difference on **velocity only** (pressure is the algebraic
  DAE variable), PI controller, accept when ‖LTE‖ ≤ atol + rtol·‖u‖ (global norms, not
  per-dof WRMS), rejection + retry, no local extrapolation, step history recorded.
  **Fixed 2026-10-07 (human):** the BDF3 candidate used EXT2 like the advancing step, so the
  convective splitting error — the dominant one at CFL-limited NSE steps — cancelled in the
  difference and the estimator was blind to it (NSE MMS: error 7× the tolerance; 100×
  tighter tol bought only 2× the steps). `nse_mms_test` AdaptiveEstimatorSeesConvection
  guards it (fails on the old code). Adaptive NSE results changed (smaller steps).
- **A Δt change is a refresh, not a rebuild** (`StokesSolver::Refresh(c0)` → only the fused
  momentum block reassembles; M, νK, B, Schur structure persist). `solver.amg_reuse`
  freezes a LOR-AMG hierarchy at c0_ref·M + νK (+ grad-div) across Δt.
- **CFL number = Nek5000's definition** (`time/cfl`, human decision 2026-10-07: "whatever is
  most accurate"): c = max over the GLL nodes of Σ_d |(J⁻¹u)_d| / Δξ_d(node), Δξ the local
  reference node spacing (one-sided at element ends, half central difference inside — Nek's
  `getdr`), so Nek's CFL targets transfer. It replaced a uniform k²·max Σ|J⁻¹u| at GL points
  that was ~2.5× larger at element edges (Q3: 9 vs 3.62 per unit |u|/h) and ~3.3× inside;
  CFL numbers recorded before 2026-10-07 (the base-mesh DFG study, the 2D-1 runs) are in
  that old measure — divide by ~2.5. `cfl_test` C1 checks the spacings against Nek's by
  hand.
- **CFL ceiling `time.cfl_max`** (0 = off):
  adaptive: dt ≤ cfl_max/c; fixed step: abort at setup/AMR events if c·dt > cfl_max (dt never
  changes silently). **Applies to BOTH convective forms** (2026-10-07): the semi-implicit
  rotational form transports vorticity explicitly and has the same threshold (§5.6).
  On AMR meshes the measured CFL number is a poor stability predictor (the max sits in
  viscous wall cells that implicit viscosity stabilizes).
- Rotational form: see §5.4 (lagged vorticity in the implicit block).

### 5.3 Linear solver

- **Outer: FGMRES on the monolithic system, always** (no MINRES, no Schur-only production
  Krylov). Block preconditioner; FGMRES is required because inner PCs are nonlinear.
- **Schur block** (`solver.schur`): **Cahouet–Chabard is the case-level default** (human
  2026-07-17); `mass` (ν M_p⁻¹, Δt-sensitive) remains, and is the LOW-LEVEL
  `StokesSolverOptions` default so solver unit tests pin what they test.
  CC = ν M_p⁻¹ + σ (B M_v⁻¹ Bᵀ)⁻¹ (consistent form, Creff–Guermond), inner CG with a FIXED
  iteration count (10) preconditioned by one symmetric LOR-AMG V-cycle on L_p. **CC
  invariants (hard rules):** no ε-shift; B M_v⁻¹ Bᵀ is matrix-free composition, never
  assembled; diagonal mass inverses are direct fused multiplies; internal pressure
  p̃ = −p_physical with exactly one sign flip at output; the inner CG operator is fixed
  linear and its AMG relaxation symmetric. `laplacian_legacy` is a comparison mode only.
  Knobs: `solver.block_shape: diag|lower|upper` (upper default), `n_inner`, `lp_vcycles`,
  `pc_quadrature: inherit|gll_collocated` (its measured 20–31% win is Stokes data — re-measure
  under NSE per §2). Details: `docs/precond_cc.md`.
- **Velocity block, CC path** (`solver.a_pc`; `cc.a_pc`): **case-level default
  `jacobi_pcg`** (human 2026-10-07): CG + Jacobi to `a_pcg_rtol` 1e-2, ≤ `a_pcg_max_iter` 50.
  Alternatives `jacobi_chebyshev` (order 4) and `loramg` (one LOR-BoomerAMG V-cycle; the
  low-level `CahouetChabardConfig` default, kept for the solver tests' baselines).
  `solver.rotation_lor: true` implies `loramg`. Mass path: `solver.preconditioner:
  jacobi|loramg` (Jacobi default). Evidence (`bench/bench_velocity_pc`, enclosed box,
  Q3/Q2, ν = 1e-3, outer FGMRES to 1e-8, outer iterations; PCG inner per outer in parens):
  ```
                      viscous-dominated (v-hat >= 2)          mass-dominated (v-hat <= 0.1)
  no grad-div  2D 32^2: loramg 26-38   cheb 106-260  pcg 14-20 (25-50)  loramg 22-30 cheb 6-7  pcg 5-7 (3)
               3D 6^3:  loramg 45-52   cheb 49-60    pcg 27-28 (13-14)  loramg 47-60 cheb 8-10 pcg 6-9 (4)
  gamma = h    2D 32^2: loramg 73-81   cheb 260-494  pcg 50-64 (50)     loramg 21-40 cheb 6-40 pcg 6-24
               3D 6^3:  loramg 173-255 cheb 171-203  pcg 65-93 (50)     loramg 38-78 cheb 8-75 pcg 6-41
  ```
  At the mass-dominated steps of CFL-limited NSE, PCG ≈ Chebyshev and 4–8× faster than
  LOR-AMG; viscous-dominated (steady, large steps), LOR-AMG is mesh-independent and wins —
  set `a_pc: loramg` there. PCG is the default because it is never the worst.
- **LOR-AMG details.** BoomerAMG on a Q1-on-GLL-nodes rediscretization (plain high-order
  AMG coarsens poorly); `SetSystemsOptions(dim, byNODES)`. MFEM's batched LOR handles only
  scalar mass/diffusion, so bessemer's vector forms use the legacy LOR path (integrators
  reused on the LOR mesh with a GLL order-1 rule). **Grad-div is in the LOR operator since
  2026-10-07** (it was omitted as "negligible"; that cost 10–30× iterations): its γ is
  `MeshSizeCoefficient(mesh, c, k^dim)`, returning the PARENT element's c·h_K for LOR
  element e (MFEM's `MakeRefined` stores k^dim children per element in order, parent =
  e / k^dim). `lor_grad_div_test` pins it (parent mapping on a non-uniform mesh; LOR
  operator equals an elasticity(λ = γ, μ = 0) reference to 1e-12; iteration guard).
- **Δt-refresh of the velocity PC** rebuilds it unless frozen (`amg_reuse`).

### 5.4 Convective forms (`physics.convective_form`)

| Form | Term | Status |
|---|---|---|
| `convective` (default) | (u·∇)u, explicit (IMEX) | done |
| `rotational` | (∇×w*)×u^{n+1} implicit, w* extrapolated; pressure = Bernoulli head | done 2026-10-02 |
| skew-symmetric | ½(u·∇)u + ½∇·(u⊗u) | **planned, gated on upstream MFEM** |

**Rotational (semi-implicit), as built** (`rotational_convection_pa_spec.md`, human
2026-10-02):
- `VectorRotationalConvectionIntegrator` (PA, quads/hexes, skew: Nᵀ = −N, zero diagonal;
  `UpdateVorticity()` per step without reassembly; `AddNodalSkewPA`; legacy
  `AssembleElementMatrix` for LOR; per-component `AssembleEA` for the GPU route). N lives in
  its own form: `Momentum()` stays symmetric (what Chebyshev/PCG/LOR-AMG see), the outer
  FGMRES applies `FullMomentum()` = Momentum (DIAG_ONE) + N (**DIAG_ZERO** on essential dofs
  — two DIAG_ONE operators halve inhomogeneous Dirichlet data). Elimination subtracts N u_D.
- w* by EXT2 (EXT1 on the starter, where N is split half implicit / half explicit).
- **Dealiased at the GL order-3p rule only** (human 2026-10-05). Kernel specializations
  cover exactly that rule for p = 1..5: (D1D, Q1D) = (2,2),(3,4),(4,5),(5,7),(6,8)
  (Q1D = (3p+2)/2); anything else hits a generic fallback (correct, ~2.5× slower on CPU,
  likely far worse on GPU). Every test/bench using N runs at the solver's rule; check with
  `MFEM_REPORT_KERNELS=1` (no `RotConv` fallback). `rotational_convection_test` and
  `point_block_jacobi_test` hard-code 3p — change them with the specialization list.
- **Pressure:** the solve yields P = p + ½|u|²; `Pressure()` returns static
  p = P − I(½|u|²), computed every step on the device by `post/kinetic_head` (not lazily —
  `OutputWriter` holds the reference). Do-nothing outflow acts on P (a modelling difference,
  immaterial for DFG). **Use grad-div with the rotational form** when forces/pressure
  matter: without it its accuracy is markedly worse (DFG 2D-1 c_D error 9.4e-4 vs 1.0e-4;
  the known rotation-form loss, Layton et al. JCP 2009).
- **Velocity PC with N** (`solver.rotation_pc`): `symmetric` (default; the a_pc PC on the
  symmetric part only), `pbj_only`, **`pbj_krylov`** (GMRES(20), rtol 1e-2, ≤ 30 its,
  point-block-Jacobi preconditioned — the recommended one). `PointBlockJacobi` inverts the
  nodal dim×dim blocks diag(d) + [s]× in closed form, one fused kernel on the true-dof
  layout. `rotation_lor: true` (N inside LOR-AMG, re-set-up each step) — **measured, leave
  off**: it helps only near |ω|Δt ≈ 1 and breaks AMG where rotation dominates
  (`bench/bench_rotation_pc`).
- **Rotation-aware Schur preconditioner** (`solver.rotation_schur: cc|tensor|auto`, or a
  map with `mode`, `criterion: max_mu|volume_fraction`, `mu_on`, `mu_off`, `vol_on`,
  `vol_off`, `inner_iterations`; human 2026-10-06, spec
  `rotational_schur_velocity_mg_spec.md` minus p-multigrid). CC ignores N and degrades with
  the rotation number μ = |αω*|/σ (3D: 38 → 309 → 878 outer iterations at μ 1.8/18/59);
  the tensor mode uses L_T = −∇·(T∇), T = (σI + [αω*]×)⁻¹ (Olshanskii), applied by a FIXED
  number of FGMRES iterations (**default 10**, measured; the spec's 3 assumed exact
  Laplacian solves) preconditioned by CC's LOR-AMG L_p V-cycle, plus CC's ν M_p⁻¹. Inner
  solve runs on σL_T with the UNSCALED V-cycle and multiplies by σ (keeps constrained rows
  consistent with CC). 2D and 3D. **Default `cc`** (bitwise today's path). **Orientation is
  load-bearing** (T, not Tᵀ: with Tᵀ the 3D μ 1.8 count is 137 vs 37); the PA path calls
  `Project(qf, transpose=true)` — honor the flag. MFEM traps: never put the tensor in a LOR
  form (scalar coefficients only, silently 1); Krylov solvers forward `SetOperator` to
  their PC (wrap in a SetOperator-proof solver); PA nonsymmetric diffusion's
  `MultTranspose` aborts. 2D CC is far more robust to rotation than 3D (99 vs 309 at μ ≈ 16):
  the tensor's payoff is mainly 3D. In DFG 2D-3 μ stayed < 0.4, so `auto` never switched.
- **Diagnostics:** `RotationNumber` (max μ, volume fraction above a threshold; device),
  `ViscousRatioDiagnostic` v̂ = ν/(σ(h/p)²) (c_p matches the spec: 20.22/29.53/40.92 for
  p = 3/4/5), `SolveStats` + `MonitoredSolver` (counts/timings of velocity and Schur PCs).
  `solver.rotation_log_interval: N` → `<output.path>/<output.name>_rotation.csv`.
- **The TGV cannot see errors in N** (its convection is a pure gradient); the MMS pins N.

**Skew-symmetric — TRAP, do not "just switch the integrator".** MFEM's
`SkewSymmetricVectorConvectionNLFIntegrator` (and the Convective one) override only
`AssembleElementGrad`; the residual/PA apply is the base class's convective one. Since
bessemer only calls the residual, swapping it in would change *nothing*, silently. Gate:
re-check those overrides when the upstream PRs land. Plan: `ConvectiveForm::SkewSymmetric`,
`Convection` picks the integrator, same 3k rule, LHS-signed `Mult`, IMEX side unchanged.
Tests to write with it: equivalence of all forms on an analytically divergence-free field;
**discrete energy conservation at ν = 0** (the only test that catches a silent convective
residual); MMS per form with forcing verified by finite differences (hand algebra was wrong
by 3× once). A default change needs measured NSE evidence (§2).

### 5.5 Adaptive mesh refinement (`amr:`; `amr_spec.md`; human 2026-10-06)

Refinement only (no derefinement), isotropic or anisotropic, Stokes and both NSE forms. Off
by default (bitwise unchanged).
- `MakeCaseMesh` is the only mesh factory. AMR on: `EnsureNCMesh()` BEFORE partitioning and
  METIS's partition passed explicitly. Hanging nodes via MFEM's P; every operator/PC works on
  NC meshes (`nc_stokes_test`: steady MMS exact on every solver path).
- **Indicator** G_{K,d} = RMS of ∂u/∂ξ_d over K (anisotropy falls out); η_K = |G_K|.
  **Marking:** θ·max η or an absolute tolerance; directions with G_d ≥ aniso_ratio·max G;
  `min_size`; `max_elements` cap (largest η first); 3D np > 1: anisotropic conflicts
  upgraded to XYZ (`ParMesh::AnisotropicConflict`); `nc_limit` 1.
- **Event = in-memory checkpoint + restart** (`Case::Adapt()`, every `amr.interval`
  accepted steps, never in the startup ramp, before output/checkpoint): export
  `IntegratorState`, destroy integrator/output before the mesh changes, refine/rebalance
  with EXACT transfer, optional `project_history` (div-free projection of the history,
  default on), rebuild, import. `amr.initial_passes` refine on the analytic IC.
- **Checkpoints of adapted runs replay the refinement log** (`RefinementRecord` per rank)
  on the initial mesh at the same np — not ParPrint (dof numbering would not match).
  Restart matches the uninterrupted run bitwise at np 1/2/4.
- Adapted meshes can differ between np = 1 and np > 1 (conflict handling, NC limit):
  cross-np comparisons on adapted meshes are not exact.
- **Refinement-only + a cap spends the budget early** (DFG 2D-3: θ = 0.3 hit 800 elements by
  t = 0.5); use a high θ (0.85) so refinement follows the evolving flow.
- **Forces right after an event:** the rebuilt integrator has no step; `Case` caches the
  pre-event force (fixed 2026-10-06; `body_force_test` covers it).

### 5.6 Lift/drag and the DFG benchmarks

- **Lift/drag** (`post/body_force`, deck `forces:` — `enabled`, `attributes`,
  `reference_velocity`, `reference_area` (nondimensionalized), `interval` (CSV
  `<path>/<name>_forces.csv`; 0 = no log, tests use 0)). John's volume-integral formulation:
  F_i = −r·v_i, r = A u + N u − Bᵀp − b the **residual of the step that produced (u, p)**
  (`StokesTimeIntegrator::MomentumResidual`, unconstrained operators, raw RHS), v_i = e_i at
  the body's velocity nodes. Independent of v_i's interior extension (tested). The
  trapezoidal starter's force is not the force at t¹ (time-averaged pressure).
  `Case::SetForceBody(attrs)`, `BodyForceVector()`, `ForceCoefficients()`.
- **DFG geometry** `mesh/cylinder_channel` (O-grid ring, exact transfinite curved nodes;
  default 208 Q3 elements). Driver `apps/dfg_cylinder`: `-c 1` 2D-1 (steady Re 20),
  `-c 2` 2D-2 (expensive tier), `-c 3` 2D-3 (U(t) = 1.5 sin(πt/8), t ∈ [0, 8], fixed steps,
  RESULT line with errors vs John 2004: c_D,max 2.950921575 at 3.93625, c_L,max 0.47795 at
  5.693125, Δp(8) = −0.1116; per-step `<out>/dfg_2d3_history.csv`; `step_wall` = time in
  `Case::Step()` only). Options: `-rot -pbj -schur -rlog -gd -apc -ext -cflt -mref -at -maxe
  -theta -aniso -rtol -cfl -dout`. Study harness: `bench/dfg3/` (`run_queue.sh` pins two
  np-4 runs to disjoint physical cores at constant load; `make_sweep.sh`, `analyze.py`,
  `report.py`).
- **2D-1 measured** (208 elements, t = 8, rel. errors c_D / c_L / Δp): convective 1.05e-4 /
  2.79e-3 / 4.06e-3; rotational 9.45e-4 / 5.32e-2 / 5.72e-3; rotational + c_gd 1 1.03e-4 /
  1.13e-2 / 5.74e-3. Slow-tier test `dfg_cylinder_slow_test` (np 4, ~9 min) pins the first
  and third (ceilings in `baselines.yaml` `dfg_2d1`).
- **2D-3 studies (`docs/imex_vs_semi_implicit.md` §6–7; concluded 2026-10-08, human: "for
  this problem there is no benefit to the rotational form")**. Work-precision study (§7):
  Dirichlet outflow for both (do-nothing is a different BC for the rotational form; bias vs
  John's values ~1e-8), nested meshes `-mref` M0/M1/M2 (208/832/3328 cells), CFL control,
  tooling + raw results in `bench/dfg3/`. On M1: both schemes accurate to CFL 0.7 with EXT2
  and degraded at 0.9 — **the rotational scheme cannot step further than IMEX**; equal
  steps and iterations at equal CFL, rotational +7–9% per step; lift comparable, **drag
  peak 50–70× worse for the rotational form (2.2e-4 vs 3–4e-6, spatial)**; EXT3 fails
  silently at 0.7 in both (IMEX: measured CFL creeps above the target; rotational: no
  signal). CFL 0.5 → 0.7 saves 29% of steps but only ~10–15% wall time (Jacobi-PCG
  iterations per step grow as σ shrinks). M2 (~6× M1 per step) was stopped unfinished.
  Earlier findings (§6):
  - **The lagged-vorticity rotational scheme is CFL-limited exactly like IMEX.** The curl of
    ω*×u^{n+1} is (u^{n+1}·∇)ω*: vorticity *transport* is explicit (EXT2). Base mesh: both
    fine at Δt = 0.0015, both unstable at 0.002 (same onset time; deviation ×~1.9 per step).
    IMEX blows up; the rotational scheme **silently** finishes with c_D,max off 34–66%
    (energy stability bounds the norm, not the error). Never sell it as "no CFL limit".
  - At equal stable Δt both schemes cost and deliver the same (base: 134 vs 138 s, c_D,max
    errors 2.2e-3 vs 2.4e-3, temporally converged); the base mesh's c_L error (7%) is
    spatial — AMR (803 elements) brings it to 2.8% and c_D,max to 1.9e-5.
  - An experiment with Picard sweeps (re-solving each step with w* = u^{n+1}) extended the
    threshold ~×1.5 per sweep but each sweep is a full solve — never cheaper than the plain
    schemes at their stable Δt; the option was removed (2026-10-07).
  - IMEX with LOR-AMG at CFL-limited steps was 3× slower than Jacobi–Chebyshev → §5.3.

### 5.7 Python interface (`cpu-python` preset, module `incns`)

Python is a **job driver only**: configure parameters + analytic IC/BC/forcing and run, no
C++ recompile. **Hard principles:** never PyMFEM; no MFEM type crosses the boundary (the
Python `Case` builds its own mesh via `MakeCaseMesh`); pure SPMD (`mpirun -np N python …`;
all MPI below the binding line; no mpi4py; `incns.rank()/size()/on_root()`); **no bulk
solution data in Python** — numerics and diagnostics are C++ routines surfaced as scalars.
Surface: `incns.Parameters` (all fields incl. `mesh.box(...)`, `amr`, `forces`, `cc`,
`rotation_schur`, `step_control` (enum `StepControl`), `cfl_target`, `dt_max`, `ext_order`,
`from_yaml`, `normalize`; enums
re-exported from `incns/__init__.py` — add new ones there, the AMR work forgot
`AmrThreshold` until 2026-10-07); `incns.Case(p)` / `Case.from_yaml(path)`;
`set_initial_velocity`, `set_forcing`, `set_dirichlet_field(group, f)`,
`velocity_dirichlet(attrs, f)`, `outflow`, `no_slip`, `face/faces/all_faces` (box names),
`run/step`, scalars (`time`, `time_dimensional`, `done`, `step_count`, `iterations`,
`element_count`, `velocity_l2_error`, `kinetic_energy`, `dissipation_rate`,
`divergence_norm`), forces (`set_force_body`, `forces`, `force_coefficients`). Fields:
Tier 1 plain callables `f(x, t)` (slow, GIL) or Tier 2 `@incns.field(dim)` (numba `cfunc`,
C ABI `void(const double* x, double t, double* out)`, no GIL; `run/step` release the GIL).
Tests `py_stokes_ex`, `py_stokes_ex_yaml`, `py_channel_noslip`, `py_stokes_amr` (np 1/2/4).

## 6. Coding conventions

- C++17, MFEM style, RAII, no raw `new`/`delete` in new code (MFEM ownership idioms aside).
  **`#pragma once`** in every header. **Doxygen on every entity, private members included**
  (`///` / `///<`; one declaration per line — a trailing `///<` documents only the last
  name); `scripts/docs.sh` must print no warnings. `.cpp` comments stay plain `//`.
- Modular: one concern per file, small classes, a unit test with every new module.
  Runtime options, never `#ifdef`. Diagnostics print on rank 0 only; every reduction global.
- **Profile hot paths** with nested `INCNS_PROFILE("name")` scopes.
- **NEVER hand-write `mfem::forall` outside the nvcc TU list** in `src/CMakeLists.txt`
  (`grad_div_integrator.cpp`, `rotational_convection.cpp`, `point_block_jacobi.cpp`,
  `kinetic_head.cpp`, `amr/gradient_indicator.cpp`, `time/cfl.cpp`,
  `precond/rotational_schur.cpp`). Elsewhere `MFEM_HOST_DEVICE` is empty and `forall`'s CUDA
  dispatch is preprocessed out: a host loop over device pointers that compiles clean,
  passes every CPU and debug-device test, and segfaults on a GPU (38/62 tests on an H100,
  2026-07-21). Prefer device-aware `Vector`/`Operator` ops (`Set`, `Add`, `/=`, `Max`,
  `InnerProduct`, PA integrators). A new kernel TU goes on the list with `LANGUAGE CUDA` and
  opens with `#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)` / `#error`. Exceptions
  without the guard (human decisions): `grad_div_integrator.cpp` (upstream-bound) and
  `point_block_jacobi.cpp` — the CMake entry is their only protection; never drop it.
  Kernels go in a NAMED namespace (nvcc rejects extended lambdas in internal-linkage scope).
- **Device-resident vectors in the solve path:** `UseDevice(true)`, `BlockVector`s with
  `Device::GetMemoryType()` (no-ops on CPU). The **debug device** (`INCNS_DEVICE=debug`,
  `scripts/debug_device.sh`) faults on unsynced host access; sweep GREEN 94/94 at np 1/2
  (2026-10-06; refined-periodic NC solves skip there) — a failure is a regression. **It
  cannot catch a host-compiled `forall`** — only a real GPU run proves the GPU path.
- **Never hand MFEM device paths a hypre-malloc'd buffer:** use
  `form.ParallelAssemble(Vector&)`, never `unique_ptr<HypreParVector>(form.ParallelAssemble())`
  (page-granular mprotect hits heap neighbours → intermittent faults on the debug device).
- **Preconditioners borrow `StokesOperator::EssentialTrueDofs()`, never the
  `BoundaryConditions` list** (2026-10-07): MFEM's Jacobi/Chebyshev smoothers keep a pointer
  and read the list on the device; on the debug device that protects the host mirror, and
  the BC list's in-place AMR rebuild (`MarkerToList`: `SetSize(0)` then a size-0
  `HostWrite`) then faults. Hit by the `jacobi_pcg` default in `amr_event_test` E1.
- **MFEM debug-device false positive (2026-10-06):** a host-valid `BlockVector` whose ≥ 1-page
  block is read on the device through the alias, then the whole vector read on the device →
  "illegal memory access" (pure-MFEM repro: 400 doubles pass, 1032 fail; ASan clean). Hit by
  refined PERIODIC NC meshes; such tests skip on the debug device only
  (`amr_test::DebugDeviceSkipsPeriodicNcSolves`). Upstream-report candidate.
- **RuleBook lifetime:** it owns every rule and must outlive every integrator AND the mesh's
  rule-keyed geometric-factor cache (a stale key at a reused address gave NaNs):
  `Case::~Case` calls `mesh.DeleteGeometricFactors()`; in tests declare the RuleBook
  BEFORE the mesh. `QuadratureInterpolator`s are cached per FES by rule address too.
- **MFEM keeps pointers to essential-dof lists** (`ConstrainedOperator`,
  `OperatorJacobiSmoother`): pass lists that outlive them, never temporaries.
- **`ParMesh::FindPoints` is collective** and expects the same points on every rank; for a
  local search call `mesh.Mesh::FindPoints(...)`.
- Formatting: `scripts/style.sh` (astyle 3.4.11), never by hand.

## 7. Testing

### 7.1 Rules
- **Every module ships a fast, targeted unit test**; prefer algebraic checks exact to
  machine precision over convergence studies. gtest + MPI (`incns_add_mpi_test` registers
  np 1, 2, 4; labels `fast` and `dev` for np 1). **A test is green only when np 2 and 4
  pass.**
- Rank-robust tolerances (~1e-13 relative on reduced quantities); no bitwise comparison
  across rank counts. Jacobi iteration counts agree across np within ±1; AMG counts are
  rank-sensitive → per-np baselines.
- **All stored baselines live in `test/baselines.yaml`**, never as magic numbers in tests.
- Convergence studies: measure errors with an elevated RuleBook rule and a tightened Krylov
  tolerance, or the curves flatten for the wrong reason.
- Tests that write files use unique per-rank-count paths and clean up (ctest runs the
  np 1/2/4 instances concurrently).

### 7.2 Tiers
- **Dev** (`scripts/dev.sh`, `ctest -L dev`): np 1 only, seconds; never licenses a commit.
- **Fast** (`ctest -L fast -j8` in `build/cpu`; target < 3 min, currently ~197 s): ~150 tests
  = every module at np 1/2/4 plus the oracles (TGV spatial/temporal orders, adaptive TGV,
  unsteady/steady MMS 2D+3D, NSE MMS, divergence, solver-health iteration baselines).
- **Slow** (`scripts/test.sh cpu -L slow`, np 4, on demand): `cc_sweep_slow_test` (full CC
  robustness grid), `dfg_cylinder_slow_test` (DFG 2D-1, ~9 min). ctest names
  `cc_sweep_slow_np4` / `dfg_cylinder_slow_np4` (select one with `-R`).
- **Debug device** (`scripts/debug_device.sh`), **ASan/UBSan** (`scripts/asan.sh`),
  **Python** (`cpu-python` build, `ctest -R py_`).
- **Expensive** (human-launched only): 3D TGV Re 1600, channel DNS, DFG 2D-2, anything
  > a few minutes or > 4 ranks.

### 7.3 Load-bearing tests (what they pin)
- `stokes_solver_test` / `unsteady_mms_test`: polynomial MMS **exact** to solver tolerance
  (steady and unsteady with quadratic g(t), time-dependent Dirichlet, nontrivial pressure,
  null space) in 2D and 3D — any h-dependent error is a bug.
- `convection_test`: the 3k rule reproduces a 3k+6 reference AND the GLL 2k−1 rule does not
  (the second half keeps the test from passing trivially).
- `nse_mms_test`: NSE temporal order 2D/3D (order tests — EXT2 splitting error is real);
  rotational order vs a same-mesh reference; every velocity PC × Schur path (incl. tensor,
  auto) marches to the same answer; BDF2/EXT3 order 3 on the MMS; the error-controlled
  estimator sees the convective error.
- `tgv_nse_test`, `tgv_stokes_temporal_test`, `adaptive_tgv_test`: oracles.
- `rule_book_test`, `multistep_coeffs_test`, `adaptive_controller_test`, `pressure_mean_test`,
  `stokes_operator_test` (collocated-mass diagonality; GL mass NOT diagonal),
  `grad_div_*_test`, `schur_quality_test` / `cc_robustness_test` / `cahouet_chabard_test` /
  `lp_surrogate_test` / `mass_inverse_test` / `mixed_poisson_test` / `nullspace_test`
  (preconditioner health + baselines), `rotational_convection_test` (2 self-contained tests:
  PA/legacy vs MFEM reference at p ≤ 3; kernel specialization + determinism p = 1..5),
  `point_block_jacobi_test` (self-contained), `rotational_schur_test` (S1/S2),
  `rotational_schur_solver_test` (S3 through StokesSolver + Case rotation log),
  `viscous_ratio_test`, `lor_grad_div_test`, `kinetic_head_test`, `cfl_test`,
  `amr_*_test`, `nc_stokes_test`, `body_force_test`, `cylinder_mesh_test`, `deck_test`,
  `checkpoint_test`, `nondim_test`, `diagnostics_test`, `bc_integration_test`.

### 7.4 GPU parity
CPU/GPU parity runs only where a CUDA build and GPUs exist (≥ 1 GPU per rank); GPU runs need
human approval, and on a cluster the batch script is drafted, not submitted.

### 7.5 PENDING GPU VALIDATION (open since 2026-10-02)
None of these kernels has been confirmed on a GPU (the human ran `rotational_convection_test`
on an H100 on 2026-10-05 — `MatchesMfemReference` was slow at p = 4, 5, hence p ≤ 3 now):
`rotational_convection.cpp` (setup/apply/nodal skew, component `AssembleEA`),
`point_block_jacobi.cpp` (no `#error` guard — confirm nvcc compiles it),
`kinetic_head.cpp` (3D shared tiles ~45 KB), `grad_div_integrator.cpp`
(`GradDivComponentIntegrator::AssembleEA`; `graddiv::DiagonalSumFactorized` 3D tiles
~48.6 KB at MAX_D1D = 14, right at the 48 KB static limit; no `#error` guard),
`amr/gradient_indicator.cpp`, `time/cfl.cpp`, `precond/rotational_schur.cpp`. AMR on
periodic meshes has no debug-device coverage (`amr_checkpoint_test` 2D, `amr_flow_test` T1
skip there) — run them on the GPU; watch the NC prolongation. Steps: `cmake --preset cuda &&
cmake --build --preset cuda`, then the fast tier with 4 GPUs; record the result here and
delete this item.
