# CLAUDE.md

Project instructions for Claude Code. Read this fully before planning any task.

## What this codebase is

A **time-dependent** high-order finite element solver for the incompressible
Navier–Stokes equations, built on **MFEM** (parallel: `mfem::ParMesh` /
`ParGridFunction`). It uses a **coupled, monolithic mixed method** — velocity and
pressure are solved together as one saddle-point system, not via operator splitting or a
pressure-correction scheme.

It builds as a **library** (`libincns`). A case is set up by a thin driver — a `.cpp`
program or a Python script — and/or a **YAML input deck** that supplies parameters, mesh,
boundary/initial conditions, and output settings. **The solver core is never edited to run
a new case;** if a case needs a source change, that's a missing library feature, not a
per-case edit.

- **Elements:** tensor-product **quadrilaterals (2D) and hexahedra (3D) only**. No
  simplices. Spaces are continuous `H1_FECollection` (Qk). Non-tensor-product meshes are
  rejected, not silently handled.
- **Orders:** velocity order `k_u` and pressure order `k_p` are independent **runtime
  options**. Default to an inf-sup–stable (Taylor–Hood) pairing `k_u = k_p + 1`
  — the concrete library default is **`k_u = 3`, `k_p = 2` (Q3/Q2)**; equal
  orders need a pressure-stabilized formulation (PSPG or similar), so warn if
  `k_u == k_p` without that enabled. (Grad–div below is *not* that stabilization.)
- **Boundary conditions (library-level):** velocity **Dirichlet** (no-slip / prescribed),
  **outflow/natural** (do-nothing traction), and **periodic** (mesh-level via
  `Mesh::MakePeriodic` + `CreatePeriodicVertexMapping`), assigned per boundary attribute
  through the BC object (see Architecture). Dirichlet DOFs are eliminated from the block
  system; outflow is natural. **Dirichlet data may be time-dependent** — in scope from
  Sprint 1: the BC coefficients are advanced (`SetTime(t)`) and the eliminated-DOF
  contribution to the RHS is rebuilt **every step**. Stale BC elimination silently drops
  temporal order, and the periodic TGV oracle can never catch it (see the unsteady MMS).
- **Pressure null space is BC-dependent.** It exists exactly when **no boundary carries an
  outflow/natural condition** — i.e. fully periodic, *or* fully enclosed by velocity
  Dirichlet (e.g. a no-slip cavity). The solver must **detect this from the BC set + mesh**,
  not assume it. When it exists, remove it by **orthogonalizing the residual/solution
  against the constant mode each FGMRES solve — never pin a pressure DOF.** Pinning injects a local error that destroys high-order accuracy near the pinned
  node and breaks symmetry; we do not use it, ever. Getting the null space wrong makes
  FGMRES stall or drift rather than diverge, so it's easy to miss.
- **Pressure normalization (post-processor):** separate from the in-solve null-space
  removal, a post-processing step shifts the pressure to **zero integral mean**,
  p ← p − (∫_Ω p)/|Ω|, on output and before any pressure error is measured. Compute the
  mean as a **mass-weighted L2 integral** (via quadrature / the mass matrix), **not** an
  average of nodal values — the nodal average is wrong at high order.
- **Outer solver:** FGMRES on the full saddle-point system with a block preconditioner.
- **Velocity block:** default **Jacobi** (`OperatorJacobiSmoother`, matrix-free / PA).
  Deliberate — see Time integration for why the block stays well-conditioned. **BoomerAMG
  is an option** for stiffer regimes, but it needs an assembled matrix (or LOR), so it
  changes the assembly route.
- **Pressure Schur complement (Sprint 1): scaled pressure mass.** Ŝ = (1/ν) M_p, applied
  as Ŝ⁻¹ = ν M_p⁻¹ (diagonal/Jacobi or a short CG on M_p — cheap either way). Deliberately
  simple: spectrally equivalent to S for **steady** Stokes, and Δt-independent, so nothing
  to refresh when the step changes. **Known, accepted limitation:** it degrades as Δt
  shrinks (it lacks the (β0/Δt) L_p⁻¹ term; the misfit sits on the low pressure modes,
  ∝ β0/(Δt·ν), and is essentially mesh-independent) — growing iteration counts at small
  Δt are *expected in Sprint 1*, not a bug to chase. **The
  preconditioner deep-dive — Cahouet–Chabard, the L_p solve, its BCs and null space — is
  Sprint 2.** Do not build any of it, or stub it, in Sprint 1.
- **Grad–div term (optional):** augmented γ (∇·u, ∇·v) on the velocity block, runtime γ.
  No native H1 grad–div integrator exists, so use `ElasticityIntegrator` with
  **λ = γ, μ = 0** (elasticity form is λ(∇·u,∇·v) + 2μ(ε(u),ε(v)); μ = 0 leaves grad–div).
  Partial assembly is supported for it. Improves mass conservation — ‖∇·u‖ should drop
  when on. **γ enters the preconditioner:** with grad–div on, the Schur mass term scales
  as **(ν+γ) M_p⁻¹**, not ν M_p⁻¹ (augmented-Lagrangian form). This applies to the
  **Sprint-1 pressure-mass block directly** — Ŝ⁻¹ = (ν+γ) M_p⁻¹ — and to the mass term of
  Cahouet–Chabard later. Miss the γ and iteration counts blow up exactly
  when γ is doing its job. Large γ also stiffens the velocity block and weakens Jacobi,
  so the γ>0 path keeps its **own iteration-count baseline** (see Fast tier).
- **Assembly:** partial assembly (`AssemblyLevel::PARTIAL`) on the tensor-product elements
  — affordable high order on CPU (sum factorization), GPU path open for later. Prefer
  `mfem::forall` for new kernels.
- **Quadrature policy:** **Gauss–Legendre everywhere by default.** The RuleBook hands out
  rules **per operator, selecting both order and 1D rule type** (it owns separate
  `IntegrationRules` containers per family — e.g.
  `IntegrationRules(0, Quadrature1D::GaussLobatto)` alongside the GL default). The one
  standing exception is a **runtime option for mass matrices: Gauss–Lobatto collocation**
  — k+1 GLL points per direction for degree-k elements, collocated with the GLL nodal H1
  basis, which makes the mass matrix **diagonal** (SEM-style lumping). This is deliberate
  under-integration (exact to degree 2k−1 vs the 2k mass integrand) — standard spectral
  element practice, spectrally accurate, and it does not degrade convergence order.
  **Precondition:** diagonality requires the GLL *nodal basis*
  (`BasisType::GaussLobatto`, MFEM's H1 default) — assert/verify the basis when the
  option is on; GLL points against a non-collocated basis silently gives a full,
  under-integrated mass matrix. Payoff: at small DNS Δt the momentum block is
  mass-dominated, so a diagonal mass makes the Jacobi block preconditioner nearly exact.
- **Output:** `ParaViewDataCollection`, written by `Run()` per the deck's output settings.
  **High-order output must be enabled** — `SetHighOrderOutput(true)` plus a
  levels-of-detail refinement ≥ `k_u` — or Q4/Q5 fields render as low-order mush, which
  looks exactly like a solver bug and isn't one. Do not "fix" the solver over a
  visualization artifact; check the output settings first.

Production runs (DNS/LES of turbulent flows) target DOE systems; **this is not where you
run them** (see Guardrails).

<!-- Resolved in 1.0b: astyle pinned to 3.4.11 (desktop env lockfile); default polynomial
     orders k_u = 3, k_p = 2 (Q3/Q2 Taylor–Hood). -->

**Repo: `bessemer`.** Env is live at `environments/desktop` (human-built, see 1.0a).

## Architecture & build order

**Modular, not monolithic.** Each concern is its own file/class with its own unit test
(see Unit tests). No god-class, no 2000-line `solver.cpp`. Rough decomposition:

```
src/
  mesh/periodic_box.{hpp,cpp}       # periodic quad/hex mesh factory
  spaces/mixed_spaces.{hpp,cpp}     # velocity/pressure ParFESpaces, block offsets
  operators/
    stokes_operator.{hpp,cpp}       # block [A Bᵀ; B 0], PA; optional grad–div on A
    convection.{hpp,cpp}            # nonlinear term, dealiased rule (Sprint 2)
    pressure_schur.{hpp,cpp}        # Schur approx — Sprint 1: (1/ν)M_p; Cahouet–Chabard here (Sprint 2)
    block_preconditioner.{hpp,cpp}  # velocity smoother (Jacobi/AMG) + pressure Schur block
  time/
    multistep_coeffs.{hpp,cpp}      # BDF + AB/EXT coefficients, variable-step aware
    time_integrator.{hpp,cpp}       # in-repo stepper (NOT mfem::ODESolver)
    adaptive_controller.{hpp,cpp}   # LTE estimate, PI controller, mixed abs/rel tol; records step history
  post/pressure_mean.{hpp,cpp}      # mass-weighted mean-zero projection
  post/output.{hpp,cpp}             # ParaViewDataCollection writer — high-order output + LOD
  quadrature/rule_book.{hpp,cpp}    # OWNS IntegrationRules (program lifetime); per-integrator order AND 1D type (GL default, GLL mass option)
  util/profiler.{hpp,cpp}           # nested scoped wall-time profiler (INCNS_PROFILE)
  bc/boundary_conditions.{hpp,cpp}  # per-attribute Dirichlet/outflow/periodic
  config/parameters.{hpp,cpp}       # Parameters struct + YAML load (yaml-cpp)
  solver/
    stokes_solver.{hpp,cpp}         # one implicit saddle-point solve
    navier_stokes_solver.{hpp,cpp}  # reuses the Stokes step; adds convection (Sprint 2)
  exact/tgv2d.hpp                   # shared analytic TGV (also used by tests)
apps/
  run_case.cpp                      # generic YAML-driven driver (no recompile per case)
  taylor_green.cpp                  # example in-code driver (analytic IC, periodic)
cases/*.yaml                        # input decks (params, mesh, BCs, output)
environments/<machine>/             # per-machine spack.yaml + committed spack.lock
python/                             # optional pybind11 bindings + example.py
test/                               # unit tests + the TGV convergence oracles
```

**Using the library.** The public surface a driver touches:
- `incns::Parameters` — physics/discretization/time/solver/output settings; `LoadYAML(path)`
  fills it from a deck. The solver takes `NavierStokesSolver(ParMesh&, const Parameters&)`;
  lightweight setters (`SetTimeScheme`, `SetFixedTimeStep`) exist for in-code/test use and
  just override fields of a default `Parameters`.
- `incns::BoundaryConditions` — `AddVelocityDirichlet(attr, coeff)`, `AddOutflow(attr)`;
  periodicity is mesh-level. Passed via `SetBoundaryConditions`. An empty BC set on a
  periodic mesh means fully periodic.
- `SetInitialVelocity(coeff)` (pressure is not an independent IC); `Run()` marches to
  `t_end` and writes output per the deck, or `Step()`/`Time()`/`Done()` for manual control;
  `Velocity()` / `Pressure()` return the fields. A YAML case and an in-code case go through
  this same surface — that's the point of the library split.

**Stokes first, then NSE — by design.** The implicit solve each step is *identical* for
unsteady Stokes and NSE: both solve the same block system
`[(β0/Δt)M + νK, Bᵀ; B, 0] x = rhs`. The only difference is the RHS — Stokes has just the
BDF history (+ forcing); NSE adds the explicitly-extrapolated, **dealiased** convection.
So **`NavierStokesSolver` composes a `StokesSolver`** (shared implicit step) plus a
`ConvectionOperator`; it must not reimplement the saddle-point machinery. (Equivalent
inheritance form: a base `TransientSolver` with a virtual `AssembleExplicitRHS()` — empty
for Stokes, convection for NSE. Composition is preferred for testability.)

**Two sprints, with a hard gate between them.** Sprint 1 delivers a complete, validated
**unsteady Stokes** library. Sprint 2 is **NSE plus the preconditioner stage** — convection
and everything that depends on it, and the Schur deep-dive (Cahouet–Chabard, the L_p solve,
its BC/null-space handling). **Do not write, scaffold, or even stub any of it — convection,
AB/EXT extrapolation, dealiasing, `NavierStokesSolver`, the NSE oracle, Cahouet–Chabard or
any pressure-Laplacian (L_p) machinery — until Sprint 1 is finished
AND the human has explicitly signed off.** During Sprint 1 the `navier_stokes_*` and
`convection.*` files stay *absent*, not empty-stubbed, and `pressure_schur` contains the
mass block only. If a Sprint-1 task appears to need
convection, it is mis-scoped — stop and ask.

**Sub-sprint protocol — correctness over throughput.** The plan below is deliberately
fine-grained: one feature per sub-sprint, strictly sequential. **No working ahead** — no
parallel scaffolding of future sub-sprints, no speculative files, no subagents racing
down the list. Each sub-sprint ends with: (1) the new module's unit tests green at
np ∈ {1, 2, 4}; (2) the **entire fast tier green** — no regressions anywhere; (3) a
commit whose message states what was verified, with what evidence, and what is *not* yet
covered; (4) **stop for human review before the next sub-sprint begins.** This project
optimizes for verified correctness, not velocity — when in doubt, add the check. Do
**not** scaffold a whole solver in one shot.

**Sprint 1 — Stokes: shared infra + full time integration + grad–div + adaptive.**

- **1.0a — Spack environment (supervised session).**
  **Status: DONE for `desktop`** — built by the human at `environments/desktop`; this
  sub-sprint re-runs only when bringing up a **new machine**. Sub-sprint 1.0b therefore
  begins by **verifying** the existing env against this sub-sprint's green criteria
  (toolchain resolution + the MFEM MPI smoke case below) — verify, don't rebuild, and if
  a check fails, report it and stop rather than modifying the human-built env.
  Environments live in
  **`environments/<machine>/spack.yaml` — one directory per machine** (laptop, workstation,
  each cluster), since compilers, externals, and MPI differ per system.
  **Spack provides the entire toolchain — the system gcc/cmake/MPI are never used for
  project code.** Two-phase bootstrap: first `spack install` a **recent GCC** (system
  compiler is used for that bootstrap build *only*) and register it; then the env pins
  every spec to that compiler (`%gcc@<version>` / `packages:all`) and includes **cmake,
  ninja, and MPI as spack packages** alongside `mfem@develop` (see Environment & build),
  hypre, METIS, yaml-cpp, and pybind11 if Python is built. No `spack external find`, no
  system packages — the **single exception**, requiring no judgment call, is cluster
  environments where the vendor MPI is declared as an external (never build MPI from
  source on a cluster). This is a **long build** — that is expected: do not kill or
  restart jobs for being slow, and report failures verbatim instead of "fixing" them by
  silently pinning MFEM to a release, swapping dependency versions, or falling back to a
  system tool (flag it, per Environment & build). Commit each machine's `spack.lock` next
  to its yaml — it records the concretized MFEM commit for that machine. **Green when:**
  `which gcc / cmake / mpicc / mpirun` all resolve inside the activated env, and a
  trivial MFEM MPI smoke case (one of MFEM's own examples is fine) compiles against the
  env and runs at np = 2 and 4 — the toolchain is proven before any repo code exists.
- **1.0b — Harness.** Everything this file references must exist before any numerics:
  CMake presets + Ninja, `scripts/env.sh` (machine resolution — see Environment & build),
  `scripts/build.sh` / `test.sh` / `style.sh`, ctest registration
  at np ∈ {1, 2, 4}, `config/style.astylerc`, the git hooks, and the baselines file —
  **all stored baselines live in `test/baselines.yaml`**, one place, never as magic
  numbers in test source. Resolve every `<!-- ADAPT -->` placeholder in this doc against
  the real repo in the same change. **Green when:** a hello-world MPI target builds and
  a trivial ctest passes at all three rank counts through `scripts/test.sh`. The fitness
  function exists before the code it judges.
- **1.1 — Mesh & spaces.** `periodic_box` factory and `mixed_spaces` (velocity/pressure
  `ParFESpace`s, block offsets), 2D quads and 3D hexes. **Green when:** module tests
  verify DOF counts against closed-form values for the configured pairs and periodic
  node identification, at np ∈ {1, 2, 4}.
- **1.2 — Quadrature RuleBook.** Ownership, per-integrator order **and 1D type**
  (GL default, GLL option). **Green when:** the *RuleBook selection & lifetime* test passes.
- **1.3 — Profiler.** Nested scoped wall-time regions, MPI max/min reduction,
  `INCNS_PROFILE` toggle. **Green when:** its module test passes (nesting depth,
  reduction correctness) and the toggle compiles both ways.
- **1.4 — Core operators, BCs, pressure post (no solver yet).** `stokes_operator` blocks
  (mass incl. the collocated-GLL option, viscous, divergence), the BC object with
  Dirichlet elimination, `pressure_mean`. **Green when:** *Collocated-mass diagonality*
  (2D and 3D), *Divergence operator*, and *Mean-zero post-processor* tests pass.
- **1.5 — Steady Stokes solve.** `stokes_solver`: one saddle-point solve, block FGMRES
  with the pressure-mass Schur block (Ŝ⁻¹ = ν M_p⁻¹), null-space detection +
  orthogonalization. **Green when:** the **steady polynomial exactness MMS** passes in
  2D and 3D (nontrivial pressure — see Unit tests), then spatial order
  (velocity/pressure) and inf-sup OK.
- **1.6 — Preconditioner health.** **Green when:** the *Schur block quality* test passes —
  FGMRES iteration count bounded / mesh-robust at fixed Δt — and its baselines are seeded
  in `test/baselines.yaml`. Δt-robustness is *out of scope* here — that's the Sprint-2
  preconditioner stage.
- **1.7 — Multistep coefficient module.** BDF + AB/EXT, variable-step aware. **Green
  when:** the *Multistep coefficients* test passes (known uniform-step values; exact
  differentiation of `1, t, t², t³` under non-uniform ratios).
- **1.8 — Unsteady Stokes, fixed step.** `time_integrator` (in-repo BDF2; BDF3 in
  test-only mode), BDF1 → BDF2 → BDF3 startup ramp. **Green when:** the **unsteady
  polynomial MMS** (exact in space *and* time, incl. time-dependent Dirichlet, 2D and 3D)
  passes — it gates this sub-sprint before any TGV rate is measured — then the TGV-Stokes
  oracle shows temporal order ≈ 2 (≈ 3 for BDF3 in test mode). The 2D TGV *velocity* also
  solves unsteady Stokes (its convective term is a pure gradient absorbed into pressure),
  so the Stokes oracle uses it with convection off — velocity matches, pressure → 0.
  Note: in Stokes the AB/EXT half is dormant (nothing to extrapolate) — pure BDF here.
- **1.9 — Adaptive stepping.** BDF2-vs-BDF3 LTE, PI controller, mixed abs/rel tol, step
  history recording, the (inert) Δt ceiling hook. Validated on unsteady Stokes. **Green
  when:** the *Adaptive controller* unit test passes and the adaptive-mode fast-tier
  check does (fewer steps than fixed-Δt at equal error, sane recorded step history,
  bounded rejections).
- **1.10 — Grad–div.** Optional γ on the velocity block, **(ν+γ) M_p⁻¹ Schur scaling**,
  γ>0 iteration baselines. **Green when:** the *Grad–div assembly* test passes, ‖∇·u‖
  drops with γ on, and the γ>0 baseline is seeded.
- **1.11 — Deck, driver & output.** `Parameters` + YAML load, `run_case` driver,
  `post/output` (ParaView, high-order + LOD). **Green when:** a YAML TGV deck reproduces
  the in-code driver's results within tolerance through the same library surface, and
  the written output renders at high order.

**Sprint 1 ends here — a complete, validated unsteady Stokes library. Get human sign-off
before touching Sprint 2.**

**Sprint 2 — Navier–Stokes + preconditioner stage (only after sign-off):**

- **2.1 — Convection operator + dealiasing.** Operator level first, no solver changes.
  **Green when:** the *Dealiasing / nonlinear-term exactness* flagship test passes.
- **2.2 — NSE solver.** AB/EXT extrapolation of the nonlinear term; `NavierStokesSolver`
  composes the Stokes step. **Green when:** the full 2D TGV oracle passes (velocity +
  NSE pressure rates).
- **2.3 — Preconditioner deep-dive.** Cahouet–Chabard (ν M_p⁻¹ + (β0/Δt) L_p⁻¹) replaces
  the mass-only Schur block — the L_p operator, its BCs and null-space handling, solver
  choices, robustness in Δt and ν. **Deliberately under-specified for now**; expect
  detailed direction from the human when this stage opens, and do not design ahead of it.

## Time integration

- **Written in-repo. Do NOT use any MFEM built-in time steppers** — no `mfem::ODESolver`
  or its subclasses. The scheme is implemented separately in this codebase.
- **Default: BDF2 / AB2 semi-implicit (IMEX).** The linear Stokes/viscous part is implicit
  via **BDF2**; the nonlinear convection is explicit via **AB2** extrapolation. Each step
  solves the coupled saddle-point system — that's the per-step FGMRES solve. The BDF2 mass
  term dominates at the small DNS Δt, which is exactly why the momentum block is
  well-conditioned and Jacobi is enough.
- **Startup:** BDF2/AB2 need two levels of history. Bootstrap the first step(s) at lower
  order (BDF1/AB1) or equivalent. A wrong startup shows up as *reduced temporal order* in
  the 2D TGV check — that's the guard, so don't paper over it.
- **Dealiasing:** the nonlinear convective term is **dealiased by over-integration** — a
  quadrature rule exact for the quadratic product (~⌈3k/2⌉ points/direction), not the
  default rule. Prevents aliasing-driven blow-up in DNS. The convection integrator carries
  its higher-order rule from the **rule book**; linear terms keep the standard rule. Do not
  "simplify" the convection term back to the default quadrature.
- **Higher-order pair: BDF3 / AB3.** Implemented alongside BDF2/AB2 **solely to drive the
  adaptive error estimator below** — it is *not* a production time scheme and is never the
  advancing solution outside of tests. (Tests do march it directly to verify 3rd order,
  because the estimator is only trustworthy if that path genuinely is 3rd order.) Needs
  three history levels, so the startup ramp extends (BDF1 → BDF2 → BDF3).
- **Adaptive timestepping (option; default stays fixed-step BDF2/AB2).** Estimate the
  local truncation error from the **difference between the order-2 and order-3 solutions**
  (BDF2/AB2 vs BDF3/AB3) — that difference approximates the LTE of the order-2 step. A
  step-size controller drives ‖LTE‖ to a target tol: Δt_new = safety · Δt ·
  (tol/‖LTE‖)^(1/3), with growth/shrink caps and **step rejection + retry** when
  ‖LTE‖ > tol. Prefer a PI (Gustafsson) controller over the bare formula for smoother
  step sequences. The solution **always advances with the order-2 (BDF2/AB2) step**; the
  order-3 solution is auxiliary, formed only for the LTE estimate — **no local
  extrapolation**.
  - **Control on velocity only.** Incompressible NS is an **index-2 DAE**: velocity is the
    differential variable, pressure is algebraic. Build the LTE measure from **u**; keep
    pressure out of the error control — controlling on the algebraic pressure is ill-posed.
  - **Mixed absolute + relative tolerance.** Independent **atol** and **rtol**, both always
    computed, combined in the acceptance threshold: accept when ‖LTE‖ ≤ atol + rtol·‖u‖.
    Setting one to ~1e-16 makes its term negligible and effectively disables it — rtol→0
    gives pure absolute, atol→0 gives pure relative. Use the **global** form (scale the
    global ‖LTE‖ by the global ‖u‖), **not** a per-DOF WRMS weight `1/(rtol|u_i|+atol)`:
    the per-DOF weight blows up where a velocity component crosses zero (2D TGV vanishes on
    whole lines), so per-DOF relative control needs an atol floor and can't be turned off.
  - **Variable-step coefficients are mandatory.** Under non-uniform Δt the BDF *and* AB
    coefficients depend on the step-size ratios — recompute them each step from the actual
    ratios. **Never reuse the uniform-step coefficients when Δt varies**; that silently
    drops order and can destabilize. This is the single most common adaptive-BDF bug.
  - **Δt-dependent operator.** The implicit block depends on Δt, so the operator and its
    preconditioner must be refreshed whenever Δt changes. Cheap with the Jacobi default —
    another reason AMG isn't default, since its setup cost penalizes frequent step changes.
    (The Sprint-1 Schur block, (1/ν) M_p, is Δt-independent — only the velocity side
    refreshes.)
  - **Δt ceiling hook (stability, not accuracy).** The LTE controller sees accuracy only.
    Once explicit convection exists (Sprint 2) it imposes a convective-CFL stability
    ceiling the estimator cannot detect — a pure-LTE controller will push Δt past it and
    thrash on rejections. The controller therefore always applies
    Δt_new ← min(Δt_new, Δt_cap) through a pluggable ceiling: inert (∞) in Sprint 1,
    wired in Sprint 2 to a convective CFL bound (global min over elements,
    ∝ h/(k_u²·|u|)). The interface lands in Sprint 1 with the controller — it is not
    bolted on later.
  - **Step history is part of the interface.** The controller records
    (t, Δt, accepted/rejected, ‖LTE‖) for every attempted step and exposes it
    programmatically. The adaptive-mode check consumes this record to assert bounded
    rejections and a sane step sequence — it is never reconstructed by parsing logs.

## Environment & build

Dependencies (MFEM, hypre, METIS, MPI, yaml-cpp; pybind11 if Python is built) come from
**spack**, not the system default — and so does the **toolchain itself**: a recent
spack-built GCC (bootstrapped once with the system compiler, then registered — the system
compiler is never used again), plus cmake, ninja, and MPI as spack packages. **The system
gcc, cmake, and MPI are never used to configure, build, or run project code**, on any
machine. `scripts/env.sh` **asserts** after activation that `gcc`, `cmake`, `mpicc`, and
`mpirun` resolve inside the spack env and aborts otherwise — a PATH leak to a system tool
is an environment bug, not something to shrug past. The one standing exception: on
clusters the vendor MPI is a declared external (see sub-sprint 1.0a); nothing else is
ever externalized. Environments are **per-machine directory envs**:
`environments/<machine>/spack.yaml` with its committed `spack.lock` beside it — one
directory per machine, because compilers, MPI, and externals differ per system. The env
for the current machine is built in **sub-sprint 1.0a** — a supervised session; nothing
else starts until its toolchain-resolution and MPI smoke checks pass.
Activate before anything: `spack env activate ./environments/<machine>`. **Known
machines: `desktop`** (the human's local build — the current default). The machine
name resolves in **one place** — `scripts/env.sh` (hostname detection with an
`INCNS_MACHINE` override, defaulting to `desktop`) — and build/test scripts source it;
never hardcode a machine path anywhere else.

Adding a new machine = adding a new `environments/<name>/` directory; existing envs and
lockfiles are never edited to make a different machine work.

**MFEM tracks the `dev` branch** (via spack) unless explicitly stated otherwise — the
exact commit is recorded in the spack env/lock. dev moves: verify capability claims
(e.g. whether an integrator has a PA path) against the *checked-out commit*, not release
notes. If an API seems missing, suspect a commit mismatch and flag it — do not silently
work around it or downgrade the approach.

Build system is CMake + Ninja. The **CPU build is the one that matters right now**:

```
cmake --preset cpu          # RelWithDebInfo, ccache on
cmake --build --preset cpu
```

A **CUDA build is optional and off by default** (`-DUSE_CUDA=ON`, or the `cuda` preset) —
enable only on a machine with a GPU + CUDA toolkit. Do not assume it is available.

- `ccache` is on. Never `rm -rf build/` — it dumps the warm cache and turns a 30 s
  rebuild into 15 min. If a build misbehaves, ask before wiping anything.
- Warnings are errors (`-Werror`). A build that warns is a build that failed.

**Environment & git are structural, not memory.** Build and test go through
`scripts/build.sh` / `scripts/test.sh`, which **activate the spack env first**, and the git
hooks call those — so activation never depends on the model remembering `spack env activate`
in a fresh shell. Git workflow: one branch per sub-sprint (`sprint1.1-spaces`,
`sprint1.5-stokes`, …),
`git status`/`git diff` before and after, **commit when the fast tier is green** with a
message naming the sub-sprint, and **never push, force-push, or touch `main`** (a human merges).
This repeats the git guardrail below because it's the operational default, not an afterthought.

## Code style

Formatting is **astyle**, matching MFEM's convention — not clang-format. Options in
`config/style.astylerc`; runner is `scripts/style.sh`.

- **astyle is version-sensitive.** Use the pinned version (**astyle 3.4.11** — from the
  desktop env lockfile; `scripts/style.sh` refuses to run on a mismatch). A different
  version silently reformats the whole tree and buries the real
  diff — if `scripts/style.sh` produces a huge diff, you have the wrong astyle: stop and
  flag it, don't commit it.

## Unit tests

The end-to-end TGV convergence oracle is an *integration* test — necessary, not sufficient.
**Every module ships with a fast, targeted unit test in the same change** — code without one
is not done. Prefer **algebraic checks exact to machine precision** over convergence studies,
so a failure localizes to the module. All live in `test/` and carry the `fast` label.

**Every unit test — all of them, no exceptions — runs under MPI at 2 and 4 ranks. A
serial pass is never sufficient.** Partition-boundary and reduction bugs are invisible at
np=1. Register each test at np ∈ {1, 2, 4} in CTest via the configured MPI launcher;
np=1 exists for debugging, but a test only counts as *green* when the 2- and 4-rank runs
pass (this stays within the 4-rank guardrail — do not go wider). Two consequences to
design tests around:
  - **"Machine precision" must be rank-count-robust.** MPI reductions reorder floating-point
    sums, so use tight *relative* tolerances (~1e-13) on globally-reduced quantities —
    never bitwise equality, and never bitwise comparison of results across rank counts.
  - **Iteration-count baselines:** Jacobi-preconditioned counts should agree across np to
    within ±1 — a wider spread is a parallel bug, not noise. AMG counts are legitimately
    rank-count-sensitive: record those baselines per-np.

The tests:

- **Dealiasing / nonlinear-term exactness (flagship).** On a single element (and a small
  periodic mesh), take polynomial velocities of degree `k_u`; the convection integrand is a
  polynomial of known degree. Assert the **over-integrated rule reproduces that integral to
  machine precision**, and that the default collocation rule does **not**. Pure quadrature
  exactness — no solver, no stepping — and exactly what pins the dealiasing that the smooth
  laminar TGV case can never stress.
- **Stokes polynomial exactness (MMS, nontrivial pressure).** Pick a divergence-free
  polynomial velocity of degree ≤ `k_u` (e.g. the curl of a polynomial streamfunction) and
  a zero-mean polynomial pressure of degree ≤ `k_p`; set f = −νΔu + ∇p and impose the exact
  velocity as Dirichlet data on the whole box. The exact solution lies in the discrete
  space, so the solve must **reproduce it to (tightened) Krylov tolerance — any
  h-dependent error is a bug, not discretization error.** This is the test that pins the
  mixed discretization, Dirichlet elimination, and the null-space/mean handling with
  **p ≠ 0** — the TGV-Stokes oracle cannot (its pressure is identically zero).
  All-Dirichlet ⇒ the pressure null space exists ⇒ orthogonalization runs against a
  nontrivial pressure; compare after the mean-zero shift (shift the exact p too).
  **Unsteady variant (also required):** multiply the solution by a quadratic g(t). BDF2
  differentiates quadratics exactly, so the **entire unsteady solve** — history, the
  BDF1→BDF2 startup ramp, forcing evaluated at t^{n+1}, and **time-dependent Dirichlet
  data** — must reproduce the exact solution to solver tolerance at *every* step. This is
  the only Sprint-1 test that exercises time-dependent Dirichlet (SetTime + per-step
  re-elimination); the periodic TGV never touches that path. **Run both variants in 2D
  and 3D** — the tiny hex case is nearly free and is Sprint 1's only 3D coverage; 3D
  tensor/PA index bugs hide until channel flow otherwise.
- **Multistep coefficients.** Uniform-step BDF2/BDF3 and AB2/AB3 equal their known values;
  under a non-uniform step ratio the operator differentiates `1, t, t², t³` exactly to the
  scheme's order. Guards the variable-step bookkeeping (the #1 adaptive bug).
- **RuleBook selection & lifetime.** Requesting (order, 1D type) returns the right rule:
  correct point count per family, and the GLL rule's abscissae **include the element
  endpoints** while the GL rule's do not — a cheap, unambiguous family fingerprint.
  Repeated requests return **stable addresses** (the same `IntegrationRule*`), guarding
  the non-owning-pointer contract the integrators rely on.
- **Collocated-mass diagonality.** With the GLL mass option on (k+1 GLL points/direction,
  GLL nodal basis), assert the mass operator is **diagonal to machine precision** — apply
  it to unit basis vectors and check off-diagonal response ≈ 0, and check its diagonal
  matches the assembled diagonal used by Jacobi. Also assert the default GL-rule mass is
  **not** diagonal (guards a silent basis/rule mismatch), and that total mass ∫1·1 is
  preserved by the lumped rule. Pure algebra — no solver.
- **Schur block quality (pressure mass).** Small periodic Stokes problem: FGMRES with the
  (1/ν) M_p block converges in a bounded, **mesh-robust** iteration count *at fixed Δt*.
  Do **not** assert Δt-robustness — iteration growth at small Δt is the known Sprint-1
  limitation, fixed by Cahouet–Chabard in Sprint 2. Seeds the solver-health baseline
  (baselines are per-Δt).
- **Divergence operator.** `B` matches the weak divergence of a known field; `Bᵀ` is `B`
  transposed (structural).
- **Grad–div assembly.** `ElasticityIntegrator(λ=γ, μ=0)` equals a reference grad–div form on
  a small mesh, element-wise.
- **Mean-zero post-processor.** A field with a known offset returns zero mass-weighted mean;
  exact for a constant shift.
- **Adaptive controller.** Synthetic LTE sequences drive the expected step-size updates,
  rejections, and the abs/rel disable-by-tiny-tol behavior.

## Test tiers — this is the loop's fitness function

Only the fast tier runs inside the agentic loop; it is the **unit tests above plus** the
convergence oracles below.

### Fast tier (`ctest -L fast`, CPU build, < 3 min total, 2–4 MPI ranks)

- **2D Taylor–Green vortex, exact solution — this is the first check.** The 2D TGV has a
  closed-form decaying solution on a fully periodic box (u = cos x sin y · e^(−2νt), and
  so on). Run it two ways: **h-refinement at fixed small Δt** → assert spatial order
  in L2 ≥ `min(k_u+1, k_p+2) − 0.2` (velocity) and `k_p+1 − 0.2` (pressure, measured after
  the mean-zero post-processor). Expected rates are **pair-dependent**, so the target comes
  from the configured `(k_u, k_p)`, not a fixed formula: Q_N/Q_{N-1} → velocity N+1,
  pressure N; Q_N/Q_{N-2} → velocity N (capped by the coupling), pressure N-1.
  **Δt-refinement at fixed fine mesh** → assert temporal order ≈ 2 for BDF2/AB2 and ≈ 3 for
  BDF3/AB3 (BDF3 marched in test-only mode — it is estimator-only in production, but the
  estimator is only valid if this path is genuinely 3rd order). This one case exercises periodicity, the dealiased
  nonlinear term, the coupled solve, the null-space handling, and the timestepper against
  an exact answer — which is why it leads.
- **Adaptive-mode check (2D TGV).** Run adaptive stepping at a set tol; assert the achieved
  global L2 error in u sits within a small factor of tol and shrinks when tol is tightened,
  and that the run finishes with a sane step history (bounded rejections). This tests the
  estimator + controller, not just the underlying schemes.
- **Divergence.** Assert ‖∇·u‖ below tol. With grad–div on it should tighten — a γ=0 vs
  γ>0 case asserting ‖∇·u‖ drops guards that path.
- **Coupled solver-health regression.** Assert the outer **FGMRES iteration count** stays
  within a band of the stored baseline. Keep **separate baselines for Jacobi vs AMG** —
  they differ, and a regression in one must not be masked by the other's baseline. Same
  for **γ=0 vs γ>0**: grad–div changes both the velocity block and the Schur scaling.
- *(Optional)* a steady manufactured solution (e.g. Kovasznay, non-periodic) to isolate
  spatial order with no temporal error, if the TGV split isn't enough to localize a bug.

All asserts use **tolerances, not bit-exact compares** — MPI FP reductions aren't
deterministic across rank counts. Fixed seeds for anything stochastic.

**Convergence-study hygiene:** measure L2 errors with an **elevated quadrature rule from
the RuleBook**, not the default — default-rule error integration pollutes high-order rates
by roughly the margin the −0.2 slack allows, so a passing solver can fail on measurement
error alone. And **tighten the Krylov tolerance for convergence runs** so algebraic error
sits well below the finest-grid discretization error — otherwise the curve flattens at the
fine end and the order assert fails for the wrong reason.

### Conditional checks (run only when the relevant build exists)
- **CPU/GPU parity.** *Only when the CUDA build is present and a GPU is available.* Run one
  small case on both builds, assert agreement to tol. Skipped cleanly on a CPU-only
  machine — never a blocker there.

### Expensive tier (manual only — NEVER in the loop)
Resolved 3D TGV at Re=1600, channel-flow DNS, anything > a few minutes or > 8 ranks.
Launched by a human via the batch scheduler. See Guardrails.

## Guardrails — hard rules, do not violate

- **NEVER submit to the batch scheduler** (`sbatch`, `srun` beyond the fast-tier rank
  count, `flux`, `bsub`). Draft the batch script and stop; a human submits it.
- **NEVER run above 4 MPI ranks or a 32³ mesh** without explicit approval, and (once CUDA
  is enabled) do not launch GPU runs without approval. If verifying seems to need a big
  run, say so and stop — do not "just try it."
- **Sprint gate:** no NSE-specific code (convection, dealiasing, AB/EXT *extrapolation of
  convection*, `NavierStokesSolver`, the NSE oracle) **and no Cahouet–Chabard / L_p
  preconditioner code** until Sprint 1 (Stokes) is complete and
  signed off — the Sprint-1 Schur block is (1/ν) M_p (with γ when grad–div is on), full
  stop. Grad–div, adaptive stepping, and the BDF/AB coefficient module are Sprint 1.
  When unsure whether something is Sprint 2, stop and ask — don't stub it.
- Do not edit `third_party/` or the vendored/spack MFEM install.
- **Never build or run project code with the system toolchain** (gcc, cmake, MPI). If
  `scripts/env.sh`'s toolchain assert fails, fix the environment or stop — do not work
  around it by calling system tools directly.
- Do not push, force-push, or touch `main`. Work on a branch.
- If a change alters numerical results (formulation, quadrature/dealiasing, time scheme,
  preconditioner, grad–div, order defaults, null-space or mean handling), say so in the
  plan and flag which fast-tier baselines need re-blessing.

## Coding conventions

- C++17, MFEM style: `mfem::` types, RAII, no raw `new`/`delete` in new code.
- **Doxygen docstrings on every public API entity.** Each header carries an `@file`
  brief; every public class/struct, free function, enum, and data member gets a Doxygen
  comment (`/** @brief ... */` with `@param`/`@return`/`@pre`/`@tparam` as applicable;
  `///` briefs for members and trivial accessors). Struct/enum members use `///` (or
  trailing `///<`) so Doxygen actually captures them — a plain `//` does not. `.cpp`
  implementation comments stay plain `//` (they document *how*, not the API).
- **Modular:** one concern per file, small classes, no god-class. A new module lands with
  its unit test in the same change.
- **Quad/hex only:** assert element geometry is `Geometry::SQUARE` / `Geometry::CUBE`;
  reject simplex meshes at load time, not downstream.
- **Pressure:** null space removed by orthogonalization only — **never pinned**; output
  normalized to zero integral mean by the post-processor (mass-weighted, not nodal).
- **Time integration is in-repo** — never reach for `mfem::ODESolver` or built-in steppers.
- **Convection stays over-integrated** — its integrator keeps its own higher-order rule;
  do not collapse it to the default quadrature.
- **Quadrature via the rule book.** Integrators take their `IntegrationRule` from the
  `RuleBook`; never hardcode a rule at the call site, and **never pass the address of a
  temporary/local `IntegrationRule` to an integrator.** MFEM integrators hold *non-owning*
  `const IntegrationRule*`, so the `RuleBook` **owns the rules and must outlive every
  operator/integrator that references them** — a destroyed rule is a dangling pointer and
  segfaults, usually far from the cause. Per-integrator **order and 1D rule type** (GL
  default; GLL for the collocated-mass option) are set through it — no integrator builds
  its own `IntegrationRules` container.
- **Profile the hot paths.** Wrap the solve and its subcomponents in nested scopes
  (`INCNS_PROFILE("name")`): outermost = total solve, with assembly, preconditioner
  setup/apply, and each Krylov solve as nested regions (convection assembly too, in Sprint 2).
  New expensive paths get a scope. Lightweight — wraps `MPI_Wtime`, no external dependency;
  the report is a nested tree with inclusive/exclusive time, % of parent, and MPI max/min
  across ranks to expose imbalance.
- **Prefer device-portable kernels** (`mfem::forall` / PA) so a future CUDA build needs no
  rewrite — but CPU is the only required target today.
- Parallel-correct by construction: every reduction is global; never assume rank 0 holds
  the whole field. Prefer `ParGridFunction` accessors over manual indexing.
- Diagnostics print on **rank 0 only** (guard with `mfem::Mpi::Root()`); per-rank
  quantities go through reductions (the profiler's max/min pattern) — never interleaved
  per-rank stdout.
- Runtime options (`k_u`, `k_p`, γ, time scheme + adaptive atol/rtol, preconditioner
  choice) and new physics are set at runtime, not via `#ifdef`.
- Formatting is astyle's job — run `scripts/style.sh`, don't hand-format.

## Definition of done

A task is done when: the **CPU build** compiles clean (no warnings), the full fast tier
(unit tests + convergence oracles) passes, any new module carries its unit test, and the plan's baselines are either unchanged or explicitly re-blessed with a note
on why. If a CUDA build is enabled, it must also compile clean and pass parity. If you
can't get the fast tier green, stop and report — do not weaken a tolerance to pass.
