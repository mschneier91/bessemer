# SPEC: Cahouet–Chabard Schur Complement Preconditioner (Consistent BM⁻¹Bᵀ Variant) for Incompressible NSE in MFEM on GPUs

**Project:** bessemer
**Status:** Draft for implementation by Claude Code
**Scope:** Preconditioner + coupled generalized-Stokes solve module. Does NOT replace the existing time integrator; it adds a fully-implicit / coupled solve path alongside it.

---

## 1. Purpose

Implement a matrix-free, GPU-resident block preconditioner for the generalized Stokes
(saddle-point) problem arising from implicit time discretization of the incompressible
Navier–Stokes equations, using the Cahouet–Chabard (CC) structure with the
**consistent mixed operator** `B M_v⁻¹ Bᵀ` in place of the assembled pressure
Laplacian `L_p`, per the robustness findings of Creff & Guermond (2024/2025)
[arXiv:2407.01783; CAMWA 191 (2025) 255–274], **with the following deliberate
deviations from that paper**:

1. **No `εM_Q + L_Q` shift.** Nullspace handling is done properly via projection
   (orthogonalization against the constant mode) and correct AMG configuration.
   The ε-shift perturbs exactly the low-frequency end of the spectrum that
   dominates the CFL-limited Schur complement for enclosed flows. Forbidden in
   this codebase.
2. **Lumped mass is an option, not a requirement.** With tensor-product elements
   and Gauss–Lobatto (GLL) points/quadrature, the velocity mass matrix is diagonal
   by construction (collocation) — that is the preferred default. For consistent
   (non-collocated) mass, a fixed-order Chebyshev/Jacobi application of `M_v⁻¹`
   is used. Simplex-style `|φ|` lumping is a fallback only.
3. **No augmented Lagrangian.** No `λ` term anywhere in v1 (C&G showed it loses
   on CPU-time despite fewer outer iterations; the grad-div block is hostile to
   our solver stack). Leave a documented extension point only.

The preconditioner must be robust (bounded outer iteration counts) with respect to
mesh size `h`, polynomial order `p`, time step `Δt` (including `τ ~ h` CFL scaling),
and viscosity `ν` over at least `ν ∈ [10⁻⁶, 1]`.

---

## 2. Mathematical formulation and conventions (read carefully — sign bugs live here)

### 2.1 Continuous problem

At each implicit stage/step (BDF-k or DIRK, advection treated explicitly or via
outer Picard/Newton — out of scope here), solve the generalized Stokes problem:

```
σ u − ∇·(ν 𝔻(u)) + ∇p = f     in Ω
∇·u = 0                        in Ω
```

with `σ = γ₀/Δt` (γ₀ = leading BDF weight, e.g. 1 for BE, 3/2 for BDF2, 11/6 for BDF3),
and `𝔻(u)` either `∇u` (Laplacian form) or `2 e(u) = ∇u + ∇uᵀ` (symmetric-gradient /
Newtonian form) — selectable, see §5.

### 2.2 Discrete system — canonical convention for this module

Velocity space `V_h ⊂ [H¹]^d` (vector H1, order `p_v`), pressure space `Q_h`
(order `p_q`, continuous H1 Taylor–Hood by default; see §5 for L2/DG variant).

Bilinear forms (all on TRUE dofs after essential-BC elimination):

```
A  := σ M_v + ν K_v                (velocity block; SPD)
B  : V_h → Q_h,  (B u)_k = ∫ ψ_k (∇·u) dx     (VectorDivergenceIntegrator)
M_v := velocity vector mass
M_p := pressure mass
```

**Canonical saddle-point system (symmetric indefinite):**

```
[ A   Bᵀ ] [ u ]   [ f ]
[ B   0  ] [ p̃ ] = [ g ]
```

with **internal pressure `p̃ = −p_physical`**. This makes the system symmetric with
`S := B A⁻¹ Bᵀ` symmetric positive (semi-)definite. The sign flip to physical
pressure happens exactly once, at output/postprocessing. Document this in the
header with a comment block; add a unit test asserting the sign convention against
an analytic solution. `g ≠ 0` arises from essential-BC elimination.

### 2.3 The preconditioner

Schur complement approximate inverse (CC with consistent operator, λ = 0),
**with the actual coefficients spelled out — these are the values, not
placeholders:**

```
P_S⁻¹ := ν M_p⁻¹  +  (γ₀/Δt) (B M_v⁻¹ Bᵀ)⁻¹
```

i.e. the pressure-mass term is scaled by the **viscosity ν**, and the mixed
Poisson term is scaled by **γ₀/Δt** (γ₀ = leading BDF weight; = 1/Δt for
backward Euler; = 0 for steady Stokes, which drops the Poisson term). In code
these appear as `nu_pc` and `sigma`, but they are ν and γ₀/Δt — `nu_pc` exists
only as an O(1) calibration override, never a different physical quantity.

**Verified convention mapping to Creff & Guermond (arXiv:2407.01783, eqs.
(2.2), (2.8), (2.9), (4.4a)) — do not re-derive, this has been checked:**

| This spec | C&G paper | Note |
|---|---|---|
| `b(v,q) = ∫q∇·v`, `B`: vel→pres | identical (their (2.2), (2.7)) | same sign, same orientation |
| `[A Bᵀ; B 0]` on `p̃ = −p_phys` | `[A −Bᵀ; B 0]` on `p` (their (2.8)/(2.14)) | substitute `p̃ = −p`: systems identical |
| `A = σM_v + νK_v`, `σ = γ₀/Δt` | `A = τ⁻¹M_V + μE_V` | `σ ↔ 1/τ`, `ν ↔ μ`; γ₀ generalizes to BDF-k |
| `S = BA⁻¹Bᵀ` (SPD) | `S = BA⁻¹Bᵀ` (their (2.9)) | identical operator |
| `P_S⁻¹ = νM_p⁻¹ + (γ₀/Δt)(BM_v⁻¹Bᵀ)⁻¹` | `C₀ = τ⁻¹(BM_V⁻¹Bᵀ)⁻¹ + μM_Q⁻¹` (their (4.4a), λ=0) | identical; mass↔ν, Poisson↔1/Δt |

- Reaction limit (σ large): `A → σ M_v ⇒ S → σ⁻¹ B M_v⁻¹ Bᵀ` ⇒ Poisson term exact.
- Viscous limit (σ → 0): `S` spectrally equivalent to `ν⁻¹ M_p` ⇒ mass term.
- `nu_pc` default is **ν for BOTH viscous forms**. C&G use the `2μ e(u):e(v)`
  form throughout and still put plain `μ` on the mass term — correctly, since
  on (near-)solenoidal fields `∫2e(u):e(v) = ∫∇u:∇v` (the grad-div part
  vanishes), so the viscous-limit equivalence constant does not pick up the 2.
  `2ν` is retained ONLY as the T3 calibration alternative for the
  sym-gradient form; adopt it only if T3 shows measurably better clustering.
- **Scaling guard (do not "fix" this):** in the periodic constant-coefficient
  case the decomposition is exact — the symbol of `S⁻¹` is `σ/|k|² + ν`. The
  `1/Δt` lives ENTIRELY in `σ = γ₀/Δt` on the Poisson term; the mass term's
  coefficient is the flat high-|k| plateau of the symbol, which is `ν`
  independent of `Δt`. Putting `1/Δt` on the mass term mis-scales the
  high-frequency band by `σ/ν = γ₀/(νΔt)` (~10⁷ at DNS parameters) and
  destroys the preconditioner. `nu_pc` is an O(1) dial around `ν`, never a
  function of `Δt`.
- All inner applications are inexact (it is a preconditioner). Inexactness is
  fine; *structural* wrongness (L_p in place of BM⁻¹Bᵀ, ε-shifts, wrong
  coefficient scalings) is not.
- One deliberate upgrade over C&G's implementation: they solve `BM_V⁻¹Bᵀ`
  systems with GMRES because their `M_V⁻¹` is an AMG-based approximate solve
  (not a fixed symmetric operator). Ours is a diagonal multiply or fixed
  Chebyshev — exactly symmetric and fixed — which is what legalizes CG (§2.4).

**Legacy/comparison mode:** `P_S⁻¹ = ν_pc M_p⁻¹ + σ L_p⁻¹` must also be
implementable behind the same interface, used ONLY for regression test T5
(demonstrating its degradation on fine meshes / tight tolerances) and for cheap
early-development smoke tests. Never the default.

### 2.4 Solver structure — there is exactly one

The outer Krylov loop runs on the **full block system** `[A Bᵀ; B 0]`. Note the
per-application work is segregated by construction: each triangular-preconditioner
application traces exactly like an inexact Uzawa sweep (pressure/Schur solve,
then velocity solve against the corrected residual), and triangular-preconditioned
Richardson IS inexact Uzawa — this design is Krylov-accelerated inexact Uzawa.
"Monolithic" here refers only to (i) where the residual is measured — the coupled
(u,p) residual, so divergence is driven to the outer tolerance with no splitting
error — and (ii) where the Krylov acceleration happens, which is what permits
loose, iteration-varying inner solves (flexibility absorbs the inexactness).
The alternative — Krylov on the Schur complement alone — requires each `S`-matvec,
hence each embedded `A`-solve, to be consistent to Krylov accuracy
(Rusten–Winther), and is not implemented as a production mode; the T3 spectrum
tests construct `P_S⁻¹S` directly as a test utility with tight A-solves. The
outer solver is **FGMRES, always** — the inner components are
tolerance/iteration-limited solves, which makes the preconditioner nonlinear, so
flexibility is non-negotiable. MINRES is not implemented and must not be added.

Full solver hierarchy (this diagram is normative):

```
FGMRES — monolithic [A Bᵀ; B 0]                        (the ONLY outer Krylov)
 └─ block upper-triangular preconditioner
     P_block = [ Â   Bᵀ ]        (also provide diag / lower-tri via config)
               [ 0   −Ŝ ]
     ├─ Â⁻¹  : fixed V-cycle(s), p-MG / LOR-AMG         (fixed linear op; no Krylov)
     └─ Ŝ⁻¹ = P_S⁻¹:
         ├─ ν_pc M_p⁻¹ : direct diagonal multiply if M_p diagonal,
         │               else fixed-order Chebyshev      (fixed linear op; no Krylov)
         └─ σ (BM_v⁻¹Bᵀ)⁻¹ : CG                          (SPD ⇒ CG, never GMRES)
             ├─ operator: B ∘ M_v⁻¹ ∘ Bᵀ with M_v⁻¹ = direct diagonal multiply
             │            or fixed-order Chebyshev       (fixed linear op)
             └─ PC: symmetric AMG V-cycle on LOR L_p     (Ortho-wrapped if singular)
```

**Inner-solver selection rule:** CG wherever the operator + preconditioner pair
is SPD; GMRES only where symmetry is genuinely unavailable. In the default
configuration nothing below the outer FGMRES needs GMRES. The complete
step-by-step execution trace of one solve is normative in §6.5.

**Krylov correctness constraints (enforce in code, not just docs):**

- Inside the inner CG on `B M_v⁻¹ Bᵀ`, the operator must be a *fixed linear
  operator* per application ⇒ `M_v⁻¹` inside it must be a direct diagonal
  multiply or fixed-order Chebyshev — never a tolerance-based solve. Config
  validation throws on violation.
- CG legality also requires the inner Poisson PC to be an SPD application:
  the `L_p` AMG must use a symmetric relaxation (Chebyshev, ℓ1-Jacobi, or
  symmetric-GS) with matched pre/post sweep counts. Assert the BoomerAMG relax
  settings at setup; a nonsymmetric V-cycle silently breaks CG.

---

## 3. Operator inventory and MFEM implementation notes

All operators live on TRUE dofs, GPU-resident, `AssemblyLevel::PARTIAL` unless
stated. Target MFEM ≥ 4.7 with CUDA or HIP device build; hypre built with GPU
support for the LOR/AMG component.

| Operator | MFEM construction | Assembly | Inverse application |
|---|---|---|---|
| `A = σM_v + νK_v` | `ParBilinearForm` on vector H1: `VectorMassIntegrator(σ)` + `VectorDiffusionIntegrator(ν)` or `ElasticityIntegrator`-equivalent for sym-gradient | PA | never inverted exactly; see Â §6.3 |
| `B` | `ParMixedBilinearForm(vel_fes, pres_fes)` + `VectorDivergenceIntegrator` | PA if supported (VERIFY — see note below), else FA fallback | n/a (rectangular) |
| `Bᵀ` | `MultTranspose` of the same operator | same | n/a |
| `M_v` | `VectorMassIntegrator` | PA | GLL-diag (default) / Chebyshev(k) / |φ|-lumped fallback |
| `M_p` | `MassIntegrator` on pressure space | PA | Chebyshev(k) w/ Jacobi (default); exact block-diag if DG pressure |
| `L_p` (inner PC only) | `DiffusionIntegrator` on pressure space | LOR-assembled → `HypreParMatrix` | 1 (or fixed n) BoomerAMG V-cycle via `LORSolver<HypreBoomerAMG>`; p-MG alternative |
| `BM_v⁻¹Bᵀ` | composed `Operator` (custom class `MixedPoissonOperator`) | matrix-free composition — NEVER assembled | inner CG, preconditioned by L_p-AMG (+ Ortho wrapper when singular) |

**Implementation verification tasks (do these first, they gate design):**

1. Check whether `VectorDivergenceIntegrator` supports `AssemblyLevel::PARTIAL`
   (and device execution) in the MFEM version pinned by bessemer, including
   `MultTranspose`. If yes: use `FormRectangularSystemOperator` to get the
   RAP/eliminated `RectangularConstrainedOperator`. If no: fallback order is
   (a) custom sum-factorized element kernel for div/grad action (tensor elements
   only), (b) fully-assembled `HypreParMatrix` for `B` alone (memory-check at
   p ≥ 6 in 3D before accepting; log a warning).
2. Check hypre device build (`HYPRE_USING_GPU`). If hypre is CPU-only, the L_p
   AMG runs on host with per-application device↔host transfers: allow with a
   loud startup warning (perf cliff), and prefer the p-MG alternative (§6.2) in
   that configuration.
3. Confirm `OrthoSolver` (mean-subtraction wrapper) is device-safe in the pinned
   MFEM version; if not, implement a 20-line device-side projector
   `q ← q − (1ᵀq/1ᵀ1) 1` (see §4 for the correct inner product).
4. Verify LOR mesh/space construction and `LORSolver<HypreBoomerAMG>` on a
   periodic mesh (periodic channel + fully periodic box). If unsupported or
   buggy in the pinned version, `lp_pc = PMG` becomes the default on periodic
   meshes (log which path was selected).

**Essential-BC consistency (classic bug source):** `A`, `B`, `Bᵀ` must all be
eliminated against the SAME essential velocity true-dof list. `B`'s columns at
essential velocity dofs are zeroed (that is what `RectangularConstrainedOperator`
does). The Schur operator and the preconditioner's `BM⁻¹Bᵀ` must use the
*eliminated* `B`; using an uneliminated `B` inside the preconditioner while the
system uses the eliminated one silently degrades convergence near boundaries.
Unit test T1c covers this.

**Quadrature:** the system operators use bessemer's standard rules. Inside the
preconditioner, cheaper rules are legal (it's a preconditioner) but keep the
default identical to the system rules to avoid confusing calibration; expose
`pc_quadrature = {inherit, gll_collocated}`.

---

## 4. Nullspace handling (replaces the ε-shift)

Applies when the pressure is defined up to a constant: all velocity boundary dofs
essential and no traction/outflow boundary ⇒ `Bᵀ·𝟙 = 0` where `𝟙` is the
coefficient vector of the constant pressure function (all-ones for nodal H1
Lagrange; for L2/DG spaces with non-nodal bases, build 𝟙 by projecting the
constant-1 function — do NOT assume all-ones).

**Detection:** `nullspace = auto` (default): singular iff **no traction/outflow
BC is registered on any boundary attribute that actually exists in the mesh**.
Do NOT phrase the check as "essential attrs cover the whole boundary" — periodic
meshes break that formulation (see below). Also expose `force_on / force_off`.

### 4.1 Periodic boundaries (first-class, not an afterthought)

MFEM periodic meshes (`Mesh::MakePeriodic`) identify dofs at mesh/space
construction: periodic directions contribute **no boundary elements** and take
**no essential BCs**. Consequences this module must handle:

- **Detection with partial or empty boundary.** Periodic channel (walls
  essential, streamwise/spanwise periodic): the only existing boundary
  attributes are the walls, all essential, no outflow ⇒ singular (pressure
  constant mode) — the detection rule above gets this right. Fully periodic
  box (TGV): the boundary-element set is EMPTY; a vacuous "all boundary
  essential" test must resolve to **singular**, not crash on empty attribute
  arrays. Unit-test both.
- **Fully periodic + σ = 0 is ill-posed for this module:** with no essential
  dofs anywhere, `A = νK_v` carries the `d` constant-velocity modes in its
  kernel. v1 behavior: if the mesh has an empty boundary-element set AND no
  essential velocity dofs AND `σ == 0`, throw with a clear message
  ("fully periodic steady Stokes requires velocity nullspace handling — not
  supported"). Velocity-nullspace projection is out of scope.
- **Operators inherit identification automatically** through the FE space's
  true-dof maps — `B`, `Bᵀ`, mass diagonals, `L_p` need no special code. Note
  the mass diagonal stays exactly diagonal under identification (entries
  accumulate across identified nodes); the `diag_direct` fast path is unaffected.
- **Mean-pressure-gradient / fixed-flow-rate forcing** for periodic channel DNS
  lives entirely in the RHS (bessemer's forcing machinery); no preconditioner
  involvement. But T6 must include the periodic-channel case so the singular
  path is exercised with walls present.
- **Verify LOR on periodic meshes** in the pinned MFEM version (added to the §3
  verification list); if broken, the `lp_pc = PMG` alternative is the periodic
  fallback.

**Treatment (all three together):**

1. Project the RHS of every singular solve onto `𝟙⊥` before iterating.
2. Wrap the inner CG on `BM⁻¹Bᵀ` and the outer application in a projector that
   re-orthogonalizes the iterate every `k_reproj` iterations (default 5) to
   control floating-point drift — CG on a singular consistent SPD system is
   correct in exact arithmetic but drifts in fp64 on ill-conditioned fine meshes.
3. AMG on the pure-Neumann `L_p`: keep the operator singular (no pinning, no
   shift). BoomerAMG tolerates this as a preconditioner when RHS/iterates are
   projected; additionally set a smoother-only or SVD-capable coarse solve if
   available. Provide `pin_dof` ONLY as a debug option behind a flag with a
   warning (it wrecks h-independence locally); never default.

**Projection inner product:** for nullspace removal of `BM⁻¹Bᵀ` (an SPD operator
on coefficient vectors), the nullspace vector is `𝟙` in the *l2* sense on true
dofs — plain Euclidean projection is correct. Do not mass-orthogonalize here.
(Mass-weighted mean is only relevant when reporting the physical mean-zero
pressure field at output.)

When a traction/outflow boundary exists, the system is nonsingular; the L_p used
inside the inner PC should carry homogeneous Dirichlet on the pressure boundary
dofs associated with the outflow attribute(s) and Neumann elsewhere
(`lp_bc = {auto, all_neumann, dirichlet_on_attrs[...]}`, default `auto`).

---

## 5. Discretization options

- **Default: Taylor–Hood** `[H1]^d` order `p` velocity / H1 order `p−1` pressure,
  hex (`Q_p/Q_{p−1}`) and tet (`P_p/P_{p−1}`), 2D and 3D, curved isoparametric
  meshes supported (PA handles geometric factors; LOR supports curved meshes).
  `p ∈ [2, 8]` must work; performance target is `p ∈ [4, 8]` on hexes.
- **Discontinuous (L2) pressure — TODO, not v1.** See §12. Do not build
  speculative abstractions for it; the only accommodation in v1 is keeping the
  three pressure-space-dependent pieces (`M_p⁻¹` strategy, `L_p` construction,
  `𝟙` construction) in separable functions rather than inlined, so the variant
  is additive later.
- **Viscous form:** `visc_form = {laplacian (default), sym_gradient}`. For
  `sym_gradient`, the Â preconditioner for the A-block still DROPS the
  cross-component coupling (C&G's Ã₃ finding): precondition with the
  block-diagonal vector-Laplacian `σM_v + νL_v`. Note grad-grad and sym-gradient
  are equivalent only for enclosed `[H¹₀]^d`; with outflow they differ and the
  traction BC meaning changes — document, don't silently convert.
- Variable viscosity (LES eddy viscosity) is a v2 hook: option
  `pc_mass_coeff = {const_nu, reciprocal_nu_field}` reserved in the config enum
  but only `const_nu` implemented in v1 (use ν = molecular or a supplied scalar
  `nu_pc`); assert-fail on the other value with a clear message.

---

## 6. Preconditioner algorithm (pseudocode + components)

### 6.1 `P_S⁻¹` application (`CahouetChabardSchurPC : mfem::Solver`)

```
Input:  r_p  (pressure-space residual, true dofs)
Output: z_p

1. z_mass ← ApplyMpInv(r_p)                      // direct diagonal multiply if M_p diagonal, else Chebyshev(k_mp)
2. if σ > 0:
     rhs ← r_p
     if singular: rhs ← Project⊥𝟙(rhs)
     w ← InnerCG( BMvInvBt, rhs;
                  pc = L_p-AMG (Ortho-wrapped if singular),
                  fixed_iters = n_inner  OR  rtol = tol_inner,   // config, default fixed_iters
                  reproject every k_reproj if singular )
   else: w ← 0                                    // steady Stokes limit: pure mass PC
3. z_p ← ν_pc · z_mass + σ · w
```

- `BMvInvBt::Mult(x) = B( MvInv( Bᵀ x ) )` with `MvInv` one of:
  - `diag_direct` (default on tensor elements with a collocated GLL basis/rule,
    where `M_v` is exactly diagonal): **precompute the reciprocal diagonal once
    at setup (`d_inv`), and apply as a single fused elementwise device multiply
    `y_i = d_inv_i · x_i`.** This path must NOT construct or route through any
    `IterativeSolver`, smoother wrapper, or one-iteration Chebyshev — it is a
    vector multiply, nothing more. Debug builds assert zero solver objects on
    this path; T1f verifies its cost ≈ one vector op. The same fast path applies
    to `M_p⁻¹` whenever the pressure basis/quadrature makes `M_p` diagonal.
  - `chebyshev(k_mv)` (default for consistent, non-collocated mass): fixed-order
    Chebyshev with Jacobi (`OperatorJacobiSmoother` diagonal), eigenvalue bounds
    from a few power iterations at setup, cached.
  - `abs_lumped`: `Λ_ii = ∫|φ_i|` fallback (needed for simplices where standard
    lumping produces nonpositive weights at p ≥ 2); applied via the same
    direct-diagonal fast path.
  - Resolution rule for `Auto`: inspect whether the assembled/PA mass diagonal
    IS the full operator (collocation) — if yes, `diag_direct`; else `chebyshev`.
    Requesting `diag_direct` on a non-diagonal mass throws at validation.
- `σ = 0` short-circuit: skip step 2 entirely (avoid a 0·(singular solve)).
- Very large σ (tiny Δt): fine as-is; scale-balance by applying the two terms
  as written (no premature factoring) and verify no catastrophic cancellation in
  T3's extreme corners.

### 6.2 Inner Poisson PC (`L_p` surrogate)

- Default: `LORSolver<HypreBoomerAMG>` on the assembled LOR `L_p`
  (`DiffusionIntegrator`, pressure space), 1 V-cycle per application
  (`lp_vcycles`, default 1), strong threshold `0.25` 3D / `0.1` 2D as starting
  points (tunable).
- Alternative (`lp_pc = pmg`): p-multigrid on the pressure space with Chebyshev
  smoothers and an assembled lowest-order AMG coarse solve — preferred when
  hypre lacks device support.
- BCs per §4. The operator stays singular in the enclosed case.

### 6.3 A-block inexact inverse `Â⁻¹`

- `Â = σM_v + νL_v` block-diagonal-by-component (drop sym-gradient coupling).
- Application: fixed cost. Default `a_pc = pmg_chebyshev` (p-MG, Chebyshev
  smoothing, LOR-AMG coarse) with `a_vcycles = 1`; alternatives:
  `lor_amg(n_vcycles)`, `jacobi_chebyshev(k)` (cheap, for σ-dominated regimes).
- NEVER a tolerance-based CG inside the block preconditioner — `Â⁻¹` is a fixed
  linear operator (fixed V-cycle count) by design; a nested tolerance solve
  inflates cost and adds nothing the outer FGMRES can't absorb.

### 6.4 Outer solver

FGMRES on the monolithic system, full stop (§2.4). Restart 50–100 (config),
abs+rel tol config, monitor per-block residual norms (momentum vs continuity)
each iteration when `verbosity ≥ 2`. No other outer solver exists in this module.

### 6.5 End-to-end anatomy of one coupled solve (normative trace)

This section is the ground truth for what executes, in what order. Start from
the assembled (eliminated, true-dof) block system — no derivation needed:

```
𝒜 x = b,    𝒜 = [ A   Bᵀ ],   x = [ u ],   b = [ f ]      A = σM_v + νK_v (SPD)
                [ B   0  ]        [ p̃ ]        [ g ]      p̃ = −p_physical (§2.2)
```

**Setup phase (once per (σ, ν) pair; reused across time steps via `Reset`):**

```
S0. Detect nullspace (§4/§4.1) → singular flag; build 𝟙 if singular
S1. Extract PA diagonals: diag(M_v), diag(M_p), diag(Â-levels for smoothers)
S2. If mass diagonal (collocated): store d_inv vectors        → diag_direct path
    else: power-iterate eigenvalue bounds, cache Chebyshev coefficients
S3. Assemble LOR L_p (HypreParMatrix) + BoomerAMG setup (symmetric relaxation)
S4. Build Â p-MG hierarchy: Chebyshev smoothers per p-level, LOR-AMG coarse setup
```

**Solve phase — right-preconditioned FGMRES** (flexible preconditioning is
inherently right-sided: we solve `𝒜 P⁻¹ y = b`, recover `x = P⁻¹ y`; the
monitored residual `b − 𝒜x` is the TRUE coupled residual, unpolluted by the
preconditioner — this is what "divergence driven to outer tolerance" means).

Per outer iteration `j`:

```
F1. OPERATOR APPLICATION  v ← 𝒜 z_j
      momentum row:   A z_u  (PA mass+diffusion action)  +  Bᵀ z_p  (PA gradient action)
      continuity row: B z_u  (PA divergence action)
F2. PRECONDITIONER APPLICATION  z_{j+1} ← P⁻¹ r      [the segregated sweep, §2.4]
      (detailed below)
F3. Arnoldi/MGS orthogonalization; store BOTH v-basis and z-basis
      (flexible ⇒ 2 full-size (u,p) vectors per iteration ⇒ restart matters)
F4. Convergence check on true residual; per-block ‖r_u‖, ‖r_p‖ at verbosity ≥ 2
```

**F2 expanded — backward substitution through `P = [Â Bᵀ; 0 −Ŝ]`, given
`r = [r_u; r_p]`:**

```
── STAGE 1: pressure/Schur block ────────────────────────────────────────────
P1.  z_p ← −P_S⁻¹ r_p        (note the minus: row 2 of P reads −Ŝ z_p = r_p)

     P_S⁻¹ r_p breaks into two additive terms (§2.3):

     P1a. MASS TERM      t ← M_p⁻¹ r_p
            diag_direct:  t_i = d_inv_i · r_p,i          (one fused device kernel)
            else:         k_mp Chebyshev steps, each = 1 M_p PA action + diag scalings

     P1b. POISSON TERM   (skipped entirely if σ = 0)
            if singular:  r̂ ← r_p − (𝟙ᵀr_p/𝟙ᵀ𝟙)𝟙       (l2 projection, §4)
            else:         r̂ ← r_p
            INNER CG on  (B M_v⁻¹ Bᵀ) w = r̂ ,  fixed n_inner iters (default 10):
              per CG iteration:
                i.   operator matvec:  q ← B ( M_v⁻¹ ( Bᵀ s ) )
                       Bᵀ action (PA) → M_v⁻¹ (diag_direct multiply, default;
                       else k_mv Chebyshev, each = 1 M_v PA action) → B action (PA)
                ii.  PC application:   1 symmetric BoomerAMG V-cycle on LOR L_p
                       (Ortho-wrapped if singular)
                iii. 2 global reductions (α, β) + axpys
                iv.  if singular and (iter % k_reproj == 0): re-project iterate ⊥ 𝟙

     P1c. COMBINE        z_p ← −( ν · t + (γ₀/Δt) · w )
            (the minus is row 2 of the triangular P; the coefficients are the
             viscosity and γ₀/Δt LITERALLY — `nu_pc`/`sigma` in code are these
             values, see §2.3)

── STAGE 2: velocity block ──────────────────────────────────────────────────
P2.  r_u' ← r_u − Bᵀ z_p     (one Bᵀ PA action)
P3.  z_u  ← Â⁻¹ r_u'         = a_vcycles p-MG V-cycle(s), fixed count, no Krylov:
       per V-cycle: Chebyshev pre-smooth at each p-level (each step = 1 Â-level
       PA action), p-restriction → … → coarse: 1 AMG V-cycle on lowest-order LOR
       → p-prolongation, Chebyshev post-smooth. Component-block-diagonal Â
       (σM_v + νL_v per component) even in sym_gradient mode (§5).

Return z = [z_u; z_p].
```

**Operation count per outer FGMRES iteration (default config: diag masses,
n_inner = 10, a_vcycles = 1):**

| Kernel | Count | Origin |
|---|---|---|
| A PA action (mass+diffusion) | 1 | F1 |
| B PA action | 1 + n_inner | F1 + P1b.i |
| Bᵀ PA action | 2 + n_inner | F1, P2 + P1b.i |
| M_v⁻¹ diag multiply | n_inner | P1b.i |
| M_p⁻¹ diag multiply | 1 | P1a |
| L_p AMG V-cycle | n_inner | P1b.ii |
| Â p-MG V-cycle (incl. smoothing actions) | 1 | P3 |
| Global reductions | (j+2) MGS + 2·n_inner CG | F3, P1b.iii |

The inner CG (n_inner × [B + Bᵀ + AMG V-cycle]) dominates the preconditioner
cost; this is the knob to tune first (n_inner vs outer iteration count trade),
and the NVTX ranges in §9 are drawn around exactly these lines.

**Degenerate paths, for completeness:** σ = 0 ⇒ P1b skipped, P_S⁻¹ = ν_pc M_p⁻¹
(pure mass) — trace contains no inner CG. Singular case ⇒ projections appear at
exactly three places: P1b RHS, every k_reproj inner iterations, and inside the
Ortho-wrapped AMG. `LaplacianLegacy` comparison mode ⇒ P1b's inner CG is replaced
by fixed V-cycle(s) on LOR L_p directly (no CG, no B/Bᵀ/M_v⁻¹ actions) — cheaper
per application, non-robust (T5).

---

## 7. Configuration surface (single POD struct, all fields listed)

```cpp
struct CahouetChabardConfig {
  // structure — outer solver is FGMRES on the monolithic system, not configurable
  BlockPCShape   block_shape     = BlockPCShape::UpperTri;    // Diag | LowerTri | UpperTri
  SchurModel     schur_model     = SchurModel::ConsistentBMB; // ConsistentBMB | LumpedBMB | LaplacianLegacy
  // physics/scaling
  double         sigma;                    // γ0/Δt; 0 ⇒ steady Stokes
  double         nu;                       // constant viscosity used in system
  double         nu_pc            = -1.0;  // <0 ⇒ ν for BOTH viscous forms (matches C&G (4.4a));
                                           // 2ν only as T3 calibration alternative for sym_gradient
  ViscousForm    visc_form        = ViscousForm::Laplacian;   // Laplacian | SymGradient
  PcMassCoeff    pc_mass_coeff    = PcMassCoeff::ConstNu;     // ConstNu | ReciprocalNuField(v2)
  // inner components
  MassInvType    mv_inv           = MassInvType::Auto;  // Auto | DiagDirect | Chebyshev | AbsLumped
  int            k_mv_chebyshev   = 4;                  // Auto → DiagDirect iff M_v exactly diagonal
  MassInvType    mp_inv           = MassInvType::Auto;  // same resolution rule for M_p
  int            k_mp_chebyshev   = 3;
  InnerStop      inner_stop       = InnerStop::FixedIters;    // FixedIters | RelTol
  int            n_inner          = 10;    // inner CG iters on BM⁻¹Bᵀ
  double         tol_inner        = 1e-3;  // if RelTol
  LpPC           lp_pc            = LpPC::LORAMG;             // LORAMG | PMG
  int            lp_vcycles       = 1;
  LpBC           lp_bc            = LpBC::Auto;               // Auto | AllNeumann | DirichletOnAttrs
  mfem::Array<int> lp_dirichlet_attrs;
  // A-block
  APC            a_pc             = APC::PMGChebyshev;        // PMGChebyshev | LORAMG | JacobiChebyshev
  int            a_vcycles        = 1;
  // nullspace
  Nullspace      nullspace        = Nullspace::Auto;          // Auto | ForceOn | ForceOff
  int            k_reproj         = 5;
  bool           allow_pin_dof    = false; // debug only; warn loudly
  // outer solver
  double         outer_rtol       = 1e-8;
  double         outer_atol       = 0.0;
  int            outer_maxit      = 200;
  int            fgmres_restart   = 100;
  // execution
  Precision      pc_precision     = Precision::FP64;          // FP64 | FP32PC (v2, reserved)
  int            verbosity        = 1;     // 0 silent; 1 summary; 2 per-iter block residuals; 3 inner traces
  bool           collect_spectrum = false; // enable Lanczos estimation hooks (T3)
};
```

Config validation at construction (fail fast, clear messages):
`schur_model == LaplacianLegacy` ⇒ warn "comparison mode, known non-robust";
`sigma < 0` ⇒ error; `nu <= 0` ⇒ error; `DiagDirect` requested on a non-diagonal
mass ⇒ error; any tolerance-based solve nested inside the inner-CG operator ⇒
error; nonsymmetric L_p AMG relaxation with inner CG ⇒ error; `pin_dof` ⇒ warn;
L2/DG pressure space ⇒ error "not supported in v1, see §12".

---

## 8. Edge cases and required behaviors (checklist — each maps to a test)

1. **Enclosed flow** (all-essential velocity boundary): singular path exercised
   end-to-end; converged pressure reported mean-zero (mass-weighted mean at
   output). [T6a lid cavity]
2. **Outflow present:** nonsingular path; `lp_bc = Auto` puts Dirichlet on the
   outflow pressure boundary. [T6b channel]
3. **σ = 0 (steady Stokes):** pure-mass PC path; iterations bounded. [T4b]
4. **Extreme Δt:** σ ∈ {1e−6 … 1e6} sweep must not NaN/overflow; iteration counts
   transition smoothly between the two limits. [T3]
5. **ν sweep** {1, 1e−2, 1e−4, 1e−6} at τ ~ h. [T3, T5]
6. **Order sweep** p = 2…8 hex, p = 2…4 tet; 2D and 3D. [T3]
7. **Curved meshes** (isoparametric ≥ quadratic geometry). [T4 on a curved
   annulus/pipe]
8. **Nonzero essential BC data** (lifting): PC operates on the homogeneous
   residual system; verify no BC-value dependence of iteration counts. [T1c]
9. **Eliminated-B consistency** (see §3): regression test comparing eliminated
   vs raw B inside the PC. [T1c]
10. **Parallel edge partitions:** ranks owning no boundary dofs; ranks with zero
    pressure dofs after partitioning (small meshes, many ranks); deterministic
    behavior of the nullspace detection collective. [T7a]
11. **Nonconforming AMR meshes:** should work through true-dof prolongations +
    PA; mark experimental, add one smoke test, defer performance to v2. [T7b]
12. **Krylov legality:** a tolerance-based mass inverse nested inside the
    inner-CG operator, or a nonsymmetric L_p AMG relaxation paired with inner
    CG, must throw at config validation. [T1d]
13. **Singular-CG drift:** long inner iteration on fine mesh with reprojection
    disabled must show drift; with `k_reproj = 5` must not. [T2b]
14. **Legacy L_p mode degradation:** on the two finest meshes at tol 1e−10,
    `LaplacianLegacy` shows iteration growth where `ConsistentBMB` stays flat —
    this is the guard that encodes WHY the default is what it is. [T5]
15. **hypre-CPU fallback:** device build with host-only hypre runs correctly
    (warn), and `lp_pc = PMG` avoids the transfers. [T7c]
16. **Memory guard:** refuse (or warn + require override) to fully assemble `B`
    in 3D at p ≥ 6; never assemble `BM⁻¹Bᵀ` under any configuration (no code
    path may exist). [code review + T1e assert]
17. **Periodic channel** (essential walls + 1–2 periodic directions): nullspace
    detection resolves singular; solve converges; iteration counts match the
    enclosed-cavity behavior. [T6d]
18. **Fully periodic box** (empty boundary-element set): detection resolves
    singular without touching empty attribute arrays; `σ = 0` throws per §4.1;
    `σ > 0` solves cleanly (TGV). [T6c + unit test]
19. **Diagonal-mass fast path:** debug assert that no solver/smoother object is
    constructed on the `diag_direct` path; profile check that `M⁻¹` application
    cost ≈ one vector multiply. [T1f + T7 profiling]

---

## 9. GPU requirements

- `mfem::Device` configured `cuda`/`hip`; every `Vector` in the hot path device-
  resident; zero host↔device transfers per PC application in the default config
  (assert via counter in debug builds; hypre-CPU fallback exempted with warning).
- All PA kernels sum-factorized on tensor elements (MFEM handles this; verify
  the div/grad path in the §3 verification task).
- Setup-time work (diagonals, Chebyshev eigenvalue estimates, LOR assembly, AMG
  setup) done once per (σ, ν) pair; provide `Reset(sigma, nu)` that rebuilds
  ONLY σ/ν-dependent scalars and the A-block PC, reusing B, mass diagonals,
  and AMG hierarchies where valid (L_p is σ/ν-independent — reuse always).
- Profiling hooks (NVTX ranges or MFEM annotations) around: outer matvec,
  Â-apply, M_p-inv, inner CG total, L_p-AMG apply. Report a per-step time
  breakdown at `verbosity ≥ 1`.
- Performance targets (acceptance, on one modern data-center GPU, 3D hex p=4–6,
  ~5–20M velocity dofs): PC application ≤ 3× the cost of one coupled-operator
  matvec; end-to-end coupled solve time recorded and compared against bessemer's
  projection step on the same mesh (expect the ~O(30×) gap of C&G — we are
  buying divergence control and stiff-stable stepping, not raw speed; record,
  don't fail on this ratio).

---

## 10. Testing and validation plan

**T1 — Operator sanity (unit, tiny meshes, CPU+GPU):**
 a. Adjoint: `⟨Bu, q⟩ = ⟨u, Bᵀq⟩` to 1e−13 (random vectors).
 b. Nullspace: `‖Bᵀ𝟙‖ ≤ 1e−13` enclosed; `> 0` with outflow. Symmetry of
    `BM⁻¹Bᵀ` via random-vector adjoint test; no negative Ritz values below −1e−12.
 c. Eliminated-B consistency + BC-lifting independence (edge cases 8, 9).
 d. Config-validation throws (edge case 12): tolerance-based mass inverse
    inside the inner-CG operator rejected; `DiagDirect` on a non-diagonal mass
    rejected; nonsymmetric L_p relaxation with inner CG rejected.
 e. PA vs full-assembly agreement at p ≤ 3 on a coarse mesh, rel. err ≤ 1e−12.
 f. Diagonal fast path (edge case 19): `diag_direct` application is bitwise
    `y_i = d_inv_i · x_i` (compare against explicit multiply); debug-build
    counter confirms zero solver/smoother constructions on this path; timing
    check that cost is within ~2× of a raw vector multiply.

**T2 — Singular-system correctness:** enclosed-domain inner solves converge to
the projected solution; drift test (edge case 13).

**T3 — Spectrum/robustness sweep (the core scientific test):** Lanczos estimate
of `κ(P_S⁻¹ S)` (with `S = B A⁻¹ Bᵀ` built directly as a test-only utility using
tight CG A-solves, small-to-mid meshes) over the full (ν, σ, h, p) grid of §8
items 4–6; PLUS outer FGMRES iteration tables on the monolithic system on larger
meshes with τ ~ h. Acceptance: iteration counts bounded and
flat (±30%) under refinement for each ν; `nu_pc` calibration for the sym-gradient form: default ν stands unless 2ν
shows measurably better clustering — record the evidence in the
test output.

**T4 — Accuracy (MMS):**
 a. Unsteady Stokes manufactured trig solution (C&G-style, div-free and
    non-div-free variants): velocity `O(h^{p+1})`, pressure `O(h^p)` in L2.
 b. Steady Stokes (σ = 0) MMS.
 c. Sign-convention test: analytic pressure recovered with correct physical sign.

**T5 — Legacy-mode regression (edge case 14):** side-by-side iteration histories,
`ConsistentBMB` vs `LaplacianLegacy`, finest two meshes, rtol 1e−10, ν sweep.
Assert consistent mode ≤ some bound AND legacy mode exceeds it (guards the
design decision permanently).

**T6 — Benchmarks:** (a) 3D lid-driven cavity (enclosed/singular path, steady via
σ=0 and pseudo-transient); (b) 2D/3D channel + backward-facing step with outflow
(nonsingular path); (c) integration test inside bessemer's time loop on the
Taylor–Green vortex at modest Re with IMEX advection — triply periodic, so this
is also the empty-boundary detection test (edge case 18); verify divergence norm
of the coupled solve is at solver tolerance (vs the splitting-error level of the
projection path) and temporal order is preserved; (d) periodic plane channel
(essential walls + periodic streamwise/spanwise, body-force driven): singular
path with walls present (edge case 17); laminar Poiseuille recovery as the
correctness check, iteration counts compared against the enclosed cavity.

**T7 — Platform:** (a) parallel edge partitions; (b) AMR smoke; (c) hypre-CPU
fallback; (d) CPU-vs-GPU agreement: identical iteration counts ±1 and final
residuals within 10× tolerance (bitwise equality is NOT expected — reductions).

---

## 11. Deliverables / file layout (within bessemer)

```
src/precond/cahouet_chabard.hpp/.cpp      // CahouetChabardSchurPC, config, validation
src/precond/mixed_poisson_op.hpp          // BMvInvBt operator (matrix-free composition)
src/precond/mass_inverse.hpp              // GLLDiag / Chebyshev / AbsLumped strategies
src/precond/block_stokes_pc.hpp           // BlockDiag/Tri wrappers, monolithic FGMRES driver
src/precond/nullspace.hpp                 // detection, projector, Ortho wrapper
src/precond/lp_surrogate.hpp              // LOR-AMG / p-MG Poisson PC, BC logic
tests/precond/t1_*.cpp … t7_*.cpp         // per §10, wired into ctest
docs/precond_cc.md                        // math conventions (§2 verbatim), usage, tuning notes
```

Update `CLAUDE.md`: add the sign convention (§2.2), the invariants "no ε-shift /
no assembled BM⁻¹Bᵀ / outer solver is FGMRES on the monolithic system, always /
diagonal mass inverses are direct multiplies, never wrapped in solvers", and the
config struct location.

## 12. Non-goals / future hooks (document, do not implement)

- **Discontinuous (L2) pressure, SEM-style `Q_p/Q_{p−2}` — TODO.** Blocked in
  part by MFEM's LOR support for high-order discontinuous spaces being immature;
  revisit when that lands. When it does: `M_p` becomes block-diagonal (exact
  local inverse via the diag/block fast path), the inner-PC Laplacian needs an
  interior-penalty DG form or an H1 auxiliary-space surrogate, `𝟙` must be
  built by projecting the constant function (not assumed all-ones), and `B` is
  structurally unchanged. v1 throws on L2 pressure spaces (§7 validation).
- Augmented Lagrangian (λ > 0) — extension point in config enum only.
- PCD/LSC convection-aware Schur approximations for high-Re large-Δt implicit
  advection — the CC structure degrades there by construction; note in docs.
- Variable-viscosity (LES) reciprocal-ν pressure mass — reserved enum.
- FP32 preconditioner path — reserved enum.

## 13. References

- Cahouet & Chabard, *Some fast 3D finite element solvers for the generalized
  Stokes problem*, IJNMF 8 (1988) 869–895.
- Creff & Guermond, *Preconditioning of the generalized Stokes problem…*,
  arXiv:2407.01783; Computers & Math. with Appl. 191 (2025) 255–274.
  (Adopted: consistent `BM⁻¹Bᵀ` over `L_p`; drop grad-div coupling in Â.
  Rejected: ε-shift; lumping-as-necessity; AL variants.)
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*, 2005
  (Schur complement spectral bounds; PCD context).
- Rusten & Winther, SIAM J. Matrix Anal. Appl. 13 (1992) (exactness requirement
  for Schur-iteration Krylov; motivates the monolithic-solve-only design).
