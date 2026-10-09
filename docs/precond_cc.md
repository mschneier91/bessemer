# Cahouet–Chabard preconditioner — math conventions, usage, tuning

Implements [`design/SPEC_cahouet_chabard_mfem.md`](design/SPEC_cahouet_chabard_mfem.md): the consistent
`B M_v⁻¹ Bᵀ` Cahouet–Chabard Schur preconditioner for the generalized Stokes
problem, per Creff & Guermond [arXiv:2407.01783; CAMWA 191 (2025) 255–274],
with the spec's deliberate deviations (no ε-shift, lumping optional not
required, no augmented Lagrangian). This file records the **normative
conventions** — sign bugs live here; read before touching `src/precond/`.

## Status / staged adaptations (human-approved 2026-07-16)

- **Â-block:** the low-level `CahouetChabardConfig` default is LOR-AMG (spec's
  sanctioned alternative); **the case-level default (decks, Python, `Case`) is
  `jacobi_pcg` since 2026-10-07** — CG with Jacobi to rtol 1e-2 (≤ 50
  iterations), measured with `bench/bench_velocity_pc` (CLAUDE.md, "Velocity
  block PC default"). `jacobi_chebyshev` is the third option. p-multigrid is
  deferred until the GPU port at PSC; `a_pc = pmg_chebyshev` is a reserved enum
  value that errors with a clear message in v1.
- CC is integrated as a **pluggable `schur_model` inside the existing
  `StokesSolver`** (one solve path), not a parallel driver module. The spec's
  normative solver hierarchy (§2.4/§6.5) is what `StokesSolver` executes when
  configured with `schur_model = ConsistentBMB`.
- Quad/hex only (repo law): no tets, no curved isoparametric meshes, no AMR.
  "Deformed elements" coverage comes from the tanh-stretched box factory
  (affine, non-uniform). `AbsLumped` mass inverse exists only as a rejected
  enum value — simplices are refused at mesh load, so the fallback it serves
  has no callers.
- T6c runs TGV-**Stokes** (convection off): NSE/IMEX advection remains gated
  behind Sprint 2 sign-off.
- Test tiers: representative T3 subset in `fast`; the full (ν, σ, h, p) sweep
  carries the `slow` ctest label (~20–35 min, ≤4 ranks, on-demand, never in
  the agentic loop); DNS-scale benchmarks stay in the human-launched
  expensive tier.

## Conventions (spec §2 — normative)

### Continuous problem

At each implicit stage/step, the generalized Stokes problem:

```
σ u − ∇·(ν 𝔻(u)) + ∇p = f     in Ω
∇·u = 0                        in Ω
```

with `σ = γ₀/Δt` (γ₀ = leading BDF weight: 1 for BE, 3/2 for BDF2, 11/6 for
BDF3; σ = 0 for steady Stokes), and `𝔻(u)` either `∇u` (Laplacian form,
default) or `2e(u)` (symmetric-gradient form).

### Discrete system — canonical convention for this module

All operators on TRUE dofs after essential-BC elimination:

```
A   := σ M_v + ν K_v               (velocity block; SPD)
B   :  V_h → Q_h,  (B u)_k = ∫ ψ_k (∇·u) dx     (VectorDivergenceIntegrator)
M_v := velocity vector mass        M_p := pressure mass
```

**Canonical saddle-point system (symmetric indefinite):**

```
[ A   Bᵀ ] [ u ]   [ f ]
[ B   0  ] [ p̃ ] = [ g ]
```

with **internal pressure `p̃ = −p_physical`**. This makes the system symmetric
with `S := B A⁻¹ Bᵀ` symmetric positive (semi-)definite. The sign flip to
physical pressure happens **exactly once, at output/postprocessing** (unit
test asserts this against an analytic solution). `g ≠ 0` arises from
essential-BC elimination.

Relation to the legacy bessemer system: `[A −Bᵀ; B 0]` on physical `p` and
`[A Bᵀ; B 0]` on `p̃ = −p` are the same equations; only the preconditioner's
triangular application and the output flip see the difference.

### The preconditioner

Schur complement approximate inverse (CC with consistent operator, λ = 0) —
**these are the actual coefficients, not placeholders:**

```
P_S⁻¹ := ν M_p⁻¹  +  (γ₀/Δt) (B M_v⁻¹ Bᵀ)⁻¹
```

- The pressure-mass term is scaled by the **viscosity ν**; the mixed Poisson
  term by **σ = γ₀/Δt** (drops entirely for steady Stokes, σ = 0).
- Reaction limit (σ large): `A → σM_v ⇒ S → σ⁻¹BM_v⁻¹Bᵀ` ⇒ Poisson term exact.
- Viscous limit (σ → 0): `S` spectrally equivalent to `ν⁻¹M_p` ⇒ mass term.
- `nu_pc` defaults to **ν for BOTH viscous forms** (C&G put plain μ on the
  mass term even with the 2μe(u):e(v) form — on (near-)solenoidal fields
  ∫2e(u):e(v) = ∫∇u:∇v). `2ν` is retained only as a T3 calibration
  alternative for sym-gradient; adopt only on measured evidence.
- **Scaling guard (do not "fix" this):** in the periodic constant-coefficient
  case the decomposition is exact — the symbol of `S⁻¹` is `σ/|k|² + ν`. The
  `1/Δt` lives ENTIRELY in σ on the Poisson term; the mass term's coefficient
  is the flat high-|k| plateau, `ν`, independent of Δt. Putting `1/Δt` on the
  mass term mis-scales the high-frequency band by `σ/ν ≈ γ₀/(νΔt)` (~10⁷ at
  DNS parameters) and destroys the preconditioner. `nu_pc` is an O(1) dial
  around ν, never a function of Δt.
- Inner applications are inexact (it's a preconditioner). Inexactness is
  fine; *structural* wrongness (L_p in place of BM⁻¹Bᵀ, ε-shifts, wrong
  coefficient scalings) is not.
- `LaplacianLegacy` (`ν_pc M_p⁻¹ + σ L_p⁻¹`) exists behind the same interface
  ONLY for regression test T5 and cheap smoke tests. Never the default.

### Convention mapping to Creff & Guermond (verified, do not re-derive)

| This module | C&G paper | Note |
|---|---|---|
| `b(v,q) = ∫q∇·v`, `B`: vel→pres | identical (their (2.2), (2.7)) | same sign/orientation |
| `[A Bᵀ; B 0]` on `p̃ = −p_phys` | `[A −Bᵀ; B 0]` on `p` (their (2.8)/(2.14)) | substitute `p̃ = −p`: identical |
| `A = σM_v + νK_v`, `σ = γ₀/Δt` | `A = τ⁻¹M_V + μE_V` | `σ ↔ 1/τ`, `ν ↔ μ` |
| `S = BA⁻¹Bᵀ` (SPD) | `S = BA⁻¹Bᵀ` (their (2.9)) | identical |
| `P_S⁻¹ = νM_p⁻¹ + σ(BM_v⁻¹Bᵀ)⁻¹` | `C₀ = τ⁻¹(BM_V⁻¹Bᵀ)⁻¹ + μM_Q⁻¹` (their (4.4a), λ=0) | identical |

### Solver structure — there is exactly one

FGMRES on the **monolithic** `[A Bᵀ; B 0]` — the only outer Krylov, always
flexible (the inner components are iteration-limited solves, i.e. a nonlinear
preconditioner). Block upper-triangular preconditioner by default:

```
FGMRES — monolithic [A Bᵀ; B 0]
 └─ P_block = [ Â   Bᵀ ]     (diag / lower-tri via config)
              [ 0   −Ŝ ]
     ├─ Â⁻¹  : fixed LOR-AMG V-cycle(s)          (fixed linear op; no Krylov)
     └─ Ŝ⁻¹ = P_S⁻¹:
         ├─ ν_pc M_p⁻¹ : diagonal multiply if M_p diagonal, else fixed-order
         │               Chebyshev                (fixed linear op; no Krylov)
         └─ σ (BM_v⁻¹Bᵀ)⁻¹ : CG (SPD ⇒ CG, never GMRES)
             ├─ operator: B ∘ M_v⁻¹ ∘ Bᵀ, M_v⁻¹ = diagonal multiply or
             │            fixed-order Chebyshev   (fixed linear op)
             └─ PC: symmetric AMG V-cycle on LOR L_p (Ortho-wrapped if singular)
```

**Krylov correctness constraints (enforced in config validation):**
- Inside the inner CG, `M_v⁻¹` must be a *fixed linear operator* — diagonal
  multiply or fixed-order Chebyshev, never a tolerance-based solve.
- The L_p AMG must use symmetric relaxation with matched pre/post sweeps
  (asserted at setup); a nonsymmetric V-cycle silently breaks CG.
- MINRES is not implemented and must not be added.

### Nullspace handling (replaces the ε-shift)

Detection: singular iff **no traction/outflow BC on any boundary attribute
that actually exists in the mesh** (this phrasing survives periodic meshes,
including the fully periodic empty-boundary-set case). Implemented by
`BoundaryConditions::PressureNullspaceExists()`.

Treatment (all three together): (1) project the RHS of every singular solve
onto `𝟙⊥`; (2) re-orthogonalize the inner-CG iterate every `k_reproj`
(default 5) iterations against fp drift; (3) keep the pure-Neumann L_p
singular — **no pinning, no shift, ever** (repo law).

Projection inner product: `𝟙`-removal for `BM⁻¹Bᵀ` is plain **Euclidean (l2)**
on true dofs. Mass-weighted mean is only for reporting the physical mean-zero
pressure at output.

Fully periodic + σ = 0 is ill-posed for this module (constant-velocity kernel
in A): construction throws with a clear message.

### Invariants (also recorded in CLAUDE.md)

1. No ε-shift / `εM_Q + L_Q` regularization anywhere.
2. `BM_v⁻¹Bᵀ` is a matrix-free composition; **no code path may assemble it**.
3. The outer solver is FGMRES on the monolithic system, always.
4. Diagonal mass inverses are direct fused multiplies — never wrapped in a
   solver/smoother object (debug asserts + T1f enforce).
5. Internal pressure is `p̃ = −p_physical`; one sign flip at output.
6. Pressure nullspace by projection/orthogonalization only — never pinned
   (pre-existing repo law; the spec's `allow_pin_dof` debug flag is
   deliberately NOT carried -- pinning was never implemented, so there is
   nothing for it to enable).
7. The spec's `outer_*`/`fgmres_restart` config fields are deliberately NOT
   carried: the outer FGMRES knobs are `StokesSolverOptions`' (deck
   `solver.rtol/atol/max_iter/kdim`) -- one source of truth.

## Usage (lands with CC.7)

`Parameters` / deck: `solver.schur: mass | cc | laplacian_legacy` plus the
`CahouetChabardConfig` fields (`n_inner`, `lp_vcycles`, `block_shape`, …).
The config struct lives in `src/precond/cahouet_chabard.hpp`.

## Tuning notes (T3 evidence, desktop, 2026-07-17)

- First knob: `n_inner` (inner CG iterations, default 10) vs outer FGMRES
  count — the inner CG dominates preconditioner cost (spec §6.5 table).
- `nu_pc`: O(1) calibration around ν only; see the scaling guard above.
- **Measured robustness** (enclosed box, Q3/Q2, defaults): outer FGMRES 19–34
  flat over σ ∈ [0, 1e5] AND ν ∈ [1e-6, 1]; the Sprint-1 mass block (same
  LOR-AMG velocity block) hits 104 at σ = 1e5 vs CC's 32. Full slow-tier grid
  (ν × σ × h × Q2..Q5, 2D+3D): worst 63. Caps in `test/baselines.yaml`
  (`cc_robustness`).
- **Where the mass block still wins:** moderate σ/ν (e.g. σ ≈ 75, ν = 1:
  mass 10 vs CC 27 outers, each CC outer ~n_inner× heavier). CC pays off at
  small Δt / small ν — DNS time steps — and for adaptive runs where its
  counts do not move. FLIPPED 2026-07-17: CC is now the case-level default (`Parameters.schur`); `solver.schur: mass` selects the old block.
- **LaplacianLegacy:** indistinguishable on enclosed domains at these sizes;
  measurably worse (and worsening under refinement) with an outflow boundary
  — the boundary treatment is exactly its structural error (the T5 guard).
- **pc_quadrature (measured Q3+Q5, Stokes):** `gll_collocated` gives outer
  counts identical to ±1 and wall time uniformly 20–31% lower than `inherit`
  through Q5. **Default stays `inherit` by decision (2026-07-17): re-measure
  under NSE before flipping** — production DNS lives at large σ/small ν where
  the Poisson term dominates the PC, exactly where the lumping misfit would
  show if ever. Instrument: `PcQuadratureCompare` in the slow tier.
- **Singular L_p coarse solve:** hypre's DEFAULT, deliberately — any
  CycleRelaxType override breaks setup on degenerate tiny 3D hierarchies, and
  the Ortho wrap + projections are the actual protection (lp_surrogate.hpp).
