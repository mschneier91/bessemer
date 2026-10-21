# Spec: hyperelasticity, St. Venant-Kirchhoff first

**Status:** approved 2026-10-21. Human: "start specking on adding hyperelasticity and a St.
Venant-Kirchhoff model", then, on this draft, "go with your recommendations, commit the spec
and start M1". The decisions of section 0 are the recommendations as written. M1
(section 7) is in progress on branch `hyperelastic`.

**Where it fits.** `docs/elasticity.md` section 5 names this as the next solid step: "Large
deformation (St. Venant-Kirchhoff, total Lagrangian, Newton), for Turek-Hron CSM and FSI".
`docs/mesh_motion.md` plans a hyperelastic mesh-motion model through the same
`MeshMotionModel` interface. The linear model cannot do either job:

- a rigid rotation is not strain-free in it;
- the Turek-Hron structural and FSI benchmarks prescribe St. Venant-Kirchhoff.

## 0. Decisions (accepted 2026-10-21)

| # | Question | Decision |
|---|---|---|
| D1 | Scope of the first delivery | Static and dynamic SVK as a standalone solid (Turek-Hron CSM1-3), then the mesh-motion model. Follower loads, neo-Hookean and near-incompressibility later. |
| D2 | How the residual and tangent are computed | bessemer's own sum-factorized partial-assembly kernels (device), with MFEM's host `HyperelasticNLFIntegrator` as the test reference only (section 3). |
| D3 | Deck interface | A new `equation: hyperelasticity` with `solid.model: st_venant_kirchhoff`; `equation: linear_elasticity` decks unchanged. |
| D4 | Neo-Hookean in the same pass? | No: SVK first, neo-Hookean as a second material through the same interface once SVK is verified (an afternoon, not a project). |
| D5 | Newton's linear preconditioner | The reference linear-elasticity LOR-AMG (built once, the existing machinery) first; a refreshed LOR tangent as an option, measured before it becomes a default. |
| D6 | Who builds it, where | A branch from main. `src/solid/` and `src/ale/` are the other thread's code: new files only, no edits to `LinearElasticity` without a bitwise check. |

## 1. Formulation

**Total Lagrangian** on the reference configuration Omega_0 (the mesh as read), as the
linear solver and the mesh motion already are. The unknown is the displacement u(X, t).

- F = I + grad_X u, the deformation gradient; J = det F.
- E = (F^T F - I) / 2, the Green-Lagrange strain.
- **SVK:** W(E) = lambda/2 (tr E)^2 + mu E : E. So S = lambda tr(E) I + 2 mu E (second
  Piola-Kirchhoff) and P = F S (first Piola-Kirchhoff).
- Lame parameters from E and Poisson's ratio exactly as `ElasticityParameters` gives them.
  2D is plane strain. Plane stress is not supported for SVK: it needs a pointwise solve
  for the out-of-plane stretch. A deck asking for it is an error.

**Static problem.** Find u = g on Gamma_D such that, for every v vanishing on Gamma_D,

    R(u; v) = int_Omega0 P(F(u)) : grad_X v dX - int_Omega0 f0 . v dX - int_GammaN T0 . v dA = 0,

with f0 the body force per reference volume and T0 the traction per reference area. These
are **dead loads**: fixed in the reference configuration, which is what Turek-Hron CSM
uses (gravity). Follower loads (pressure on the deformed surface, the fluid traction in
FSI) are the coupling step's job.

**Linearization (Newton).** The tangent in direction w:

    K_T(u)[w; v] = int ( grad w S + F dS ) : grad v,
    dS = lambda tr(dE) I + 2 mu dE,
    dE = sym(F^T grad w).

- It is symmetric: the hyperelastic tangent always is. It is positive definite on stable
  equilibrium branches and can be indefinite near buckling.
- At u = 0 it is exactly the linear elasticity operator. So the existing LOR-AMG on the
  reference is the tangent's preconditioner in the small-strain limit, and it degrades as
  strains and rotations grow.

**Dynamics.** rho_0 u_tt + R_int(u) = f(t), with the consistent mass at rho_0 as today.

- **Generalized-alpha** (Chung-Hulbert) applies unchanged to the nonlinear system, with
  the internal force evaluated at u_{n+1-alpha_f}. Each step is a Newton solve for
  u_{n+1} with the effective tangent c M + (1 - alpha_f) K_T.
- **BDF2** on (u, v) as today, with tangent c0^2 M + K_T.
- **BDF3 stays off** for undamped solids (elasticity.md section 1).
- **Energy:** for SVK, generalized-alpha at rho_inf = 1 does not conserve energy exactly
  as it does for the linear model; an energy-momentum scheme (Simo-Tarnow, built for SVK)
  would. This is noted, not planned.

**SVK's known weakness.** It is not polyconvex, and W stays finite as an element is
crushed (J -> 0). Strong compression can therefore invert elements without the energy
resisting. That is acceptable for Turek-Hron (bending, moderate strain, large rotation)
and for mesh motion with stiffening. It is the reason to add neo-Hookean next (D4).

## 2. What exists to build on

- **`LinearElasticity`:** vector H1 Q_k on GLL nodes ordered by nodes, the Gauss rule of
  order 2k + 1 (k + 1 points per direction), Dirichlet handling, generalized-alpha and
  BDF, energies, the probe, LOR-AMG in systems mode. The hyperelastic solver mirrors its
  API (AddFixed, AddDisplacement, AddTraction, SetBodyForce, Solve, Solve(t), Initialize,
  Step, energies, DisplacementAt), so `ElasticityCase` and the mesh motion can drive either.
- **`MeshMotionModel`:** the interface a hyperelastic mesh-motion model implements, with
  Newton inside Solve(t).
- **MFEM:**
  - `HyperelasticNLFIntegrator` with `HyperelasticModel` (EvalW / EvalP / AssembleH). It
    ships neo-Hookean and inverse-harmonic models, but no SVK, and it is host-only
    element assembly, with the deformed position x = X + u as its unknown.
  - `NewtonSolver`.
  - MFEM 4.10's dFEM (`fem/dfem/`, forward-mode differentiable operators on the device),
    untested here.
- **The GPU lesson from the linear model:** MFEM's `ElasticityIntegrator` PA launches
  dim (p+1)^dim threads per block, so 3D Q6 cannot run on a GPU (status.md 2026-10-20).
  Its apply is also dense O(p^6) per element (vecdivdiv_spec.md section 1). Anything
  hyperelastic must avoid that path.

## 3. Discretization and kernels (D2)

**The operators.** The residual R(u) and the tangent action K_T(u) w, matrix-free in
partial assembly, at the linear solver's rule. That rule is 7 points per direction at Q6,
inside MFEM's interpolation limits, unlike the Oseen work's 10.

**One kernel family, templated on the material.** Each apply:

1. loads the element's displacement (or direction);
2. forms the reference gradient by sum factorization, O(p^4);
3. maps it to grad_X with the stored J_0^-1;
4. applies the pointwise law;
5. contracts back with the transposed gradient.

Thread blocks are Q1D x Q1D with a loop over z, as the in-repo `GradDivIntegrator` kernels
do, so no block exceeds CUDA's limit at Q6.

Pointwise data per quadrature point:

- the reference geometry: J_0^-1 and w det J_0, fixed for the run;
- for the tangent, F and S at the Newton linearization point (15 numbers in 3D), formed
  once per Newton iteration by the residual's pass.

The material is a struct of `MFEM_HOST_DEVICE` functions, so neo-Hookean (D4) is a new
struct, not new kernels:

    P = Stress(F, lambda, mu)                 // first Piola-Kirchhoff
    dP = Tangent(F, S, dF, lambda, mu)        // grad w S + F dS
    W = Energy(F, lambda, mu)

lambda and mu are per element, so the mesh motion's Jacobian stiffening carries over.

**The test reference.** A small SVK `HyperelasticModel` (EvalW, EvalP, AssembleH) plugged
into MFEM's `HyperelasticNLFIntegrator`, host and fully assembled, in the tests only. The
kernels must match it to roundoff.

**Alternatives considered.**

- **MFEM's integrator for real runs:** quick, but host-only with dense element matrices.
  At Q6 in 3D that is CPU-only and slow, the same trap as the elasticity PA.
- **dFEM:** writing only the pointwise law and getting the tangent by automatic
  differentiation is attractive for several materials. But its performance at Q6, its
  device maturity and its fit with bessemer's LOR preconditioners are unknown. The MFEM
  upgrade plan already has a "single-kernel dFEM probe" pending. If that probe goes well,
  dFEM becomes the second implementation of the same material interface, checked against
  the own kernels.

## 4. Solvers (D5)

**Newton.**

- Relative and absolute residual tolerances, an iteration cap, and a backtracking line
  search on the residual norm (off by default, on for large loads).
- For static solves, load stepping: `solid.load_steps` ramps the body force, tractions and
  prescribed displacements together.
- In dynamics, each step starts from the extrapolated displacement; in mesh motion, from
  the previous move's displacement.
- Reported per solve: Newton iterations, linear iterations, the residual history (so the
  quadratic convergence a consistent tangent should give is visible), and the load steps.

**Linear solves.**

- CG on the tangent (symmetric; positive definite on stable branches); GMRES as an option
  for indefinite cases.
- Default preconditioner: one BoomerAMG V-cycle (systems mode) on the LOR linear-elasticity
  operator at the reference, plus c M in dynamics, built once and reused across Newton
  iterations. This is the existing machinery, factored out rather than duplicated
  (section 5).
- Option: a refreshed LOR tangent every N Newton iterations, assembled on the host on the
  LOR mesh (the HO and LOR dofs coincide, so the displacement transfers directly) through
  the SVK `HyperelasticModel` of section 3.
- Jacobi as today.
- Which default wins is measured on CSM1-3 and the mesh-motion test, not assumed.

## 5. Code and deck (D3, D6)

**New files:**

- `src/solid/hyperelastic_material.hpp`: SVK (later neo-Hookean), device functions.
- `src/solid/hyperelastic_operator.{hpp,cpp}`: residual and tangent operators (a CUDA
  translation unit).
- `src/solid/hyperelasticity.{hpp,cpp}`: `Hyperelasticity`, the solver (Newton, load
  steps, dynamics).
- `src/ale/hyperelastic_motion.{hpp,cpp}`: `HyperelasticMotion`, the mesh-motion model,
  with a `Factory` like `LinearElasticMotion`'s.
- `test/hyperelasticity_test.cpp`, `test/verify_hyperelasticity_test.cpp`.
- `cases/turek_hron_csm{1,2,3}.yaml` and the beam mesh.

**Existing files:**

- `run_case` and `Parameters` dispatch `equation: hyperelasticity`; the `solid` section
  gains `model` and the Newton keys.
- `ElasticityCase` builds either solver. Its summary gains a `newton` block: iterations
  per solve or step, load steps, and the most Newton iterations in one solve.
- The LOR-AMG builder is shared with `LinearElasticity` by moving it to a helper. Any
  change inside `LinearElasticity` is checked bitwise on its tests and decks at np 1, 2
  and 4 before it merges, so existing outputs do not move.

**Deck keys (proposed):**

```yaml
equation: hyperelasticity
solid:
  model: st_venant_kirchhoff   # neo_hookean later
  youngs_modulus: 1.4e6        # or shear_modulus + poisson_ratio (Turek-Hron give mu_s, nu_s)
  poisson_ratio: 0.4
  density: 1000.0
  body_force: [0.0, -2000.0]   # rho_s g per reference volume
  newton_rtol: 1.0e-10
  newton_max_iter: 25
  load_steps: 1
  line_search: false
```

## 6. Verification

**Fast tier** (`hyperelasticity_test`, np 1, 2 and 4 on the CPU; np 1 and 2 on GPUs):

- **H1 (material):**
  - P(F) against the closed form at random F;
  - the tangent against finite differences of P (error O(eps^2));
  - frame indifference, P(QF) = Q P(F);
  - zero stress for any rotation;
  - at F = I + eps H, P equals the linear sigma(eps H) to O(eps^2).
- **H2 (operators):** the PA residual and tangent action equal MFEM's
  `HyperelasticNLFIntegrator` with the SVK model (assembled), at a finite random
  displacement, on distorted 2D Q3 and 3D Q2 meshes and a small Q6 mesh, to 1e-12.
- **H3 (tangent consistency):** a Taylor test of the PA residual against the PA tangent.
- **H4 (homogeneous finite deformation):** an affine displacement with constant F
  (stretch plus shear, J != 1), the exact traction P(F) N on the free faces, prescribed
  displacement on the others. It is reproduced exactly, and Newton converges quadratically.
- **H5 (large rigid rotation):** 90 and 180 degrees prescribed on the whole boundary. The
  interior rotates exactly with zero strain energy. The linear model's spurious strain on
  the same test documents why this solver exists.
- **H6 (small loads):** the cantilever converges to `LinearElasticity`'s displacement as
  the load goes to zero (difference O(load^2)).
- **H7 (dynamics):** generalized-alpha and BDF2 at order 2 on a separable nonlinear motion,
  the body force made by a generator script (sympy) at build time.
- **H8 (end to end):** static and dynamic decks through `ElasticityCase`, summary written.

**Verify tier** (`verify_hyperelasticity_test`, np 4, thresholds in `baselines.yaml` under
a new key, never re-blessing existing ones):

- **V1:** a nonlinear manufactured solution in 2D and 3D; L2 rate k + 1, H1 rate k.
- **V2:** Turek-Hron CSM1 and CSM2 (static) and CSM3 (dynamic):

  | Test | Material | Point A reference |
  |---|---|---|
  | CSM1 (static, g = 2) | rho_s 1e3, nu_s 0.4, mu_s 0.5e6 | u_x -7.1877e-3, u_y -66.1023e-3 |
  | CSM2 (static, g = 2) | rho_s 1e3, nu_s 0.4, mu_s 2.0e6 | u_x -0.469000e-3, u_y -16.9739e-3 |
  | CSM3 (dynamic, g = 2) | as CSM1 | u_x -14.305 +- 14.305 [1.0995] e-3, u_y -63.607 +- 65.160 [1.0995] e-3 |

  The geometry is the flag of Turek and Hron (2006): a beam 0.35 x 0.02, clamped to the
  rigid cylinder (centre (0.2, 0.2), diameter 0.1). Point A = (0.6, 0.2) is the midpoint
  of its free end. Gravity acts on the beam only.

  The values above come from the FeatFlow benchmark tables as quoted by search extracts
  and a downstream paper; the official page was unreachable at drafting. The bracket is
  read as the frequency in Hz. **Confirm every number against Turek and Hron, "Proposal
  for numerical benchmarking of fluid-structure interaction between an elastic object and
  laminar incompressible flow", in Fluid-Structure Interaction, Springer 2006
  (doi:10.1007/3-540-34596-5_15), before any of it becomes a test threshold.**

- **V3 (mesh motion):** the DFG cylinder rotated by a large angle about its centre.
  `JacobianRange` stays positive where the linear model's goes negative, every node
  returns on moving back to rest, and Newton converges warm-started from the last move.
- **GPU:** the fast tier at np 1 and 2, and a 3D Q6 run, which MFEM's elasticity PA
  cannot do.

## 7. Milestones

1. **Material and operators:** H1-H3 on CPU and GPU, with the MFEM reference model in the
   tests.
2. **Static solver:** Newton, load steps, the deck, H4-H6 and H8 static; CSM1/CSM2 decks
   and V1, V2 static.
3. **Dynamics:** H7, H8 dynamic, CSM3 (V2 dynamic).
4. **Mesh motion:** `HyperelasticMotion` and V3.
5. **Later, each needing its own OK:**
   - neo-Hookean (D4);
   - follower pressure loads for FSI;
   - the refreshed LOR tangent as a default;
   - a mixed u-p form for nearly incompressible solids;
   - an energy-momentum integrator;
   - replacing MFEM's `ElasticityIntegrator` PA in `LinearElasticity` with the new kernel
     at F = I. This would fix the 3D Q6 GPU limit for the linear solver too, but it
     changes linear results at roundoff.

Each milestone ends with the fast tier at CPU np 1, 2 and 4 and GPU np 1 and 2, a write-up
in `docs/elasticity.md` (or a new `docs/hyperelasticity.md`), and a merge to main after
review.

## 8. Risks and open questions

- **Newton robustness at large loads.** Load stepping and the line search are the answer
  for CSM1/CSM2. CSM3's dynamics should not need them.
- **The fixed reference preconditioner at large rotations.** The Turek-Hron flag's tip
  moves about 0.066 against a length of 0.35, a rotation of roughly 10-20 degrees (an
  estimate). Mesh motion may rotate much more; V3 measures how far the fixed LOR-AMG
  holds.
- **Memory** (an estimate, not measured): the tangent's 15 numbers per point and the
  geometry's 10 add roughly 1.5-2x the linear PA data. The LOR-AMG still dominates
  (elasticity.md section 3), so problem sizes stay close to the linear solver's.
- **SVK in compression** (section 1): documented, with neo-Hookean as the remedy.
- **Plane stress:** not supported for SVK (section 1).
