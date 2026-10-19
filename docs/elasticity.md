# Linear elasticity

bessemer's first solid solver, and the first step toward fluid-structure
interaction (FSI) with an ALE fluid (the Quaini / Veneziani / Quarteroni line of
work). It is a standalone solver today: a deck with `equation:
linear_elasticity` runs one static solve, or with `solid.analysis: dynamic` a
march in time with generalized-alpha or the fluid's BDF2/BDF3 (human
2026-10-19: "I want A and C implemented").

## 1. Formulation

Small strains, isotropic and homogeneous, displacement-only, posed on the
reference configuration (for a linear model the Lagrangian and the current
configuration coincide). Find u in [H1 Q_k]^dim, u = g on Gamma_D, with

    (2 mu eps(u), eps(v)) + (lambda div u, div v) = (f, v) + (t, v)_{Gamma_N}

for every v vanishing on Gamma_D, eps(u) = (grad u + grad u^T) / 2 and
sigma = lambda tr(eps) I + 2 mu eps. Material from Young's modulus E and
Poisson's ratio nu: mu = E / (2 (1 + nu)), lambda = E nu / ((1 + nu)(1 - 2 nu)).
2D is plane strain by default; `solid.plane: stress` uses lambda* = 2 lambda mu /
(lambda + 2 mu) = E nu / (1 - nu^2).

Boundary groups (`boundary_conditions`, every real boundary in exactly one):
`fixed` (u = 0), `displacement` (`value`: a constant u), `traction` (`value`: a
constant sigma n, force per unit area) and `free` (zero traction). A body force
per unit volume is `solid.body_force`. At least one fixed or displacement group
is required (otherwise the rigid motions are free).

**Dynamics** (`solid.analysis: dynamic`): rho u_tt + K u = f(t) with the
consistent mass M (density `solid.density`), from rest at t = 0 (the deck's
loads act from t = 0; the library's `Initialize` takes any initial
displacement and velocity) to `time.t_final` in steps of `time.dt`. Every
scheme solves (K + c M) u^{n+1} = r once per step:

- `generalized_alpha` (default): Chung & Hulbert (J. Appl. Mech. 60, 1993) on
  (u, v, a), alpha_m = (2 rho_inf - 1)/(rho_inf + 1), alpha_f = rho_inf /
  (rho_inf + 1), gamma = 1/2 - alpha_m + alpha_f, beta = (1 - alpha_m +
  alpha_f)^2 / 4; c = (1 - alpha_m) / ((1 - alpha_f) beta dt^2). Second order,
  unconditionally stable; `solid.rho_inf` (default 0.8) sets the damping of
  the unresolved high frequencies, 1 being the trapezoidal rule (energy
  conserved exactly). The initial acceleration solves M a0 = f(0) - K u0 (zero
  on the Dirichlet boundary).
- `bdf`: the fluid's variable-step BDF (`time/multistep_coeffs`) on the
  first-order system u_t = v, rho v_t + K u = f: with v^{n+1} = c0 u^{n+1} +
  S_u, (K + c0^2 M) u^{n+1} = f^{n+1} - M (c0 S_u + S_v). The first step is a
  trapezoidal step, then BDF2, then BDF3 (`solid.bdf_order: 3`). **BDF3 is not
  A-stable**: an undamped mode with omega dt between about 0.5 and 2 grows (4 %
  per step at omega dt = 1), and a fine mesh always has such modes, so BDF3
  suits short or physically damped runs only. The fluid does not see this
  because its spectrum is dissipative. BDF2 is A-stable but damps resolved
  modes noticeably more than generalized-alpha (D3 below: 22 % of the energy
  in 60 steps against 10 % at rho_inf 0.5).

Dirichlet data and loads may depend on time in the library (their
coefficients' `SetTime`); deck values are constants.

**Locking.** A displacement-only discretization locks as nu -> 1/2. High orders
(the default is the fluid's Q6) delay it a long way; a mixed u-p formulation
(the Stokes saddle point with a pressure mass block, which bessemer's Taylor-Hood
machinery already solves) is the fix if nearly incompressible solids are
needed. Not built.

## 2. Code

- `src/solid/elasticity_parameters.hpp`: the `solid` deck section
  (`ElasticityParameters`, `PlaneModel`, `ElasticityPC`, `SolidAnalysis`,
  `SolidTimeScheme`, `GeneralizedAlpha()`).
- `src/solid/linear_elasticity.{hpp,cpp}`: `LinearElasticity`, the solver. Vector
  H1 on GLL nodes ordered by nodes; MFEM's `ElasticityIntegrator` in partial
  assembly at the RuleBook's Gauss-Legendre rule of order 2k + 1 (k + 1 points per
  direction); Dirichlet data projected on the boundary and eliminated; CG
  preconditioned by one BoomerAMG V-cycle in systems mode on the LOR (Q1 on the
  GLL nodes) rediscretization, the same machinery as the fluid's velocity block
  (`solid.preconditioner: jacobi` uses the operator's diagonal instead).
  Dynamics adds a PA `VectorMassIntegrator` form; K + c M is applied as the two
  forms on true dofs (no combined form, no assembled matrix), and a change of c
  (the BDF ramp, a shortened last step) rebuilds only the preconditioner.
  Post-processing: strain and kinetic energy (operator applies, on the device),
  external work, max |u|, the displacement at a point (search cached).
- `src/solid/elasticity_case.{hpp,cpp}`: `ElasticityCase`, the deck-driven case
  (boundary groups, body force, `Run()`: the static solve or the march,
  ParaView output of `displacement` (and `velocity`), the history CSV, the run
  summary). A separate class from the flow `Case`: the unknown is different,
  and FSI will couple one of each. ALE mesh motion is itself an elasticity-type
  (or harmonic) extension, which this class can provide.
- `apps/run_case` dispatches on `equation: linear_elasticity`.
- Example decks: `cases/cantilever_2d.yaml` and `cantilever_2d_dynamic.yaml`
  (the load applied suddenly; both run by ctest at np 1/2/4),
  `cases/cook_membrane.yaml` (+ `cook_membrane.mesh`; run it from the repository
  root).

The summary (`<path>/<name>_summary.json`) has `status`, `case` (material,
order, plane model), `mesh` (elements, dofs), `solver` (preconditioner,
iterations, final residual, converged, solve time), for a dynamic run
`dynamics` (scheme, dt, t_final, steps, CG iterations in all and the most in
one step) and `results` (`strain_energy`, `external_work`, `max_displacement`,
`kinetic_energy` and `total_energy` when dynamic, and the `probe` point's
displacement when `solid.probe` is set). `output.history: true` writes
`<path>/<name>_history.csv`: step, t, dt, CG iterations, kinetic, strain and
total energy, and the probe's displacement, one row per step.

## 3. Memory per DOF (estimated, not measured)

Per vector DOF, 3D, double precision:

| Piece | Q2 | Q6 | Note |
|---|---|---|---|
| Operator data (MFEM PA) | ~180 B | ~85 B | 20 numbers per quadrature point: Jacobian (9), lambda, mu, and a 9-number work array |
| Mass data (dynamics) | ~9 B | ~4 B | one number per quadrature point |
| CG vectors (r, z, p) + constrained-operator work (2) + the operator's three local vectors | ~70 B | ~70 B | |
| State and work (dynamics) | ~60-80 B | ~60-80 B | generalized-alpha u (2 levels), v, a, f^n + 4 work; BDF3 4 levels of u and v |
| LOR-AMG (systems mode) | ~2-3 kB | ~2-3 kB | the LOR matrix (81 nonzeros per row) and the hierarchy |

So the AMG dominates: a static or dynamic solve fits roughly 25-35M DOFs on an
80 GB H100.
Two later wins: a sum-factorized elasticity kernel of bessemer's own (MFEM's
apply evaluates the full element basis at every point, which is slow at Q6;
the in-repo grad-div kernel is half of it already), and point-block Jacobi /
Chebyshev preconditioning where dynamics makes the system mass-dominated (small
dt), as for the fluid; `solid.preconditioner: jacobi` is the first step there
and needs no assembled matrix. BoomerAMG with rigid-body modes would improve the static AMG but
needs the byVDIM ordering MFEM's elasticity PA does not support.

## 4. Verification

Fast tier (`linear_elasticity_test`, np 1/2/4; pins its own orders):

- E1: the PA operator equals MFEM's assembled `ElasticityIntegrator` matrix
  (2D distorted mesh at Q3, 3D at Q2) to 1e-12.
- E2: a full quadratic displacement is reproduced exactly at Q2 (2D, 3D), one
  face held, the exact traction on all others, the exact body force.
- E3: the Timoshenko-Goodier cantilever (Theory of Elasticity, 1970, section 21;
  plane stress, an exact cubic) is reproduced at Q3, tip deflection
  P L^3 / (3 E I) included.
- E4: the energy identity 2 W = (f, u) + (t, u) with zero Dirichlet data, and
  LOR-AMG and Jacobi reaching the same displacement (LOR-AMG in fewer
  iterations).
- E5: a deck through `ElasticityCase`, summary written.
- `run_case_cantilever_np*`: the cantilever deck end to end.

Fast tier, dynamics (`elastodynamics_test`, np 1/2/4):

- D1: the generalized-alpha parameters in closed form (rho_inf 1, 0.5, 0).
- D2: temporal orders on a separable motion u = s(x) T(t), s quadratic (in
  the space, so the semi-discrete solution is exact and every error is the
  integrator's), 2D and 3D: generalized-alpha at rho_inf 1, 0.5 and 0 order 2,
  BDF2 order 2, BDF3 order 3, in both u and v (measured 1.99-2.05 and
  2.94-3.03).
- D3: free vibration: generalized-alpha at rho_inf 1 conserves the energy
  1/2 v.M v + 1/2 u.K u to 2e-15; rho_inf 0.5 and BDF2 dissipate it.
- D4: the fixed-fixed bar's first longitudinal mode (nu = 0, so u =
  sin(pi x) cos(omega t) is exact with free sides; omega = pi sqrt(E / rho)):
  second order after 1.25 periods for both schemes, and generalized-alpha at
  rho_inf 1 matches the trapezoidal rule's analytic phase error 2 atan(omega
  dt / 2) per step to 2 %.
- D5: a dynamic deck through `ElasticityCase`: a load applied suddenly at
  t = 0 satisfies the trapezoidal rule's exact energy balance E(t) = f . u(t),
  the last step lands on t_final, the summary and the history file.

Verify tier (`verify_elasticity_test`, np 4; settings in `baselines.yaml`
`verification.elasticity`), each study at fixed orders and at the deck default:

- V1/V2: a manufactured solution u_i = prod sin(pi x_d) (div u != 0, so both
  Lame terms enter) in 2D and 3D, one face held, exact traction elsewhere:
  L2 rate k + 1, H1 rate k.
- V3: Lame's thick-walled cylinder under internal pressure (Timoshenko &
  Goodier, section 28; plane strain) on a curved isoparametric quarter annulus:
  the same rates.
- V4: Cook's membrane (Cook 1974; E = 1, nu = 1/3, total shear 1): the
  vertical displacement at the loaded edge's midpoint (48, 52) converges to
  the literature's 23.96 in plane stress (e.g. Pian & Sumihara 1984) and
  21.520 in plane strain (Auricchio et al., CMAME 194, 2005), within 0.2 %.
  The top corner (48, 60), sometimes named as the reference point, converges
  to about 25.2 instead (checked with an independent Q2 code).

- V5: the deck's default time scheme (generalized-alpha, rho_inf 0.8 today) on
  D2's separable motion at the default order: rate 2 in u and v (measured
  1.998, 1.999).
- V6: a clamped-free beam (L / D = 20, plane stress, Q6 40 x 2) started from
  its Euler-Bernoulli first mode: the first natural frequency from the tip's
  zero crossings over three periods is 0.151 % below Euler-Bernoulli's (1.8751
  / L)^2 sqrt(E I / (rho A)), where Timoshenko's first-order shear and rotary
  inertia correction predicts -0.15 %; the test allows 0.5 %.

The Turek-Hron CSM1-3 structural benchmarks use a St. Venant-Kirchhoff
material: they belong to the large-deformation step, not to this linear model.

## 5. Next

1. Large deformation (St. Venant-Kirchhoff, total Lagrangian, Newton), for
   Turek-Hron CSM and FSI.
2. The ALE fluid and the coupling.
3. Measured memory and throughput on a GPU (nothing here has run on one).
