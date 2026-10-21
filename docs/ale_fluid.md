# The ALE fluid

A flow case whose mesh moves at every step: the arbitrary Lagrangian-Eulerian
(ALE) form of the incompressible Navier-Stokes and unsteady Stokes equations,
the third step toward fluid-structure interaction after the linear elasticity
solver ([elasticity.md](elasticity.md)) and the mesh motion
([mesh_motion.md](mesh_motion.md)). Human 2026-10-21: "can you go scope out what
needs to be done to implement ALE now?", then "Alright can you go get started
on this on CPUs". The scope and the decisions behind it (IMEX first; the mesh
moved every step by pseudo-elasticity with Jacobian stiffening; the cached
geometry kept for fixed meshes, with a refresh path for moving ones; the
partitioned FSI coupling to the Turek-Hron problems) are in the scoping
document of 2026-10-21 (stage numbers below are its stages).

This first piece (stages 0-1) is the library: `Case::SetMeshMotion` with
IMEX. No deck keys yet (stage 2), a full rebuild of the solvers after each move
(stage 3 replaces it with an in-place refresh), CPU only so far.

## 1. Formulation

The non-conservative ALE form on the current mesh Omega(t):

    du/dt|_X + ((u - w) . grad) u - nu Lap u + grad p = f,   div u = 0,

with du/dt|_X the time derivative following a mesh point and w the mesh
velocity. Every operator is assembled on the mesh at the new time t^{n+1}. The
nodal BDF history needs no transfer: a dof belongs to a mesh node, and the
history values ride with it, so the BDF combination of the nodal values is the
derivative along the node path (Formaggia and Nobile 2004, the BDF-ALE
schemes).

- **The mesh velocity** w^{n+1} is the BDF derivative of the node positions at
  the order of the step (BDF1 on the starter, then BDF2 or BDF3;
  `MeshMotion::MeshVelocity`), interpolated from the mesh's nodal space onto
  the velocity space (`MeshVelocityField`; exact when the velocity order is at
  least the geometry order).
- **The mesh convection** is explicit: the convective velocity u - w adds
  -C(w) u to the left side, and bessemer puts C(w^{n+1}) u* on the right, u*
  the EXT extrapolation of the history at the convection's order (EXT1 on the
  starter). C is linear in the wind, so no history of w is kept.
  `MeshConvection`: MFEM's `ConvectionIntegrator` per component, partial
  assembly, at the dealiased 3k rule of the fluid's convection. It is added
  for Stokes too: the ALE time derivative needs it whatever the equation.
- **Moving walls** take u = w_h, the discrete mesh velocity, as their
  Dirichlet data (`Case::MeshVelocityCoefficient()`). With it a rigidly moved
  box carries the fluid exactly (test A3).
- **The geometric conservation law.** With the operators on the new mesh and
  w the BDF derivative of the nodes, a uniform flow stays uniform on any mesh
  motion to the solver tolerance (test A2, 2D and 3D): the discrete GCL holds
  for this non-conservative form. A conservative form (mass matrices of
  several meshes in the BDF sum) would need the GCL in its own discrete
  version; it is not built.
- **The CFL number** is taken from u - w.

**Refused with a moving mesh (setup aborts):** OIFS (`time.convection: auto`
picks IMEX when the mesh moves; an explicit `oifs` is refused), the rotational
form, error-controlled steps, AMR. OIFS on a moving mesh needs the geometry at
every RK stage, the riskiest later stage of the scope.

## 2. Code

- `src/ale/mesh_convection.{hpp,cpp}`: `MeshVelocityField` (w on the velocity
  space and its coefficient) and `MeshConvection` (y = C(w) u).
- `StokesTimeIntegrator::SetMeshMotion(MeshMotion&, MeshVelocityField&)`,
  before the first step. Each step moves the mesh first
  (`MoveMesh`: `MeshMotion::Move(t^{n+1})`, w updated, then every solver and
  the IMEX convection rebuilt on the new geometry), then assembles and solves
  as on a fixed mesh, with `AddMeshConvection` in the right-hand side (the
  trapezoidal starter and `AssembleBdfRhs`).
- **A failed solve on a moving mesh** (`time.on_solver_failure: cut_step`)
  moves the mesh back first (`MeshMotion::Revert`, which also restores a
  history level the move had dropped), then retries at the cut step.
- `Case::SetMeshMotion(MeshMotion&)` before the first step; `MeshVelocity()`,
  `MeshVelocityCoefficient()`. The `MeshMotion` is the caller's (built on the
  case's mesh before the `Case`, as in `test/ale_fluid_test.cpp`); any
  `MeshMotionModel` drives it (the linear elastic one with stiffening by
  default; the test uses a prescribed displacement field).

**Why a full rebuild after every move, for now.** MFEM deletes a mesh's cached
geometric factors when its nodes change (`NodesUpdated`), and bessemer's own
divergence (`operators/lean_vector_divergence.cpp`), grad-div
(`operators/grad_div_integrator.cpp`) and OIFS kernels keep raw pointers to
them for speed. A solver kept across a move would read freed memory. The
rebuild is correct and slow (the pressure LOR-AMG setup every step); stage 3
adds the in-place refresh for moving meshes and keeps the cached pointers for
fixed ones (human 2026-10-21: "we'll want both options").

## 3. Verification (`test/ale_fluid_test.cpp`, fast tier, np 1/2/4)

| Test | What it pins | Result |
|---|---|---|
| A1 | zero motion through the moving-mesh path = the fixed-mesh run (Navier-Stokes Taylor-Green) | to 1e-9 |
| A2 | free stream on a mesh whose interior oscillates, 2D and 3D (the GCL) | u uniform to 1e-9, p constant to 1e-8 |
| A3 | a box moved rigidly with u = w on its walls, constant and oscillating motion, Stokes and NSE | u = w_h to 1e-9 |
| A4 | a linear Stokes flow on a mesh moved at constant rates: translated exact; stretched, the starter's O(dt^2) error | translation exact; rate 2.02 |
| A5 | a vetoed solve on a moving mesh: the mesh goes back, the step is cut and retried, the nodes end where the motion says | to 1e-12 |
| A6 | Taylor-Green on an oscillating mesh, Q3, h = 1/2, 1/4, 1/8: velocity L2 | rates 4.03, 4.32 (theory 4) |
| A7 | the same at Q5, dt = 0.04, 0.02, 0.01 | rates 2.83, 2.24 (theory 2, BDF2/EXT2) |

A4's stretched case is not exact because the starter's explicit C(w) u^0
reads u^0 on the moved nodes, off by dt (grad u) w where grad w is not zero; the
BDF2 steps after it are exact for this flow.

Run in the cloud session on a scratch build (MFEM v4.10 from GitHub, apt hypre
and Open MPI, not the Spack toolchain); CI is the reference build. Never run on
a GPU: `MeshVelocityField::Update` projects on the host once per step.

## 4. Next

1. Deck surface (stage 2): a `mesh_motion:` section in flow decks with
   per-group boundary motion, nondimensionalization, the node history in
   checkpoints.
2. In-place refresh after a move (stage 3): the geometric data re-read, the
   pressure LOR-AMG allowed to go stale for N steps, the velocity PC
   refreshed; the fixed-mesh path keeps its cached pointers.
3. Modal (rigid) motion without a per-step elastic solve (stage 4).
4. Validation (stage 5): the in-line oscillating cylinder (Dütsch et al. 1998)
   and transverse lock-in.
5. Then the FSI stages: St. Venant-Kirchhoff solid, the Turek-Hron geometry,
   the partitioned coupling with dynamic Aitken and the Robin conditions, to
   all nine Turek-Hron problems.
