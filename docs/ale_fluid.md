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

Built so far: the library (stages 0-1, `Case::SetMeshMotion` with IMEX) and
the deck surface (stage 2, section 3 below: boundary `motion`, checkpoints of
the node history), the solvers refreshed in place after each move (stage 3,
section 2), earlier mesh solves reused (stage 4); CPU only so far.

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
- **The CFL number** is taken from u - w: the CFL step control's, the
  progress line's, the fixed-step `cfl_max` check's, the first step's cap and
  the error mode's ceiling (until 2026-10-23 all but the first took u).

**Refused with a moving mesh (setup aborts):** OIFS (`time.convection: auto`
picks IMEX when the mesh moves; an explicit `oifs` is refused), the rotational
form, error-controlled steps, AMR. OIFS on a moving mesh needs the geometry at
every RK stage, the riskiest later stage of the scope.

## 2. Code

- `src/ale/mesh_convection.{hpp,cpp}`: `MeshVelocityField` (w on the velocity
  space and its coefficient) and `MeshConvection` (y = C(w) u).
- `StokesTimeIntegrator::SetMeshMotion(MeshMotion&, MeshVelocityField&)`,
  before the first step. Each step moves the mesh first
  (`MoveMesh`: `MeshMotion::Move(t^{n+1})`, w updated, then every solver
  refreshed on the new geometry and the IMEX convection rebuilt), then
  assembles and solves
  as on a fixed mesh, with `AddMeshConvection` in the right-hand side (the
  trapezoidal starter and `AssembleBdfRhs`).
- **A failed solve on a moving mesh** (`time.on_solver_failure: cut_step`)
  moves the mesh back first (`MeshMotion::Revert`, which also restores a
  history level the move had dropped), then retries at the cut step.
- `Case::SetMeshMotion(MeshMotion&)` before the first step (a deck's `motion`
  groups build their own, `Case::BuildDeckMotion`); `MeshVelocity()`,
  `MeshVelocityCoefficient()`. The `MeshMotion` is the caller's (built on the
  case's mesh before the `Case`, as in `test/ale_fluid_test.cpp`); any
  `MeshMotionModel` drives it (the linear elastic one with stiffening by
  default; the test uses a prescribed displacement field).

- **The refresh after a move** (stage 3; deck `mesh_motion.refresh:
  in_place | rebuild`, default `in_place`; human 2026-10-22: "go to stage
  3"). `in_place` keeps every solver and calls
  `StokesSolver::UpdateGeometry(rebuild)`: the operator re-assembles what
  depends on the geometry (`StokesOperator::UpdateGeometry`: the mass and its
  diagonal, the divergence, gamma = c h_K with its per-element values, the
  momentum block at the next `SetMassCoeff`), the Cahouet-Chabard
  preconditioner its masses and their inverses
  (`CahouetChabardSchurPC::UpdateGeometry`, `MassInverse::Refresh`), the
  pressure mean its weights, the guess state its operator images. Re-assembly
  re-reads MFEM's new geometric factors, so the kernels' cached pointers are
  valid again. Only the AMG hierarchies (the pressure LOR-AMG and a LOR-AMG
  velocity block) and the Chebyshev eigenvalue estimates are left an earlier
  mesh's, rebuilt every `mesh_motion.amg_interval` moves (default 10, by the
  move count, so a restart keeps the schedule): they only precondition. The
  velocity PCG's Jacobi or point-block blocks are rebuilt at every move.
  `rebuild` is the old path, a new solver after every move.
  Dütsch deck to t = 1 (np 4, 124 steps; outer and inner iterations
  unchanged): the refresh 1.42 s with `rebuild`, 1.16 s in place with the
  AMG rebuilt every move, 0.39 / 0.35 / 0.26 s at intervals 5 / 10 / 25; the
  run 11.2 s against 9.1-9.6 s (-15 to -19 %, run-to-run noise ~0.5 s). With
  the AMG rebuilt at every move the in-place path equals `rebuild` to 2e-16
  (test A10). The fixed-mesh path is untouched (no move, no refresh).
- **What survives a rebuild** (`refresh: rebuild`): the outer solver's initial-guess state
  (the projection's kept solutions, or the extrapolation's previous one) is
  handed from the last step's solver to the new one, its operator images
  recomputed on the moved mesh (`StokesTimeIntegrator::CopyGuessState`).
  Without it every step started cold: on the oscillating-cylinder deck to
  t = 0.3 (np 4) the outer iterations were 1,336 in 116 steps, 545 with it,
  and the run 3.3 s against 2.0 s; with zero motion through the same path the
  run takes 0.82 s, as the fixed mesh does.
- **The mesh solve starts from the kept node displacements extrapolated to
  the new time** (quadratic, `MeshMotion::Options::guess_order`), to the
  tolerance a zero guess would have had (`LinearElasticity::Solve(t, guess)`);
  deterministic across a restart, since the checkpoint carries those states.
  3D box, 8^3 Q3, np 4: the mesh solves' time 5.3 s to 3.1 s over 10 steps
  (about 20 CG iterations per move). The summary's `mesh_motion` section
  reports the iterations (`solve_iterations`).
- **Earlier mesh solves are reused** (stage 4, the modal shortcut, without a
  key of its own): the linear elastic displacement is linear in the boundary
  data, so `LinearElasticMotion` keeps up to `mesh_motion.basis_size` (12)
  orthonormalized data vectors and their solutions, and a move whose data lie
  in their span (to 1e-12 relative) is the same combination of the solutions,
  with no CG. A rigid or modal motion (a few shapes times functions of t) then
  costs as many solves as it has shapes: the oscillating-cylinder deck to
  t = 0.3 makes one solve in 116 moves (24 CG iterations in all), and its
  mesh motion falls from 1.27 s to 0.07 s; the run takes 1.46 s against 3.26
  s before stages 2-4's solver work. General motions fall back to CG and keep
  filling the basis up to its size. Checkpoints carry the basis
  (`MeshMotionModel::SaveState`); test `mesh_motion_test` M6 (two shapes:
  two solves, the reuse equal to solving every move to 1e-11, a restart exact).
- **A closed domain must keep its volume.** With no outflow and moving
  `no_slip` walls (u = w), a motion that changes the volume leaves the
  incompressible fluid no solution; the deck path checks the volume after
  every move and aborts at a change above 1e-6 relative
  (`StokesTimeIntegrator::RequireConstantVolume`; test
  `run_case_rejects_volume_change`). Before this check such a deck iterated
  to the solver's cap.

**Why the refresh is needed.** MFEM deletes a mesh's cached
geometric factors when its nodes change (`NodesUpdated`), and bessemer's own
divergence (`operators/lean_vector_divergence.cpp`), grad-div
(`operators/grad_div_integrator.cpp`) and OIFS kernels keep raw pointers to
them for speed. A solver kept across a move without re-assembly would read
freed memory. Stages 0-2 rebuilt every solver after every move: 9 % of a step
on the 2D cylinder deck and 6 % on an 8^3 Q3 box (np 4, profile regions
`time_integrator::build_solver`), the pressure LOR-AMG setup included. The
in-place refresh above re-assembles instead and keeps the AMG for
`amg_interval` moves; the fixed-mesh path keeps its cached pointers (human
2026-10-21: "we'll want both options"). To be re-measured at production sizes
and on a GPU, where the AMG setup's share may differ.

## 3. Decks

A flow deck (`equation: navier_stokes` or `stokes`) moves its mesh when a
boundary group carries `motion`: the displacement of that group's nodes from
their reference positions, one expression per component in the reference
x, y, z and t, with named `parameters` (the expression syntax of
docs/mesh_motion.md section 3):

```yaml
boundary_conditions:
  - {select: [walls], type: no_slip}            # fixed
  - select: [cylinder]
    type: no_slip                               # u = w on the moving wall
    motion: ["0", "A * sin(2 * pi * f * t)"]
    parameters: {A: 0.02, f: 3.0}
  - {select: [inflow], type: velocity, ...}     # fixed
  - {select: [outflow], type: outflow}          # fixed
```

- **Every group without `motion` is held fixed**, whatever its flow type; no
  sliding boundaries yet.
- **A moving `no_slip` group takes the mesh velocity** (u = w_h); a moving
  `velocity` group keeps its own data.
- **The interior** follows the linear elastic model with Jacobian stiffening,
  set by section `mesh_motion` (`poisson_ratio`, `stiffening`, `order`, the CG
  settings; its `steps` is not used: the mesh moves at every time step). The
  node order is the mesh's (the velocity order on the built-in geometries).
- **Refused:** dimensional decks (`nondim: dimensional`), AMR, `time.convection:
  oifs`, `step_control: error`, the rotational form, periodic meshes. Setup
  aborts with the reason.
- **Checkpoints** carry the node history (`<checkpoint>/mesh_motion.r<rank>`:
  the move count, the last BDF levels of the node displacements and their
  times), so a restart at the same rank count continues the march exactly
  (test A9).
- **Summary:** a `mesh_motion` section (moves, the det F range over the mesh,
  the largest node displacement).

Example: `cases/ale_oscillating_cylinder.yaml`, the DFG 2D-3 channel at Re 100
with the cylinder moved across it at 0.2 D near the shedding frequency (Q3,
IMEX, CFL control; 2,099 steps to t = 2 in 55 s at np 4 on the cloud scratch
build; det F stays in [0.89, 1.11]). An example of the keys, not a validated
benchmark.

**Validated:** the in-line oscillating cylinder of Dütsch et al. (1998), Re 100,
KC 5, in [benchmarks/oscillating_cylinder/](../benchmarks/oscillating_cylinder/):
c_d 2.091 and c_i 1.460 against 2.09 and 1.45 (other computations 1.45-1.48),
converged to about 0.1 % in mesh, time step and domain size; 8 min at np 4 on the
CPU. Transverse lock-in and Dütsch's velocity profiles are not compared yet.

## 4. Verification (`test/ale_fluid_test.cpp`, fast tier, np 1/2/4)

| Test | What it pins | Result |
|---|---|---|
| A1 | zero motion through the moving-mesh path = the fixed-mesh run (Navier-Stokes Taylor-Green) | to 1e-9 |
| A2 | free stream on a mesh whose interior oscillates, 2D and 3D (the GCL) | u uniform to 1e-9, p constant to 1e-8 |
| A3 | a box moved rigidly with u = w on its walls, constant and oscillating motion, Stokes and NSE | u = w_h to 1e-9 |
| A4 | a linear Stokes flow on a mesh moved at constant rates: translated exact; stretched, the starter's O(dt^2) error | translation exact; rate 2.02 |
| A5 | a vetoed solve on a moving mesh: the mesh goes back, the step is cut and retried, the nodes end where the motion says | to 1e-12 |
| A6 | Taylor-Green on an oscillating mesh, Q3, h = 1/2, 1/4, 1/8: velocity L2 | rates 4.03, 4.32 (theory 4) |
| A7 | the same at Q5, dt = 0.04, 0.02, 0.01 | rates 2.83, 2.24 (theory 2, BDF2/EXT2) |
| A8 | a deck whose whole boundary moves rigidly (`motion` on a no_slip group): IMEX chosen, u = w, the nodes translated | to 1e-9 |
| A9 | a checkpoint mid-march of a deck with a moving wall and an elastically moved interior, restarted | `rebuild`: = the uninterrupted run to 1e-13; `in_place`: to 1e-10 (the restart's AMG is built on its own mesh) |
| A10 | the in-place refresh against `rebuild`: AMG every move and every 3 moves; default path, LOR-AMG velocity block, the Schur outer method | to 2e-16 (every move), < 1e-12 otherwise |

A4's stretched case is not exact because the starter's explicit C(w) u^0
reads u^0 on the moved nodes, off by dt (grad u) w where grad w is not zero; the
BDF2 steps after it are exact for this flow.

Run in the cloud session on a scratch build (MFEM v4.10 from GitHub, apt hypre
and Open MPI, not the Spack toolchain); CI is the reference build. Never run on
a GPU: `MeshVelocityField::Update` projects on the host once per step.

## 5. Next

1. The the FSI stages: St. Venant-Kirchhoff solid, the Turek-Hron geometry,
   the partitioned coupling with dynamic Aitken and the Robin conditions, to
   all nine Turek-Hron problems.
