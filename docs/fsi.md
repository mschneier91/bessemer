# Fluid-structure interaction

The flow of a deck coupled with an elastic solid: the partitioned (segregated)
coupling of the ALE fluid ([ale_fluid.md](ale_fluid.md)) and the solid
([elasticity.md](elasticity.md)), iterated to convergence in every time step.
Stage F3 of the FSI plan (the scoping document of 2026-10-21, section 11):
human 2026-10-21, "it's a segregated solve and you'll leverage gslib right?
You also should use atkin acceleration and be iterating back and forth like
is common in the literature", and the target, all nine Turek-Hron problems,
fully coupled and iterated; 2026-10-22, "then do the coupling".

Built so far (CPU only): Dirichlet-Neumann (the default) and Robin-Neumann
subiterations (section 2) with constant, dynamic Aitken or IQN-ILS updates,
the linear (small-strain) solid, static or dynamic, matching or non-matching
interfaces by point search. Not yet: Neumann-Dirichlet and Robin-Robin (F4),
the St. Venant-Kirchhoff solid (F1, the maintainer's), restarts, a GPU run.

## 1. The coupling

A time step from t^n to t^{n+1} (`FsiCase::Step`):

1. **Predict** the interface displacement d^{n+1} from the committed steps
   (`fsi.predictor`: 0 the last one, 1 linear, 2 quadratic extrapolation).
2. **Fluid trial step** with the interface at d: the mesh moves with the
   interface displacement as the boundary data of the mesh motion (the motion
   space on the reference mesh holds d; `Case::InterfaceDisplacement()`), the
   interface is a wall moving with the mesh (u = w there, boundary type
   `interface`), and the fluid steps (`Case::TrialStep`), recording nothing.
3. **The traction.** The fluid's force on the interface is its momentum
   residual at the interface dofs, r = A u + N u - B^T p - b, the same one the
   lift and drag come from (John's volume-integral form, developing.md 5.6): the
   force on the body against a test function v is -r . v. A traction field
   t on the fluid's interface with the same moments solves M_Gamma t = -rho_f r
   at the interface dofs, M_Gamma the boundary mass matrix on the moved
   interface (CG with Jacobi; a boundary-sized problem). Its integral is
   rho_f times the force on the interface exactly (test F3). The traction is
   interpolated to the solid's interface nodes and applied as the solid's
   traction boundary condition.
4. **Solid step** from the start of the step (`LinearElasticity::ImportState`
   of the state at t^n, then `Step(dt)`), or a static solve when
   `solid.analysis: static` (quasi-static coupling, right for a steady
   problem such as FSI1).
5. **Residual.** The solid's displacement interpolated to the fluid's
   interface nodes, d~ = G(d), and r = d~ - d in the true dofs of the motion
   space. Converged when |r| <= fsi.rtol |d~| + fsi.atol: the fluid's trial
   step is committed (`Case::CommitTrialStep`: the step is recorded, logged
   and written as any other). Otherwise the fluid step is undone
   (`Case::UndoTrialStep`: the mesh moves back, the march's state is restored;
   the solvers keep their projection bases) and the next d comes from the
   accelerator (section 2). A step that does not converge in
   `fsi.max_iterations` stops the run (`status: diverged`).

This is the Dirichlet-Neumann coupling (`fsi.coupling: dirichlet_neumann`):
the fluid gets the interface's position and velocity, the solid the fluid's
traction. The fluid's interface velocity is the mesh velocity, the BDF
derivative of the interface's positions, so the kinematic condition holds in
the fluid's own discrete sense (geometric consistency), not as the solid's
velocity.

**The solid's traction is applied on the reference interface.** The traction
is a Cauchy traction on the moved interface; the small-strain solid integrates
it over the undeformed one. The two differ at the order of the strain, the
small-strain model's own error. The St. Venant-Kirchhoff solid will take the
nominal traction.

## 2. Robin-Neumann (`fsi.coupling: robin_neumann`)

The fluid's interface takes a Robin condition instead of u = w (Badia, Nobile
and Vergara 2008): sigma_f n + alpha_f u = tau + alpha_f w, tau the fluid's
traction of the previous subiteration and w the mesh velocity of the solid's
interface; the solid takes the fluid's traction as before. At the fixed point
tau is the fluid's own traction, so u = w and the coupled solution is
Dirichlet-Neumann's, whatever alpha_f (test F6: the two agree to 1e-7 over two
impulsive steps). The fluid's side: `BoundaryConditions::AddRobin` (a
boundary mass alpha (u, v) in the momentum block, `StokesOperator::Robin`;
the load alpha (w, v) + (tau, v) assembled on the current mesh every step,
`StokesTimeIntegrator::SetBoundaryTraction`); the momentum residual leaves
the Robin terms out, so forces and the solid's load stay the physical ones
(`robin_test` R1-R3: an exact Robin channel flow, the residual identity, the
no-slip limit as alpha grows).

**The iterate is the traction, not the displacement.** A step: the solid
under -rho_f tau, the mesh to its interface, the fluid with Robin data tau,
its traction tau~; residual tau~ - tau, relative to |tau~|, and the
accelerator updates tau. Iterating on d instead leaves tau as hidden state,
and IQN-ILS, which models d~(d), stalled (|r|/|d~| ~ 0.5 for 60
subiterations). tau is a field on the interface, not the residual's moments:
the moments of one traction change as the interface moves, and reading them
on the mesh of the start of the step moved the converged flag by 3 % on an
impulsive start.

**alpha_f.** Badia, Nobile and Vergara's rho_s H / dt for a thin wall
(`fsi.alpha_fluid`, or `fsi.structure_thickness` H > 0); by default face by
face, rho_s h / dt with h the solid's thickness behind the face (a ray cast
through the solid's boundary along the face normal), capped at c_P dt, the
P-wave's reach in a step (`FsiCase::BuildRobinCoefficient`): on the
Turek-Hron flag 4,000 on its sides and 54,800 on its tip at dt 0.005.

**On the flag it is slower than Dirichlet-Neumann** (human 2026-10-22: the
default is Dirichlet-Neumann with IQN-ILS):

| | DN, Aitken | DN, IQN-ILS | RN, Aitken | RN, IQN-ILS |
|---|---|---|---|---|
| `cases/fsi_flag.yaml`, 10 steps | 201 | 103 | fails | 171 |
| FSI1 (coarse, t = 15), per step | fails at step 1 | 4.4 | fails at step 1 | 6.6 |

Both reach the same FSI1 answer (drag +0.30 % against the reference in
both, lift -0.48 % and -0.35 %, tip u_y -0.06 %, u_x +6.5 %).

The flag is a wall wetted on both sides. Its bending modes want alpha_f near
their impedance (larger diverges), its through-thickness squeeze mode is
some fifty times stiffer and converges at about 1 - alpha_f / Z_squeeze per
subiteration (0.97 measured): no single alpha_f serves both. solids4foam
reports the same on Turek-Hron (FSI3: 151 subiterations per step for
Robin-Neumann against 9 for Dirichlet-Neumann with IQN-ILS; FSI1 17.6
against 5.9; their issues #493 and #529), with separate coefficients for the
two faces' symmetric and antisymmetric motion as the open remedy. The
parameter formulas in the literature are derived for thin walls wetted on
one side (arteries). A Robin-Neumann subiteration also costs about three
times a Dirichlet-Neumann one on FSI1: the flow's solve is harder with the
Robin term in it (the pressure preconditioner does not see it).

## 3. The accelerators (`fsi.acceleration`)

- **`iqn_ils`** is the default (human 2026-10-22); Aitken fails FSI1's first
  step under both conditions.
- **`aitken`**: dynamic Aitken relaxation
  (Kuttler and Wall 2008), d_{k+1} = d_k + omega_k r_k, omega_0 =
  `fsi.relaxation` at the first subiteration of every step, then omega_k =
  -omega_{k-1} r_{k-1}.(r_k - r_{k-1}) / |r_k - r_{k-1}|^2 within
  [-omega_max, omega_max].
- **`constant`**: omega = `fsi.relaxation` throughout.
- **IQN-ILS**: the interface quasi-Newton method with a least-squares
  inverse Jacobian (Degroote, Bathe and Vierendeels 2009): with V the
  differences of the residuals and W those of the outputs d~, d_{k+1} = d~_k
  + W c, c the least-squares solution of V c = -r_k (modified Gram-Schmidt
  with global inner products; a column whose part orthogonal to the newer
  ones is below 1e-4 of its norm is dropped). `fsi.reuse` keeps the columns of
  that many earlier steps. The filter matters: at 1e-10 the least-squares
  problem picked up the fluid and mesh solvers' noise once the residual was
  near 1e-5 and the iteration diverged (static flag, `fsi_test` F5).

Measured on `fsi_test` F4 (Turek-Hron geometry at Q2/Q1, the FSI1 solid and
densities rho_s = rho_f = 1000, an impulsive start, rtol 1e-8): IQN-ILS 27
subiterations in two steps, IQN-ILS reusing two steps 25, Aitken 54-62
(rank-dependent), the constant relaxation at 0.1 4.7 times Aitken's over four
steps, at 0.5 divergent. A density ratio of one is the hard case of
Dirichlet-Neumann coupling (the added-mass effect; Causin, Gerbeau and Nobile
2005); on this flag IQN-ILS handles it better than the Robin condition does
(section 2), and FSI3 is the test still ahead.

## 4. The interface maps (`fsi/interface_transfer`)

Both directions interpolate a field at the destination's interface nodes:
the solid's displacement to the nodes of the motion space on the fluid's
reference mesh, the fluid's traction (on the fluid's velocity space, taken
to the reference mesh: the same dof layout) to the solid's interface nodes.
Both meshes are in their reference configurations, so the search runs once,
at setup: every rank gathers all the interface nodes and searches its own
elements (`Mesh::FindPoints`, local); the lowest rank that finds a node owns
it; a node no rank finds aborts the setup. A transfer is then one evaluation
per owned node and one sum over the ranks of the interface's values.

On the Turek-Hron meshes the fluid and solid interfaces share their nodes
(`mesh.cylinder_channel.part: solid`) and the map is the identity on the
interface (a linear field crosses exactly both ways; test F1). On
non-matching interfaces it is interpolation, not a conservative projection.
MFEM's `FindPointsGSLIB` is the planned alternative for large or 3D
interfaces (the human's gslib decision; MFEM is built with gslib in every
real environment, not in the cloud scratch build these were developed on).

## 5. The deck

The flow deck as usual, with the wetted boundary as a group of type
`interface`, the solid in section `solid` (its material, `analysis`, time
scheme, `probe`), and section `fsi`:

```yaml
boundary_conditions:
  - {select: [flag], type: interface}     # u = w, moves with the solid
fsi:
  enabled: true
  solid_fixed: [cylinder]   # the solid mesh's clamp
  solid_interface: []       # empty: the fluid's interface names
  fluid_density: 1000       # the solid's load is rho_f times the kinematic one
  coupling: dirichlet_neumann  # dirichlet_neumann | robin_neumann
  alpha_fluid: 0            # robin_neumann: <= 0 face by face (section 2)
  structure_thickness: 0    # robin_neumann: > 0 one alpha = rho_s H / dt
  acceleration: iqn_ils     # iqn_ils | aitken | constant
  relaxation: 0.5           # omega_0 (aitken), omega (constant), first step (iqn_ils)
  omega_max: 1
  reuse: 0                  # iqn_ils: earlier steps kept
  rtol: 1.0e-6
  atol: 1.0e-12
  max_iterations: 50
  predictor: 2
  print_level: 0            # 1: every subiteration's residual on rank 0
```

The solid's mesh is the solid part of the deck's geometry: today
`mesh.geometry: cylinder_channel` with a flag, whose `part: solid` is built
next to the fluid mesh with the same interface nodes. **Set the flow solver's
tolerance well below `fsi.rtol`** (`solver.rtol` 1e-9 against 1e-6 in
[cases/fsi_flag.yaml](../cases/fsi_flag.yaml)): the subiterations cannot
converge below the flow solve's own error, and IQN-ILS fed with that noise
wanders off (at the default `solver.rtol` the smoke deck's residual stalled
near 1e-5 and the step failed). A static solid (`solid.analysis: static`)
needs a far larger gap: with no inertia to set against the fluid's added
mass, the subiteration map amplifies the flow solve's error about 1e5 times
(`fsi_test` F5: `solver.rtol` 1e-10 left the residual wandering at 1e-5
against `fsi.rtol` 1e-8; 1e-12 converges in 14-18 subiterations). Time steps are fixed
(`time.step_control: fixed`); a failed fluid solve inside the coupling stops
the run instead of cutting the step. Restarts are refused (the solid's state
is not checkpointed yet). The run writes `<name>_fsi.csv` (per step: time,
subiterations, final residual, the last relaxation factor, the force on the
interface, the probe's displacement) and a `fsi` section in the summary;
the reference keys `displacement_x` and `displacement_y` compare the probe's
displacement. Example: [benchmarks/turek_hron/fsi1.yaml](../benchmarks/turek_hron/fsi1.yaml)
(results in its README: on the coarse mesh with the linear solid drag +0.30 %,
lift -0.48 %, tip u_y -0.06 %, u_x +6.5 %, of which +4.9 % is the bending
shortening a small-strain solid lacks). A steady problem has to be marched well
past its inflow ramp: FSI1's flag still creeps at t = 5 (u_y 4 % short) and is
steady from t = 10.

## 6. Tests (`fsi_test`, fast tier, np 1/2/4)

F1 the maps are exact on a linear field both ways; F2 a fluid trial step
undone and taken again reproduces itself, and a committed one equals a
plain step; F3 the traction integrates to rho_f times the force on the flag
(the residual's sum, and through the map on the solid's interface); F4
coupled steps converge with IQN-ILS (with and without reuse) and Aitken, the
fluid's interface sits at the solid's, IQN-ILS needs fewer subiterations than
Aitken; F5 a static solid on a ramped start; F6 Robin-Neumann converges to
Dirichlet-Neumann's steps, with u = w on the interface. `robin_test` R1-R3
covers the fluid's Robin condition alone. The deck end to end:
`run_case_fsi_flag` ([cases/fsi_flag.yaml](../cases/fsi_flag.yaml), ten
dynamic steps, 9-12 subiterations each with IQN-ILS, 5 s at np 2).
