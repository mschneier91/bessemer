# Fluid-structure interaction

The flow of a deck coupled with an elastic solid: the partitioned (segregated)
coupling of the ALE fluid ([ale_fluid.md](ale_fluid.md)) and the solid
([elasticity.md](elasticity.md)), iterated to convergence in every time step.
Stage F3 of the FSI plan (the scoping document of 2026-10-21, section 11):
human 2026-10-21, "it's a segregated solve and you'll leverage gslib right?
You also should use atkin acceleration and be iterating back and forth like
is common in the literature", and the target, all nine Turek-Hron problems,
fully coupled and iterated; 2026-10-22, "then do the coupling".

Built so far (CPU only): Dirichlet-Neumann subiterations with constant,
dynamic Aitken or IQN-ILS updates of the interface displacement, the linear
(small-strain) solid, static or dynamic, matching or non-matching interfaces
by point search. Not yet: the Robin conditions (F4; Robin-Neumann is to be the
default), Neumann-Dirichlet, the St. Venant-Kirchhoff solid (F1, the
maintainer's), restarts, a GPU run.

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

## 2. The accelerators (`fsi.acceleration`)

- **`aitken`** (default; human 2026-10-21): dynamic Aitken relaxation
  (Kuttler and Wall 2008), d_{k+1} = d_k + omega_k r_k, omega_0 =
  `fsi.relaxation` at the first subiteration of every step, then omega_k =
  -omega_{k-1} r_{k-1}.(r_k - r_{k-1}) / |r_k - r_{k-1}|^2 within
  [-omega_max, omega_max].
- **`constant`**: omega = `fsi.relaxation` throughout.
- **`iqn_ils`**: the interface quasi-Newton method with a least-squares
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
2005): the Robin conditions of stage F4 are the remedy, and FSI3 the test.

## 3. The interface maps (`fsi/interface_transfer`)

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

## 4. The deck

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
  acceleration: aitken      # aitken | constant | iqn_ils
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
near 1e-5 and the step failed). Time steps are fixed
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

## 5. Tests (`fsi_test`, fast tier, np 1/2/4)

F1 the maps are exact on a linear field both ways; F2 a fluid trial step
undone and taken again reproduces itself, and a committed one equals a
plain step; F3 the traction integrates to rho_f times the force on the flag
(the residual's sum, and through the map on the solid's interface); F4
coupled steps converge with IQN-ILS (with and without reuse) and Aitken, the
fluid's interface sits at the solid's, IQN-ILS needs fewer subiterations than
Aitken; F5 a static solid on a ramped start. The deck end to end:
`run_case_fsi_flag` ([cases/fsi_flag.yaml](../cases/fsi_flag.yaml), ten
dynamic steps, 9-12 subiterations each with IQN-ILS, 5 s at np 2).
