# Fluid-structure interaction

The flow of a deck coupled with an elastic solid: the partitioned (segregated)
coupling of the ALE fluid ([ale_fluid.md](ale_fluid.md)) and the solid
([elasticity.md](elasticity.md)), iterated to convergence in every time step.
Stage F3 of the FSI plan (the scoping document of 2026-10-21, section 11):
human 2026-10-21, "it's a segregated solve and you'll leverage gslib right?
You also should use atkin acceleration and be iterating back and forth like
is common in the literature", and the target, all nine Turek-Hron problems,
fully coupled and iterated; 2026-10-22, "then do the coupling".

Built so far (CPU only): the four interface conditions, Dirichlet-Neumann
(the default), Robin-Neumann, Neumann-Dirichlet and Robin-Robin (sections
2-4), with constant, dynamic Aitken or IQN-ILS updates, the linear
(small-strain) solid or the St. Venant-Kirchhoff one (`fsi.solid`, stage F5;
Dirichlet-Neumann and dynamic only; FSI1-3 in
benchmarks/turek_hron/README.md), static or dynamic, matching or non-matching
interfaces by point search. Not yet: the St. Venant-Kirchhoff solid under the
other three conditions, restarts, a GPU run.

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

**The solid's traction** (`fsi.solid`). The fluid's moments are those of a
Cauchy traction on the moved interface, sigma n dGamma, and those of the
nominal traction (per reference area) on the reference one, P N dGamma_0: the
same numbers. The St. Venant-Kirchhoff solid (`fsi.solid: hyperelasticity`,
`Hyperelasticity`, total Lagrangian) takes the nominal one, M_Gamma0 t_0 =
-rho_f r with the reference interface's mass (`FsiCase::NominalTractionFromLoad`),
a dead load within each subiteration (the subiterations themselves carry the
follower part, so the solid needs no follower-load linearization). The
small-strain solid (`linear_elasticity`, the default) takes the Cauchy traction
of the moved interface on its undeformed one; the two differ at the order of
the strain, the small-strain model's own error, and the F3/F4 results stay
bitwise. The other conditions pass the solid a traction field per current
area (Robin-Neumann, Robin-Robin) or take its reaction as one
(Neumann-Dirichlet), so they refuse the St. Venant-Kirchhoff solid until their
iterate becomes the moments. `FsiSolid` (fsi/fsi_solid.hpp) is the common
interface: an adapter over either class, whose methods have the same names.

**The St. Venant-Kirchhoff solid must be dynamic** (`solid.analysis:
dynamic`; a static one is refused at setup, test
`run_case_rejects_static_svk_fsi`). A static solid has no mass, the worst
added-mass case for Dirichlet-Neumann: the interface velocity w ~ d / dt
turns a small displacement into a large load change, and the flag's bending
compliance turns that back into a large displacement. On FSI1's first step
the second subiteration's displacement jumps from 6e-6 m to 0.011 m with
either solid (traced 2026-10-23: every Newton solve converged, CG to 1e-13).
The linear solid stays linear however far out IQN-ILS explores, and the
subiterations recover; the SVK flag meets trial displacements of 0.1-0.5 m,
longer than the flag itself, folds and finds spurious equilibria, and the
subiterations never come back. The dynamic flag reaches the same steady
state (benchmarks/turek_hron/fsi1_svk.yaml).

**The force ripple at a density ratio of one** (FSI3, 2026-10-23). With
rho_s = rho_f the added mass turns the coupling's leftover interface error
into a force through 1/dt^2: at `fsi.rtol` 1e-4 the lift carried a
step-to-step ripple that grew with the flag's motion (+-25,000 against a
reference amplitude of 150 by t = 4.5), the displacement smooth. Undamped
generalized-alpha (rho_inf 1) made it 17x worse than a BDF2 solid, whose
velocity matches the fluid's BDF2 mesh velocity. The FSI2/FSI3 decks
therefore step the solid by BDF2 and couple to `fsi.rtol` 1e-6 (the ripple
100x smaller, 4.4 -> 5.0 subiterations per step), with `solver.rtol` 1e-9:
the flow solve must sit well below the coupling tolerance, or IQN-ILS stalls
on its noise (at the default it stopped in the third step).

**Newton at tiny loads.** The SVK residual has a roundoff floor near
eps E |Omega_s| (4.5e-11 on the Turek-Hron flag), and both of Newton's
relative tests can sit below it when the load is small (the first steps of a
ramp), so Newton runs to its cap. The FSI decks set `solid.newton_atol: 1e-9`.

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

## 3. Neumann-Dirichlet (`fsi.coupling: neumann_dirichlet`)

The roles swapped: the solid's interface is prescribed, the fluid's loaded.
The iterate is the interface displacement d. A subiteration: the solid with
its interface at d (`LinearElasticity::AddDisplacement` with the transferred
field), its reaction R there (`LinearElasticity::TrackReaction`: the
unconstrained residual at the essential dofs, unweighted by the scheme:
generalized-alpha's residual is (1 - alpha_f) R^{n+1} + alpha_f R^n over
(1 - alpha_f), the trapezoidal start's R^0 + R^1, so the previous reaction is
part of the state), R as a traction field on the solid's interface, moved to
the fluid's and handed to it as a traction boundary (a Robin boundary with
alpha = 0); the fluid on the mesh at d gives its interface velocity u, and
d~ = d + (u - w) / c0, w = c0 d + (older levels) the mesh velocity's BDF:
the displacement u integrates to (`Case::MeshVelocityLead`). Residual d~ - d,
so u = w at the fixed point. The clamp's nodes are left out of R (the
corner node between the flag and the cylinder carries the clamp's reaction);
the two conditions treat that node differently, which moves the velocity
next to the flag's root by 5e-4 of its scale and leaves the rest within
1e-6 (test F7).

**It suits soft solids, not the Turek-Hron flag.** Its convergence is the
reciprocal of Dirichlet-Neumann's: on a flag 1,000 times softer (E 1.4e3,
F7) it takes 30 subiterations over two steps against Dirichlet-Neumann's
148; on the FSI1 flag (E 1.4e6) the flag's through-thickness squeeze mode,
stiff under a prescribed displacement and free under a fluid traction, gives
the map a gain of order 100, and IQN-ILS does not get under 0.1 in 50
subiterations on the first step of `cases/fsi_flag.yaml`. FSI1 (coarse,
IQN-ILS) stops the same way in its first two steps.

## 4. Robin-Robin (`fsi.coupling: robin_robin`)

Robin-Neumann's fluid side plus a Robin condition on the solid:
sigma_s n_s + alpha_s w = -sigma_f n + alpha_s u, alpha_s = `fsi.alpha_solid`
(> 0, no default). Linearized in d, alpha_s w is an elastic support
k = alpha_s c0 on the interface (`LinearElasticity::AddSupport`: k (u, v)
on its faces in the stiffness and both preconditioners), c0 the BDF weight
at the march's order (1.5 / dt under BDF2). The iterate is the solid's
Robin data g (kinematic, as Robin-Neumann's tau): the solid takes
-rho_f g with the support, so sigma_s n_s = -rho_f g - k d; the fluid's
Robin data is g + (k / rho_f) d, and its traction tau~, u and w give
g~ = tau~ - (alpha_s / rho_f)(u - w) - (k / rho_f) d. At the fixed point
(alpha_f + alpha_s)(u - w) = 0 and the tractions balance, whatever k
(it only has to be the same in both places). alpha_s = 0 is Robin-Neumann
(test F6: both within 1e-6 of Dirichlet-Neumann).

**On the flag it gets Dirichlet-Neumann's subiterations, not its cost.**
`cases/fsi_flag.yaml` (3 steps): Robin-Neumann 59 subiterations, alpha_s 100
/ 1,000 / 10,000 / 1e5: 59 / 53 / 55 / 107, 1e6 diverges; Dirichlet-Neumann
33. FSI1 (coarse, t = 15, IQN-ILS, np 4) with alpha_s 1,000: 4.3
subiterations per step (Dirichlet-Neumann 4.4, Robin-Neumann 6.6), the same
answer (drag +0.31 %, lift -0.38 %, tip u_y -0.04 %, u_x +6.5 %), but
1,095 s against Dirichlet-Neumann's 389 s: each subiteration's flow solve is
harder with the Robin term in it, as for Robin-Neumann. The parameter formulas in the
literature (Badia, Nobile and Vergara 2008; Gerardo-Giorda, Nobile and
Vergara 2010) are for thin walls wetted on one side, as for Robin-Neumann.

## 5. The accelerators (`fsi.acceleration`)

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
(section 2). On FSI3 itself, IQN-ILS to 1e-6 takes 4.5 subiterations per
step (benchmarks/turek_hron/README.md).

## 6. The interface maps (`fsi/interface_transfer`)

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

## 7. The deck

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
  solid: linear_elasticity  # | hyperelasticity (St. Venant-Kirchhoff, solid.model)
  coupling: dirichlet_neumann  # | robin_neumann | neumann_dirichlet | robin_robin
  alpha_fluid: 0            # robin_*: <= 0 face by face (section 2)
  structure_thickness: 0    # robin_*: > 0 one alpha = rho_s H / dt
  alpha_solid: 0            # robin_robin: the solid's Robin coefficient, > 0
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

## 8. Tests (`fsi_test`, fast tier, np 1/2/4)

F1 the maps are exact on a linear field both ways; F2 a fluid trial step
undone and taken again reproduces itself, and a committed one equals a
plain step; F3 the traction integrates to rho_f times the force on the flag
(the residual's sum, and through the map on the solid's interface); F4
coupled steps converge with IQN-ILS (with and without reuse) and Aitken, the
fluid's interface sits at the solid's, IQN-ILS needs fewer subiterations than
Aitken; F5 a static solid on a ramped start; F6 Robin-Neumann and Robin-Robin
(alpha_s 1,000) converge to Dirichlet-Neumann's steps (velocity to 1e-6, the
tip to 1e-6 of the displacement), with u = w on the interface; F7 so does
Neumann-Dirichlet on a soft flag, in fewer subiterations; F8 the St.
Venant-Kirchhoff solid reaches the linear one's steps at small strain (the
tip to 1e-4 of the displacement). `robin_test` R1-R3
covers the fluid's Robin condition alone. The deck end to end:
`run_case_fsi_flag` ([cases/fsi_flag.yaml](../cases/fsi_flag.yaml), ten
dynamic steps, 9-12 subiterations each with IQN-ILS, 5 s at np 2).
