# Mesh motion

The mesh-motion half of an ALE (arbitrary Lagrangian-Eulerian) formulation, the
second step toward fluid-structure interaction after the linear elasticity
solver ([elasticity.md](elasticity.md)). Human 2026-10-20: "add the capability
to deform a mesh using the linear elasticity solver ... make it general that a
different type of solver (hyper elasticity etc) can be used". Since
2026-10-21 a deck can deform its mesh by prescribed boundary displacements
(`equation: mesh_motion`, section 3; human: "add the capability to deform a
given mesh using the structural solver ... specify like a function on the
boundary"). Since 2026-10-21 a flow case can move its mesh at every step
with a `MeshMotion` (`Case::SetMeshMotion`, IMEX, no deck keys yet): the ALE
fluid, [ale_fluid.md](ale_fluid.md).

## 1. Formulation

The mesh nodes follow x = X + d(X, t), X the node positions of the initial
(reference) mesh. The displacement d solves a pseudo-solid problem on the
reference mesh with Dirichlet data only: d = g(X, t) on the moving boundaries
(the body, later the solid's interface displacement) and d = 0 on every other
boundary. The default model is small-strain linear elasticity with Jacobian
stiffening (Stein, Tezduyar and Benney, J. Appl. Mech. 70 (2003) 58-63):

    (2 mu s_K eps(d), eps(v)) + (lambda s_K div d, div v) = 0,
    s_K = (V_mean / V_K)^chi,

V_K the element's reference volume and V_mean the mean over the mesh. With
chi = 1 (the default) the small elements, typically the boundary layer at a
moving body, are stiffer and move nearly rigidly, so the large elements away
from it absorb the deformation; chi = 0 is a homogeneous solid. E = 1 (a
displacement-driven problem does not depend on it), Poisson's ratio 0.3 by
default, plane strain in 2D.

**Total displacement from the reference mesh** (not increments on the current
mesh). The operator and its LOR-AMG are built once for the run; each move is
one CG solve; a boundary that returns to its place brings every node back
exactly, and nothing drifts over a long run. The cost: linear elasticity on
the reference mesh is accurate for moderate motions and degrades for large
rotations (a rotation is not strain-free in the linear model). That is the job
of a hyperelastic model through the same interface (section 2); an incremental
mode on the current mesh is the other standard remedy, not built.

**High-order geometry.** The model solves at the order of the mesh's nodal
space by default, and the displacement is interpolated onto that space at the
nodes (the identity when the two spaces are the same). So the geometry stays
isoparametric, and a boundary moved by a rigid motion keeps its exact shape at
the nodes: the DFG cylinder's nodes stay on the moved circle to roundoff (test
M3). A mesh without nodes is given nodes at construction
(`MeshMotion::Options::geometry_order`, default the displacement order).

**Mesh velocity.** `MeshMotion` keeps the node displacements of the last
`history` states (default 4) and returns w = sum_j c_j x^{n+1-j}, the
variable-step BDF derivative of the node positions (`time/multistep_coeffs`)
at the newest state, exact for node motions polynomial in time up to the BDF
order. That is the discrete mesh velocity a BDF-ALE fluid step uses (Formaggia
and Nobile's BDF-ALE schemes); which one satisfies the geometric conservation
law is the ALE fluid's choice, and it is made there.

**Mesh quality.** `JacobianRange()` returns the minimum and maximum of
det F = det(dx/dX), the ratio of the current to the reference Jacobian
determinant, at the Gauss-Lobatto points of every element. min > 0 is
necessary for a valid mesh; it is sampled, not a bound (a positive Bernstein
bound would be).

**Choosing the settings.** What the stiffening buys, and what it costs, depends
on the mesh, so here is what was measured (2026-10-21, 2D, np 4, one move,
`cases/`-style decks; det F is the minimum over an element's nodes):

| Mesh and motion | chi | min det F near the body / wall layer | min det F elsewhere | CG its |
|---|---|---|---|---|
| DFG cylinder (208 Q3), cylinder moved 0.05 (one radius) | 0 | 0.62 | 0.68 | 27 |
| | 1 | 0.90 | 0.62 | 29 |
| | 2 | 0.95 | 0.58 | 32 |
| same, moved 0.1 (two radii, 0.05 from the wall) | 0 | 0.17 | 0.22 | 27 |
| | 1 | 0.84 | 0.12 | 29 |
| | 2 | 0.91 | 0.10 | 32 |
| 2 x 1 rectangle, 16 x 16 Q4, tanh layer (beta 3) at the bottom, parabola 0.3 | 0 | 0.68 | 0.68 | 22 |
| | 1 | 0.84 | 0.63 | 23 |
| | 2 | 0.75 | 0.54 | 22 |

("Near the body": elements with their centre within 0.08 of the cylinder's;
"wall layer": the bottom row.) So stiffening does not make the mesh as a whole
better: it moves the distortion out of the small elements into the large ones.
That is the trade wanted when the small elements are a boundary layer whose
shape the flow needs, and the large ones have room.

- **`stiffening` (chi).** Keep the default **1** when the mesh is graded toward
  a moving body or wall (the usual case): it kept the cylinder's ring 2.6 times
  less distorted (max |det F - 1| 0.15 against 0.39) at the same overall
  minimum. On a uniform mesh it does nothing (s_K = 1 everywhere; the example
  decks). **0** gives the best overall minimum det F and suits a mesh whose
  small elements do not matter more than the rest. **2** protects the body's
  ring further (0.09) but costs the far field, can make a graded wall layer
  worse than chi = 1 (the rectangle), and adds CG iterations (up to +19 %); use it
  only when the ring must stay nearly rigid, and check `det_f_min`. In these
  runs chi = 1 was the better balance.
- **`poisson_ratio`.** Keep **0.3**. Raising it resists volume change: on the
  uniform rectangle 0.45 lifted min det F from 0.68 to 0.74 but lowered the
  element Jacobian ratio from 0.89 to 0.84 (more shear) and took 33 % more CG
  iterations. Try 0.4-0.45 only when compression (det F well below 1) is what
  fails; never near 0.5, where the displacement formulation locks.
- **Order.** The default (the velocity order) keeps the geometry isoparametric
  with the flow; a lower order is cheaper but gives the fluid a coarser
  geometry than its velocity space.
- **`steps`.** They do not change the final shape (total displacement); use
  them to see the motion in ParaView or to find when det F drops.
- **When it is not enough.** The second cylinder case is near the limit: two
  radii of motion leave min det F 0.1-0.2, close to inversion. Watch
  `det_f_min` and `jacobian_ratio_min`, and treat values below about 0.2 as
  a warning (a working rule; its effect on a flow is not measured yet). Large
  motions and rotations (which linear elasticity on the reference mesh treats
  as strain) need the hyperelastic model, a finer mesh where the deformation
  lands, or a remesh; none of these is built yet.

## 2. Code

- `src/ale/mesh_motion_model.hpp`: `MeshMotionModel`, the interface a model
  implements, and `MeshMotionModelFactory`. A model is built on the reference
  mesh at an order MeshMotion gives it, receives fixed and displaced boundary
  attributes (`AddFixed`, `AddDisplacement` with a `VectorCoefficient` of X
  and t), and on `Solve(t)` produces `Displacement()`, a vector H1 field on the
  reference mesh. A hyperelastic model (St. Venant-Kirchhoff or neo-Hookean,
  total Lagrangian, Newton) implements the same interface, with Newton inside `Solve`; so would a
  harmonic or biharmonic extension.
- `src/ale/linear_elastic_motion.{hpp,cpp}`: `LinearElasticMotion`, the model
  above on `LinearElasticity` (`solid/`), with `Factory(options)` for
  MeshMotion and `StiffeningFactors(mesh, chi)`. `LinearElasticity` gained
  `Options::element_stiffness` (a per-element factor on lambda and mu, carried
  into the LOR preconditioner through the parent element, as the fluid's
  grad-div coefficient is) and `Solve(t)` (a static solve with the data at t).
- `src/ale/mesh_motion.{hpp,cpp}`: `MeshMotion`. Construction copies the mesh
  (the reference, which never moves) and builds the model on the copy.
  `Move(t)` solves, sets the nodes to X + d and calls `Mesh::NodesUpdated()`
  (MFEM's cached geometric factors are dropped; face-neighbour nodes are
  re-exchanged when the mesh has face-neighbour data). A boundary attribute
  given neither `AddFixed` nor `AddDisplacement` is fixed: there are no
  free or sliding mesh boundaries yet.

Usage:

```cpp
incns::LinearElasticMotion::Options elastic; // nu 0.3, chi 1
incns::MeshMotion motion(mesh, incns::MeshMotion::Options(),
                         incns::LinearElasticMotion::Factory(elastic));
mfem::VectorFunctionCoefficient body(dim, [](const mfem::Vector& X, double t,
                                             mfem::Vector& d) { ... });
motion.AddDisplacement(body_attribute, body); // the rest stays fixed
motion.Move(t);                               // the mesh's nodes are moved
motion.MeshVelocity(2, w);                    // BDF2 node velocity
```

**What it does not do yet.** Anything that cached the mesh's geometry before a
move (a flow `Case`'s operators, the grad-div coefficient's element sizes, the
CFL estimator's Jacobians) is stale after it: rebuilding or refreshing those is
the ALE fluid's work. Refinement after construction aborts (the reference copy
would have to be refined in step). Sliding boundaries (zero normal
displacement along flat walls) are not done. Memory: the reference copy (topology and nodes), the node
history (`history` node vectors) and the elasticity solver (its LOR-AMG
dominates, elasticity.md section 3).

## 3. Deforming a mesh from a deck

`equation: mesh_motion` makes `run_case` deform the deck's mesh instead of
solving a flow (`ale/mesh_motion_case`, `MeshMotionCase`). Every real boundary
is in exactly one `boundary_conditions` group, of type `displacement` (moved
by the given displacement of its reference coordinates) or `fixed`. A
displacement is a constant `value: [dx, dy(, dz)]` or an `expression`, one
string per component in x, y, z (the reference coordinates) and t, with
optional named `parameters` (`config/expression`: `+ - * / ^`, sin, cos, tan,
their inverses and hyperbolic forms, exp, log, sqrt, abs, min, max, pow,
atan2, pi, e). The `mesh_motion` section picks the model (`linear_elastic`,
the only one today), its Poisson ratio and stiffening exponent, the order
(default `discretization.order_u`; a curved geometry keeps its own nodes'
order), the CG settings, and `steps`: the moves are at t_k = k t_final / steps
(`time.t_final`), so an expression with t ramps the deformation in and one
without it is a single static deformation. Every move solves from the
reference mesh (total displacement), so the result at t_final does not depend
on the number of steps.

```yaml
equation: mesh_motion
mesh: {dim: 2, elements: [16, 8], lengths: [2.0, 1.0], periodic: [false, false]}
mesh_motion: {order: 4, steps: 4}
time: {t_final: 1.0}
boundary_conditions:
  - select: [ymin]
    type: displacement
    expression: ["0", "A * t * 4 * x * (L - x) / L^2"]
    parameters: {A: 0.3, L: 2.0}
  - {select: [xmin, xmax, ymax], type: fixed}
output: {enabled: true, history: true, path: mesh_deform_2d, name: mesh_deform_2d}
```

Output: ParaView (`output.enabled`; cycle 0 is the reference mesh, then every
`output.interval` moves and the last) with the fields `displacement` and
`det_F` on the deformed high-order mesh; `<name>_history.csv` per move
(`output.history`: CG iterations, det F range, the smallest element Jacobian
ratio, the largest displacement, the boundary error); the summary's
`results`: `det_f_min`, `det_f_max`, `det_f_min_over_moves`,
`jacobian_ratio_min` (min det J / max det J over an element, smallest over
the mesh; the reference mesh's is under `mesh`), `max_displacement` and
`boundary_error` (the largest |x - X - g(X, t)| over the displaced boundaries'
nodes: roundoff when the mesh reproduces the data). A move that inverts the
mesh (min det F <= 0) stops the run with status `diverged`. Example decks:
`cases/mesh_deform_2d.yaml` (a rectangle's bottom pushed up by a parabola)
and `cases/mesh_deform_cube.yaml` (a cube's bottom face pushed up by a sin^2
bump). The `displacement` expressions work in `equation: linear_elasticity`
decks too. Not yet: free or sliding mesh boundaries, AMR or periodic meshes
(refused), velocity boundary groups given by expressions.

## 4. Verification

Fast tier, `mesh_motion_test`, np 1/2/4 (pins its own orders):

- M1: a translation of every boundary is reproduced exactly with stiffening
  on (2D Q3 and 3D Q2 on distorted meshes, which get non-uniform factors):
  the nodes move by the translation to 1e-9 and det F = 1; a uniform mesh gets
  factor 1 everywhere.
- M2: an affine motion d = A X + b of every boundary is reproduced exactly
  without stiffening, on the curved DFG mesh (its own order 3) and a 3D box:
  det F = det(I + A) at every sample point, and the moved volume is det(I + A)
  times the reference one (which also checks that the geometry caches were
  refreshed).
- M3: the DFG cylinder translated by (0.01, 0.03) with only the body named
  (inflow, outflow and walls held by default): the body's nodes lie on the
  moved circle to 1e-12, the other boundaries' nodes did not move, det F > 0,
  and with chi = 1 the largest |det F - 1| over the O-grid ring's elements is
  under 0.75 of chi = 0's (measured 0.159 against 0.247, and min det F 0.79
  against 0.77); moving back to rest returns every node.
- M4: the mesh velocity is exact for a motion quadratic in time at BDF2 and
  cubic at BDF3, on variable steps (0, 0.1, 0.25, 0.33).
- M5: a model that is not elasticity (a test model that interpolates a given
  field) plugs in through the factory, the nodes follow it exactly, and the
  boundaries nobody named reach the model as fixed.

`mesh_motion_case_test` (decks with the defaults, a small order): C1 the 2D
rectangle with a ramped parabola, C2 the 3D cube with a sin^2 bump (the
displaced boundary's nodes at X + g(X, t) to roundoff, the held faces' nodes
unmoved, det F > 0, the peak node moved by the peak), C3 a constant
displacement of every boundary is a translation (det F = 1).
`expression_test` E1-E4 pins the expression parser. `run_case_mesh_deform`
runs `cases/mesh_deform_2d.yaml` end to end.

## 5. Next

0. Run the decks and tests on a GPU (human 2026-10-21: "next step is to have
   it tested on GPU"); the expression evaluation and the boundary-error check
   are host loops, at setup and once per move.
1. The ALE fluid: started 2026-10-21 (IMEX, library only, a full rebuild
   after each move; [ale_fluid.md](ale_fluid.md)); next there the deck
   section, the in-place refresh and the validation.
2. A hyperelastic mesh-motion model (shares the large-deformation solid of
   the Turek-Hron step).
3. Sliding boundaries; an incremental (current-configuration) mode if large
   rotations need it; a warm-started solve (the previous displacement as the
   CG guess, with a tolerance that does not depend on it).
