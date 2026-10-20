# Mesh motion

The mesh-motion half of an ALE (arbitrary Lagrangian-Eulerian) formulation, the
second step toward fluid-structure interaction after the linear elasticity
solver ([elasticity.md](elasticity.md)). Human 2026-10-20: "add the capability
to deform a mesh using the linear elasticity solver ... make it general that a
different type of solver (hyper elasticity etc) can be used". It is a library
feature today: nothing in the flow solver moves its mesh yet (that is the ALE
fluid, the next step), and there are no deck keys.

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
displacement along flat walls), deck keys, the Python bindings and a GPU run
are not done. Memory: the reference copy (topology and nodes), the node
history (`history` node vectors) and the elasticity solver (its LOR-AMG
dominates, elasticity.md section 3).

## 3. Verification

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
  under half of chi = 0's; moving back to rest returns every node.
- M4: the mesh velocity is exact for a motion quadratic in time at BDF2 and
  cubic at BDF3, on variable steps (0, 0.1, 0.25, 0.33).
- M5: a model that is not elasticity (a test model that interpolates a given
  field) plugs in through the factory, the nodes follow it exactly, and the
  boundaries nobody named reach the model as fixed.

## 4. Next

1. The ALE fluid: the convective velocity u - w, the mesh velocity's
   geometric conservation law with BDF, the flow operators refreshed after
   each move, a `mesh_motion:` deck section; verified on a moving-domain MMS.
2. A hyperelastic mesh-motion model (shares the large-deformation solid of
   the Turek-Hron step).
3. Sliding boundaries; an incremental (current-configuration) mode if large
   rotations need it; a warm-started solve (the previous displacement as the
   CG guess, with a tolerance that does not depend on it).
