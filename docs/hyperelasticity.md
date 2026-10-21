# Hyperelasticity

Large-deformation solids in the total Lagrangian form, St. Venant-Kirchhoff (SVK) first.
The design and its decisions are in [design/hyperelasticity_spec.md](design/hyperelasticity_spec.md).
Human 2026-10-21: "start specking on adding hyperelasticity and at venant Kirkhoff
model", then "go with your recommendations, commit the spec and start M1", then "after
this is a good time to also start optimizing memory and then do performance engineering".

**State (2026-10-21):** milestone 1 (the material and its operators) and a memory step.
There is no solver, deck or boundary condition yet: that is milestone 2 (Newton, static,
Turek-Hron CSM1/CSM2).

## 1. What exists

- `src/solid/hyperelastic_material.hpp`: `StVenantKirchhoff`, three `MFEM_HOST_DEVICE`
  functions templated on the dimension. `Stress(F) -> S, P = F S`,
  `Tangent(F, S, dF) -> dP = dF S + F dS` with `dS = lambda tr(dE) I + 2 mu dE` and
  `dE = sym(F^T dF)`, and `Energy(F) -> W`. Matrices are column-major. A new material is
  a new struct with the same three functions, not new kernels (spec D2, D4).
- `src/solid/hyperelastic_operator.{hpp,cpp}`: `HyperelasticOperator`.
  - `Mult(u)` is the internal force R(u) = int P(F(u)) : grad_X v dX on true dofs, with
    F = I + grad_X u.
  - `GetGradient(u)` is the tangent K_T(u).
  - `Energy(u)` is the strain energy.
  - Each apply is one fused kernel per element. It forms the reference gradients of all
    components by sum factorization (MFEM's `kernels::internal` helpers, as in
    `GradDivIntegrator`), applies the law at every point, and contracts with the
    transposed gradient. It does no element matrices and makes no host round trips.
  - Requirements: quads and hexes, orders 1 to 7, and the linear solver's rule (Gauss,
    order 2k + 1, k + 1 points per direction). lambda and mu are per element (the mesh
    motion's stiffening).
  - No boundary conditions: the solver constrains the operator.
- `TangentStorage` sets what the tangent keeps from its linearization point:
  - `Deformation` (the default): F at every point, with S recomputed in each apply.
  - `StressAndDeformation`: F and S.
  - `Displacement`: the displacement's element vector only, with F and S recomputed by
    one more gradient pass.
- The operator can borrow its work vectors from an `ElementScratch`. It keeps no L-vectors
  where the prolongation is the identity (one rank, no hanging nodes).
- `bench/bench_hyperelastic.cpp` reports, per order and storage: the bytes per vector dof,
  the device memory added, and the residual, linearization and tangent times. MFEM's
  linear `ElasticityIntegrator` in partial assembly is the reference.

**Changed from the spec.** Section 3 of the spec planned to keep F and S (15 numbers per
point, the stress symmetric). The default keeps F only (9 numbers): on an H100 it is
smaller and faster (section 3 below).

## 2. Tests

`test/hyperelasticity_test.cpp` (`hyperelasticity_test`):

- **H1:** the material at a point. It is checked against the closed form and an
  independent SVK `HyperelasticModel`. The tangent is checked against central differences
  of the stress. It also checks frame indifference, zero stress under a rigid rotation, and
  the small-strain limit.
- **H2:** residual and tangent against MFEM's `HyperelasticNLFIntegrator`, which the test
  runs with its own SVK model (2D Q3 and Q6, 3D Q2; 3D Q6 the residual only), to 1e-12
  relative.
- **H3:** Taylor test of the residual against the tangent, and of the energy against the
  residual.
- **HD:** applies are bitwise repeatable.
- **HM:** the three storages give the same tangent to 1e-13 and order their memory as
  expected; borrowed buffers leave results bitwise unchanged.

Passed on the CPU at np 1, 2 and 4 (RM, jobs 49563606 and 49563998) and on H100s at np 1
and 2 (jobs 49563894 and 49564199).

## 3. Memory and speed

The benchmark used a 3D unit cube of hexes at about 2e7 vector dofs per H100 (code at
856a03d, job 49563894, np 1). It reports bytes per vector dof that the operator owns, and
the same with work vectors borrowed from an `ElementScratch`. Rates are million vector
dofs per second.

| p | storage | B/dof | B/dof borrowed | residual Mdof/s | tangent Mdof/s | MFEM linear PA Mdof/s |
|---|---------|------:|------:|------:|------:|------:|
| 4 | F (default) | 128 | 98 | 2655 | 2580 | 1085 |
| 5 | F (default) | 113 | 86 | 3221 | 3262 | 557 |
| 6 | F,S | 142 | 117 | 3827 | 3502 | cannot launch |
| 6 | F (default) | 104 | 79 | 3826 | 3698 | cannot launch |
| 6 | u | 79 | 54 | 3826 | 2798 | cannot launch |
| 7 | F (default) | 98 | 75 | 5882 | 5585 | cannot launch |

- At Q6 (7 points per direction, 0.53 points per vector dof), the bytes go to:
  - the geometry, J^-1 and w det J (10 numbers per point): about 42 B/dof;
  - F: 38 B/dof (F,S: 76; u: 13);
  - the operator's own E-vectors, in and out: 25 B/dof (borrowed: 0).

  At np 2 the operator also owns two L-vectors (16 B/dof). The first construction also
  computes the mesh's geometric factors at the rule (about 38 B/dof at Q6). The mesh keeps
  them until the rule's owner clears them.
- The default `Deformation` has half the stored data of `StressAndDeformation` and its
  tangent is 5 % faster on the H100: there is less to read, and recomputing S is cheap.
  On the CPU the two are equally fast. `Displacement` saves another 25 B/dof but costs
  30 % in the tangent.
- MFEM's linear elasticity PA does strictly less work (no material law) and is 2.4x to
  5.9x slower at Q3-Q5. In 3D at Q6 it cannot launch: it asks for dim (p + 1)^dim threads
  per block ([status.md](status.md)). These kernels use (p + 1)^2.
- Low orders underuse the GPU. A block holds one element of (k + 1)^2 threads, so
  Q2 (1.2 Gdof/s) is slower than MFEM's linear PA (1.85).
- Two H100s (np 2, ~4e7 dofs, job 49564199): at Q6, 3.49 Gdof/s per GPU in the tangent
  and 3.57 in the residual, i.e. 94 % and 93 % of one GPU.
- One CPU core (RM, np 1, ~3e5 dofs): the Q6 tangent 11 Mdof/s against MFEM's linear PA
  at 0.7.

## 4. Notes for developers

- **nvcc and `if constexpr`.** nvcc rejects an extended `__host__ __device__` lambda that
  first captures an outer variable inside an `if constexpr`; g++ accepts it. The kernels
  therefore capture such variables at their top with `MFEM_CONTRACT_VAR` and define their
  constants (D2) inside the lambda. To check a CUDA translation unit without waiting for
  a GPU node, rerun the GPU build's nvcc command for that file on a CPU node with
  `-ccbin` set to the host g++. The GPU environment's own compiler wrappers cannot run on
  CPU nodes.
- MFEM's `QuadratureInterpolator::MultTranspose` is unimplemented (it aborts). The
  operator does its own transposed gradients.

## 5. Next

- **Performance engineering:** Nsight Systems and Compute on the Q6 kernels (queued). Then
  use the measurements to decide on several elements per block at low orders, the register
  pressure, and the geometry computed on the fly (42 of the 104 B/dof).
- **Milestone 2:** Newton with load steps, the fixed reference LOR-AMG preconditioner, the
  deck (`equation: hyperelasticity`), CSM1 and CSM2.
