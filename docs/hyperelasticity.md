# Hyperelasticity

Large-deformation solids in the total Lagrangian form, St. Venant-Kirchhoff (SVK) first.
The design and its decisions are in [design/hyperelasticity_spec.md](design/hyperelasticity_spec.md).
Human 2026-10-21: "start specking on adding hyperelasticity and at venant Kirkhoff
model", then "go with your recommendations, commit the spec and start M1", then "after
this is a good time to also start optimizing memory and then do performance engineering".

**State (2026-10-21):** milestone 1 (the material and its operators), a memory step, and
a performance step (the 3D kernels on a device by lines, section 4). There is no solver,
deck or boundary condition yet: that is milestone 2 (Newton, static, Turek-Hron
CSM1/CSM2).

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

The benchmark used a 3D unit cube of hexes at about 2e7 vector dofs per H100, with the 3D
kernels by lines (job 49569573, np 1; the first kernels: job 49563894). It reports bytes per vector dof that the operator owns, and
the same with work vectors borrowed from an `ElementScratch`. Rates are million vector
dofs per second.

| p | storage | B/dof | B/dof borrowed | residual Mdof/s | tangent Mdof/s | MFEM linear PA Mdof/s |
|---|---------|------:|------:|------:|------:|------:|
| 2 | F (default) | 222 | 169 | 5400 | 4218 | 1846 |
| 3 | F (default) | 156 | 118 | 11594 | 9252 | 862 |
| 4 | F (default) | 128 | 98 | 14074 | 11753 | 1078 |
| 5 | F (default) | 113 | 86 | 13816 | 11958 | 553 |
| 6 | F,S | 142 | 117 | 14543 | 11990 | cannot launch |
| 6 | F (default) | 104 | 79 | 14482 | 14087 | cannot launch |
| 6 | u | 79 | 54 | 14535 | 8923 | cannot launch |
| 7 | F (default) | 98 | 75 | 16549 | 14886 | cannot launch |

- At Q6 (7 points per direction, 0.53 points per vector dof), the bytes go to:
  - the geometry, J^-1 and w det J (10 numbers per point): about 42 B/dof;
  - F: 38 B/dof (F,S: 76; u: 13);
  - the operator's own E-vectors, in and out: 25 B/dof (borrowed: 0).

  At np 2 the operator also owns two L-vectors (16 B/dof). The first construction also
  computes the mesh's geometric factors at the rule (about 38 B/dof at Q6). The mesh keeps
  them until the rule's owner clears them.
- The default `Deformation` has half the stored data of `StressAndDeformation` and its
  tangent is 15 % faster on the H100 (1.39 against 1.64 ms at Q6): there is less to
  read, and recomputing S is cheap. On the CPU the two are equally fast. `Displacement`
  saves another 25 B/dof but costs 58 % in the tangent at Q6, and 2.7x at Q7, where its
  18 stacks of shared memory leave 2 blocks per SM.
- MFEM's linear elasticity PA does strictly less work (no material law) and is 2.3x
  slower at Q2 and 11x to 22x slower at Q3-Q5. In 3D at Q6 it cannot launch: it asks for
  dim (p + 1)^dim threads per block ([status.md](status.md)). These kernels use
  (p + 1)^2.
- Throughput rises with the order: an element is a block of (k + 1)^2 threads, so Q2 (9
  threads) uses the GPU least, 4.2 Gdof/s against 14 at Q6.
- Two H100s (np 2, ~4e7 dofs, job 49569724): at Q6, 11.6 Gdof/s per GPU in the tangent
  and 11.8 in the residual, 82 % of one GPU (94 % with the first kernels: the parallel
  prolongation and its exchange did not get faster, so presumably their share grew; not
  profiled).
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
- **The 3D kernels on a device, by lines.** Nsight Compute on the first kernels at Q6
  (job 49569116) found MFEM's register tensors in local memory: the reference gradients
  of the three components (9 N values per thread) and as many fluxes. That memory
  overflowed L1 and L2 to DRAM. The tangent kernel moved 8.7 GB per apply for 2.3 GB of
  data, at 25 % occupancy, in 4.97 ms. The 3D kernels on a device now work by lines, as
  the momentum apply of change 56 (`operators/lean_vector_diffusion.cpp`):
  - an N x N block per element;
  - in each stage of the sum factorization, a thread per line of outputs;
  - the 1D matrices as kernel parameters;
  - the gradients, then in place the fluxes, in shared memory: 9 stacks of N planes,
    37 KB per block at Q6, 18 stacks for `Displacement`.

  No instance uses local memory or a stack. The Q6 tangent kernel moves exactly its data
  (2.0 GB) in 1.02 ms, at 31 % of the FP64 peak and 60 % of the DRAM bandwidth (job
  49569573). A Q6 apply went from 5.13 to 1.35 ms (residual) and from 5.31 to 1.39 ms
  (tangent), 3.8x; the other orders gained 2.7x to 5.7x. 2D and the CPU keep the first
  kernels, so CPU results are unchanged. The output E-vector is written, not added to, so
  these applies skip zeroing it.

## 5. Next

- **More performance.** At Q6 a tangent apply is now 0.92 ms of kernel and 0.47 ms of
  the element restriction's gather and scatter (nsys, job 49569573). Next in line:
  - read the input through the restriction's map inside the kernel. That removes the
    gather pass and the input E-vector (13 B/dof). The scatter stays a separate pass,
    because atomics would make the sums order-dependent.
  - compute the geometry on the fly: J^-1 is 40 % of the tangent's DRAM traffic and
    42 B/dof.
  - fix the shared-memory strides at N = 7. They were designed for rows of 8 lanes, and
    ncu counts 30M bank conflicts per tangent apply.
- **Milestone 2:** Newton with load steps, the fixed reference LOR-AMG preconditioner, the
  deck (`equation: hyperelasticity`), CSM1 and CSM2.
