# Hyperelasticity

Large-deformation solids in the total Lagrangian form, St. Venant-Kirchhoff (SVK) first.
The design and its decisions are in [design/hyperelasticity_spec.md](design/hyperelasticity_spec.md).
Human 2026-10-21: "start specking on adding hyperelasticity and at venant Kirkhoff
model", then "go with your recommendations, commit the spec and start M1", then "after
this is a good time to also start optimizing memory and then do performance engineering".

**State (2026-10-21):** milestone 1 (the material and its operators), a memory step, and
a performance step in two rounds: the 3D kernels on a device by lines, then with the
gather and scatter fused and the derivatives collocated (section 4). There is no solver,
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
  - `Displacement`: the displacement only (its element vector; in 3D on a device what the
    kernels read: its true dofs and ghosts, or its L-vector), with F and S recomputed by
    one more gradient pass.
- The operator can borrow its work vectors from an `ElementScratch`. It keeps no L-vectors
  where the prolongation is the identity (one rank, no hanging nodes). In 3D on a device
  it keeps no E-vectors (the gather and scatter are fused into the kernels), and on a
  conforming space no L-vectors at any rank count (`TrueDofExchange`, section 4).
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

`test/true_dof_exchange_test.cpp` (`true_dof_exchange_test`), for vector and scalar
spaces at orders 1 to 3:

- **X1:** the exchange's gather is exactly MFEM's P.
- **X2:** its reduce is P^T to roundoff and writes every true dof.
- **X3:** two reduces agree bitwise.

np 4 on the CPU covers dofs that three or more ranks share.

Passed on the CPU at np 1, 2 and 4 and on H100s at np 1 and 2 at every step; for round 3
(section 4), job 49575822 on the CPU and, on H100s, job 49575048 together with round 4.

## 3. Memory and speed

The benchmark used a 3D unit cube of hexes at about 2e7 vector dofs per H100, with the 3D
kernels of round 2 (job 49572810, np 1; round 1: job 49569573; the first kernels: job
49563894). It reports bytes per vector dof that the operator owns, and the same with work
vectors borrowed from an `ElementScratch`. Rates are million vector dofs per second.

| p | storage | B/dof | B/dof borrowed | residual Mdof/s | tangent Mdof/s | MFEM linear PA Mdof/s |
|---|---------|------:|------:|------:|------:|------:|
| 2 | F (default) | 206 | 180 | 6301 | 4500 | 1846 |
| 3 | F (default) | 142 | 126 | 13912 | 9586 | 862 |
| 4 | F (default) | 116 | 104 | 17149 | 13672 | 1078 |
| 5 | F (default) | 101 | 91 | 16253 | 13994 | 553 |
| 6 | F,S | 129 | 121 | 20305 | 13903 | cannot launch |
| 6 | F (default) | 91 | 84 | 20278 | 16706 | cannot launch |
| 6 | u | 62 | 54 | 20307 | 10233 | cannot launch |
| 7 | F (default) | 85 | 79 | 22111 | 18883 | cannot launch |

- At Q6 (7 points per direction, 0.53 points per vector dof), the bytes go to:
  - the geometry, J^-1 and w det J (10 numbers per point): about 42 B/dof;
  - F: 38 B/dof (F,S: 76; u, its L-vector: 8);
  - the slots of the dofs elements share: 8 B/dof (borrowed: 0);
  - the fused scatter's maps (each E-dof's destination, the shared dofs' lists): 3 B/dof.

  Before round 2 the operator's E-vectors took 25 B/dof instead of the slots and maps
  (11), so it owns 13 B/dof less; with borrowed buffers it holds 4.5 B/dof more (the
  maps cannot be borrowed).

  At np 2 the operator also owns two L-vectors (16 B/dof). The first construction also
  computes the mesh's geometric factors at the rule (about 38 B/dof at Q6). The mesh keeps
  them until the rule's owner clears them.
- The default `Deformation` has half the stored data of `StressAndDeformation` and its
  tangent is 17 % faster on the H100 (1.17 against 1.41 ms at Q6): there is less to
  read, and recomputing S is cheap. On the CPU the two are equally fast. `Displacement`
  saves another 30 B/dof but costs 63 % in the tangent at Q6 and 82 % at Q7, where its
  18 stacks of shared memory leave 2 blocks per SM.
- MFEM's linear elasticity PA does strictly less work (no material law) and is 2.4x
  slower at Q2 and 11x to 25x slower at Q3-Q5. In 3D at Q6 it cannot launch: it asks for
  dim (p + 1)^dim threads per block ([status.md](status.md)). These kernels use
  (p + 1)^2.
- Throughput rises with the order: an element is a block of (k + 1)^2 threads, so Q2 (9
  threads) uses the GPU least, 4.5 Gdof/s against 16.7 at Q6.
- Two H100s (np 2, ~4e7 dofs, job 49573345): at Q6, 13.3 Gdof/s per GPU in the tangent
  and 15.4 in the residual, 79 % and 76 % of one GPU (94 % with the first kernels: the
  parallel prolongation and its exchange did not get faster, so presumably their share
  grows with each kernel gain; not profiled).
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
  kernels, so CPU results are unchanged.
- **Round 2: the gather and scatter fused, the derivatives collocated** (human: "Keep
  optimizing"). At Q6 the element restriction's gather and scatter were 0.47 of the
  tangent's 1.39 ms. The kernels now read the L-vector through the restriction's gather
  map. They write a dof one element holds straight into the output L-vector, and a
  shared dof's contribution into a slot. `ScatterSlots` then sums each shared dof's slots
  in element order: deterministic, with no atomics and no E-vectors.
  - The restriction's 0.47 ms became 0.16 ms. But the index loads cost the kernels 12-22
    registers (nvcc hoists them wherever the source reads them, job 49573235), and the Q6
    tangent fell from 6 blocks per SM to 4: 1.30 ms (job 49571253).
  - The sum factorization is therefore collocated: B along each direction to the values
    at the points, then D = G B^-1 there. That is 6 N^4 multiply-adds per component and
    pass instead of 8 N^4, and it took the Q6 kernels to 128-140 registers and back to 6
    blocks per SM; shared memory (37 KB per block) is now the limit.
  - A Q6 apply: residual 1.35 -> 0.97 ms, tangent 1.39 -> 1.17 ms (job 49572810).
    Capping the registers instead (`__launch_bounds__` with 5 or 6 blocks) gave the same
    Q6 times with small spills, so the gain is mostly occupancy. Collocation costs the
    Q2-Q3 tangent 6-10 % against round 2 without it (one more sync per component on small
    blocks); it stays ahead of round 1 there.
  - The outputs are written, not added to, so these applies zero nothing.
- **Round 3: the shared dofs exchanged directly.** At np 2, MFEM's device prolongation
  cost 0.15 ms (P u) and 0.17 ms (P^T y) per Q6 apply: all of the np 2 overhead (job
  49573701). Most of it copies every owned dof between the true-dof vector and the
  L-vector (`SetSubArray`, `ExtractSubArray`).
  - On a conforming space the kernels now read and write true dofs directly.
    `TrueDofExchange` (`operators/true_dof_exchange.hpp`) sends the owners' shared dofs
    to the ranks that hold them as ghosts. It returns the ghosts' contributions, which
    the owners add in a fixed order (by neighbor rank, then position): deterministic,
    with no atomics. Only shared dofs move, over GPU-aware MPI.
  - The message order follows MFEM's group topology (each group's master and its group
    number there), with no MFEM internals.
  - A dof is written directly only if one local element holds it and no other rank
    shares it. The others go through slots, and `ScatterSlots` writes their sums into the
    true-dof vector or the ghosts' buffer.
  - Nonconforming spaces keep the L-vector and MFEM's prolongation.
  - Measured together with round 4 (section 3).
  - Gotcha: filling a `Vector` through sub-vector aliases on the device does not mark
    the base's device copy valid. The base's next `Read()` brings the stale host data
    back (job 49574179, the `Displacement` storage). Write through the base's own
    `Write()` pointer instead.

## 5. Next

- **More performance.** At Q6 a tangent apply is now 1.01 ms of kernel and 0.16 ms of
  `ScatterSlots` (nsys, job 49572810). Next in line:
  - compute the geometry on the fly: J^-1 is a third of the tangent's DRAM traffic and
    42 of its 91 B/dof.
  - fit 7 blocks per SM: the shared memory per block (37 KB at Q6) is the limit now.
    Unpadded stacks would take 28 KB, but the padding was designed for rows of 8 lanes
    and fits N = 7 poorly anyway (ncu counts 16M bank conflicts per tangent apply).
  - weak scaling: round 3 (section 4) replaces MFEM's prolongation on conforming spaces;
    what is left of the exchange could overlap with the interior elements' work.
- **Milestone 2:** Newton with load steps, the fixed reference LOR-AMG preconditioner, the
  deck (`equation: hyperelasticity`), CSM1 and CSM2.
