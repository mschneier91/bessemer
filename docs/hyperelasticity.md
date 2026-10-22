# Hyperelasticity

Large-deformation solids in the total Lagrangian form, St. Venant-Kirchhoff (SVK) first.
The design and its decisions are in [design/hyperelasticity_spec.md](design/hyperelasticity_spec.md).
Human 2026-10-21: "start specking on adding hyperelasticity and at venant Kirkhoff
model", then "go with your recommendations, commit the spec and start M1", then "after
this is a good time to also start optimizing memory and then do performance engineering".

**State (2026-10-22):** milestone 1 (the material and its operators), a memory step, and
a performance step in five rounds (section 4):

1. the 3D kernels on a device by lines;
2. the gather and scatter fused and the derivatives collocated;
3. the shared dofs exchanged directly instead of through MFEM's prolongation;
4. 7 blocks per SM;
5. the exchange overlapped with the interior elements' work.

Milestone 2 has begun with the static solver, `Hyperelasticity`: Newton with load
steps, boundary conditions and loads (sections 1, 2 and 4). Its deck
(`equation: hyperelasticity`) and Turek-Hron CSM1/CSM2 come next (section 5).

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
- `src/solid/hyperelasticity.{hpp,cpp}`: `Hyperelasticity`, the static solver
  (milestone 2). Its API is `LinearElasticity`'s static part: `AddFixed`,
  `AddDisplacement`, `AddTraction` (P N, per reference area) and `SetBodyForce` (per
  reference volume). The loads are dead loads. `Solve(t)` runs from the current
  displacement.
  - Newton on the operator's residual and tangent, `options.load_steps` ramping the loads
    and the Dirichlet data together from zero. An optional backtracking line search
    halves the step until the residual norm drops (Armijo, 1e-4).
  - Each correction is solved by CG, preconditioned by one BoomerAMG V-cycle on the LOR
    linear elasticity of the reference configuration (the tangent at u = 0), built once
    (spec D5). It is a copy of `LinearElasticity`'s; sharing it would need that class's
    bitwise check (spec section 5).
  - A load step converges when its residual (true dofs, Dirichlet rows out) falls below
    `max(newton_atol, newton_rtol r_ref)`. Here r_ref is the largest of the step's first
    residual, its load's norm, and the right-hand side its Dirichlet increment makes
    (section 4).
  - Reported: Newton iterations (total and the most in one step), CG iterations, the
    residual history, strain energy, external work and the largest displacement.
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

The static solver (milestone 2):

- **H4:** a homogeneous finite deformation (stretch and shear, det F != 1). The
  displacement is prescribed on the faces through the origin, and the exact traction
  P(F) N acts on the others. It is reproduced to 1e-10 in 2D and 3D in at most 8 Newton
  iterations per load step: the full load at once, and in three load steps with the line
  search. The residuals fall quadratically, e.g. in 2D 1.3e-2, 3.0e-4, 1.6e-7, 5.4e-14.
- **H5:** a rigid rotation by 90 and 180 degrees, prescribed on the whole boundary in 8
  solves. The interior rotates to 1e-10 with |strain energy| < 1e-16, where
  `LinearElasticity` on the same data stores more than 0.1.
- **H6:** a 4 x 1 cantilever under its weight at loads 1e-4 and 5e-5. The difference to
  `LinearElasticity` falls as the load squared (ratio 4.0015).

Passed on the CPU at np 1, 2 and 4 and on H100s at np 1 and 2 at every step; for the
current code, jobs JOB_CPU (CPU) and JOB_GPU (GPU).

## 3. Memory and speed

The benchmark used a 3D unit cube of hexes at about 2e7 vector dofs per H100, with the
kernels of round 5 (job 49576431, np 1; round 4: job 49575048; round 2: job 49572810;
round 1: job 49569573; the first kernels: job 49563894). It reports bytes per vector dof that the operator owns, and
the same with work vectors borrowed from an `ElementScratch`. Rates are million vector
dofs per second.

| p | storage | B/dof | B/dof borrowed | residual Mdof/s | tangent Mdof/s | MFEM linear PA Mdof/s |
|---|---------|------:|------:|------:|------:|------:|
| 4 | F (default) | 117 | 105 | 18308 | 14751 | 1078 |
| 5 | F (default) | 102 | 93 | 16197 | 13891 | 553 |
| 6 | F,S | 130 | 123 | 20212 | 15552 | cannot launch |
| 6 | F (default) | 93 | 85 | 20202 | 17194 | cannot launch |
| 6 | u | 63 | 56 | 20234 | 12007 | cannot launch |
| 7 | F (default) | 86 | 80 | 21478 | 18834 | cannot launch |

Q2 and Q3 were last measured in round 2 (job 49572810): 4500 and 9586 Mdof/s in the
tangent, against MFEM's 1846 and 862.

- At Q6 (7 points per direction, 0.53 points per vector dof), the bytes go to:
  - the geometry, J^-1 and w det J (10 numbers per point): about 42 B/dof;
  - F: 38 B/dof (F,S: 76; u, its L-vector: 8);
  - the slots of the dofs elements share: 8 B/dof (borrowed: 0);
  - the fused scatter's maps (each E-dof's destination, the shared dofs' lists): 3 B/dof;
  - the exchange's local-dof sources: 1.4 B/dof (at np > 1 also its gather map to true
    dofs and ghosts, 2 B/dof; at np 1 that map is MFEM's).

  Before round 2 the operator's E-vectors took 25 B/dof instead of the slots and maps, so
  it owns 11 B/dof less. With borrowed buffers it holds 6 B/dof more, because the maps
  cannot be borrowed.

  At np 2 it owns 94 B/dof (job 49576431): no L-vectors since round 3 (before: 16 B/dof
  more). The first construction also computes the mesh's geometric factors at the
  rule (about 38 B/dof at Q6). The mesh keeps them until the rule's owner clears them.
- The default `Deformation` has half the stored data of `StressAndDeformation` and its
  tangent is 10 % faster on the H100 (1.14 against 1.26 ms at Q6): there is less to read,
  and recomputing S is cheap. On the CPU the two are equally fast. `Displacement` saves
  another 30 B/dof but costs 43 % in the tangent at Q6 and 2x at Q7, where its 18 stacks
  of shared memory leave 2 blocks per SM.
- MFEM's linear elasticity PA does strictly less work (no material law) and is 2.4x
  slower at Q2 and 11x to 25x slower at Q3-Q5. In 3D at Q6 it cannot launch: it asks for
  dim (p + 1)^dim threads per block ([status.md](status.md)). These kernels use
  (p + 1)^2.
- Throughput rises with the order: an element is a block of (k + 1)^2 threads, so Q2 (9
  threads) uses the GPU least, 4.5 Gdof/s (round 2) against 17.2 at Q6.
- Two H100s (np 2, ~4e7 dofs, job 49576431): at Q6, 15.8 Gdof/s per GPU in the tangent
  and 18.2 in the residual, 92 % and 90 % of one GPU (Q4-Q7: 89-93 %).
  - Earlier rounds reached 87 % and 84 % (round 4) and 79 % and 76 % (round 2, through
    MFEM's prolongation). MFEM's P u and P^T y alone take 0.165 ms each at Q6 there.
  - About 0.1 ms per apply is left: the exchange's messages (~1.4 MB each way) and the
    syncs around them, less what the overlap hides.
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
  - Measured together with round 4 (section 3): at np 2, Q6, 87 % of one GPU in the
    tangent and 84 % in the residual, against 79 % and 76 %.
  - Gotcha: filling a `Vector` through sub-vector aliases on the device does not mark
    the base's device copy valid. The base's next `Read()` brings the stale host data
    back (job 49574179, the `Displacement` storage). Write through the base's own
    `Write()` pointer instead.

- **Round 4: 7 blocks per SM.** The stacks are now unpadded, N^3 each. A thread touches
  its own column, and the rows that the transposed derivatives read go to lanes in
  broadcast. The two scratch regions keep a padded z stride and now also carry the y
  stage's planes.
  - At Q6: 31 KB per block for 9 stacks instead of 37 KB, so 7 blocks per SM instead of
    6 (`Displacement`'s 18 stacks: 4 instead of 3).
  - A Q6 apply at np 1 (job 49575048): tangent 1.17 -> 1.13 ms (F), 1.41 -> 1.25 (F,S),
    1.92 -> 1.63 (`Displacement`); residual 0.97 -> 0.95.
  - Q4 gained 6-7 %, and Q5 and Q7 are unchanged. `Displacement` is 8 % slower at Q7
    (not investigated).

- **Round 5: the exchange overlapped.**
  - On a rank with ghosts, the elements that read none run while the gather's messages
    travel, and the others after.
  - On the way back, the ghosts' slots are summed first and sent, and the other slots are
    summed while those sums travel.
  - The slotted dofs are ordered ghosts first. Every sum keeps its order, so results do
    not change. A rank without ghosts runs one launch as before; at np 2 that is rank 0,
    which owns the interface.
  - At np 2, Q6 (job 49576431): residual 1.20 -> 1.15 ms and tangent 1.40 -> 1.33 ms per
    apply, 90 % and 92 % of one GPU.
  - At np 1 the exchange's gather map is MFEM's own (2 B/dof less).

- **The Dirichlet increment goes through the tangent.** At first, each load step moved
  the Dirichlet dofs to their new values and left the rest in place. That strained a
  layer of boundary elements, and Newton started far away: in H4's 2D case the first
  step raised the residual from 1.4 to 880, and it took 11 iterations. Now a step's first
  Newton correction carries the increment: it is prescribed on the Dirichlet dofs
  (`ConstrainedOperator::EliminateRHS`), and the free dofs follow through the tangent,
  as in deal.II's step-44. H4 then needs 6 iterations from 0.51. The line search skips
  that first correction, whose residual has new data.
- **Newton's tolerance has a roundoff floor.** On H6's small cantilever (Q3, 8 x 2
  elements, loads near 1e-4) the residual stopped at about 3e-11 of the load. Element
  contributions that are much larger than the load cancel at every node. A slender beam
  and a finer mesh raise the floor, so the default `newton_rtol` is 1e-8, not the 1e-10
  the spec's sample deck had. Quadratic convergence leaves little behind it: the
  displacement error is about the tangent's inverse times the residual.

## 5. Next

- **More performance.**
  - Measure the scaling beyond two GPUs, at np 4 and 8 on one node. Tests stay at np 1
    and 2 on GPUs, but benchmarks at 4-8 H100s are allowed.
  - What is left at np 2 is about 0.1 ms per Q6 apply. Intra-node GPU MPI goes over
    InfiniBand (rc_mlx5) by design, not CUDA IPC (change 42).
  - Geometry on the fly: **shelved** (human 2026-10-21). The operator's ~93 B/dof are
    under 5 % of the LOR-AMG's estimated 2-3 kB/dof (elasticity.md section 3), so it
    would save under 2 % of the solver's memory, at a cost in speed.
- **Milestone 2, the rest:**
  - the deck (`equation: hyperelasticity`, `solid.model`, the Newton keys) through
    `ElasticityCase`, and H8;
  - the displacement at a point (`DisplacementAt`) for point A;
  - Turek-Hron CSM1 and CSM2, the beam mesh, and V1 and V2 static. The reference values
    must be checked against Turek and Hron (2006) before they become thresholds.
