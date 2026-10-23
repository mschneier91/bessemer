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

Milestone 2 (static): the solver `Hyperelasticity` (Newton with load steps, boundary
conditions and dead loads), its deck `equation: hyperelasticity` through `ElasticityCase`,
and the Turek-Hron CSM1 and CSM2 decks, which reproduce the benchmark's point-A
displacements to 1-2e-4 (sections 1-4). Milestone 3 (dynamics): generalized-alpha and
BDF2 on the nonlinear system, the dynamic deck, and CSM3, whose oscillation matches the
benchmark's to 0.4-1.7 % at its time step.

## 1. What exists

- `src/solid/hyperelastic_material.hpp`: `StVenantKirchhoff`, three `MFEM_HOST_DEVICE`
  functions templated on the dimension. `Stress(F) -> S, P = F S`,
  `Tangent(F, S, dF) -> dP = dF S + F dS` with `dS = lambda tr(dE) I + 2 mu dE` and
  `dE = sym(F^T dF)`, and `Energy(F) -> W`. Matrices are column-major. A new material is
  a new struct with the same three functions, not new kernels (spec D2, D4).
  `LinearElastic` is small-strain linear elasticity, `Tangent` only (dP = lambda tr(dF)
  I + mu (dF + dF^T), St. Venant-Kirchhoff's tangent at F = I bit for bit): with it the
  operator is `LinearElasticity`'s stiffness K. It stores no linearization, `Mult` is K u
  from grad u itself, and `AssembleDiagonal` (the tangent's at u = 0, for either
  material) gives its Jacobi smoother.
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
    `max(newton_atol, newton_rtol r_ref)`, or when a Newton correction falls below
    `newton_dx_rtol` (default 1e-10) times the displacement. Here r_ref is the largest of
    the step's first residual, its load's norm, and the right-hand side its Dirichlet
    increment makes. On fine meshes the correction test decides, because the residual
    has a roundoff floor (section 4).
  - Reported: Newton iterations (total and the most in one step), CG iterations, the
    residual and correction histories, strain energy, external work, the largest
    displacement, and the displacement at a point (`DisplacementAt`).
  - Dynamics (milestone 3): rho u_tt + R(u) = f(t) with the consistent mass at rho,
    `Initialize(t0, u0, v0)` and `Step(dt)` as `LinearElasticity`'s. Every scheme is one
    Newton solve per step of c M u^{n+1} + R(theta u^{n+1} + (1 - theta) u^n) = b, the
    static solve's loop with c = 0, theta = 1:
    - generalized-alpha (Chung-Hulbert, `rho_inf`), the internal force and the load at
      t_{n+1-alpha_f} (theta = 1 - alpha_f), the initial acceleration from M a0 = f -
      R(u0);
    - BDF2 on (u, v) after one trapezoidal step (theta = 1); BDF3 is not offered (it is
      not A-stable, elasticity.md section 1).
    The preconditioner is then the LOR-AMG of K_0 + (c / theta) rho M, rebuilt when dt
    changes. Newton starts from u^n.
- **For the FSI coupling** (docs/fsi.md), the API `LinearElasticity` gives `FsiCase`, with
  the same semantics: `AddSupport(attr, k)` (an elastic support, k (u, v) on the faces per
  reference area, in the residual at the scheme's point, the tangent and the LOR-AMG: the
  Robin couplings' solid side), `TrackReaction()` / `Reaction()` (the unconstrained
  residual at the essential dofs, unweighted by the scheme: generalized-alpha's is
  (1 - alpha_f) R^{n+1} + alpha_f R^n there, the trapezoidal start's R^0 + R^1; exact for
  a linear internal force, to first order in the step for SVK's), `ExportState()` /
  `ImportState()` (a subiteration repeats the step) and `TotalIterations()`. The traction
  of `AddTraction` is per reference area (nominal), the load the solid sees in each
  subiteration of a partitioned coupling.
- **The deck**, `equation: hyperelasticity`, static or dynamic (`solid.analysis`, the
  linear deck's time keys). `ElasticityCase` builds either
  solver and applies the deck's boundary groups to it; `run_case` and
  `incns.run_deck` dispatch both equations. The `solid` section takes the linear keys
  (material, order, body force, CG, probe) and, for this equation, `model:
  st_venant_kirchhoff`, `newton_rtol`, `newton_atol`, `newton_dx_rtol`,
  `newton_max_iter`, `load_steps` and `line_search` (docs/deck_reference.md). Plane
  stress, BDF3 and the Jacobi preconditioner are rejected. The summary gains a `newton`
  block; a dynamic run's history CSV a `newton_iterations` column. The linear decks'
  results, the dynamic one's history included, are bitwise unchanged (np 1, 2 and 4).
  - `cases/turek_hron_csm1.yaml`, `cases/turek_hron_csm2.yaml`: the flag of the
    `cylinder_channel` geometry (its solid part) clamped to the cylinder under gravity,
    Q4 on 16 x 2 elements, point A as the probe (section 3).
  - `cases/turek_hron_csm3.yaml`: the same flag released from rest, dt 0.005 to t = 10,
    generalized-alpha at rho_inf 1, the history of point A.
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

- **H7:** generalized-alpha (rho_inf 0.8) and BDF2 are second order in time on a
  separable finite motion, u = 0.1 s(X) (cos 4t + sin 3t / 2) with s quadratic (gradient
  up to 0.14), in 2D and 3D at Q2, whose space holds it exactly. The body force and the
  traction P N are exact, so every error is the integrator's. Halving dt divides the
  errors by 3.99-4.05; at most 3 Newton iterations per step.
- **H8:** a deck through `ElasticityCase`: a 4 x 1 cantilever bent by about 20 % of its
  length. Its tip on the mid-line moves back along the beam, where the linear model's
  stays by symmetry. The summary is written with its `newton` block. Dynamic: a load
  applied suddenly at rest, at rho_inf 1 (the midpoint rule); E(t) = f . u(t) holds to
  5e-6 (not to the solver tolerance as in the linear D5: SVK's energy is not quadratic),
  and the history CSV is written.
- `run_case_csm1` runs `cases/turek_hron_csm1.yaml` through `run_case` (fast tier).

- **H10:** a dynamic cantilever's state exported after three steps; two more steps, then
  the state imported and the two steps again: the same displacement, velocity and
  reaction (bitwise on a host), generalized-alpha and BDF2.
- **H11:** on H4's 2D deformation (displacement on two faces, P N on the others) the
  reaction on the held faces' dofs sums to minus the applied force, to 1e-10.
- **H12:** H6's cantilever with an elastic support on its top face: the difference to
  `LinearElasticity` with the same support falls as the load squared (ratio 3.99998).

`test/verify_hyperelasticity_test.cpp` (verify tier, np 4; settings in
`test/baselines.yaml`, `verification.hyperelasticity`):

- **V1:** a manufactured finite deformation, u_i = 0.05 prod_d sin(pi x_d) (gradient up
  to 0.16, J >= 0.84), held on the whole boundary, under the body force -Div P. The test
  evaluates P on the host, apart from the kernels. Rates k + 1 (L2) and k (H1) at Q2-Q4
  in 2D and Q2-Q3 in 3D, and at the default order. At amplitude 0.1 SVK loses strong
  ellipticity (section 4).
- **V2:** CSM1 and CSM2 through a deck at Q6 on 16 x 2 and 32 x 4 flag meshes: point A
  within 5e-4 (relative, each component) of the benchmark's values and converging.
  CSM3 at Q4 on 16 x 2, dt 0.005, to t = 10: point A's mean and amplitude within 2.5 %
  and its frequency within 1 % of the benchmark's at the same dt (section 3).

Passed on the CPU at np 1, 2 and 4 and on H100s at np 1 and 2 at every step; for the
current code, jobs 49597065 (CPU) and 49592163 (GPU).

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

### Turek-Hron CSM1 and CSM2

The flag (0.35 x 0.02) is clamped to the cylinder and carries gravity, rho_s g = 2000 per
reference volume, with nu_s = 0.4 in plane strain. CSM1 has mu_s = 0.5e6 (E = 1.4e6), and
CSM2 has mu_s = 2.0e6 (E = 5.6e6). The reference values come from the benchmark's own
table (FeatFlow, "CSM tests", level 4+2, 164092 Q2 dofs), read on 2026-10-22. They match
the values the spec quoted. Displacement of point A (0.6, 0.2), x 1e-3, np 4 (jobs
49586969, 49587586):

| Mesh, order | dofs | CSM1 u_x | CSM1 u_y | CSM2 u_x | CSM2 u_y |
|---|---|---|---|---|---|
| 16 x 2, Q2 | 390 | -7.14143 | -65.8629 | -0.465907 | -16.9105 |
| 16 x 2, Q4 (the decks) | 1386 | -7.18279 | -66.0767 | -0.468671 | -16.9670 |
| 16 x 2, Q6 | 2990 | -7.18607 | -66.0939 | -0.468892 | -16.9716 |
| 32 x 4, Q6 | 10550 | -7.18631 | -66.0952 | -0.468908 | -16.9720 |
| 64 x 8, Q6 | 39494 | -7.18634 | -66.0953 | -0.468910 | -16.9720 |
| FeatFlow 4+2 | 164092 | -7.18766 | -66.1023 | -0.468999 | -16.9739 |

- The values converge to within 1.8e-4 (u_x) and 1.1e-4 (u_y) of the reference. That is
  about the spread of the benchmark's own finest levels (CSM1 u_y: -66.0965 at 4+0,
  -66.1029 at 4+3, -66.1009 at 5+0).
- Newton takes 6 iterations (CSM1) and 5 (CSM2) from rest in one load step. The first
  correction, the linear solution, raises CSM1's residual about 10^4-fold; the iterations
  then converge quadratically without a line search.

### Turek-Hron CSM3

CSM1's flag is released from rest under gravity (rho_s = 1000) and oscillates. The
benchmark reports point A as mean +- amplitude [frequency in Hz] at three time steps;
its values move with dt by up to 2.4 % (u_x mean -14.384, -14.645, -14.305 e-3 at dt
0.02, 0.01, 0.005). Ours at np 4, to t = 10, mean and amplitude from the extremes over
t >= 5, the frequency from the upward crossings of the mean (job 49596583; u x 1e-3):

| Run | u_x | u_y | Frequency (u_x, u_y) | Wall |
|---|---|---|---|---|
| Q4 16 x 2, dt 0.005, rho_inf 1 (the deck) | -14.548 +- 14.548 | -63.942 +- 65.500 | 1.0954, 1.0942 | 54 s |
| the same, rho_inf 0.8 | -14.543 +- 14.544 | -63.923 +- 65.499 | 1.0954, 1.0941 | 56 s |
| Q6 16 x 2, dt 0.005 | -14.553 +- 14.554 | -63.957 +- 65.511 | 1.0953, 1.0940 | 99 s |
| Q4 32 x 4, dt 0.005 | -14.548 +- 14.549 | -63.944 +- 65.502 | 1.0954, 1.0941 | 136 s |
| Q4 16 x 2, dt 0.0025 | -14.612 +- 14.613 | -64.440 +- 65.113 | 1.0948, 1.0945 | 74 s |
| Q4 16 x 2, dt 0.005, BDF2 | -14.440 +- 14.439 | -63.817 +- 65.354 | 1.0934, 1.0948 | 66 s |
| FeatFlow 4+0, dt 0.005 | -14.305 +- 14.305 | -63.607 +- 65.160 | 1.0995 | |

- Space is converged to 2e-4 (Q6, or the 32 x 4 mesh, against the deck).
- The time step moves u_y's mean by 0.8 % from dt 0.005 to 0.0025: the extremes of a
  signal that carries the higher bending modes, whose phases dt shifts, not the
  fundamental's second-order error (omega dt = 0.035). The benchmark's values move the
  same way, so the comparison is at its dt.
- At dt 0.005 the deck is within 0.53 % (u_y mean and amplitude), 1.7 % (u_x) and
  0.4-0.5 % (frequency) of the benchmark. Numerical damping hardly matters for this mode (rho_inf
  0.8 vs 1: 0.03 %); BDF2 damps the higher modes and lands 0.2 % away.
- At most 3 Newton iterations per step, about 140 CG iterations per step.

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
- **Newton's residual has a roundoff floor; its corrections do not.** Element
  contributions much larger than the load cancel at every node. The floor was 3e-11 of
  the load on H6's small cantilever. On the CSM1 flag (job 49587586) it grew from
  2.5e-9 (16 x 2, Q4) to 1.2e-7 (64 x 8, Q6, 4e4 dofs). With the residual test alone
  (`newton_rtol` 1e-8), the 32 x 4 Q5 and Q6 flags ran into the 25-iteration cap with
  u(A) converged. The relative correction ||dx|| / ||u|| kept falling quadratically to
  about 1e-15 on every mesh (CSM1: 2.1e-5, 1.1e-8, 1e-15). So a step also converges at
  `newton_dx_rtol` = 1e-10 (on by default), five orders above where the corrections
  stop. `newton_rtol` stays at 1e-8, not the 1e-10 of the spec's sample deck.
- **SVK loses strong ellipticity in moderate compression.** V1 first used amplitude 0.1:
  J down to 0.69 in 2D (E = 1, nu = 0.3). There the acoustic tensor's least eigenvalue
  is -0.025 (+0.18 at amplitude 0.05), so the continuous problem is ill-posed. Newton
  converged quadratically on 4 x 4 meshes and diverged on finer ones, which resolve the
  unstable short waves (job 49589795; np 1 and np 4 identical). This is the weakness the
  spec names (section 1) and the reason for neo-Hookean next. Mesh motion with SVK should
  watch the element Jacobians.

## 5. Next

- **More performance.**
  - Measure the scaling beyond two GPUs, at np 4 and 8 on one node. Tests stay at np 1
    and 2 on GPUs, but benchmarks at 4-8 H100s are allowed.
  - What is left at np 2 is about 0.1 ms per Q6 apply. Intra-node GPU MPI goes over
    InfiniBand (rc_mlx5) by design, not CUDA IPC (change 42).
  - Geometry on the fly: **shelved** (human 2026-10-21). The operator's ~93 B/dof are
    under 5 % of the LOR-AMG's estimated 2-3 kB/dof (elasticity.md section 3), so it
    would save under 2 % of the solver's memory, at a cost in speed.
- **Milestone 4, the hyperelastic mesh motion: parked** (human 2026-10-22: "What is
  standard in the literature? That's what we should be doing"). An SVK model behind
  `MeshMotionModel` (kept on a local branch, not on main) did worse than the linear one on
  the DFG cylinder turned about its centre with the walls held (stiffening chi = 1):
  min det F 0.76 / 0.37 at 30 / 40 degrees against the linear model's 0.89 / 0.82, and
  Newton failed from about 45 degrees, where the linear model stays valid to 90 (it
  inverts at 100). Line search, four load steps per move and 5-degree moves trace the same
  converged path and fail at the same angle, so it is the material's compression weakness
  (section 4), not the solver. The literature's standard (Shamanskiy & Simeon, Comput.
  Mech. 2021, seven techniques on Turek-Hron FSI2): linear elasticity with Jacobian
  stiffening, what `LinearElasticMotion` is, for moderate motions (it handled 1.6-1.9
  times FSI2's); biharmonic extension first for large ones; a log-neo-Hookean
  pseudo-solid for extreme ones (its ln J term keeps elements from inverting). Nobody uses
  SVK for mesh motion. The Turek-Hron cases need only the linear model.
- **Later, each needing the human's OK** (spec section 7): neo-Hookean (now motivated by
  that measurement), follower pressure loads for FSI, the refreshed LOR tangent, a mixed
  u-p form, an energy-momentum integrator.
- The fixed reference LOR-AMG took 50-80 CG iterations per Newton correction on the CSM
  decks (CG rtol 1e-10). Large rotations in mesh motion may need the refreshed LOR
  tangent (spec D5).
