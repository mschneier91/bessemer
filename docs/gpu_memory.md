# GPU memory: what was recovered, and what is left

The record of the GPU memory work: each change, numbered, with what it saved, and what is
left. **The current working list (Q6, toward 100M DOFs on one H100) is the next section.**
Rounds one to three (2026-10-13 and 14) were measured on Taylor–Green (Re 1600) with the case defaults
(Q3/Q2, IMEX BDF2, grad-div γ = 2h, Cahouet–Chabard Schur PC, PCG with point-block Jacobi
on the velocity block), about ten steps per run, one NVIDIA H100 80 GB on Bridges-2, CUDA
backend. Memory is the device's peak in use (`nvidia-smi`, and the profiler gauge of
change 1), in units of 1000 MiB; step times are the mean from step 3 on, with the
synchronizing profiler on.

## Q6: toward 100M DOFs on one H100 (the working list)

Human 2026-10-16: the goal is 100M DOFs on one 80 GB H100, and this section is the place
to refer to while working toward it ("where is a central file which we can refer to as we
work through memory optimization?"). Phases (human): first the items with no expected time
cost, each measured and merged when it works; then those that may cost a little or must be
measured; those that could cost meaningful throughput or change the answer are recorded
here and held. Single-precision AMG is held because it changes the answer (human).

**Where it stands.** Taylor–Green Re 1600, Q6/Q5, OIFS at CFL 2, ~25 steps, one rank on one
H100, `profile: sync`, 42³ elements = 57.3M DOFs (48.0M velocity, 9.3M pressure). Peak is
`nvidia-smi`'s; B/DOF divides it by all 57.3M DOFs.

| | Coupled (FGMRES) | Schur (`solver.outer: schur`) |
|---|---|---|
| main aa6aa7e (job 49342093) | 71,781 MiB (1,314 B/DOF) | 58,463 MiB (1,070 B/DOF) |
| + change 17 (job 49343807) | 66,311 MiB (1,214 B/DOF) | 56,485 MiB (1,034 B/DOF) |
| + change 18 (job 49345234) | 63,029 MiB (1,153 B/DOF) | 53,203 MiB (974 B/DOF) |
| + change 19 (job 49345424) | 61,955 MiB (1,134 B/DOF) | 48,127 MiB (881 B/DOF) |
| + change 20 (job 49345599) | 57,603 MiB (1,054 B/DOF) | 46,031 MiB (842 B/DOF) |

100M DOFs fit only below about 850 B/DOF (the card's 81,559 MiB over 100M), less headroom.
Last measured capacity (before change 17): coupled 57.3M runs and 70.4M does not; Schur
70.4M runs and 85.5M does not. GPU iteration counts vary from run to run, and the coupled
peak moves with the largest solve's FGMRES iterations: about 0.85 GiB per iteration (two
system-sized vectors of 437 MiB).

**The ledger at 57.3M** (main aa6aa7e, job 49342093; each region's own growth, MiB). A
velocity vector is 366 MiB, a velocity element vector (E-vector) 582, a pressure vector 71;
the 3k rule of OIFS has 10³ points per element (74.1M points), the momentum and divergence
rule 8³ (37.9M), the collocated mass rule 7³ (25.4M). "Accounted" means the listed arrays add
up to the region within 1 %; otherwise the split is an estimate.

| Piece | MiB | What it holds | |
|---|---|---|---|
| OIFS stepper setup | 5,087 + 1,249 | the Jacobians at the 10³ points (9 numbers per point; computed by `OifsConvection` because MFEM's interpolator stops at 8 points), plus u, p, p_static and the mass diagonal and its reciprocal | accounted (`time_integrator::setup`) |
| OIFS wind data | 7,366 | per-wind quadrature data (3 winds × 3 numbers per point, 5,087), the winds at the points to build it (1,696), the wind E-vector (582) | accounted (`oifs::setup`) |
| OIFS apply scratch | 2,268 | two E-vectors, two local-dof vectors (removed by change 17), the mass right-hand side | accounted (`oifs::rhs`) |
| Divergence | 5,212 | MFEM's PA data W adj(J) (9 per point at 8³, 2,602) and the mesh's cached Jacobians at 8³ (2,602; grad-div reads them at apply, change 13) | accounted |
| Mass | 3,158 | the PA form's E-vector pair (1,164), PA data (194), the diagonal (366), the mesh's Jacobians at the GLL rule (1,746) | sum 3,470: not reconciled |
| Momentum | 2,222 + first apply 2,776 | compact diffusion metric (6 per point at 8³, 1,735), grad-div α; first apply: an identity RAP pair (732, removed by change 17), 2,044 not identified | partly |
| Pressure LOR-AMG | 4,122 + 242 | hierarchy and matrix (hypre pool capped) | |
| Cahouet–Chabard | 1,784 + 1,040 + 736 | velocity mass inverse (an identity RAP pair, removed by change 17; reciprocal diagonal 366, the same values as OIFS's; 686 not identified), inner CG, B M⁻¹ Bᵀ vectors | partly |
| Velocity PC + diagonal | 1,542 + 736 | Jacobi-PCG work vectors, momentum diagonal and the kept part | |
| Time integrator | 2,208 + 736 + 1,444 + 1,472 | BDF right-hand side, commit, forcing (708 removed by change 17), OIFS advection stages | |
| Coupled only | 8,800 + 7,446 + 1,174 | FGMRES V and Z (~20 system vectors at 437), the projection basis (16 system vectors), step_reference | |
| Coupled warm-up | 6,654 | first applies of the block operator and preconditioner; moved, not added (Schur allocates the same at its first solve, in `solve_true/rhs`) | probe 49342945 |
| Schur only | 1,392 + ~1,280 | velocity CG vectors; the pressure projection basis (18 pressure vectors) | |
| Not in any region | 3,216 Schur / 8,200 coupled, + 1,482 | `case::run`'s own growth, and `step`'s | unknown |

**Phase 1: no time cost expected.**

- **Done, change 17:** no RAP over identity prolongations, no OIFS local-dof copies at one
  rank, no kept copy of OIFS's mass diagonal, no kept zero forcing. 3,724 MiB less in both
  methods, every result bitwise unchanged.
- **Done, change 18:** the collocated mass applied as its diagonal on conforming spaces,
  without the mesh's GLL-rule Jacobians; the momentum block's c0 M as a diagonal shift. 3,282 MiB less in both
  methods, results within roundoff, and the steps 10–14 % faster.
- **Done, change 19:** our own partially assembled blocks from MFEM's public pieces, their
  element vectors (and the OIFS kernel's) lent by one shared pair. 2,820 MiB less in both
  methods, bitwise unchanged, no time cost.
- The pressure preconditioner's reciprocal mass diagonal and OIFS's hold the same values
  (366 MiB): share them where the lifetimes allow.
- **Tabled (human 2026-10-17): full-size work vectors.** Where the prolongation is not the
  identity (hanging nodes, or
  two or more ranks): the RAP wrappers' pairs (about 1.9 GiB at 57.3M), the OIFS mass CG and
  the Cahouet–Chabard Chebyshev mass inverse (both only when the mass is not diagonal) are
  never in use at the same time either: one pool of full-size vectors.

**Meshes with hanging nodes (estimate, not measured).** Changes 17 and 18 rest on an
identity prolongation and a diagonal true-dof mass, which hanging nodes break: about 0.7 GiB
of change 17 applies there and none of change 18, and the non-diagonal mass adds the OIFS
mass CG and a Chebyshev mass inverse (+1.5–3 GiB). Roughly 1,100 B/DOF Schur and 1,350
coupled at 57.3M against 963 and 1,153 conforming; the shared scratch (next) and phase 2
apply in full. To measure: Taylor–Green with a refined band (as oifs_test O3).

**Phase 2: a small time cost possible; measure.**

- **Done, change 20: the divergence from the cached Jacobians.** 2,606 MiB less in both
  methods; the divergence and gradient applies 15–17 % slower, the step 0.5–1.4 %.
  **Follow-up:** W adj(J) once per element in shared memory, to win the kernel time back.
- **OIFS winds combined inside the apply kernel**: drops the per-wind data and its build
  array (6,783 MiB). The apply would read about one number per point instead of nine and
  interpolate the wind itself: expected neutral or faster, but measure.

**Held: could cost meaningful throughput, or change the answer.**

| Item | Saving | Why held |
|---|---|---|
| OIFS Jacobians on the fly from the mesh nodes | 5,087 MiB | cheap on straight-sided meshes; roughly doubles the substep interpolation on curved ones |
| ... or stored in single precision | 2,543 MiB | no time cost; changes the answer (~1e-7 relative in the convection) |
| Geometry on the fly in the momentum and divergence operators | cached Jacobians 2,602 + diffusion metric 1,735 | the velocity solve is the hottest kernel: perhaps 5–15 % |
| FGMRES basis in single precision | ~4.3 GiB | cheaper per iteration, but iteration counts may grow; the answer moves within the tolerance |
| Projection basis in single precision | coupled 3.6 GiB, Schur ~0.6 | a different initial guess: the answer moves within the tolerance |
| Schur projection with the solutions only (S-orthonormal basis) | ~0.6 GiB | the same: a different guess |
| `projection_k` 8 → 4 | coupled 3.6 GiB | more iterations per step |
| Single-precision AMG | ~2 GiB | changes the answer (human 2026-10-16); the single-precision plan comes after all other work |
| One level of aggressive AMG coarsening | 2.1 GB at Q3 64³ | AMG settings are the human's call ("an optimization for another day") |

**Next measurement:** the 3.2–8.2 GiB that no region claims. An allocation census (Nsight
Systems with CUDA memory backtraces, one short run at 21³) would name every device
allocation and its call site; to try before guessing at it.

## Result (rounds one to three, Q3)

| | Largest mesh on one H100 | Velocity dofs | Peak at 32³ | Peak at 64³ |
|---|---|---|---|---|
| Before (morning) | 56³ | 14.2 M | 23.7 GB | out of memory (62 GB per GPU on two) |
| After changes 2–3 | 69³ | 26.6 M | 9.8 GB | 63.9 GB |
| After changes 4–5 | — | — | — | 51.8 GB |
| After changes 7–9 | 80³ | 41.5 M | — | 43.5 GB (38 GB after step 1) |
| After changes 11–12 (2026-10-14) | 85³ | 49.7 M | — | 37.6 GB |
| After changes 13–15 | 87³ | 53.3 M | — | 35.8 GB |
| After change 16 | **88³** | **55.2 M** | — | **34.7 GB** |

Per velocity dof: 5.2 kB at 56³ before, 1.95 kB at 80³ after the first two rounds,
1.53 kB at 87³ after change 15, 1.48 kB at 88³ after change 16. Sweeps: after change 9, 64³ 43.5 GB, 72³ 59.0, 76³
68.0, 80³ 78.9, 81³ out of memory; after change 12, 82³ 73.2 GB, 84³ 78.1, 85³ 80.5, 86³
out of memory; after change 15, 86³ and 87³ run (at FGMRES restart lengths 3–6 and 1)
and 88³ does not; after change 16, 88³ runs (restart length 1, 32.4 s per step) and 89³
does not (job 49216290).

Near the limit the meshes run with a short FGMRES restart (change 15): 87³ converged in
12.3 outer iterations per step at restart length 1, 29.9 s per step, about 20 % slower than
proportional to 64³. For production at full speed the practical limit is about 85³.

**Step time at 64³ through the third round** (same iteration counts, 10.3–10.4 outer
iterations per step):

| | Peak | Step time |
|---|---|---|
| MFEM's vector-diffusion storage (changes 1–10 and 12) | 42.8 GB | 10.72 s |
| Change 11, compact storage | 37.6 GB | 10.54 s |
| Change 13, grad-div stores α only | 35.3 GB | 11.11 s |
| Change 14, refresh work kept across Δt | 35.8 GB | 9.84 s |
| Change 16, CFL Jacobians per call | 34.7 GB | 10.02 s (within noise) |

## The changes

1. **A memory gauge in the profiler.** `Profiler::SetGauge`; with `output.profile: sync`
   on a GPU, `DeviceMemoryInUse` (`cudaMemGetInfo`) is read at every region entry and
   exit, and the profile CSV gains three columns per region: what its first call left
   allocated, the same over all its calls, and the peak while it was open. Every number
   below comes from it. (`src/util/profiler.*`, `src/util/device_sync.cpp`)

2. **11 GB of GPU stack reserved by MFEM's vector-diffusion diagonal.**
   `VectorDiffusionIntegrator::AssembleDiagonalPA` (3D) runs a kernel whose per-thread
   work arrays are sized by the device maximum (14 points per direction), 44 kB of stack
   per thread; CUDA reserves that for every concurrent thread (270,336 on an H100) on
   the first call and keeps it: **11.1 GB, independent of the problem**.
   `LeanVectorDiffusionIntegrator` overrides only the diagonal with a direct sum that has
   no work arrays; it equals MFEM's to 8e-16 and the apply is bit-identical. Written up
   for upstream: [upstream/mfem_vector_diffusion_diagonal.md](upstream/mfem_vector_diffusion_diagonal.md)
   (not yet reported). (`src/operators/lean_vector_diffusion.*`)

3. **The trapezoidal starter solver released before the BDF solver is built.** Two
   whole solvers (operator, preconditioners, AMG) coexisted from step 2 on: 3.1 GB at
   32³. Superseded by change 7, which removes the second solver altogether.

4. **Diagnostics that store nothing.** `FlowDiagnostics` (energy, dissipation,
   divergence) held three partially assembled forms, about **10 GB at 64³**, built even
   with the diagnostics CSV off because the run summary asks for the energy. Now one
   device pass interpolates u and ∇u at the quadrature points and sums the three
   integrands. Case setup at 64³: 19.8 → 7.5 GB. (`src/post/flow_sums.*`,
   `src/post/diagnostics.*`)

5. **The unconstrained viscous operator on demand.** Every solver built ν K for the
   explicit right-hand-side terms, but only the trapezoidal starter uses it: **6.75 GB
   at 64³** (MFEM's vector-diffusion PA data, see "What is left"). Built on first use.
   (`StokesOperator::ViscousUnconstrained`, `ReleaseViscousUnconstrained`)

6. **The grad-div coefficient from the fluid's viscosity in every solver.** Not a saving
   by itself: the starter (built with ν/2 on its viscous term) computed γ's
   ν-proportional part from ν/2, so its γ differed from the BDF steps'. With
   `grad_div_nu_ref` γ is fixed for the run, which change 7 needs. (Changes step 1 only,
   and only when `physics.grad_div_nu` ≠ 0.)

7. **One solver for the starter and the BDF steps.** `StokesSolver::Refresh(c0, ν)` lets
   a solver change its mass factor and the factor on its viscous term (ν/2 → ν); the
   pressure AMG depends on neither. `StokesTimeIntegrator::AdoptSolver` hands the
   starter's solver to BDF2 (and BDF2's to BDF3) instead of building another: no second
   operator, AMG hierarchy or solve workspace; the hand-over frees 8.9 GB at 64³. Not
   with the rotational form or error-controlled steps. Test
   `SchurQuality.RefreshWithViscosityMatchesAFreshSolver`.

8. **The starter's viscous operator freed before its solve.** After change 7 the peak
   moved into step 1, where that operator (6.75 GB at 64³), the starter and the solve
   workspace coexisted. It is needed for one product in the starter's right-hand side
   and is now freed right after it. With change 9: **64³ 51.8 → 43.5 GB**.

9. **The CFL estimator's velocity copies per call.** Two element-wise copies of the
   velocity were kept for the run (0.8 GB at 64³), and one was an interpolation to the
   GLL points that, with the GLL nodal basis, are the dofs themselves. Now one copy per
   call, no interpolation: 2.0 → 1.2 GB at 64³ (the rest is the Jacobians at the nodes,
   which it needs). (`src/time/cfl.*`)

10. **The FGMRES restart length capped to the free memory** (not a saving: a guard).
    FGMRES holds two system-sized vectors per outer iteration up to its restart length
    (0.37 GB per iteration at 64³, 0.73 GB at 80³). With `time.on_solver_failure:
    cut_step` (the case-level default) the restart length is lowered below
    `solver.kdim` before each solve if its vectors would not fit, and a solve that then
    fails rejects the step and retries at half the step (held for 5 steps) instead of
    ending the run. `abort` restores the old behaviour.

Also added, defaults unchanged: `solver.lp_strong_threshold`, `solver.lp_agg_levels`,
`solver.lp_print_level` for the pressure AMG.

### Third round (2026-10-14)

Human: "do all of these but keep a path for the rotational form too that requires more
storage, also ... do the vector diffusion thing too ... except keep a path to the built
in mfem way"; then "do 1" (change 13), "Yes, do that for sure" (14), "yes" (15), "go ahead
and make that CFL change" (16).

11. **Compact vector-diffusion storage** (the default; `discretization.vector_diffusion:
    compact | mfem`). `LeanVectorDiffusionIntegrator` stores, for a scalar coefficient, one
    symmetric geometric matrix per quadrature point shared by the velocity components: 6
    numbers in 3D where MFEM stores 27 (the full matrix once per component). Setup uses
    MFEM's own expressions and the apply is MFEM's shared-memory kernel reading the shared
    matrix: **bit-identical to MFEM's apply** in all 16 tested cases (2D/3D, orders 2 and
    3, both orderings, constant and variable ν; diagonal within 1.2e-15). Anything else
    (matrix or vector coefficients, non-tensor elements, libCEED) falls back to MFEM's
    path, and `mfem` selects it outright. **64³: 42.8 → 37.6 GB**, and 2 % faster per step
    (less data read). (`src/operators/lean_vector_diffusion.*`)

12. **Nine small items, with the rotational form keeping what it needs.** (a) The momentum
    block and (b) the divergence block share one P^T A P between their constrained and
    unconstrained forms (each had its own, with two work vectors), and the mass operator
    lost a constraint wrapper it never needed. (c) Point-block Jacobi frees its build
    inputs (superseded by change 14). (d) Without a rotation term its 3×3 inverses are
    symmetric and stored as 6 numbers, bit-identical to the full 9. (e) The raw
    right-hand side is kept only when forces are computed (`keep_rhs`). (f) The lagged
    velocity w* is allocated only for the rotational form. (g) The mass-path Schur block
    is built only on the mass path. (h) Cahouet–Chabard frees its plain pressure mass
    once the γ-weighted one exists, except with the rotational form, whose Schur
    preconditioner uses it. (i) γ is kept per element, not per quadrature point (125×
    smaller at Q3). Also: the old velocity preconditioner is freed before its
    replacement is built. **About 0.7 GB at the 64³ peak**, not the 2.4 GB estimated:
    MFEM allocates a work vector on the device at its first use, and the duplicate
    wrappers removed by (a) and (b) are never applied without walls or forces, so they
    had cost nothing in Taylor–Green. They do in wall-bounded cases.

13. **Grad-div stores α only.** Its PA data was adj(J) (9 numbers) and α = γ w / det J
    per point; now only α, with adj(J) recomputed in the kernels from the Jacobians MFEM
    already caches for that rule (shared with the diffusion and convection forms). **64³:
    37.6 → 35.3 GB.** The cost: the velocity-block apply is 5 % slower (32.5 vs 30.9 s
    over 117 applications), and the point-block Jacobi rebuild 44 % slower, which change
    14 removes. (`src/operators/grad_div_integrator.*`)

14. **What does not change with Δt is kept across refreshes** (a speed change, not a
    saving). Every Δt change rebuilt the momentum diagonal from scratch (0.40 s at 64³)
    and a new point-block Jacobi with the grad-div coupling (0.81 s): 1.22 s of an 11 s
    step. Now the diagonal is c0·(mass diagonal) + a kept diagonal of ν K + γ grad-div,
    assembled once (again after a viscosity change) from the momentum form's own
    integrators through a proxy form; and the preconditioner and its CG are updated in
    place with the new diagonal. Refresh: **1.22 → 0.009 s per step; 64³ step 11.11 →
    9.84 s**, for **0.5 GB** kept (the diagonal and coupling the preconditioner holds, and
    the kept diagonal). The updated preconditioner is bit-identical to a fresh one, and a
    refreshed solver matches a fresh solver exactly on the CPU (1.6e-12 on the GPU).

15. **The restart-length guard measures the first solve's workspace** instead of guessing
    it. A solver's first application allocates its preconditioners' workspace for the run:
    9.2 GB at 64³ (the pressure AMG's 5.8 GB of it). Change 10 allowed 20 % of the memory
    in use for it (4 GB at 64³), and at 86³ the solve ran out of memory. Now
    `StokesSolver::WarmUp()` applies the operator and the preconditioner once to a random
    vector before the first solve (1.7 s at 64³, once per run; the solve that follows is
    identical), the free memory is read after it, and the formula counts the 2k + 4
    system-sized vectors a solve holds (FGMRES's 2k + 2, the solve's right-hand side and
    solution). 86³ and 87³ now run, at restart lengths 3–6 and 1.

16. **The CFL estimator's Jacobians computed per call.** The estimator evaluates Nek5000's
    CFL at the GLL nodes and needs the Jacobians there; MFEM cached them in the mesh for
    the whole run (1.2 GB at 64³). They are now computed at each call with MFEM's own
    routine, without the cache (bit-identical, test C4), between solves and so away from
    the peak: **64³ 35.8 → 34.7 GB**; the limit moves to 88³. The definition and every CFL number are unchanged. With the
    collocated mass (BDF3, OIFS) the mass form caches the same Jacobians and nothing is
    saved. (`src/time/cfl.*`)

### Fourth round (2026-10-16, Q6)

Human: "lets phase this ... do the no cost expected items see where that gets us and commit
and merge it when its working". Measured at Q6 (the working list above).

17. **No copies through identity prolongations; three kept copies dropped** (bitwise
    unchanged). On one rank MFEM's prolongation is an `IdentityOperator`, and
    `StokesOperator::TrueDofOperator` still wrapped the mass, divergence, momentum and
    viscous operators in a `RAPOperator` over it: two velocity-sized work vectors per
    operator, allocated on the device at its first apply, only to copy. It now returns the
    form itself, as MFEM's `FormSystemMatrix` does. Likewise the OIFS convection kernel
    skips its local-dof copies at one rank, `OifsAdvector` keeps the reciprocal of the mass
    diagonal but not a copy of the diagonal, and a zero body force (the default) is no
    longer assembled and kept. CPU job 49343803 ran main and the change on Taylor–Green (Q6
    OIFS, Q3 IMEX), the channel (walls, forcing, OIFS) and DFG 3D (inflow, outflow, error
    control) at np 1 and 2: every output byte-identical. Q6 Taylor–Green at 57.3M DOFs
    (job 49343807 against 49342093): 3,724 MiB less of the profiler's per-region growth in
    each method; peak 71,781 → 66,311 MiB coupled (about 1.7 GiB of it two fewer FGMRES
    iterations in the largest solve, run-to-run variation on the GPU) and 58,463 → 56,485
    MiB Schur. Two or more ranks: unchanged (the prolongation is not the identity there).
    Tests: CPU smoke/fast at np 1, 2 and 4, 73/73 each; GPU at np 1 and 2, 71/71 each.

18. **The collocated mass applied as its diagonal** (results change at roundoff). Under
    OIFS (and BDF3) the velocity mass is collocated, GLL nodes and rule, so exactly
    diagonal; it was still a partially assembled form: an element-vector pair, PA data and,
    through the mesh's cache, the Jacobians at the GLL points (9 numbers per point) for the
    run, read again by the momentum form's mass term at every Δt change. Now `M` is a
    `DiagonalOperator` over its diagonal, which `AssembleCollocatedMassDiagonal` computes
    from MFEM's det J at the GLL points without the cache, and the momentum block's c0 M is
    a diagonal shift added at each apply with c0 read then (`DiagonalShiftOperator`), so a
    new Δt re-assembles nothing (`src/operators/collocated_mass.*`). Only on a conforming
    space: with hanging nodes the true-dof mass P^T D P couples them to the nodes they are
    constrained by, and the form stays (oifs_test O3 caught a first version that did not
    check). Nothing caches the GLL Jacobians now, so change 16 holds with the collocated
    mass too. Against MFEM's partial
    assembly on a curved mesh: the diagonal within 1e-14 and the apply within 1e-13 (new
    StokesOperator tests, 2D and 3D, plus the momentum diagonal after refreshes). CPU job 49345233 ran main and the change at np 1 and 2: Taylor–Green Q6 OIFS within 4e-13 relative and the channel within 4e-14, iteration counts unchanged; Taylor–Green Q3 IMEX and DFG 3D byte-identical.
    Q6 Taylor–Green at 57.3M DOFs (job 49345234 against 49343807): 3,282 MiB less of the
    per-region growth in both methods (mass 3,158 → 592; the CC mass inverse's setup 680;
    the momentum form's mass data 194); peak coupled 66,311 → 63,029 MiB, Schur 56,485 →
    53,203 MiB. Steps 3–25 took 25.4 s instead of 28.3 (coupled, −10 %) and 29.0 s instead of
    33.6 (Schur, −14 %): the mass apply is now a multiply (the same to 0.1 s in a second
    run, job 49344488). Tests: CPU smoke/fast at np 1, 2 and 4, 73/73 each; GPU at np 1 and 2, 71/71 each.

19. **Partially assembled blocks from MFEM's public pieces, with shared element vectors**
    (bitwise unchanged). MFEM's forms keep a pair of element vectors each for the run (its
    mixed form also a full-size vector for adding its result), with no way to hand them a
    workspace. The mass (when it is not diagonal), momentum, divergence and starter viscous
    blocks are now `PaOperator` / `PaMixedOperator` (`src/operators/pa_operator.*`): the
    integrators' public `AssemblePA` / `AddMultPA` and the element restrictions, step for
    step as MFEM's form does them, with the element vectors lent by one `ElementScratch`
    pair (`src/operators/element_scratch.*`) that the time integrator owns and the OIFS
    convection kernel borrows too (its wind, input and output element vectors). No apply
    runs inside another, so one pair serves them all. A non-diagonal mass still gets its
    diagonal from MFEM's form (a temporary one), for its parallel and hanging-node
    assembly. A first version re-pointed MFEM's protected members instead; rejected in
    favour of public pieces only (human: "what you just proposed is much better"). CPU job
    49345423: change 18 and change 19 byte-identical on the four decks at np 1 and 2. Q6
    Taylor–Green at 57.3M DOFs (job 49345424 against 49345234): 2,820 MiB less of the
    per-region growth in both methods (the divergence's first-apply vectors, the OIFS
    element vectors, a momentum-form pair); peak coupled 63,029 → 61,955 MiB and Schur
    53,203 → 48,127 MiB (the peaks fall at different points of the run; the per-region sum
    is the like-for-like figure). Steps unchanged (25.3 and 28.7 s for steps 3–25). Tests: CPU smoke/fast at np 1, 2 and 4, 73/73 each; GPU at np 1 and 2, 71/71 each.

20. **The divergence reads the mesh's cached Jacobians** (phase 2; results change at
    roundoff). MFEM's partial assembly of `VectorDivergenceIntegrator` stored W adj(J) at
    every point (9 numbers at the 2k + 2 rule), next to the mesh's cached Jacobians at the
    same rule, which grad-div reads at every apply (change 13). `LeanVectorDivergenceIntegrator`
    (`src/operators/lean_vector_divergence.*`) keeps only a pointer to that cache and forms
    W adj(J) in its kernels (those of `kernel_specializations.cpp`, MFEM's, with the
    components unrolled) with MFEM's expressions; 3D, orders 2 to 7, else MFEM's path. New
    test against MFEM (orders 2 to 7, both kernel variants, a constant coefficient, the 2D
    fallback): within 1e-13. CPU job 49345598: Taylor–Green Q6 within 3.4e-13 of change 19;
    the Q3 cases byte-identical. Q6 Taylor–Green at 57.3M DOFs (job 49345599 against
    49345424): 2,606 MiB less in both methods (the divergence region 5,212 → 2,606: what is
    left is the shared Jacobian cache); peak coupled 61,955 → 57,603 MiB, Schur 48,127 →
    46,031 MiB. **The cost:** the divergence apply 4.54 → 5.23 ms and the gradient 6.56 →
    7.65 ms (+15 % and +17 %): at Q6 the kernel goes one velocity component at a time
    (register pressure) and reads 6 of the 9 Jacobian entries per component, 18 reads per
    point against 9 stored values. Per step +0.5 % (coupled) and +1.4 % (Schur); steps 3–25
    25.3 → 25.8 s and 28.7 → 29.1 s. To win it back: form W adj(J) once per element in
    shared memory (37 KB at Q6) before the component loop. Tests: CPU smoke/fast at np 1, 2 and 4, 74/74 each; GPU at np 1 and 2, 72/72 each.

## Where the memory is now (64³, one H100, after change 15)

From the profiler gauge (job 49215743); the peak is in the FGMRES solve of a BDF2 step.

| Piece | Memory | Note |
|---|---|---|
| The first solve's workspace | 9.2 GB | the pressure AMG's 5.8 GB (allocated at its first V-cycle); CC's inner vectors 0.65; velocity CG 0.5 |
| Convection | 4.5 GB | half the 5-point Jacobians, shared by every 5-point form; half its own PA data |
| Schur preconditioner setup | 4.5 GB | the pressure LOR-AMG hierarchy 3.2; velocity mass inverse 1.0; pressure mass 0.3 |
| Divergence operator | 2.25 GB | |
| Momentum operator | 2.2 GB | was 9.6 GB: compact diffusion (change 11), α-only grad-div (13) |
| Mass operator | 1.9 GB | includes the 4-point Jacobians |
| CFL estimator | 1.2 GB | 0 after change 16 |
| Velocity preconditioner | 0.65 GB | the point-block inverses, and the diagonal and coupling kept by change 14 |
| FGMRES basis and solve vectors | ~5.6 GB | transient, per solve, at 10–14 iterations |

## MFEM's vector-diffusion storage: taken (change 11)

`VectorDiffusionIntegrator::AssemblePA` stores the full dim × dim geometric matrix for
each velocity component, 27 numbers per point in 3D, where a scalar coefficient needs one
symmetric matrix, 6 numbers. On 2026-10-13 this was left alone (human: "that is a big
change I'm not really comfortable with since that is like the core mfem library"); on
2026-10-14 the human asked for it with MFEM's path kept selectable, and it is change 11.
The upstream issue still stands: a compact storage for the scalar case in MFEM itself, the
same report as change 2's ([upstream/mfem_vector_diffusion_diagonal.md](upstream/mfem_vector_diffusion_diagonal.md)).

## hypre's device memory pool (2026-10-16)

Spack's hypre recipe configures every CUDA build without Umpire with
`--enable-device-memory-pool`. The pool's defaults (bin growth 8, no largest bin, no cache
limit) round every hypre device allocation up to the next power of 8 and keep every freed
block. That was most of the pressure AMG's "first-application workspace" (change 15): the
hierarchy is built lazily in the first V-cycle, and its matrices, interpolation and setup
temporaries all went through the pool. Taylor-Green Q7 at ~128³ (one H100, profile: sync):

- hypre's defaults: AMG setup 3,284 MiB, run peak 13,089 MiB;
- `HYPRE_SetGPUMemoryPoolSize(2, 1, 20, 64 MiB)` (only blocks up to 1 MiB pooled): AMG setup
  594 MiB, run peak 10,371 MiB (-21 %; Q3 -680 MiB). AMG setup +5 ms at Q7, +15 ms at Q3;
  steps unchanged (0.2546 against 0.2534 s).
- Frequent rebuilds, the 2D square cylinder adapting every 5 steps to t = 20 (116 adapts in
  1,149 steps, job 49341452): 134.1 s of solve time with the cap, 134.4 s with hypre's
  defaults.
- Capacity at Q6 (throughput.md): the largest Taylor-Green that ran on one H100 went from
  39.2M to 57.3M DOFs (70.4M with the Schur outer method).

`ConfigureDevice` applies that cap on every GPU run; `INCNS_HYPRE_POOL` overrides it. The pool
exists because `cudaMalloc` is slow and `cudaFree` synchronizes the device: it pays when AMG
hierarchies are rebuilt often. Ours is built once per mesh. If a case does rebuild often
(AMR adapting nearly every step, the velocity LOR-AMG option rebuilding on dt changes) and
the cap costs time there, the fallbacks are (human 2026-10-16: keep them on record):

- **Rebuild hypre with `--enable-device-malloc-async`** (CUDA's stream-ordered allocator:
  cheap allocation and release without power-of-8 rounding; in hypre 2.33's configure).
  Spack's recipe has no variant for it, so it takes a package override in our environment.
- **Umpire** (`+umpire` builds of MFEM and hypre): one shared, coalescing pool for both
  libraries, with per-allocator statistics. The larger change.

Both are environment rebuilds (AGENTS.md rule 3). Measured alongside: MFEM's LORSolver kept
its LOR mesh, space and batched-assembly scratch for the solver's lifetime; freeing them
after assembly saves 460 MiB at Q3 and Q7 (6d90d0c). BoomerAMG's strength threshold 0.5 / 0.6
grew the hierarchy (+0.4 / +1.4 GB), and one level of aggressive coarsening did not shrink it
at Q7.

## What was left at Q3

Superseded by the Q6 working list above; kept for the record. Nothing large at the time. At 64³:

- **The mass operator on the 5-point rule** (its Jacobians shared with the other
  forms, which all use that rule; after change 16 the mass's 4-point rule is the only
  other one with cached Jacobians): about 1.15 GB, but the mass apply would do 5³ instead of 4³ points
  per element, roughly 5–10 % on the velocity solve. Left (human, 2026-10-14, after
  asking about the throughput trade-off).
- **Krylov vectors in single precision:** about 2.5 GB of the FGMRES basis. Not tried; it
  changes the outer solver's arithmetic.
- **Pressure AMG.** One level of aggressive coarsening (`solver.lp_agg_levels: 1`) saved
  2.1 GB at 64³ with the same outer iterations over 11 steps; the strength threshold
  saved nothing. Its first-application workspace (5.8 GB, change 15) was the largest
  single piece: mostly hypre's pool rounding, capped since 2026-10-16 (above). Human: AMG
  options have trade-offs, "an optimization for another day".
- **MFEM's scratch per form:** every partially assembled form keeps two element-by-
  element vectors for the run, about 3 GB over the mass, momentum, divergence and
  convection forms. MFEM internals.
- **The FGMRES basis:** an outer iteration on the pressure alone would make its vectors
  ~10× smaller, a much larger change.
