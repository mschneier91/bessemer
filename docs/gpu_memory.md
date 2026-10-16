# GPU memory: what was recovered, and what is left

The record of the 2026-10-13 and 2026-10-14 memory work: each change, numbered, with what
it saved, and what is left. Measurements are Taylor–Green (Re 1600) with the case defaults
(Q3/Q2, IMEX BDF2, grad-div γ = 2h, Cahouet–Chabard Schur PC, PCG with point-block Jacobi
on the velocity block), about ten steps per run, one NVIDIA H100 80 GB on Bridges-2, CUDA
backend. Memory is the device's peak in use (`nvidia-smi`, and the profiler gauge of
change 1), in units of 1000 MiB; step times are the mean from step 3 on, with the
synchronizing profiler on.

## Result

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

## What is left

Nothing large. At 64³:

- **The mass operator on the 5-point rule** (its Jacobians shared with the other
  forms, which all use that rule; after change 16 the mass's 4-point rule is the only
  other one with cached Jacobians): about 1.15 GB, but the mass apply would do 5³ instead of 4³ points
  per element, roughly 5–10 % on the velocity solve. Left (human, 2026-10-14, after
  asking about the throughput trade-off).
- **Krylov vectors in single precision:** about 2.5 GB of the FGMRES basis. Not tried; it
  changes the outer solver's arithmetic.
- **Pressure AMG.** One level of aggressive coarsening (`solver.lp_agg_levels: 1`) saved
  2.1 GB at 64³ with the same outer iterations over 11 steps; the strength threshold
  saved nothing. Its first-application workspace (5.8 GB, change 15) is the largest
  single piece. Human: AMG options have trade-offs, "an optimization for another day".
- **MFEM's scratch per form:** every partially assembled form keeps two element-by-
  element vectors for the run, about 3 GB over the mass, momentum, divergence and
  convection forms. MFEM internals.
- **The FGMRES basis:** an outer iteration on the pressure alone would make its vectors
  ~10× smaller, a much larger change.
