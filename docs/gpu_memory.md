# GPU memory: what was recovered, and what is left

The record of the 2026-10-13 memory work: each change, numbered, with what it saved, and
the one real opportunity left. Measurements are Taylor–Green (Re 1600) with the case
defaults (Q3/Q2, IMEX BDF2, grad-div γ = 2h, Cahouet–Chabard Schur PC, Jacobi-PCG velocity
block), about ten steps per run, one NVIDIA H100 80 GB on Bridges-2, CUDA backend. Memory
is the device's peak in use (`nvidia-smi`, and the profiler gauge of change 1).

## Result

| | Largest mesh on one H100 | Velocity dofs | Peak at 32³ | Peak at 64³ |
|---|---|---|---|---|
| Before (morning) | 56³ | 14.2 M | 23.7 GB | out of memory (62 GB per GPU on two) |
| After changes 2–3 | 69³ | 26.6 M | 9.8 GB | 63.9 GB |
| After changes 4–5 | — | — | — | 51.8 GB |
| After changes 7–9 | **80³** | **41.5 M** | — | **43.5 GB** (38 GB after step 1) |

Per velocity dof: 5.2 kB at 56³ before, 1.95 kB at 80³ after. Single-GPU sweep after all
changes: 64³ 43.5 GB, 72³ 59.0, 76³ 68.0, 80³ 78.9, 81³ out of memory.

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

## Where the memory is now (64³, one H100, 43.5 GB peak)

| Piece | Memory | Note |
|---|---|---|
| Momentum operator | 9.6 GB | 6.75 GB of it MFEM's vector-diffusion PA data |
| Solve workspace | ~8 GB | mostly the pressure AMG's at its first apply (5.8 GB) |
| Convection | 4.5 GB | half the 5-point Jacobians, shared by every 5-point form; half its own PA data |
| Schur preconditioner setup | 4.1 GB | 77 % the pressure LOR-AMG hierarchy |
| FGMRES basis | ~5 GB at 14 iterations | transient, per solve |
| Divergence operator | 2.3 GB | |
| Mass operator | 1.9 GB | includes the 4-point Jacobians |
| CFL estimator | 1.2 GB | Jacobians at the GLL nodes |

## The last real opportunity: MFEM's vector-diffusion storage

`VectorDiffusionIntegrator::AssemblePA` stores, at every quadrature point, the full
dim × dim geometric matrix for **each** velocity component: 9 × 3 = 27 numbers per point
in 3D. For a scalar coefficient the three components share one symmetric matrix, so 6
would do. At 64³ the momentum operator's copy is 6.75 GB where 1.5 GB would suffice: about
**5.2 GB, 12 % of the peak**, which would move the single-GPU limit from 80³ to roughly
83–84³ (an extrapolation, not a measurement).

Taking it means storing the 6 numbers ourselves and writing the apply kernel that reads
them, i.e. replacing MFEM's own tuned, specialised vector-diffusion apply, the most
frequently called kernel in the solver. **Decided against (human, 2026-10-13: "that is
a big change I'm not really comfortable with since that is like the core mfem
library").** The better route is upstream: the same issue as change 2's write-up, a
compact storage for the scalar-coefficient case.

## Looked at and left

- **Pressure AMG.** One level of aggressive coarsening (`solver.lp_agg_levels: 1`) saved
  2.1 GB at 64³ with the same outer iterations over 11 steps; the strength threshold
  saved nothing. The 3.6 GB the AMG still allocates at its first apply is unexplained.
  Human: AMG options have trade-offs, "an optimization for another day".
- **Convection operator.** Nothing left over: the Jacobians are shared, the rest is
  MFEM's data.
- **Jacobians per quadrature rule** (~9 GB over four rules at 64³). Collocated mass would
  share the CFL estimator's rule (1.2 GB), but changes the discretisation.
- **The FGMRES basis.** Fewer outer iterations shrink it; an outer iteration on the
  pressure alone would make its vectors ~10× smaller, a much larger change.
