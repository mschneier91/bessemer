# Throughput: measurement, results and decisions

The record of the throughput work (from 2026-10-14): how throughput is measured, what the
solver spends its time on, and every change kept or rejected, with its measurement. It
follows the maintainer's throughput plan (tiers 0 to 5), adapted to bessemer (below).
Headline metric: **DOF-steps per second per GPU** (velocity plus pressure true dofs times
steps, per second of step wall time) on one H100 80 GB, Bridges-2, CUDA backend.

## Where it stands (2026-10-15, 17:00)

One H100. Taylor-Green 32^3 Q3/Q2 from t = 0 (IMEX; 20 + 50 or 10 + 30 steps, 3 repeats)
and the turbulent channel at Re_tau 180 replayed from t = 60 (OIFS CFL 2, 3 + 30 steps),
each row adding to the previous one:

| | TGV 32^3 s/step | Speed-up | Channel s/step | Speed-up |
|---|---|---|---|---|
| Before (main 0aabb9b; the channel's spin-up binary a6870f8), defaults | 1.44 | | 0.961 | |
| Code changes only, defaults unchanged (1a specializations, fused kernel) | 1.08 | 1.33x | 0.873 | 1.10x |
| + the solver settings (project k 8, rhs tolerance, `n_inner` 3, `rtol` 1e-6) | 0.101 | 14.2x | 0.503 | 1.91x |
| + OIFS: Dirichlet data reused across the RK stages (1d) | | | 0.189 | 5.1x |
| + unrolled viscous and divergence kernels (1a) | 0.058 | 24.8x | 0.157 | 6.1x |
| + OIFS: deterministic classification, one convection kernel per stage (1d) | | | 0.082 | 11.7x |
| + fused kernel with a thread per point, host overhead | **0.043** | **33.8x** | 0.058 | 16.5x |
| + `time.oifs_cfl: 1.5` (opt-in substep CFL) | | | **0.050** | **19.2x** |

The solver settings and `time.oifs_cfl` are opt-in deck keys (the defaults are unchanged,
pending the human); everything else is code. Gates: the expensive Taylor-Green gate passes
after every kernel change (the dissipation curve to t = 20 within 5.9e-5 of the default
run's peak with all of it; that run takes 127 s against 3,390 s); the channel statistics
gate passes for the solver settings (t = 60 to 120) and every OIFS change was checked from
t = 60 to 75 against MKM (Re_tau 177.4 to 178.7, U+ within 7.7 to 9.0 %, the sampling
noise of 15 units). Details in the sections below.

**Production check, TGV 64^3 to t = 20** (one H100, CFL 0.3, job 49220279), against the
run of 2026-10-14 (`runs/tgv_re1600/tgv_n64_1gpu`):

| | Wall time | Outer its/step | s/step | Peak dissipation |
|---|---|---|---|---|
| 2026-10-14, main | 12 h 44 min (45,847 s) | 13.7 | 13.9 | 3.10846 at t = 8.892 |
| now, `n_inner` 3, `rtol` 1e-6, project k 8, rhs | **1 h 01 min** (3,672 s) | 1.8 | 1.07 | 3.10843 at t = 8.892 |

12.5x; the dissipation curve within 2.4e-4 of its peak, the energy at t = 20 within 1.7e-4
(`runs/tgv_re1600/tgv_n64_fast`).

## Adaptations of the plan (2026-10-14)

- bessemer solves velocity and pressure **together**: one FGMRES on the block system,
  preconditioned by a block preconditioner (Cahouet-Chabard Schur, PCG with point-block
  Jacobi on the velocity block). So the replay harness re-solves the block system, and
  MINRES (2c) does not apply: the inner solves run to a tolerance, which needs FGMRES.
- Already in place: the Taylor-Green case and its reference (`benchmarks/tgv_re1600`),
  the phase profiler (`output.profile: sync`), the semi-implicit rotational convection
  and OIFS (Tier 4 is a comparison of what exists).
- Deferred by the human (2026-10-14): all single-precision work (1b and Krylov vectors in
  float), libCEED (1a step 3), Tier 5 (needs more than two GPUs), and the periodic-hill
  and Re_tau 550 channel cases. The channel at Re_tau 180 is being spun up (job 49216608,
  `runs/channel180`).

## The harness

- **`bench/throughput/bench.py`**: runs a case at fixed steps and reports, per size, the
  median over repeats of the mean step wall time, outer iterations per step and
  DOF-steps/s, plus the cheap correctness quantities (final kinetic energy, largest
  divergence norm, kinetic-energy budget closure) and, with `--profile sync`, the phase
  breakdown from the profile CSV. `compare` flags a change outside the 4 % noise band and
  prints the energy and divergence differences; `--log` appends to a decision log.
  Standard library only.
- **`bench/cases/tgv_re1600`**: Taylor-Green Re 1600, Q3/Q2, IMEX BDF2, **fixed** dt =
  0.5/N (CFL about 0.29 at t = 0), so every change does the same work per step.
  `gate.json` holds the gate tolerances.
- **Replay** (`--restart <checkpoint>`): the run continues a checkpoint at its next step
  size, fixed, and so re-solves the same systems with whatever solver options are given.
  A checkpoint holds the BDF history, so the restarted step is the original one. Frozen
  checkpoint: TGV 32^3 at t = 9 (`runs/bench/cases/tgv_n32_t9/chk` beside the repo;
  meta and md5 in the job log of 49216992), near the dissipation peak.
- **Kernel census**: `--wrap "nsys profile -t cuda,nvtx ..."`; the profiler's regions are
  NVTX ranges on a GPU (`Profiler::SetMarkers`), so `nsys stats --report nvtx_kern_sum`
  attributes kernel time to the solver's phases without the synchronizing profiler.
- Protocol: 20 warm-up and 50 measured steps, 3 repeats, median (replay: 2 and 30, 1).

## Tier 0 results (2026-10-14/15, jobs 49216707)

**Baseline, TGV 32^3** (2.92 M DOF): 1.44 s per step, 11.6 outer iterations,
2.03e6 DOF-steps/s, spread 1 % over three repeats.

**Saturation curve** (one H100, 5 + 10 steps, 1 repeat):

| N^3 | DOF | s/step | DOF-steps/s |
|---|---|---|---|
| 16 | 0.36 M | 0.30 | 1.23e6 |
| 24 | 1.23 M | 0.66 | 1.88e6 |
| 32 | 2.92 M | 1.27 | 2.29e6 |
| 40 | 5.70 M | 2.56 | 2.23e6 |
| 48 | 9.84 M | 4.32 | 2.28e6 |
| 56 | 15.6 M | 6.27 | 2.49e6 |
| 64 | 23.3 M | 9.33 | 2.50e6 |
| 72 | 33.2 M | 13.2 | 2.51e6 |
| 80 | 45.6 M | 16.3 | 2.80e6 |

The knee is near **3 M DOF per GPU** (32^3): 92 % of the saturated throughput there, 75 %
at 1.2 M, 49 % at 0.36 M. The benchmarks use 32^3.

**Where the time goes** (Nsight Systems, TGV 32^3, 13 steps; GPU busy about 90 % of the
wall time; kernel time by NVTX phase, inclusive):

| Phase | Share |
|---|---|
| FGMRES solve | 97 % |
| ... Schur preconditioner (Cahouet-Chabard) | 56 % |
| ...... inner CG on B M_v^-1 B^T, 10 iterations per apply | 50 % |
| ......... velocity mass inverse (Chebyshev, order 4) | 23 % |
| ......... B and B^T | 27 % |
| ...... pressure LOR-AMG V-cycles | 6 % |
| ... velocity block (PCG with point-block Jacobi) | 32 % |
| ... FGMRES's own applies and glue | 9 % |

**By kernel**: MFEM's vector mass apply 23.8 %, our grad-div apply 17.3 %, MFEM's
divergence transpose 15.4 % and divergence 14.0 %, our compact diffusion 13.4 %, hypre's
SpMV about 5 %. **The vector mass and both divergence kernels ran MFEM's generic
versions** (sizes at run time): 53 % of the kernel time (Tier 1a, below).

**Glue**: 61,000 `cudaMemcpy` calls in 13 steps (27 us each, about 8 % of the wall time),
52,000 of them device-to-device, and 57,500 inside the Schur preconditioner: 36,000 in the
hypre V-cycles (about 27 per cycle) and 14,000 in MFEM's Chebyshev mass inverse (10 per
apply). About 20,000 kernel launches per step.

## Tier 2: the initial guess and the tolerance (2a, 2b)

**The finding.** FGMRES's tolerance was relative to the initial residual (`solver.rtol`
1e-10, MFEM's `rel_tol`), so the warm start (the previous step's solution) saved no
iterations: a better guess only made the answer more accurate. Hence two options
(defaults unchanged so far):

- `solver.tolerance_reference: rhs`: stop at `rtol` times the right-hand side's norm.
- `solver.initial_guess: extrapolate` (linear in time) or `project` (least squares over
  the last `solver.projection_k` solutions; Fischer's projection with the residual
  minimized in the 2-norm, since the block system is indefinite). Cost: one block-operator
  application and 2k inner products per step; memory 2k system-sized vectors (2.9 GB at
  64^3 for k = 8).

**Measured on one H100** (job 49216992; energy differences against the default at most
2e-11 relative, divergence unchanged):

| Case | Variant | Outer its/step | DOF-steps/s | Speed-up |
|---|---|---|---|---|
| TGV 32^3 from t = 0 (20 + 50 steps, 3 repeats) | default | 11.6 | 2.03e6 | |
| | rhs | 7.5 | 3.18e6 | +56 % |
| | rhs + extrapolate | 5.0 | 4.61e6 | +127 % |
| | rhs + project k 8 | 3.7 | 5.89e6 | +190 % |
| | rhs + project k 16 | 3.3 | 6.47e6 | +219 % |
| TGV 32^3 replayed from t = 9 (30 steps) | default | 15.0 | 1.39e6 | |
| | rhs | 12.0 | 1.75e6 | +26 % |
| | rhs + extrapolate | 10.8 | 1.91e6 | +37 % |
| | rhs + project k 4 | 10.4 | 1.98e6 | +42 % |
| | rhs + project k 8 | 9.1 | 2.25e6 | +61 % |
| | rhs + project k 16 | 8.1 | 2.52e6 | +81 % |
| TGV 64^3 replayed from t = 19 (12 steps) | default | 15.0 | 1.39e6 | |
| | rhs + project k 8 | 5.3 | 3.70e6 | +165 % |

On the CPU (TGV 12^3, 8 ranks, job 49216916): 13.0 -> 8.9 (rhs), 7.4 (extrapolate), 5.2
(project k 8) outer iterations per step.

## Tier 1a: MFEM's kernels at our sizes

`AddKernelSpecializations` (`src/operators/kernel_specializations.*`) registers, through
MFEM's public kernel registries, the compile-time-sized vector mass (k+1, k+1) and vector
divergence and transpose (k+1, k, k+2) kernels for velocity orders 2 to 4, 2D and 3D;
`StokesOperator` calls it before assembling. Same code with fixed sizes: **bit-identical
on the GPU** (all 12 tested cases), within 3e-16 on the CPU (the host compiler vectorizes
differently).

TGV 32^3 from t = 0 (job 49217262): **1.44 -> 1.15 s per step, +24 %**, same iterations.
Per apply: vector mass 652 -> 152 us, divergence 1311 -> 1119 us, transpose 1331 ->
1086 us. Kernel time after it: grad-div 22.6 %, compact diffusion 17.4 %, divergence
transpose 16.3 %, divergence 15.5 %, mass 7.2 %, hypre 6 %.

## Tier 3: the Cahouet-Chabard inner solve

The Schur preconditioner solves B M_v^-1 B^T with a fixed 10 CG iterations
(`solver.n_inner`), each a B^T, a Chebyshev M_v^-1 (order 4) and a B application, with one
pressure LOR-AMG V-cycle as its preconditioner. Replayed from t = 9 on top of the
projection (k 8, rhs; job 49217262):

| Variant | Outer its/step | DOF-steps/s | vs reference | dE/E |
|---|---|---|---|---|
| reference (`n_inner` 10) | 9.13 | 2.75e6 | | |
| `n_inner` 5 | 9.20 | 3.55e6 | +30 % | 4e-13 |
| `n_inner` 3 | 9.33 | 4.03e6 | +47 % | 2e-12 |
| `n_inner` 3, 2 V-cycles | 9.33 | 3.90e6 | +42 % | 2e-12 |
| `discretization.mass: collocated` | 7.93 | 3.96e6 | +44 % | 2.2e-4 |
| collocated, `n_inner` 5 | 8.03 | 5.14e6 | +87 % | 2.2e-4 |
| collocated, `n_inner` 3 | 8.33 | 5.62e6 | +105 % | 2.2e-4 |

The inner solve was far more accurate than the outer iteration needs: 3 inner iterations
cost 0.2 outer iterations per step and save 47 %. The collocated (GLL, diagonal) velocity
mass makes M_v^-1 exact and cheap, but it is a different discretization (the energy moves
by 2e-4 in 30 steps); it is OIFS's default already. Not a default change without the
human's decision.

**Expensive gate** (job 49217952): Taylor-Green 32^3 at CFL 0.25 to t = 20 with
`initial_guess: project` (k 8) and `tolerance_reference: rhs`, against the earlier run
with the defaults (`runs/tgv_re1600/tgv_n32_cfl025`): the dissipation curves agree to
**2.1e-9 of the peak**, the energy at t = 20 to 3.1e-9 (`bench/throughput/gate_tgv.py`,
tolerance 2 %). Same 1924 steps: **13.7 -> 6.0 outer iterations per step, 1.76 -> 0.67 s
per step, 3390 -> 1304 s for the run** (this run also has Tier 1a).

**The outer tolerance (2b)**, replayed from t = 9 on top of the projection (k 8, rhs):

| `solver.rtol` | Outer its/step | Speed-up | dE/E (30 steps) |
|---|---|---|---|
| 1e-10 (default) | 9.13 | | |
| 1e-8 | 5.40 | +60 % | 4.5e-11 |
| 1e-6 | 2.60 | +205 % | 9.6e-10 |

The velocity block's inner tolerance (`solver.a_pcg_rtol`, 0.01) is right where it is:
0.03 costs 6 %, 0.1 costs 19 % (more outer iterations).

## Tier 1: the viscous and grad-div terms in one kernel

`discretization.fuse_grad_div` (default true): `LeanVectorDiffusionIntegrator::
FuseGradDiv` applies the grad-div term in the compact diffusion kernel. One pass computes
every component's reference gradients and grad-div's scalar t = alpha div u (adj J), the
second forms each component's flux O grad u_c + t adj(J)[:, c] and contracts it once;
`GradDivIntegrator` keeps its data for the diagonal and the point-block Jacobi blocks and
skips its own apply. Results agree with the two separate applies to 2.5e-16 (test L3,
2D and 3D, orders 2 and 3). TGV 32^3 from t = 0, same binary (job 49218056): **1.159 ->
1.079 s per step, +7.4 %**; per apply 3.02 ms against 2.23 + 1.71 ms.

## Decision log

| # | Change | Kept | Measured (one H100) | Job |
|---|---|---|---|---|
| 1 | NVTX ranges, `bench.py`, replay, census (Tier 0) | yes | (instrumentation) | 49216707 |
| 2 | `tolerance_reference: rhs` (option) | yes, opt-in | TGV 32^3: +56 % | 49216992 |
| 3 | `initial_guess: extrapolate` (option) | yes, opt-in | with 2: +127 % | 49216992 |
| 4 | `initial_guess: project` (option) | yes, opt-in | with 2, k 8: +190 %; 64^3: +165 %; gate passes | 49216992, 49217952 |
| 5 | MFEM kernel specializations (1a) | yes, always | +24 %, bit-identical on the GPU | 49217262 |
| 6 | Fused viscous + grad-div kernel | yes, default (`fuse_grad_div`) | +7.4 % | 49218056 |
| 7 | `a_pcg_rtol` 0.03 / 0.1 | no | -6 % / -19 % | 49217952 |
| 8 | `n_inner` 10 -> 3 / 1 (option) | yes, opt-in (gate passes) | +54 % / +73 % (replay t = 9) | 49218488 |
| 9 | `mass: collocated` (discretization) | not a default | +44 %, changes the answer by 2e-4 | 49217262 |
| 10 | `rtol` 1e-8 / 1e-6 with 2 + 4 | yes, opt-in (TGV gate passes) | +60 % / +205 % (replay) | 49217952, 49219918 |
| 11 | `pc_quadrature: gll_collocated` | no | -7 to -10 % (more outer its) | 49218488 |
| 12 | velocity block / block shape variants | no (defaults best) | -1 to -68 % | 49219918 |
| 13 | OIFS Dirichlet data reused across the RK stages (1d) | yes | channel +166 % | 49233522 |
| 14 | Unrolled viscous kernels (1a) | yes | TGV +41 % | 49236242 |
| 15 | Unrolled divergence kernels in MFEM's tables (1a) | yes | TGV +21 %, channel +21 % | 49242191 |
| 16 | Diagnostics' flow sums, a thread per point (1e) | yes | 4.0 -> 0.56 ms per call | 49244351 |
| 17 | Deterministic OIFS classification | yes (human) | channel +36 % | 49244351 |
| 18 | OIFS convection forms re-assembled in place | no | changed MMS results, crashes | CPU |
| 19 | One OIFS convection kernel per RK stage | yes | channel +38 % | 49253619 |
| 20 | `time.oifs_cfl` 1.5 | yes, opt-in | channel +28 % (then), no measurable accuracy change | 49251387 |
| 21 | Fused kernel: recompute / stash variants | no | 255 registers or a stack (ptxas) | - |
| 22 | Fused kernel with a thread per point | yes | TGV +11 % | 49255498 |
| 23 | Host overhead (persistent FGMRES, work vectors, caches) | yes | channel +45 %, TGV +34 % (with 22) | 49256543 |
| 24 | Rotational form | deferred (human) | energy loss on the channel | 49242722, 49245661 |
| 25 | Fused restriction scatter | no (human: no nondeterminism) | needs atomics | - |
| - | 1b single precision, libCEED, Tier 5 | deferred (human) | | |

## Tier 3, continued: fewer inner iterations; PC-only GLL masses rejected

Replayed from t = 9 with the projection (k 8, rhs) and the fused kernel (job 49218488):

| Variant | Outer its/step | DOF-steps/s | vs `n_inner` 10 | dE/E |
|---|---|---|---|---|
| `n_inner` 10 | 9.13 | 3.02e6 | | |
| `n_inner` 3 | 9.33 | 4.65e6 | +54 % | 2e-12 |
| `n_inner` 2 | 9.63 | 4.96e6 | +64 % | 5e-12 |
| `n_inner` 1 | 10.27 | 5.21e6 | +73 % | 1e-13 |
| `pc_quadrature: gll_collocated`, `n_inner` 3 | 15.67 | 2.76e6 | -9 % | 2e-12 |
| `pc_quadrature: gll_collocated`, `n_inner` 5 | 13.63 | 2.79e6 | -7 % | 3e-12 |

The PC-only GLL masses (`solver.pc_quadrature`, already a deck key) make M_v^-1 a diagonal
multiply but a worse preconditioner for the consistent-mass system: more outer iterations
than they save. Rejected.

**Expensive gates** (TGV 32^3, CFL 0.25, to t = 20, against the default run; job 49218488):

| Configuration | Dissipation curve difference / peak | E(20) difference | Its/step | Run wall |
|---|---|---|---|---|
| default (reference run) | | | 13.7 | 3390 s |
| project k 8 + rhs | 2.1e-9 PASS | 3.1e-9 | 6.0 | 1304 s |
| + `n_inner` 3 | 2.2e-9 PASS | 3.5e-9 | 6.4 | 784 s |
| + `n_inner` 3 + PC GLL masses | 9.3e-10 PASS | 1.4e-10 | 9.1 | 1002 s |
| ... + `rtol` 1e-8 | 6.5e-7 PASS | 4.1e-7 | 5.1 | 615 s |
| ... + `rtol` 1e-6 | 6.8e-5 PASS | 7.8e-5 | 2.2 | 317 s |
| `mass: collocated` + project + rhs + `n_inner` 3 | 4.3e-2 FAIL | 1.3e-1 | 6.3 | 660 s |

The collocated (lumped GLL) velocity mass changes this under-resolved flow by 4 % in the
dissipation and 13 % in the energy at t = 20: a discretization choice, not a solver
setting, and not adopted.

**Headline** (all kept code, project k 8, rhs, `n_inner` 3):
TGV 32^3 from t = 0: **1.44 -> 0.248 s per step, 5.8x** (2.03e6 -> 1.18e7 DOF-steps/s);
TGV 64^3 replayed from t = 19: **16.75 -> 3.16 s per step, 5.3x** (1.39e6 -> 7.40e6).

## Tier 3a: the velocity block and the block shape (kept as they are)

Replayed from t = 9 on top of project k 8, rhs, `n_inner` 3 (job 49219918):

| Variant | Outer its/step | vs reference |
|---|---|---|
| reference (`a_pc: jacobi_pcg`, point-block Jacobi, `a_pcg_rtol` 0.01, `block_shape: upper`) | 9.33 | |
| `a_pc: jacobi_chebyshev` | 29.9 | -46 % |
| `a_pc: loramg` (with or without `amg_reuse`) | 60.3 | -68 % |
| `a_pcg_pc: jacobi` | 9.33 | -1 % (noise) |
| `a_pcg_rtol` 0.003 | 8.77 | -10 % |
| `block_shape: lower` | 11.7 | -25 % |
| `block_shape: diag` | 22.3 | -66 % |
| `projection_k` 16 | 9.33 | 0 % |

## The fastest gated configuration (2b, with Tier 3)

`tolerance_reference: rhs`, `initial_guess: project`, `projection_k` 8, `rtol` 1e-6 and
`n_inner` 1 to 3 (job 49219918):

| Case | Default | Fast | Speed-up |
|---|---|---|---|
| TGV 32^3 from t = 0, `n_inner` 3 | 1.44 s/step, 11.6 its | 0.101 s/step, 1.28 its | **14.2x** |
| TGV 32^3 from t = 0, `n_inner` 1 | | 0.113 s/step, 1.80 its | 12.8x |
| TGV 64^3 replayed from t = 19, `n_inner` 1 | 16.75 s/step, 15 its | 1.15 s/step, 2.0 its | **14.6x** |
| Gate TGV 32^3 to t = 20, `n_inner` 1 | 3390 s | 287 s; dissipation 1.1e-4, E(20) 1.4e-4: PASS | 11.8x |
| Gate TGV 32^3 to t = 20, `n_inner` 3 | 3390 s | 326 s; dissipation 5.0e-5, E(20) 1.1e-4: PASS | 10.4x |
| Channel Re_tau 180 replayed from t = 60 (OIFS, walls) | 0.89 s/step, 14 its | 0.50 s/step, 4.0 its | 1.77x |

Channel, step by step: project + rhs +16 %, + `n_inner` 3 +36 %, + `rtol` 1e-6 +77 %. The
channel gains less: an OIFS step (CFL 2) changes the solution more than an IMEX step,
and its convection substeps are work the solver changes do not touch.

`rtol` 1e-6 is relative to the right-hand side; on Taylor-Green its effect (1e-4 in the
energy at t = 20) is far below the discretization's, but it has been gated on Taylor-Green
only. Before it becomes a default it should pass the channel statistics gate too.

### Recommended deck settings (opt-in; the defaults are unchanged)

```yaml
solver:
  tolerance_reference: rhs   # rtol relative to the right-hand side
  initial_guess: project     # least squares over earlier solutions
  projection_k: 8            # 2 x 8 system-sized vectors (2.9 GB at 64^3)
  n_inner: 3                 # Cahouet-Chabard inner CG (default 10)
  rtol: 1.0e-8               # conservative; 1.0e-6 passed the Taylor-Green gate
```

## Tier 4: the time step, against the DNS

TGV Re 1600 to t = 20 with the fast solver settings (project k 8, rhs, `n_inner` 3,
`rtol` 1e-6), compared with the spectral DNS (`benchmarks/tgv_re1600/plot.py`; jobs
49220334, 49220279, earlier runs):

| Run | Steps | Run wall | Peak dissipation error | Largest curve error |
|---|---|---|---|---|
| 32^3 IMEX CFL 0.25, default solver | 1924 | 3390 s | 15.6 % | 16.8 % |
| 32^3 IMEX CFL 0.25, fast solver | 1924 | 326 s | 15.6 % | 16.8 % |
| 32^3 IMEX CFL 0.25, `mass: collocated` | 1982 | 660 s | 16.1 % | 19.6 % |
| 32^3 OIFS CFL 2, default solver | 259 | 492 s | 10.2 % | 14.3 % |
| 32^3 OIFS CFL 2, fast solver | 259 | 168 s | 10.2 % | 14.3 % |
| 32^3 rotational IMEX CFL 0.5, fast solver | 1165 | 371 s | 25.9 % | 30.3 % |
| 64^3 IMEX CFL 0.3, default solver (2026-10-14) | 3284 | 45,847 s | 2.5 % | 3.2 % |
| 64^3 IMEX CFL 0.3, fast solver | 3293 | 3,672 s | 2.5 % | 3.2 % |
| 64^3 OIFS CFL 2, fast solver | 499 | 2,078 s | 0.96 % | 5.6 % |

The fast solver settings change no answer (identical errors); they change the time. At
32^3, OIFS at CFL 2 is closer to the DNS than IMEX and, with the fast solver, 20x faster
than the default IMEX run; the semi-implicit rotational form at CFL 0.5 is the furthest.
(All three "fail" the gate against the IMEX run because the gate compares two runs of the
same scheme; against the DNS, OIFS is the better one at 32^3.) At 64^3 (job 49224583) it is a
trade-off: OIFS at CFL 2 has the peak closer to the DNS (0.96 % against 2.5 %) but a larger
deviation elsewhere on the curve (5.6 % against 3.2 %), in 35 min against IMEX's 61 min:
6.6x fewer steps at 3.8x the cost each (the convection substeps). IMEX stays the accurate
choice for this case (human 2026-10-12: IMEX, not OIFS, for it); OIFS with the fast solver
is the fast one. No run had a kinetic-energy increase.

**Saturation with the fast settings** (5 + 10 steps, from t = 0, job 49220334):

| N^3 | DOF | s/step | DOF-steps/s |
|---|---|---|---|
| 16 | 0.36 M | 0.024 | 1.55e7 |
| 24 | 1.23 M | 0.048 | 2.56e7 |
| 32 | 2.92 M | 0.103 | 2.84e7 |
| 40 | 5.70 M | 0.186 | 3.06e7 |
| 48 | 9.84 M | 0.318 | 3.10e7 |
| 64 | 23.3 M | 0.735 | 3.17e7 |
| 72 | 33.2 M | 1.03 | 3.24e7 |

About 12x the original saturated throughput (2.5e6); the knee stays near 1 to 3 M DOF per
GPU (80 to 90 % of the saturated rate at 24^3 to 32^3).

## The channel at Re_tau 180 (spin-up, job 49216608)

24 x 18 x 24 Q3/Q2 elements, OIFS CFL 2, t = 0 to 120 in 7 h 15 min on one H100 (28,628
steps, before the solver changes). Averaged over t = 60 to 120 (2,877 samples): **Re_tau
178.21 against 178.12** (0.05 %: the momentum balance is converged), bulk velocity 16.56
(the DNS's is about 15.7; the streamwise resolution, dx+ ~ 15.5, is coarse). Against Moser,
Kim & Mansour (fetched 2026-10-15 with the human's approval; `benchmarks/channel/plot.py`):
the mean velocity U+ within 8.6 % for y+ >= 5, and the largest absolute differences in wall
units u_rms 0.17, v_rms 0.10, w_rms 0.10, u'v' 0.066 (6 to 9 % of their peaks): the
coarse-mesh signature (the log-layer overshoot of the bulk velocity), with the momentum
balance exact. Frozen restart at t = 60 for the benchmarks:
`runs/bench/cases/chan180_t60`.

## The channel at Re_tau 180: before and after (jobs 49225563, 49219918, 49225579)

Replayed from the frozen t = 60 checkpoint (OIFS CFL 2, 3 + 30 steps, one H100), each row
adding to the previous one:

| | s/step | Outer its/step | DOF-steps/s | Speed-up |
|---|---|---|---|---|
| Spin-up binary (a6870f8, before the throughput work) | 0.961 | 14.0 | 9.79e5 | |
| Code changes only, defaults (1a, fused kernel) | 0.873 | 14.0 | 1.08e6 | 1.10x |
| + project k 8, rhs tolerance | 0.768 | 10.8 | 1.23e6 | 1.25x |
| + `n_inner` 3 | 0.658 | 12.4 | 1.43e6 | 1.46x |
| + `rtol` 1e-6 | 0.503 | 4.0 | 1.87e6 | 1.91x |
| + OIFS Dirichlet data reused in the substeps (1d, below) | 0.189 | 4.0 | 4.98e6 | 5.08x |
| + unrolled viscous and divergence kernels (1a, below) | 0.157 | 4.0 | 6.01e6 | 6.14x |
| + deterministic OIFS classification (1d) | 0.116 | 4.0 | 8.12e6 | 8.29x |
| + one OIFS convection kernel per stage (1d) | 0.082 | 4.0 | 1.14e7 | 11.7x |
| + per-point fused kernel, host overhead | 0.058 | 4.0 | 1.61e7 | 16.5x |
| + `time.oifs_cfl: 1.5` | **0.050** | 4.0 | 1.88e7 | **19.2x** |

The solver settings gave 1.9x; the rest came from OIFS's substeps, the kernels and the host
overhead (jobs 49233522, 49242191, 49244351, 49253619, 49256543). The outer iteration count
(4 per step against TGV's 1.3) is unchanged by all of it: the next item.

**Statistics gate (Tier 2 close, job 49225579).** The channel restarted at t = 60 with the
fast settings (project k 8, rhs, `n_inner` 3, `rtol` 1e-6) to t = 120, against the
spin-up's own averages over the same window and against Moser, Kim & Mansour:

| t = 60 to 120 | Re_tau | max U+ error (y+ >= 5) | u_rms | v_rms | w_rms | u'v' | its/step | s/step |
|---|---|---|---|---|---|---|---|---|
| spin-up (defaults) | 178.21 | 8.64 % | 0.170 | 0.098 | 0.098 | 0.066 | 13.8 | 0.914 |
| fast settings | 178.16 | 8.54 % | 0.162 | 0.100 | 0.110 | 0.065 | 4.0 | 0.469 |

(DNS Re_tau 178.12; the rms columns are the largest absolute differences from the DNS in
wall units.) Same statistics to within the sampling noise of a 60-unit window, in 1 h 54
min instead of about 3 h 39 min. **Gate passes.**

## Re-census with the fast settings (job 49225616)

**Taylor-Green 32^3** (0.101 s/step, 1.3 outer iterations): the GPU is the bottleneck
(kernel time about the step time). By kernel: the fused viscous + grad-div apply 49 %,
divergence transpose 10 %, divergence 10 %, the diagnostics' flow sums 5 % (`output.interval:
1` in the benchmark deck), vector mass 5 %, element restriction 4 %, the convection apply 2
%, hypre SpMV 2 %. 1,500 launches per step, hidden behind the kernels.

**Channel** (0.57 s/step): `oifs::advect` (the history advected by the RK4 substeps) is
**80 %** of the step, and the GPU is busy only about 20 % of it: 6,700 launches, 830
`cudaMemcpy` (137 us each) and about 100 `cudaMalloc` / `cudaFree` per step. The solve is
the remaining 20 %. Hence the next item.

## 1a, continued: the roofline, and registers on the stack (jobs 49233585, 49236242)

Nsight Compute on the production kernels inside the solver (TGV 32^3 Q3/Q2 with the fast
settings; the OIFS convection from the channel replay), per apply:

| Kernel | Time | DRAM | Local-memory traffic | DRAM, % of peak | FP64, % of peak | FLOP/byte |
|---|---|---|---|---|---|---|
| fused viscous + grad-div (ours) | 3.00 ms | 5.7 GB | 6.3 GB | 57 % | 2.2 % | 0.39 |
| MFEM vector divergence (4, 3, 5) | 1.15 ms | 2.1 GB | 2.3 GB | 56 % | 2.2 % | 0.39 |
| MFEM vector divergence transpose | 1.10 ms | 1.9 GB | 2.2 GB | 52 % | 2.3 % | 0.43 |
| MFEM vector mass (4, 4) | 0.17 ms | 0.15 GB | none | 27 % | 5.5 % | 2.0 |
| MFEM convection (4, 5), OIFS substeps | 0.05 ms | 0.05 GB | none | 26 % | 5.9 % | 2.3 |

(Peaks: 3.35 TB/s, 33.5 TFLOP/s FP64 without tensor cores. Local-memory traffic counted at
L1.) The mass and convection kernels are where a sum-factorized kernel should be: bound by
shared memory / L1 (60 % and 83 % of its throughput). The other three moved about ten times
the data their operators need, nearly all of it local memory.

**Cause** (`ptxas -v`): no register spills, but 720 to 792 bytes of **stack** per thread.
The 3D kernels built on MFEM's register tiles (`kernels::internal::vd_regs3d_t`, 3 x 3 x Q1D
values per thread) index them with a component loop that is not unrolled (MFEM's
`LoadDofs3d`, `Grad3d`, `GradTranspose3d` and `WriteDofs3d` loop over the components without
`MFEM_UNROLL`), so the compiler cannot keep them in registers. MFEM's own divergence kernels
have the same pattern.

**Fix, our kernels** (`lean_vector_diffusion.cpp`): `MFEM_UNROLL(VDIM)` on the component
loops of the 3D compact and fused kernels. No stack for velocity orders 2 to 4 (the fused
kernel at 248 registers, the compact one at 96). Fused kernel **3.00 -> 1.09 ms** (DRAM 5.7
-> 0.67 GB, no local traffic; 12 % occupancy, L1/shared 61 %); **TGV 32^3 fast 0.1017 ->
0.0721 s per step, +41 %**, same iterations (job 49236242). The fused-against-separate test
agrees to 2.5e-16.

**Fix, the divergence** (`kernel_specializations.cpp`): our copies of MFEM's 3D divergence
and transpose kernels with the component loops unrolled (the same arithmetic per
component), installed in MFEM's own dispatch tables under MFEM's keys (dim, trial d1d,
test d1d, q1d) for velocity orders 2 to 4, so `VectorDivergenceIntegrator` calls them. MFEM
exposes the table read-only (`GetDispatchTable`); the entry is written through a
`const_cast` of that (non-const) static. No stack, 126 to 168 registers.

Transpose **1.10 -> 0.47 ms**, divergence **1.12 -> 0.32 ms** per apply; **TGV 32^3 fast
0.0700 -> 0.0580 s per step (+21 %)**, the channel replay 0.189 -> 0.157 s (+21 %)
(job 49242191). Against MFEM's generic kernels (`kernel_specializations_test`, now orders
2 to 4): 2D bitwise, 3D within 5e-16 on the CPU.

**Both kernel changes**, TGV 32^3 fast: **0.1017 -> 0.0580 s per step (+75 %)**, same
iterations. **Expensive gate** (32^3 CFL 0.25 to t = 20, fast settings, job 49242191):
passes, the dissipation curve within 4.4e-5 of the default run's peak (2.5e-5 of the
earlier fast run's), the final energy within 8.6e-5; **186 s** for the run (the default
run of 2026-10-14: 3,390 s).

After them (Nsight Systems): the fused kernel 29 % of the kernel time (0.94 ms per apply,
now bound by L1/shared memory at 12 % occupancy: its 248 registers), the diagnostics'
flow sums 9 % (1e, below), vector mass 9 %, divergence transpose 7 %, divergence 5 %,
element restrictions 8 %. Still GPU-bound: kernel time about the step time.

## 1d: OIFS's substeps (2026-10-15)

The channel's OIFS step was 80 % `oifs::advect` with the GPU idle most of the time
(re-census above). Four changes, each measured on the channel replay (fast settings, one
H100) and each passing the OIFS tests:

| Change | Channel s/step | Speed-up | Job |
|---|---|---|---|
| (fast settings, unrolled kernels) | 0.503 | | |
| Dirichlet data reused across the RK stages when it does not change | 0.189 | +166 % | 49233522 |
| (the unrolled viscous and divergence kernels, 1a) | 0.157 | +21 % | 49242191 |
| Deterministic inflow/outflow classification (human decision, below) | 0.116 | +36 % | 49244351 |
| One convection kernel per RK stage, the winds combined at the points | 0.082 | +38 % | 49253619 |
| `time.oifs_cfl: 1.5` (opt-in; default 0.5) | see below | +28 % at 0.114 | 49251387 |

**Dirichlet data.** At every RK stage (about 85 per channel step) the Dirichlet data was
projected on the host into a whole velocity grid function and gathered: a device round
trip of the velocity each time. Now it is evaluated at three times of the step's window
(collective, exact comparison) and reused when it does not change (no-slip walls, a
constant inflow); constant coefficients (`BoundaryConditions::DirichletIsConstant`) are
projected once per space. Outputs unchanged.

**The classification** (where the substeps impose the Dirichlet data: inflow and
tangential nodes) evaluated the wind at every boundary node through its element on the
host, every step, and tested `w.n <= 1e-10 |w| |n|`. On a no-slip wall that evaluated
`w` is roundoff, so its sign decided whether a wall node was imposed. **Human (2026-10-15):
make it deterministic.** The wind is read from the node's dofs and a node is free only
where its outward normal velocity exceeds 1e-10 of the largest wind on the boundary; zero
data with a zero wind on every Dirichlet dof skips the loop. The OIFS MMS tests give the
same numbers as before (the old evaluation imposed every wall node there too).

**One convection kernel per stage** (`time/oifs_convection`, `OifsConvection`). The
substeps applied MFEM's ConvectionIntegrator once per component and wind: nine applies
per RK4 stage, about 600 per channel step. The form is linear in the wind, so each wind's
quadrature data (W adj(J) u, as MFEM's `PAConvectionSetup3D`) is stored once per step and
one kernel per stage combines the winds with the stage's weights at each point and
applies the result to all three components (MFEM's `SmemPAConvectionApply3D` structure;
no stack, 46-56 registers). Against MFEM's per-component applies: 2-4e-16
(`oifs_convection_test`, orders 2 to 4). 3D at orders 2 to 4; 2D keeps MFEM's forms.
Channel from t = 60 to 75: Re_tau 178.41, U+ within 8.2 % of MKM (before: 177.41, 9.0 %).

**Substep CFL** (`time.oifs_cfl`, the CFL number of each RK4 substep; default 0.5):

| Substep CFL | Channel replay s/step | Channel Re_tau / max U+ error | TGV 32^3 peak dissipation error vs DNS | TGV run |
|---|---|---|---|---|
| 0.5 | 0.114 | 177.4 / 9.0 % | 10.24 % | 81 s |
| 1.0 | 0.097 (+17 %) | 178.5 / 7.8 % | 10.20 % | 71 s |
| 1.5 | 0.089 (+28 %) | 178.7 / 7.7 % | 10.21 % | 69 s |

The substeps' own error is far below OIFS's splitting error: 1.5 costs nothing measurable
(channel statistics within the sampling noise of a 15-unit window; TGV OIFS CFL 2 against
the DNS unchanged to 0.04 %). Measured before the combined kernel, which shrinks the
substeps' share; a default change is the human's decision.

**Tried and reverted:** re-assembling the convection forms in place (instead of new forms
at every step) changed the OIFS MMS results (`NseMms.OifsTemporalOrder2D` 9.1e-8 -> 1.1e-6
against the exact solution) and segfaulted nondeterministically; cause not found. The
combined kernel replaced those forms in 3D anyway.

## 1e: the diagnostics' flow sums

`flowsums::Sums3D` (kinetic energy, dissipation, divergence for `output.diagnostics`) ran
one thread per element over a direct Q^3 x D^3 sum: 4.0 ms per call at TGV 32^3 Q3, 9 % of
the kernel time with diagnostics every step (the TGV decks' setting). `Sums3DPoints` gives
every quadrature point a thread (the same per-point arithmetic, the element's sums in the
same order): **4.03 -> 0.56 ms** (job 49244351). Outside the step timer (the TGV benchmark's
s/step does not move; the run time does).

## 1a, continued: the fused kernel's occupancy

After the unrolling (above) the fused viscous + grad-div kernel kept every component's
gradient for a column of Q1D points per thread: 248 registers, 12 % occupancy, and the
compiler would not reuse slots when the gradients were recomputed or stashed (variants
measured with ptxas: 255 registers, or a stack). `FusedApply3DPoints` gives each
quadrature point a thread (Q1D^3 per element, the gradients in shared memory, MFEM's
convection-kernel structure): **56 registers, 55 % occupancy, 1.09 -> 0.65 ms per apply**,
now limited by shared-memory throughput (94 % of L1/shared). TGV 32^3 fast **0.0579 ->
0.0521 s/step (+11 %)** (job 49255498). The fused-against-separate test agrees to 1.7e-16.
Registered for the compile-time sizes with Q1D <= 7 (static shared memory <= 48 KB);
(7, 8) and the runtime-size fallback keep the column kernel.

## MFEM kernel audit (fixes held until MFEM 4.10, human 2026-10-15)

`cuobjdump --dump-resource-usage` on libmfem and run_case, crossed with the census
(`~/bessemer_jobs/audit.sh`, `audit.py`): 1,169 of 3,591 kernels have a stack frame, nearly
all generic fallbacks or integrators the solver does not use. Of the kernels that run:
MFEM's nonlinear convection apply (IMEX, `SmemPAConvectionNLApply3D<4,5>`) has 960 bytes
of stack: **4.5 % of TGV's kernel time** (1.2 % on the channel); the forcing's linear-form
assembly 32 bytes (1 %); the LOR assembly for the pressure AMG up to 4,800 bytes (setup,
<= 0.4 %). Everything else that runs is clean. hypre's sparse products (cuSPARSE, about 6
% of TGV) are outside this audit. After the upgrade: re-run the audit; the candidate is the
IMEX convection kernel.

## Tier 4 on the channel: the time step (human 2026-10-15: the channel Re_tau 180 instead of the 2D square cylinder)

Each probe restarted from the frozen t = 60 checkpoint (its first dt scaled to the target)
and ran to t = 75 with the fast solver settings, statistics over 60..75 (job 49242722; a
first attempt, 49237210, ran its first probe on the CPU: the device setting was missing
from the probe command):

| Treatment | CFL | Result | Re_tau | max U+ error | wall per time unit |
|---|---|---|---|---|---|
| OIFS | 2 | stable, statistics match | 178.0 | 8.2 % | 39 s |
| OIFS | 4 | stable, statistics drift (u_bulk 15.2 vs 16.5, u_rms off by 1.1) | 179.4 | 17.5 % | 24 s |
| OIFS | 8 | blew up at t = 61 (NaN: the abort is the multistep coefficients' ordering check) | | | |
| IMEX | 0.5 | stable; v_rms, w_rms off by 0.8 (noise near the limit) | 183.8 | 1.8 % | 94 s |
| IMEX | 1 | the step collapsed 10x (growing velocity); 50-minute probe limit | | | |
| rotational | 0.5 | energy loss: KE 5690 -> 830 in 15 units | 116 | 23 % | |

**OIFS at CFL 2 is the throughput choice** with matched statistics: 2.4x cheaper per unit of
simulated time than IMEX's best stable setting; larger steps degrade the statistics before
they go unstable. **The rotational form fails on the channel** with the default solver
settings too (t = 60 to 63: KE 1958 against 1899 with the fast settings; the divergence norm
jumps 5x on the first step), so it is not a solver-tolerance effect. **Human: abandoned for
now, a separate thread** (first step then: a fresh-start rotational channel run, to separate
the scheme from the restart). An abort message for NaN velocities (now: the multistep
ordering check) is a small open item.

## Plan items not done, and why

| Item | Status | Reason |
|---|---|---|
| 1b single precision (AMG, Krylov vectors) | deferred | human (2026-10-14) |
| 1a libCEED | deferred | human (2026-10-14); a dFEM probe follows the MFEM 4.10 upgrade instead |
| 1c Krylov launch audit | absorbed | TGV is GPU-bound (launches hidden); the channel's host overhead was treated directly (below) |
| AMG settings (V-cycles, coarsening, strength) | deferred | human ("an optimization for another day") |
| Tier 5 multi-GPU | deferred | human: more than two GPUs needed |
| periodic hill | deferred | human |
| square cylinder Tier 4 | replaced | human: the channel instead (no GPUs for 2D) |
| rotational form | deferred | fails on the channel (above); a separate thread |

## Host overhead (human 2026-10-15: "strictly free, do that for sure")

An attribution of the channel step's CUDA runtime calls to the innermost profiler region
(Nsight Systems export, `~/bessemer_jobs/ho_attr.py`) found about 23 ms of an 86 ms step in
`cudaMalloc` / `cudaFree` (each free synchronizes the device) and host round trips:

| Where | Per step before | Fix |
|---|---|---|
| OIFS: the winds copied from the history | 5 mallocs, 3 frees, 10.7 ms | borrowed pointers |
| FGMRES (MFEM allocates its Krylov vectors in every Mult) | 10 / 10, 2.4 ms | `PersistentFGMRESSolver`: MFEM's Mult line for line, the vectors kept (bitwise the same iterates) |
| CFL rate, once per wind | 9 / 9, 1.9 ms | `ConvectiveCfl::MaxRate`: one Jacobian evaluation |
| OIFS boundary values, the step's RHS, solution, commit | ~10 / 10 + 8 copies | kept work vectors; the history level that drops out lends its memory (`Vector::Swap`; MFEM's Vector has no move) |
| the forcing (a new linear form every step) | 5 / 4 | a constant coefficient assembled once |
| the solve's block vectors; the Dirichlet projection per solve | 9 / 9 + 9 copies | kept; constant (no-slip) data cached on the essential dofs |

Results unchanged (OIFS MMS numbers identical; fast tier CPU np 1 and 2, GPU np 1). With
the per-point fused kernel (job 49256543): **channel 0.0726 -> 0.0501 s/step (+45 %,
substep CFL 1.5), TGV 32^3 0.0571 -> 0.0426 (+34 %)** -- TGV is GPU-bound, but the frees'
synchronizations had kept the host from running ahead. Expensive TGV gate passes (5.9e-5);
the channel from t = 60 to 75 at substep CFL 1.5: Re_tau 178.18, U+ within 7.7 %. Left: about
26 frees per step outside any region, a few in the solve and the initial guess (being
traced), and hypre's AMG V-cycles (about 2,000 launches and 400 small copies per channel
step; AMG settings deferred by the human).

**Reproducibility (human 2026-10-15):** results must stay bitwise reproducible run to run in
bessemer's own code -- no atomics or nondeterministic sums (so no fused restriction
scatter); results moving at roundoff against an older binary are fine; library-level GPU
nondeterminism (hypre, cuSPARSE) is accepted.
