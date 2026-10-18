# Throughput: measurement, results and decisions

The record of the throughput work (from 2026-10-14): how throughput is measured, what the
solver spends its time on, and every change kept or rejected, with its measurement. It
follows the maintainer's throughput plan (tiers 0 to 5), adapted to bessemer (below).
Headline metric: **DOF-steps per second per GPU** (velocity plus pressure true dofs times
steps, per second of step wall time) on one H100 80 GB, Bridges-2, CUDA backend.

## Where it stands (2026-10-16)

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
| + `time.oifs_cfl: 1.5` (opt-in substep CFL) | | | 0.050 | 19.2x |
| + `n_inner` 2, host overhead round 2 (7d880f0, a173072) | 0.0381 | 37.8x | 0.0435 | 22.1x |
| + the constant-pressure projection on the device (747a294) | 0.0339 | 42.5x | 0.0382 | 25.2x |
| + step-relative `rtol` 1e-3 (b1bc283; the decks, 4440522), the CFL rate once per level (eb328eb), a new dt re-assembles only the mass term (9a36852) | **0.0314** | **45.9x** | **0.0259** | **37.1x** |

The solver settings and `time.oifs_cfl` are deck keys, set by the Taylor-Green and channel
decks (4440522); since 2026-10-16 the case-level defaults are the same configuration at `rtol`
1e-6 (see "Restarts with the projected guess, and the defaults"); `rtol` depends on the
problem (see "The step-relative tolerance").
Everything else is code. A production channel run (CFL-controlled steps, statistics every 5
steps) from t = 0 to 120 costs **0.0289 s of wall time per step**: the spin-up's 7 h 15 min
is **13.8 min** (job 49270033; "Final runs"). Gates: the expensive Taylor-Green gate passes
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
| `n_inner` 3, `rtol` 1e-6, project k 8, rhs (2026-10-15) | 1 h 01 min (3,672 s) | 1.8 | 1.07 | 3.10843 at t = 8.892 |
| **the deck** (step `rtol` 1e-3, project k 8, `n_inner` 2, CFL 0.25; 2026-10-16, job 49270034) | **17.8 min** (1,065 s) | 1.48 | 0.271 | 3.10643 at t = 8.892 |

12.5x, then **43.0x**. The rhs 1e-6 run's dissipation curve is within 2.4e-4 of its peak and the
energy at t = 20 within 1.7e-4 (`runs/tgv_re1600/tgv_n64_fast`); the deck's within 5.8e-3 and
4.6e-4, and against the 512^3 DNS as close as main's run (peak error 2.60 % against 2.54 %;
"Final runs").

**The polynomial order** (Taylor-Green, Q2 to Q6 at N ~ 96, 128 and 192 nodes per direction;
"The polynomial order" below): Q3 is the fastest per DOF and step; Q4 and Q5 reach a given
accuracy fastest and in the least memory (2.5 % peak-dissipation error: Q5 at 130^3 nodes in
356 s and 13.8 GiB, Q3 at 192^3 in 1,056 s and 36.3 GiB); memory per DOF falls with p.

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
| 26 | Host overhead round 2 (pressure mean, projection pool, flow sums, RecordStep) | yes | TGV +4.6 %, channel within noise; bitwise on the CPU | 49263939, 49263929 |
| 27 | `n_inner` 2 (fast set) | yes, opt-in | channel +12.5 %, TGV +5.6 %; TGV gate and channel 60..120 pass | 49265246, 49265767 |
| 28 | Velocity PCG tolerance 0.003 / 0.03 | no | within noise / slower | 49256626, 49265246 |
| 29 | Projection depth 4 / 16, extrapolated guess, lower block, no grad-div in the Schur PC | no | within noise or slower | 49256626 |
| 30 | NaN reported by the CFL rate | yes (fix) | - | 49266195 |
| 31 | `DeviceOrthoSolver` (the constant-pressure projection on the device) | yes | channel +13.9 %, TGV +12.3 %; bitwise | 49266911, 49266904 |
| 32 | Channel statistics' line sums on the device | yes | production wall per step -19 % (with 31) | 49266911 |
| 33 | Fused PCG for the velocity block | not done | sized at ~3 % (TGV), ~1 % (channel) | - |
| 34 | The fast set as defaults | not as it is | rhs-relative 1e-6 is wrong on DFG 2D-3, aborts on DFG 2D-1 | 49266126, 49268064 |
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

### Recommended deck settings (2026-10-14; superseded by the defaults of 2026-10-16)

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
| fused PCG (velocity block) | not done | sized at ~3 % of TGV, ~1 % of the channel (below) |
| production-only costs (dt changes) | open | ~4 ms per channel step, below; some change iterates |
| hypre's own host work, AMG settings | deferred | human ("not a task for here"; AMG another day) |

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

## The channel's linear solve (human 2026-10-15; jobs 49256626, 49265246, 49265767)

Why 4 outer iterations per step against TGV's 1.3: FGMRES does not stagnate. From the
projected guess (about 10 times below ||b||) the first iteration cuts the residual ~160
times and each further one ~10 times; 1e-6 of ||b|| takes four. Of the solver settings
other than AMG's (human: AMG deferred), only the Cahouet-Chabard inner iterations helped,
`n_inner` 3 having been chosen for TGV at the default tolerance and never revisited:

| Channel replay (substep CFL 1.5) | s/step | its/step | vs `n_inner` 3 |
|---|---|---|---|
| `n_inner` 3 (repeat: -0.8 %) | 0.0495 | 4.00 | |
| `n_inner` 2 | 0.0440 | 4.00 | +12.5 % |
| `n_inner` 1 | 0.0453 | 5.00 | +9.4 % |
| velocity PCG tolerance 0.003 (with 3 / 2 / 1) | 0.0483 / 0.0462 / 0.0484 | 3.53 / 4.00 / 5.00 | within noise or slower |
| projection depth 4 / 16; extrapolated guess | 0.0494 / 0.0527 / 0.0487 | 4.0 / 4.23 / 4.0 | within noise / -5 % / within noise |
| lower-triangular block; no grad-div in the Schur PC | 0.0704 / 0.0700 | 6.17 / 6.13 | -29 % |

TGV 32^3: `n_inner` 2 +5.6 % (0.0425 -> 0.0403), 1 +1.9 %. **Gates:** TGV 32^3 CFL 0.25 to
t = 20 passes for both (6.1e-5 and 1.05e-4 of the peak against the default run). The
channel from t = 60 to 120 at substep CFL 1.5, the same binary (a 15-unit window had given
`n_inner` 2 a 9.4 % U+ error, outside the earlier 7.7-9.0 %; the long window settles it):

| t = 60 to 120 | Re_tau | max U+ error | u_rms | v_rms | w_rms | u'v' | its/step | s/step |
|---|---|---|---|---|---|---|---|---|
| `n_inner` 2 | 178.42 | 7.57 % | 0.148 | 0.096 | 0.105 | 0.060 | 4.18 | 0.0498 |
| `n_inner` 3 | 178.31 | 8.25 % | 0.131 | 0.095 | 0.098 | 0.061 | 4.01 | 0.0543 |

`n_inner` 2 joins the fast set.

## Host overhead, round 2 (7d880f0, a173072; jobs 49263939, 49263929)

The remaining per-step allocations, from CUDA backtraces counted step to step (an earlier
count divided the run's setup and teardown -- hypre's pool freed at exit -- by the steps):
the pressure mean rebuilt a linear form and three vectors at every solve (7 pairs);
`SolutionProjection` allocated two vectors per step and deep-copied every kept vector when
its `std::vector` grew (mfem::Vector has no move); the flow sums' per-element vectors (4);
`Case::RecordStep`'s velocity copy (1). Now: `PressureMean` keeps the space's functional;
the projection reuses a pool and swaps vectors in; the small vectors are kept. Left: the CFL
rate's element copy and Jacobians (3, kept per call on purpose: memory at 64^3), MFEM's
`RectangularConstrainedOperator::EliminateRHS` (1: `Operator::AddMult`'s temporary), the flow
sums' element copy (1). TGV +4.6 %, the channel within noise. Bitwise: the CPU build's final
checkpoints are byte-identical old vs new (TGV 8^3 with projection k 4, the channel).

## The constant-pressure wrap: MFEM's OrthoSolver ran on the host (747a294)

On the channel the GPU was idle a third of each step (16.9 of 52.5 ms, ho2's profile); 32 of
the gaps per step, ~160 us each, sat around the Cahouet-Chabard pressure AMG. CUDA backtraces
of the copies in them: `mfem::OrthoSolver::Orthogonalize` (MFEM 1c19aba, marked "TODO:
GPU/device implementation") subtracts the mean of its input and output in a host loop --
two device-host round trips of the pressure vector around every AMG call. Every singular
pressure (periodic or enclosed: TGV, the channel) paid it. `DeviceOrthoSolver`
(precond/nullspace.hpp) is the same algorithm with the subtraction as a device vector
operation; `nullspace_test` checks it against MFEM's bitwise, and the CPU checkpoints are
byte-identical. **Channel replay 0.0435 -> 0.0382 s/step (+13.9 %), TGV 0.0381 -> 0.0339
(+12.3 %)**; GPU idle per channel step 15.3 -> 9.0 ms, gaps over 100 us 35 -> 4 (job
49266911). An upstream candidate; re-check at the MFEM upgrade.

(CPU sampling to see the host in those gaps does not work on the GPU nodes -- Nsight records
no samples there, though the login node passes its check; the CUDA backtraces found it.)

## Channel statistics on the device (df7e87e)

A production channel run spent ~6 ms per step (10-11 % of its wall time) outside the step
timer (sum of step walls against the run wall, four runs), invisible in s/step: the
statistics' plane averages, a host loop over the elements (~28 ms per sample, every 5
steps). A device kernel now forms the per-element line sums with the loop's arithmetic in
its order; the host adds them into the stations in the same order. The saved sums move at
roundoff (largest 1.1e-13 absolute); the profiles as written and the flow are identical.
**Production channel t = 60..63: wall per step 0.0536 -> 0.0434 s (-19 %), outside the steps
5.39 -> 0.33 ms** (job 49266911, with the previous section).

## Production runs against the replay (job 49266347)

The replay's fixed dt never exercises what a CFL-controlled run does at nearly every step
(dt changed on 2,630 of 3,538 steps): with the region profiler, per step, the projection's
images rebuilt when the mass coefficient drifts 5 % (+1.8 ms), the momentum operator
re-assembled (+1.3 ms), a second CFL-rate evaluation (+0.8 ms; OIFS evaluates the same
velocity again for its substeps), and 4.2 instead of 4.0 outer iterations with `n_inner` 2.
Not done: sharing the CFL rate between the step choice and OIFS is bitwise-free (~2 %); a
looser rebuild threshold and an in-place mass-coefficient update change iterates at the
tolerance level or need the momentum operator restructured. The human's call.

## Run-to-run reproducibility on the GPU (job 49263939)

Two runs of the same binary on one H100 give different final checkpoints (TGV 32^3, the
channel): GPU runs are already not bitwise reproducible, presumably in hypre's GPU AMG or
cuSPARSE (not traced; human 2026-10-15: library-level GPU nondeterminism is accepted). The
CPU build is: old vs old and old vs new byte-identical. "Bitwise-neutral" changes are
checked on the CPU build.

## NaN in the velocity (a173072)

The CFL rate's element max (and MPI's MAX) skipped a NaN -- every comparison with it is
false -- so a diverged velocity gave a finite rate and the run went on until an unrelated
check failed (the rotational channel run aborted in the multistep coefficients' ordering).
The kernel now reports a NaN node as an infinite rate and the steps abort with "the
velocity is not finite (NaN or Inf) at t = ...". Finite rates unchanged (cfl_test C5).

## The fast settings on the other benchmarks (human 2026-10-15, before the defaults decision)

Gated so far on TGV and the channel only, where dt sits near CFL 1-2. On the DFG 2D
benchmarks (`cases/dfg_2d3.yaml`: fixed dt 0.001; `cases/dfg_2d1.yaml`: error-controlled
steps to steady state), CPU, np 4 (jobs 49266126, 49268064):

| DFG 2D-3 (John 2004: c_D,max 2.9509, c_L,max 0.4780) | wall | its/step | c_D,max | c_L,max |
|---|---|---|---|---|
| the deck (`rtol` 1e-10 of the initial residual) | 242 s | 10.3 | 2.943 | 0.5178 |
| + projected guess (k 8) | 239 s | 10.9 | 2.943 | 0.5178 |
| + projected guess, `n_inner` 2 | 153 s | 19.6 | 2.943 | 0.5178 |
| projected, `n_inner` 2, `rtol` 1e-6 of the initial residual | 94 s | 11.7 | 2.943 | 0.5178 |
| **projected, `n_inner` 2, rhs-relative 1e-10** | **39 s** | 4.4 | **2.943** | **0.5178** |
| projected, `n_inner` 2, rhs-relative 1e-8 | 36 s | 2.9 | 2.945 | 0.5180 |
| the fast set (rhs-relative 1e-6, `n_inner` 3 / 2) | 35 / 19 s | 1.4 | 3.246 / 3.142 | 0.637 / 0.595 |

DFG 2D-1 (Schaefer & Turek): the deck 160 s; rhs-relative 1e-10 35 s with the same c_D, c_L
and Delta p; rhs-relative 1e-8 20 s with c_L moved 2 %; the fast set (1e-6) **aborts**
(the error controller rejects step after step: the solve's error is of the size of its
local-error tolerance). (The deck's own c_L,max error, 8 %, is this coarse mesh's.)

**Why:** the rhs-relative tolerance measures the residual against ||b||, and at a small dt
||b|| is the mass term (c0 M u^n, c0 ~ 1500 here): 1e-6 of it is loose against what a step
changes. Near CFL 1-2 (TGV, the channel) it is not. **For the defaults:** the projected
guess, `n_inner` 2 and the rhs-relative reference are safe everywhere tested, at a tight
`rtol` (1e-10: DFG 2D 4.6-6.2x faster than the decks); `rtol` 1e-6 only where dt is
CFL-limited near 1-2 (decks that opt in). A tolerance scaled by the step's own change (the
residual of the previous solution) would cover both; a design decision, not made.

## Re_tau 550 (job 49237225)

`benchmarks/channel/channel_retau550.yaml`, 36 x 48 x 36 Q3/Q2 (62,208 elements), one H100,
the morning's binary and the opt-in settings with `rtol` 1e-8 (5-7 outer its/step):
t = 0 to 40 in 8 h 57 min (34,547 steps, ~0.93 s/step). Statistics over t = 20 to 40 against
Lee & Moser / MKM at Re_tau 543.5: **Re_tau 546.25 (+0.5 %)**, max U+ error 3.56 % (y+ >= 5),
largest absolute differences u_rms 0.127, v_rms 0.070, w_rms 0.079, u'v' 0.023 (wall
units). With today's code, an estimate from the Re_tau 180 scaling (6x the elements, rtol
1e-6): ~0.25-0.3 s/step, ~2.5-3 h; not measured.

## Fused PCG for the velocity block: sized, not done

The velocity block is 15.0 of TGV 32^3's 42.6 ms per step (job 49255498's profile): kernels
12.8 ms, of which the vector operations (CG's adds, dots and fill) 1.47 ms, and the GPU idle
1.8 ms, mostly the ~30 dot-product round trips per step. A fused, one-reduction CG would
recover ~1.2 ms: ~3 % of TGV, ~1 % of the channel. MFEM's vector mass in the same block is
already near its shared-memory bound. Not worth another copy of MFEM's code now.

## The step-relative tolerance (b1bc283; the decks: 4440522)

`solver.tolerance_reference: step` stops FGMRES at ||r|| <= rtol ||b - A x_prev||, x_prev the
previous solution: the residual of "nothing changed", which scales with what the step has to
move. The rhs reference (rtol ||b||) is loose at a small dt, where ||b|| is mostly c0 M u^n:
it was wrong on DFG 2D-3 and 3D-3Z and made the error-controlled DFG 2D-1 and 3D-1Z abort
(above). With `initial_guess: project` the reference costs no operator application: the
projection keeps A x_prev from when it recorded the solution, and a new mass factor adds
(c0 - c0_prev) M x_u on the free velocity rows. The other references are unchanged (CPU final
checkpoints byte-identical; job 49268547).

| Step-relative, projected guess, `n_inner` 2 (jobs 49268547, 49268551, 49269400) | wall | against |
|---|---|---|
| DFG 2D-1, `rtol` 1e-6 | 25 s | the deck 160 s; c_D, c_L, Delta p the deck's to ~1e-4 |
| DFG 2D-3, `rtol` 1e-6 | 29 s | 242 s; c_D,max, c_L,max the deck's |
| DFG 3D-1Z, `rtol` 1e-6 | 114 s | 937 s |
| DFG 3D-3Z, `rtol` 1e-6 / 1e-4 / 1e-3 | 186 / 171 / 171 s | 557 s; c_D,max 3.21116 / 3.21132 / 3.21223 (deck 3.21116; rhs 1e-6: 3.30849) |
| TGV 32^3, `rtol` 1e-3 | 0.0314 s/step, gate 82 s | rhs 1e-6: 0.0339 s/step; dissipation within 1.7e-4 of the peak |
| channel replay, `rtol` 1e-3 | 0.0259 s/step | rhs 1e-6: 0.0377 s/step |
| channel production, t = 60..120, `rtol` 1e-3 | 0.0317 s wall/step | 0.0434; Re_tau 178.34, max U+ error 7.60 % (the sampling noise) |

The TGV and channel decks set it explicitly (`rtol` 1e-3, project k 8, `n_inner` 2; the
channel decks `time.oifs_cfl` 1.5). `rtol` still depends on the problem: 1e-3 where dt is
CFL-limited near 1-2, 1e-6 where a small fixed dt or an error controller sets it (DFG).

## The CFL rate once per history level (eb328eb)

A production channel step evaluated the CFL rate twice for the same velocity (the step
choice, then OIFS's substep count with the two older levels), each time recomputing the
Jacobians at the GLL nodes. `ConvectiveCfl::MaxRateAt` keeps each level's rate keyed by its
time; the step choice goes through the advector's cache. Bitwise (cfl_test C6; CPU
checkpoints and channel statistics byte-identical, job 49268350). Production channel wall per
step 0.0439 -> 0.0434 s (job 49268351).

## A new dt re-assembles only the mass term (9a36852)

`StokesOperator::SetMassCoeff` rebuilt the whole momentum form at every step-size change
(every integrator's partial-assembly data, the fusion, the true-dof and constrained
operators): ~1.3 ms per change on the channel, whose CFL-controlled steps change dt at ~75 %
of the steps. Only c0 M depends on dt; the mass integrator's AssemblePA alone recomputes it,
and the diagonal is c0 mass_diag + rest_diag as before. Bitwise (CPU production channel and
DFG 2D-1's 3,116 error-controlled steps; job 49268935). Production channel 0.0425 -> 0.0413 s
wall per step (job 49269819).

## The projection's rebuild threshold (8f7c7c2)

The projected guess recomputes its kept images (k operator applications and a Gram-Schmidt)
once c0 = beta0 / dt has moved 5 % since they were made; `solver.projection_rebuild_threshold`
(default 0.05, human 2026-10-15) sets it. Keeping M x for each kept solution would make a
rebuild k vector additions (~2 % more on the production channel, estimated) for k more
velocity-size vectors (1.4 GB at 64^3); not done (human: memory weighs more). Batching the
Gram-Schmidt's inner products (one reduction per pass) gave nothing measurable (production
channel 0.0413 -> 0.0418 s, job 49269460) and was dropped.

## Restarts with the projected guess, and the defaults (2026-10-16)

With `initial_guess: project`, a restart matched the uninterrupted run only to the solver
tolerance (relative differences 1e-10 to 4e-7 in checkpoint_test and channel_statistics_test
S5, job 49269820): the projection's kept solutions and images, and the step reference's
A x_prev, were not checkpointed. The human chose (a), checkpointing them, over (b)
tolerance-level restarts and (c) the previous solution as the default guess. A checkpoint
now holds each built solver's guess state (`StokesTimeIntegrator::ExportSolverStates`: the
starter's, BDF2's and BDF3's), handed to that solver when the restarted run builds it:
restarts are exact again on the CPU (np 1 and 2, jobs 49276071, 49276460: relative difference
0 for the adaptive, CFL-controlled, channel-statistics and AMR restarts; fixed steps 0 at the
default `rtol` and 1.7e-15 at the tests' 1e-10, the roundoff that test has always allowed
there). On GPUs two runs agree only to the solver tolerance (hypre's device AMG is not
run-to-run deterministic): at `rtol` 1e-6 checkpoint_test, channel_statistics_test S5,
deck_test and nondim_test differed by 3e-9 to 1e-6 (job 49276194), so those reproducibility
tests pin `rtol` 1e-10 (the guess and the reference stay the defaults) and agree to 3e-12 to
5e-11 (job 49276461). The cost: up to 2k + 1 system-sized vectors more per solver in each
checkpoint (17 at k 8), written from aliases of the solvers' vectors (no extra device memory).
Taylor-Green 64^3: 0.32 -> 1.69 s per checkpoint, 337 MB -> 1.8 GB (job 49276194), so the
Taylor-Green decks checkpoint every 1000 steps instead of 200 (~1 % of the run). Checkpoints
from before read as before (the solvers then start afresh).

The case-level defaults are now `initial_guess: project` (k 8), `tolerance_reference: step`,
`rtol` 1e-6 and `n_inner` 2 (`Parameters`; `StokesSolverOptions` keeps its own for the solver
unit tests and their baselines). The DFG decks drop their `rtol` 1e-10 for the default
(validated in "The step-relative tolerance"); the Taylor-Green and channel decks keep their
explicit `rtol` 1e-3.

## Final runs (2026-10-16; one H100, the 4440522 binary, the committed decks)

| | Before | Now | Speed-up |
|---|---|---|---|
| Channel Re_tau 180, t = 0 to 120 (spin-up and statistics) | 7 h 15 min, 28,628 steps (a6870f8, defaults, 2026-10-14) | **13.8 min** (826 s), 28,567 steps, 2.30 its/step, 0.0289 s wall/step (job 49270033) | **31.6x** |
| Taylor-Green 64^3 Q3/Q2, t = 0 to 20 | 12 h 44 min (45,847 s), 3,284 steps, 13.7 its/step (main, 2026-10-14) | **17.8 min** (1,065 s), 3,927 steps, 1.48 its/step (job 49270034) | **43.0x** |

The channel's statistics over t = 60 to 120: Re_tau 178.22 (DNS 178.12), max U+ error 8.02 %
(y+ >= 5), largest absolute differences u_rms 0.166, v_rms 0.095, w_rms 0.093, u'v' 0.063 (wall
units): the sampling noise of the earlier gates (7.4 to 9.0 %). Taylor-Green against the
512^3 DNS: peak dissipation error 2.60 % (main's run 2.54 %), largest |eps - ref| 3.23 % of
the peak (3.16 %). Against main's run the curve is within 1e-3 of the peak to t = 12 and
5.8e-3 at t = 15.4, in the turbulent decay (the deck's CFL 0.25 against main's 0.3 differs
too; the rhs 1e-6 run was within 2.4e-4); the gate (0.02) passes.

## Orders 5 and 6: compile-time kernels (c521d65; human 2026-10-16, before the order study)

Velocity orders 5 and 6 ran MFEM's runtime-size kernels (arrays for 14 points per
direction) for the vector mass, the 3D divergence and its transpose, the IMEX nonlinear
convection and the diagnostics' flow sums. They are registered at those sizes now
(`kernel_specializations.cpp`, `flow_sums.cpp`). At Q1D >= 7 nvcc stops unrolling MFEM's
`Grad3d` loop over directions, and kernels holding 3 x 3 x Q1D gradient arrays put them on
the stack (1.0-1.2 KB per thread, ptxas): the 3D divergence (K >= 5) and the viscous-only
kernel (Q1D >= 7) work one velocity component at a time (`operators/component_kernels.hpp`;
same arithmetic, no stack). The fused viscous + grad-div kernel at (7, 8) keeps its column
variant: a two-pass one-component version (no stack, 255 registers, the gradients twice)
was slower.

| Per apply, one H100 (jobs 49270936, 49271021) | Q5 (19^3) before | after | Q6 (16^3) before | after |
|---|---|---|---|---|
| flow sums (every diagnostics step) | 24.5 ms | 1.10 ms | 53.5 ms | 1.33 ms |
| vector mass | 0.40 ms | 0.26 ms | 0.39 ms | 0.25 ms |
| divergence / transpose | 0.79 / 0.76 ms | 0.28 / 0.32 ms | 0.53 / 0.48 ms | 0.26 / 0.34 ms |
| nonlinear convection (IMEX) | 1.03 ms | 0.87 ms | 1.38 ms | 1.23 ms |
| viscous only (grad-div off) | 0.87 ms | 0.50 ms | 0.65 ms | 0.53 ms |
| fused viscous + grad-div (two-pass, rejected) | | | 0.87 ms | 1.01 ms |

Kernel time of 13-step runs, setup included: Q5 1.59 -> 0.98 s, Q6 2.90 -> ~1.8 s. Still on
the stack at these sizes, MFEM's own kernels (fixes held for MFEM 4.10): the vector mass at
Q1D 5-7 (240-336 bytes) and the convection apply at (6, 8), (7, 10) (1.5-1.9 KB), and the
fused kernel's column variant at (7, 8) (1.2 KB). Setup at Q6 also costs: the viscous
diagonal 0.16 s and grad-div's nodal off-diagonal blocks 0.31 s (once per run).

## The polynomial order: time to solution and memory (human 2026-10-15; jobs 49271024, 49271184)

Taylor-Green Re 1600 from t = 0 to 20 on one H100 with the n32 deck's settings (IMEX, CFL
0.25, step `rtol` 1e-3, project k 8, `n_inner` 2, grad-div on) and Q_p / Q_{p-1} at about the
same number of velocity nodes per direction, N = n p; compile-time kernels at every order
(c521d65). Peak memory: `nvidia-smi` every 250 ms (the process, CUDA context included; the
profiler's gauge needs `profile: sync`, which would slow the runs). The dissipation's error
against the 512^3 DNS (`plot.py`): at its peak, and the largest over t <= 20, as a fraction
of the peak.

| p | elements | N | DOFs | steps | its/step | wall to t = 20 | DOF-steps/s | peak memory | bytes/DOF | peak error | largest error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 48^3 | 96 | 2.76M | 1,595 | 1.59 | 87 s | 5.09e7 | 7.0 GiB | 2,732 | 34.4 % | 37.2 % |
| 3 | 32^3 | 96 | 2.92M | 1,928 | 1.53 | **79 s** | **7.09e7** | 6.3 GiB | 2,336 | 15.6 % | 16.8 % |
| 4 | 24^3 | 96 | 3.03M | 2,262 | 1.48 | 102 s | 6.72e7 | 6.5 GiB | 2,299 | 7.0 % | 8.8 % |
| 5 | 19^3 | 95 | 3.01M | 2,790 | 1.47 | 141 s | 5.95e7 | 6.2 GiB | 2,206 | **4.8 %** | 9.3 % |
| 6 | 16^3 | 96 | 3.17M | 2,809 | 1.46 | 169 s | 5.25e7 | 6.4 GiB | 2,172 | 7.3 % | 8.5 % |
| 2 | 64^3 | 128 | 6.55M | 2,223 | 1.53 | 250 s | 5.84e7 | 13.5 GiB | 2,209 | 24.7 % | 24.7 % |
| 3 | 42^3 | 126 | 6.59M | 2,539 | 1.52 | **218 s** | **7.67e7** | 11.1 GiB | 1,808 | 12.3 % | 14.0 % |
| 4 | 32^3 | 128 | 7.18M | 2,970 | 1.45 | 287 s | 7.42e7 | 14.0 GiB | 2,090 | 4.6 % | 9.2 % |
| 5 | 26^3 | 130 | 7.72M | 3,418 | 1.43 | 356 s | 7.41e7 | 13.8 GiB | 1,923 | **2.5 %** | 7.6 % |
| 6 | 21^3 | 126 | 7.16M | 4,118 | 1.40 | 456 s | 6.46e7 | 13.0 GiB | 1,950 | 3.5 % | 4.9 % |
| 2 | 96^3 | 192 | 22.12M | 3,306 | 1.48 | 1,166 s | 6.27e7 | 41.9 GiB | 2,034 | 10.7 % | 12.4 % |
| 3 | 64^3 | 192 | 23.33M | 3,926 | 1.48 | **1,056 s** | **8.67e7** | 36.3 GiB | 1,670 | 2.6 % | 3.2 % |
| 4 | 48^3 | 192 | 24.22M | 4,614 | 1.41 | 1,357 s | 8.24e7 | 37.4 GiB | 1,656 | 1.9 % | 3.9 % |
| 5 | 38^3 | 190 | 24.09M | 5,270 | 1.38 | 1,532 s | 8.29e7 | 35.3 GiB | 1,573 | **0.18 %** | 3.3 % |
| 6 | 32^3 | 192 | 25.33M | 6,024 | 1.36 | 2,345 s | 6.51e7 | 36.3 GiB | 1,538 | 1.1 % | 3.5 % |

- **Per DOF and step, Q3 is the fastest** at every size (7.1, 7.7 and 8.7e7 DOF-steps/s at
  N ~ 96, 128, 192), Q4 and Q5 3 to 16 % below it, Q2 and Q6 the slowest (Q6 25 % below Q3 at
  N = 192: its fused (7, 8) column kernel and MFEM's convection at (7, 10) keep stack arrays).
- **Steps grow with p at a fixed N.** The CFL limit is taken at the GLL nodes, whose spacing
  near the element faces shrinks faster than h / p: at N = 192 Q6 takes 6,024 steps, Q3
  3,926, Q2 3,306. That is most of Q6's cost (its time per step is 1.45x Q3's).
- **Iterations per step** fall slightly with p (1.48 to 1.36 at N = 192).
- **Memory carries no high-order penalty.** Bytes per DOF fall with p (at N = 192: 2,034 for Q2,
  1,670 for Q3, 1,538 for Q6, the CUDA context included). Q2 needs the most: 41.9 GiB at 22M DOFs.
- **Accuracy at a fixed N rises steeply with p** (the peak-dissipation error, Q2 to Q6):
  34 / 16 / 7.0 / 4.8 / 7.3 % at N ~ 96, 25 / 12 / 4.6 / 2.5 / 3.5 % at N ~ 128, and
  10.7 / 2.6 / 1.9 / 0.18 / 1.1 % at N ~ 192. Q6 trails Q5 at every N: its elements are
  larger (16^3 against 19^3 elements at N ~ 96).
- **Time to an accuracy favours Q4 and Q5.** About 2.5 % peak error: Q5 at N = 130 takes 356 s
  and 13.8 GiB, Q3 at N = 192 1,056 s and 36.3 GiB (3.0x the time, 2.6x the memory). About
  5 %: Q5 at N = 95 takes 141 s and 6.2 GiB, Q4 at N = 128 287 s. Q2 at N = 192 (10.7 %, 1,166 s)
  is less accurate than Q4 at N = 96 (7.0 %, 102 s).
- **The largest error over t <= 20 levels off at 3.2 to 3.9 % of the peak at N = 192 for every
  p >= 3.** Each run's dissipation peaks about 0.09 early (t = 8.83-8.89 against the DNS's
  8.97), so past N ~ 128 at Q5 the spatial resolution no longer sets this error. The time step
  (CFL 0.25, BDF2) or the step tolerance (1e-3) would; they were not separated here.
- **On this flow, Q4 and Q5 give the best time and memory to a given accuracy.** Q3 is the
  fastest per DOF. Q2 is dominated everywhere. Q6 pays 1.5x Q3's steps at the same N, and its
  kernels are the least tuned.

## Two GPUs (job 49271026, c521d65)

The GPU tests at 2 MPI ranks, one H100 per rank: the fast tier passes (68 tests; without
nse_mms and tgv_nse, which time out on GPUs at any rank count). Taylor-Green 64^3 Q3/Q2
(23.3M DOFs) from t = 0, 5 + 20 steps, the decks' configuration: 0.219 s per step on one
H100, 0.157 s on two (1.40x, 70 % parallel efficiency), 1.20 against 1.25 iterations per
step.

## The default polynomial order: OIFS, the time step, grad-div and memory (human 2026-10-16)

Goal (human): the default p that reaches a target accuracy (largest error 10 / 5 / 2 %) with
the least run time and memory. Taylor-Green Re 1600 to t = 20 on one H100, OIFS (substeps at
CFL 1.5 on the GLL spacing; BDF3), Q_p / Q_{p-1}, about 96 to 192 nodes per direction. Error:
the largest |eps - eps_DNS| over t <= 20 as a percentage of the DNS peak (plot.py), and its
time average. Jobs 49290475/6 (CFL 2, 96^3 to 192^3), 49302516 (grad-div on / off),
49309226/7 (outer CFL 1, 2, 4, 160^3, memory), 49314402 (grad-div scaling), 49316573
(CFL 0.5, 160^3 at CFL 1); binaries main 9fa50a8 and 97fb45e.

**Grad-div on (gamma = 2 h_K), outer CFL 2** (wall: solve time to t = 20; memory: nvidia-smi peak):

| p | N | DOFs | steps | wall | peak memory | largest error | mean error |
|---|---|---|---|---|---|---|---|
| 3 | 96 / 126 / 159 / 192 | 2.9 / 6.6 / 13.3 / 23.3M | 260 / 336 / 445 / 500 | 22 / 59 / 149 / 294 s | 6.7 / 11.9 / 23.2 / 39.0 GiB | 14.1 / 11.8 / 5.4 / 5.6 % | 2.9 / 1.7 / 0.9 / 1.1 % |
| 4 | 96 / 128 / 160 / 192 | 3.0 / 7.2 / 14.0 / 24.2M | 303 / 391 / 483 / 586 | 30 / 86 / 203 / 404 s | 6.7 / 14.8 / 26.3 / 40.1 GiB | 11.2 / 5.7 / 9.0 / 5.6 % | 2.8 / 1.8 / 1.4 / 1.1 % |
| 5 | 95 / 130 / 160 / 190 | 3.0 / 7.7 / 14.4 / 24.1M | 366 / 441 / 553 / 667 | 37 / 110 / 243 / 522 s | 6.5 / 14.6 / 24.9 / 37.7 GiB | 6.8 / 7.6 / 5.3 / 5.0 % | 1.9 / 1.5 / 1.3 / 1.1 % |
| 6 | 96 / 126 / 162 / 192 | 3.2 / 7.2 / 15.2 / 25.3M | 369 / 525 / 688 / 761 | 56 / 152 / 392 / 721 s | 6.7 / 13.7 / 25.8 / 38.8 GiB | 6.6 / 2.7 / 3.1 / 2.7 % | 1.4 / 0.7 / 0.6 / 0.6 % |
| 7 | 98 / 126 / 161 / 189 | 3.4 / 7.3 / 15.1 / 24.5M | 419 / 529 / 751 / 893 | 72 / 176 / 461 / 849 s | 7.0 / 13.4 / — / 37.1 GiB | 10.2 / 3.1 / 5.1 / 3.0 % | 1.5 / 0.9 / 0.7 / 0.7 % |

- **OIFS against IMEX at the same N: 3.3 to 3.7x faster** (the IMEX study above runs at CFL
  0.25; OIFS takes 6 to 8x fewer implicit solves at about twice the cost per step).
- **The outer step.** With OIFS it is limited by accuracy, not stability; NekRS runs its
  characteristics at CFL 2-4 measured at the GLL nodes, as here. Largest error at 128^3 by
  outer CFL 0.5 / 1 / 2 / 4: Q3 15.1 / 14.4 / 11.8 / 20.8 %, Q4 7.8 / 7.4 / 5.7 / 20.3 %,
  Q5 8.4 / 8.3 / 7.6 / 17.7 %, Q6 4.4 / 4.1 / 2.7 / 14.3 %, Q7 5.0 / 4.7 / 3.1 / 15.4 %.
  CFL 4 is out at every p (the dissipation rise near t = 6.3 is mistimed). CFL 1 is within
  0.3-0.7 points of CFL 0.5. CFL 2 carries a 1.5-2.7 point time error that partly cancels
  the spatial error, so its numbers above flatter every order; the time-averaged error barely
  moves with CFL. At a fixed N the GLL CFL rule takes 1.85x more steps at Q7 than at Q3 (the
  smallest GLL gap is 0.45 of the average spacing at Q7, 0.83 at Q3); at matched outer dt the
  time errors were similar, so the outer step is accuracy-limited at every order.
- **The largest error is noisy past the dissipation peak** (under-resolved turbulence): Q4 is
  worse at 160^3 (8.4 % at CFL 1) than at 128^3 (7.4 %). Differences under 1-2 points are noise.
- **Cheapest run per target, time error removed (CFL 1):** 10 %: Q4 at 128^3 (7.4 %, 115 s,
  14.8 GiB); Q3 needs ~160^3. 5 %: **Q6 at 126^3 (4.1 %, 212 s, 13.7 GiB)**, Q7 at 126^3
  (4.7 %, 234 s), Q5 at 160^3 (3.7 %, 326 s, 25 GiB); Q3 and Q4 do not reach it by 192^3.
  2 %: none yet (Q6 and Q7 sit at 2.7-3.8 % up to 192^3).

**The velocity block** (job 49302516, ~128^3, CFL 2; jacobi_pcg, iterations per outer
iteration from the new history columns, 97fb45e):

| p | grad-div | outer its/step | velocity CG its | s/step | largest error | max ‖div u‖ |
|---|---|---|---|---|---|---|
| 3 | 2 h_K / off | 2.3 / 5.5 | 16.8 / 2.0 | 0.18 / 0.17 | 11.8 / 9.1 % | 1.1 / 21.7 |
| 5 | 2 h_K / off | 2.1 / 4.9 | 28.5 / 2.0 | 0.25 / 0.17 | 7.6 / 7.3 % | 0.54 / 17.9 |
| 7 | 2 h_K / off | 2.0 / 4.7 | 38.4 / 2.0 | 0.33 / 0.17 | 3.1 / 5.7 % | 0.34 / 17.0 |

- **All of the per-step growth with p is grad-div under point-block Jacobi.** Without it the
  velocity CG takes 2 iterations and a step costs 0.17 s at every order (equal DOFs). With
  gamma = 2 h_K the coefficient grows with the element size, 2.3x from Q3 to Q7 at equal
  DOFs, and so does the block's stiffness against the mass term (gamma dt / dx_min^2: ~10
  at Q3, ~53 at Q7); Q7 starts hitting the 50-iteration cap.
- **Grad-div earns its keep:** without it the outer iterations go 2 -> 5, ‖div u‖ is 20-50x
  larger and Q6 / Q7's error grows (at CFL 1: 4.1 -> 7.6 %, 4.7 -> 7.4 %). Q3 is better
  without it (14.4 -> 10.8 %).
- **gamma = c h_K / k_u** (job 49314402; c = 2 k_u is the old 2 h_K): the velocity CG's
  iterations follow c, nearly independent of p: c = 6 gives 16.8 / 19.7 / 21.5 / 22.2 / 24.5
  at Q3-Q7 (against 16.8 to 38.4 with 2 h_K). Q6: c = 2 / 6 / 12 run in 101 / 124 / 152 s
  with 4.0 / 3.3 / 2.7 % (CFL 2) and ‖div u‖ 1.26 / 0.65 / 0.41. Q7: c = 1 / 2 / 6 / 14 in
  91 / 104 / 135 / 172 s. (Q5 at c = 1 ran during a filesystem stall; its time is void.)

**Memory per DOF is flat in p at a fixed N** (~1,800-2,200 B/DOF), so high order saves
memory only by needing fewer DOFs. Breakdown at ~128^3 from the profiler's gauge (profile:
sync), B/DOF, Q3 / Q5 / Q7: pressure LOR-AMG 418 / 705 / 752; vectors (state, right-hand
sides, scratch, solver work) 328 / 412 / 392; operator data and geometric caches 422 / 271 /
229; OIFS 246 / 204 / 186; projection basis and FGMRES growth after step 1 174 / 171 / 150;
the CUDA context 141 / 120 / 128; FGMRES 66 / 65 / 81. (Each figure is a region's own net
allocation; a release booked in an enclosing region lowers that region instead. The
trapezoidal starter's nu K, 67 / 38 / 28 in its own region, is freed right after its one
product, so it is not held at the peak; the vectors are that much larger.) OIFS keeps one
Jacobian array: on the column path (more than 8 points) it builds its own from the nodes and
never asks for MFEM's cached one. The pressure space is Q_{p-1}, so at equal velocity resolution Q7 has 2.1x the
pressure unknowns of Q3, which eats the operators' savings with p (2.55 -> 1.39 GB).
- **The pressure LOR-AMG** (jobs 49318797, 49321252; branch memory-audit): hypre 2.33 is built
  with its device memory pool (Spack's recipe adds `--enable-device-memory-pool` to any CUDA
  build), whose defaults round every allocation up to a power of 8 and cache every freed
  block. `HYPRE_SetGPUMemoryPoolSize(2, 1, 20, 64 MiB)` (pool only blocks up to 1 MiB) cut the
  hierarchy from 3,284 to 594 MiB and the Q7 peak 13,089 -> 10,371 MiB (-21 %; Q3 -680 MiB),
  setup +5 ms, steps unchanged. MFEM's LORSolver kept its LOR mesh, space and batched-assembly
  scratch for its lifetime: freeing them after assembly saves 460 MiB at Q3 and Q7 (6d90d0c).
  BoomerAMG's strength threshold 0.5 / 0.6 made the hierarchy larger, aggressive coarsening
  did not shrink it. The assembled LOR matrix holds 1.4 GB at Q7 for ~0.4 GB of nonzeros; a
  deep copy made by hypre freed nothing (job 49341452), so it is one copy, ~1 GB of it not
  yet explained.
- **After both fixes Q7 is at 1,488 B/DOF** (1,946 before).
- **Measured capacity at Q6** (Taylor-Green, OIFS CFL 2, ~25 steps, one H100, nvidia-smi
  peak; jobs 49341518, 49341752), largest size that ran / first that did not:

  | | 25.3M DOFs | largest that ran | first out of memory |
  |---|---|---|---|
  | main before the fixes, coupled | 38.8 GiB, 1,647 B/DOF | 39.2M (65.7 GiB) | 57.3M |
  | both fixes, coupled | 32.1 GiB, 1,360 B/DOF | 57.3M (67.9 GiB) | 70.4M |
  | both fixes, `solver.outer: schur` | 26.3 GiB, 1,117 B/DOF | 70.4M (70.6 GiB) | 85.5M |

  The jump before the fixes at 39.2M (1,800 B/DOF) is the pool's power-of-8 rounding. 100M
  DOFs needs ~850 B/DOF, about 20 % below Schur's 1,076. What is left has not been measured:
  sharing MFEM's per-form scratch and the time integrator's vectors, the AMG in single
  precision, OIFS's Jacobians and wind in single precision or on the fly.

**Decision (human 2026-10-16): Q6/Q5 by default and gamma = 6 h_K / k_u** (`grad_div_h`
default < 0 = 6 / k_u: 2 h_K at Q3, so Q3 runs are unchanged). Q6 is the cheapest order to
5 % (212 s at CFL 1, 152 s at CFL 2) with about half the memory of the lower orders that come
close, and the scaled grad-div cuts its time by about a fifth. These numbers assume OIFS at
CFL 2 (substeps 1.5), which became the default the same day (`time.convection: auto`; human:
"it honestly seems to always be better").

## The solver matrix at Q6: one and two H100s (2026-10-17/18)

Every solver configuration at ~22M DOFs (Q6/Q5, `output.profile: sync`; the decks are
`runs/verify/decks/thr_*`): Taylor-Green 32^3 (OIFS at CFL 2, IMEX) and the curved DFG
3D cylinder channel (level 2, nz 5), coupled and Schur, at np 1 and np 2 (one H100 per
rank), on the change-32 code:

| Case | np 1 s/step | np 2 s/step | Efficiency |
|---|---|---|---|
| TGV OIFS coupled | 0.511 | 0.363 | 70 % |
| TGV OIFS Schur | 0.616 | 0.395 | 78 % |
| TGV IMEX coupled | 0.308 | 0.225 | 69 % |
| TGV IMEX Schur | 0.341 | 0.216 | 79 % |
| Cylinder OIFS coupled | 1.118 | 0.730 | 77 % |
| Cylinder OIFS Schur | 1.209 | 0.735 | 82 % |
| Cylinder IMEX coupled | 0.832 | 0.582 | 72 % |
| Cylinder IMEX Schur | diverged (fixed by change 34) | | |

The DFG Q3 AMR run and the channel at Re_tau 180 are slower at np 2: under ~2M DOFs per
GPU, launches and latency dominate.

What it found, and what was done:

- **The Stokes step projected Dirichlet data on the host at every solve**: 230 ms of every
  step on the cylinder, about 20 % (`stokes_solver::solve_true`'s own time). Change 33: the
  separable data's shapes are projected once per space and scaled on the device.
- **The Schur method stalled on the curved IMEX case**: the first velocity solve's
  constraint residual met the coupled target, so the pressure was never updated. Change 34
  gives the pressure its own target (1.6-2.8 FCG iterations per solve).
- **The momentum apply at Q6 ran the column kernel** (its thread-per-point version needs
  64 KB of shared memory). Change 36: dynamic shared memory; 0.94 -> 0.57 ms per apply.
- **All communication was staged through the host.** Change 37: GPU-aware MPI by default.
- **The pressure LOR-AMG scales worst**: 2x (Taylor-Green) to 4x (cylinder) slower per call
  at np 2 with half the work per GPU. GPU-aware MPI cuts its time by about a quarter on
  Taylor-Green; the rest is docs/multi_gpu.md's Phase 2 (AMG settings are the human's call).
- **`oifs::advect`'s own time on the cylinder (~200 ms per step) and the curved IMEX
  convection (158 ms per step)** were mostly the outflow condition's host round trips:
  change 39 (below).

**Change 36** (prototype measurement, job 49397767; one H100, 21.6M DOFs):

| Case | s/step before | after | Change |
|---|---|---|---|
| TGV OIFS coupled | 0.511 | 0.416 | -19 % |
| TGV OIFS Schur | 0.622 | 0.474 | -24 % |
| TGV IMEX coupled | 0.317 | 0.261 | -18 % |
| Cylinder OIFS coupled | 1.111 | 0.986 | -11 % |

**Change 37** (job 49397771; np 2 on two H100s, 21.6M DOFs):

| Case | s/step off | on | Change | Pressure AMG, whole run |
|---|---|---|---|---|
| TGV OIFS coupled | 0.371 | 0.297 | -20 % | 1.69 -> 1.27 s |
| TGV IMEX coupled | 0.225 | 0.182 | -19 % | 1.35 -> 1.10 s |
| Cylinder OIFS | 0.743 | 0.719 | -3 % | 3.52 -> 3.31 s |

**Changes 36 + 37 as landed** (job 49400578; change 35 -> 36 + 37, s/step):

| Case | np 1 | np 2 (36 alone, GPU-aware off) | np 2 (36 + 37) |
|---|---|---|---|
| TGV OIFS coupled | 0.517 -> 0.412 (-20 %) | 0.359 -> 0.334 (-7 %) | 0.250 (-30 %) |
| TGV OIFS Schur | 0.712 -> 0.535 (-25 %) | | |
| TGV IMEX coupled | 0.325 -> 0.258 (-21 %) | | |
| TGV IMEX Schur | 0.611 -> 0.474 (-22 %) | 0.429 -> 0.356 (-17 %) | 0.262 (-39 %) |
| Cylinder OIFS coupled | 0.889 -> 0.786 (-12 %) | 0.640 -> 0.569 (-11 %) | 0.573 (-10 %) |

GPU-aware MPI does nothing for the cylinder at np 2 (0.569 -> 0.573); its pressure AMG
stays 17-18 ms per call.

**What the cylinder's step spends besides kernels** (Nsight Systems with CUDA call
stacks, jobs 49400579 and 49401163; the regions are NVTX ranges):
- The outflow condition (directional do-nothing) copied a whole velocity vector to the
  host and back per evaluation: ~120 ms per step OIFS, ~113 ms IMEX. **Change 39** moves
  only the outflow elements' dofs (the same sums, bitwise): cylinder OIFS 0.775 -> 0.707
  s per step (-9 %), IMEX 0.510 -> 0.455 s (-11 %) at np 1 (job 49401382).
- `RectangularConstrainedOperator::EliminateRHS` called MFEM's default `Operator::AddMult`
  on the divergence block, which allocates and frees a full-size vector per call (the
  free 5.6-7.8 ms with its device synchronization), twice per step. **Change 40**: the PA
  blocks add into y themselves (bitwise). Taylor-Green OIFS -1.5 %, IMEX Schur -7 %; the
  cylinder within its run-to-run variation (job 49401559).
- The curved CFL rate (the old path) allocates an element copy and the Jacobians per
  call; `oifs::classify` allocates its transient flag vectors (change 32's memory
  choice). A few ms per step each in the call-stack profile; not changed.

**Change 38**: change 36's kernel at Q7 ((8, 9): 729 threads, 93 KB of dynamic shared
memory). 0.71 -> 0.44 ms per apply; Taylor-Green Q7 24^3 (17M DOFs) OIFS 0.399 -> 0.310
s per step (-22 %), IMEX Schur 0.424 -> 0.324 (-24 %) (jobs 49400684, 49401382).

**Tried and rejected**: three blocks per SM for the Q6 kernel (`__launch_bounds__(512,
3)`: 40 registers, ~800 bytes of spills): 0.57 -> 0.81 ms per apply, +15 % per step (job
49401164). Two blocks (change 36) stay.

**DFG 3D-3Z Q3 with change 32** (OIFS's Dirichlet data per stage without projections;
job 49390028): 151 -> 60 s of step wall time conforming, 698 -> 189 s with AMR
(`oifs::dirichlet` had 380 s and `oifs::classify` 109 s of the AMR run); c_D within
4.4e-5 relative, c_L within 1e-4 absolute (the GPU's run-to-run level).

**Held or dropped this round.**
- Geometry on the fly (Jacobians from the nodes in every kernel, ~74 B/DOF): +26-28 % per
  step on Taylor-Green, +14 % on curved DFG Q6; held by the human (2026-10-17), a candidate
  only for the memory profile.
- hypre's prolongation at np 1: helps one rank only; dropped (human 2026-10-17).

**Open, no memory trade-off**: the AMG at np > 1 (instrumentation first); batched
reductions in FGMRES and CG (docs/multi_gpu.md, Phase 3); the remaining per-step
allocations (the curved CFL rate, OIFS's classification). **Trade-off candidates** for a
throughput profile: hypre's keep-transpose (a stored restriction).
