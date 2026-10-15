# Throughput: measurement, results and decisions

The record of the throughput work (from 2026-10-14): how throughput is measured, what the
solver spends its time on, and every change kept or rejected, with its measurement. It
follows the maintainer's throughput plan (tiers 0 to 5), adapted to bessemer (below).
Headline metric: **DOF-steps per second per GPU** (velocity plus pressure true dofs times
steps, per second of step wall time) on one H100 80 GB, Bridges-2, CUDA backend.

## Where it stands (2026-10-15, 04:30)

Taylor-Green from t = 0 on one H100 (32^3: 20 + 50 steps, 3 repeats; 64^3: replayed from
t = 19, 12 steps):

| | TGV 32^3 s/step | Outer its | Speed-up | TGV 64^3 s/step | Speed-up |
|---|---|---|---|---|---|
| Before (main, 0aabb9b) | 1.44 | 11.6 | | 16.75 | |
| Code changes only, defaults unchanged (1a, fused kernel) | 1.08 | 11.6 | 1.33x | | |
| + project k 8, rhs tolerance | 0.369 | 3.7 | 3.9x | 6.31 | 2.7x |
| + `n_inner` 3 | 0.248 | 3.9 | 5.8x | 3.16 | 5.3x |
| + `rtol` 1e-6 | **0.101** | 1.3 | **14.2x** | **1.15** (`n_inner` 1) | **14.6x** |

Every row below the second passes the expensive Taylor-Green gate (dissipation curve to
t = 20 within 1.1e-4 of the default run's peak). The solver settings are opt-in deck keys;
the defaults are unchanged. On the turbulent channel (OIFS, walls) the same settings give
1.77x.

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
