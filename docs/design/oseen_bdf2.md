# Semi-implicit and implicit BDF2 (Oseen) schemes: what was built, what was learned, why it is parked

**Status: parked, 2026-10-21 (human decision).** Branch `oseen-bdf2` in `bessemer-c36`
(on main 689dd2f; nothing merged). The existing solvers are untouched: the branch adds new
files (`src/oseen/`, `apps/oseen_case.cpp`, `apps/oseen_cylinder.cpp`, two tests) and only
registers them in the three `CMakeLists.txt` files. All runs below are on Bridges-2; job IDs
are given so every number can be traced to its log in `~/bessemer_logs/`.

## Summary

The goal (human, 2026-10-20): "find a problem where implicit or semi implicit beats the
various imex schemes", with turbulence as the requirement ("If the scheme won't work for
turbulence then it is useless"). The starting point was a recipe from a separate 2D research
code: BDF2 with every step an Oseen solve, preconditioned by low-order-refined (LOR) AIR for
the velocity block and PCD for the pressure Schur complement, all on GPUs.

What came out of it:

1. **GPU AIR works in 3D on hypre 2.33**, mesh-independently, where classical AMG does not.
   The hypre dropping bug the recipe warned about is in 2.33 too; a 10-line patch fixes it.
2. **Implicit clearly wins on steady flow**: DFG 3D-1Z reaches its steady state in 80 steps
   (4 min on 16 CPU cores) against 14,920-21,245 OIFS steps. On lift/drag (DFG 2D-3, 3D-3Z)
   it is competitive: drag is right at any step, lift needs small steps.
3. **On vortex-dominated flow (Taylor-Green) PCD fails** at CFL 5-10. The cause is the OSS
   stabilization in PCD's pressure-space operator. Without it PCD converges.
4. **An augmented-Lagrangian Schur side fixes the pressure block**: strong grad-div
   (gamma = 100 nu) with Cahouet-Chabard weighted by (nu + gamma) gives 8-14 outer
   iterations at CFL 10, flat in time, *if* the velocity block is solved well.
5. **The velocity block is what fails.** Neither modified AL (block Gauss-Seidel over the
   components) nor any LOR variant fixes it. The LOR operator is a poor stand-in for the
   high-order convection operator: with its blocks inverted exactly, the outer count is still
   27 at CFL 1.4 and 75 at CFL 8, against about 14 with the true high-order block. The gap
   grows roughly linearly with CFL.
6. **So larger steps buy nothing.** On Taylor-Green 12^3 at Q6 (CPU), the total work to a
   fixed time is flat in CFL from 1.8 to 8.3. The best AL configuration is 1.8x slower than
   production OIFS at lower accuracy, and 2.9x slower at matched accuracy.
7. **The GPU path is not finished.** Q6 hits three MFEM kernel-size limits; two are worked
   around, one (determinants at 10 points per direction) is open. No GPU timing was measured.

Making implicit pay for turbulence-type flow would take a velocity preconditioner whose
quality does not degrade with CFL in the convection-dominated, high-order regime. That is a
research problem, not tuning, so the scheme is parked.

## 1. Where it started

### The recipe

From a 2D research code (MFEM 4.10, hypre 3.2), handed over 2026-10-19/20:

- Taylor-Hood Q_k/Q_{k-1} on GLL nodes, GLL-lumped velocity mass.
- Skew-symmetric linearized convection C(w) + 1/2 (div w) M.
- OSS stabilization (Codina's orthogonal subscales):
  S = K_tau - C_tau^T W_tau^-1 C_tau, with tau at the element centre.
- Grad-div gamma = nu, left out of the velocity preconditioner.
- Semi-implicit BDF2 (one Oseen solve per step, wind extrapolated) and implicit BDF2
  (Picard, to a tolerance or a fixed count), in defect-correction form.
- FGMRES with a block upper-triangular preconditioner:
  - **Velocity block:** one AIR V-cycle per component on an LOR operator
    sigma D + nu K_lor + C_lor + SD_lor, with streamline diffusion SD from the high-order
    parent's tau, assembled on the host every 10 steps; classical AMG below a cell Reynolds
    number of 3.
  - **Schur complement:** PCD, A_p^-1 (sigma + N_p M_p^-1) + nu M_p^-1, with N_p the
    pressure-space convection plus OSS plus an inflow term.

### The constraints

- "Whatever you do does not have any impact on the outputs of the existing solvers."
- "Classical AMG won't work so if no AIR then forget about it."
- Everything on GPUs eventually.

A 2026-10-07 note in `docs/status.md` ("not interested in implicit methods at the moment")
recorded an earlier Picard experiment with no Oseen preconditioner, never cheaper than IMEX.
The recipe's point was the preconditioner, so the question was reopened.

### The gate: GPU AIR on hypre 2.33

The stack has hypre 2.33, not 3.2. A static check found device Neumann AIR, one-point
interpolation and device dropping in 2.33. A probe on a 3D Q1 operator like the recipe's LOR
one (FGMRES + one V-cycle, one H100, job 49498361) settled it:

| 3D Q1 LOR-like operator | AIR, 118k -> 912k dofs | classical AMG |
|---|---|---|
| nu 1e-3, constant wind, steady | 10 -> 12 | 39 -> 37 |
| nu 1e-5, constant wind, steady | 21 -> 24 | 74 -> 100 |
| nu 1e-5, vortex wind, steady | 27 -> 32 | 133 -> 164 |
| nu 1e-5, CFL 2 (with the mass) | 7 -> 7 | 17 -> 17 |
| nu 1e-1 (diffusion-dominated), steady | 23 -> 43 | 14 -> 15 |

GPU AIR works in 3D, so the work went ahead.

**hypre's dropping bug.** The recipe's bug is in 2.33 too: `DropSmallEntriesDevice`
compresses the off-rank column map without rebuilding the communication package, so AIR with
`ADropTol` crashes at np >= 2. The fix is a 10-line patch (in
`$PROJECT/hypre-src/hypre-2.33.0`, branch v2.33.0-commpkg). With it, AIR with dropping
converges at np 2 and its setup is 4.6x faster on the hardest case (job 49528973). The
patched stack (environment `psc_gpu_hypfix`) was never made the default. Doing so needs the
human's OK and the stuck Spack database lock cleared. Until then the code sets `ADropTol` to 0
above one rank.

A side finding: the GPU hypre is built without GPU-aware MPI, and at np 2 an AIR iteration
took about 37 ms against 3.6 ms at np 1 on the same problem.

## 2. What was built

`src/oseen/` (about 2,200 lines):

- **`OseenWindData`**: the wind and its derived data at the quadrature points, on the device:
  w, 1/2 div w, tau_c per element, tau w w^T, tau w, tau at the GLL points.
- **`ScalarOseenOperator`, `ComponentwiseOperator`**: the Oseen convection plus OSS, in
  partial assembly, applied per velocity component. The byNODES vector space is the scalar
  space block by block.
- **`OseenLorPreconditioner`**: the LOR operator, host-assembled, with AIR or AMG per
  component. Options:
  - diagonal grad-div blocks (`SetGradDiv`);
  - the velocity coupling: block Jacobi (default) or upper block Gauss-Seidel (modified AL);
  - GMRES block solves (`block_rtol`, diagnostics);
  - the streamline-diffusion factor and 1/2 div w (diagnostics);
  - a V-cycle counter.
- **`OseenPcdSchurPC`**: PCD on bessemer's Cahouet-Chabard machinery, with the N_p OSS and
  convection switchable.
- **`OseenStepper`**: semi-implicit and implicit BDF2 (BDF1 first step). Schur choice: PCD,
  or bessemer's `CahouetChabardSchurPC` with the grad-div weighting (the AL Schur side).
  Optional inner FGMRES on the velocity block (`velocity_inner_rtol` / `maxit`), constant body
  force, forces from the momentum residual.

Drivers and tests:

- **Drivers:**
  - `apps/oseen_cylinder` (DFG 2D-3);
  - `apps/oseen_case`, deck-driven: AMR off, deck boundary conditions, body force, channel
    statistics, kinetic energy (`-ke`), every solver option as a flag.
- **`test/oseen_operator_test`**:
  - O1: true dofs by component;
  - O2: partial vs full assembly, also at Q6;
  - O3: skew symmetry;
  - O4: OSS symmetric and PSD on affine meshes, bounded on curved ones;
  - O5: bitwise determinism;
  - O6: wind data at Q6 against a host evaluation.
- **`test/oseen_lor_pc_test`**:
  - L1: the grad-div blocks reassemble the vector grad-div form;
  - L2: tight block solves invert the block diagonal or upper triangle;
  - L3: determinism.
- All pass at CPU np 1, 2 and 4. `oseen_lor_pc_test` also passes on GPUs at np 1 and 2.

One finding on the way: the OSS term is positive semidefinite on affine elements only. On a
strongly curved test mesh its smallest eigenvalue is -0.3 % of its largest.

## 3. Phase 1 -- the recipe as written, on the cylinder benchmarks (CPU)

**DFG 2D-3**, level 2, Q3, CPU np 4, against bessemer's OIFS at the decks' tolerance:

| Scheme | CFL max | c_D,max error | c_L,max error | dp(8) error | Wall |
|---|---|---|---|---|---|
| implicit, 2 Picard, dt 0.01 | 21 | 1.3e-5 | -3.7 % | -0.8 % | 105 s |
| implicit, 2 Picard, dt 0.005 | 11 | 7.2e-6 | -0.87 % | +0.3 % | 192 s |
| implicit, 2 Picard, dt 0.0025 | 5.3 | 1.9e-5 | -0.25 % | +0.15 % | 333 s |
| semi-implicit, dt 0.0025 | 5.3 | 1.8e-5 | +1.4 % | -0.2 % | 703 s |
| OIFS, dt 0.005 | 10.6 | 6.5e-3 | 1.5 % | 0.5 % | 381 s |
| OIFS, dt 0.0025 | 5.3 | 9.1e-4 | 0.20 % | 0.05 % | 512 s |
| OIFS, CFL 4 | 4.3 | 3.8e-4 | 0.13 % | 0.02 % | 472 s |
| OIFS, CFL 2 (decks' default) | 2.35 | 3.6e-5 | 0.018 % | 0.015 % | 706 s |

- **Drag:** implicit gets the drag peak right to about 1e-5 at every step tried, up to
  CFL 21. OIFS's drag error grows fast with the step.
- **Lift:** at equal CFL (5.3) the two are comparable, and implicit is 1.5x cheaper. For about
  0.02 % lift, OIFS at CFL 2 is about as cheap as implicit would be.
- **Semi-implicit** was the weakest option, as the recipe found.

Moving to Q6 and 3D:

- **2D-3 at Q6:** OIFS needs about twice the steps it needs at Q3. Implicit is about 2x
  cheaper at moderate lift accuracy (0.3 %) and better on drag; at 0.05 % lift the two are
  comparable.
- **DFG 3D-1Z (steady, Re 20): a clear win.**
  - Implicit reached the steady state in 80 steps, 4 min on 16 CPU cores, with c_D within
    0.005 % and c_L within 0.02 % of FeatFlow.
  - The OIFS verification run took 21,245 steps (48 min on an H100) with AMR, and 14,920
    steps without.
- **DFG 3D-3Z at Q6 (lift/drag), implicit at CFL 11:**
  - 1,600 steps; c_D,max within 0.18 % of FeatFlow, the same to 6 digits at half the step.
  - OIFS without AMR needed 11,433 steps and was off by 0.41 %.
  - But an implicit step cost 3.4 s on the CPU (34 FGMRES iterations), so this one would have
    been decided on GPUs.

PCD worked on the cylinders: 5-27 iterations per step. These are through-flows with smooth
wind gradients away from the body.

## 4. Phase 2 -- vortex-dominated flow: PCD fails

Taylor-Green Re 1600, Q6/Q5 on 12^3, semi-implicit to 1e-6, CPU (job 49513119), FGMRES
iterations per step (mean, max) over the run to t = 10:

| CFL | PCD (the recipe) | Cahouet-Chabard (PCD without N_p) |
|---|---|---|
| ~2 | 27-78 (max 157) | 29-67 (max 76) |
| ~5 | 208-500: at the 500 cap from t ~ 4 | 76-144 (max 159) |
| ~10 | 427-500: at the cap nearly every step | 179-300 (max 333) |

PCD failed already in the laminar vortical phase, before any turbulence.

**Why Cahouet-Chabard degrades but never fails.** Write F = H + N, with H the symmetric
positive definite part and N the skew convection. Then the exact Schur complement
B F^-1 B^T has a positive-definite symmetric part, and Cahouet-Chabard is SPD. So the
preconditioned operator's field of values stays in the right half-plane, and its distance
from zero shrinks roughly like 1/CFL. Iterations grow with CFL but never stall.

**Why PCD failed, diagnosed** (job 49529209, CFL 5 / 10):

| Variant | PCD | Cahouet-Chabard |
|---|---|---|
| Taylor-Green, N_p without OSS | **68 / 126**, no cap hits | (above) |
| Translated Taylor-Green (U_0 = (4, 0, 0)) | 53 / 104 | 71 / 206 |

The OSS stabilization in N_p was the cause. Without it, PCD beats Cahouet-Chabard. With a
mostly uniform wind, where PCD's commutator assumption nearly holds, PCD also works. Neither
variant removed the growth with CFL.

The human then set the requirement: turbulence. They ruled out Creff and Guermond's
projection-method comparison as a benchmark ("projection methods have a ton of weaknesses on
real geometries with obstacles").

## 5. Phase 3 -- the augmented Lagrangian: the Schur side solved

A literature survey of PCD alternatives (LSC excluded by the human) pointed to the augmented
Lagrangian:

- Benzi and Olshanskii 2006;
- Heister and Rapin 2013 (grad-div stabilization as an AL term);
- Farrell, Mitchell and Wechsung 2019.

With a strong grad-div term gamma (div u, div v) in F, the Schur complement tends to
(nu + gamma)^-1 M_p plus the sigma part, whatever the convection. All the convection
difficulty moves into the velocity block. bessemer already has the matching preconditioner:
`CahouetChabardSchurPC` with the grad-div weighting `StokesSolver` uses.

**The gamma probe** (Taylor-Green 12^3 at CFL 10, jobs 49529758 / 49530376):

| Schur side | Velocity block | Outer its/step |
|---|---|---|
| CC, gamma = nu | 1 V-cycle | 168.9 |
| CC (nu + gamma), gamma = 10 nu | 1 V-cycle | 103.7 |
| CC (nu + gamma), gamma = 100 nu | 1 V-cycle | 146.8 |
| CC (nu + gamma), gamma = 100 nu | inner FGMRES to 1e-2 | **12.7**, flat 12-15 |
| PCD without N_p OSS, gamma = 100 nu | inner FGMRES to 1e-2 | **8.2**, flat 8-9 |

**The production-style AL check** (bessemer's Cahouet-Chabard with per-element (nu + gamma),
LOR diagonal grad-div blocks, CFL ~9, job 49530533):

| Configuration | Outer its | Inner its/step | s/step |
|---|---|---|---|
| CC, gamma = nu, 1 V-cycle (baseline) | 169 | -- | 115 |
| AL, gamma = 100 nu, 1 V-cycle | 99.7 (rising 72 -> 145) | -- | 62 |
| AL, gamma = 100 nu, inner 1e-1 | **13.6** (flat 12-17) | 134 | 67 |
| AL, gamma = h_K (~840 nu, bessemer's default), 1 V-cycle | 429, cap hits | -- | 170 |
| AL, gamma = h_K, inner 1e-1 | **10.2** | 428 | 129 |
| AL, gamma = h_K, inner 1e-1, no LOR grad-div blocks | 10.2 | 526 | 191 |

**Conclusion of this phase.** With grad-div plus the matching Schur weighting, the pressure
side is solved: 8-14 outer iterations, flat in time at CFL 10. Only the velocity block remains
hard.

The grad-div gamma is part of the discretization (it changes the solution), not a free solver
parameter. gamma = 100 nu is milder than bessemer's default h_K. This was never
accuracy-checked against the default; the parked state uses gamma = 100 nu.

## 6. Phase 4 -- the velocity block

### 6.1 Modified AL: tried and killed

The obvious suspect was the grad-div coupling *between* components, which the per-component
LOR drops. Options considered:

- **Robust multigrid** with vertex-star patch smoothers (Schoberl; Farrell et al.): about
  6,600 unknowns per patch at Q6 in 3D. Not realistic on GPUs.
- **Auxiliary-space methods:** research.
- **Systems AMG:** coarse spaces blind to divergence-free fields.
- **Modified AL** (Benzi, Olshanskii, Wang, IJNMF 66, 2011): keep the upper block triangle of
  the augmented velocity block over the components and apply one back-substitution sweep.

What their paper shows for modified AL:

- unsteady (sigma ~ 1/h, i.e. CFL ~ 1 only), gamma = 1: 19-26 iterations in 3D at
  nu = 1e-3, against 6-7 for the full block;
- steady: very sensitive to gamma, and it degrades as nu falls (65 iterations);
- the diagonal blocks are anisotropic (1 + gamma/nu);
- plain Cahouet-Chabard was usually faster overall "except for some cases with small viscosity
  and large time steps".

Farrell et al. call it "much more sensitive to the choice of gamma" and say it degrades with
Reynolds number.

(Correction to a claim made during the work: Leveque, He and Olshanskii 2025 use block
Gauss-Seidel over Runge-Kutta stages, not velocity components. Their velocity block uses
vertex-star multigrid.)

**Implementation.** Upper block Gauss-Seidel in `OseenLorPreconditioner`. The off-diagonal
LOR blocks gamma (d_j u_j, d_i v_i) are assembled once, at the vertex rule. One sweep costs
the same 3 V-cycles plus 3 sparse products.

**The kill test** (Taylor-Green 12^3, CFL ~10, np 8, job 49531481). The kill line, set before
the run: about 40 outer iterations with exact block solves at a useful gamma. "Exact" means
each LOR block solved to 1e-6 by GMRES.

| gamma | Block Jacobi, 1 V-cycle | Modified AL, 1 V-cycle | Block Jacobi, exact | Modified AL, exact | Inner HO solve 1e-1 |
|---|---|---|---|---|---|
| 10 nu | 91.2 | 90.9 | 83.2 | 83.0 | 32.4 |
| 100 nu | 99.7 | 89.1 | 74.8 | **64.5** | 13.6-13.7 |
| h_K | 429 (op3) | 391 (cap hit) | -- | 232 | 10.2 |

Modified AL saves about 10 % at 100 nu and nothing at 10 nu. With exact blocks it needs 64.5
outer iterations, past the kill line. Killed.

The table also shows:

- **gamma = 10 nu:** the Schur side is weak again (32 outer with a near-exact velocity solve).
- **gamma = h_K:** the dropped coupling dominates.
- **gamma = 100 nu** is the workable middle.
- **At every gamma, the exact low-order blocks are far from the high-order solve.**

### 6.2 The real bottleneck: the LOR operator as a stand-in

If solving every LOR block exactly still leaves 65-83 outer iterations while solving the
high-order block gives 13-32, the problem is the LOR operator itself, not AIR. AIR solves the
LOR blocks well: each block reaches 1e-6 in about 7-8 preconditioned iterations.

**Is it the stabilization?** The LOR keeps all of tau (w.grad u, w.grad v); the high-order
OSS removes the resolved part. LOR variants, gamma = 100 nu, exact blocks (job 49532033):

| LOR streamline-diffusion factor | 0 | 1/6 | 1/2 | 1 (recipe) | 2 | 1 + 1/2 div w |
|---|---|---|---|---|---|---|
| Outer its, exact LOR blocks | 93.0 | 79.0 | 76.0 | 74.8 | 74.5 | 74.8 |

No. Less streamline diffusion is worse, more changes nothing, and the 1/2 div w term changes
nothing. An OSS-consistent LOR operator was therefore not built.

**How the mismatch scales with CFL** (job 49533440; first two steps, exact LOR blocks):

| CFL | 1.4 | 2.8 | 4.5 | ~8 |
|---|---|---|---|---|
| Outer its, exact LOR blocks | 26.5 | 40 | 54 | 75 |

The high-order block gives about 13-15 at these CFLs. So the LOR operator costs about 2x the
outer iterations at CFL 1.4 and about 5x at CFL 8, growing roughly linearly with CFL. At
CFL 1.8 a single V-cycle per block (22.9 outer) is as good as exact LOR blocks, so the V-cycle
is not the issue.

For mass and diffusion, the LOR operator on GLL points is provably spectrally equivalent to
the high-order one, uniformly in the order. For convection there is no comparable guarantee
that we know of. These numbers show the gap opening as convection dominates (cell Reynolds
number ~140 here).

### 6.3 Inner-outer combinations: the total work is invariant

With the LOR a weak stand-in, the velocity block needs a Krylov solve on the true high-order
block, with the LOR AIR inside. Each inner iteration reduces the residual by only about 0.78.
At CFL ~10, gamma = 100 nu (jobs 49532033, 49531481):

| Velocity solve | Outer its | V-cycles/step | s/step (shared node) |
|---|---|---|---|
| 1 V-cycle | 99.7 (rising) | 299 | 62-73 |
| fixed 2 / 3 / 5 inner its | 56 / 38 / 23 | 337 / 339 / 343 | 77 / 75 / 70 |
| inner to 3e-1 | **20.7** (flat) | **327** | **62.7** |
| inner to 1e-1 | 13.6 | 365-400 | 67-72 |
| inner to 1e-1 inside modified AL | 13.7 | 365 | 72 |

About 330 V-cycles per step however they are split. The 3e-1 inner tolerance is the best mix.
A well-matched convection-diffusion preconditioner would need roughly 3-4x fewer
(an estimate: 0.1-0.3 reduction per iteration instead of 0.78).

## 7. Phase 5 -- the decisive CPU comparison against OIFS

The human asked for two cheap levers, a looser outer tolerance and a moderate CFL, and for "no
GPU runs until you've gotten something convincing working on CPUs".

Taylor-Green Re 1600, Q6/Q5 on 12^3, np 8 each, t = 0 -> 3, all runs on one shared node at
the same time (job 49533440). The baseline is bessemer's production OIFS deck (CFL 2,
step-referenced rtol 1e-3, projected initial guess): **110 s** including setup, 48 steps,
kinetic energy 30.51840 at t = 3. AL runs use gamma = 100 nu, -schur al, semi-implicit; times
are summed step times without setup.

| dt (CFL max) | Velocity solve | Outer tol | Outer its | V-cycles to t = 3 | Time to t = 3 | KE(3) vs OIFS |
|---|---|---|---|---|---|---|
| 0.3 (8.3) | inner 3e-1 | 1e-6 | 20.4 | 3183 | 554 s | -0.95 % |
| 0.3 (8.3) | inner 3e-1 | 1e-4 | 13.8 | 1437 | 295 s | -0.95 % |
| 0.3 (8.3) | inner 3e-1 | 1e-3 | 9.9 | 801 | 210 s | -0.95 % |
| 0.2 (5.7) | inner 3e-1 | 1e-6 | 22.1 | 2835 | 503 s | -0.45 % |
| 0.2 (5.7) | inner 3e-1 | 1e-3 | 8.9 | 735 | 223 s | -0.45 % |
| 0.2 (5.7) | 1 V-cycle | 1e-6 | 60.7 | 2730 | 520 s | -0.45 % |
| 0.125 (3.6) | inner 3e-1 | 1e-6 | 22.2 | 3072 | 554 s | -0.19 % |
| 0.125 (3.6) | inner 3e-1 | 1e-3 | 6.9 | 843 | **202 s** | -0.19 % |
| 0.125 (3.6) | 1 V-cycle | 1e-6 | 40.0 | 2883 | 513 s | -0.19 % |
| 0.0625 (1.8) | inner 3e-1 | 1e-6 | 14.8 | 3720 | 598 s | -0.05 % |
| 0.0625 (1.8) | inner 3e-1 | 1e-3 | 5.5 | 1131 | 314 s | -0.05 % |
| 0.0625 (1.8) | 1 V-cycle | 1e-6 | 22.9 | 3300 | 485 s | -0.05 % |

1. **The looser tolerance is free.** Going from 1e-6 to 1e-3 changes the kinetic energy by
   about one part in a million and cuts the cost 2-3x. This is the same 1e-3 production OIFS
   uses.
2. **The CFL buys nothing.** The work to t = 3 is flat in CFL: about 3,000 V-cycles at 1e-6,
   about 750-1,100 at 1e-3. The iterations per step grow as fast as the steps shrink.
3. **OIFS wins.** At matched accuracy (CFL 1.8; -0.05 % is the spatial-discretization
   difference between the two schemes) AL is 2.9x slower. At its best, CFL 3.6, it is 1.8x
   slower and less accurate.
4. **Large steps lose accuracy.** BDF2 at CFL 8 is already 1 % low in kinetic energy by
   t = 3, while OIFS at CFL 2 is not. For a resolved turbulent flow the time step the physics
   needs is near CFL 1, where OIFS is cheap.

## 8. GPU status (incomplete)

The first end-to-end GPU run (channel Re_tau 180, job 49531026) died in its first step. Every
`oseen_case` configuration did, defaults included (job 49531663). At Q6 the code hit MFEM
kernel-size limits:

1. **Interpolation values.** MFEM's runtime-size quadrature-interpolator kernels size shared
   memory for 8 points per direction on CUDA (`MAX_INTERP_1D`), unchecked. The convection rule
   k + 2k has 10 at Q6. Fixed: compile-time wind-value kernels are registered by the Oseen code
   only (bab4696).
2. **Physical derivatives.** MFEM's physical-derivative kernel at 10^3 threads per element asks
   for more registers than a block has (job 49531904). Fixed: the wind's reference derivatives
   go through the mesh-Jacobian kernels bessemer already compiles, with our own J^-1 (07a5e2e).
3. **Open: determinants.** MFEM's `MassIntegrator` (the 1/2 div w term) asks for Jacobian
   determinants at the 10-point rule, and MFEM's determinant kernel cannot launch there (too
   many registers; the shared-memory variant would need 144 KB). The fix would be our own
   sum-factorized weighted-mass kernel, with weights W detJ 1/2 div w formed from J in
   `OseenWindData::Update`.

The LOR preconditioner test passes on GPUs at np 1 and 2.

Two further GPU costs were known but not addressed:

- the host LOR assembly takes 25-90 s at 1-3M dofs, so a GPU step would need a device
  assembly kernel;
- the unpatched hypre's dropping bug (and its lack of GPU-aware MPI) at np >= 2.

No GPU time per step was ever measured. The channel break-even, for reference: OIFS takes
24 ms per step at CFL 2, so an implicit step at CFL 10 would have to stay under about 120 ms.

## 9. Conclusions

1. **Where implicit wins: steady and slowly varying flow.** DFG 3D-1Z took 80 steps against
   ~15-21k. On a cylinder the drag is right at any step. That is the case for an implicit
   option, and it holds with the recipe's preconditioner as built.
2. **The pressure side is solvable for vortical flow at large CFL**, by grad-div plus the
   (nu + gamma)-weighted Cahouet-Chabard (or PCD without OSS in N_p): 8-14 outer iterations,
   flat, at CFL 10.
3. **The velocity side is not, with LOR + AIR.** AIR handles the low-order operator fine, but
   the low-order operator does not stand in for the high-order convection-dominated operator.
   The mismatch grows about linearly with CFL. So the work per unit of simulated time is flat
   in CFL, and large steps buy nothing. The stabilization, the 1/2 div w term and the grad-div
   coupling (modified AL) were each ruled out as the cause.
4. **For turbulence-type flow the scheme loses to OIFS**: 1.8-2.9x slower on CPU, and less
   accurate at the steps where it would need to win.
5. **What would change the picture**: a velocity preconditioner for high-order continuous
   Galerkin convection-diffusion whose quality does not degrade with CFL and cell Reynolds
   number, that runs on GPUs. Candidates, all research:
   - high-order element-block or patch smoothers around the LOR AIR correction;
   - a low-order surrogate that matches the high-order convection better than the Galerkin +
     streamline-diffusion LOR (for example an upwind discretization on the LOR mesh);
   - p-multigrid with convection-robust smoothing.

## 10. If this is picked up again

- **Code:** branch `oseen-bdf2` in `bessemer-c36`, through a2c8022 (variants, `-ke`) on
  07a5e2e (GPU, WIP).
  - The worktree `bessemer-c36-mal` (branch `oseen-mal`, the same code without the GPU commit)
    and its `build/cpu` (257 MB) can be removed.
  - Neither branch is pushed to GitHub.
- **Job scripts:** `~/bessemer_jobs/os/` (DFG, Taylor-Green, PCD, AL, channel) and
  `~/bessemer_jobs/mal/` (modified AL, GPU fixes, LOR variants, CFL sweep).
- **Logs:** `~/bessemer_logs/<name>-<job>.out` with run directories beside them.
- **Before any GPU work:** the determinant kernel (section 8), a device LOR assembly, and the
  patched hypre as the default.
- **Never done:**
  - the directional do-nothing term on backflow (implicit);
  - deck integration (`time.convection: semi_implicit | implicit`);
  - AMR;
  - the pressure null space for enclosed flows;
  - an accuracy check of gamma = 100 nu against bessemer's default grad-div;
  - the bitwise every-deck comparison against main (section 0 of the plan). Nothing was
    merged, so it was not needed.
- **Cheapest win if wanted:** the implicit scheme for steady problems (3D-1Z).
