# Semi-implicit and implicit BDF2 (Oseen) schemes on the GPU: plan

Human, 2026-10-19: implement the method of a recipe from a separate 2D research code
(MFEM 4.10, hypre 3.2): semi-implicit BDF2 (one Oseen solve per step, the wind
extrapolated) and fully implicit BDF2 (Picard to a tolerance, or a fixed number of
iterations), with every linear solve on the GPU. "I want to make sure that whatever you do
does not have any impact on the outputs of the existing solvers." "Classical AMG won't work
so if no AIR then forget about it." Built on branch `oseen-bdf2`; merged only when the human
has seen the results.

This reverses the 2026-10-07 note in status.md ("not interested in implicit methods at the
moment": a CFL-free option). The Picard sweeps tried then had no Oseen preconditioner and
were never cheaper than IMEX at its stable step; the recipe's point is the preconditioner.

## 0. The guarantee: existing outputs unchanged

- New code in new files (`src/oseen/`, a new app, new tests). Existing files gain only
  what dispatches to it, behind new option values; nothing an existing deck can reach
  changes.
- Before any merge: every committed deck through main's CPU binary and the branch's,
  bitwise identical at np 1, 2 and 4 (c39cmp); the CPU suite (smoke, fast) at np 1, 2, 4;
  the GPU suite at np 1 and 2; `test/baselines.yaml` untouched.

## 1. The gate: GPU AIR on hypre 2.33

The stack's hypre is 2.33 (CUDA, its own device pool, no Umpire), the recipe's 3.2.
Static check (2026-10-19): 2.33 has the device Neumann AIR restriction
(`hypre_BoomerAMGBuildRestrNeumannAIR_assembleRdiag`), one-point interpolation
(`BuildInterpOnePntDevice`) and `ParCSRMatrixDropSmallEntriesDevice`; distance-1 and
distance-2 AIR have no device code (the recipe: only Neumann AIR with PMIS works on a GPU).
**The recipe's hypre bug is in 2.33 too**: `DropSmallEntriesDevice` compresses the off-rank
column map without rebuilding the matvec communication package (np >= 2 with `ADropTol`).
Probe (`~/bessemer_jobs/air/air_probe.cpp`): a 3D Q1 operator like the recipe's LOR one
(vertex-rule diffusion, Gauss convection + streamline diffusion with a Q6 parent tau),
FGMRES + one V-cycle, 48^3 to 144^3, nu 1e-1 to 1e-5, constant and vortex winds, AIR against
classical AMG, np 1 and 2, `ADropTol` 1e-2 and 0. RESULT at np 1 (job 49498361,
2026-10-20, one H100): **GPU AIR works in 3D** -- every run converged, with the true
residual checked (1e-8), and the convection-dominated counts barely grow with the mesh:

| 3D Q1 LOR-like operator | AIR, 118k -> 912k dofs | classical AMG |
|---|---|---|
| nu 1e-3, constant wind, steady | 10 -> 12 | 39 -> 37 |
| nu 1e-5, constant wind, steady | 21 -> 24 | 74 -> 100 |
| nu 1e-5, vortex wind, steady | 27 -> 32 | 133 -> 164 |
| nu 1e-5, CFL 2 (with the mass) | 7 -> 7 | 17 -> 17 |
| nu 1e-1 (diffusion-dominated), steady | 23 -> 43 | 14 -> 15 |

AIR setup 0.05-0.07 s (0.2-0.4 s for the steady high-Peclet cases at 912k); classical AMG
wins where diffusion dominates, hence the recipe's switch at Re_h = 3. The LOR operator's
MFEM host assembly took 25 s at 912k dofs: in 3D it needs a device kernel. The np 2 half
(the dropping bug) is still queued; at np > 1 the code sets ADropTol to 0 until then.

## 2. The recipe mapped onto bessemer

| Recipe | bessemer today | Plan |
|---|---|---|
| Taylor-Hood Q_k/Q_{k-1}, GLL nodes, curved quads | the same, 2D/3D, hexes, Q6/Q5 default, curved, AMR (hanging nodes) | as is |
| GLL-lumped velocity mass | `discretization.mass: collocated` (OIFS default) | required by the new schemes |
| skew convection C(w) + 1/2 (div w) M | convective form only, explicit (IMEX) or OIFS; skew "gated on MFEM" -- that TRAP is MFEM's NLF integrator, not the linear form with a given wind used here | new: linear PA operator per velocity component (the velocity space is byNODES, so a scalar space applies to each component block) |
| grad-div gamma = nu, left out of F^-1 | gamma = 6 h_K / k_u, kept in the LOR source (without it LOR-AMG needed 10-30x the iterations, 2026-10-07) | the new schemes default to the recipe's gamma = nu: a component-wise F^-1 cannot hold gamma = 6 h/k |
| directional do-nothing, implicit on backflow steps | directional do-nothing, explicit (IMEX) | implicit: assembled (host) only on steps with backflow |
| OSS stabilization (tau-weighted projection, PA) | none (high order + grad-div) | new, only in the new schemes |
| forces from the residual of the discrete momentum equation | `MomentumResidual` + `post/body_force` (the same John formula) | reuse |
| BDF2, BDF1 first step | trapezoidal starter -> BDF2 (-> BDF3) | BDF1 first step, as the recipe |
| FGMRES + block upper triangular P | `BlockStokesPC` (UpperTri) in FGMRES | reuse |
| F^-1: one AIR (or AMG) V-cycle on a mixed-quadrature LOR operator sigma D + nu K_lor + C_lor + SD_lor (parent tau), host assembly every 10 steps | LOR-AMG on mass + diffusion + grad-div (MFEM batched LOR, device) | new LOR operator per component; host assembly first (the recipe's), a device kernel for convection + SD if the host cost dominates (expected in 3D: 12 s per rebuild at 2.4M DOFs in 2D) |
| S^-1: PCD, A_p = B D^-1 B^T assembled by SpGEMM, one AMG V-cycle | Cahouet-Chabard: sigma (B M^-1 B^T)^-1 applied matrix-free by inner CG with an LOR-AMG pressure Laplacian | A_p^-1 by the CC machinery: assembled B D^-1 B^T at Q5 in 3D would hold ~1e9 nonzeros on the DFG 3D mesh (each pressure row reaches the pressure dofs of 27 elements) |
| N_p: pressure-space convection + OSS + the inflow term | none | new (PA), the recipe's |
| M_p^-1: Chebyshev(3) on the consistent pressure mass | `precond/mass_inverse.hpp` | reuse or add Chebyshev |
| device coefficients (wind, tau, 1/2 div w at quadrature points) | QuadratureInterpolator / geometric factors in places | new, the recipe's |

## 3. Order of work

1. The high-order Oseen operator (component-wise C(w) + 1/2 div w + OSS, device
   coefficients); tests: PA against the assembled operator, OSS positive semidefinite on a
   small mesh, CPU against GPU.
2. The LOR operator and its AIR/AMG V-cycle (the `Re_h >= 3` switch).
3. PCD.
4. The stepper (semi; implicit with a fixed-iteration mode), a driver on DFG 2D-3 (John's
   reference: c_d,max 2.950921575, c_l,max 0.47795, dp(8) -0.1116), CPU then GPU.
5. 3D: DFG 3D-3Z against FeatFlow and the OIFS runs of 2026-10-19 at matched accuracy;
   the LOR assembly cost; a device kernel if needed.
6. Deck integration (`time.convection: semi_implicit | implicit`), validation of section 0,
   write-up.

## 4. Status (2026-10-20 morning)

Built on `oseen-bdf2`: `src/oseen/` -- `OseenWindData` (device quadrature data: w, 1/2
div w, tau terms), `ScalarOseenOperator` + `ComponentwiseOperator` (the Oseen convection
with OSS per velocity component), `OseenLorPreconditioner` (host LOR assembly, AIR or AMG
per component), `OseenPcdSchurPC` (PCD on the CC machinery, N_p with OSS and the inflow
term), `OseenStepper` (semi-implicit; implicit with a Picard tolerance or a fixed count);
`apps/oseen_cylinder` (DFG 2D-3); `test/oseen_operator_test` (O1-O5, CPU np 1/2/4).

Found on the way: the OSS term is positive semidefinite on affine elements only. The
GLL-lumped projection mass bounds the Gauss mass from above when the Jacobian is constant;
on a strongly curved test mesh S has eigenvalues down to -0.3 % of its largest. The recipe
reported positivity on curved quadrilaterals (milder curvature around a cylinder).

Not yet: the directional do-nothing term (implicit, on backflow steps), the deck
integration, the device LOR kernel for 3D, the pressure null space (enclosed flows).
