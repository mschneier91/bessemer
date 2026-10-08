# IMEX vs. semi-implicit rotational time stepping in bessemer

This document explains the two Navier–Stokes time-stepping schemes bessemer
offers — the **IMEX convective scheme** and the **semi-implicit rotational
scheme** — exactly as they are implemented, and reports head-to-head
comparisons on the DFG 2D-3 benchmark (flow around a cylinder with
time-dependent inflow, V. John's reference case): fixed steps on the base
mesh and with adaptive mesh refinement (§6), and a work-precision study under
CFL-controlled steps on nested meshes (§7). The questions: **which scheme is
better, and can the semi-implicit scheme take time steps beyond the
convective CFL limit and still give a better answer?**

## Summary

**On DFG 2D-3 the semi-implicit rotational scheme has no advantage over the
IMEX scheme** (studies of 2026-10-06 to 2026-10-08).

- **Same step limit.** Both schemes treat transport explicitly — IMEX the
  velocity, the rotational scheme the vorticity (§5) — and both lose accuracy
  at the same CFL number. With BDF2/EXT2 both are accurate up to a
  (Nek-scale) CFL of 0.7 and both degrade at 0.9 (§7.3). The rotational
  scheme cannot step past the CFL limit and still give a better answer.
- **Same work per step, slightly more time.** At equal CFL both take the
  same steps and the same solver iterations; the rotational scheme costs
  7–9% more per step (§7.4).
- **Accuracy.** Lift is comparable (peak value 0–20% worse for the
  rotational scheme, peak time sometimes better). The drag peak is ~50–70×
  worse on the medium mesh (2.2e-4 vs 3–4e-6 relative): a spatial error of
  the rotational form that does not change with Δt (§7.3).
- **Failure modes.** Past the limit both fail *silently* on this flow. IMEX
  (with EXT3) does not blow up — the CFL controller keeps shrinking Δt — but
  its measured CFL creeps above the target, which is detectable. The
  rotational scheme shows nothing: steady CFL, bounded forces, and no vortex
  shedding (§7.3, §6.1).
- **Extrapolation order.** BDF2/EXT3 fails silently at CFL 0.7 in both
  schemes where BDF2/EXT2 is accurate, and gains no accuracy here (the error
  is spatial). The case-level default is therefore BDF2/EXT2 (2026-10-08).
- **Larger steps save less than the step count suggests.** From CFL 0.5 to
  0.7 the step count drops 29% but the wall time only ~10–15%: Jacobi-PCG's
  iterations per step grow as $\sigma = 1.5/\Delta t$ shrinks, so the total
  velocity-solve work over a run is nearly independent of $\Delta t$ (§7.4).

*Scope:* one 2D laminar flow (Re ≤ 100); Q3/Q2; meshes up to ~19k unknowns
(the finest level was started but stopped once the picture was clear);
Dirichlet outflow standing in for a proper rotational-form outflow
condition. Possible strengths of the rotational form — energy stability,
robustness at high Re or under-resolution, 3D — are not tested here.

---

## 1. What both schemes share

Both solve the incompressible Navier–Stokes equations

$$\partial_t u + (u\cdot\nabla)u - \nu\Delta u + \nabla p = f,\qquad \nabla\cdot u = 0$$

with the same spatial discretization and the same outer solver:

- **Space.** Continuous Taylor–Hood elements on quadrilaterals/hexahedra:
  velocity $Q_k$, pressure $Q_{k-1}$ ($k = 3$ by default, so $Q_3/Q_2$).
  Partial assembly throughout; the nonlinear term (convection in one scheme,
  rotation in the other) is integrated with an over-integrated (dealiased)
  Gauss–Legendre rule of order $3k$.
- **Time.** Second-order backward differentiation (BDF2) with variable-step
  coefficients. The time derivative, the viscous term and the pressure are
  always implicit. The first step is a trapezoidal (Crank–Nicolson-type)
  starter.
- **Grad-div stabilization (optional, used in the study).** The term
  $\gamma(\nabla\cdot u, \nabla\cdot v)$ with $\gamma = c_{gd}\,h_K$ (order
  $h$, element-wise) is added to the implicit velocity block. It penalizes
  the discrete divergence and improves mass conservation; it never enters the
  Schur-complement preconditioner.
- **Linear algebra.** Each step solves **one linear saddle-point system**
  $$\begin{pmatrix} A & B^T \\ B & 0\end{pmatrix}\begin{pmatrix} u^{n+1} \\ p^{n+1}\end{pmatrix} = \begin{pmatrix} b \\ g\end{pmatrix}$$
  monolithically with FGMRES and a block upper-triangular preconditioner whose
  pressure block is a Cahouet–Chabard Schur approximation. There is no
  pressure projection/splitting and no Newton iteration in either scheme.

The schemes differ **only in how the nonlinear term is treated** — and that
single choice decides the stability limit, the structure of $A$, the
preconditioner, and what the computed pressure means.

## 2. The IMEX convective scheme (`physics.convective_form: convective`)

**Idea:** treat everything linear implicitly and the nonlinear convection
term **explicitly**, by extrapolating it from previous time levels.

With $N(u) = (u\cdot\nabla)u$ (convective form), step $n \to n+1$ is

$$\frac{\beta_0 u^{n+1} + \beta_1 u^{n} + \beta_2 u^{n-1}}{\Delta t} - \nu\Delta u^{n+1} + \nabla p^{n+1} = f^{n+1} - \big(2N(u^{n}) - N(u^{n-1})\big),\qquad \nabla\cdot u^{n+1} = 0,$$

written here for constant $\Delta t$ ($\beta_0 = 3/2$, $\beta_1 = -2$,
$\beta_2 = 1/2$); bessemer uses the variable-step BDF2 and extrapolation
(EXT2) weights computed from the actual step history. The first step uses
EXT1 ($N(u^0)$) inside the trapezoidal starter.

Consequences:

- **The implicit operator is the Stokes operator.**
  $A = \sigma M + \nu K + \gamma G$ with $\sigma = \beta_0/\Delta t$ is
  symmetric positive definite and does not change with the flow. The
  velocity block is preconditioned by a Jacobi–Chebyshev smoother (at the
  small, mass-dominated steps the CFL limit forces, it beats bessemer's
  default LOR-AMG V-cycle by 3×, §6.3); the Schur block by Cahouet–Chabard,
  $\hat S^{-1} = \nu M_p^{-1} + \sigma (BM_v^{-1}B^T)^{-1}$. Cheap per step,
  robust, and the iteration count barely depends on the flow.
- **Explicit convection imposes a CFL stability limit.** The extrapolated
  term is a (two-step) explicit treatment of advection, stable only if
  $\Delta t \lesssim C\,h/(k^2|u|)$. bessemer measures this with Nek5000's
  directional CFL number
  $$\mathrm{CFL} = \Delta t \max_{\text{GLL nodes } n} \sum_d \frac{\big|(J_n^{-1}u_n)_d\big|}{\Delta\xi_d(n)}$$
  ($J$ the element Jacobian, so $J^{-1}u$ is the velocity in reference
  coordinates; $\Delta\xi_d(n)$ the local reference spacing of the
  Gauss–Lobatto nodes at node $n$ along direction $d$, one-sided at element
  ends — Nek's definition, so Nek's CFL targets carry over). Beyond the
  limit the scheme **blows up** — the solver still converges each step, the
  solution just grows without bound.
- **AMR makes the limit tighter.** Halving $h$ near the cylinder halves the
  stable $\Delta t$ for the *whole* domain (one global step size).
- **Accuracy.** Second order in time; the splitting error of the
  extrapolation is $O(\Delta t^2)$. The pressure $p$ is the static pressure.

## 3. The semi-implicit rotational scheme (`physics.convective_form: rotational`)

**Idea:** rewrite the nonlinear term in rotation form and make it *linearly
implicit*: the vorticity is lagged (extrapolated), but the velocity it acts
on is the unknown $u^{n+1}$.

The identity $(u\cdot\nabla)u = (\nabla\times u)\times u + \nabla\tfrac12|u|^2$
moves the kinetic energy into a modified pressure, the **Bernoulli head**
$P = p + \tfrac12|u|^2$:

$$\partial_t u + (\nabla\times u)\times u - \nu\Delta u + \nabla P = f,\qquad \nabla\cdot u = 0.$$

Step $n \to n+1$ (constant $\Delta t$ shown):

$$\frac{\beta_0 u^{n+1} + \beta_1 u^{n} + \beta_2 u^{n-1}}{\Delta t} + \omega^*\times u^{n+1} - \nu\Delta u^{n+1} + \nabla P^{n+1} = f^{n+1},\qquad \nabla\cdot u^{n+1} = 0,$$

with $\omega^* = \nabla\times w^*$ and $w^* = 2u^n - u^{n-1}$ (EXT2,
variable-step weights in the code; $w^* = u^0$ on the first step, where the
trapezoidal starter splits $\omega^*\times u$ half implicit / half explicit
like the viscous term). Nothing is explicit after the first step.

Consequences:

- **Energy-stable for any step — but not CFL-free (§5).** The rotation term
  is skew:
  $(\omega^*\times u, u) = 0$ for *any* $\omega^*$ and $u$, so it does no
  work. Testing the scheme with $u^{n+1}$ gives the same discrete energy
  estimate as unsteady Stokes with BDF2: the scheme is energy-stable for
  every $\Delta t$. The step size is limited by **accuracy only**.
- **The implicit operator changes every step and is nonsymmetric.**
  $A = \sigma M + \nu K + \gamma G + N(\omega^*)$, $N$ skew. The velocity
  block is preconditioned by point-block Jacobi (exact inverses of the nodal
  $d\times d$ blocks $\mathrm{diag}(A) + [\,\omega^*]_\times$, which see the
  rotation) inside a short GMRES (`solver.rotation_pc: pbj_krylov`).
  The Schur complement $BA^{-1}B^T$ now depends on the rotation number
  $\mu = |\omega^*|/\sigma \approx \tfrac23|\omega^*|\Delta t$: Cahouet–Chabard
  ignores $N$ and degrades as $\mu$ grows, so bessemer can switch per step to
  Olshanskii's **rotating-Darcy tensor** preconditioner,
  $\hat S^{-1} \approx \nu M_p^{-1} + \big(-\nabla\cdot(T\nabla)\big)^{-1}$,
  $T = (\sigma I + [\omega^*]_\times)^{-1}$
  (`solver.rotation_schur: cc|tensor|auto`; CLAUDE.md, "Rotation-aware Schur
  preconditioner"). **Large steps move the cost from the number of steps to
  the cost of each linear solve.**
- **Accuracy.** Second order in time (the lagged vorticity commits an
  $O(\Delta t^2)$ error). In space the rotation form is known to be less
  accurate than the convective form: the solver's pressure is $P$, which
  contains $\tfrac12|u|^2$ and is far less smooth near walls, and its error
  feeds back into the velocity with a factor $1/\nu$ (Layton, Manica, Neda,
  Olshanskii & Rebholz, *J. Comput. Phys.* 228, 2009). Grad-div
  stabilization is the standard remedy: on the steady DFG 2D-1 benchmark
  bessemer measured a drag error of $9.4\times10^{-4}$ without it and
  $1.0\times10^{-4}$ with $c_{gd} = 1$ (the convective scheme's level).
- **Pressure.** The solve produces $P$; `Pressure()` reports the static
  pressure $p = P - I_h(\tfrac12|u|^2)$. A do-nothing outflow condition acts
  on $P$, not $p$ — a slightly different outflow model (immaterial for the DFG
  cases, measured).

## 4. Side by side

| | IMEX convective | semi-implicit rotational |
|---|---|---|
| Nonlinear term | $(u\cdot\nabla)u$, extrapolated, on the right-hand side | $\omega^*\times u^{n+1}$, in the matrix, $\omega^*$ extrapolated |
| Stability | CFL-limited ($\Delta t \lesssim C h/(k^2|u|)$) | energy-stable for any $\Delta t$ |
| Velocity block | SPD, flow-independent; LOR-AMG | nonsymmetric, re-formed every step; point-block Jacobi + GMRES |
| Schur block | Cahouet–Chabard (robust) | Cahouet–Chabard, degrading with $\mu$; tensor/auto switch |
| Cost per step | low, nearly constant | higher, grows with $\Delta t$ (via $\mu$) |
| Pressure solved for | static $p$ | Bernoulli head $P$ (static $p$ recovered) |
| Spatial accuracy | standard | needs grad-div to match |

## 5. Why the semi-implicit rotational scheme is CFL-limited too

The energy argument in §3 is correct — the lagged rotation term does no work,
so the discrete kinetic energy stays bounded for every $\Delta t$ — but it
bounds only the *energy*, not the *accuracy* or the smoothness of the
solution. Look at what the scheme does to the **vorticity**. In 2D, for a
divergence-free $u$,

$$\nabla\times(\omega^*\,\hat z\times u^{n+1}) = \nabla\cdot(\omega^* u^{n+1}) = (u^{n+1}\cdot\nabla)\,\omega^*,$$

so taking the curl of the semi-implicit momentum equation gives the
vorticity equation

$$\frac{\beta_0\omega^{n+1} + \beta_1\omega^n + \beta_2\omega^{n-1}}{\Delta t} + (u^{n+1}\cdot\nabla)\,\underbrace{(2\omega^n - \omega^{n-1})}_{\omega^*} = \nu\Delta\omega^{n+1}.$$

The advecting velocity is new, but the **advected quantity is extrapolated
from old time levels**: vorticity transport is treated *explicitly*, with the
same BDF2/EXT2 structure the IMEX scheme uses for velocity transport. The
semi-implicit rotational scheme is, in effect, an IMEX scheme for the
vorticity — and explicit advection is CFL-limited, whatever the energy
estimate says. (In 3D the vortex-stretching part $(\omega^*\cdot\nabla)u^{n+1}$
is implicit in $u$, but the advection of $\omega$ is still explicit.)

What the energy bound changes is only **how the instability shows up**:
the IMEX scheme blows up (forces $\to\infty$, a hard failure), while the
rotational scheme's growth saturates at the energy bound and leaves a bounded
but meaningless solution full of grid-scale vorticity — a *silent* failure.

Two ways to make the rotational form genuinely CFL-free: converge the
nonlinearity within each step (Picard sweeps with $w^* \leftarrow u^{n+1}$, or
Newton; an experimental Picard option was measured in §6.2 and removed), or use the Oseen
linearization of the convective form, $(w^*\cdot\nabla)u^{n+1}$, where the
transported field is the unknown and transport is fully implicit (not
implemented in bessemer).

## 6. The study: DFG 2D-3 (flow around a cylinder, time-dependent inflow)

**Problem** (Schäfer & Turek 1996; reference values from V. John, *Reference
values for drag and lift of a two-dimensional time-dependent flow around a
cylinder*, IJNMF 44, 2004). Channel $[0, 2.2]\times[0, 0.41]$, cylinder of
diameter $D = 0.1$ centred at $(0.2, 0.2)$, $\nu = 10^{-3}$, parabolic inflow
$U(0, y, t) = 4\cdot1.5\,\sin(\pi t/8)\,y(0.41 - y)/0.41^2$, no-slip walls
and cylinder, do-nothing outflow, $u = 0$ at $t = 0$, $t \in [0, 8]$. The
Reynolds number rises to 100 at $t = 4$ and falls again; vortex shedding
starts near $t \approx 4$–5. Quantities (drag/lift coefficients referred to
the maximal mean inflow velocity 1):

| | reference (John 2004) |
|---|---|
| $c_{D,\max}$ | 2.950921575 at $t = 3.93625$ |
| $c_{L,\max}$ | 0.47795 at $t = 5.693125$ |
| $\Delta p(t = 8) = p(0.15, 0.2) - p(0.25, 0.2)$ | −0.1116 |

Forces by John's volume-integral formulation (bessemer's `post/body_force`),
evaluated every step; maxima located by a parabola through the discrete
maximum and its neighbours.

**Discretization (both schemes).** $Q_3/Q_2$ on the DFG cylinder-channel mesh
(208 curved quadrilaterals, `mesh/cylinder_channel`); grad-div
$\gamma = 1\cdot h_K$ (order $h$) in both schemes; fixed steps; outer FGMRES
to $10^{-8}$. Velocity preconditioner: IMEX — Jacobi–Chebyshev (see §6.3);
rotational — point-block Jacobi + GMRES. Schur: Cahouet–Chabard for IMEX,
`auto` (Cahouet–Chabard / rotating-Darcy tensor) for the rotational scheme.
All runs on 4 MPI ranks of one desktop CPU (`apps/dfg_cylinder -c 3`).

**CFL numbers** reported below: the largest value seen during the run
(sampled every 5 steps). The base-mesh runs of §6.1–6.2 were measured with
bessemer's EARLIER definition, $k^2\max_q\sum_d|(J^{-1}u)_d|$ at
Gauss–Legendre points, which is about 2.5× the Nek-style number of §2 now
used (Q3: 9 vs 3.62 per unit $|u|/h$ at element edges); divide those columns
by ~2.5. Later sections use the Nek-style number.

### 6.1 Base mesh (no AMR): the stability threshold

| scheme | $\Delta t$ | outcome | max CFL | $c_{D,\max}$ (err) | $t(c_{D,\max})$ err | $c_{L,\max}$ (err) | $t(c_{L,\max})$ err | $\Delta p(8)$ err | wall |
|---|---|---|---|---|---|---|---|---|---|
| IMEX | 0.001 | ok | 1.31 | 2.94451 (0.22%) | 0.0013 | 0.5117 (7.1%) | 0.034 | 1.0% | 153 s |
| rotational | 0.001 | ok | 1.31 | 2.94398 (0.24%) | 0.0002 | 0.5107 (6.9%) | 0.027 | 0.16% | 174 s |
| IMEX | 0.0015 | ok | 1.96 | 2.94452 (0.22%) | 0.0012 | 0.5127 (7.3%) | 0.033 | 0.94% | 134 s |
| rotational | 0.0015 | ok | 1.96 | 2.94399 (0.24%) | 0.0002 | 0.5113 (7.0%) | 0.027 | 0.30% | 138 s |
| IMEX | 0.002 | **blew up** at $t = 3.28$ | — | — | — | — | — | — | — |
| rotational | 0.002 | **wrong** (ran to $t = 8$) | 2.76 | 3.949 (34%) | 0.83 | 0.195 (59%) | 1.10 | 0.38% | 153 s |
| IMEX | 0.0025 | **blew up** at $t = 2.80$ | — | — | — | — | — | — | — |
| rotational | 0.0025 | **wrong** (ran to $t = 8$) | 3.35 | 4.906 (66%) | 0.41 | 0.523 (9.4%) | 1.78 | 7.2% | 141 s |

- **Both schemes become unstable at the same step size**, $\Delta t \approx
  0.0017$ (CFL ≈ 2–2.7 in the earlier measure, ≈ 0.8–1.1 Nek-style, reached
  as the inflow nears its peak). At $\Delta t = 0.002$ the rotational solution leaves the converged
  one at $t \approx 3.03$, the IMEX one blows up at $t = 3.28$; a close-up at
  $\Delta t = 0.004$ shows the deviation growing by a factor ≈ 1.9 per step
  — an exponential instability, not a large truncation error (§5).
- **The rotational scheme fails silently.** It finishes the run, the forces
  stay finite, $\Delta p(8)$ is even nearly right (the instability fades
  once the inflow decays below the threshold) — but $c_{D,\max}$ is off by
  34–66% and the lift peak is in the wrong place by a full time unit.
- **Below the threshold both are converged in time** ($\Delta t$ = 0.0015
  and 0.001 agree to $10^{-5}$) and agree with each other to $2\times10^{-4}$
  in $c_{D,\max}$. The remaining error vs John is spatial: this mesh is too
  coarse for the lift (7%).
- **Cost at equal stable $\Delta t$ is the same** (134 vs 138 s at
  $\Delta t = 0.0015$). The rotational scheme's rotation number stayed below
  0.4, so `auto` never left Cahouet–Chabard.

### 6.2 Making the rotational scheme step past the threshold: Picard sweeps

An experimental option (since removed, 2026-10-07) re-solved each step $K$
times with $w^* \leftarrow u^{n+1}$ (§5):

| sweeps | $\Delta t$ | max CFL | $c_{D,\max}$ (err) | $t(c_{D,\max})$ err | $c_{L,\max}$ (err) | wall |
|---|---|---|---|---|---|---|
| 1 | 0.002 | 2.62 | 2.94397 (0.24%) | 0.0001 | 0.5094 (6.6%) | 256 s |
| 1 | 0.0025 | 3.27 | 2.94397 (0.24%) | 0.0001 | 0.5088 (6.5%) | 232 s |
| 1 | 0.004 | 5.22 | 3.686 (25%) ✗ | 0.020 | 0.412 (14%) | 194 s |
| 3 | 0.004 | 5.23 | 2.94395 (0.24%) | 0.0001 | 0.5063 (5.9%) | 366 s |
| 3 | 0.008 | 13.5 | 4.920 (67%) ✗ | 0.031 | 0.381 (20%) | 279 s |

The sweeps do move the threshold out — roughly ×1.5 per sweep, so the
stable $\Delta t$ grows about in proportion to the number of solves — but
each sweep is a full saddle-point solve (the outer iteration count per step
doubles to quadruples). **No configuration beat the plain schemes at
$\Delta t = 0.0015$ (≈135 s) on cost for the same answer.** Large steps would
need many sweeps, or Newton, and Picard's contraction itself degrades as the
CFL number grows.

### 6.3 A solver finding along the way

With the default Cahouet–Chabard velocity preconditioner (LOR-AMG on
$\sigma M + \nu K$), the IMEX scheme needed ~50 outer iterations per step
(460 s at $\Delta t = 0.0015$); with **Jacobi–Chebyshev** it needs ~15
(134 s) for the identical answer. At CFL-limited steps the velocity block is
mass-dominated ($\sigma = 1.5/\Delta t \approx 1000$), which is exactly
where a diagonal-based smoother is near-exact and the LOR-AMG — which at the
time also omitted the grad-div term from its low-order operator (fixed since;
CLAUDE.md, H5) — is not. All IMEX
numbers in this study use Jacobi–Chebyshev (`apps/dfg_cylinder -apc
jacobi_chebyshev`). Following up, a sweep over the viscous ratio
(`bench/bench_velocity_pc`) found CG with Jacobi to a loose tolerance at least
as good everywhere and robust in the viscous-dominated regime too, and it is
now bessemer's case-level default (`solver.a_pc: jacobi_pcg`; CLAUDE.md,
"Velocity block PC default").

### 6.4 With adaptive mesh refinement (do-nothing outflow, fixed steps)

AMR events every 0.25 time units (anisotropic, θ = 0.85, element cap 800),
same discretization and solvers as §6.1–6.3 (rotational: `auto` Schur, PBJ +
GMRES velocity PC). Each run adapts its own mesh, and the wall times came
from runs sharing the machine, so treat cost as indicative only.

| scheme | $\Delta t$ | outcome | elements | $c_{D,\max}$ err | $c_{L,\max}$ err | $t(c_{L,\max})$ err | $\Delta p(8)$ err | wall |
|---|---|---|---|---|---|---|---|---|
| IMEX | 4e-4 | ok | 803 | 1.9e-5 | 2.8% | 0.009 | 0.06% | 3280 s |
| rotational | 4e-4 | ok | 800 | 1.0e-4 | 4.4% | 0.024 | 0.26% | 2691 s |
| IMEX | 8e-4 | **blew up** at $t = 4.01$ | — | — | — | — | — | — |
| rotational | 8e-4 | ok | 775 | 1.0e-4 | 4.4% | 0.024 | 0.25% | 2303 s |

The rotational scheme ran at twice the step that blew IMEX up, with the same
errors as its own $\Delta t = 4\cdot10^{-4}$ run. Read with §7 in mind:
those errors were large (lift 4.4% off, 1.6× IMEX's), dominated by
something other than the time step — these runs used the do-nothing
outflow, which acts on the Bernoulli head in the rotational form (§7) — so
the extra time error at $8\cdot10^{-4}$ did not show. On nested meshes with
a Dirichlet outflow for both schemes (§7), the rotational scheme's accurate
range ends at the same CFL number as IMEX's.

## 7. Work-precision study (2026-10-08)

**Question:** to reach a given accuracy, how much wall time does each scheme
need? A fair comparison holds everything else equal and lets each scheme use
its best solver configuration.

**Setup.**
- **Problem:** DFG 2D-3 as in §6, but with a **Dirichlet outflow** (the
  inflow profile $U(t)$ imposed at $x = 2.2$) for both schemes. Do-nothing
  acts on the Bernoulli head in the rotational form, which is a different —
  and, for this form, notoriously poor — outflow condition; until the
  rotational scheme gets its own outflow term, both use the same Dirichlet
  outflow. Errors are against John's reference values. The outflow bias this
  introduces is negligible: IMEX on M1 at CFL 0.5 with do-nothing vs
  Dirichlet outflow differs by $1\cdot10^{-8}$ (relative) in $c_{D,\max}$,
  $7\cdot10^{-7}$ in $c_{L,\max}$, $2\cdot10^{-6}$ in $\Delta p(8)$. $\Delta
  p(8)$ is published to four digits (−0.1116), so its error cannot resolve
  below ~$10^{-3}$ and is not used for ranking.
- **Meshes** (nested, `apps/dfg_cylinder -mref L`: every cell count ×$2^L$,
  every grading ratio to the power $2^{-L}$, cylinder exact at each level):
  M0 208 cells (~4.9k unknowns), **M1 832 cells (~18.9k)**, M2 3328 cells
  (~74k). $Q_3/Q_2$, grad-div $\gamma = h_K$, outer FGMRES to $10^{-8}$,
  CFL-controlled steps (`time.step_control: cfl`, Nek-scale CFL number).
- **Timing:** 4 MPI ranks per run; two runs at a time, each rank pinned to
  its own physical core of the 8-core desktop, with a constant two-job load
  (`bench/dfg3/run_queue.sh`). A neighbour slows IMEX by 8.6% and the
  rotational scheme by 6.6% — the same within 2%, so ratios are fair.
  `step_wall` is the time inside `Case::Step()` (the forces and CFL
  monitoring are excluded). Runs timed at different times of day differ by up
  to ~10–15% from interactive use of the machine; iteration counts are
  immune to that.
- **What ran:** the M1 sweep. M0 was dropped (at ~1.2k unknowns per rank its
  timings measure MPI latency, not the schemes). M2 costs ~6× M1 per step
  (~3 h per run at CFL 0.7); its CFL 0.7 pair was started and then stopped
  once the M1 results settled the question.

### 7.1 Solver configuration (calibration)

M1, CFL 0.5, BDF2/EXT3, $t \in [0, 4.5]$ (through the drag peak):

| scheme | velocity PC | Schur | step wall | outer its/step |
|---|---|---|---|---|
| IMEX | **Jacobi-PCG** | Cahouet–Chabard | **497 s** | 11.5 |
| IMEX | Jacobi–Chebyshev | Cahouet–Chabard | 666 s | 15.9 |
| IMEX | LOR-AMG | Cahouet–Chabard | stopped, ~3× slower | 28 |
| rotational | **PBJ + GMRES** | **Cahouet–Chabard** | **545 s** | 11.0 |
| rotational | symmetric (Jacobi-PCG) | Cahouet–Chabard | 576 s | 12.4 |
| rotational | PBJ + GMRES | `auto` | 594 s | 11.0 |

The `auto` Schur never left Cahouet–Chabard: the rotation number peaked at
$\mu = 0.13$ against the switch threshold 20. At CFL-limited steps $\sigma =
1.5/\Delta t$ dominates $|\alpha\omega^*|$, and the tensor (Olshanskii)
preconditioner only pays for $\mu \gg 1$ — steps far past the CFL limit,
where the scheme is inaccurate anyway (§7.3). Its 9% higher time was the
machine, not $\mu$: both runs did identical iterations in every time window,
and before the overnight pause the two agree to ≤1%.

### 7.2 M1 results

| scheme | EXT | CFL | outcome | steps | step wall | s/step | outer/step | $c_{L,\max}$ err | $t(c_{L,\max})$ err | $c_{D,\max}$ err |
|---|---|---|---|---|---|---|---|---|---|---|
| IMEX | 3 | 0.3 | ok | 17976 | 1281 s | 0.071 | 9.5 | 3.6e-3 | 4.3e-4 | 4.3e-6 |
| IMEX | 3 | 0.5 | ok | 10787 | 831 s | 0.077 | 11.2 | 3.5e-3 | 5.1e-4 | 4.4e-6 |
| IMEX | 3 | 0.7 | **wrong** | 9219 | 1179 s | 0.128 | 16.1 | 43% | 0.42 | 1.5e-2 |
| IMEX | 3 | 0.9 | **wrong** (stopped) | — | — | — | — | — | — | — |
| IMEX | 2 | 0.7 | ok | 7706 | 821 s | 0.107 | 12.5 | 4.5e-3 | 1.7e-4 | 3.3e-6 |
| IMEX | 2 | 0.9 | degraded | 6093 | 779 s | 0.128 | 15.0 | 5.9% | 0.018 | 3.6e-3 |
| rotational | 3 | 0.3 | ok | 17978 | 1371 s | 0.076 | 9.1 | 4.2e-3 | 4.5e-4 | 2.2e-4 |
| rotational | 3 | 0.5 | ok | 10789 | 1045 s* | 0.097* | 10.8 | 4.1e-3 | 3.7e-4 | 2.2e-4 |
| rotational | 3 | 0.7 | **wrong** | 6828 | 1216 s | 0.178 | 18.2 | 58% | 0.91 | 47% |
| rotational | 2 | 0.7 | ok | 7708 | 898 s | 0.117 | 12.1 | 4.7e-3 | 3.4e-4 | 2.2e-4 |
| rotational | 2 | 0.9 | degraded | 6021 | 966 s | 0.161 | 15.2 | 2.1% | 0.026 | 6.7e-2 |

\* timed on a different neighbour than its IMEX counterpart (see §7.4).
Raw results: `bench/dfg3/results_2026-10-08/`.

### 7.3 Accuracy and the step limit

- **Below the limit the time error is negligible on M1.** CFL 0.3 and 0.5
  give the same errors for each scheme; the errors are spatial.
- **The drag peak is the rotational form's weak spot:** 2.2e-4 at every CFL
  and both EXT orders, against 3–4e-6 for IMEX — a spatial error of the
  form. From M0 to M1 IMEX's drag error fell ~500×, the rotational scheme's
  ~10× (§6.1: 2.2e-3 vs 2.4e-3 on M0). Lift is comparable.
- **BDF2/EXT3 fails silently at CFL 0.7 in both schemes; BDF2/EXT2 is
  accurate there.** Lift histories at CFL 0.7 against an accurate run:

  | $c_L$ at $t$ = | 4.0 | 5.0 | 5.5 | 6.0 |
  |---|---|---|---|---|
  | accurate (rotational, CFL 0.5) | −0.047 | +0.096 | −0.476 | −0.184 |
  | IMEX, EXT2 | −0.047 | +0.096 | −0.477 | −0.185 |
  | rotational, EXT3 | +0.018 | +0.074 | +0.097 | +0.153 |
  | IMEX, EXT3 | −0.062 | −0.035 | −0.551 | −0.038 |

  The rotational/EXT3 run leaves the solution at $t \approx 1.7$ (inflow
  $U \approx 0.94$), its drag plateaus near 3.66 against the true peak 2.95,
  the cylinder never sheds, and the flow heals as the inflow decays ($t >
  6$) — with a steady measured CFL of 0.81, i.e. no warning. IMEX/EXT3 is
  kept from blowing up by the CFL controller: spurious velocity growth makes
  it shrink $\Delta t$, and the measured CFL creeps from 0.77 to 1.08
  against the 0.7 target — a usable warning sign. This matches the stability
  regions of the explicit term — stable radius $|\lambda\Delta t|$ along rays
  at an angle from the negative real axis (90° = undamped advection):

  | | 90° | 85° | 80° | 70° | 45° | 0° |
  |---|---|---|---|---|---|---|
  | BDF2/EXT2 | ~0 | 0.48 | 0.60 | 0.78 | 1.09 | 1.33 |
  | BDF2/EXT3 | 0.63 | 0.62 | 0.62 | 0.60 | 0.58 | 0.57 |

  EXT3 wins only within ~10° of the imaginary axis (nearly undamped
  advection, the high-Re regime where NekRS uses it); this flow's spectrum,
  damped by viscosity and grad-div, sits where EXT2's region is larger.
- **With EXT2 the two schemes reach the same limit.** Both are accurate at
  CFL 0.7 and both degrade at 0.9 — IMEX's lift by 5.9%, the rotational
  scheme's drag by 6.7%. The rotational scheme cannot take a larger step than
  IMEX and stay accurate.

### 7.4 Cost

- **At equal CFL the schemes do the same work.** Identical step counts (the
  controller sees the same flow) and the same outer and inner iterations
  per step. The rotational step costs 7–9% more wall time in the runs timed
  side by side (CFL 0.3: 0.076 vs 0.071 s; EXT2 at 0.7: 0.117 vs 0.107 s);
  the 26% at CFL 0.5 compares runs timed a night apart.
- **Each scheme's cheapest accurate run** (both EXT2, CFL 0.7): IMEX 821 s,
  rotational 898 s, with the same lift accuracy and 70× better drag for
  IMEX. On M1 the rotational scheme cannot reach IMEX's drag accuracy at any
  cost.
- **Larger steps save less than the step count.** From CFL 0.5 (EXT3) to
  0.7 (EXT2):

  | IMEX | CFL 0.5 | CFL 0.7 | change |
  |---|---|---|---|
  | steps | 10787 | 7706 | −29% |
  | outer iterations per step | 11.2 | 12.5 | +11% |
  | velocity CG iterations per step | 74 | 97 | +31% |
  | velocity CG iterations, whole run | 799k | 751k | −6% |

  The velocity block is $\sigma M + \nu K$ + grad-div with $\sigma =
  1.5/\Delta t$; a bigger step makes it less mass-dominated and Jacobi-PCG
  needs more iterations. Over a whole run the velocity-solve work is nearly
  independent of $\Delta t$ (877k / 799k / 751k iterations at CFL 0.3 / 0.5
  / 0.7). Fitting per-iteration costs to the runs timed under the same
  conditions puts the wall-time gain at ~15%; the measured 1% compares a
  quiet night with a busy morning. A velocity preconditioner whose iteration
  count does not grow as $\sigma$ shrinks would make larger CFL targets pay.

**Reproduce:** `bench/dfg3/make_sweep.sh "-conv -apc jacobi_pcg" "-rot -pbj
-schur cc" > sweep.q`, then `setsid nohup bench/dfg3/run_queue.sh sweep.q
OUT "<ballast args>" &`; tables with `bench/dfg3/analyze.py
OUT/summary.txt`, the HTML report with `bench/dfg3/report.py`.

