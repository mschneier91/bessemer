# IMEX vs. semi-implicit rotational time stepping in bessemer

This document explains the two Navier–Stokes time-stepping schemes bessemer
offers — the **IMEX convective scheme** and the **semi-implicit rotational
scheme** — exactly as they are implemented, and reports a head-to-head
comparison on the DFG 2D-3 benchmark (flow around a cylinder with
time-dependent inflow, V. John's reference case) with adaptive mesh
refinement. The question the comparison answers: **which scheme is better,
and can the semi-implicit scheme take time steps beyond the convective CFL
limit and still give a better answer?**

RESULTS_PLACEHOLDER

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
  $\Delta t \lesssim C\,h/(k^2|u|)$. bessemer measures this with a
  directional, order-aware CFL number
  $$\mathrm{CFL} = \Delta t\; k^2 \max_{\text{quad. points}} \sum_d \big|(J^{-1}u)_d\big|$$
  ($J$ the element Jacobian, so $J^{-1}u$ is the velocity in reference
  coordinates per unit element; $k^2$ accounts for the clustering of the
  high-order nodes). Beyond the limit the scheme **blows up** — the solver
  still converges each step, the solution just grows without bound.
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

- **No convective CFL limit.** The rotation term is skew:
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

Two ways to make the rotational form genuinely CFL-free (§7): converge the
nonlinearity within each step (Picard sweeps with $w^* \leftarrow u^{n+1}$;
bessemer has an experimental `solver.rotation_picard: K`), or use the Oseen
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

**CFL number** reported below: the largest value of the directional measure
of §2 seen during the run (sampled every 5 steps).

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
  0.0017$ (CFL ≈ 2–2.7 in bessemer's measure, reached as the inflow nears its
  peak). At $\Delta t = 0.002$ the rotational solution leaves the converged
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

With `solver.rotation_picard: K`, each step is re-solved $K$ times with
$w^* \leftarrow u^{n+1}$ (§5):

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

AMR_PLACEHOLDER
