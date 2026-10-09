# OIFS: sub-stepped convection

Operator-integration-factor splitting (Y. Maday, A. T. Patera, E. M. Rønquist, *An
operator-integration-factor splitting method for time-dependent problems: application to
incompressible fluid flow*, J. Sci. Comput. **5** (1990) 263–292), as used in Nek5000.
It removes the convective CFL limit from the BDF step: convection is integrated by
explicit substeps inside each step, so the step itself can run at CFL 2–4.

| | |
|---|---|
| deck | `time.convection: oifs` (default `imex`), `time.oifs_cfl` (substep CFL, default 0.5) |
| C++ | `Parameters::convection_treatment`, `Parameters::oifs_cfl`; `TimeIntegratorOptions` likewise |
| code | `src/time/oifs.{hpp,cpp}` (`OifsAdvector`), `StokesTimeIntegrator` (`AssembleBdfRhs`, `OifsBoundaryValues`, the starter) |
| tests | `oifs_test` O1–O3, `nse_mms_test` Oifs*, `directional_do_nothing_test` D4, `deck_test` ConvectionKeys |

Scope: Navier–Stokes, convective form, fixed or CFL-controlled steps. Error-controlled
steps are rejected by design (human decision, 2026-10-09: an OIFS step is chosen as a
multiple of the CFL step anyway, so an estimator's second solve buys nothing).

## How it works

**The scheme.** BDF-$k$ approximates the time derivative by
$\sum_{j=0}^{k} c_j u^{n+1-j}$. OIFS replaces each history level $u^{n+1-j}$ by
$\tilde u_j(t^{n+1})$, that level carried forward by pure advection,

$$\partial_s \tilde u_j + (w(s)\cdot\nabla)\tilde u_j = 0 \ \text{ on } [t^{n+1-j}, t^{n+1}],
\qquad \tilde u_j(t^{n+1-j}) = u^{n+1-j}.$$

The step then solves the Stokes system

$$(c_0 M + \nu K)\,u^{n+1} - B^T p = F^{n+1} - M \sum_{j\ge1} c_j\,\tilde u_j(t^{n+1}),$$

with no explicit $N(u)$ on the right-hand side, so the step is not CFL-limited.

**One combined field.** Advection is linear in $\tilde u$, so the weighted sum is advected
as one field (Nek's way): start from $c_k u^{n+1-k}$ at the oldest time, advect to the next
history time, add the next level, and continue to $t^{n+1}$. The advection covers
$k\,\Delta t$ once.

**Wind.** $w(s)$ is the Lagrange polynomial through the last $m$ velocities, $m$ the
extrapolation order (3 under BDF3; `LagrangeWeights` in `time/multistep_coeffs`). It
interpolates over the past and extrapolates over the last interval.

**Substeps.** Classical RK4. The number of substeps per history interval is chosen so that
each substep's CFL number (Nek's definition, `time/cfl`) is at most `time.oifs_cfl`, using
the largest CFL rate over the wind's history velocities.

**Space.**
- MFEM's `ConvectionIntegrator` per velocity component on a scalar companion space, with
  partial assembly and the dealiasing rule of the IMEX convection (order $3k$).
- One operator per wind level. $C(w)$ is linear in $w$, so each RK stage combines the
  operators with that stage's interpolation weights; nothing is reassembled per stage.
- The substeps use the row-sum-lumped GLL mass $M_L$ (diagonal), so applying its inverse
  is a pointwise scaling.

**Defaults under OIFS** (all automatic at the case level):
- BDF3 (`time.order` auto = 0 → `Parameters::BdfOrder()`; human decision, 2026-10-09),
  which extrapolates the wind at order 3;
- the collocated GLL mass in the Stokes step (`Parameters::CollocatedMass()`), see
  [the mass mismatch](#the-mass-mismatch) below.

## Edge cases

Several of these are not about OIFS itself. Two (§1) come from doing OIFS with a
non-diagonal finite-element mass, which Nek5000's spectral-element setting, with its
diagonal mass everywhere, never has.

### 1. A non-diagonal finite-element mass

#### Boundary values in the BDF right-hand side

With the consistent mass $M$, the boundary nodes' values of
$\varphi = \sum_{j\ge1} c_j\tilde u_j$ enter the **interior** rows through $M$'s
off-diagonal entries, so they have to be right. At an inflow node the true characteristic
comes from outside the domain, so

$$\tilde u_j(x) = u(x - \tau_j w,\ t^{n+1-j}) \approx u_D(t^{n+1-j}) - \tau_j\,(u\cdot\nabla)u,
\qquad \tau_j = t^{n+1} - t^{n+1-j},$$

not the frozen boundary data $u_D(t^{n+1-j})$. `OifsBoundaryValues` therefore sets the
imposed nodes to

$$\sum_{j\ge1} c_j\,u_D(t^{n+1-j}) - \Big(\sum_{j\ge1} c_j\tau_j\Big)(u\cdot\nabla)u\big|^{n+1},$$

with the convective acceleration recovered as $M_L^{-1}N(u)$ and extrapolated. BDF2 and
BDF3 satisfy $\sum c_j\tau_j = -1$ and $\sum c_j\tau_j^2 = 0$ for any step ratio, and BDF3
also $\sum c_j\tau_j^3 = 0$, so the higher-order terms of the expansion cancel to the
scheme's order.

Without it, the MMS error stalled at 4.6e-4 ($h = 1/3$) whatever $\Delta t$ was, and
converged only like $h^2$ (`nse_mms_test` OifsTemporalOrder2D measures it).

With the collocated mass now used under OIFS, $M$ is diagonal on conforming meshes and this
coupling mostly disappears. It still matters in two places:
- on hanging-node (AMR) meshes, where the collocated mass $P^T D P$ is not diagonal;
- for lift and drag, which come from the momentum residual at the wall rows. There the
  boundary values make OIFS's residual equal IMEX's $N(u)$ (`nse_mms_test`
  OifsWallBounded2D: wall force equal to IMEX's to 1.5e-3).

Evaluating $(u\cdot\nabla)u$ pointwise at the boundary nodes instead (exactly zero on
no-slip walls) was tried and changed the error by less than 1%; it was reverted.

#### The mass mismatch

This was the costly one. The substeps advect with $M_L^{-1}C$, and the Stokes step then
multiplies the result by $M$. As $\Delta t \to 0$ the scheme therefore converges to

$$M\,\frac{du}{dt} + M M_L^{-1} N(u) + \dots \qquad\text{instead of IMEX's}\qquad
M\,\frac{du}{dt} + N(u) + \dots$$

- For resolved smooth fields $M M_L^{-1} \approx I$: on a wall-bounded MMS the gap to IMEX
  was only 1.6e-3 (4×4 mesh) and 1.6e-4 (8×8), shrinking with $h$.
- At the square cylinder's sharp corners it is not close to $I$, and it is not
  energy-neutral. The result was over-energetic shedding: on the Re 200 square cylinder
  (AMR, Q3/Q2) C_D was +3% and C_L,rms +12% above IMEX **at every step size**, from CFL 4
  down to CFL 0.5.

The fix is the same diagonal mass in both places, as in Nek by construction. Case-level
OIFS always uses the collocated GLL mass. With it, OIFS converges to IMEX's discretization:
the MMS gap drops to 5e-7 (`nse_mms_test` OifsMatchesImexWithCollocatedMass2D pins both
sides), and on the cylinder OIFS is within 1% of IMEX (with the same mass) at CFL 2.

### 2. Intrinsic to OIFS, or to any characteristic splitting

#### The characteristic boundary rule

Pure advection takes data only where characteristics enter. In the substeps, Dirichlet
data (times the combined field's accumulated weight) is imposed only where $w\cdot n \le 0$,
which includes walls since $w = 0$ there; outflow nodes are left free. Imposing data at
outflow nodes over-constrains the advection and cost an $O(h^2)$ boundary layer.
`OifsAdvector::ClassifyBoundary` decides the sign per boundary node from the newest wind;
the sign can change within the advection span, which is a minor effect.

#### The time error follows particle paths

This is the real price of OIFS. Its time error is the BDF error **along trajectories**
($\Delta t^k\, D^{k+1}u/Dt^{k+1}$, a material derivative), while IMEX's is the change at a
fixed point plus the extrapolation error of $N$. Where the flow pattern is nearly fixed but
fluid crosses it quickly (attached shear layers, corners, a steady vortex), OIFS's error is
much larger than IMEX's at the same $\Delta t$:

- Wall-bounded MMS at $\Delta t = 0.005$: BDF2-OIFS error 8.6e-3, IMEX 5.45e-3 (all
  spatial; both with the consistent mass).
- BDF3 cuts the trajectory error 3.5–7× (orders 2.42 and 2.80 from an exact start,
  `nse_mms_test` OifsBdf3TemporalOrder2D) — the reason BDF3 is the OIFS default.
- Taylor–Green vortex (`oifs_test` O2): BDF3-OIFS errors 0.074 / 0.072 at CFL 2 / 4 in 11 / 9
  steps, against IMEX's 0.097 at CFL 0.5 in 38 steps. BDF2-OIFS with the consistent mass
  gave 0.164 / 0.204.
- Square cylinder, Re 200, BDF3 and the collocated mass, against IMEX at CFL 0.5 with the
  same mass (C_D 1.4381, C_L,rms 0.3957, St 0.1564, 9,103 steps):

  | OIFS | C_D | C_L,rms | St | steps |
  |---|---|---|---|---|
  | CFL 1 | −0.2% | −0.7% | +0.1% | 4,379 |
  | CFL 2 | −0.8% | −0.9% | +0.1% | 2,191 |
  | CFL 4 | +2.7% | +21% | −3.6% | 1,120 |

  CFL 2 is accurate to 1% in 4.2× fewer steps (an OIFS step costs about 1.3–1.5× an IMEX
  step); CFL 4 loses the lift amplitude.

#### The directional do-nothing outflow term

Braack & Mucha's term $-\tfrac12\int_{S_1}(u\cdot n)_-\,u\cdot\varphi$
([outflow_conditions.md](outflow_conditions.md)) is stiff at backflow nodes: its rate there
is about $6|u\cdot n|/h$ for Q3.

- **Inside the substeps** (linearized in the advected field, $-\tfrac12\int_{S_1}(w\cdot
  n)_-\,\psi_j\psi_i$, which makes the substep advection exactly energy-stable): the
  exponentiated history no longer reduces to the plain term under the BDF combination once
  the step is large. On the paper's §5.1 square, $j_1$ was 2.3% off at CFL 2 and converged
  only as the CFL fell to 0.25. Removed.
- **Explicit and extrapolated in the BDF step**, exactly IMEX's treatment
  (`SubtractOifsDirectionalDoNothing`, using `Convection::MultDirectionalDoNothing`): it
  reproduces IMEX's steady state at every CFL ($j_1$, $j_2$ equal to IMEX's to 5e-5 at
  CFL 2; `directional_do_nothing_test` D4) and was stable to CFL 8 on that problem. It is
  only conditionally stable: backflow much faster than the local flow could need the
  semi-implicit version (the term with an extrapolated velocity, in the momentum block),
  which is the documented fallback.

### 3. Multistep startup

The first step is the trapezoidal starter with $u^0$ advected. Under OIFS its local error
is $O(\Delta t^2)$, not $O(\Delta t^3)$:

- the wind is frozen at $u^0$ (first-order extrapolation);
- the old-time half of the viscous term, $\tfrac{\nu}{2}Ku^0$, is evaluated at the fixed
  point instead of being advected.

That caps a **fixed-step** BDF3-OIFS march at global order 2: from the starter, the
wall-bounded MMS shows orders 2.54 / 2.19 / 1.93 instead of approaching 3. It is invisible
with BDF2, which is second order anyway, and negligible in CFL-controlled runs, which start
at `time.dt` and grow at most ×1.2 per step. The order tests seed the exact history through
`ImportState` (`MmsConfig::exact_history` in `nse_mms_test`) instead.
