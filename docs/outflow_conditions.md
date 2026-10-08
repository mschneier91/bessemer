# Outflow boundary conditions

Boundaries marked **outflow** (`BoundaryConditions::AddOutflow`, deck BC type `outflow`)
carry a do-nothing condition. With the IMEX convective form, bessemer imposes the
**directional do-nothing** condition there by default.

## Classical do-nothing

$$T(u,p)\,n = 0 \quad\text{on } S_1, \qquad T(u,p) = \nu\nabla u - pI .$$

$T$ uses the full velocity gradient, matching bessemer's vector-Laplacian viscous term. The
condition is natural: integrating the viscous and pressure terms by parts produces
$\int_{S_1} T(u,p)\,n\cdot\varphi$, which the condition sets to zero, so nothing is
assembled.

## Directional do-nothing (the default with the IMEX convective form)

M. Braack, P. B. Mucha, *Directional do-nothing condition for the Navier–Stokes
equations*, J. Comput. Math. **32**(5) (2014) 507–521, doi:10.4208/jcm.1405-m4347.

$$T(u,p)\,n - \tfrac12 (u\cdot n)_-\, u = 0 \quad\text{on } S_1,
\qquad (u\cdot n)_- = \min(u\cdot n, 0),$$

so the weak form gains $-\tfrac12\int_{S_1}(u\cdot n)_-\,u\cdot\varphi\,ds$ on its left-hand
side (the paper's eq. 2.9).

**Why.** Test the momentum equation with $\varphi = u$. The viscous and pressure boundary
terms vanish by the do-nothing condition, but the convective term leaves one behind (with
$\operatorname{div} u = 0$ and $u = 0$ on the walls):

$$((u\cdot\nabla)u,\,u) = \tfrac12\int_{S_1}(u\cdot n)\,|u|^2\,ds .$$

Where flow leaves ($u\cdot n > 0$) this carries kinetic energy out, which is harmless. Where
flow re-enters ($u\cdot n < 0$, backflow) it pumps energy in. That contribution is cubic in
$u$ while viscous dissipation is quadratic, so nothing bounds it: this is the classical
condition's backflow instability, and why its existence theory needs small inflow. The
added term contributes $-\tfrac12\int_{S_1}(u\cdot n)_-|u|^2$, which cancels exactly the
negative part. The energy balance becomes

$$\tfrac12\frac{d}{dt}\|u\|^2 + \nu\|\nabla u\|^2 + \tfrac12\int_{S_1}(u\cdot n)_+|u|^2
= (f, u),$$

with every term on the left non-negative: energy can leave through the outflow but never
enter.

**What it changes.** Where $u\cdot n \ge 0$ the term vanishes and the condition *is* the
classical one, so flows that leave through the whole boundary (the DFG cylinder benchmarks,
the paper's vortex street) are unchanged. Where flow enters, the traction balances half the
incoming momentum flux, which damps the backflow. In the paper's driven-square test
(§5.1), the inflow through the do-nothing side drops by ~5% at $\nu = 0.05$ and ~16% at
$\nu = 0.005$, and their steady solver stops converging with the classical condition at
$\nu \le 5\cdot10^{-4}$ but not with this one.

## Using it

| | |
|---|---|
| deck | `physics.outflow: directional` (default) or `classical` |
| C++ | `Parameters::outflow`, `TimeIntegratorOptions::outflow` (enum `OutflowCondition`) |
| Python | `p.outflow = incns.OutflowCondition.Directional` / `.Classical` |
| DFG app | `apps/dfg_cylinder -ddn` (default) / `-cdn` |

It applies to Navier–Stokes with `physics.convective_form: convective` on every outflow
boundary. Stokes has no convection, so there is nothing to apply. **The rotational form
keeps the classical condition**, which acts on its Bernoulli head. That is a poor outflow
condition for that form, and fixing it is an open item.

## Implementation

- `operators/directional_do_nothing` assembles the term as a boundary-face linear form for
  a given velocity, over the outflow faces only. It uses host (legacy) assembly, takes the
  outward normal from the face transformation (independent of how the mesh orients its
  boundary elements), and uses the same $3k$ quadrature as the dealiased interior
  convection.
- `Convection::EnableDirectionalDoNothing` adds it to $N(u)$. The time integrator enables
  it when the option is `Directional` and the boundary conditions have outflow attributes.
- **Time discretization: explicit.** The term is the boundary part of the convective flux,
  so it is evaluated on the velocity history and EXT-extrapolated together with $N(u)$ on
  the right-hand side. The implicit operator and every preconditioner are unchanged, and
  it adds no time-step restriction beyond the convective CFL limit. A semi-implicit
  version, $-\tfrac12\int_{S_1}(u^*\cdot n)_-\,u^{n+1}\cdot\varphi$ with $u^*$
  extrapolated, would be a symmetric positive semidefinite boundary term in the momentum
  block. It would make the backflow cancellation hold independent of $\Delta t$, but the
  operator would change every step and the Jacobi diagonal would need it. It is the
  fallback if outlet backflow ever destabilizes a run below the CFL limit.
- Lift/drag (John's volume integral) reuses the step's right-hand side, so it includes
  the term automatically.

## Tests (`test/directional_do_nothing_test.cpp`)

- **D1** The term's value and the outward normal match hand-computed integrals face by face
  for a polynomial field on a box.
- **D2** With the term on, $\langle N(u) + D(u), u\rangle = \tfrac12\int(u\cdot n)_+|u|^2
  - \tfrac12\int(\operatorname{div}u)|u|^2$ to round-off, in 2D and 3D: the backflow part of
  the boundary flux is gone. The volume part remains because Taylor–Hood velocities are only
  weakly divergence-free.
- **D3** The paper's §5.1 test (unit square, do-nothing at $x = 0$, no-slip elsewhere,
  $f = (\sin x + \sin y, 0)$, $\nu = 0.05$), marched to steady state, reproduces its
  Table 5.1:

  | | $j_1 = \int_{S_1}(u\cdot n)_-$ | paper | $j_2 = \int_{S_1}(u\cdot n)_+\|u\|^2$ | paper |
  |---|---|---|---|---|
  | classical | −4.4984e-2 | −4.498e-2 | 6.113e-4 | 6.109e-4 |
  | directional | −4.2703e-2 | −4.269e-2 | 5.321e-4 | 5.318e-4 |
