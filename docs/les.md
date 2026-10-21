# Large-eddy simulation: the subgrid model

How bessemer's LES option works, what it costs, how to use it and what the tests pin.
Decided 2026-10-16 (human: "alright why don't you go put in the LES model you
recommended", "don't forget the symmetric gradient!"); the background memo (which models
exist, why these two, when LES is worth it against DNS) is the project's LES memo.
Conventions and rules: [developing.md](developing.md) §5.8.

## 1. What it is

One explicit term on the right-hand side of the momentum equation, the divergence of a
subgrid stress with an eddy viscosity:

    du/dt + (u.grad)u - nu lap u - div(2 nu_t(grad u) S(u)) + grad p = f,
    S(u) = 1/2 (grad u + grad u^T).

Weak form: the term adds `(2 nu_t S(u), S(v))` to the left-hand side, with `nu_t` a
pointwise function of the resolved velocity gradient and the local filter width. Two
models, selected by `physics.subgrid.model`:

| Model | nu_t | Constant | Vanishes in |
|---|---|---|---|
| `sigma` (Nicoud, Toda, Cabrit, Bose & Lee, Phys. Fluids 23 (2011) 085106) | (C Delta)^2 sigma_3 (sigma_1 - sigma_2)(sigma_2 - sigma_3) / sigma_1^2, sigma_i the singular values of grad u | C = 1.35 | pure shear (y^3 at a wall), solid rotation, axisymmetric and isotropic expansion, any two-component flow |
| `wale` (Nicoud & Ducros, Flow Turbul. Combust. 62 (1999) 183) | (C Delta)^2 (S^d:S^d)^{3/2} / ((S:S)^{5/2} + (S^d:S^d)^{5/4}), S^d the traceless symmetric part of (grad u)^2 | C = 0.5 | pure shear (y^3), isotropic expansion; NOT solid rotation |
| `constant` | the coefficient itself | - | verification only (test S4) |

`sigma` is the recommended one: it is zero in every laminar case a model should leave
alone, including the near-wall y^3 behaviour without a damping function, and it costs the
same. `wale` is there for comparison (it is what most codes offer). Both are 3D models; a
2D flow is treated as the 3D flow with u_z = 0 and d/dz = 0, so the sigma model is
identically zero in 2D (use `wale` to exercise the path in 2D tests). The filter width is
`Delta_K = filter_width * h_K / k_u`: the node spacing of the element (h_K the
volume-equivalent element size, k_u the velocity order), `filter_width` 1 by default.

Deck:

```yaml
equation: navier_stokes
physics:
  nu: 2.5e-4
  subgrid:
    model: sigma       # none | constant | wale | sigma
    coefficient: -1    # < 0: the model's published value (1.35 sigma, 0.5 wale)
    filter_width: 1    # Delta_K = filter_width * h_K / order_u
```

Only with `navier_stokes` and the convective form (`physics.convective_form:
convective`): the term sits beside the explicit convection. The rotational form and
Stokes are refused at load time. Every `run_case` summary then carries a `subgrid`
section: the model, the coefficient and filter width in use, `nu_t_max`, `nu_t_mean`
(volume average), `nu_t_max_over_nu` and `dissipation` (`int 2 nu_t S:S`, the energy the
model removes per unit time), all on the final velocity. Compare `dissipation` with the
viscous dissipation in the diagnostics CSV to see how much of the dissipation the model
carries: Pope's 80 % rule of thumb for a resolved LES is that the resolved motions carry
80 % of the kinetic energy; a model dissipating more than the molecular viscosity on a
flow meant to be nearly resolved is a sign the mesh is too coarse.

## 2. Why explicit, why the stress form

**Explicit, EXT-extrapolated with the convection.** Under IMEX the term is evaluated on
the same history levels as `N(u)` and extrapolated with the same weights
(`Convection::Mult` returns `N(u) + (2 nu_t S(u), S(v))` plus the directional do-nothing
term); the trapezoidal starter and the error-controlled BDF3 candidate get it for free.
Under OIFS, where `N` lives in the substeps, the term is extrapolated in the BDF step
exactly as the directional do-nothing term is (`SubtractOifsExplicitTerms`,
`Convection::MultExplicitTerms`). So the implicit block, its preconditioners, the Schur
complement and every solver test are untouched, nothing is assembled, and the term costs
two kernel launches per history level.

Stability: an explicit diffusion term is limited by `nu_t dt / h_node^2`. For an eddy
viscosity `nu_t ~ (C Delta)^2 |grad u|` with `Delta = h_node` this is
`C^2 |grad u| dt ~ C^2 CFL`, about 0.1 at CFL 0.5 for either constant: the convective
CFL limit the scheme already obeys covers it. (A constant `nu_t`, the verification
model, has no such protection: keep `nu_t dt / h_node^2` below about 0.1.)

**The stress form with the symmetric gradient.** With a variable viscosity the two
classical forms differ: `div(nu_t grad u)` and `div(2 nu_t S(u))` agree only when
`nu_t` is constant and `div u = 0` (their difference is `grad u^T grad nu_t`). The
divergence of a symmetric stress is the physically meaningful one (momentum
conservation, the energy identity `(2 nu_t S(u), S(u)) >= 0`), so the term is built on
`S(u)` and tested against it (S2). The bulk viscous term keeps its Laplacian form because
its `nu` is constant.

## 3. Implementation

`src/operators/subgrid_model.hpp`: the pointwise models (`subgrid::Wale`,
`subgrid::Sigma`, `SingularValues3`), `MFEM_HOST_DEVICE`, on a padded 3 x 3 gradient.
`src/operators/subgrid_stress.{hpp,cpp}`: `SubgridStress`, the operator (an nvcc TU; on
the list in `src/CMakeLists.txt`). `Convection::EnableSubgrid/MultSubgrid/
MultExplicitTerms`; `TimeIntegratorOptions::subgrid`; `Parameters::subgrid` and the
deck keys; `Case::WriteSummary`'s `subgrid` section.

Two kernels per apply, at the dealiasing rule (order 3k, Convection's; `nu_t` is not a
polynomial in the gradient, and one rule for all nonlinear terms keeps one set of
geometric factors):

1. One thread per quadrature point: interpolate `grad u` from the element's dofs
   (directly, no sum factorization), pull it to physical space with `J^-1`, evaluate the
   model, form `tau = 2 nu_t S(u)` and write `D_mc = w |J| sum_j tau_cj (J^-1)_mj`, the
   stress weighted and pulled back to reference coordinates (dim^2 numbers per point).
2. One thread per element dof: `y_dc = sum_q sum_m D_mc(q) dphi_d/dxi_m(q)`.

Then the element restriction's transpose and `P^T` (hanging nodes like every other
operator). Both kernels are O(k^6) per element in 3D rather than the sum-factorized
O(k^4): a few percent of a step at k = 6 by count, not measured on a GPU yet; the
sum-factorized version is the follow-up if a profile asks for it.

**Memory.** No per-step state. The scratch of an apply (the element vectors in and
out, the stress at the points, one local-dof vector) is allocated and freed inside it:
about 150 B per velocity dof at Q3 and at Q6 while it lives, zero between applies. The
model keeps 8 B per element (the filter widths squared). The alternative, keeping the
scratch between steps to avoid a `cudaMalloc` and a synchronizing `cudaFree` per apply
(what the integrator does for its own work vectors, 2026-10-15), would cost those
150 B per dof permanently; it was not taken (memory first, human 2026-10-16), and is a
measurement away if the allocations show in a profile.

**Singular values.** Nicoud et al.'s appendix computes the singular values from the
invariants of `g^T g` by Cardano's formula. Near a wall the small singular values are
O(y) and O(y^2) against an O(1) shear, and a closed-form eigenvalue near a double root is
accurate to about sqrt(machine epsilon) in absolute terms: the formula returned
`sigma_3 = 0`, hence `nu_t = 0`, at y = 1e-3 where the true value is 2e-9 (checked
against LAPACK's SVD while writing this). `SingularValues3` uses one-sided Jacobi
rotations instead (high relative accuracy for every singular value, Demmel & Veselic
1992): right to 1e-12 relative down to y = 1e-6, in 2-3 sweeps. The y^3 behaviour is
the model's point near walls; S1 pins it at y = 1e-2, 1e-3, 1e-4.

## 4. Tests (`subgrid_test`, fast tier, np 1/2/4)

- S1 the pointwise models: the vanishing cases of each model, random gradients against
  an independent evaluation (MFEM's eigenvalue routine, DenseMatrix algebra), the y^3
  wall behaviour of both.
- S2 with a constant `nu_t` the operator equals MFEM's `ElasticityIntegrator(lambda = 0,
  mu = nu_t)` fully assembled on the same rule, to 1e-12 relative, on a curved
  partitioned mesh, 2D and 3D, p = 2 and 3.
- S3 the filter width is `h_K / k_u`; the dissipation statistic equals `<Mult(u), u>`
  and is positive (the term does nonnegative work); the statistics are the same on every
  rank.
- S4 a constant `nu_t` carried explicitly converges to the implicit run at `nu + nu_t`
  at second order in dt on the 2D Taylor-Green vortex, under IMEX and under OIFS. The
  implicit reference carries grad-div with `gamma = nu_t`: on a periodic H1 space
  `(2 S(u), S(v)) = (grad u, grad v) + (div u, div v)` exactly, and the discrete
  Taylor-Hood velocity is not exactly divergence-free, so a constant `nu_t` in the stress
  form is `nu_t` in the Laplacian form plus a grad-div term `gamma = nu_t` (without it
  the two runs differed by 7.5e-5 at every dt).
- S5 the sigma model through `Case` and `Parameters` on the 3D Taylor-Green vortex:
  steps, finite statistics, `nu_t` small against `nu` on the two-component initial
  field.

Not covered yet: a turbulent case (the TGV at Re 1600 on a coarse mesh against the DNS
dissipation curve is the natural first one, `benchmarks/tgv_re1600`), the models on a
GPU (the kernels follow the other nvcc TUs; CI builds the CPU only), and the Python
bindings (`physics.subgrid` is not exposed there).

## 5. References

- Nicoud, F., Ducros, F.: Subgrid-scale stress modelling based on the square of the
  velocity gradient tensor. Flow, Turbulence and Combustion 62 (1999) 183-200.
- Nicoud, F., Toda, H. B., Cabrit, O., Bose, S., Lee, J.: Using singular values to
  build a subgrid-scale model for large eddy simulations. Physics of Fluids 23 (2011)
  085106.
- Demmel, J., Veselic, K.: Jacobi's method is more accurate than QR. SIAM J. Matrix
  Anal. Appl. 13 (1992) 1204-1245.
- Pope, S. B.: Ten questions concerning the large-eddy simulation of turbulent flows.
  New J. Phys. 6 (2004) 35.
