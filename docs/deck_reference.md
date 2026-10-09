# Deck reference

Every key a YAML deck (`apps/run_case <deck.yaml>`) understands, with its default. Omitted keys keep their defaults; an unknown key is an error that names the closest valid key.

**Generated** by `build/cpu/apps/run_case --deck-reference` from the loader itself (`src/config/parameters.cpp`); `deck_test` fails when this file is out of date. Do not edit by hand.

## Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `equation` | stokes \| navier_stokes | `stokes` | Equation set: unsteady Stokes or incompressible Navier-Stokes. |
| `device` | string | `cpu` | MFEM device backend (cpu, cuda, debug); the env var INCNS_DEVICE wins. |
| `initial_velocity` | string | `zero` | Named initial condition: zero or taylor_green_2d (others come from Python or C++). |
| `restart` | string | `""` | Checkpoint directory to restart from (same number of MPI ranks). |
| `boundary_conditions` | list | `[]` | Boundary groups, each {select: [faces or attributes], type: no_slip \| outflow \| velocity_dirichlet, group: name}; faces are xmin, xmax, ymin, ymax, zmin, zmax or all. |

## `physics`

Physical parameters.

| Key | Type | Default | Description |
|---|---|---|---|
| `physics.nu` | real | `1` | Kinematic viscosity (dimensionless: 1/Re). |
| `physics.Re` | real | `unset` | Reynolds number; sets nu = 1/Re (dimensionless mode; not with nu). |
| `physics.grad_div` | real | `0` | Grad-div stabilization scale c_gd (0 = off). |
| `physics.grad_div_scale` | h \| nu | `h` | Grad-div coefficient: c_gd * h_K per element, or c_gd * nu. |
| `physics.convective_form` | convective \| rotational | `convective` | Nonlinear term: explicit (u.grad)u, or the semi-implicit rotational form (the pressure becomes the Bernoulli head). |
| `physics.outflow` | directional \| classical | `directional` | Condition on outflow boundaries: Braack-Mucha directional do-nothing (stable under backflow) or classical do-nothing. |

## `nondimensionalization`

Input scaling: dimensionless (default) or dimensional with reference scales.

| Key | Type | Default | Description |
|---|---|---|---|
| `nondimensionalization.mode` | dimensionless \| dimensional | `dimensionless` | dimensional: lengths, times and nu carry units and are rescaled by L_ref, U_ref. |
| `nondimensionalization.L_ref` | real | `1` | Reference length (dimensional mode). |
| `nondimensionalization.U_ref` | real | `1` | Reference velocity (dimensional mode). |
| `nondimensionalization.rho` | real | `1` | Density, for dimensional forces. |

## `discretization`

Finite element orders and mass.

| Key | Type | Default | Description |
|---|---|---|---|
| `discretization.order_u` | int | `3` | Velocity polynomial order k_u. |
| `discretization.order_p` | int | `2` | Pressure polynomial order k_p (k_u - 1). |
| `discretization.collocated_mass` | bool | `false` | Use the diagonal GLL collocated velocity mass (always on under OIFS). |

## `mesh`

Box mesh (quads/hexes) for apps/run_case.

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.dim` | int | `2` | Spatial dimension, 2 or 3. |
| `mesh.elements` | int[dim] | `[4, 4]` | Elements per direction. |
| `mesh.lengths` | real[dim] | `[6.28319, 6.28319]` | Box edge lengths (the box starts at the origin). |
| `mesh.periodic` | bool[dim] | `[true, true]` | Periodic directions; non-periodic faces need boundary_conditions. |
| `mesh.stretch` | none \| tanh | `[none, none]` | Per-direction node clustering: uniform, or two-sided tanh toward both ends. |
| `mesh.stretch_beta` | real[dim] | `[2, 2]` | tanh clustering strength per direction. |

## `time`

Time integration.

| Key | Type | Default | Description |
|---|---|---|---|
| `time.dt` | real | `0.01` | Step size (fixed steps), or the first step (cfl / error control). |
| `time.t_final` | real | `1` | End time. |
| `time.step_control` | cfl \| fixed \| error | `cfl` | How dt is chosen: dt = cfl_target / CFL rate before every step (Navier-Stokes), constant, or BDF2/BDF3 error control (IMEX only). |
| `time.adaptive` | bool | `unset` | Older spelling: true = step_control error, false = fixed. |
| `time.cfl_target` | real | `0.5` | Target CFL number (Nek5000's definition) under step_control cfl: ~0.5 for IMEX, 2 for OIFS. |
| `time.cfl_max` | real | `0` | CFL ceiling (0 = off): caps error-controlled dt, aborts fixed steps above it. |
| `time.dt_max` | real | `0` | Largest step under cfl control (0 = no cap). |
| `time.convection` | imex \| oifs | `imex` | Convection: explicit and extrapolated (IMEX, CFL < ~0.7), or OIFS sub-stepping (CFL 2 practical; BDF3 and the collocated mass automatically). |
| `time.oifs_cfl` | real | `0.5` | CFL number of each OIFS substep. |
| `time.order` | int | `0` | BDF order 2 or 3; 0 = auto (3 under OIFS, 2 under IMEX). |
| `time.ext_order` | int | `2` | Extrapolation order of the explicit/lagged nonlinear term, 2 or 3. |
| `time.atol` | real | `1e-08` | Error control: absolute tolerance on the velocity error estimate. |
| `time.rtol` | real | `1e-06` | Error control: relative tolerance on the velocity error estimate. |

## `solver`

Linear solver: FGMRES on the monolithic velocity-pressure system with a block preconditioner.

| Key | Type | Default | Description |
|---|---|---|---|
| `solver.rtol` | real | `1e-10` | FGMRES relative tolerance. |
| `solver.atol` | real | `0` | FGMRES absolute tolerance. |
| `solver.max_iter` | int | `2000` | FGMRES iteration cap. |
| `solver.kdim` | int | `200` | FGMRES restart size. |
| `solver.print_level` | int | `-1` | Solver verbosity (-1 = quiet). |
| `solver.schur` | cc \| mass \| laplacian_legacy | `cc` | Pressure Schur block: Cahouet-Chabard, scaled pressure mass, or the legacy Laplacian CC (comparison only). |
| `solver.a_pc` | jacobi_pcg \| jacobi_chebyshev \| loramg | `jacobi_pcg` | Velocity-block preconditioner (Cahouet-Chabard path): Jacobi-PCG, Jacobi-Chebyshev, or one LOR-AMG V-cycle (best when viscous-dominated). |
| `solver.a_pcg_rtol` | real | `0.01` | jacobi_pcg: inner CG tolerance. |
| `solver.a_pcg_max_iter` | int | `50` | jacobi_pcg: inner CG iteration cap. |
| `solver.preconditioner` | jacobi \| loramg | `jacobi` | Velocity-block preconditioner on the mass Schur path. |
| `solver.block_shape` | upper \| lower \| diag | `upper` | Block preconditioner shape. |
| `solver.n_inner` | int | `10` | Cahouet-Chabard: fixed inner CG iterations on the pressure Poisson block. |
| `solver.lp_vcycles` | int | `1` | Cahouet-Chabard: LOR-AMG V-cycles preconditioning the inner CG. |
| `solver.pc_quadrature` | inherit \| gll_collocated | `inherit` | Quadrature of the preconditioner's operators. |
| `solver.amg_reuse` | bool | `false` | Freeze the LOR-AMG hierarchy across dt changes. |
| `solver.rotation_pc` | symmetric \| pbj_only \| pbj_krylov | `symmetric` | Rotational form: velocity preconditioner (pbj_krylov recommended). |
| `solver.rotation_lor` | bool | `false` | Rotational form: put the rotation term into LOR-AMG (measured not to pay off). |
| `solver.rotation_log_interval` | int | `0` | Rotational form: log rotation-number statistics every N steps (0 = off). |
| `solver.rotation_schur` | cc \| tensor \| auto | `cc` | Rotational form: Schur preconditioner (a mode, or a map with the keys below). |
| `solver.rotation_schur.mode` | cc \| tensor \| auto | `cc` | Schur mode. |
| `solver.rotation_schur.criterion` | max_mu \| volume_fraction | `max_mu` | auto: switch on the maximum rotation number or the volume fraction above it. |
| `solver.rotation_schur.mu_on` | real | `20` | auto: switch to tensor above this rotation number. |
| `solver.rotation_schur.mu_off` | real | `10` | auto: switch back below this rotation number. |
| `solver.rotation_schur.vol_on` | real | `0.001` | auto (volume_fraction): switch-on fraction. |
| `solver.rotation_schur.vol_off` | real | `0.00025` | auto (volume_fraction): switch-off fraction. |
| `solver.rotation_schur.inner_iterations` | int | `10` | tensor: fixed inner FGMRES iterations. |

## `output`

ParaView output and diagnostics.

| Key | Type | Default | Description |
|---|---|---|---|
| `output.enabled` | bool | `false` | Write ParaView output. |
| `output.path` | string | `.` | Output directory prefix. |
| `output.name` | string | `case` | Collection name. |
| `output.interval` | int | `1` | Write every N accepted steps. |
| `output.diagnostics` | bool | `false` | Also log kinetic energy, dissipation and \|\|div u\|\| to a CSV. |

## `amr`

Adaptive mesh refinement (refinement only).

| Key | Type | Default | Description |
|---|---|---|---|
| `amr.enabled` | bool | `false` | Turn AMR on (the mesh becomes nonconforming). |
| `amr.interval` | int | `50` | Adapt every N accepted steps. |
| `amr.initial_passes` | int | `0` | Refinement passes on the initial condition. |
| `amr.passes_per_event` | int | `1` | Refinement passes per adaptation event. |
| `amr.anisotropic` | bool | `false` | Split only the directions with large gradients. |
| `amr.aniso_ratio` | real | `0.5` | Anisotropic: split direction d if its gradient >= ratio * the largest. |
| `amr.threshold_mode` | relative \| absolute | `relative` | Mark where the indicator >= theta * its max (relative) or >= tolerance (absolute). |
| `amr.theta` | real | `0.5` | Relative marking fraction. |
| `amr.tolerance` | real | `0.05` | Absolute marking threshold (velocity units). |
| `amr.min_size` | real | `0` | Do not split below this element extent. |
| `amr.max_elements` | int | `0` | Element cap (0 = none). |
| `amr.nc_limit` | int | `1` | Maximum hanging-node level difference. |
| `amr.rebalance` | bool | `true` | Rebalance the partition after refining. |
| `amr.project_history` | bool | `true` | Divergence-free projection of the time history after an event. |
| `amr.write_indicator` | bool | `false` | Write the refinement indicator with the output. |

## `forces`

Lift and drag on a body (John's volume integral of the momentum residual).

| Key | Type | Default | Description |
|---|---|---|---|
| `forces.enabled` | bool | `false` | Compute forces. |
| `forces.attributes` | int list | `[]` | Boundary attributes of the body. |
| `forces.reference_velocity` | real | `1` | U in C = 2F / (rho U^2 A). |
| `forces.reference_area` | real | `1` | A in C = 2F / (rho U^2 A) (a length in 2D). |
| `forces.interval` | int | `1` | Log to <path>/<name>_forces.csv every N steps (0 = no log). |

## `checkpoint`

Rolling checkpoints.

| Key | Type | Default | Description |
|---|---|---|---|
| `checkpoint.enabled` | bool | `false` | Write rolling checkpoints. |
| `checkpoint.path` | string | `chk` | Checkpoint directory (overwritten each time). |
| `checkpoint.interval` | int | `10` | Write every N accepted steps. |
