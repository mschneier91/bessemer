# Deck reference

Every key a YAML deck (`apps/run_case <deck.yaml>`) understands, with its default. Omitted keys keep their defaults; an unknown key is an error that names the closest valid key.

**Generated** by `build/cpu/apps/run_case --deck-reference` from the loader itself (`src/config/parameters.cpp`); `deck_test` fails when this file is out of date. Do not edit by hand.

## Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `equation` | stokes \| navier_stokes | `stokes` | Equation set: unsteady Stokes or incompressible Navier-Stokes. |
| `device` | string | `cpu` | MFEM device backend (cpu, cuda, debug); the env var INCNS_DEVICE wins. |
| `initial_velocity` | string | `zero` | Named initial condition: zero, uniform (see the initial section) or taylor_green_2d; anything else comes from Python or C++. |
| `restart` | string | `""` | Checkpoint directory to restart from (same number of MPI ranks). |
| `boundary_conditions` | list | `[]` | Boundary groups, each {select: [names or attributes], type: no_slip \| outflow \| velocity \| velocity_dirichlet, ...}. Names: box faces xmin, xmax, ymin, ymax, zmin, zmax; the geometry's boundary names; or all. type velocity takes value: [u, v] (constant), or profile: parabolic with u_max and height (u_x = 4 u_max y (height - y) / height^2), optionally time_profile: ramp \| sine with time_scale T (ramp: sin^2(pi t / 2T) for t < T; sine: sin(pi t / T)). velocity_dirichlet takes its field from Python or C++ by the group name. Every real boundary needs exactly one group when run by run_case. |

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
| `mesh.geometry` | box \| square_cylinder \| cylinder_channel | `box` | Domain: the box above, a square cylinder in a large domain (Joly et al. 2012; boundaries inflow, outflow, sides, body), or the DFG channel with a cylinder (boundaries inflow, outflow, walls, cylinder). The cylinder geometries are 2D. |
| `mesh.square_cylinder.side` | real | `1` | Square side D. |
| `mesh.square_cylinder.upstream` | real | `60` | Inlet distance from the square's centre. |
| `mesh.square_cylinder.downstream` | real | `120` | Outlet distance from the centre. |
| `mesh.square_cylinder.half_height` | real | `60` | Side boundaries at y = +-half_height. |
| `mesh.square_cylinder.n_face` | int | `6` | Cells along each face of the square (even). |
| `mesh.square_cylinder.corner_ratio` | real | `1.5` | Cell growth from the corners to mid-face. |
| `mesh.square_cylinder.far_ratio` | real | `1.3` | Cell growth upstream, sideways and far downstream. |
| `mesh.square_cylinder.wake_ratio` | real | `1.15` | Cell growth from the square into the near wake. |
| `mesh.square_cylinder.wake_h` | real | `0.5` | Cell width in the near wake. |
| `mesh.square_cylinder.wake_end` | real | `10` | x where the near wake's uniform cells end. |
| `mesh.cylinder_channel.n_side` | int | `4` | Cells along each side of the O-grid square. |
| `mesh.cylinder_channel.n_ring` | int | `3` | Radial cell layers around the cylinder. |
| `mesh.cylinder_channel.n_down` | int | `16` | Cells downstream of the O-grid. |
| `mesh.cylinder_channel.level` | int | `0` | Nested uniform refinement level (cell counts x 2^level). |

## `initial`

initial_velocity: uniform -- a uniform flow with an optional shedding trigger.

| Key | Type | Default | Description |
|---|---|---|---|
| `initial.velocity` | real[dim] | `[1, 0]` | The uniform velocity. |
| `initial.perturbation` | real | `0` | Amplitude (relative to \|velocity\|) of a Gaussian bump exp(-\|x - c\|^2) added to the second velocity component; breaks the symmetry so vortex shedding starts early (0 = none). |
| `initial.perturbation_center` | real[dim] | `[0, 0]` | The bump's centre c. |

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
| `output.progress` | real | `0` | Print a progress line every this many time units (0 = never): t, dt and its range, CFL and where it peaks, forces, iterations, cells, wall time, ETA. |
| `output.history` | bool | `false` | Write a per-step <path>/<name>_history.csv (t, dt, c_d, c_l, iterations, cells, step wall time). |

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
| `amr.every_time` | real | `0` | Adapt every this many time units instead of every interval steps (0 = step-based); use it with CFL-controlled steps. |
| `amr.start_time` | real | `0` | Time-based events only from this time on (spin up on the coarse mesh). |
| `amr.end_time` | real | `0` | Time-based events only up to this time (0 = no end). |
| `amr.first_event_passes` | int | `1` | Refinement passes at the first time-based event (stops when a pass changes nothing). |

## `forces`

Lift and drag on a body (John's volume integral of the momentum residual).

| Key | Type | Default | Description |
|---|---|---|---|
| `forces.enabled` | bool | `false` | Compute forces. |
| `forces.attributes` | int list | `[]` | Boundary attributes of the body. |
| `forces.boundaries` | string list | `[]` | The body by boundary name (e.g. [body], [cylinder], [ymin]); alternative to attributes. |
| `forces.reference_velocity` | real | `1` | U in C = 2F / (rho U^2 A). |
| `forces.reference_area` | real | `1` | A in C = 2F / (rho U^2 A) (a length in 2D). |
| `forces.interval` | int | `1` | Log to <path>/<name>_forces.csv every N steps (0 = no log). |
| `forces.statistics` | bool | `false` | Evaluate the coefficients every step; the run summary reports final values, maxima and averages over shedding periods (mean C_D, mean and rms C_L, Strouhal number = reference_area / (reference_velocity period) in 2D). |
| `forces.average_periods` | int | `10` | Shedding periods (the last ones) the averages cover. |

## `checkpoint`

Rolling checkpoints.

| Key | Type | Default | Description |
|---|---|---|---|
| `checkpoint.enabled` | bool | `false` | Write rolling checkpoints. |
| `checkpoint.path` | string | `chk` | Checkpoint directory (overwritten each time). |
| `checkpoint.interval` | int | `10` | Write every N accepted steps. |
| `checkpoint.at_time` | real | `0` | Also write one checkpoint once t >= at_time, to branch runs from (0 = off; independent of enabled). |

## `probes`

Point values reported in the run summary.

| Key | Type | Default | Description |
|---|---|---|---|
| `probes.pressure_difference` | two points | `[]` | [[x1, y1], [x2, y2]]: the summary reports p(x1) - p(x2) at the end (e.g. the DFG benchmark's front-back pressure difference). |

## `reference`

Benchmark values the run summary compares against (relative errors; absolute for times).

| Key | Type | Default | Description |
|---|---|---|---|
| `reference.cd` | real | `unset` | Final drag coefficient. |
| `reference.cl` | real | `unset` | Final lift coefficient. |
| `reference.cd_mean` | real | `unset` | Period-mean drag coefficient. |
| `reference.cl_mean` | real | `unset` | Period-mean lift coefficient. |
| `reference.cl_rms` | real | `unset` | Period-rms lift coefficient. |
| `reference.strouhal` | real | `unset` | Strouhal number. |
| `reference.cd_max` | real | `unset` | Maximum drag coefficient. |
| `reference.t_cd_max` | real | `unset` | Time of the maximum drag. |
| `reference.cl_max` | real | `unset` | Maximum lift coefficient. |
| `reference.t_cl_max` | real | `unset` | Time of the maximum lift. |
| `reference.pressure_difference` | real | `unset` | The pressure-difference probe's value. |
