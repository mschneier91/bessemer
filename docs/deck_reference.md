# Deck reference

Every key a YAML deck (`apps/run_case <deck.yaml>`) understands, with its default. Omitted keys keep their defaults; an unknown key is an error that names the closest valid key.

**Generated** by `build/cpu/apps/run_case --deck-reference` from the loader itself (`src/config/parameters.cpp`); `deck_test` fails when this file is out of date. Do not edit by hand.

## Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `equation` | stokes \| navier_stokes \| linear_elasticity \| mesh_motion | `stokes` | Equation set: unsteady Stokes, incompressible Navier-Stokes, a small-strain linear elastic solid, static or dynamic (section solid), or mesh_motion: deform the mesh by prescribed boundary displacements (section mesh_motion). |
| `device` | string | `cpu` | MFEM device backend (cpu, cuda, debug); the env var INCNS_DEVICE wins. |
| `initial_velocity` | string | `zero` | Named initial condition: zero, uniform, channel (see the initial section), taylor_green_2d or taylor_green_3d (u = (sin x cos y cos z, -cos x sin y cos z, 0) on a 2 pi periodic box); anything else comes from Python or C++. |
| `restart` | string | `""` | Checkpoint directory to restart from (same number of MPI ranks). |
| `boundary_conditions` | list | `[]` | Boundary groups, each {select: [names or attributes], type: no_slip \| outflow \| velocity \| velocity_dirichlet, ...}; a solid (equation linear_elasticity) takes type fixed \| displacement \| traction \| free instead, displacement and traction with value: [x, y(, z)] (a traction is sigma n, force per unit area); a displacement may instead give expression: one string per component in x, y, z, t (+ - * / ^, sin, cos, exp, sqrt, abs, min, max, ..., pi), with optional parameters: {name: value} usable in it. equation mesh_motion takes fixed \| displacement. Names: box faces xmin, xmax, ymin, ymax, zmin, zmax; the geometry's boundary names; or all. type velocity takes value: [u, v] (constant), or profile: parabolic with u_max and height (u_x = 4 u_max y (height - y) / height^2; in 3D times 4 z (width - z) / width^2, width defaulting to height), optionally time_profile: ramp \| sine with time_scale T (ramp: sin^2(pi t / 2T) for t < T; sine: sin(pi t / T)). velocity_dirichlet takes its field from Python or C++ by the group name. A flow boundary may move (the ALE fluid, docs/ale_fluid.md): motion: one string per component, its mesh displacement in the reference x, y, z and t, with parameters as for expression; a moving no_slip wall takes u = w (the mesh velocity), boundaries without motion keep their mesh in place, and the interior follows by the model of section mesh_motion, at every step. Every real boundary needs exactly one group when run by run_case. |

## `physics`

Physical parameters.

| Key | Type | Default | Description |
|---|---|---|---|
| `physics.nu` | real | `1` | Kinematic viscosity (dimensionless: 1/Re). |
| `physics.Re` | real | `unset` | Reynolds number; sets nu = 1/Re (dimensionless mode; not with nu). |
| `physics.grad_div_h` | real | `-1` | Grad-div stabilization gamma (div u, div v) on the velocity block, gamma = grad_div_h * h_K + grad_div_nu * nu: the part proportional to the element size h_K. < 0: 6 / k_u, i.e. gamma = 6 h_K / k_u (2 h_K at k_u = 3), scaling with the resolution. Both 0 = off. |
| `physics.grad_div_nu` | real | `0` | Grad-div: the part of gamma proportional to the viscosity (see grad_div_h). |
| `physics.convective_form` | convective \| rotational | `convective` | Nonlinear term: explicit (u.grad)u, or the semi-implicit rotational form (the pressure becomes the Bernoulli head). |
| `physics.outflow` | directional \| classical | `directional` | Condition on outflow boundaries: Braack-Mucha directional do-nothing (stable under backflow) or classical do-nothing. |
| `physics.subgrid.model` | none \| constant \| wale \| sigma | `none` | Subgrid-scale model of a large-eddy simulation, an eddy viscosity nu_t in the explicit stress term (2 nu_t S(u), S(v)) beside the convection (navier_stokes, convective form only; docs/les.md). sigma: Nicoud et al. 2011 (recommended; zero in pure shear, solid rotation and two-component flow). wale: Nicoud & Ducros 1999. constant: nu_t = coefficient everywhere (verification). none: DNS. |
| `physics.subgrid.coefficient` | real | `-1` | The model constant: C_sigma (sigma), C_w (wale) or nu_t itself (constant). < 0: the model's published value, 1.35 for sigma and 0.5 for wale. |
| `physics.subgrid.filter_width` | real | `1` | Filter width as a multiple of the element's node spacing: Delta_K = filter_width * h_K / order_u. |

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
| `discretization.order_u` | int | `6` | Velocity polynomial order k_u. |
| `discretization.order_p` | int | `5` | Pressure polynomial order k_p (Taylor-Hood: k_u - 1, also what a deck that sets only order_u gets). |
| `discretization.fuse_grad_div` | bool | `true` | Apply the grad-div term with the viscous term in one kernel (the velocity gradients once per apply; compact vector_diffusion only). false: two kernels. |
| `discretization.vector_diffusion` | compact \| mfem | `compact` | Storage of the viscous term's partial-assembly data. compact: one symmetric matrix per quadrature point shared by the velocity components (6 numbers in 3D). mfem: MFEM's VectorDiffusionIntegrator, one per component (27); the same operator to roundoff. |
| `discretization.mass` | auto \| collocated \| consistent | `auto` | Velocity mass matrix. auto: collocated under OIFS, consistent under IMEX. collocated: GLL quadrature, diagonal on conforming meshes (OIFS's substeps then invert it pointwise). consistent: exact quadrature (under OIFS, a mass solve per substep stage). OIFS's substeps always invert this same mass. |
| `discretization.hanging_mass` | lumped \| exact | `lumped` | The collocated mass at hanging nodes (AMR), where it is not diagonal. lumped: its row sums (Nek5000's non-conforming lumping), so it stays diagonal; with one hanging level at least the conforming weight on straight-sided and low-order-geometry elements, hence amr.nc_limit 1 is required; checked at every assembly (the run aborts if a patch is not covered). exact: P^T D P, with a mass solve per OIFS substep stage. |

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
| `mesh.geometry` | box \| square_cylinder \| cylinder_channel \| file | `box` | Domain: the box above; a square cylinder in a large domain (Joly et al. 2012; 2D; boundaries inflow, outflow, sides, body); the DFG channel with a cylinder (2D, or extruded in z in 3D; boundaries inflow, outflow, walls, cylinder, and flag with a Turek-Hron flag); or a mesh file (mesh.file; boundaries named by the file's Gmsh physical groups and mesh.boundary_names). |
| `mesh.file` | string | `""` | geometry file: a Gmsh (.msh, ASCII 2.2 or 4.x) or MFEM (.mesh) file of quadrilaterals (2D) or hexahedra (3D); relative paths are relative to the working directory. |
| `mesh.boundary_names` | name: attribute | `{}` | geometry file: boundary names for attribute numbers, added to (and overriding) the Gmsh file's named physical groups, e.g. {inlet: 1, walls: 3}. |
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
| `mesh.cylinder_channel.length` | real | `2.2` | Channel length (x). |
| `mesh.cylinder_channel.cx` | real | `0.2` | Cylinder centre x. |
| `mesh.cylinder_channel.cy` | real | `0.2` | Cylinder centre y. |
| `mesh.cylinder_channel.n_side` | int | `4` | Cells along each side of the O-grid square. |
| `mesh.cylinder_channel.n_ring` | int | `3` | Radial cell layers around the cylinder. |
| `mesh.cylinder_channel.height` | real | `0.41` | Channel height (y). |
| `mesh.cylinder_channel.radius` | real | `0.05` | Cylinder radius. |
| `mesh.cylinder_channel.box_half` | real | `0.1` | Half-width of the O-grid's outer square (> radius). |
| `mesh.cylinder_channel.ring_grading` | real | `1.5` | Radial growth ratio of the ring layers (> 1 clusters at the cylinder). |
| `mesh.cylinder_channel.n_up` | int | `2` | Cells upstream of the O-grid. |
| `mesh.cylinder_channel.n_down` | int | `16` | Cells downstream of the O-grid. |
| `mesh.cylinder_channel.n_below` | int | `2` | Cells below the O-grid. |
| `mesh.cylinder_channel.n_above` | int | `2` | Cells above the O-grid. |
| `mesh.cylinder_channel.down_grading` | real | `1.1` | Growth ratio of the downstream cells toward the outflow. |
| `mesh.cylinder_channel.outer_grading` | real | `1` | Growth ratio of the upstream, lower and upper cells away from the O-grid (1: uniform). |
| `mesh.cylinder_channel.flag_length` | real | `0` | Turek-Hron flag behind the cylinder: its length from the cylinder's downstream end to its tip (0: no flag; Turek & Hron 2006: 0.35). Boundary `flag`. |
| `mesh.cylinder_channel.flag_thickness` | real | `0.02` | Flag thickness, centred on y = cy (Turek & Hron: 0.02). |
| `mesh.cylinder_channel.n_flag` | int | `8` | Cells along the flag downstream of the O-grid. |
| `mesh.cylinder_channel.n_flag_y` | int | `1` | Cells across the flag. |
| `mesh.cylinder_channel.part` | fluid \| solid | `fluid` | fluid: the channel around the cylinder (and flag); solid: the flag itself (boundaries flag and cylinder, its clamp), with the fluid mesh's nodes on the interface. |
| `mesh.cylinder_channel.level` | int | `0` | Nested uniform refinement level (cell counts x 2^level). |
| `mesh.cylinder_channel.nz` | int | `4` | 3D: element layers in z (x 2^level). |
| `mesh.cylinder_channel.depth` | real | `0.41` | 3D: extent in z; the z = 0 and z = depth faces join the walls. |

## `initial`

Settings of the named initial conditions. uniform: a uniform flow with an optional shedding trigger. channel (walls at y = 0 and y = 2 delta, x streamwise): Reichardt's mean profile for Re_tau = u_tau delta / nu, u_tau = sqrt(f_x delta) from forcing, plus a divergence-free perturbation that vanishes with its normal derivative at the walls.

| Key | Type | Default | Description |
|---|---|---|---|
| `initial.velocity` | real[dim] | `[1, 0]` | uniform: the uniform velocity. |
| `initial.perturbation` | real | `0` | uniform: amplitude (relative to \|velocity\|) of a Gaussian bump exp(-\|x - c\|^2) added to the second velocity component, which starts vortex shedding early. channel: amplitude of the perturbation relative to the centreline velocity. 0 = none. |
| `initial.perturbation_center` | real[dim] | `[0, 0]` | uniform: the bump's centre c. |
| `initial.seed` | int | `1` | channel: seed of the perturbation's random mode phases. |

## `forcing`

A constant body force per unit mass.

| Key | Type | Default | Description |
|---|---|---|---|
| `forcing.body_force` | real[dim] | `[0, 0]` | The force f, e.g. a mean pressure gradient: [1, 0, 0] drives a channel of half-height 1 at u_tau = 1, so Re_tau = 1 / nu. Zero = none. |

## `channel_statistics`

Channel statistics (walls normal to y, x and z homogeneous; a box mesh without refinement): time averages of x-z plane-averaged U, V, W and the Reynolds stresses at every velocity-node height, the wall shear stress, u_tau and Re_tau. Written to <path>/<name>_profiles.csv and the run summary; carried through checkpoints.

| Key | Type | Default | Description |
|---|---|---|---|
| `channel_statistics.enabled` | bool | `false` | Collect them. |
| `channel_statistics.start_time` | real | `0` | Average over t >= start_time (skip the transient). |
| `channel_statistics.interval` | int | `1` | Sample every this many steps. |

## `time`

Time integration.

| Key | Type | Default | Description |
|---|---|---|---|
| `time.dt` | real | `0.01` | Step size (fixed steps), or the first step (cfl / error control). |
| `time.t_final` | real | `1` | End time. |
| `time.step_control` | cfl \| fixed \| error | `cfl` | How dt is chosen: dt = cfl_target / CFL rate before every step (Navier-Stokes), constant, or BDF2/BDF3 error control (IMEX only). |
| `time.adaptive` | bool | `unset` | Older spelling: true = step_control error, false = fixed. |
| `time.cfl_target` | real | `-1` | Target CFL number (Nek5000's definition) under step_control cfl. <= 0: 2 under OIFS, 0.5 under IMEX. |
| `time.cfl_max` | real | `0` | CFL ceiling (0 = off): caps error-controlled dt, aborts fixed steps above it. |
| `time.dt_max` | real | `0` | Largest step under cfl control (0 = no cap). |
| `time.on_solver_failure` | abort \| cut_step | `cut_step` | abort: a solve that does not converge within solver.max_iter stops the run, and solver.kdim is used as given. cut_step: it rejects the step, multiplies dt by retry_factor and solves again (fixed and cfl step control), and on a GPU the FGMRES restart length is lowered below solver.kdim when its Krylov vectors would not fit in the free memory. |
| `time.retry_factor` | real | `0.5` | cut_step: dt multiplier after a failed solve, in (0, 1). |
| `time.retry_hold` | int | `5` | cut_step: accepted steps after a rejected one during which dt may not grow. |
| `time.retry_max` | int | `8` | cut_step: cuts allowed within one step before the run stops. |
| `time.retry_dt_min` | real | `0` | cut_step: stop the run instead of cutting dt below this (0 = no floor). |
| `time.convection` | auto \| imex \| oifs | `auto` | Convection: auto (OIFS where it applies -- Navier-Stokes, the convective form, fixed or cfl steps -- else IMEX), explicit and extrapolated (IMEX, CFL < ~0.7), or OIFS sub-stepping (CFL 2; BDF3 and, with discretization.mass auto, the collocated mass by default). |
| `time.oifs_cfl` | real | `1.5` | CFL number of each OIFS substep. |
| `time.oifs_on_the_fly` | bool | `unset` | Older spelling of memory.max_dofs. |
| `time.order` | int | `0` | BDF order 2 or 3; 0 = auto (3 under OIFS, 2 under IMEX). |
| `time.ext_order` | int | `2` | Extrapolation order of the explicit/lagged nonlinear term, 2 or 3. |
| `time.atol` | real | `1e-08` | Error control: absolute tolerance on the velocity error estimate. |
| `time.rtol` | real | `1e-06` | Error control: relative tolerance on the velocity error estimate. |

## `memory`

Trading speed for capacity.

| Key | Type | Default | Description |
|---|---|---|---|
| `memory.max_dofs` | bool | `false` | Fit more DOFs on each GPU at a cost in speed: OIFS at velocity orders 6-7 on straight-sided elements rebuilds the convection's quadrature data at every substep instead of storing it (about 90 B/DOF less, about 6-7 % slower per step). A run whose GPU is 85 % full suggests it where it applies. |

## `solver`

Linear solver: FGMRES on the monolithic velocity-pressure system with a block preconditioner.

| Key | Type | Default | Description |
|---|---|---|---|
| `solver.rtol` | real | `1e-06` | FGMRES relative tolerance. |
| `solver.atol` | real | `0` | FGMRES absolute tolerance. |
| `solver.max_iter` | int | `2000` | FGMRES iteration cap. |
| `solver.kdim` | int | `200` | FGMRES restart size. |
| `solver.tolerance_reference` | initial_residual \| rhs \| step | `step` | What solver.rtol is relative to. initial_residual: \|\|r\|\| <= rtol \|\|r_0\|\| (MFEM's; a better initial guess then saves no iterations). rhs: \|\|r\|\| <= rtol \|\|b\|\|, so that solver.initial_guess pays off (loose at a small dt: \|\|b\|\| holds c0 M u_n). step: \|\|r\|\| <= rtol \|\|b - A x_prev\|\|, the previous solution's residual -- rtol of what the step changes, at any dt, with initial_guess project at no extra operator application. |
| `solver.initial_guess` | previous \| extrapolate \| project | `project` | The outer solve's initial guess. previous: the last step's solution. extrapolate: its linear extrapolation in time (one more system vector). project: the least-squares projection onto the last solver.projection_k solutions (2 projection_k system vectors). Pair with tolerance_reference: rhs. |
| `solver.projection_k` | int | `8` | initial_guess project: the most solutions kept (>= 1). |
| `solver.projection_rebuild_threshold` | real | `0.05` | initial_guess project: recompute the kept solutions' images (k operator applications) once the mass factor beta0/dt has moved by more than this fraction since they were made; until then the guess uses the older operator's images (valid, less good). |
| `solver.print_level` | int | `-1` | Solver verbosity (-1 = quiet). |
| `solver.schur` | cc \| mass \| laplacian_legacy | `cc` | Pressure Schur block: Cahouet-Chabard, scaled pressure mass, or the legacy Laplacian CC (comparison only). |
| `solver.a_pc` | jacobi_pcg \| jacobi_chebyshev \| loramg | `jacobi_pcg` | Velocity-block preconditioner (Cahouet-Chabard path): Jacobi-PCG, Jacobi-Chebyshev, or one LOR-AMG V-cycle (best when viscous-dominated). |
| `solver.a_pcg_pc` | auto \| jacobi \| pbj | `auto` | jacobi_pcg: what preconditions the inner CG. jacobi: the diagonal. pbj: point-block Jacobi, the dim x dim block at each velocity node, which holds the grad-div term's coupling of the components. auto: pbj when grad-div is on, jacobi otherwise. |
| `solver.schur_grad_div` | bool | `true` | Cahouet-Chabard: put the grad-div coefficient gamma in the Schur preconditioner's pressure-mass term (the mass weighted by 1/(nu + gamma) instead of nu M_p^-1). |
| `solver.a_pcg_rtol` | real | `0.01` | jacobi_pcg: inner CG tolerance. |
| `solver.a_pcg_max_iter` | int | `50` | jacobi_pcg: inner CG iteration cap. |
| `solver.preconditioner` | jacobi \| loramg | `jacobi` | Velocity-block preconditioner on the mass Schur path. |
| `solver.block_shape` | upper \| lower \| diag | `upper` | Block preconditioner shape. |
| `solver.n_inner` | int | `2` | Cahouet-Chabard: fixed inner CG iterations on the pressure Poisson block. |
| `solver.lp_strong_threshold` | real | `-1` | Pressure LOR-AMG: BoomerAMG strength threshold (< 0: 0.25 in 3D, 0.1 in 2D). On a GPU (PMIS coarsening) 0.5 and 0.6 gave LARGER hierarchies than 0.25 (Taylor-Green Q7, 2026-10-16). |
| `solver.lp_agg_levels` | int | `0` | Pressure LOR-AMG: levels of aggressive coarsening (0 = none). |
| `solver.lp_print_level` | int | `0` | Pressure LOR-AMG: hypre's setup print level (1 prints the grid and operator complexities). |
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
| `solver.outer` | coupled \| schur | `coupled` | Outer iteration. coupled: FGMRES on the velocity-pressure system with the block preconditioner. schur: flexible CG on the pressure Schur complement B A^-1 B^T (Uzawa), each operator application a velocity solve; same preconditioners and the same coupled-residual stopping rule (solver.rtol, atol, tolerance_reference), a few pressure and velocity vectors instead of the coupled Krylov basis. Needs solver.schur cc or laplacian_legacy and a symmetric momentum block (not the rotational form). initial_guess project then projects the pressure only. |
| `solver.log_interval` | int | `0` | Write the per-step solver log (<name>_solver.csv: iterations, inner work, timings, final coupled residual, device memory) every N steps (0 = off). |
| `solver.uzawa.inner_tolerance` | relaxed \| tight \| loose | `relaxed` | outer schur: the velocity solve inside each Schur application. relaxed: min(eps_max, max(eps_tight, relax_c tol / \|\|r\|\|)). tight: eps_tight. loose: eps_loose. |
| `solver.uzawa.eps_tight` | real | `1e-10` | outer schur: tight inner relative tolerance (relaxed: the floor). |
| `solver.uzawa.eps_loose` | real | `0.001` | outer schur: loose inner relative tolerance. |
| `solver.uzawa.eps_max` | real | `0.01` | outer schur, relaxed: the loosest inner relative tolerance. |
| `solver.uzawa.relax_c` | real | `1` | outer schur, relaxed: the constant c of the schedule. |
| `solver.uzawa.inner_max_iter` | int | `500` | outer schur: iteration cap of each velocity solve. |
| `solver.uzawa.max_cycles` | int | `3` | outer schur: CG cycles, each ended by a velocity solve and a true pressure-residual check that restarts the iteration if the residual drifted above the tolerance. |
| `solver.uzawa.history` | bool | `false` | outer schur: record every iteration (recurrence residual, inner tolerance and iterations, coupled residual of the iterate: one momentum application each) for solver.log_interval. |

## `output`

ParaView output and diagnostics.

| Key | Type | Default | Description |
|---|---|---|---|
| `output.enabled` | bool | `false` | Write ParaView output. |
| `output.path` | string | `.` | Output directory prefix. |
| `output.name` | string | `case` | Collection name. |
| `output.interval` | int | `1` | Write every N accepted steps. |
| `output.diagnostics` | bool | `false` | Log t, kinetic energy (1/2 int \|u\|^2), dissipation (nu int \|grad u\|^2) and \|\|div u\|\| to <path>/<name>_diagnostics.csv every interval steps (independent of enabled; appended on restart). |
| `output.progress` | real | `0` | Print a progress line every this many time units (0 = never): t, dt and its range, CFL and where it peaks, forces, iterations, cells, wall time, ETA. |
| `output.history` | bool | `false` | Write a per-step <path>/<name>_history.csv (t, dt, c_d, c_l, iterations, OIFS substeps, cells, step wall time, velocity-block inner iterations, inner solves at their cap). |
| `output.profile` | off \| wall \| sync | `wall` | Where the run's time went: at the end, apps/run_case writes the profiler's region tree (calls, inclusive and exclusive wall seconds, max and min over the ranks) to <path>/<name>_profile.csv and its top levels to the summary. wall: plain wall time; on a GPU, asynchronous kernels are charged to the region that next waits for the device. sync: wait for the device at every region boundary, so GPU time is charged where it is spent (slower: for profiling runs). off: neither. |

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
| `amr.threshold_mode` | relative \| absolute | `absolute` | Mark where the indicator >= theta * its max (relative) or >= tolerance (absolute). |
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
| `reference.re_tau` | real | `unset` | Channel: Re_tau from the time-averaged wall shear stress. |

## `solid`

The solid of equation linear_elasticity: small strains, isotropic and homogeneous, displacement-only on the reference configuration (docs/elasticity.md). Units are the deck's own, consistent ones.

| Key | Type | Default | Description |
|---|---|---|---|
| `solid.youngs_modulus` | real | `1` | Young's modulus E. |
| `solid.poisson_ratio` | real | `0.3` | Poisson's ratio (-1 < nu < 1/2). A displacement-only discretization locks as nu -> 1/2; high orders delay it. |
| `solid.plane` | strain \| stress | `strain` | 2D model: plane strain, or plane stress (lambda* = 2 lambda mu / (lambda + 2 mu)). Ignored in 3D. |
| `solid.order` | int | `0` | Displacement polynomial order; <= 0: discretization.order_u, so the solid matches the fluid's velocity at a future interface. |
| `solid.body_force` | real[dim] | `[0, 0]` | A constant body force per unit volume. |
| `solid.rtol` | real | `1e-10` | CG relative tolerance. |
| `solid.atol` | real | `0` | CG absolute tolerance. |
| `solid.max_iter` | int | `2000` | CG iteration cap. |
| `solid.preconditioner` | loramg \| jacobi | `loramg` | CG preconditioner: one BoomerAMG V-cycle (systems mode) on the low-order-refined operator, or the operator's diagonal (no setup, iterations grow with the resolution). |
| `solid.probe` | real[dim] | `unset` | A point whose displacement the run summary reports (e.g. the tip of a cantilever). |
| `solid.analysis` | static \| dynamic | `static` | static: one solve of K u = f. dynamic: rho u_tt + K u = f marched from rest at t = 0 to time.t_final in fixed steps of time.dt (the last one shortened to land on t_final); the loads act from t = 0. |
| `solid.density` | real | `1` | Mass density rho (dynamic analysis). |
| `solid.time_scheme` | generalized_alpha \| bdf | `generalized_alpha` | Dynamic time integrator: Chung-Hulbert generalized-alpha (second order, high-frequency damping set by rho_inf), or the fluid's variable-step BDF (order bdf_order) on the first-order system (u, v), started by one trapezoidal step. |
| `solid.bdf_order` | int | `2` | BDF order, 2 or 3. BDF3 is not A-stable: an undamped mode with omega dt between about 0.5 and 2 grows (4 % per step at omega dt = 1), so it suits short or damped runs only; BDF2 and generalized-alpha are unconditionally stable. |
| `solid.rho_inf` | real | `0.8` | Generalized-alpha spectral radius at infinite frequency, in [0, 1]: 1 is the trapezoidal rule (energy-conserving, no damping), smaller values damp the unresolved high frequencies more. |

## `mesh_motion`

Equation mesh_motion: the mesh is deformed by the boundary groups of type displacement, every other boundary held (type fixed), the displacement extended into the domain by a pseudo-solid (docs/mesh_motion.md). A flow deck whose boundaries have motion uses the same model to move its mesh at every step (docs/ale_fluid.md); steps is then not used.

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh_motion.model` | linear_elastic | `linear_elastic` | The extension model: small-strain linear elasticity on the reference mesh with Jacobian stiffening (Stein, Tezduyar and Benney 2003). |
| `mesh_motion.poisson_ratio` | real | `0.3` | Poisson's ratio of the pseudo-solid (-1 < nu < 1/2); its Young's modulus does not matter (the problem is displacement-driven). |
| `mesh_motion.stiffening` | real | `1` | Jacobian stiffening exponent chi >= 0: each element's stiffness times (V_mean / V_K)^chi, so small elements move nearly rigidly; 0 is a homogeneous solid. |
| `mesh_motion.order` | int | `0` | Displacement and geometry order; <= 0: discretization.order_u. A curved geometry (which has its own nodes) keeps its order. |
| `mesh_motion.steps` | int | `1` | Moves at t_k = k time.t_final / steps, k = 1..steps (the expressions' t); 1 is a single static deformation at t_final. |
| `mesh_motion.rtol` | real | `1e-10` | CG relative tolerance. |
| `mesh_motion.atol` | real | `0` | CG absolute tolerance. |
| `mesh_motion.max_iter` | int | `2000` | CG iteration cap. |
| `mesh_motion.basis_size` | int | `12` | Earlier solutions kept for reuse (0: none). The displacement is linear in the boundary data, so a move whose data lie in the span of earlier data (a rigid or modal motion: a few shapes times functions of t) is the same combination of the earlier solutions, with no solve. |
| `mesh_motion.preconditioner` | loramg \| jacobi | `loramg` | CG preconditioner: one BoomerAMG V-cycle on the low-order-refined operator, or its diagonal. |
