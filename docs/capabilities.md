# bessemer: what the library can do

A high-order finite element solver for incompressible flow, solids, moving meshes and
fluid-structure interaction, built on MFEM, running MPI-parallel on CPUs and NVIDIA GPUs.

The one-page map of what bessemer does, how well it is validated, and what it does not do:
check here first whether a request is supported. **Keep it current:** whoever lands a
capability, a limit or a validated result updates this page in the same change
([AGENTS.md](../AGENTS.md) rule 8).

**As of** 2026-10-27: everything on `main`, validated on CPUs at 1, 2 and 4 ranks and on
H100s at 1 and 2; the flow statistics, which landed after Bridges-2 went down, on A100s at 1
and 2 instead ([status.md](status.md)).

Details live elsewhere: [deck_reference.md](deck_reference.md) has every input key (it is
generated from the code), [using.md](using.md) how to run cases and read their outputs,
[status.md](status.md) what is in progress, the method docs in this directory how things
work, and the benchmark READMEs under [../benchmarks](../benchmarks) the validation runs.

---

## At a glance

| Area | What exists |
|---|---|
| Equations | Unsteady Stokes, incompressible Navier-Stokes (DNS or LES), linear elasticity, hyperelasticity (St. Venant-Kirchhoff, neo-Hookean), mesh motion, FSI |
| Discretization | Continuous Taylor-Hood Q_k/Q_(k-1) on quads and hexes, any order (default Q6/Q5), partial assembly, grad-div stabilization |
| Time | BDF2/BDF3 with IMEX or OIFS convection; CFL, fixed or error-controlled steps; automatic step cutting on solver failure |
| Solvers | Monolithic velocity-pressure FGMRES or Uzawa Schur CG; Cahouet-Chabard and mass Schur preconditioners; Jacobi-PCG, point-block Jacobi, Chebyshev and LOR-AMG velocity preconditioners |
| Meshes | Periodic/stretched boxes, DFG cylinder channel (2D/3D, with Turek-Hron flag), square cylinder, Gmsh/MFEM files; adaptive refinement and optional coarsening, with hanging nodes |
| Moving meshes | Pseudo-solid mesh motion (linear elastic or neo-Hookean), ALE Navier-Stokes |
| Solids | Static/dynamic, small or large deformation, generalized-alpha or BDF, adaptive steps |
| FSI | Partitioned, strongly coupled: Dirichlet-Neumann, Robin-Neumann, Neumann-Dirichlet, Robin-Robin, each with the linear or the hyperelastic solid; Aitken or IQN-ILS; adaptive coupled steps |
| I/O | YAML decks, Python module, C++ drivers; JSON run summary, CSV logs, ParaView, checkpoint/restart |
| Statistics | Running mean velocity, pressure and Reynolds stresses on any fixed mesh, sampled on lines, azimuthally averaged profiles and wall bins (wall pressure and shear), with a half-window convergence check; channel plane averages |
| Platforms | MPI on CPUs; NVIDIA GPUs (CUDA, H100-validated), GPU-aware MPI |

---

## 1. Incompressible flow

### Equations and discretization
- `equation: stokes` (unsteady Stokes) or `navier_stokes`.
- **Monolithic:** velocity and pressure solved together each step, so there is no splitting
  error.
- **Taylor-Hood** continuous spaces, orders set at run time (`order_u`, `order_p`; default
  Q6/Q5). Isoparametric curved geometry.
- **Partial assembly** throughout, with in-house fused kernels: a compact symmetric
  viscous term and grad-div fused with viscosity.
- **Grad-div stabilization** gamma(div u, div v), with gamma = c_h h_K + c_nu nu. On by
  default (gamma = 6 h_K / k_u).
- **Velocity mass:** consistent, or collocated (GLL, diagonal). Hanging nodes take lumped
  or exact mass.
- **Nonlinear term:**
  - convective form, explicit;
  - rotational form, semi-implicit, with the pressure as the Bernoulli head and dedicated
    preconditioners.
- **Nondimensionalization:** dimensionless (nu = 1/Re), or dimensional with L_ref, U_ref
  and rho.

### Time integration
- **BDF2 or BDF3**, extrapolation order 2 or 3.
- **Convection:**
  - **IMEX:** explicit, extrapolated, CFL up to about 0.7.
  - **OIFS:** operator-integration-factor sub-stepping (Nek5000's approach), CFL about 2,
    BDF3 and the collocated mass.
  - `auto` picks OIFS where it applies.
- **Step control:**
  - `cfl`: dt set to a target CFL rate (Nek5000 definition), with `dt_max` and `cfl_max`.
  - `fixed`.
  - `error`: BDF2/BDF3 local error estimate with a PI controller (IMEX).
- **Failure recovery (`cut_step`, default):**
  - A failed linear solve rejects the step and retries at `retry_factor`·dt, with
    `retry_hold`, `retry_max` and `retry_dt_min`.
  - On a GPU the FGMRES restart length shrinks to fit free memory.

### Linear solvers and preconditioners
- **Outer iteration:**
  - FGMRES on the coupled system (default);
  - or flexible CG on the pressure Schur complement (Uzawa), with relaxed inner
    tolerances.
- **Schur block:** Cahouet-Chabard (default), scaled pressure mass, or the legacy
  Laplacian.
- **Pressure Poisson:**
  - Low-order-refined (LOR) BoomerAMG V-cycles inside fixed inner CG iterations.
  - Strength threshold 0.25 in 2D and 3D (2D was 0.1 until 2026-10-24).
  - Aggressive coarsening optional.
- **Velocity block:**
  - Jacobi-PCG with scalar or point-block Jacobi (the dim×dim nodal blocks that carry
    grad-div);
  - Jacobi-Chebyshev;
  - LOR-AMG in systems mode.
- **Fold-safe LOR:** on strongly deformed curved meshes, LOR sub-elements that fold are
  detected at every setup (a device kernel) and assembled with |det J|. This keeps the pressure and velocity
  LOR operators SPD and fixed the FSI2 stalls. Meshes without folds are unaffected
  (bitwise).
- **Block shapes:** upper, lower or diagonal.
- **Initial guess:** previous, linear extrapolation, or least-squares projection onto the
  last k solutions (default).
- **Tolerance reference:** initial residual, right-hand side, or the step's change
  (default).
- **Rotational form:**
  - point-block Jacobi or GMRES velocity preconditioning;
  - tensor or automatic Schur switching on the rotation number.
- **AMG reuse** across dt changes, GPU-friendly hypre options, a per-step solver log.

### Turbulence
- **LES:** explicit eddy-viscosity stress, `physics.subgrid.model`:
  - `sigma` (Nicoud et al. 2011, recommended);
  - `wale` (Nicoud & Ducros 1999);
  - `constant` (verification).
  - The filter width scales with node spacing.
  - `none` is DNS.
- Subgrid statistics (max and mean nu_t, subgrid dissipation) go in the run summary.

### Boundary and initial conditions, forcing
- **Boundary types:**
  - `no_slip`;
  - `velocity`: constant, or parabolic in 2D and 3D, optionally ramped or sine-modulated
    in time;
  - `velocity_dirichlet`: any field from Python or C++, by group name;
  - `outflow`: Braack-Mucha directional do-nothing, stable under backflow, or classical;
  - `interface` (FSI).
- **Moving walls:** a `motion:` expression per component in x, y, z and t, with
  parameters. Walls take u = w.
- **Robin conditions** on flow boundaries, in the library and used by the FSI couplings.
- Every boundary must be assigned exactly once; omissions are refused with the name.
- **Initial conditions:** zero; uniform (with an optional shedding trigger); turbulent
  channel (Reichardt profile plus a divergence-free perturbation); Taylor-Green 2D and 3D;
  any field from Python or C++.
- **Forcing:** a constant body force, for example a channel's mean pressure gradient.

### Adaptive mesh refinement
- **Refinement:** isotropic or anisotropic (a per-direction gradient test).
- Nonconforming meshes with hanging nodes (`nc_limit`), on curved geometry.
- **Marking:** gradient indicator with relative (theta) or absolute marking; size floor
  and element cap.
- **Events:** step-based or time-based (start, end, first-event passes); initial passes on
  the initial condition.
- **Coarsening (`amr.derefine`, optional, off by default):**
  - After an event's refinement passes, a family of sibling elements merges when twice
    its largest indicator is below `derefine_fraction` (default 0.5) of the refinement
    threshold.
  - That bound is the merged element's own, so a merged element is not refined again at
    the next event (hysteresis).
  - Every BDF/EXT history level is replaced by the same linear map: its exact L2
    projection onto the coarse space, then the divergence-free projection. The multistep
    history stays consistent, so no ramp is needed, and an event cannot raise the energy.
  - MFEM's own transfer, which interpolates, would miss the field's moments by 3e-7 to
    8e-6 of its norm; the projection keeps them to 1e-14.
  - 3D needs isotropic refinement (an MFEM limit).
- Partition rebalancing, and a divergence-free projection of the time history after each
  event.
- Indicator output, checkpoint support (refinements and coarsenings replay exactly), and
  `apps/mesh_census` for reviewing where a run refined.

---

## 2. Geometry and meshes
- `box`: 2D/3D, periodic in any direction, two-sided tanh clustering per direction.
- `cylinder_channel`: the DFG/Schäfer-Turek channel with an O-grid around the cylinder.
  - 2D, or 3D extruded in z.
  - Grading controls and nested uniform refinement levels.
  - An optional **Turek-Hron flag**; `part: solid` gives the flag's own mesh with matching
    interface nodes.
- `square_cylinder`: a square cylinder in a large domain (Joly et al. 2012), with wake
  grading.
- `file`: Gmsh (.msh 2.2 or 4.x) or MFEM (.mesh) quads or hexes. Boundary names come from
  Gmsh physical groups or `mesh.boundary_names`.

---

## 3. Moving meshes (ALE)
- **`equation: mesh_motion`:** deform a mesh by prescribed boundary displacements. The
  interior follows a pseudo-solid:
  - `linear_elastic`, with Jacobian stiffening so small elements move nearly rigidly;
  - or `neo_hookean`, frame-indifferent, resists element collapse, one Newton solve per
    move, for large rotations.
  - Solution-basis reuse skips solves for rigid or modal motions.
  - LOR-AMG CG.
- **ALE Navier-Stokes:**
  - Boundaries with `motion:` expressions move the mesh every step.
  - Non-conservative ALE form; the discrete geometric conservation law holds (free-stream
    test).
  - The convective CFL uses u − w.
  - Solvers are refreshed in place after each move, with an AMG rebuild interval, or
    rebuilt fully.
  - Failed steps revert the mesh.
- **Restrictions on moving meshes:** IMEX only (no OIFS), no rotational form, no AMR. No
  error-controlled steps for a fluid alone; FSI uses its own coupled-step control.

---

## 4. Solid mechanics
- **`equation: linear_elasticity`:**
  - small-strain, isotropic, homogeneous; 2D (plane strain or plane stress) or 3D;
  - orders 1-7 with an in-house operator, including 3D Q6 on the GPU;
  - static or dynamic.
- **`equation: hyperelasticity`:**
  - large deformation, total Lagrangian;
  - **St. Venant-Kirchhoff** or compressible **neo-Hookean**;
  - Newton with load steps and optional line search, residual and correction convergence
    tests;
  - static or dynamic.
- **Dynamics:**
  - Chung-Hulbert generalized-alpha, with rho_inf-controlled damping;
  - or the variable-step BDF2/BDF3 on the first-order system.
- **Adaptive steps:**
  - Error control with a PI controller on local error estimates (Zienkiewicz-Xie for
    generalized-alpha, Milne's device for BDF).
  - `cut_step` retries on Newton or CG failure.
- **Boundary conditions and loads:** `fixed`, `displacement`, `traction`, `free`;
  constants or space-time expressions. Body force. Point probes in the summary.
- **Solver:** CG with LOR-BoomerAMG (systems mode) or Jacobi.
- Checkpoint and restart of dynamic runs (bitwise on a host).

---

## 5. Fluid-structure interaction (`fsi` section)
- **Partitioned and strongly coupled:** subiterations every time step.
- **Interface conditions:**
  - Dirichlet-Neumann (default);
  - Robin-Neumann (Badia-Nobile-Vergara);
  - Neumann-Dirichlet;
  - Robin-Robin.
  - Robin coefficients come face by face from the solid's thickness, or are set.
- **Solid:** linear elastic, or hyperelastic SVK or neo-Hookean (dynamic), under all four
  conditions.
  - With the hyperelastic solid, the quantities exchanged at the interface are the load's
    nodal moments, which do not depend on the configuration. That matches the solid's
    total-Lagrangian loading.
  - Each condition reaches Dirichlet-Neumann's coupled solution (test F15).
- **Acceleration:** constant relaxation, dynamic Aitken, or IQN-ILS quasi-Newton.
  - IQN-ILS can reuse columns from earlier steps and filters near-dependent ones.
  - A failed attempt drops the reused columns.
- **Backtracking after a failed solid solve (`fsi.backtracks`, default 4):**
  - The subiteration's input is halved toward the last completed one, up to four times,
    before the step is cut.
  - On the CPU it carries Robin-Neumann with the hyperelastic solid through an impulsive
    start, where IQN-ILS's first step throws the load to 2.3 times itself.
  - That start sits at the edge of Robin-Neumann's convergence. On a GPU, roundoff can still
    send it down a path that cuts the step, so Robin-Robin is the condition for it there.
- **Predictor:** constant, linear or quadratic extrapolation (variable-step aware).
- **Mesh motion:** linear elastic or neo-Hookean (the latter needed for large-amplitude
  flags like FSI2).
- **Interfaces:** matching or non-matching (point search).
- **Coupled step control:**
  - `fixed`, `cfl` (the fluid's CFL step), or `error` (the smaller of the fluid CFL step
    and the solid's error-controlled step).
  - `cut_step` undoes the flow trial, the solid state and the accelerator's attempt, then
    retries.
- **Checkpoint and restart**, including the accelerator state, bitwise on a host.
- **Outputs:** a per-step CSV of subiterations, residual, forces and tip displacement;
  reference comparison of probe displacements.
- **GPU:** the FSI test suite passes on one and two H100s (2026-10-24). On one GPU it
  takes about 30 minutes. Production FSI runs so far have been on CPUs.

---

## 6. Outputs and post-processing
- **Run summary JSON (always):**
  - status (running, ok, diverged);
  - settings, step statistics, mesh size, diagnostics;
  - forces and shedding statistics, probes, subgrid statistics, channel statistics,
    flow statistics with their first-half / second-half comparison;
  - profile;
  - errors against `reference` values.
- **Per-step and periodic files:**
  - progress lines with an ETA;
  - history CSV (dt, coefficients, iterations, substeps, cells, wall time);
  - diagnostics CSV (kinetic energy, dissipation, ||div u||);
  - solver log CSV;
  - forces CSV.
- **Forces:** John's volume-integral lift and drag on named boundaries. Optional
  per-step statistics: means, rms, maxima with times, Strouhal number over the last N
  periods.
- **Probes:** pressure difference between two points; solid point displacement.
- **Channel statistics:** plane- and time-averaged U, V, W, Reynolds stresses, wall shear,
  u_tau, Re_tau, carried through restarts.
- **Flow statistics (`flow_statistics`):** for any fixed mesh, e.g. a nozzle or a
  separated flow.
  - Running time averages at every velocity and pressure node: the mean velocity and
    pressure and the Reynolds stresses. A sample is one pass over the nodes, nothing is
    communicated, and the solution is unchanged.
  - Everything else is computed from the averages only when written (at the end and at
    every checkpoint):
    - lines (Cartesian);
    - radial profiles at axial stations, averaged over the azimuth, in cylindrical
      components;
    - wall bins along an axis: mean wall pressure and wall shear stress, signed along
      the axis.
    - Each with the mean velocity, pressure, Reynolds stresses and the mean flow's
      viscous stress.
  - The fields also go to ParaView (`mean_velocity`, `mean_pressure`, `reynolds_normal`,
    `reynolds_shear`).
  - Convergence: the halves of the averaging window are compared per set and quantity
    in the summary, with CSVs of each half.
  - Carried through restarts.
- **ParaView:** high-order output (`.pvd`), optional AMR indicator.
- **Profiler:** a region tree with calls and inclusive and exclusive time (max and min
  over ranks). On the GPU, `sync` mode attributes kernel time and device memory per
  region.
- **Analysis scripts:**
  - `turek_hron/forces.py` and `fsi.py` (mean ± amplitude, frequency; `--smooth` removes
    the force ripple, `--periods N`);
  - `oscillating_cylinder/morison.py` (Fourier c_d and c_i);
  - `flow_stats.py` for the flow statistics CSVs: normalization (reference velocity and
    length, axial origin, pressure relative to a reference point, stresses over
    ½ρU²), jet-breakdown indicators along a centerline, separation and reattachment
    points on a wall, the half-window table;
  - `plot.py` per benchmark;
  - `fetch_references.py`.

## 7. Checkpoint and restart
- Rolling checkpoints every N steps, plus one at a chosen time for branching runs.
- What restarts:
  - flows, including the solver's initial-guess state, so a restart repeats the
    uninterrupted run's solves;
  - AMR meshes (the refinement and coarsening history replays exactly), channel and flow
    statistics;
  - dynamic solids;
  - FSI.
- Restart requires the same MPI rank count.

---

## 8. Running it
- **YAML decks:** `apps/run_case deck.yaml`. Unknown keys are errors that suggest the
  closest valid key; `run_case --deck-reference` prints every key.
- **Python (`incns`):**
  - `run_deck` takes a path, dict or `Parameters`;
  - `Parameters.from_dict` and `from_yaml` reach every key;
  - `Case` steps a flow from Python;
  - analytic fields are plain callables or numba-compiled `@incns.field`.
- **C++ drivers:** `dfg_cylinder`, `square_cylinder`, `taylor_green`, `unsteady_mms_3d`,
  `mesh_census`.
- **Platforms:**
  - MPI on CPUs;
  - NVIDIA GPUs via CUDA (H100-validated), GPU-aware MPI by default;
  - device backends `cpu`, `cuda`, `debug` (a host device that catches missing
    host/device syncs).
- **GPU memory:** about 1.95 kB per velocity DOF. One 80 GB H100 holds about 80³ Q3/Q2
  elements (41.5 M velocity DOFs). `memory.max_dofs` trades about 6-7 % speed for about
  90 B/DOF at orders 6-7.
- **Multi-GPU:** parallel efficiency at 4 GPUs about 77-83 % on the measured cases
  ([multi_gpu.md](multi_gpu.md)).
- **Tooling:**
  - a Spack toolchain (`scripts/setup.sh`);
  - `scripts/doctor.sh` (environment check, build, smoke run), `build.sh`, `test.sh`;
  - style (astyle), ASAN and debug-device scripts.

## 9. Verification and testing
- About 100 MPI gtest suites, run at 1, 2 and 4 ranks, in tiers:
  - **fast:** every module, order and exactness oracles;
  - **verify:** convergence rates against exact solutions at the case defaults;
  - **slow**;
  - **Python**.
  - GitHub Actions runs style, warnings-as-errors, Doxygen and the fast and Python tiers;
    verify runs periodically.
- **Exact-solution checks:**
  - Kovasznay flow;
  - manufactured solutions: steady, unsteady and 3D;
  - Taylor-Green Stokes temporal order;
  - free-stream preservation on moving meshes;
  - elasticity and hyperelasticity spatial and temporal orders;
  - AMR events, refinement and coarsening, exact at every step for a polynomial solution;
    coarsening never raises the energy;
  - each FSI condition against Dirichlet-Neumann, for both solids;
  - flow statistics: the running moments, the line, profile and wall samples and the
    half-window split exact for fields the space holds; the solution bitwise unchanged
    with them on;
  - restart bitwise tests.

---

## 10. Validated benchmarks

| Benchmark | Result |
|---|---|
| Taylor-Green vortex, Re 1600 (3D, GPU) | Dissipation peak within 0.17 % of the 512³ DNS at 252³ nodes (Q6); energy never rises |
| Turbulent channel, Re_tau 180 / 395 (GPU) | Re_tau within 0.13 % of Moser-Kim-Mansour; U+ within 1.5 % / 0.7 % |
| DFG 3D-1Z (steady) | c_D 0.006 %, c_L 0.03 % (Q6 + AMR) |
| DFG 3D-3Z (unsteady) | c_D,max 0.17 %, c_L,min 0.39 % |
| DFG 2D-1 / 2D-3 | 2D-1: c_D 1e-4, c_L 3e-3 relative. 2D-3: c_D,max within 0.3 %, c_L,max within about 9 % (base mesh) |
| Square cylinder, Re 200 | C_D 1.43, C_L,rms 0.39, St 0.157 (paper 1.44, 0.42, 0.151) |
| Turek-Hron CFD1 / CFD2 / CFD3 | Drag +0.12 % (L1) / +0.07 %; CFD3 drag +0.03 %, lift amplitude −0.24 % (L1) |
| Turek-Hron CSM1 / CSM2 / CSM3 | CSM1-2 point-A displacement within 1-2e-4; CSM3 oscillation within 0.4-1.7 % |
| Turek-Hron FSI1 | Drag +0.11 %, lift −0.5 % (L1); SVK solid: u_y −1.4 % |
| Turek-Hron FSI2 | Level 2 (Q4/Q3, about 209k fluid unknowns) against FEAT's finest refined results (level 4, 304k Q2/P1 unknowns): displacement amplitudes +0.8 / +0.3 %, drag amplitude +0.3 % (mean −0.1 %), lift 0.0 %, frequencies equal (1.93, 3.86). Force amplitudes good to about ±2 % (the run stopped at t = 12.8). The 2006 reference values are coarser (level 2 is 1.5-3 % above them) |
| Turek-Hron FSI3 | Level 1 (Q4/Q3, 41k velocity unknowns) sits at FEAT's level 3 (77k Q2/P1). Its amplitudes against FEAT's levels 3 / 4: displacements −4.4 / −7.2 % (u_x) and −2.3 / −3.8 % (u_y), drag −3.5 / −6.4 % (mean −0.3 %), lift −3.9 / −5.8 %; frequency +0.3 %. Refining from the coarse mesh moved them toward FEAT's finest, as with FSI2. Against the 2006 values: displacements −1 / −2.4 %; drag amplitude +13 %, because the 2006 drag amplitude is an outlier |
| Oscillating cylinder (Dütsch, ALE) | c_d 2.091, c_i 1.460 (reference 2.09 / 1.45), converged to about 0.1 % |
| Cantilever, Cook's membrane | Timoshenko-Goodier deflection and Cook's tip displacement, orders verified |

---

## 11. Not supported, or limits
- **Elements:** quadrilaterals and hexahedra only (no triangles or tetrahedra); continuous
  Taylor-Hood only.
- **Physics:** isothermal, Newtonian, single-phase incompressible flow. No energy
  equation or heat transfer, no RANS models.
- **AMR:** coarsening is opt-in, and in 3D isotropic only. AMR is not combined with
  moving meshes.
- **Moving meshes:** IMEX only (no OIFS, no rotational form, no AMR). No sliding
  boundaries.
- **FSI:**
  - The hyperelastic solid must be dynamic.
  - Its Newton can fail on the coupling's early iterates where the linear solid would not.
    Dirichlet-Neumann on a very soft flag (E = 1.4e3 on the test flag) throws the flag out
    15 thicknesses and cuts steps. Neumann-Dirichlet, the condition for soft solids, does
    not.
  - FSI3 has been refined to level 1, one level short of FSI2: 2-7 % off FEAT's finest.
    Level 2 would take about 8× level 1's 7 h at 32 cores (estimate).
- **Solid loads:** dead loads only (no follower or pressure loads).
- **Flow statistics:** fixed meshes only (no AMR, mesh motion or FSI): the averages are
  kept per node. Radial profiles are 3D only (lines in 2D). A wall bin is an exact area
  average only when its edges lie on element edges. Sample points outside the discrete
  mesh are NaN. No inflow profile for pipes yet (the parabolic profile is a rectangular
  duct's in 3D). Cost, measured on one A100 (Taylor-Green Q6, 6.0M velocity DOFs, one line and
  five profile stations): 4.2 ms per step (2.3 % of the 0.185 s step), 0.07 s of setup, 0.15 s
  per write of the sample sets, and 148 MiB of device memory, 9 numbers per velocity node and 1
  per pressure node ([status.md](status.md)).
- **Restarts:** same MPI rank count.
- **GPUs:** NVIDIA/CUDA only, built for one architecture at a time (currently sm_90,
  H100).
  - In bessemer, an A100 needs only `cuda_arch=80`. Nothing uses H100-only features, and
    the custom kernels' shared memory (at most about 93 KB per block) fits the A100's 163
    KB.
  - The dependency stack (MFEM, hypre, libCEED, UCX) must be rebuilt for sm_80, and the
    MPI transport settings rechecked on the new network.
  - The memory and speed figures were measured on 80 GB H100s.
- **Meshes from files:** Gmsh is not in the toolchain; the sphere benchmark still needs
  a hex mesh.
