# Using bessemer

How to run the solver: build it, drive a case (deck, Python or C++), start from the example
decks, and read the results. No solver code changes are ever needed to run
a case; if one seems necessary, that is a missing library feature (see
[developing.md](developing.md)).

## 1. Check the setup, build once

The toolchain comes entirely from Spack (see [install/](install/)). Start with the doctor:
it checks the environment step by step and says how to fix whatever fails. With `--run` it
also builds if needed and runs a 2-second smoke case; do that where builds and MPI runs are
allowed (on a cluster, a compute node rather than a login node).

```sh
scripts/doctor.sh               # check only: builds and runs nothing
scripts/doctor.sh --run         # also build if needed and smoke-run; --build forces a rebuild
```

Then, as needed:

```sh
scripts/build.sh cpu            # the solver, tests and drivers  -> build/cpu/
scripts/build.sh cpu-python     # also the Python module         -> build/cpu-python/
scripts/test.sh cpu -L fast     # optional: the fast test tier (~4 min, 180 tests)
```

The scripts activate the environment themselves. To run `mpirun` by hand, activate it in
your shell first: `. scripts/env.sh`.

## 2. Three ways to drive a case

All three go through the same `incns::Case`, so they behave identically.

### A YAML deck (no compiling)

```sh
. scripts/env.sh
mpirun -np 4 build/cpu/apps/run_case cases/tgv2d_stokes.yaml
```

`run_case` builds the mesh named by `mesh.geometry`:
- `box`: periodic and/or walled, optionally stretched;
- `square_cylinder`: a square cylinder in a large domain;
- `cylinder_channel`: the DFG cylinder channel, 2D, or extruded in z with `mesh.dim: 3`
  (the DFG 3D benchmarks);
- `file`: any quad or hex mesh from a Gmsh or MFEM file (`mesh.file`).

Boundary groups select box faces (`xmin` ... `zmax`) or the geometry's named boundaries:
`inflow`, `outflow`, `sides` / `walls`, `body` / `cylinder`, and for a Gmsh file its
physical-group names (more with `mesh.boundary_names`). A constant or parabolic velocity
is given right in the deck, optionally ramped or sine-modulated in time; in 3D the
parabolic profile is a product of parabolas in y and z. Every real
boundary must be in exactly one group: a deck that leaves one out is refused with the
boundary named, rather than run with a silent do-nothing wall. Omitted
keys take library defaults, and a key the loader doesn't know is an error that names the
closest valid key. **Every key, with its default and allowed values:
[deck_reference.md](deck_reference.md)** (generated from the loader; `run_case
--deck-reference` prints it). The sections at a glance:

| Section | Keys |
|---|---|
| (top level) | `equation: stokes \| navier_stokes`, `initial_velocity: zero \| uniform \| channel \| taylor_green_2d \| taylor_green_3d`, `device`, `restart` |
| `initial` | `velocity`, `perturbation`, `perturbation_center` (uniform), `seed` (channel) |
| `forcing` | `body_force`: a constant force per unit mass, e.g. a channel's mean pressure gradient |
| `physics` | `nu`, `grad_div_h`, `grad_div_nu`, `convective_form: convective \| rotational`, `outflow: directional \| classical` |
| `discretization` | `order_u`, `order_p`, `mass: auto \| collocated \| consistent`, `vector_diffusion: compact \| mfem`, `fuse_grad_div` |
| `mesh` | `dim`, `elements`, `lengths`, `periodic`, `stretch`, `geometry`, `file`, `boundary_names`, `square_cylinder.*`, `cylinder_channel.*` |
| `time` | `dt`, `t_final`, `order`, `step_control: cfl \| fixed \| error`, `cfl_target`, `cfl_max`, `dt_max`, `ext_order`, `convection: imex \| oifs`, `oifs_cfl`, `atol`, `rtol`, `on_solver_failure: cut_step \| abort`, `retry_factor`, `retry_hold`, `retry_max`, `retry_dt_min` |
| `solver` | `rtol`, `atol`, `tolerance_reference: initial_residual \| rhs`, `initial_guess: previous \| extrapolate \| project`, `projection_k`, `max_iter`, `kdim`, `schur`, `schur_grad_div`, `a_pc`, `a_pcg_pc`, `preconditioner`, `block_shape`, `n_inner`, `lp_vcycles`, `lp_strong_threshold`, `lp_agg_levels`, `lp_print_level`, `rotation_pc`, `rotation_schur`, ... |
| `boundary_conditions` | a list of `{select: [xmin, ymax, ...] or attributes, type: no_slip \| outflow \| velocity_dirichlet}` |
| `amr` | `enabled`, `interval`, `theta` or `tolerance`, `anisotropic`, `min_size`, `max_elements`, ... |
| `forces` | `enabled`, `attributes` or `boundaries`, `reference_velocity`, `reference_area`, `interval`, `statistics` |
| `channel_statistics` | `enabled`, `start_time`, `interval`: plane and time averages for a channel (below) |
| `output` | `enabled`, `path`, `name`, `interval`, `diagnostics`, `progress`, `history` |
| `checkpoint` | `enabled`, `path`, `interval` |
| `nondimensionalization` | dimensionless (default) or dimensional reference scales |

Examples: `cases/tgv2d_stokes.yaml`, `cases/stokes_mms.yaml`,
`cases/channel_noslip.yaml`.

### Python (analytic fields, no compiling)

```sh
. scripts/env.sh
export PYTHONPATH=build/cpu-python/python
mpirun -np 4 python examples/python/start_here.py     # a fully commented first case
```

Python is a job driver: it sets parameters and analytic initial/boundary/forcing fields,
and all numerics stay in C++. Run it as `mpirun -np N python case.py`; there is no
`mpi4py`. Fields are plain callables `f(x, t)` or, for speed, `@incns.field(dim)`
(numba-compiled). More examples: `examples/python/stokes_ex/`.

### A C++ driver (complex geometry)

Geometries beyond boxes have drivers in `apps/`: `dfg_cylinder` (the DFG channel with a
cylinder) and `square_cylinder` (a square cylinder in a large domain). Each prints its
options with `--help`.

**Standard benchmarks** (Taylor–Green Re 1600, turbulent channel Re_τ 180/395, DFG 3D)
have their own decks, reference data and comparison scripts in
[../benchmarks/](../benchmarks/README.md).

## 3. Example decks

Starting points for new cases: copy the closest one. Each is a deck in `cases/`; run it
with `mpirun -np 4 build/cpu/apps/run_case cases/<deck>.yaml`. Those with literature
values report their errors in the summary's `reference` section. Wall times are for 4 MPI ranks on an 8-core desktop.

| Deck | Case | Time | Expected |
|---|---|---|---|
| `square_cylinder_re200_oifs.yaml` | square cylinder, Re 200 (Joly et al. 2012), OIFS at CFL 2 | ~13 min | C_D 1.43, C_L,rms 0.39, St 0.157 (paper 1.44, 0.42, 0.151) |
| `square_cylinder_re200.yaml` | the same with IMEX at CFL 0.5 | ~42 min | C_D 1.44, C_L,rms 0.40, St 0.157 |
| `dfg_2d1.yaml` | DFG 2D-1 (steady, Re 20; Schäfer–Turek) | ~1–2 min | relative errors C_D 1e-4, C_L 3e-3, Δp 4e-3 |
| `dfg_2d3.yaml` | DFG 2D-3 (unsteady; John 2004), fixed dt 0.001 | ~3 min | C_D,max within 0.1%, C_L,max within ~8% (base mesh), Δp(8) within 1% |
| `cavity.yaml` | lid-driven cavity, Re 100 (a fast smoke case) | seconds | `status: ok` |

The C++ drivers `apps/square_cylinder` and `apps/dfg_cylinder` run the same benchmarks
with command-line flags and extra study options (`--help`); they end with a `RESULT
key=value ...` line. (Note: `dfg_cylinder -c 3` defaults to `-dt 0.01`, beyond the
stability limit; pass `-dt 0.001`.)

## 4. Choosing the time stepping

- **IMEX** (default, `time.convection: imex`): explicit convection, BDF2. The step is
  limited to CFL ≈ 0.5–0.7. Accurate and the cheapest per step.
- **OIFS** (`time.convection: oifs`): convection sub-stepped inside each step, so CFL 2 is
  practical. BDF3 and the collocated mass are switched on automatically. On the square
  cylinder, CFL 2 matches IMEX to 1% in 4.2× fewer steps, about 3× less wall time. CFL 4
  loses accuracy on lift amplitudes. Details: [oifs_implementation.md](oifs_implementation.md).
- **Step control:** `cfl` (default; dt follows `cfl_target`), `fixed`, or `error` (IMEX
  only, BDF2/BDF3 error estimate).
- **Check that the kinetic energy never rises** in a periodic or enclosed flow without
  forcing. At Re 1600 IMEX was unstable at CFL 0.5 and the run carried on without an
  error ([../benchmarks/tgv_re1600/README.md](../benchmarks/tgv_re1600/README.md)).
- **When the solver fails to converge** (`time.on_solver_failure`):
  - `cut_step` (**default**): the step is rejected and solved again at half the step
    (`retry_factor` 0.5); the step may not grow for the retried step and 5 more
    (`retry_hold`), then grows back (CFL control: as usual; fixed steps: 1.2× per step
    to the original). At most `retry_max` (8) cuts per step, never below `retry_dt_min`;
    then the run stops. Each rejection is logged, and the summary counts
    `rejected_steps`. On a GPU the FGMRES restart length is also lowered below
    `solver.kdim` before each solve when its Krylov vectors (two system-sized vectors per
    outer iteration) would not fit in the free memory; the log reports it. The free
    memory is read after the solver's preconditioners have been applied once, since
    their first application allocates their workspace (9 GB at 64³ Q3). So
    `solver.kdim` bounds the Krylov memory, `solver.max_iter` the work per attempt, and
    a hard step costs time instead of the run. Near the memory limit the restart length
    can fall to a few vectors: Taylor–Green at 87³ Q3/Q2 ran on one H100 at restart
    length 1, about 20 % slower per step than proportional to 64³.
  - `abort`: the run stops on a failed solve, and `solver.kdim` and `solver.max_iter` are
    used exactly as given (the behaviour before 2026-10-13).
  Error-controlled steps (`step_control: error`) keep their own rejection logic.
- **The outer solve's initial guess and tolerance** (`solver.initial_guess`,
  `solver.tolerance_reference`; defaults unchanged). By default the tolerance is relative to
  the initial residual, so the warm start (the previous step's solution) saves no
  iterations. With `tolerance_reference: rhs` the solve stops at `rtol` times the
  right-hand side's norm, and a better guess stops it sooner: `extrapolate` (linear in time,
  one more system-sized vector) or `project` (least squares over the last `projection_k`
  solutions, 2 `projection_k` system-sized vectors, e.g. 2.9 GB at 64³ for 8). On a small
  Taylor–Green on CPU: 13.0 outer iterations per step by default, 8.9 with `rhs`, 7.4 with
  `extrapolate`, 5.2 with `project` (k 8), the energy unchanged to 4e-11
  ([throughput.md](throughput.md)).

## 5. Outputs

- **Run summary (always):** `run_case` writes `<output.path>/<output.name>_summary.json`
  (or `--summary <file>`): `status` (`running` while it runs, then `ok` or `diverged`,
  exit code 2), the case settings, time-step statistics, mesh size, diagnostics, forces,
  probes, and `reference` errors. Read this rather than the log. From Python:
  `case.write_summary(path, status)`.
- **Progress lines:** `output.progress: <time interval>` prints t, dt and its range, the
  CFL number and where it peaks, forces, iterations, cells, wall time and an ETA.
- **History:** `output.history: true` writes `<path>/<name>_history.csv`, one row per step.
- **Profile (always, unless `output.profile: off`):** `run_case` writes
  `<path>/<name>_profile.csv` at the end: every profiled region with its nesting path,
  call count and inclusive and exclusive wall seconds (max and min over the ranks). The
  summary's `profile` section has the top three levels. On a GPU use
  `output.profile: sync` for a profiling run: plain wall time charges an asynchronous
  kernel to whichever region next waits for the device; `sync` waits at every region
  boundary, which is slower but puts the time where it is spent. The file's second line
  says which mode produced it.
- **Force statistics:** `forces.statistics: true` evaluates C_D and C_L every step; the
  summary reports final values, maxima (with times), and means, rms and the Strouhal
  number over the last `forces.average_periods` shedding periods.
- **ParaView:** `output.enabled: true` writes `<path>/<name>/` with high-order output; open
  the `.pvd` file.
- **Grad-div is on by default.** Every run adds γ(∇·u, ∇·v) to the velocity block with
  γ = `physics.grad_div_h`·h_K + `physics.grad_div_nu`·ν (h_K the element size; defaults
  2 and 0, so γ = 2h). Set either part, or both; both 0 turns it off. The old keys
  `physics.grad_div` and `physics.grad_div_scale` are rejected with a message.
- **Diagnostics CSV:** `output.diagnostics: true` writes t, kinetic energy ½∫|u|²,
  dissipation ν∫|∇u|² and ‖∇·u‖ to `<path>/<name>_diagnostics.csv` every
  `output.interval` steps. It works without field output, and a restart appends to it.
- **Channel statistics:** `channel_statistics.enabled` (walls normal to y; a box mesh
  without refinement).
  - **What:** x–z plane averages at every velocity-node height, time-averaged from
    `start_time` on: U, V, W and the Reynolds stresses u'u', v'v', w'w', u'v'. Also the
    wall shear stress and from it u_τ and Re_τ, plus the bulk velocity.
  - **Where:** the summary's `channel` section, and `<path>/<name>_profiles.csv` (raw and
    wall units), written at the end of the run and at every checkpoint.
  - **Restarts:** the averages travel with the checkpoint, so a restarted run continues
    them exactly.
- **Forces:** `forces.enabled` with the body's attributes gives lift and drag coefficients
  (John's volume-integral method) in `<path>/<name>_forces.csv`.
- **Checkpoints:** `checkpoint.enabled`; restart with `restart: <dir>` at the same rank
  count.
- **Where an AMR run refined:** `apps/mesh_census -d <deck> -c <checkpoint>` rebuilds the
  mesh from a checkpoint pass by pass and counts elements per streamwise band (for
  checking an `amr.end_time` freeze; benchmarks/square_cylinder_re1000/README.md).

## 6. Resources

- Stay at 4 MPI ranks or fewer on a workstation. Larger runs belong on a cluster, and batch
  jobs are submitted by the agent once you approve the cost (AGENTS.md rule 3; a cluster with
  its own page is listed in [environments/README.md](../environments/README.md)).
- After interrupting a run, reap stray MPI processes with `scripts/reap.sh`.
- GPU runs are not validated yet ([developing.md](developing.md) §7.5).
- **GPU memory:** one 80 GB H100 holds up to 80³ Q3/Q2 elements (41.5 M velocity dofs)
  with the Taylor–Green defaults, about 1.95 kB per velocity dof; leave headroom for
  harder cases (each outer iteration adds two system-sized vectors). What was done to
  get there and what is left: [gpu_memory.md](gpu_memory.md). `output.profile: sync` on a
  GPU adds device memory per region to the profile CSV.

## 7. When something goes wrong

- **Run `scripts/doctor.sh` first.** It finds most setup problems.
- **`parameters: unknown key(s) ... (did you mean ...?)`:** a typo in the deck; every valid
  key is in [deck_reference.md](deck_reference.md).
- **`the deck declares no boundary_conditions` / `has no boundary condition`:** every real
  boundary needs exactly one group; the message names the boundary.
- **`status: diverged` in the summary:** the forces or the velocity blew up; the
  `divergence` field says when. Lower `time.cfl_target` (IMEX) or use OIFS.
- **`env.sh: ... System-toolchain leak -- aborting`:** the Spack environment isn't
  installed, or it was just edited and spack re-resolved it. See
  [install/spack.md](install/spack.md) ("When `env.sh` fails").
- **`implicit solve did not converge`:** usually the step is too large for IMEX. Lower
  `time.cfl_target`, or switch to OIFS.
- **`time.convection: oifs needs ... fixed or cfl step control`:** OIFS has no error
  control, by design. Use `step_control: cfl`.
