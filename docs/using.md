# Using bessemer

How to run the solver: build it, drive a case (deck, Python or C++), start from the example
decks, and read the results. No solver code changes are ever needed to run
a case; if one seems necessary, that is a missing library feature (see
[developing.md](developing.md)).

## 1. Check the setup, build once

The toolchain comes entirely from Spack (see [install/](install/)). Start with the doctor:
it checks the environment step by step, builds if needed, runs a 2-second smoke case, and
says how to fix whatever fails.

```sh
scripts/doctor.sh               # add --build to force a rebuild
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
| `physics` | `nu`, `grad_div`, `convective_form: convective \| rotational`, `outflow: directional \| classical` |
| `discretization` | `order_u`, `order_p`, `mass: auto \| collocated \| consistent` |
| `mesh` | `dim`, `elements`, `lengths`, `periodic`, `stretch`, `geometry`, `file`, `boundary_names`, `square_cylinder.*`, `cylinder_channel.*` |
| `time` | `dt`, `t_final`, `order`, `step_control: cfl \| fixed \| error`, `cfl_target`, `cfl_max`, `dt_max`, `ext_order`, `convection: imex \| oifs`, `oifs_cfl`, `atol`, `rtol` |
| `solver` | `rtol`, `atol`, `max_iter`, `kdim`, `schur`, `a_pc`, `preconditioner`, `block_shape`, `n_inner`, `lp_vcycles`, `rotation_pc`, `rotation_schur`, ... |
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

## 5. Outputs

- **Run summary (always):** `run_case` writes `<output.path>/<output.name>_summary.json`
  (or `--summary <file>`): `status` (`running` while it runs, then `ok` or `diverged`,
  exit code 2), the case settings, time-step statistics, mesh size, diagnostics, forces,
  probes, and `reference` errors. Read this rather than the log. From Python:
  `case.write_summary(path, status)`.
- **Progress lines:** `output.progress: <time interval>` prints t, dt and its range, the
  CFL number and where it peaks, forces, iterations, cells, wall time and an ETA.
- **History:** `output.history: true` writes `<path>/<name>_history.csv`, one row per step.
- **Force statistics:** `forces.statistics: true` evaluates C_D and C_L every step; the
  summary reports final values, maxima (with times), and means, rms and the Strouhal
  number over the last `forces.average_periods` shedding periods.
- **ParaView:** `output.enabled: true` writes `<path>/<name>/` with high-order output; open
  the `.pvd` file.
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

## 6. Resources

- Stay at 4 MPI ranks or fewer on a workstation. Larger runs belong on a cluster, and batch
  jobs are submitted by a human (see [install/bridges2.md](install/bridges2.md)).
- After interrupting a run, reap stray MPI processes with `scripts/reap.sh`.
- GPU runs are not validated yet ([developing.md](developing.md) §7.5).

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
