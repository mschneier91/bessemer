# Using bessemer

How to run the solver: build it, drive a case (deck, Python or C++), reproduce the
validated benchmarks, and read the results. No solver code changes are ever needed to run
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
scripts/test.sh cpu -L fast     # optional: the fast test tier (~3.5 min, 171 tests)
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

`run_case` builds the box (periodic and/or walled, optionally stretched), the square
cylinder in a large domain, or the DFG cylinder channel (`mesh.geometry`). Boundary groups
select box faces (`xmin` ... `zmax`) or the geometry's named boundaries (`inflow`,
`outflow`, `sides` / `walls`, `body` / `cylinder`); a constant or parabolic velocity
(optionally ramped or sine-modulated in time) is given right in the deck. Every real
boundary must be in exactly one group: a deck that leaves one out is refused with the
boundary named, rather than run with a silent do-nothing wall. Omitted
keys take library defaults, and a key the loader doesn't know is an error that names the
closest valid key. **Every key, with its default and allowed values:
[deck_reference.md](deck_reference.md)** (generated from the loader; `run_case
--deck-reference` prints it). The sections at a glance:

| Section | Keys |
|---|---|
| (top level) | `equation: stokes \| navier_stokes`, `initial_velocity: zero \| taylor_green_2d`, `device`, `restart` |
| `physics` | `nu`, `grad_div`, `convective_form: convective \| rotational`, `outflow: directional \| classical` |
| `discretization` | `order_u`, `order_p`, `collocated_mass` |
| `mesh` | `dim`, `elements`, `lengths`, `periodic`, `stretch` |
| `time` | `dt`, `t_final`, `order`, `step_control: cfl \| fixed \| error`, `cfl_target`, `cfl_max`, `dt_max`, `ext_order`, `convection: imex \| oifs`, `oifs_cfl`, `atol`, `rtol` |
| `solver` | `rtol`, `atol`, `max_iter`, `kdim`, `schur`, `a_pc`, `preconditioner`, `block_shape`, `n_inner`, `lp_vcycles`, `rotation_pc`, `rotation_schur`, ... |
| `boundary_conditions` | a list of `{select: [xmin, ymax, ...] or attributes, type: no_slip \| outflow \| velocity_dirichlet}` |
| `amr` | `enabled`, `interval`, `theta` or `tolerance`, `anisotropic`, `min_size`, `max_elements`, ... |
| `forces` | `enabled`, `attributes`, `reference_velocity`, `reference_area`, `interval` |
| `output` | `enabled`, `path`, `name`, `interval`, `diagnostics` |
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

## 3. Validated cases you can reproduce

Each is a deck in `cases/`: run it with `mpirun -np 4 build/cpu/apps/run_case
cases/<deck>.yaml`. The summary's `reference` section reports the errors against the
literature values. Wall times are for 4 MPI ranks on an 8-core desktop.

| Deck | Case | Time | Expected |
|---|---|---|---|
| `square_cylinder_re200_oifs.yaml` | square cylinder, Re 200 (Joly et al. 2012), OIFS at CFL 2 | ~13 min | C_D 1.43, C_L,rms 0.39, St 0.157 (paper 1.44, 0.42, 0.151) |
| `square_cylinder_re200.yaml` | the same with IMEX at CFL 0.5 | ~70 min | C_D 1.44, C_L,rms 0.40, St 0.157 |
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
- **Diagnostics CSV:** `output.diagnostics: true` writes kinetic energy, dissipation and
  ‖∇·u‖ per step.
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
  [install/desktop.md](install/desktop.md).
- **`implicit solve did not converge`:** usually the step is too large for IMEX. Lower
  `time.cfl_target`, or switch to OIFS.
- **`time.convection: oifs needs ... fixed or cfl step control`:** OIFS has no error
  control, by design. Use `step_control: cfl`.
