# Using bessemer

How to run the solver: build it, drive a case (deck, Python or C++), reproduce the
validated benchmarks, and read the results. No solver code changes are ever needed to run
a case; if one seems necessary, that is a missing library feature (see
[developing.md](developing.md)).

## 1. Build once

The toolchain comes entirely from Spack (see [install/](install/)). With the machine's
environment installed:

```sh
scripts/build.sh cpu            # the solver, tests and drivers  -> build/cpu/
scripts/build.sh cpu-python     # also the Python module         -> build/cpu-python/
scripts/test.sh cpu -L fast     # optional: the fast test tier (~3.5 min, 159 tests)
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

`run_case` handles box geometries (periodic and/or walled, optionally stretched). Omitted
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

Wall times are for 4 MPI ranks on an 8-core desktop.

| Case | Command | Time | Expected |
|---|---|---|---|
| Square cylinder, Re 200 (Joly et al. 2012), IMEX | `mpirun -np 4 build/cpu/apps/square_cylinder -re 200 -tf 160 -at 1 -amr-end 90 -tol 0.25 -minh 0.2 -maxe 3000 -pi 5 -out sq200` | ~70 min | C_D 1.44, C_L,rms 0.40, St 0.157 (paper: 1.44, 0.42, 0.151) |
| Same, OIFS at CFL 2 | the same plus `-oifs -cflt 2` | ~15–20 min | within 1% of IMEX with the same mass |
| DFG 2D-1 (steady, Re 20) | `scripts/test.sh cpu -R dfg_cylinder_slow_np4` (a slow-tier test) | ~9 min | relative errors C_D 1.1e-4, C_L 2.8e-3, Δp 4.1e-3 vs Schäfer–Turek |
| DFG 2D-3 (unsteady) | `mpirun -np 4 build/cpu/apps/dfg_cylinder -c 3 -dt 0.0015` (fixed steps: the default `-dt 0.01` is beyond the stability limit) | ~2–3 min | C_D,max within 0.2%, C_L,max within ~7% on the base mesh, vs John 2004 |

Both drivers end with one machine-readable `RESULT key=value ...` line. `square_cylinder`
also prints progress every `-pi` time units (time, dt and its range, CFL, C_D, C_L, cells,
ETA) and writes `<out>/history.csv` per step.

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

- **`env.sh: ... System-toolchain leak -- aborting`:** the Spack environment isn't
  installed, or it was just edited and spack re-resolved it. See
  [install/desktop.md](install/desktop.md).
- **`implicit solve did not converge`:** usually the step is too large for IMEX. Lower
  `time.cfl_target`, or switch to OIFS.
- **`time.convection: oifs needs ... fixed or cfl step control`:** OIFS has no error
  control, by design. Use `step_control: cfl`.
