# bessemer

A high-order finite element solver for the **incompressible Navier–Stokes equations** (and
unsteady Stokes), built on [MFEM](https://mfem.org).

- **Monolithic:** velocity and pressure are solved together as one saddle-point system per
  step, with no projection or splitting error. FGMRES with a Cahouet–Chabard block
  preconditioner.
- **High order on quads and hexes:** Taylor–Hood Q3/Q2 by default (orders are runtime
  options), partial assembly throughout.
- **Time stepping:** BDF2 with explicit convection (IMEX), or OIFS sub-stepped convection
  with BDF3 for steps at CFL 2. CFL-controlled, fixed or error-controlled steps.
- **Adaptive mesh refinement** (refinement only, anisotropic), lift/drag, a directional
  do-nothing outflow, checkpoint/restart.
- **A library, not a monolith:** a case is a YAML deck, a Python script or a small C++
  driver. MPI-parallel; the GPU path is built in but not yet validated.

**Using a coding agent?** Point it at [AGENTS.md](AGENTS.md).

## Validated against the literature

| Benchmark | Result |
|---|---|
| Square cylinder, Re 200 (Joly, Etienne & Pelletier 2012) | C_D 1.443 (paper 1.44), C_L,rms 0.403 (0.42), St 0.157 (0.151); all within 5% |
| Same, OIFS at CFL 2 | within 1% of IMEX in 4.2× fewer steps |
| DFG 2D-1, Schäfer–Turek (Re 20) | relative errors C_D 1.1e-4, C_L 2.8e-3, Δp 4.1e-3 |
| DFG 2D-3, John 2004 (unsteady) | C_D,max error 1.9e-5 with AMR |
| Directional do-nothing, Braack & Mucha 2014 (Table 5.1) | both conditions within 0.1% |
| Manufactured solutions, Taylor–Green vortex | design orders in space and time; Stokes MMS exact to solver tolerance |

Each is a deck in `cases/` (or a test); see [docs/using.md](docs/using.md) §3.

## Quickstart

The whole toolchain, compiler included, comes from Spack (`environments/<machine>/`;
setup in [docs/install/](docs/install/)). Then:

```sh
scripts/doctor.sh                        # check the setup, build, smoke-run (seconds)
scripts/test.sh cpu -L fast              # the fast test tier at 1, 2 and 4 MPI ranks
. scripts/env.sh
mpirun -np 4 build/cpu/apps/run_case cases/dfg_2d3.yaml   # -> dfg_2d3/dfg_2d3_summary.json
```

Python (no compiling once the module is built):

```sh
scripts/build.sh cpu-python
. scripts/env.sh && export PYTHONPATH=build/cpu-python/python
mpirun -np 4 python examples/python/start_here.py
```

## Documentation

| | |
|---|---|
| [docs/using.md](docs/using.md) | Running cases: decks, Python, the benchmark drivers, outputs |
| [docs/deck_reference.md](docs/deck_reference.md) | Every deck key with its default (generated from the loader) |
| [docs/install/](docs/install/) | Building the Spack environment (desktop, PSC Bridges-2) |
| [docs/developing.md](docs/developing.md) | Rules, design and testing for changing the solver |
| [docs/status.md](docs/status.md) | Where the project stands, open items, decision log |
| `docs/*.md`, [docs/design/](docs/design/) | Method write-ups and design specs |

## Status

The CPU path is the validated one. GPU kernels exist but haven't been confirmed on a GPU.
Production DNS/LES runs target DOE systems. Current work and open items:
[docs/status.md](docs/status.md).

## License

BSD 3-Clause, the same as MFEM: see [LICENSE](LICENSE). The license covers this source
code. A compiled bessemer links third-party libraries under their own licenses (among
them MUMPS, CeCILL-C, and SuiteSparse's UMFPACK, GPL), which apply if you distribute
binaries.
