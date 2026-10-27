# bessemer

A high-order finite element solver for the **incompressible Navier–Stokes equations** (and
unsteady Stokes), with linear and hyperelastic solids, moving meshes and fluid-structure
interaction, built on [MFEM](https://mfem.org) and running on CPUs and GPUs.

**bessemer is used through a coding agent.** You describe the flow you want; an agent
(Claude Code, Codex, Cursor or similar) running in this checkout sets up the machine,
writes the case, runs it and reports the results. The repository is written for that
agent: its entry point is [AGENTS.md](AGENTS.md), and the docs, scripts, deck format and
run summaries are built so an agent can work without you knowing the codebase, the input
format or MFEM.

## Getting started

```sh
git clone https://github.com/mschneier91/bessemer.git
cd bessemer
claude                 # or whichever coding agent you use
```

Then ask it to set things up. It shows you a plan first: what gets installed where. The
first setup builds the whole toolchain from Spack, compiler included, which on a workstation takes a few
hours and about 20 GB; the agent starts only when you agree. Later sessions start in
seconds.

## What to ask

| You say | The agent |
|---|---|
| "Set this up on my machine and check that it works." | Builds the Spack environment and bessemer, then runs a smoke case |
| "Simulate flow past a cylinder in a channel at Re 100. What's the Strouhal number?" | Writes a deck from the closest example, tells you the expected run time, runs it, and reads the shedding statistics from the run summary |
| "Run a lid-driven cavity at Re 1000 to steady state; I want to look at the flow." | Writes the deck with ParaView output, runs it, and tells you which file to open |
| "Can bessemer do a backward-facing step? A heated cavity?" | Answers from [docs/capabilities.md](docs/capabilities.md) and the deck reference, and says plainly when something isn't supported yet |
| "Where did development leave off?" | Reads [docs/status.md](docs/status.md) and summarizes the open items |

**What the agent asks you first:** using more than 4 MPI ranks, a 3D mesh larger than 32³,
or any GPU; submitting a batch job (it tells you the cost, then submits once you agree); changing the
Spack environment; merging to `main`. These are rules in [AGENTS.md](AGENTS.md).

**What you get back:** every run writes a `summary.json` with its status, step statistics,
forces, shedding statistics and errors against reference values when the case has them.
The agent reports from that file, not from the log. Runs can also write ParaView output
and per-step history.

## What it solves

The full list, with the validated benchmark results and the limits:
[docs/capabilities.md](docs/capabilities.md). In short:

**Incompressible flow** (`equation: navier_stokes` or `stokes`)

- **Monolithic:** velocity and pressure are solved together as one saddle-point system per
  step, with no projection or splitting error.
- **High order on quads and hexes:** continuous Taylor–Hood spaces with the orders as
  runtime options, partial assembly throughout, grad-div stabilization (on or off, scaled
  with h, ν or both).
- **Time stepping:** BDF with convection either sub-stepped along characteristics (OIFS,
  Nek5000's approach, at CFL numbers above 1) or explicit and extrapolated (IMEX); the
  convective or the semi-implicit rotational form; CFL-controlled, fixed or
  error-controlled steps; a failed solve cuts the step and retries.
- **Linear solvers:** two outer methods, FGMRES on the coupled system or flexible CG on the
  pressure Schur complement (Uzawa, [docs/uzawa.md](docs/uzawa.md)); a Cahouet–Chabard or
  pressure-mass Schur preconditioner; several velocity-block preconditioners (Jacobi-PCG,
  point-block Jacobi, Chebyshev, low-order-refined AMG); projected initial guesses across
  steps.
- **Large-eddy simulation:** the sigma and WALE subgrid models as an explicit eddy-viscosity
  stress ([docs/les.md](docs/les.md)); off means DNS.
- **Geometry and boundaries:** boxes (periodic, stretched), the DFG cylinder channel in 2D
  and 3D, a square cylinder in a large domain, and Gmsh or MFEM mesh files with named
  boundaries; inflow profiles, no-slip, and a directional do-nothing outflow that stays
  stable under backflow.
- **Adaptive mesh refinement** (isotropic or anisotropic refinement, hanging nodes, and
  optional coarsening with an L2-projected multistep history), lift and drag, shedding
  statistics, checkpoint and restart.
- **Flow statistics:** running mean velocity, pressure and Reynolds stresses, sampled on
  lines, azimuthally averaged radial profiles and wall bins (wall pressure and shear
  stress), with a check that the averages have converged; channel plane averages.

**Solids** (`equation: linear_elasticity`): small-strain linear elasticity, static or
dynamic (generalized-alpha or BDF), plane strain or plane stress
([docs/elasticity.md](docs/elasticity.md)). `equation: hyperelasticity`: large
deformation (St. Venant-Kirchhoff or neo-Hookean), static or dynamic, Newton with load
steps; Turek-Hron CSM1-3 decks ([docs/hyperelasticity.md](docs/hyperelasticity.md)).

**Moving meshes and fluid-structure interaction:** a pseudo-solid mesh deformation with
Jacobian stiffening (`equation: mesh_motion`, [docs/mesh_motion.md](docs/mesh_motion.md)),
the ALE fluid on a mesh moved by prescribed boundary motion
([docs/ale_fluid.md](docs/ale_fluid.md)), and the flow coupled with the linear or the
hyperelastic solid by partitioned subiterations in every step (section `fsi`,
[docs/fsi.md](docs/fsi.md)): Dirichlet-Neumann, Robin-Neumann, Neumann-Dirichlet or
Robin-Robin, with Aitken or IQN-ILS acceleration; Turek-Hron FSI1-3 decks.

**Running it:** MPI-parallel on CPUs and NVIDIA GPUs (CUDA). Cases are YAML decks, Python
scripts (module `incns`: `run_deck`, `Parameters.from_dict`, or step a `Case` yourself) or
small C++ drivers.

The defaults (orders, time scheme, CFL target, solver settings) are measured choices that
change as the studies move; the generated [deck reference](docs/deck_reference.md) lists
every key with its current default, and the method docs in [docs/](docs/) say why.

## Without an agent

Everything the agent does goes through scripts and decks, so you can do it by hand:

```sh
scripts/doctor.sh --run                             # check the setup, build, smoke-run
                                                    # (on a cluster: on a compute node)
. scripts/env.sh                                    # the Spack toolchain in this shell
mpirun -np 4 build/cpu/apps/run_case cases/dfg_2d1.yaml
build/cpu/apps/run_case --deck-reference            # every deck key
```

| To | Read |
|---|---|
| Run cases: decks, Python, C++ drivers, the example decks, outputs | [docs/using.md](docs/using.md) |
| Every deck key and its default | [docs/deck_reference.md](docs/deck_reference.md) |
| Run the standard benchmarks (Taylor–Green Re 1600, turbulent channel, DFG 3D) | [benchmarks/](benchmarks/README.md) |
| Install: the Spack setup (`scripts/setup.sh`), machines with their own page | [docs/install/spack.md](docs/install/spack.md), [environments/](environments/README.md) |
| Change the solver: rules, design decisions, tests | [docs/developing.md](docs/developing.md) |

## Testing

Tests are MPI gtests run at 1, 2 and 4 ranks, in tiers: a **fast** tier (every module, the
order and exactness oracles), a **verify** tier (convergence rates against exact solutions,
run through the case-level defaults so it follows them as they change), a **slow** tier,
and the **Python** tests. GitHub Actions runs the style check, a warnings-as-errors build,
Doxygen, the fast tier and the Python tests on every push and pull request, and the verify
tier every third day and on request ([docs/developing.md](docs/developing.md) §7).

## Status

The flow solver, the solid solver, mesh motion and LES all run on CPUs and on GPUs (NVIDIA
H100); the benchmarks have been run at the defaults with results in each benchmark's
README. Where development stands, what was decided and what is open:
[docs/status.md](docs/status.md).

## License

BSD 3-Clause, the same as MFEM: see [LICENSE](LICENSE). The license covers this source
code. A compiled bessemer links third-party libraries under their own licenses (among
them MUMPS, CeCILL-C, and SuiteSparse's UMFPACK, GPL), which apply if you distribute
binaries.
