# bessemer

A high-order finite element solver for the **incompressible Navier–Stokes equations** (and
unsteady Stokes), built on [MFEM](https://mfem.org).

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

Then ask it to set things up. The first setup builds the whole toolchain from Spack,
compiler included, which takes a few hours. Later sessions start in seconds.

## What to ask

| You say | The agent |
|---|---|
| "Set this up on my machine and check that it works." | Builds the Spack environment and bessemer, then runs a smoke case |
| "Simulate flow past a cylinder in a channel at Re 100. What's the Strouhal number?" | Writes a deck from the closest example, tells you the expected run time, runs it, and reads the shedding statistics from the run summary |
| "Run a lid-driven cavity at Re 1000 to steady state; I want to look at the flow." | Writes the deck with ParaView output, runs it, and tells you which file to open |
| "Can bessemer do a backward-facing step? A heated cavity?" | Answers from the docs and the deck reference, and says plainly when something isn't supported yet |
| "Where did development leave off?" | Reads [docs/status.md](docs/status.md) and summarizes the open items |

**What the agent asks you first:** using more than 4 MPI ranks, a 3D mesh larger than 32³,
or any GPU; submitting a batch job (it drafts the script, you submit it); changing the
Spack environment; merging to `main`. These are rules in [AGENTS.md](AGENTS.md).

**What you get back:** every run writes a `summary.json` with its status, step statistics,
forces, shedding statistics and errors against reference values when the case has them.
The agent reports from that file, not from the log. Runs can also write ParaView output
and per-step history.

## The solver

- **Monolithic:** velocity and pressure are solved together as one saddle-point system per
  step, with no projection or splitting error. FGMRES with a Cahouet–Chabard block
  preconditioner.
- **High order on quads and hexes:** Taylor–Hood Q3/Q2 by default (orders are runtime
  options), partial assembly throughout.
- **Time stepping:** BDF2 with explicit convection (IMEX), or OIFS sub-stepped convection
  with BDF3 for steps at CFL 2. CFL-controlled, fixed or error-controlled steps.
- **Adaptive mesh refinement** (refinement only, anisotropic), lift/drag, a directional
  do-nothing outflow, checkpoint/restart.
- **Cases** are YAML decks, Python scripts or small C++ drivers. MPI-parallel; the GPU
  path is built in but not yet validated on a GPU.

## Without an agent

Everything the agent does goes through scripts and decks, so you can do it by hand:

```sh
scripts/doctor.sh                                   # check the setup, build, smoke-run
. scripts/env.sh                                    # the Spack toolchain in this shell
mpirun -np 4 build/cpu/apps/run_case cases/dfg_2d1.yaml
```

[docs/using.md](docs/using.md) covers running cases, [docs/install/](docs/install/) the
setup, [docs/developing.md](docs/developing.md) changing the solver, and
[docs/deck_reference.md](docs/deck_reference.md) lists every deck key.

## Status

The CPU path is validated; the GPU path hasn't been run on a GPU yet. Where development
stands and what's open: [docs/status.md](docs/status.md).

## License

BSD 3-Clause, the same as MFEM: see [LICENSE](LICENSE). The license covers this source
code. A compiled bessemer links third-party libraries under their own licenses (among
them MUMPS, CeCILL-C, and SuiteSparse's UMFPACK, GPL), which apply if you distribute
binaries.
