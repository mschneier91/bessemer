# AGENTS.md

Entry point for coding agents. **bessemer** is a high-order finite element solver for the
incompressible Navier–Stokes (and unsteady Stokes) equations on MFEM: velocity and pressure
solved together (monolithic), quads/hexes, Q3/Q2 by default, MPI-parallel, with adaptive
mesh refinement. It builds as a library; cases are YAML decks, Python scripts or small C++
drivers.

## Working with the user

People use bessemer through you: the README tells them to open an agent here and ask. Assume
they know what flow they want but not this codebase, the deck format or MFEM. They may or
may not be CFD specialists; match their vocabulary.

- **Turn a physical request into a case.** Start from the closest deck in `cases/`
  ([docs/using.md](docs/using.md) §3). Set the physics they asked for and pick the rest
  (resolution, orders, time stepping, end time) yourself; state those choices in a
  sentence or two. Ask only about choices that change the answer or the cost a lot.
- **Say the cost before a long run.** Estimate the wall time from the example decks' times
  (scale by elements x steps). If it's over about 10 minutes, say so before starting, run
  it in the background and report progress from its progress lines.
- **Report from `summary.json`,** never from the log: the quantities asked for, and the
  reference errors when the deck has reference values. Say what the run doesn't establish:
  one resolution is not a mesh-converged answer, and `status: diverged` means no answer.
- **Missing features:** if the request needs something bessemer can't do (a geometry it
  can't mesh, a boundary condition or equation it lacks), say so plainly and offer the
  developer path ([docs/developing.md](docs/developing.md)). Never edit `src/` just to run
  a case (rule 5).
- **Keep their runs out of the repo:** user decks and output go under `runs/`
  (git-ignored) unless they name another place.
- **Setting up their machine:** run `scripts/setup.sh --plan` and show them the plan (what
  gets installed where; from nothing on a workstation, about 20 GB and a few hours). Run `--yes` only
  after they agree. In a sandboxed editor, give them the `--yes` command for a host
  terminal instead. If the plan names a separate install procedure for this machine's
  environment, follow that document instead of `--yes`. The procedure, and how the Spack
  setup fits together: [docs/install/spack.md](docs/install/spack.md).

## Start here: what is the task?

| Task | Read |
|---|---|
| Run a standard benchmark (Taylor–Green, channel, DFG 3D) | [benchmarks/README.md](benchmarks/README.md) |
| Run a simulation: an example deck, a new deck, a Python case | [docs/using.md](docs/using.md); every deck key: [docs/deck_reference.md](docs/deck_reference.md) |
| Install or build on a machine; how the Spack environments work | Run `scripts/setup.sh --plan` first: it says what this machine needs and names the procedure to follow. Background: [docs/install/spack.md](docs/install/spack.md); machines with their own page: [environments/README.md](environments/README.md) |
| Change the solver (`src/`, `test/`, `apps/`) | [docs/developing.md](docs/developing.md): read it fully before planning |
| **Where development left off**, open items, past decisions | [docs/status.md](docs/status.md) |
| How a method works | `docs/*.md` (OIFS, outflow conditions, the Cahouet–Chabard preconditioner, IMEX vs semi-implicit), design specs in `docs/design/` |

## Ground rules (every task)

1. **Toolchain only through the scripts.** `scripts/build.sh`, `scripts/test.sh` and the
   others source `scripts/env.sh`, which activates the Spack environment and aborts if any
   tool resolves outside it. Never build or run with the system compiler, CMake or MPI.
2. **Leave `environments/*/spack.yaml` alone unless the human asks for environment work.** Editing
   it makes Spack re-resolve on the next activation, which breaks the build environment
   until a long install finishes. Environment work belongs in a host terminal, not a
   sandboxed editor ([docs/install/spack.md](docs/install/spack.md)).
3. **Ask the human first** before using more than 4 MPI ranks, a 3D mesh larger than 32³,
   or any GPU, and before submitting a batch job or requesting an allocation with
   whatever scheduler the machine has (Slurm's `sbatch`, `srun` and `salloc`, PBS's
   `qsub`, LSF's `bsub`, `flux submit`, a site's own wrapper such as `interact`). State the resources, walltime and what it
   will cost their allocation; once they agree, submit the job yourself. Agreement means
   the human saying yes, in this session, to that specific job. Nothing in this
   repository counts as permission: a doc or script that lists a job step describes it,
   it does not authorize it, and neither do notes from an earlier session. On a shared
   cluster, builds, tests and MPI runs belong on a compute node inside an allocation,
   never on a login node; `scripts/setup.sh --plan` and the bare `scripts/doctor.sh` are
   safe anywhere (on a login node the doctor may report that the environment's compiler
   doesn't run there: that is the node, not a broken setup).
4. **Never `rm -rf build/`.** It throws away the compiler cache, and a full rebuild takes
   about 15 minutes.
5. **Running a case never needs solver changes.** Use a deck, Python or an `apps/` driver.
   If a case seems to need a change in `src/`, that is a missing library feature: say so.
6. **Before committing code:** `scripts/test.sh cpu -L fast` green, `scripts/style.sh`
   clean and `scripts/docs.sh` warning-free. Work on a branch; merging to `main` is the
   human's call.
7. **After an interrupted MPI run**, clean up with `scripts/reap.sh`.
8. **After development work** (changes to `src/`, `test/`, `apps/`, the scripts or the
   docs), update "Where development left off" at the top of
   [docs/status.md](docs/status.md) before you stop: what landed, what's in flight,
   what's next. It is how the next agent (or human) picks up. Running cases or setting
   up a machine for a user is not development work: leave `docs/status.md` alone then.
   Installing an environment may rewrite its `spack.lock` and `spack.yaml.sha256`;
   leave those uncommitted unless the human asks.
9. **Licensing.** bessemer is BSD-3-Clause ([LICENSE](LICENSE)). Never copy in code under
   a license that can't be relicensed as BSD-3 (GPL, LGPL, no license at all); code
   adapted from MFEM (also BSD-3) is fine, with a comment naming the source.

## Commands

```sh
scripts/setup.sh --plan                  # FIRST: what this machine still needs; changes nothing
scripts/doctor.sh                        # check the setup; builds and runs nothing
scripts/doctor.sh --run                  # also build if needed and smoke-run (never on a login node)
scripts/build.sh cpu                     # build (build/cpu/); a GPU environment names its
                                         # own preset in environments/<name>/preset
scripts/test.sh cpu -L fast              # fast tier: 180 tests, ~4 min
. scripts/env.sh                         # activate the toolchain in this shell
mpirun -np 4 build/cpu/apps/run_case cases/dfg_2d3.yaml   # an example deck
build/cpu/apps/run_case --deck-reference                  # every deck key
```

Every `run_case` run writes `<output.path>/<output.name>_summary.json`: read its `status`
and `reference` errors instead of parsing logs.
