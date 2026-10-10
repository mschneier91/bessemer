# AGENTS.md

Entry point for coding agents. **bessemer** is a high-order finite element solver for the
incompressible Navier–Stokes (and unsteady Stokes) equations on MFEM: velocity and pressure
solved together (monolithic), quads/hexes, Q3/Q2 by default, MPI-parallel, with adaptive
mesh refinement. It builds as a library; cases are YAML decks, Python scripts or small C++
drivers.

## Start here: what is the task?

| Task | Read |
|---|---|
| Run a simulation: a validated benchmark, a new deck, a Python case | [docs/using.md](docs/using.md); every deck key: [docs/deck_reference.md](docs/deck_reference.md) |
| Install or build on a machine | [docs/install/desktop.md](docs/install/desktop.md), [docs/install/bridges2.md](docs/install/bridges2.md) (PSC GPU cluster) |
| Change the solver (`src/`, `test/`, `apps/`) | [docs/developing.md](docs/developing.md): read it fully before planning |
| **Where development left off**, open items, past decisions | [docs/status.md](docs/status.md) |
| How a method works | `docs/*.md` (OIFS, outflow conditions, the Cahouet–Chabard preconditioner, IMEX vs semi-implicit), design specs in `docs/design/` |

## Ground rules (every task)

1. **Toolchain only through the scripts.** `scripts/build.sh`, `scripts/test.sh` and the
   others source `scripts/env.sh`, which activates the Spack environment and aborts if any
   tool resolves outside it. Never build or run with the system compiler, CMake or MPI.
2. **Leave `environments/*/spack.yaml` alone unless the task is the environment.** Editing
   it makes Spack re-resolve on the next activation, which breaks the build environment
   until a long install finishes. Environment work belongs in a host terminal, not a
   sandboxed editor ([docs/install/desktop.md](docs/install/desktop.md)).
3. **Ask the human first** before using more than 4 MPI ranks, a 3D mesh larger than 32³,
   or any GPU, and before submitting anything to a batch scheduler (draft the job script
   and let the human submit it).
4. **Never `rm -rf build/`.** It throws away the compiler cache, and a full rebuild takes
   about 15 minutes.
5. **Running a case never needs solver changes.** Use a deck, Python or an `apps/` driver.
   If a case seems to need a change in `src/`, that is a missing library feature: say so.
6. **Before committing code:** `scripts/test.sh cpu -L fast` green, `scripts/style.sh`
   clean and `scripts/docs.sh` warning-free. Work on a branch; merging to `main` is the
   human's call.
7. **After an interrupted MPI run**, clean up with `scripts/reap.sh`.
8. **Before you stop**, update "Where development left off" at the top of
   [docs/status.md](docs/status.md): what landed, what's in flight, what's next. It is how
   the next agent (or human) picks up.
9. **Licensing.** bessemer is BSD-3-Clause ([LICENSE](LICENSE)). Never copy in code under
   a license that can't be relicensed as BSD-3 (GPL, LGPL, no license at all); code
   adapted from MFEM (also BSD-3) is fine, with a comment naming the source.

## Commands

```sh
scripts/doctor.sh                        # FIRST: check the setup, build if needed, smoke-run
scripts/build.sh cpu                     # build (build/cpu/)
scripts/test.sh cpu -L fast              # fast tier: 171 tests, ~3.5 min
. scripts/env.sh                         # activate the toolchain in this shell
mpirun -np 4 build/cpu/apps/run_case cases/dfg_2d3.yaml   # a validated benchmark deck
build/cpu/apps/run_case --deck-reference                  # every deck key
```

Every `run_case` run writes `<output.path>/<output.name>_summary.json`: read its `status`
and `reference` errors instead of parsing logs.
