# The assistant server (MCP)

`mcp_server/bessemer_mcp.py` is a [Model Context Protocol](https://modelcontextprotocol.io)
server. It gives an AI assistant tools to set up, vet, run and read bessemer cases on
the user's own machine. With it the assistant doesn't have to learn the repository
first. It works from Claude Code in this checkout, and also from any other MCP client
(Claude Desktop, Cursor, ...) pointed at the checkout.

It is one Python file using only the standard library (Python 3.8 or newer), so there is
nothing to install. It drives the user's own build of `run_case`, which needs a build
first (`scripts/setup.sh --plan`, then `scripts/build.sh cpu`; see [using.md](using.md)).
It runs on the user's machine only: no scheduler, no remote hosts, no credentials.

## Connecting it

**Claude Code in this checkout:** the repository's `.mcp.json` registers the server.
Claude Code asks once whether to enable it.

**Another client**, or Claude Code elsewhere: register the command
`python3 <checkout>/mcp_server/bessemer_mcp.py`. For Claude Code that is

```sh
claude mcp add bessemer -- python3 /path/to/bessemer/mcp_server/bessemer_mcp.py
```

and for Claude Desktop an entry in its `claude_desktop_config.json`:

```json
{"mcpServers": {"bessemer": {"command": "python3",
                             "args": ["/path/to/bessemer/mcp_server/bessemer_mcp.py"]}}}
```

Optional environment variables: `BESSEMER_RUN_CASE` (the binary, default
`build/<preset>/apps/run_case`), `BESSEMER_ENV` (`auto`, `spack` or `none`: run through
`scripts/env.sh`, plainly, or Spack when `env.sh` works, the default), `BESSEMER_MPIEXEC`
(the launcher without `env.sh`, default `mpirun`), `BESSEMER_RUNS` (where new cases and
the run registry go, default `runs/`).

## The tools

| Tool | What it does |
|---|---|
| `check_setup` | The build, cores, memory, toolchain. If nothing is built, it says how to set up |
| `list_examples` | The decks in `cases/` and `benchmarks/` with their header comments |
| `deck_keys` | Searches every deck key (type, default, values, description), read from this build's `run_case --deck-schema`, so it is never stale |
| `read_doc` | `using`, `capabilities`, `deck_reference`, `benchmarks`, `install`, this page, or a benchmark's README |
| `new_case` | Copies an example deck into `runs/<name>/<name>.yaml` (comments kept), applies dotted-key changes (`{"physics.nu": 1e-3}`), points the output at `runs/<name>/`, then checks it |
| `write_deck` | Replaces a deck under `runs/` with new text, then checks it |
| `check_deck` | `run_case <deck> --check`: loads the deck, builds the mesh, checks that every boundary is in exactly one group, counts the DOFs; no solver is built. Returns the loader's error message, or the size, the settings in effect, a memory estimate and the step count when `dt` is fixed |
| `run_deck` | Checks the deck, then runs it in the background with MPI (ranks default `min(4, cores)`, at most the core count); returns a run id |
| `run_status` | State, latest progress line (t, dt, CFL, iterations, wall time, the run's own ETA), errors, log tail; without an id, the recent runs |
| `stop_run` | Stops a run's whole process group |
| `read_results` | The run's `summary.json` (status, time, mesh, forces, reference errors, ...), the history CSV's ranges, the output files, and what one run doesn't establish |

The same documents are also served as MCP resources (`bessemer://docs/<name>`). The
server's instructions to the assistant follow [AGENTS.md](../AGENTS.md) ("Working with
the user"): start from the closest example, state the choices made, say the cost
before a long run, report from the summary, and say plainly when bessemer can't do
something.

**Estimates.** The memory estimate is 0.6 to 1.5 kB per velocity DOF, from the GPU
measurements in [gpu_memory.md](gpu_memory.md). AMR grows the mesh during a run. A deck
alone can't give an honest wall time for CFL-controlled steps, so the ETA comes from
the run itself: `run_status` reads it from the progress lines (`new_case` adds
`output.progress` when the deck has none).

**Runs** are recorded in `runs/.mcp/<id>/` (the command, the log and the exit code), so
`run_status` works across server restarts. Runs start in the repository root, as on
the command line, so relative mesh paths in decks resolve as usual.

## `run_case` options it uses

```sh
run_case <deck.yaml> --check   # vet the deck; one JSON object; exit 0 (ok) or 3 (error)
run_case --deck-schema         # every deck key as JSON, keyed by its path
```

Both are driver options only: a run is unchanged. `--check` turns MFEM errors into
exceptions so their message reaches the report. Run it on one rank, because an error
on one rank of several would leave the others waiting.

The test `mcp_server_test` (fast tier) drives the server over stdio: setup, keys,
examples, a new case, the errors a user sees for a misspelled key and an uncovered
boundary, a run at one rank, and that the run's DOFs match the check's.
