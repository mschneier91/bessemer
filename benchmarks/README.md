# Benchmarks

Standard CFD benchmarks with published reference data: a deck (or several, for a
resolution ladder), the reference data with its citation, and a script that compares a
run with it. They are separate from [`cases/`](../cases/), which holds small example
decks.

| Benchmark | Tests | Reference | Decks | Compute |
|---|---|---|---|---|
| [tgv_re1600/](tgv_re1600/) | transition to turbulence in a periodic box: E(t), dissipation rate | spectral DNS on 512³ (high-order workshop case C3.5) | 16³, 32³, 64³ elements | 16³ and 32³ fit the desktop limits; 64³ needs GPU/many ranks |
| [channel/](channel/) | wall-bounded turbulence: U⁺(y⁺), Reynolds stresses, Re_τ | Moser, Kim & Mansour 1999, Re_τ 178 and 392 | Re_τ 180, 395 | 180 fits the desktop limits (slow); 395 needs GPU/many ranks |
| [dfg_3d/](dfg_3d/) | laminar 3D flow past a cylinder: drag, lift | FeatFlow (Bayraktar, Mierka & Turek 2012) | 3D-1Z (steady, Re 20), 3D-3Z (unsteady) | desktop |
| [sphere_re300/](sphere_re300/) | 3D wake and forces with AMR | Johnson & Patel 1999 | needs a hex mesh file first | — |

**Status (2026-10-10):** the infrastructure is in place and unit-tested: the decks load,
the solver features they use are tested (`test/channel_statistics_test`,
`test/benchmark_setup_test`), and each plot script passes `--selftest` once the reference
data is fetched (below). **None of the
benchmarks has been run yet**; the human runs them. Record results in each benchmark's
README and in [docs/status.md](../docs/status.md).

## Running one

```sh
mpirun -np 4 build/cpu/apps/run_case benchmarks/<benchmark>/<deck>.yaml
python3 benchmarks/<benchmark>/plot.py <the run's output file>   # see the deck header
```

- Runs write into `<output.path>` relative to the working directory. Run from a scratch
  directory (or `runs/`) to keep the checkout clean.
- Long runs checkpoint. Restart with `restart: <checkpoint dir>` added to the deck, at the
  same rank count. Channel averages and the diagnostics log continue across restarts.
- Anything above 4 MPI ranks, above 32³ elements in 3D, or on a GPU needs the human's
  approval first (AGENTS.md rule 3). The decks say which ones do.
- The plot scripts print a comparison table with the standard library alone. They draw
  plots when matplotlib is available, which is in `environments/stack.yaml`; the desktop
  environment gains it at its next refresh.

## Reference data

**Not committed:** none of the sources states redistribution terms. Fetch it once per
checkout:

```sh
python3 benchmarks/fetch_references.py
```

This downloads every dataset from its original source and writes the CSVs in
`*/reference/` (git-ignored), each with its citation and source URL in its header. Until
then, the plot scripts print the fetch command and exit, and their fast-tier self-tests
report *Skipped*.
