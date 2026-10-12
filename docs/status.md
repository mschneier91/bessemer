# Project status

The living handoff log: where the project stands, what was decided (dated, with who
decided), and what is open. The maintainer and their agents update it as work lands.
Rules and design live in [developing.md](developing.md); this file is the *state*.

## Where development left off (2026-10-10)

Keep this section short and current: whoever stops working updates it (AGENTS.md rule 8).
The dated log below keeps the history.

**NOT YET TESTED on the changes below (human 2026-10-10: merge and push anyway, test
after). Do these before relying on them:**
- **The CPU fast tier on the desktop** (`scripts/test.sh cpu -L fast`, `cpu` preset). The
  changes were only run on Bridges-2, where the one build is the CUDA one. Watch the AMR
  tests: the marker's element cap now treats equal indicators as a group (below), which
  can change which elements are refined near `amr.max_elements`.
- **`scripts/docs.sh`**: Bridges-2 has no Doxygen.
- **The Python tests and the debug-device sweep**: queued on Bridges-2 when this was
  written, not yet seen to pass.
- **The GPU fast tier at np 4 and the slow tier.** With two H100s (two ranks per card) the
  np 4 pass got through 16 of 60 tests in 28 minutes (15 passed, `nse_mms_test` timed out)
  and was stopped; the slow tier never ran. They need one GPU per rank: 4 H100s.
- **The CPU backend of the CUDA build is not a CPU reference.** `INCNS_DEVICE=cpu` in
  `build/cuda` passes 175 of 183; the 8 failures are inside hypre ("Error during solve",
  `amr_checkpoint_test`, `amr_event_test`, `unsteady_mms_test` at np 1 and 2; they pass at
  np 4) and MFEM's cuSPARSE path (`point_block_jacobi_test` np 2 and 4). All eight pass on
  the CUDA backend; the hypre ones match the July note about a CUDA-built hypre on the
  host. Not confirmed against a CPU build: that needs the desktop, or `psc_cpu`.

**In flight (branch `macos-gcc`, 2026-10-10, first cold-start run of `setup.sh` on a
fresh machine, an Apple Silicon MacBook Air, macOS 15.6.1): not merged.**
- **`setup.sh --yes` failed building `gcc@14.3.0`** at configure: `Configuration
  aarch64-apple-darwin24.6.0 not supported`. Spack's gcc recipe carries Iain Sandoe's
  aarch64-darwin patches only up to 14.2.0. Fix on the branch: `setup.sh` uses
  `gcc@14.2.0` on Apple Silicon (`docs/install/spack.md`, "The compiler"). The rest of the
  setup on macOS is **not yet verified**: resolve, install, build and the fast tier are
  still ahead on this machine; update this entry with what they find.
- **`gcc@14.2.0` then failed in libsanitizer** (~50 min in; configure and everything
  before it passed): linking `libasan.8.dylib` crashes Apple's linker, `ld: Assertion
  failed: (entry.toAddend < 0x400000), function encodeSectionReloc` (CLT 16.3.0,
  ld-1167.4.1, macOS 15.6.1, arm64). Spack's gcc has a `libsanitizer` variant (default on).
  Fix on the branch (human 2026-10-10): `gcc@14.2.0~libsanitizer` on macOS. Cost: no GCC
  `libasan`. That matters less than first thought: bessemer is compiled by Apple clang
  (below), whose own sanitizer runtime `scripts/asan.sh` should use; not yet run there.
  `-ld_classic` still exists in this ld but is deprecated.
- **`spack concretize` then failed** (human 2026-10-10: "we just need to use system
  clang"): `python: CPython does not compile with GCC on macOS yet, use clang` (a Spack
  conflict `platform=darwin %gcc`; it reaches Python through numba/llvmlite, matplotlib
  and `ninja`). Fix on the branch: `c`/`cxx` require `apple-clang@<detected>` (external,
  prefix `/usr`), `fortran` stays the Spack-built `gcc@14.2.0~libsanitizer`; `setup.sh`
  writes both on Darwin and `env.sh` whitelists the `apple-clang` external. Concretizes
  in ~20 s with that (hypre resolved 3.1.0 there, `desktop` has 3.2.0). **Not yet
  verified:** the install, the build, and `env.sh`'s assertion with `gcc`/`g++` resolving
  to `/usr/bin` (its banner also prints clang's version as `gcc=`).
- **`spack install` hung at 33/69 on `python@3.12.13`** (human's Terminal panel in the
  Claude desktop app; stopped, 0% CPU, for an hour; the progress line kept counting).
  Observed: the build worker, `make` and `python.exe check_extension_modules.py` were in
  `ps` state `T` (job-control stop), in a process group that was not the terminal's
  foreground group. **Cause not established**: the theory is a terminal-mode call
  (`readline` import) from a background group; the signal was not captured, and the gcc
  build ran fine from the same kind of tab. Rerunning with stdin detached
  (`spack -e ENV install < /dev/null`) was started; whether that is the fix is not
  confirmed until that Python build passes the extension check. Not tried in Terminal.app.
- **`setup.sh`'s install check was wrong after an interrupted install.** It reported
  step 6 "done" when the environment's view contained MFEM; Spack refreshes the view on
  Ctrl-C with whatever is built so far (MFEM was, python/numpy/ninja were not), so a
  rerun went straight to the build, which failed with "unable to find ... Ninja".
  **Fixed (2026-10-11):** it asks Spack (`spack -e ENV find -L`: every root spec marked
  `[+]`, none ` -  `), only when the lock is current (so `--plan` still changes nothing).
  Checked: this environment reads "installed (all 12 root specs)"; the marker format of an
  uninstalled root was confirmed on a scratch environment; a half-installed real
  environment was not reproduced (the parsing was tested on canned output).
- **Re 200 OIFS deck reproduced on the Mac (2026-10-11, M3 MacBook Air, clang, np 4):**
  `cases/square_cylinder_re200_oifs.yaml`, status ok, 2,191 steps, 3,000 elements (the
  deck's cap, reached at t ~ 50; 47 AMR events), 53,526 velocity + 11,842 pressure dofs,
  wall 1,936 s (32.3 min; the deck says ~13 min on the desktop, so ~2.5x here: a
  fanless Air with four performance cores). Periodic statistics, 10 periods,
  period spread 0.04%:

  | | this run | desktop (GCC, validated 2026-10-09) | paper | vs paper |
  |---|---|---|---|---|
  | C_D mean | 1.4265 | 1.4265 | 1.44 | -0.9% |
  | C_L,rms | 0.3924 | 0.3923 | 0.42 | -6.6% |
  | St | 0.1565 | 0.1565 | 0.151 | +3.6% |

  So the clang build agrees with the GCC desktop run to the digits quoted (same step
  count), and the C_L,rms gap to the paper is the desktop's too, not a platform effect.
  One run on one AMR level: not a mesh-converged answer. This is step 0 of the Re
  1000 -> 10000 ramp below.
- **Fast tier on the Mac (2026-10-11, `scripts/test.sh cpu -L fast`, clang, M3 Air):**
  180 of 180 pass (0 failed), 269 s of test time (4 min 31 s wall), after the three
  clang fixes. 3 of the 180 were *skipped*, as designed: `benchmark_plot_tgv_re1600`,
  `_channel` and `_dfg_3d` need `python3 benchmarks/fetch_references.py` (exit 77, not
  fetched on this machine). This closes the macOS bring-up of `scripts/setup.sh` ->
  build -> doctor -> case -> fast tier. **Not run on the Mac:** the slow tier,
  `scripts/asan.sh`, the Python tests (`cpu-python` preset), `scripts/docs.sh` (Doxygen),
  `scripts/debug_device.sh`.
- **Re 1000 convergence benchmark added (2026-10-11):** `benchmarks/square_cylinder_re1000/`
  (decks l05, l1, l2, l3; `analyze.py`: batch-means statistics and spectral peaks, level
  differences against the error bar, `--save`; README with the protocol). Plan:
  `docs/design/re1000_convergence_plan.md`. The ladder halves AMR tolerance and `min_size`
  per level; `t_final` 130, AMR to t = 60. L0.5 and L1 were started on the Mac; L2 and L3
  are for the desktop or HPC. **Not yet verified:** a restart with a larger `t_final`, and
  everything on Linux. `analyze.py` was tested only on the Mac's histories.
- **Re 1000 -> 10000 ramp (human 2026-10-11), decisions so far:** Re 1000 runs on this
  machine (an M3 MacBook Air), not Bridges-2; "converged" means a 1-2% change between
  AMR levels (C_D, C_L,rms, St). Only step 0 is authorised: the Re 200 OIFS deck
  (`cases/square_cylinder_re200_oifs.yaml`) reproducing on this machine, after the
  install, build and doctor. Everything past it (Re 500 calibration, Re 1000
  self-convergence, Re 10k) waits for the human. Open before step 2: whether a restart
  may change Re, whether error-bar statistics exist, what Re 1000 costs here (the
  desktop's IMEX estimate was ~5 h at np 4): answered by the probe below.
- **Re 1000 cost probe (2026-10-11, this Mac, np 4, OIFS CFL 2, BDF3):**
  `runs/re1000/re1000_l1_probe.yaml` = the Re 200 OIFS deck with Re 1000, `min_size`
  0.1, `max_elements` 6000, `t_final` 30. Status ok, 1,076 steps, 17.2 min wall (Re 200
  took 3.9 min to t = 30). The C_D, C_L numbers in it (C_D up to 3.2, C_L +-2.4) are
  the start-up transient, not results.

  | t | elements | dt | s/step | its |
  |---|---|---|---|---|
  | 5 | ~1,600 | 0.034 | ~0.6 | 19-21 |
  | 14 | 1,908 | 0.028 | ~0.7 | 19 |
  | 20 | 2,782 | 0.025 | ~0.9 | 16 |
  | 26 | 4,167 | 0.026 | ~1.3 | 16 |
  | 30 | 4,994 | 0.016 (last step, cut to land on t_final) | ~1.7 | 15 |

  **Measured:** cost is ~0.33-0.40 ms per element per step at np 4 (the Re 200 run
  gave the same), so per-step cost follows the element count; dt stayed at 0.021-0.026
  for t = 8-28 (Re 200: 0.072); the mesh reached 4,994 elements at t = 30 (Re 200 reached
  3,000 only at t ~ 50), still growing ~120/time unit, so the 6,000 cap will bind from
  about t ~ 35 and level 1 is cap-limited, not tolerance-limited.
  **Extrapolated (assumptions in brackets):**
  - Full t = 160 at level 1: ~5,200-7,200 more steps [dt 0.018-0.025, mesh at the 6,000
    cap from t ~ 35] x 1.9-2.4 s [0.35 ms/element] = 2.7-4.8 h, so **~3-5 h in total**.
    Risk: the transient is long (|C_L| up to 2.4 at t = 30), the Re 200 deck averages
    over t = 90-160 and Re 1000 in 2D may not be periodic, so t_final may need to be
    longer; not measured.
  - Convergence series [halving `tolerance` and `min_size`, cap raised to match, elements
    x2-3 and dt x0.4-0.5 per level, so wall x4-7.5]: L2 ~12-38 h (~20 h), L3 ~50-280 h
    (~4 days or more). L1 is an overnight run, L2 a day, **L3 is not realistic on this
    machine** (and thermal behaviour of a fanless Air over days is unknown).
  - Re 2000 / 5000 at level-1 resolution [work ~ Re^1.5, status.md]: ~11 h / ~45 h.
  - **Re 10000 does not fit here.** 1-3e6 unknowns (status.md) at the measured ~22
    dofs/element is ~45-140k elements, ~16-48 s/step [0.35 ms/element, optimistic: no
    iteration growth], ~200k steps at dt ~8e-4 = 37-110 days; memory ~4-12 GB [~4 KB per
    unknown, NOT measured] against 8 GB total RAM (`sysctl hw.memsize` = 8,589,934,592).
    That is HPC work (status.md: Bridges-2).
- **First build on clang (`scripts/build.sh cpu`), 2026-10-11:** configure passed (AppleClang
  17.0.0 through the Spack `mpicxx`, MFEM and GTest found). Then compile stopped at
  `src/precond/mass_inverse.hpp:187:13: error: private field 'comm_' is not used
  [-Werror,-Wunused-private-field]` (clang-only warning, GCC doesn't have it; header
  included from `cahouet_chabard.hpp`). Not fixed: it is a `src/` change (ask). A
  keep-going pass (`ninja -k 0`) compiled 106 of 173 targets and found three distinct
  causes, all clang-only (GCC accepts them), all in `src/`; the other 67 targets depend
  on the failed ones, so more may appear once these are fixed:
  1. `src/precond/mass_inverse.hpp:187` `comm_` is set (line 69) and never read:
     `-Wunused-private-field`; 57 of the errors, via every includer of the header.
  2. `src/config/parameters.cpp:747` `DeckReader r(YAML::Node(YAML::NodeType::Map));` is a
     most-vexing parse for clang (`-Wvexing-parse`, then three follow-on errors).
  3. `src/config/deck_reader.cpp:119-120` a lambda captures the structured bindings
     `parent`/`leaf`: a C++20 feature, the build is C++17 (`-Wc++20-extensions`).
  **Fixed (2026-10-11, human approved):** commit `2870c38` (remove the unused member;
  brace-initialize the `DeckReader`; hold the `SplitPath` pair and take references). The
  whole `cpu` preset then built, 125/125, with no further clang errors (only the
  linker's "ignoring duplicate libraries" warnings); `scripts/style.sh check` clean;
  `scripts/doctor.sh --run` all good (cavity smoke run at 2 ranks, 3 s).
- **`scripts/env.sh` failed when sourced from zsh** (the macOS login shell, and what
  AGENTS.md's `. scripts/env.sh` runs in): `'gcc' -> '/usr/bin/gcc' is NOT inside
  spack`. Cause: the declared-external prefixes are one newline-separated string
  iterated with an unquoted `for _p in $_spack_ext_prefixes`; zsh does not word-split
  unquoted expansions, so with TWO declared externals (the Spack gcc and `apple-clang`,
  /usr) the pattern match never saw `/usr`. One external worked, which is why Linux
  never saw it. **Fixed (2026-10-11, human asked):** a `while read` loop over a here-doc
  (and an empty line is skipped, since the pattern would be `/*`). Checked: `. scripts/
  env.sh` returns 0 under `sh`, `bash` and `zsh` on this Mac. Not run on Linux.
- **`ccache` is not installed** on this machine: the build prints "compiler launcher
  'ccache' not found on PATH -- building without it". AGENTS.md rule 4 (never
  `rm -rf build/`) assumes the cache; here every rebuild is a full one.
- **Setup layer restructured (human 2026-10-11, "keep one setup.sh as the single entry
  point ... move platform facts into small sourced files"):** `scripts/platform/darwin.sh`
  and `linux.sh` hold the GCC spec, the `c`/`cxx` compiler, the job count and the
  `apple-clang` external; `setup.sh` sources the one `uname -s` picks (`INCNS_PLATFORM`
  overrides). Checked on this Mac: the generated `spack.yaml` machine settings are
  identical to before, `--plan` shows `gcc@14.2.0~libsanitizer` here and `gcc@14.3.0` under
  `INCNS_PLATFORM=linux`, an unknown platform exits 2 with a message, and `nproc` -> `sysctl
  -n hw.ncpu` fixes the `-j 4` fallback. **Not run on Linux.** `env.sh`'s zsh bug, the
  install check and the `src/` warnings are NOT part of this change (shared bugs, below).
- **Compiler version in the docs (fixed 2026-10-11):** `docs/install/spack.md` and the
  `setup.sh` header, `desktop.md` and the YAML examples no longer name a version: the
  GCC version is written once, in `scripts/platform/<uname -s>.sh` (human 2026-10-11: a
  different gcc is coming).

**Just landed (`gpu-validation`, merged to `main` and pushed 2026-10-10): the Bridges-2
GPU environment rebuilt, the setup machinery made machine-neutral, and the first GPU run
of the fast tier.**
- **GPU fast tier (H100, CUDA backend, MFEM `1c19aba`, hypre 2.33.0 +cuda):** np 2 60/60.
  np 1 61/63: `nse_mms_test` and `tgv_nse_test` timed out at 15 min while sharing one card
  (every np 1 test uses GPU 0); both pass at np 2 (428 s, 224 s), and `nse_mms_test` passed
  alone at np 1 (13/13, 291 s) before the last changes. Run GPU tests two at a time at
  most, and with Open MPI's core binding off (`PRTE_MCA_hwloc_default_binding_policy=none`):
  concurrent `mpirun`s otherwise pin to the same cores.
- **AMR marker, element cap** (`refinement_marker.cpp`): indicators equal to within 1e-8
  relative are marked together. When the cap would split such a group, the whole group is
  marked if that exceeds `amr.max_elements` by at most 10% (human: a small overshoot is
  fine), and none of it otherwise. Why: the bisection used exact comparisons, so in a
  symmetric flow roundoff chose which members of a tied group were refined. A CPU restart
  is bitwise, a GPU one reproduces the state to ~1e-11: `amr_checkpoint_test` 2D differed
  by 74% on the GPU. It now matches to 1e-13 (112 elements; 118 before, cap 120).
- **`amr.threshold_mode` defaults to `absolute`** (human: refine to the tolerance you
  need, not to an element count; no cap by default). It was `relative`. Every test, app
  and example that sets `theta` now sets `relative` explicitly, so their results are
  unchanged. A deck that sets only `amr.theta` must add `threshold_mode: relative`.
- **Tolerances (human-approved):** `amr_event_test` E2 compares attempted steps to 1e-9
  on every backend (GPU: ~2e-11). `nc_stokes_test` S4 allows iteration counts to differ by
  3 (was 1): on a GPU hypre's coarsening draws from one random stream per process, so an
  AMG hierarchy depends on every setup before it (the same problem took 21 to 24
  iterations depending on what had run; the two meshes differed by up to 2; counts are
  identical on a host backend). `HYPRE_SetUseGpuRand(0)` removes the effect but made the
  GPU tests 3 to 11 times slower, so it is not used. **Open:** reseed the GPU stream per
  AMG setup instead; until then a restarted GPU run builds different hierarchies from the
  uninterrupted one.
- **Build:** the CUDA compute capability comes from the environment's `cuda_arch`
  (`scripts/env.sh` exports `INCNS_CUDA_ARCH`), no longer a hardcoded 90.
- **Setup machinery:** `scripts/setup.sh --plan` is the one entry point. An environment
  can carry `INSTALL` (its own procedure; `setup.sh` reports and runs nothing), `hosts`
  (full-hostname patterns), `preset` and `spack.yaml.sha256` (written by
  `scripts/envlock.sh`: a lock is stale when its `spack.yaml` changed, whatever the file
  times say). `scripts/doctor.sh` only checks; `--run` builds and smoke-runs.
  `scripts/test.sh` takes the preset as its first argument. `setup.sh --yes` refuses on a
  cluster login node without `--on-login-node`.
- **Agents:** rule 3 asks before any job or allocation and says nothing in the repo is
  permission; the committed `.claude/settings.json` only denies or asks. Allow rules are
  personal (`.claude/settings.local.json`, git-ignored).
- **Bridges-2:** `psc_gpu` re-resolved (MUMPS, the MFEM pin, `py-pip`) and installed with
  `environments/psc_gpu/install.sbatch` (one H100, 10 min for the five new packages);
  numba 0.66.0 from the wheel folder. `psc_cpu` (the same stack, no CUDA) is resolved and
  mirrored, not installed; it resolved hypre 3.1.0 where `psc_gpu` has 2.33.0, and is
  pinned to the GPU nodes' compiler until there is an RM allocation. Neither has
  `py-matplotlib`.

**Just landed (`benchmark-infra`, merged to `main` and pushed 2026-10-10): the benchmark
infrastructure** (human
2026-10-10: "do not run any of the actual test problems, I will do that at a later time, I
just want the test infrastructure set up"). Re_tau pair 180/395, constant pressure
gradient and `benchmarks/<case>/` were the agent's choices (the human didn't pick).
- Deck `forcing.body_force`; initial conditions `taylor_green_3d` and `channel` (Reichardt
  + curl perturbation, exactly divergence-free, zero with its normal derivative at the
  walls); `mesh.geometry: file` (`mesh.file`, Gmsh `$PhysicalNames` -> boundary names,
  `mesh.boundary_names`); the DFG channel in 3D (`Extrude2D`, curved nodes kept, z faces
  join `walls`; product parabolic inflow); `output.diagnostics` without field output,
  appended on restart; `reference.re_tau`.
- `post/channel_statistics`: plane and time averages at every velocity-node height
  (U, V, W, u'u', v'v', w'w', u'v'), tau_w, u_tau, Re_tau, bulk velocity; Save/Load exact,
  carried by Case's checkpoints; summary section `channel`, `<name>_profiles.csv`.
- `benchmarks/`: tgv_re1600 (n16/32/64), channel (Re_tau 180/395), dfg_3d (3D-1Z/3Z),
  sphere_re300 (README: needs a hex mesh). Reference data is NOT committed (human: the
  sources state no redistribution terms): `python3 benchmarks/fetch_references.py`
  downloads it from the original sources into git-ignored `*/reference/*.csv`;
  `plot.py --selftest` is in the fast tier and skips (exit 77) until it's fetched.
- Tests: `channel_statistics_test` S1-S5 (exact plane averages, wall shear, trapezoid,
  Save/Load, restart through a checkpoint == uninterrupted), `benchmark_setup_test`
  F1-F2, I1-I2, M1-M2, B1. None of the benchmarks has been run.
- Found while testing: `solver.atol` defaults to 0, so a step that starts from an exact
  steady state aborts ("implicit solve did not converge"): its residual cannot drop by
  rtol. Worth a small absolute floor by default (changes baselines: the human's call).

**Just landed (`oifs-mass`, merged to `main` and pushed 2026-10-10):** OIFS's substeps invert the BDF step's own mass
operator, so the two can no longer differ (human: "I literally want it to error out if
the two aren't consistent, using the diagonal for both of them as the default is fine").
Pointwise for a diagonal mass (collocated, conforming: the default, unchanged results),
Jacobi-CG per RK stage otherwise (consistent mass; collocated at hanging nodes, ~6
its/stage), always the FULL inverse (a constrained solve left 3.6e-4 vs IMEX). Setup
checks M^-1(Mz) = z and aborts otherwise. OIFS now matches IMEX with either mass (wall MMS
3.0e-6 both; was 1.6e-3 mismatched). AMR runs change slightly (exact P^T D P instead of
row-sum lumping). Deck option `discretization.mass: auto | collocated | consistent`
(replaces `collocated_mass`; auto = collocated under OIFS, consistent under IMEX).

**To evaluate later (human 2026-10-10: "we'll do a bunch of numerical experiments at a
later date"): collocated vs consistent mass in general**, for IMEX and OIFS, as the
default. Questions: accuracy per DOF on resolved benchmarks (TGV, channel, DFG), cost
(the consistent mass under OIFS adds a CG solve per substep stage: ~2x wall on the 6x6
cavity, ~6 its/stage at hanging nodes), and robustness. Data so far: on the Re 200 square
cylinder the collocated mass moves IMEX by C_D -0.4% / C_L,rms -1.8% (a resolution
effect: both should converge to the same answer); on the 6x6 cavity, KE differs 1.1%.
Benchmark plan (3D TGV, channel Re_tau 180/395, sphere, DFG 3D,
Ethier-Steinman) proposed; waiting on the human's answers (Re_tau pair, forcing,
directory layout, GPU-hours).

**Just landed (branch `license`, merged to `main` and pushed 2026-10-09, human: "merge and
push"):** BSD-3-Clause `LICENSE` (copyright Michael
Schneier; human: "it should be me"). README rewritten so the agent is the way in: what to
ask, what the agent asks first, what comes back. Human: the test cases are not the
standard CFD benchmarks to showcase (literature table removed; `cases/` decks are now
"example decks"), and the "production DNS/LES on DOE systems" line removed. AGENTS.md
gains "Working with the user" and rule 9 (licensing); user runs go under `runs/`.
Spack for anyone's machine (human: "someone points their agent at the repo ... it can do
everything from installing with spack (with users permissions of course)"):
`environments/stack.yaml` (portable stack, + py-matplotlib), `scripts/setup.sh` (plan,
then `--yes`; builds gcc@14.3.0 and declares it external), `scripts/machine.sh` (no default
machine any more; this checkout has `environments/.machine` = desktop),
`docs/install/spack.md` (how it all fits, for agents). Not yet run: `setup.sh --yes` on a
fresh machine. Next: that test (host terminal, clean VM or container); the TGV example
(deferred by the human).

**Just landed (branch `agent-first`, merged to `main` and pushed 2026-10-09, human:
"commit merge and push"):**
- Environment refresh: MFEM 4.10.1-dev with MUMPS, hypre 3.2, libCEED 1.0. MFEM pinned
  to `1c19aba` because of an upstream regression (PR #5297, reported 2026-10-09).
- Agent-first front door: `AGENTS.md`, short README, `docs/using.md`,
  `docs/install/desktop.md`, CLAUDE.md split into `developing.md` + this file.
- Generated deck reference (`docs/deck_reference.md`); unknown deck keys are errors.
- `scripts/doctor.sh`: setup check; `--run` adds the build-if-missing and smoke run (seconds).
- Benchmarks as decks: `mesh.geometry` (square cylinder, DFG channel), deck velocity
  BCs (constant / parabolic, ramp / sine), boundary coverage check, `uniform` initial
  flow, time-based AMR, force statistics, progress / history, pressure probe,
  reference values, and `summary.json` from every `run_case` run. Decks:
  `cases/square_cylinder_re200[_oifs].yaml`, `dfg_2d1.yaml`, `dfg_2d3.yaml`,
  `cavity.yaml`. The OIFS deck reproduced the validated result (C_D 1.4275, C_L,rms
  0.3940, St 0.1568).

**Validated:** `dfg_2d1.yaml` (relative errors C_D 1.05e-4, C_L 2.79e-3, dp 4.06e-3 = the
slow-tier values, 70 s), `dfg_2d3.yaml` (dt 0.001; C_D,max 0.08%, 172 s),
`square_cylinder_re200_oifs.yaml` (C_D 1.4265, C_L,rms 0.3923, St 0.1565 in 2,191 steps,
13 min: the driver's run to every digit). **IMEX deck run:** `square_cylinder_re200.yaml` validated 2026-10-09: C_D 1.4436,
C_L,rms 0.4034, St 0.1568 in 11,130 steps, 42 min at 4 ranks. (An earlier run that seemed
to stop at t = 90 had been killed by the agent itself, to restart it on the corrected deck;
the restart never launched.) Against the OIFS deck: C_D 1.2% and C_L,rms 2.8% higher, St
equal; the "within 1%" figure is IMEX with OIFS's collocated mass, not this deck.
Debug-device sweep: green, 114/114. The full sweep after the merge failed only
`run_monitor_test` R3 (AMR on a periodic Taylor-Green box), the known MFEM periodic-NC
debug-device false positive; R3 now skips there like `amr_flow_test.cpp:88`. Bridges-2: `psc_gpu` was
re-resolved on the cluster on 2026-10-10 (MUMPS and the pin).

**Next, in order:**
1. A cold-start test: a fresh agent with no context, given real tasks (install check,
   reproduce a benchmark from its deck, set up a new case); fix where it stumbles.
2. HPC preparation for Re 10k / 100k (2D): decide whether OIFS at CFL 2 becomes the
   default; GPU validation (developing.md §7.5) or a CPU strong-scaling check;
   statistics with error bars.
3. When MFEM fixes PR #5297's regression: drop the pin in both `spack.yaml` files.

**Open decisions for the human:** OIFS CFL 2 as the default for production runs.

**Open finding (2026-10-09):** DFG 2D-3 on the refreshed stack differs from the 10-07
study: C_D,max 2.94858 at dt 0.001 (study 2.94451, a 0.14% spatial shift) and dt 0.0015
now diverges at t = 3.91 (the study found it stable; threshold ~0.0017). Bitwise the same
with the pre-OIFS code (`530f93d`) on the new stack, so not this branch's code; the
square cylinder is bitwise unchanged across the refresh (driver OIFS CFL 2: 1.4265 /
0.3923 / 0.1565, 2,191 steps, both stacks). Suspects: MFEM 17d1afc → 1c19aba (curved
DFG mesh / quadrature?) or a code change between the 10-07 study and 10-08. The deck uses
dt 0.001.

## Log: state and decisions (last full update 2026-10-08)

**Branches.** `main` includes `rotational-schur` (merged and pushed 2026-10-07:
rotation-aware Schur PC, DFG 2D-3 driver, `jacobi_pcg` default + grad-div in LOR-AMG, step
control `fixed|error|cfl` with Nek's CFL and the BDF3/EXT3 estimator, case-level CFL control
at 0.5, the debug-device ess-list fix developing.md §6) and **`dfg-scheme-comparison`** (fast-forward
merged and pushed 2026-10-08, human: "merge and push everything"). New work goes on a new
branch; merge/push only when the human says so. dfg-scheme-comparison: DFG app `-mref` nested meshes, per-step
history CSV and `step_wall`; `bench/dfg3/` (pinned two-slot queue runner, sweep generator,
analysis, HTML report generator, raw results of 2026-10-08); the work-precision write-up
(`docs/imex_vs_semi_implicit.md` §7 + summary at its top); **case-level default back to
BDF2/EXT2** (human 2026-10-08; developing.md §5.2). The charts were published as a private claude.ai
artifact (regenerate with `bench/dfg3/report.py OUT.html
bench/dfg3/results_2026-10-08/findings.html bench/dfg3/results_2026-10-08/sweep.txt`).

**Scheme comparison concluded (human 2026-10-08): for DFG 2D-3 there is no benefit to the
semi-implicit rotational form** — same CFL limit as IMEX, 7–9% more per step, comparable
lift, 50–70× worse drag; both fail silently past the limit (developing.md §5.6, the doc §7). The finest
mesh (M2) was started and stopped unfinished at the human's call.

**`directional-do-nothing`** (fast-forward merged and pushed 2026-10-08, human): Braack &
Mucha's directional do-nothing outflow condition for the IMEX convective form, default on,
explicit (on the RHS with the convection; no operator or preconditioner change — a
semi-implicit LHS version is the fallback if outlet backflow ever destabilizes below the CFL
limit) (developing.md §4 BCs; `docs/outflow_conditions.md`); `directional_do_nothing_test`; DFG app
`-cdn` for the classical condition.

**Square cylinder (Joly, Etienne & Pelletier, J. Fluids Struct. 28 (2012) 232–243;
human 2026-10-08) — branch `square-cylinder`, fast-forward merged and pushed 2026-10-08
(human: "commit merge and push this stuff")**: `src/mesh/square_cylinder.{hpp,cpp}`,
`apps/square_cylinder.cpp`,
`ConvectiveCfl::RateAndLocation` + `Case::ConvectiveCflNumber(Vector* where)` (where the
CFL rate peaks), `cylinder_mesh_test` M3. Details developing.md §5.6. The paper PDF `big_domain.pdf` sits
untracked in the repo root (do not commit it; copyright).
- **Re = 200, α = 0 — DONE, all within the human's 5%** (IMEX, Q3/Q2, CFL 0.5, EXT2, AMR):
  C_D 1.4434 (paper 1.44, +0.2%), mean C_L 0.0003 (0.001), C_L,rms 0.4034 (0.42, −4.0%),
  St 0.1568 (0.151, +3.8%), 10 periods, period spread 0.03%. Fine AMR (tol 0.12, 5,003
  cells vs 3,001): C_D 1.4425, C_L −0.0003, C_L,rms 0.4025, St 0.1569 (11,518 steps, dt
  5.3e-3–2.5e-2) — every coefficient within 0.2% of the base run → wake-converged; corner resolution (smallest cell 0.125 in
  both) and dt were NOT varied. dt 6.2e-3–2.5e-2 (mean 1.4e-2, ~440 steps/period vs the
  paper's ~32 at UΔt/D = 0.2); 11,138 steps, 69 min at np 4. Remaining −4% / +3.8% may be the
  paper's large implicit step (untested). Literature scatter (their Table 1): C_L,rms
  0.32–0.55, St 0.142–0.170.
- **Re = 1000 — moved to HPC (human 2026-10-08: "that whole campaign will be left to the
  hpc")**. A desktop level-1 run (tol 0.25, min size 0.1, cap 6000) was stopped at t = 20:
  dt 1.5–4e-3 (5–9× below Re 200), 4,708 cells at t = 20 and still refining, CFL peak in the
  near wake ((2.4, −0.3) → (9.4, −0.4)), ~0.44 s/step at np 4 → ~5 h for t = 160 at that
  level. No public benchmark: plan = self-convergence over 2–3 AMR levels (halve
  tolerance AND min size; track cells/unknowns, dt range, C_D, C_L,rms, St; converged when
  < 1–2% change) + a CFL 0.5 vs 0.25 branch from the t = 90 checkpoint; check periodicity
  (2D at Re 1000 may be irregular → batch-means error bars). Consider `-amr-start` (spin up
  on the coarse mesh) to cut the startup cost.
- **dt sawtooth on AMR meshes (open):** once shedding starts, every ~0.13 time units the CFL
  rate jumps ~1.5× in ONE step → the controller cuts dt ~1.9e-2 → ~1.0e-2 and regrows ×1.2.
  Forces unaffected (C_D jitter ~1e-4); ~10–20% extra steps. Likely a fast mode at its
  stability limit at CFL 0.5 on the AMR (hanging-node) mesh — conforming DFG meshes were
  stable to 0.7. Diagnose with the CFL location now printed; a 0.4 target likely removes it.
- **Human's goal: Re = 10k and 100k on HPC** (2D on purpose; human knows it is not
  physical). Estimates (2D: unknowns ∝ Re, dt ∝ Re^-1/2, work ∝ Re^3/2, ~30× per decade):
  Re 10k ~1–3e6 unknowns, smallest cell ~0.02 (Q3), dt ~8e-4; Re 100k ~1–3e7, ~0.006,
  ~2.5e-4; ~1e4 core-hours at Re 100k if the solver scales. Prerequisites agreed so far:
  the dt-sawtooth diagnosis; GPU validation (developing.md §7.5) or a CPU strong-scaling check of the CC
  Schur / LOR-AMG; an element-order comparison at Re 1000 (Q5–Q7 likely better per unknown
  at high Re); OIFS (subcycled convection, CFL 2–4) before Re 100k; time-based AMR
  scheduling (start / interval / end) in Case/Parameters for decks (now only in the
  driver); statistics with error bars; same-np restart constrains job chaining. OIFS now
  exists (`oifs` branch: CFL 2 within 1% of IMEX at Re 200, see above). Batch
  scripts: submitted by the agent once the human approves the cost (changed 2026-10-10).
- **AMR stays refinement-only (human 2026-10-08):** derefinement breaks the multistep
  history (refinement transfers every BDF/EXT level exactly; coarsening must project them),
  and single-step schemes are order-bound. For a statistically stationary turbulent wake,
  refine-only converges to resolving the wake region anyway; bound the startup waste with
  `-amr-start` (spin up on the coarse mesh — spin-up only has to reach the state, not
  accurately) and `-amr-end`; a remesh-and-restart (multistep startup ramp) is the fallback.

**`oifs` (branch; human 2026-10-08 "give a shot at implementing the OIFS stuff", 2026-10-09
"you should be running everything with BDF3 with the OIFS stuff"; fast-forward merged and
pushed 2026-10-09, human: "merge and push"):** OIFS (developing.md §5.2), BDF3 by default under OIFS, collocated GLL mass always under
OIFS. Square cylinder Re 200 (same AMR as the base IMEX run, np 4; vs IMEX CFL 0.5 with
the SAME collocated mass: C_D 1.4381, C_L,rms 0.3957, St 0.1564, 9,103 steps):
```
OIFS BDF3, collocated   C_D            C_L,rms         St             steps
  CFL 1                 1.4346 -0.2%   0.3931 -0.7%    0.1565 +0.1%   4,379
  CFL 2                 1.4265 -0.8%   0.3923 -0.9%    0.1565 +0.1%   2,191
  CFL 4                 1.4772 +2.7%   0.4778 +21%     0.1507 -3.6%   1,120
```
**CFL 2 is the sweet spot: within 1% of IMEX in 4.2× fewer steps (~3× less wall time;
an OIFS step costs ~1.3–1.5× an IMEX step).** CFL 4 loses the lift amplitude. History of
the study (each step measured): BDF2 + consistent BDF mass converged in dt to the WRONG
answer (C_D 1.49, C_L,rms 0.45: +3% / +12% at CFL 4 → 0.5) — the mass mismatch (developing.md §5.2),
not the wall treatment (pointwise wall acceleration: no change, reverted), the force
evaluation (wall MMS: = IMEX to 1.5e-3), hanging nodes (`oifs_test` O3) or the outflow
(IMEX classical = directional BITWISE on this domain: no outlet backflow). BDF2's
trajectory time error at CFL 4 was ~4× BDF3's (C_D). OIFS also has no dt sawtooth (steady
dt at every CFL). The collocated mass itself moves IMEX by −0.4% C_D / −1.8% C_L,rms.
DDN under OIFS done (explicit, extrapolated in the BDF step: Braack & Mucha's Table 5.1
to IMEX's values at CFL 2, stable to CFL 8). **Error control under OIFS: not pursued
(human 2026-10-09: "we are always going to need to do some multiple of CFL basically,
so maybe we just skip the error estimator")** — rejected at validation. Open: OIFS cost
per step (substep CFL 0.5 — try 1.0); the fixed-step BDF3-OIFS startup (O(dt²) starter);
whether the case-level default for HPC runs should become OIFS CFL 2.

**Environment refresh (2026-10-09, human):** full desktop refresh with current spack:
MFEM 4.10.1-dev **+MUMPS**, hypre 3.2.0, libCEED 1.0.0, SuiteSparse 7.14 (developing.md
§3). MFEM's head broke 3D parallel anisotropic AMR (PR #5297; reported upstream), so MFEM
is **pinned to `1c19aba`** in both `spack.yaml` files until it is fixed. On the pinned
stack: fast tier 159/159, Python 12/12. The Bridges-2 lock (`psc_gpu`) was re-resolved on that
machine on 2026-10-10 with the same MUMPS + pin edit.

**`agent-first` (branch, 2026-10-09, human: "an agent first repo where someone can just
point their agent at it"):** `AGENTS.md` front door, short README, `docs/using.md`,
`docs/install/desktop.md`, this file and `developing.md` split out of CLAUDE.md (which now
imports them), specs moved to `docs/design/`. Next on that track: `scripts/doctor.sh`,
validated decks for the benchmarks (needs the drivers' AMR scheduling / averaging in the
library), `summary.json` per run, a generated deck reference, a cold-start agent test.

**Open follow-ups** (none started):
- Outflow condition for the rotational form: do-nothing acts on the Bernoulli head
  ("notoriously bad" for this form, human); the DFG studies used a Dirichlet outflow instead.
- A velocity PC whose iteration count does not grow as σ = β₀/Δt shrinks (Jacobi-PCG's
  does, so CFL 0.5 → 0.7 saves 29% of steps but only ~10–15% of wall time) — e.g. the
  p-multigrid velocity block (spec Part D).
- Warn when the measured CFL creeps above `cfl_target` (the IMEX/EXT3 failure symptom on
  DFG); the rotational form's silent failure shows no such signal.
- If the rotational form is pursued: find the source of its DFG drag error (the form's
  spatial error vs the force evaluation from the Bernoulli head).
- `auto` Schur: compute μ every N steps (it costs ≤1% per step now; low priority).
- Fast tier ~177–197 s, near the 3-min budget; critical path `tgv_nse_test_np2`.
- LOR-AMG with strong grad-div in 3D is still 4–5× its no-grad-div count; BoomerAMG's
  elasticity mode is the standard fix but needs `Ordering::byVDIM` (velocity is byNODES).
- Rotation-aware Schur `auto` thresholds (spec's 20/10) uncalibrated in 3D; default `cc`.
- p-multigrid velocity block (spec Part D Level 1) not built (human: PBJ only for now).
- Skew-symmetric convective form: gated on upstream MFEM (developing.md §5.4 TRAP).
- Not pursued (human 2026-10-07: "not interested in implicit methods at the moment"): a
  CFL-free option (Oseen linearization (w*·∇)u^{n+1}, or converged Newton/Picard steps).
- PENDING GPU VALIDATION list (developing.md §7.5).
- Upstream candidates: the MFEM debug-device BlockVector false positive (developing.md §6).
- 2D NSE TGV pressure-rate oracle never built (human accepted the 3D MMS, 2026-07-24).
- `VectorRotationalConvectionComponentIntegrator::AssembleEA` is untested.
- Convective-form TGV at ν = 0.05 is pre-asymptotic on coarse Q3 meshes (4×4 → 8×8 raises
  the error 0.012 → 0.024 before 16×16 drops it to 0.0027); oracles run at ν = 1.

**Design specs** in `docs/design/` (authoritative design records; developing.md wins where they
disagree): `docs/design/SPEC_cahouet_chabard_mfem.md` (+ `docs/precond_cc.md`),
`docs/design/rotational_convection_pa_spec.md`, `docs/design/rotational_schur_velocity_mg_spec.md`,
`docs/design/amr_spec.md`, `docs/design/vecdivdiv_spec.md`, `docs/install/bridges2.md`.
