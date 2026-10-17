# Project status

The living handoff log: where the project stands, what was decided (dated, with who
decided), and what is open. The maintainer and their agents update it as work lands.
Rules and design live in [developing.md](developing.md); this file is the *state*.

## Where development left off (2026-10-16)

Keep this section short and current: whoever stops working updates it (AGENTS.md rule 8).
The dated log below keeps the history.

**Defaults: Q6/Q5, grad-div gamma = 6 h_K / k_u, OIFS at CFL 2 (2026-10-16, human: "go with
p=6 and your grad-div scaling recommendation", "make oifs the default it honestly seems to
always be better").** The study is in throughput.md ("The default
polynomial order"): Taylor-Green Re 1600 with OIFS, 96^3 to 192^3, outer CFL 0.5-4. Q6 is the
cheapest order to a 5 % largest dissipation error (126^3: 212 s at CFL 1, 13.7 GiB); Q3 and
Q4 do not reach it by 192^3. With gamma = 2 h_K the velocity PCG's iterations grew 17 -> 38
from Q3 to Q7 at equal DOFs; 6 h_K / k_u keeps them at 17-25 and leaves Q3 unchanged.
`discretization.order_u` defaults to 6 and `order_p` to 5 (a deck that sets only `order_u`
gets k_u - 1); `physics.grad_div_h` defaults to < 0 = 6 / k_u (`Parameters::GradDivH()`).
`time.convection` defaults to `auto` (`Parameters::Convection()`): OIFS where it applies
(Navier-Stokes, the convective form, fixed or CFL steps), IMEX for Stokes, the rotational
form and error-controlled steps; `time.cfl_target` defaults to auto (`CflTarget()`: 2 under
OIFS, 0.5 under IMEX) and `time.oifs_cfl` to 1.5, the study's settings. The verify tier
follows the defaults by design, so it now verifies Q6 with OIFS. **Open:** no order
reaches 2 % by 192^3; validation on the channel at Q6.

**GPU memory toward 100M DOFs on one H100 (2026-10-16, branch memory-audit).** The pressure
LOR-AMG was 39 % of a Q7 run (752 B/DOF). Freeing MFEM's LOR scaffolding after assembly
(6d90d0c) saves 460 MiB; capping hypre's device pool (it rounds every allocation up to a
power of 8 and caches every freed block) saves 2.7 GB at Q7 128^3 with no time cost
(ConfigureDevice; docs/gpu_memory.md). Q7 1,946 -> 1,488 B/DOF. Measured capacity at Q6
(Taylor-Green, one H100): 39.2M DOFs before (out of memory at 57.3M), 57.3M with the fixes
(out at 70.4M), 70.4M with `solver.outer: schur` (out at 85.5M). The working list (the
measured ledger at 57.3M, the phases, the held items): gpu_memory.md, "Q6: toward 100M DOFs".
Change 17 (no RAP over identity prolongations, three kept copies dropped) saves 3.6 GiB at
57.3M in both methods, bitwise unchanged; change 18 (the collocated mass as its diagonal)
3.2 GiB more, at roundoff and 10–14 % faster; change 19 (our own PA blocks from MFEM's
public pieces, shared element vectors) 2.8 GiB more, bitwise unchanged. Peak at 57.3M:
coupled 61,955 MiB, Schur 48,127 MiB (881 B/DOF); change 20 (the divergence from the
cached Jacobians, phase 2) 2.6 GiB more at 0.5–1.4 % per step: coupled 57,603 MiB, Schur
46,031 MiB (842 B/DOF); changes 21-25 (OIFS keeps only its combined wind data,
the on-the-fly OIFS apply as an opt-in, IMEX's convection on the fly, the lean vector
mass): OIFS coupled 50,465 MiB, Schur 40,351 MiB (738 B/DOF); IMEX coupled 44,745 MiB,
Schur 32,165 MiB (589 B/DOF).

**Q6 on curved meshes on a GPU (2026-10-17, change 26).** The 3D verification (human: "run
the 3d verification tests ... to really be sure we didn't introduce any bugs") found the Q6
DFG 3D decks stopping at setup on the H100 with an illegal address -- in main and in the main
of before the memory work alike; the CPU runs them. The cylinder meshes take the velocity
order as their geometry order (7 nodes per direction at Q6), and the dealiased convection's
10-point rule (MFEM's form on curved elements: IMEX's explicit term, OIFS's wall nodes) asks
MFEM's quadrature interpolator for the Jacobians, whose runtime-size kernels hold 8 per
direction on a GPU, unchecked. `AddInterpolationKernels` registered compile-time kernels for
linear nodes only; it now covers node orders 1-7 at 9 and 10 points (3D; 9-11 in 2D). Curved
meshes at Q7 (11 points, where MFEM's 3D kernel cannot launch 11^3 threads per element): the
dealiased convection takes the OIFS kernel's stored path instead of MFEM's form (change 27,
human: "ya do that small change for Q7").

**Branch `verification` (2026-10-15, PR #1, merged 2026-10-16): CI and the verification
tier (Phase 1 of the verification audit).** Human: "start phase 1", "the defaults will be
changing over time", rate threshold theory - 0.3 approved, grad-div "an open question",
"can you set up the CI ... are you able to handle the caching?".
- **CI** (`.github/workflows/ci.yml`, `environments/ci/spack.yaml`; developing.md §7.6):
  style, `cpu` build, Doxygen, fast tier on every push to `main` and every PR; the verify
  tier every third day (human: "it's free right?": yes, public repo), by dispatch, or
  with the PR label `verify`. Spack stack cached by the Spack release + the env's hash:
  the cold build took ~1 h (2026-10-15); with the stack and ccache warm a run takes
  ~11 min (run 38807839901: build 33 s, fast tier 198/198 in 5.3 min, verify 3/3 in
  3.7 min).
- **Verify tier** (label `verify`, np 4; developing.md §7.2): `verify_spatial_test`,
  `verify_kovasznay_test`, `verify_temporal_test`, helpers in `test/verify_util.hpp`,
  levels and the 0.3 margin in `baselines.yaml` `verification`. Written in a sandbox
  with no toolchain: compiled and run only in CI. **All three pass in CI** (run
  38805467828; 90 / 73 / 58 s at np 4); the rates are in the run summary and PR #1.
- **Next (audit Phases 2-4):** provenance (commit, defaults) in `summary.json`, splitting
  `nse_mms_test` / `tgv_nse_test` (they time out on a GPU), then the solution-
  verification and validation studies on Bridges-2.

**Ranks that own no elements run (2026-10-16, branch `throughput`).** `benchmark_setup_test`
M1 (a 3-element Gmsh mesh) segfaulted at np 4 on rank 3, which owns no elements (job
49276869; with the pinned MFEM and 4.10 alike). The CPU test jobs ran np 1 and 2 only, so
it went unnoticed. `StokesSolver::SolveTrue` built its kept block vectors only when their
total size changed. An empty rank's offsets are {0, 0, 0} from the start, so it never built
them, and `GetBlock(0)` dereferenced a BlockVector with no blocks. It now compares the
block layout. Same family in the checkpointed guess state (32287da): whether x_prev / A
x_prev existed was read off the vectors' sizes. An empty rank wrote no `guess_*_prev` /
`guess_*_ax` files, so a restart aborted there, and the import would have marked A x_prev
absent on that rank alone. `SolverGuessState::has_previous_x` / `has_step_ax` carry it now,
the same on every rank. New `empty_partition_test` puts every element on rank 0, so the
other ranks are empty at np 2 and 4. A Stokes cavity and Navier-Stokes Taylor-Green (IMEX,
OIFS, CFL control) match the default partition to 1e-8, and a restart with the guess state
(project, extrapolate) is bitwise. Bitwise-neutral otherwise: the old and new `run_case`
write byte-identical checkpoints (cavity, np 2, both guesses). smoke|fast green at np 1, 2
and 4 on the CPU (job 49277774, 72 tests each). Next: the CPU test jobs should also run
np 4 (`-R "_np4$"`).

**Schur complement (Uzawa) outer method, as an option (2026-10-16, branch
`claude/schur-cg-solver-9sf1r5`, PR #2, merged 2026-10-16, human: "merge to main and
push"). Written in a cloud session that cannot build; CI is its only build: style,
`-Werror` build, Doxygen and the fast tier pass there (CPU, np 1/2/4, 219/219). Never run
on a GPU, nothing timed.**
Human: "implement the uzawa stuff here as an option for the solver" (the proposal is
[design/uzawa_schur_cg_spec.md](design/uzawa_schur_cg_spec.md)); on the initial guess,
"ok ya that makes sense" (pressure-only projection under Schur). `solver.outer: coupled |
schur` (default coupled), `solver.uzawa.*` (flexible CG only, human: "drop both" for plain CG
and pressure FGMRES; inner tolerance `relaxed` | `tight` | `loose`, ...; not with the
rotational form), `solver.log_interval` (per-step solver log,
either method, with device memory). New `SchurComplementOperator`, `SchurSolver`,
`VelocityCG`; `StokesSolver` dispatches on `outer`; the pressure projection is
checkpointed. Memory (human: "hyper focused on ... minimizing memory usage"): the Schur
path builds nothing coupled-only and holds 4 velocity + 2k + 6 pressure vectors, 49
B/vdof against 115 before the trim (counted, not measured; uzawa.md §4). Departures
from the proposal and the study plan: [uzawa.md](uzawa.md). **Next:**
`bench/uzawa/study.sh` inside a GPU allocation (Phases 0, 2, 3, and `capacity` for the
memory; submits nothing). Gaps: the Python bindings do not expose the new keys.

**CI builds bessemer for the Spack stack's target (2026-10-16, branch
`claude/schur-cg-solver-9sf1r5`).** `persistent_fgmres_test` failed in CI at np 1 and
4 (on `main` too): FGMRES(4) differed from MFEM's by up to 4e-13 relative against a
1e-14 limit, and even FGMRES(50) was not bitwise (1-2e-15). The arithmetic outside
libmfem is MFEM's own Givens rotations and back substitution, copied; MFEM is built
by Spack with `-march=x86-64-v3` (FMA contraction), bessemer was built for the
baseline x86-64. The build step now sets `CXXFLAGS=-march=x86-64-v3`, and the fast tier passes
219/219 (CI run 38963922762). The differences a passing test prints are not in the CI
log, so "bitwise" is not confirmed, only within the test's 1e-14. A desktop or cluster
build without the stack's `-march` may show the same failure.

**Duplicate deck keys are an error; bench.py's `--set` replaces template keys
(2026-10-16, branch `throughput`).** `bench.py run --set discretization.order_u=5` had
appended `order_u: 5` after the template's `order_u: 3`, and the loader read the first
(job 49270645 ran at order 3). Now `write_deck` replaces a key the template sets (any
section, nested keys too) and adds the rest at the end of their section; the loader aborts
on a key a map sets twice, naming the key, the section and the lines
(`DeckReader::DuplicateKeys`; tests `Deck.DuplicateKeysAreReported`,
`Deck.EveryCaseDeckLoads`, `run_case_rejects_duplicate_key`). Before this, a `--set` had
no effect when the template set that `discretization` or `time` key, or when it named any
other section the template has (`physics`, `mesh`, `output`, ...: a second section);
`solver.*` and new `discretization` / `time` keys worked.
Verified on the CPU (job 49276422: smoke|fast at np 1 and 2) and on two H100s (job 49277617:
`deck_test` and `run_case_rejects_duplicate_key` at np 1 and 2; `bench.py` runs with
`--set discretization.order_u=5` and `--set mesh.elements=...` ran at the overridden order
and mesh). `scripts/docs.sh` not run (no doxygen on Bridges-2).

**Throughput merged into main; new solver defaults (2026-10-16).** The `throughput` branch
(2026-10-14 to 16; record and decision log: [throughput.md](throughput.md)) is in `main`.
One H100 against main of 2026-10-14: Taylor-Green 64^3 Q3/Q2 to t = 20 in **17.8 min instead
of 12 h 44 min** (43x), the channel at Re_tau 180 from t = 0 to 120 in **13.8 min instead of
7 h 15 min** (31.6x), 37-46x per step; compile-time kernels for velocity orders 2 to 6; an
order study (Q4 and Q5 reach a given accuracy fastest; Q3 is the fastest per DOF). **Case-level
defaults** (human 2026-10-16, option (a) in throughput.md's "Restarts with the projected
guess, and the defaults"): `solver.initial_guess: project` (k 8), `tolerance_reference: step`,
`rtol` 1e-6, `n_inner` 2; checkpoints keep the solvers' guess state, so restarts stay bitwise
exact. `rtol` is per problem: the Taylor-Green and channel decks set 1e-3 (CFL-limited
steps); a small fixed dt or an error controller wants 1e-6. Next (human): MFEM 4.10 with the
PR #5464 patch, then a single-kernel dFEM probe.

**Throughput plan, overnight 2026-10-14/15 (branch `throughput`, not merged).** Human:
up to 80 GPU SU, push only the branch, "as far as possible", projection "absolutely ...
as an option", single precision / libCEED / Tier 5 / new cases deferred. Record and
decision log: [throughput.md](throughput.md). Kept: the benchmark harness (`bench/`), NVTX
ranges, MFEM kernel specializations (+24 %), the fused viscous + grad-div kernel (+7 %),
and the opt-in `solver.initial_guess` / `solver.tolerance_reference`; Tier 3 found
`solver.n_inner` 3 (+54 %) and, with the projection, `rtol` 1e-6. With all of it: TGV
32^3 14.2x, 64^3 14.6x per step, and the 64^3 run to t = 20 in **1 h 01 min instead of 12 h
44 min** with the same dissipation curve (2.4e-4). Defaults unchanged pending the human
(decided 2026-10-16, above).

**GPU memory, third round (2026-10-14): the single-H100 limit is 88^3 Q3/Q2
(55.2 M velocity dofs); about 85^3 at full speed; 64^3 steps 8 % faster.** Human: "do all
of these but keep a path for the rotational form ... do the vector diffusion thing too
... except keep a path to the built in mfem way", then "do 1", "Yes, do that for sure",
"yes", "go ahead and make that CFL change". Changes 11-16 in
[gpu_memory.md](gpu_memory.md):
- **Compact vector-diffusion storage**, the default (`discretization.vector_diffusion:
  compact | mfem`): 6 numbers per point instead of MFEM's 27, apply bit-identical to
  MFEM's in all 16 tested cases. 64^3: 42.8 -> 37.6 GB, 2 % faster.
- **Nine small items** (shared P^T A P wrappers, symmetric point-block inverses, `keep_rhs`,
  w* only for the rotational form, the plain pressure mass freed, gamma per element, ...):
  0.7 GB at the Taylor-Green peak; more with walls and forces.
- **Grad-div stores alpha only** (adj(J) recomputed from MFEM's cached Jacobians): 64^3
  37.6 -> 35.3 GB, the velocity-block apply 5 % slower.
- **Delta-t-independent refresh work kept** (momentum diagonal split as c0 M_diag + a kept
  nu K + grad-div diagonal; point-block Jacobi and its CG updated in place): refresh
  1.22 -> 0.009 s per step, 64^3 step 11.11 -> 9.84 s, for 0.5 GB.
- **Restart-length guard measures the first solve** (`StokesSolver::WarmUp`, once per
  solver, 1.7 s at 64^3): the 20 % allowance missed the 9.2 GB workspace and 86^3 ran out
  of memory; now 86^3 and 87^3 run, at restart lengths 3-6 and 1 (87^3: 29.9 s per step,
  ~20 % slower than proportional).
- **CFL Jacobians per call** (MFEM's routine, not its cache; bit-identical): 64^3
  35.8 -> 34.7 GB; 88^3 runs (restart length 1, 32.4 s per step), 89^3 does not.
- **Tests:** new `StokesOperator.MomentumDiagonalAfterRefreshes2D/3D`,
  `PbjGradDiv.P4_NewDiagonalMatchesAFreshBuild`, `SchurQuality.WarmUpLeavesTheSolveUnchanged`,
  `Cfl.C4_PerCallJacobiansEqualTheCachedOnes`, R3 rewritten. Before the CFL change:
  fast tier CUDA np 1 64/64 (one H100), CPU np 1 and np 2 66/66. With it: CPU np 1 66/66.
  With the CFL change (after the commit, at the human's request): CUDA np 1 64/64, host
  device 10/10, CPU np 1 and np 2 66/66 (jobs 49216290, 49216292); CUDA np 2 (two H100s,
  job 49216291) still queued at the time of writing.

**CPU environment works on RM nodes (2026-10-14).** `environments/psc_cpu` re-pinned to
the b2rm gcc (`hosts` selects it on `r*` nodes, `preset cpu`), `reuse: false`, numpy
pinned below 2.5 for numba; Open MPI built without CUDA (the install check reads its
`opal_built_with_cuda_support` flag). CPU build plus fast tier on 32 RM-shared cores:
np 1 and np 2 66/66. The point-block Jacobi abort seen with the CUDA build on the host
device did not recur (CUDA build, host device, 2026-10-14) and never appears in the CPU
build.

**A failed solve cuts the step, by default (2026-10-13, on `main`).** Human: "if the
outer solver fails ... cut the timestep and then let it try to increase again", "a hold of
5 steps", "just option 1 and 2". `time.on_solver_failure: cut_step` (default) | `abort`;
`retry_factor` 0.5, `retry_hold` 5, `retry_max` 8, `retry_dt_min` 0. With `cut_step` the
FGMRES restart length is capped before each solve to the free GPU memory. A "max-DOF"
mode (size the mesh to the memory) was dropped: with AMR the mesh grows during the run.
Tests: `step_retry_test` R1 (fixed steps: two cuts, hold, growth back; error 0.002705
against 0.002696 undisturbed), R2 (CFL), R3 (the restart-length formula). Fast tier
CUDA np 1 64/64 (one H100) and np 2 64/64 (two H100s, one per rank), 2026-10-14; the new
`SchurQuality.RefreshWithViscosity...` first hung at np 2 (a rank-0-only collective in
the test's print, found with stack traces; fixed). The cap near the memory limit was
first tested 2026-10-14 and failed (86^3 out of memory), fixed by the warm-up above.
**Not tested:** a real (not injected) solver failure; `nse_mms_test`, `tgv_nse_test`.
Memory record: [gpu_memory.md](gpu_memory.md).

**Taylor-Green 64^3 at CFL 0.3 on one H100** (job 49142049, 2026-10-14;
human: "back off the 75^3 run to 64^3 ... increase the CFL target a little bit to .3"):
**Finished**: t = 20 in 12 h 45 min (3284 steps, 13.9 s per step, ~25 SU); energy never
increased, so CFL 0.3 is stable here and matches CFL 0.25 in E to 0.02 points. Max
−dE/dt 0.01294 at t = 8.95 (reference 0.01286 at 8.97); E within 0.6 % to the peak, 5 %
low at t = 20 (under-resolved decay). Table in
[../benchmarks/tgv_re1600/README.md](../benchmarks/tgv_re1600/README.md). The
restart-length cap stayed at 119 (of 200), far above the ~15 iterations used. Output
`runs/tgv_re1600/tgv_n64_1gpu` beside the repo on Bridges-2.

**GPU memory, second round (2026-10-13, on `main`): the single-H100 limit is 80^3 Q3/Q2
(41.5 M velocity dofs).** Human: "tackle the first 3", "go ahead with the single solver
change", "make that simple change to the CFL stuff too".
- **Measured** (Taylor-Green defaults, ~11 steps, one H100): 64^3 43.5 GB (38 GB after the
  first step), 72^3 59.0, 76^3 68.0, **80^3 78.9 GB**, 81^3 out of memory; 1.95 kB per
  velocity dof. 64^3: 10.7 s per step, 10.4 outer iterations. Headroom: FGMRES holds two
  system-sized vectors per outer iteration (0.37 GB each pair at 64^3, 0.73 GB at 80^3);
  these runs peaked at 14 iterations.
- **Diagnostics hold nothing**: `FlowDiagnostics` sums the integrands in one device pass
  (`post/flow_sums`, nvcc) instead of three assembled forms (10 GB at 64^3, built even
  with the CSV off: the summary asks for the energy). `All()` gives the three values.
- **One solver for the starter and the BDF steps**: `StokesSolver::Refresh(c0, nu)` and
  `StokesTimeIntegrator::AdoptSolver` hand the trapezoidal starter (built at nu / 2) to
  BDF2, and BDF2's to BDF3. Not with the rotational form or error control (old path).
  Test `SchurQuality.RefreshWithViscosityMatchesAFreshSolver`.
- **gamma uses the fluid's viscosity in every solver** (`grad_div_nu_ref`): the starter
  computed its nu-proportional part from nu / 2. Changes step 1 only, and only when
  `physics.grad_div_nu` is not 0.
- **The unconstrained viscous operator** (6.75 GB at 64^3) is built on demand and freed
  right after the starter's right-hand side; the BDF solver never builds it.
- **CFL estimator**: its two velocity copies live per call, and with the GLL nodal basis
  nothing is interpolated (2.0 -> 1.2 GB at 64^3; the rest is Jacobians at the nodes).
- **Pressure AMG knobs**, defaults unchanged: `solver.lp_strong_threshold`,
  `solver.lp_agg_levels`, `solver.lp_print_level`. At 64^3 the threshold changes nothing;
  one level of aggressive coarsening saves 2.1 GB with the same iterations over 11
  steps (not tried through the turbulent phase). Human 2026-10-13: AMG options are "an
  optimization for another day".
- **Looked at and left**: the convection operator (half of its 4.5 GB is the Jacobians it
  shares with the other 5-point forms, half MFEM's own data); the Schur setup (77 % AMG);
  the momentum form's 27-per-point diffusion data (human: do not replace MFEM's core
  operator; reversed 2026-10-14 with MFEM's path kept, above).
- **Tests**: fast tier CUDA np 1 63/63 before the last three small changes, then 14
  selected np 1 tests (time stepping, CFL, Schur, checkpoint, AMR event) pass. np 2 (two
  H100s, one per rank): fast tier 64/64 with the step-retry work on top (2026-10-14). `nse_mms_test`,
  `tgv_nse_test` and a CPU build not run.
- **Taylor-Green 64^3 at CFL 0.25** (two H100s, cancelled at t = 10.8 to rerun on one GPU
  with this code): energy never increases; max -dE/dt 0.01294 at t = 8.95 (reference
  0.01286 at 8.97); viscous dissipation within 1 % of the reference to the peak; energy
  0.5 % low at the peak, 1.5 % at t = 10.5. `runs/tgv_re1600/tgv_n64` beside the repo.

**GPU memory, first round (2026-10-13): 64^3 Q3/Q2 fits one 80 GB H100; the limit is 69^3.**
Human: "find out how many dofs I can fit on a GPU", then "implement both".
- **Measured** (Taylor-Green defaults, ~10 steps, one H100): before, 56^3 (14.2 M velocity
  dofs) at 74.5 GB and 60^3 out of memory; after the two fixes below 64^3 at 63.9 GB,
  **69^3 (26.6 M velocity + 2.6 M pressure dofs) at 78.4 GB, 70^3 out of memory**; about
  3.0 kB per velocity dof.
- **The profiler has a gauge** (`Profiler::SetGauge`; `output.profile: sync` on a GPU
  installs `DeviceMemoryInUse`): three more CSV columns per region, what its first call
  left allocated, the same over all calls, and the peak while it was open.
- **Fix 1: 11 GB of GPU stack.** MFEM's `VectorDiffusionIntegrator::AssembleDiagonalPA`
  reserves 44 kB of stack per GPU thread on its first call (work arrays sized by
  `DofQuadLimits::MAX_*`), 11 GB on an H100 for the life of the process, independent of
  the problem. `LeanVectorDiffusionIntegrator` overrides only that diagonal. Written up
  for upstream in [upstream/mfem_vector_diffusion_diagonal.md](upstream/mfem_vector_diffusion_diagonal.md);
  **not reported to MFEM yet**.
- **Fix 2: the trapezoidal starter solver** is built on first use and released before
  the first BDF solver is built (not under OIFS or error control), so two whole operators
  and preconditioners never coexist; a resumed march never builds it.
- **Tests:** fast tier CUDA np 1 63/63 (without `nse_mms_test`, `tgv_nse_test`);
  `lean_vector_diffusion_test`, `profiler_test`, `pbj_grad_div_test` also at np 2. **Not
  run:** the fast tier at np 2, a CPU build, OIFS or adaptive stepping after fix 2 beyond
  the np 1 tier.
- **Where the rest goes at 64^3 (63.9 GB)**, candidates, none acted on: 14.9 GB in
  `case::setup` outside the time integrator, with no regions yet; vector-diffusion PA data
  13.5 GB (27 numbers per point where 6 would do, in the momentum form and again in
  `viscous_unconstrained`); the pressure LOR-AMG about 9 GB for 2.1 M unknowns (3 GB at
  setup, 5.8 GB at its first apply); 4.9 GB unattributed in `time_integrator::setup`; each
  momentum reassembly allocates 9.4 GB before the old form is freed.
- Feature costs at 32^3 (23.7 GB before the fixes): grad-div 1.1 GB; OIFS +4.4 GB;
  `a_pc: loramg` +21.7 GB; PBJ, the Schur grad-div term and diagnostics within 0.3 GB.

**Branch `profile-output` (2026-10-12, not merged): the profiler's results are saved.**
Human: "can we save out the profiler results so we know where the compute is being
spent?", ahead of the 3D Taylor-Green study.
- `apps/run_case` writes `<path>/<name>_profile.csv` next to the summary (every region:
  nesting path, calls, inclusive and exclusive wall seconds, max and min over the ranks),
  and the summary gets a `profile` section (top three levels). Deck key
  `output.profile: off | wall | sync`, default `wall`. `sync` waits for the GPU at every
  region boundary (`incns::DeviceSync`, a new nvcc TU `util/device_sync.cpp`), so GPU time
  is charged where it is spent. `Case::WriteProfile`, `Profiler::WriteCsv`/`SetSync`.
- Tests: `profiler_test` (the hook; the exact CSV), new `profile_output_test` W1 (a short
  run's file: nesting, one step region per step, children fit in their parent). Bridges-2,
  CUDA backend, np 1 and 2: pass, with `deck_test` and `run_case`; style clean. **Not
  run:** a CPU build; the Python bindings do not expose `output.profile`.
- **First 3D profile** (`tgv_re1600_n16` to t = 0.5, one H100, np 1; 18 steps, 112 s):
  87 % of the run is `inner_cg`, the inner CG of the Cahouet-Chabard Schur
  preconditioner (106 applies, 0.93 s each, for 32,768 pressure unknowns); 12 % is in
  `case::run` outside the steps (not broken down: the OIFS substeps have no region);
  assembly and the outer FGMRES are negligible. `wall` and `sync` agree to 0.3 % here and
  cost the same. **Next:** regions inside the inner solve (the LOR-AMG cycle, B M^-1 B^T)
  and around the OIFS substeps, then the same at 32^3: is 0.93 s per apply a fixed cost of
  a small problem on a GPU, or does it scale?
- **Answered (same day): the AMG smoother.** With regions inside the Schur preconditioner
  (`cc::lor_amg`, `cc::bmb`, `cc::gradient`, `cc::divergence`,
  `cc::velocity_mass_inverse`, `cc::pressure_mass_inverse`; `velocity_block::apply`,
  `schur_block::apply`; the time integrator's and Case's pieces; `ProfiledOperator`),
  `cc::lor_amg` was 88.6 % of an IMEX run: 86 ms per V-cycle at 16^3, 670 ms at 32^3. It
  was on the GPU, but smoother 8 (l1 symmetric Gauss-Seidel) runs there as triangular
  solves. With l1-Jacobi (18): 0.74 ms per cycle, the 16^3 run 16 s instead of 145 s, same
  7 outer iterations. Now `SymmetricAmgSmoother()` (18 on a GPU, 8 on a host) and
  `VerifyHypreAmgOnDevice` (aborts on any option outside hypre's GPU list), test
  `hypre_device_test`. After that fix the diagnostics were 77 % of the run (host element
  loops): rewritten as `FlowDiagnostics` (PA forms, device), test
  `Diagnostics.AssembledFormsMatchTheFreeFunctions`.
- **Grad-div on by default, point-block Jacobi under the velocity PCG** (human, same day).
  Deck: `physics.grad_div_h` (default 2) and `physics.grad_div_nu` (default 0),
  γ = c_h·h_K + c_ν·ν; the old `grad_div` / `grad_div_scale` deck keys are rejected.
  `solver.a_pcg_pc: auto | jacobi | pbj`, `auto` = PBJ with the grad-div nodal blocks when
  there is a grad-div term. New test `pbj_grad_div_test`. The TGV decks are IMEX at CFL 0.5
  (human: "I'd rather not use OIFS for this just imex").
- **First build and runs (jobs 49098413, 49098829; CUDA, one H100, np 1).** Built first
  time. Fast tier np 1: 59/64; `nse_mms_test` and `tgv_nse_test` time out at 900 s as on
  2026-10-10; `deck_test` failed on the stale deck reference; **`oifs_test` O2 fails**
  (OIFS error 0.0175 / 0.0453 against 1.2 x the IMEX error 0.0039: moved by today's
  defaults, cause not found); `pbj_grad_div_test` P1 failed (the test's host probing is
  suspected, P2 passes: PBJ 665 / 1699 CG iterations against Jacobi 744 / 2021). np 2 not
  run. Taylor-Green 16³ to t = 0.5: no grad-div 7 outer iterations, 3.0 s; γ = 2h 28
  iterations, 11.8 s, PBJ = scalar Jacobi (uniform mesh: the nodal coupling cancels);
  momentum reassembly 150 ms per Δt change against 3 ms. 32³: 3.6 s per step (55 s
  before the smoother fix). Hence, human 2026-10-13: γ in the Schur PC's mass term
  (`solver.schur_grad_div`) and γ cached per mesh -- written, see below for their state.
- **Second round (2026-10-13, jobs 49099073, 49099090, 49099146): built, tested, NOT
  merged.** Fast tier, CUDA, one / two H100s, without `nse_mms_test` and `tgv_nse_test`
  (they time out on a GPU; untested on this code): **np 1 62/62; np 2 61/62**.
  `nc_stokes_test` S4 fails at np 2: Cahouet-Chabard iterations 25 against 21 on the two
  meshes at step 0, limit 3 (human 2026-10-10), errors equal to 1e-8. Same kind of spread
  as the hypre GPU random-stream effect that limit was set for, but wider than seen
  before (the AMG smoother on a GPU is now l1-Jacobi). **Limit set to 4 (human
  2026-10-13); the test was NOT rerun with it, and the edit is not astyle-checked.**
  Taylor-Green 64^3 (job 49099148) ran out of GPU memory on one 80 GB H100 after the
  first step: it needs two GPUs or a smaller mesh. `oifs_test` O2 now pins `grad_div_h = 0` (with gamma = 2h the 8x8
  mesh spatial error falls 30x, IMEX 0.097 -> 0.0033, and OIFS at 4-8x the step loses on
  time error: 0.018 / 0.045). `pbj_grad_div_test` passes with host-pointer probing.
  Taylor-Green 32^3 to t = 0.5 (25 steps): defaults 12 outer iterations, 36.3 s;
  `schur_grad_div: false` 23, 73.5 s; `a_pcg_pc: jacobi` 12, 35.8 s; `grad_div_h: 0` 7,
  17.5 s. Momentum reassembly 15 ms (was 1.17 s).
- **Taylor-Green Re 1600, 32^3 Q3/Q2 to t = 20** (one H100, 39 min, 1087 steps, 15-16
  outer iterations; `runs/tgv_re1600/tgv_n32` beside the repo on Bridges-2): max -dE/dt
  0.01292 at t = 10.1, max viscous dissipation 0.01150 at t = 10.1 (reference 0.01286
  at 8.97): the peak is about 1.1 late, under-resolved as expected at 96 nodes. Not yet
  compared point by point (`plot.py`; the reference CSV is not fetched). 64^3 queued
  (job 49099148, 6 h limit, checkpointing; estimated ~10 h to t = 20).
- **STATE (superseded where the items above say so): written, NOT compiled, NOT run** (human: no runs until all of it is written).
  Only the profiling regions were built and tested (CUDA, np 1 and 2, 20/20) before the
  later edits. Uncompiled: `hypre_device`, `FlowDiagnostics`, the grad-div options,
  `AddNodalOffDiagonalPA`, the PBJ extension, `a_pcg_pc`, both new tests. Unformatted
  (`scripts/style.sh format`); `docs/deck_reference.md` not regenerated. **Expect Case-level
  results to move** with γ = 2h on by default (iteration counts, forces; pin
  `grad_div_h: 0` to reproduce old numbers). List what moves; do not re-bless `test/baselines.yaml`. The Python
  bindings do not expose `output.profile` or `a_pcg_pc`. Not pushed (human: wait until the
  grad-div work is done).

**Branch `mesh-census` (2026-10-12, merged): `apps/mesh_census`**, to check whether an
`amr.end_time` freeze of a refinement-only AMR run is safe. `amr/mesh_census` counts
elements per streamwise band; `ReplayMeshCensus` rebuilds a checkpointed run's mesh pass by
pass (as a restart does). Test `mesh_census_test` C1 (exact band counts on a box), C2
(replay = the run's final mesh).

**Restart fixes (2026-10-12):** a restart replaced the per-step history instead of
continuing it (`post/run_monitor` appends now; `benchmarks/common.py` drops rows a later
segment supersedes, using the step's start t - dt); a restart with forces aborted in the
summary before its first step (`Case::WriteSummary` gated forces on StepCount instead of an
available force). Tests `run_monitor_test` R5, R6 (R6 aborts without the fix: checked).

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
  One run on one AMR level: not a mesh-converged answer.
- **Fast tier on the Mac (2026-10-11, `scripts/test.sh cpu -L fast`, clang, M3 Air):**
  180 of 180 pass (0 failed), 269 s of test time (4 min 31 s wall), after the three
  clang fixes. 3 of the 180 were *skipped*, as designed: `benchmark_plot_tgv_re1600`,
  `_channel` and `_dfg_3d` need `python3 benchmarks/fetch_references.py` (exit 77, not
  fetched on this machine). This closes the macOS bring-up of `scripts/setup.sh` ->
  build -> doctor -> case -> fast tier. **Not run on the Mac:** the slow tier,
  `scripts/asan.sh`, the Python tests (`cpu-python` preset), `scripts/docs.sh` (Doxygen),
  `scripts/debug_device.sh`.
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
2. When MFEM fixes PR #5297's regression: drop the pin in both `spack.yaml` files.

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
- **dt sawtooth on AMR meshes (open):** once shedding starts, every ~0.13 time units the CFL
  rate jumps ~1.5× in ONE step → the controller cuts dt ~1.9e-2 → ~1.0e-2 and regrows ×1.2.
  Forces unaffected (C_D jitter ~1e-4); ~10–20% extra steps. Likely a fast mode at its
  stability limit at CFL 0.5 on the AMR (hanging-node) mesh — conforming DFG meshes were
  stable to 0.7. Diagnose with the CFL location now printed; a 0.4 target likely removes it.
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
