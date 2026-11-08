# FDA nozzle: what to run on the GPU machine

Written 2026-11-08 by the cloud thread "Run FDA nozzle Re 500 smoke test" for the agent
working in Mike's local session (cluster and GPU access). That session cannot see the
project's other threads, so this file holds everything it needs. Mike said go on the
Re 500 smoke test in that thread; AGENTS.md rule 3 still applies to you: before each job,
state its resources, walltime and cost and get Mike's yes in your session.

## What is new and where it is

- **`profile: pipe`** for deck velocity boundaries (src/bc/deck_boundary,
  src/config/parameters): Poiseuille flow in a circular pipe, 2 u_mean (1 − (r/R)²) along
  `axis` through `center`, ramped by `time_profile`. With `perturbation: eps > 0` it adds
  six travelling azimuthal modes to the axial velocity (azimuthal numbers 1, 2, 3, 1, 2, 3,
  frequencies k f/6 for f = `perturbation_frequency`, each of amplitude eps u_mean / √6,
  phases from `perturbation_seed`). They vanish at the wall and carry no net flux. The
  deck had only planar profiles (constant, parabolic) before, so a circular inflow needed
  this. Test: `deck_boundary_test` B5. Without a perturbation the data is separable in
  time, which is the cached fast path. With one it is projected on the host at every
  solve and every OIFS stage: watch for that in the profile (stage 2).
- **benchmarks/fda_nozzle/**: the mesh script (`fda_nozzle_mesh.py`, gmsh), a 148-hex
  test mesh (`fda_nozzle_tiny.msh`), four decks, `run.sbatch` (Delta header; Bridges-2
  header in its comments), README.md, this file.
- Branch / PR: see the PR "FDA nozzle: pipe inflow profile, decks and HPC handoff". Use
  `main` once it is merged; until then check out its branch.

## The case

Lengths in throat diameters (d = 1), throat mean velocity U_th = 1, so the throat
Reynolds number is Re = 1/ν and time is in d/U_th. Inlet pipe D = 3 from z = −27.67 (12
long), a 20° cone, the throat (10 long), the sudden expansion at z = 0, the outlet pipe to
z = 30. Inlet mean velocity (d/D)² = 1/9. Boundaries `inlet`, `outlet`, `walls`.

Level-0 mesh: 56,320 hexes, geometry order 3, minimum scaled Jacobian 0.55.

| order | velocity + pressure DOFs | GPU memory (estimate) |
|---|---|---|
| Q3/Q2 | 4.63M + 0.46M (measured) | ~5–8 GB |
| Q6/Q5 (the default) | ~43.5M | ~45 GB: one 80 GB H100, or two 40 GB A100s |

The memory figures are scaled from gpu_memory.md's curved-mesh numbers (about 1 kB per
DOF for OIFS on curved meshes), not measured on this mesh. The smoke test measures them.

## Step 0: build and mesh (login node, then one short job)

```sh
git fetch && git checkout <the PR branch, or main once merged>
scripts/build.sh cuda                       # in a job if the login node can't compile
pip install --user gmsh                     # needs libGLU; a conda gmsh works too
mkdir -p runs/fda_nozzle
python3 benchmarks/fda_nozzle/fda_nozzle_mesh.py --level 0 --order 3 \
    -o runs/fda_nozzle/fda_nozzle_l0.msh    # ~10 s, 126 MB, prints 56320 hexahedra
```

The decks expect the mesh at `runs/fda_nozzle/fda_nozzle_l0.msh` and run from the
checkout's root (`run.sbatch` does that). `runs/` is git-ignored.

## Stage 1: Re 500 (laminar)

Run these in order. Stop and report to Mike after each one that fails a check.

**1a. Smoke test: `fda_nozzle_re500_smoke.yaml`.** Q3/Q2 to t = 5, one GPU, ParaView
every 50 steps, `output.profile: sync`.
`sbatch -n 1 --gpus-per-node=1 -t 0:30:00 run.sbatch fda_nozzle_re500_smoke.yaml`.
Estimate: 2–10 min including setup.
Checks:
- `runs/fda_nozzle/re500_smoke/re500_smoke_summary.json`: `status: ok`.
- The history CSV: outer iterations per step stay bounded (no climb to `solver.max_iter`),
  no step cuts, dt settles after the ramp (t > 2) near the CFL-2 step. I estimated
  dt ≈ 0.04 for Q3 from the mesh spacing; the run's own dt is what counts.
- The profile CSV (`re500_smoke_profile.csv`): peak device memory and time per step.
  Report both: they size everything below.
- ParaView: a laminar jet leaves the throat along the axis; the inflow profile is
  parabolic and axisymmetric; nothing odd at the step corner or the outlet.
- `python3 benchmarks/flow_stats.py halves runs/fda_nozzle/re500_smoke/re500_smoke_summary.json`
  runs (the averages here are over a transient, so only check they exist).

**1b. Time to steady state at Q3: `fda_nozzle_re500_q3.yaml`.** Q3/Q2 to t = 1000, one
GPU, checkpoints every 500 steps, statistics from t = 900.
Why so long: the run starts from rest, and the inlet pipe's velocity profile develops on
its viscous time. Its slowest mode decays as exp(−t/195) (R²/(ν j₀²), R = 1.5, ν = 0.002),
so the inlet pipe takes t ≈ 900 to settle to 1 %. The throat (time scale ~22) and the jet
should settle much sooner, and finding out when is the point of this run.
Estimate: ~23,000 steps at about 0.2 s (H100) to 0.35 s (A100) each, 1.3–2.3 h; ask for
3 h.
Checks:
- `status: ok`.
- The diagnostics CSV and the stats: when the kinetic energy, the centreline velocity in
  the jet and the reattachment point stop changing (< 0.1 % over 50 time units). Record
  the time for the throat/jet separately from the inlet pipe's.
- If it is still drifting at t = 1000, restart with a larger t_final:
  raise t_final and the statistics' start_time in the deck, then
  `run.sbatch fda_nozzle_re500_q3.yaml runs/fda_nozzle/re500_q3/chk` (the script appends
  `restart:` to a copy).
- `flow_stats.py halves` on the summary: the half-window differences should be small
  (steady flow).
- `flow_stats.py centerline .../re500_q3_stats_centerline.csv` and
  `flow_stats.py wall .../re500_q3_stats_wall_jet.csv` (and `_wall_far`): the jet's
  centreline decay and the reattachment point of the outlet recirculation.
- That time sets t_final and start_time for 1c.

**1c. Re 500 at the defaults: `fda_nozzle_re500.yaml`.** Q6/Q5, ~43.5M DOFs. One H100, or
two A100s (`sbatch -n 2 --gpus-per-node=2 ...`). Before submitting, shorten t_final
(1000 in the deck) and flow_statistics.start_time to what 1b showed.
Estimate: dt ≈ 0.013 (Q6 spacing at CFL 2), ~7,700 steps per 100 time units, about
1.3–1.6 s each: **~3 h per 100 time units** (to t = 1000, ~30 h). I scaled the step
cost from the curved cylinder's measured Q6 steps (throughput.md, ~31 ns per DOF per step on
one H100), not measured here. Split it into jobs within the queue limit and continue each
from the checkpoint.
Checks: as 1b, plus Q3 against Q6. The centreline velocity, the wall pressure and the
reattachment point should agree to the plotted precision if Q3 is already converged. If
they don't, Q6 is the reference and Q3 was not resolved.

**Comparison with FDA's data.** FDA's PIV and pressure data are at
https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks (not in the repo; no
redistribution terms checked). The decks sample the centreline, radial profiles at
z/d = −22, −16, −10.5, −5, −2, 0, 2, 4, 6, 8, 10, 12, 15, 20 and the wall in 0.25 d bins.
I took those stations from memory of Hariharan et al. 2011 (FDA's sudden-expansion
locations, −0.088 to 0.080 m with d = 4 mm). Check them against the data release before
comparing. Normalize with `flow_stats.py normalize --u-ref 1 --l-ref 1`. FDA reports
velocities in m/s at its flow rate: U_th = Re·μ/(ρ d) with its blood analogue's
properties (ρ 1056 kg/m³, μ 3.5 mPa·s).

**Report to Mike after stage 1:** for each run, the status, wall time, steps, dt range,
iterations per step, peak GPU memory, time per step; the time to steady state; Q3 against
Q6; and against FDA's data where it is in hand. Record the results in a "Results" section of
benchmarks/fda_nozzle/README.md (there is none yet; create it, as the other benchmarks
have) and in docs/status.md. Running cases is not development work, but benchmark results
go there (benchmarks/README.md).

## Stage 2: Re 2000 inflow sweep (only after stage 1 passes and Mike says go)

The question: how far downstream the jet breaks down as a function of the inflow
disturbance level. This is the open Re 2000 question between Sánchez Abad 2020 (Nek5000:
needs inflow eddies) and Jain 2020 (LBM: doesn't).

- Deck: `fda_nozzle_re2000.yaml` (ν = 5e-4, Q6/Q5 defaults, perturbation 0.01, to t = 400,
  statistics from t = 100). Make one copy per member, changing `perturbation`,
  `output.path`, `output.name` and `checkpoint.path` together:
  ```sh
  for eps in 0 0.001 0.003 0.01 0.03; do
    tag=re2000_eps$(echo $eps | tr . p)
    sed -e "s/perturbation: 0.01,/perturbation: $eps,/" \
        -e "s/re2000_eps0p01/$tag/g" benchmarks/fda_nozzle/fda_nozzle_re2000.yaml \
        > benchmarks/fda_nozzle/fda_nozzle_$tag.yaml
  done
  ```
  The amplitude set is my suggestion. Mike may want a different one, and a repeat with
  another `perturbation_seed` for the most sensitive member.
- First member alone (eps = 0.01): check that the flow is unsteady and that the half-window
  comparison converges. Check the profile for the host projection of the perturbed inflow
  (the boundary-condition and `oifs` regions); if it is a large share of a step, tell
  Mike: separable multi-mode data would be a library change.
- **Resolution is the first open question here.** The level-0 jet region has 0.15–0.4 d
  cells, about 0.03–0.07 d between Q6 nodes. Whether that resolves the shear-layer
  breakdown at Re 2000 is unknown. Run one member at Q5 as well as Q6 (or on the level-1
  mesh) before trusting the sweep.
- Cost estimate: same dt as Re 500 at Q6 (same velocities), ~30,000 steps to t = 400,
  11–14 h per member on one H100 or two A100s; five members ~60–70 H100-hours (twice
  that in A100-hours). Mike's call.
- The inlet pipe's viscous time is 4× Re 500's (780), so from rest it never develops
  within the run. The 9:1 contraction dominates what the throat sees, but report the
  inlet-station profile so Mike can judge. An initial condition with the pipes in
  Poiseuille flow would fix this; it is not built (a library change).
- Output per member: the centreline (breakdown location: `flow_stats.py centerline`),
  the profiles, the wall bins, the summary's half-window comparison.

## Stage 3: Re 6500 p-ladder (later; Mike decides)

The pitch is the first grid-converged Re 6500 result with error bars: the level-1 mesh
(450,560 hexes; `--level 1`) at Q3, Q4, Q5: about 40M, 99M and 198M DOFs, which needs
1, 2 and 4 H100s (or 2, 3–4 and 6–8 A100s; more than 4 ranks needs Mike's yes, rule 3).
Rough cost at Q5: dt ≈ 0.009, ~1.6 s per step on 4 H100s, ~10 h per 200 time units.
Unresolved before it starts:
- **Inflow.** The inlet pipe is at Re ≈ 2,170 there, transitional. Fehn, Wall &
  Kronbichler (arXiv 1811.06752) found a turbulent precursor pipe essential, and bessemer
  has no periodic file meshes for one. Use the pipe profile with a perturbation for now and
  treat the inflow as a studied parameter, as in stage 2.
- Mesh sizing against the Kolmogorov scale from the stage 2 runs.

## Files a run writes (in `runs/fda_nozzle/<name>/`)

`<name>_summary.json` (status, steps, dt, iterations, flow statistics' half-window
comparison), `_history.csv` (per step), `_diagnostics.csv` (energy, dissipation, div u),
`_profile.csv`, `_stats_{centerline,inlet,throat,outlet,wall_upstream,wall_jet,wall_far}.csv`
(+ `_half1/_half2`),
`chk/` (rolling checkpoint). Report from the summary, not the log (AGENTS.md).

## What the cloud thread checked (CPU only, 4 cores, scratch build)

Scratch build: MFEM v4.10 + #5464, apt hypre 2.28 and Open MPI, not the Spack toolchain.
- `deck_boundary_test` (B5, the pipe profile) at np 1/2/4 and `benchmark_setup_test`
  (B1: every deck here loads) at np 1/2, and `deck_test` at np 1: pass.
- The coarse nozzle (992 hexes, Q2/Q1, Re 500, np 4) to t = 8: `status: ok`, 67 steps, no
  cuts, 9–12 outer iterations per step. The inlet profile at z = −27.6 matches
  2 u_mean (1 − (r/R)²) to 1e-3. Downstream it is still nearly a plug at t = 3–5: the
  start from rest, as stage 1b's note explains.
- The same with `perturbation: 0.05` to t = 3: `status: ok`, 40 steps.
- The level-0 mesh at Q3/Q2 (the smoke deck, np 4): loads with its boundary names, 56,320
  elements, 4,628,739 velocity + 460,545 pressure DOFs. Three steps of dt 0.001 ran `ok`
  (34, 3 and 12 outer iterations), 8.3 GB of host memory. CPU step times (15–130 s) say
  nothing about the GPU.
- The flow statistics sets find their points. Earlier 0.25-wide wall bins upstream were
  empty (NaN) where the inlet cells are 1 long, so the walls are now three sets with bins
  sized to the cells. One flow_statistics message reported 9 sample points outside the
  mesh (NaN), 30 after the station change; which ones was not checked. Check the CSVs in
  the smoke run.
- **Not done:** no run past the ramp on the real mesh, so the dt, iterations per step and
  time to steady state are estimates (Mike stopped the 40-minute CPU run). Nothing on a
  GPU.
