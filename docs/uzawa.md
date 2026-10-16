# The Schur complement (Uzawa) outer method

`solver.outer: schur` solves each step's saddle-point system by flexible CG on the
pressure Schur complement instead of FGMRES on the coupled system. The proposal it
implements is [design/uzawa_schur_cg_spec.md](design/uzawa_schur_cg_spec.md) (human
2026-10-16: "implement the uzawa stuff here as an option for the solver"). The coupled
method stays the default. Whether Schur should replace it is the study at the end of this
page, which needs GPU runs.

**State (2026-10-16): builds and passes its tests in CI (GitHub Actions, CPU, np 1/2/4);
never run on a GPU.** Nothing on this page has been timed, and the memory figures in §4
are counts of the vectors the code allocates, not measurements. The code was written in a
cloud session that cannot build Bessemer: CI is its only build so far.

## 1. What it solves

Both methods solve the same eliminated system on the Cahouet-Chabard path. The system is
`[A B^T; B 0] [u; p~] = [f; g]` on the internal pressure p~ = -p. A is the constrained
momentum block (identity rows on the essential velocity dofs). B is the eliminated
divergence, so B^T d is zero on those rows. The Schur complement is S = B A^-1 B^T. Each
application of S is one B^T, one velocity solve and one B
(`SchurComplementOperator`, `src/solver/schur_complement.hpp`). The pressure residual is
r = B u - g with u = A^-1 (f - B^T p).

`SchurSolver` (`src/solver/schur_solver.hpp`) runs each step as follows:

1. **The tolerance** is FGMRES's, τ = max(atol, rtol ‖b - K x₀‖), using the same
   `solver.rtol`, `atol` and `tolerance_reference`. It is split into τ/√2 for the
   pressure residual and τ/√2 for the momentum residual. Both are checked on true
   residuals, so the final coupled residual ‖b - K x‖ is at most τ. Both methods stop on
   the same quantity, and `SolveStats::final_residual` and `tolerance` report it for
   either method.
2. **System 1** computes u = A^-1 (f - B^T p₀) from the warm start (u₀, p₀). The proposal
   solves A u₀ = f first and then applies S once to p₀. Folding p₀ into the right-hand
   side saves that velocity solve.
3. **The pressure projection** applies when `initial_guess: project` is set (below).
4. **A flexible CG cycle** on S δp = r follows. u is carried along with p
   (u -= γ v, where v = A^-1 B^T d is the velocity of the application), so u tracks
   A^-1 (f - B^T p) without another solve.
5. **System 3** computes u = A^-1 (f - B^T p), warm-started from the carried u.
6. **The true pressure residual** of (u, p) is checked. If inexact inner solves let it
   drift above τ/√2, another cycle starts (`solver.uzawa.max_cycles`, default 3).

Velocity solves outside S (Systems 1 and 3) are explicit corrections du = A^-1 r of the
true momentum residual. At most 4 passes are made, each to max(1e-13, ½ target/‖r‖)
relative, until ‖r‖ ≤ τ/√2. Every velocity solve, inside S or outside, uses one CG
(`VelocityCG`, `src/solver/velocity_cg.hpp`): MFEM's CG algorithm and stopping test, but
working in its right-hand side (§4). Its preconditioner is the fixed linear one inside the
coupled path's velocity PCG: Jacobi or point-block Jacobi under `a_pc: jacobi_pcg`, and
otherwise the coupled path's velocity PC itself.

## 2. Options (deck `solver.outer`, `solver.uzawa.*`)

| Key | Default | Meaning |
|---|---|---|
| `solver.outer` | `coupled` | `schur` selects this method. It needs `solver.schur: cc` or `laplacian_legacy`, and not the rotational form. |
| `uzawa.inner_tolerance` | `relaxed` | Relative tolerance of the velocity solve inside S: `tight` (`eps_tight`), `loose` (`eps_loose`) or `relaxed`: min(`eps_max`, max(`eps_tight`, `relax_c` τ_p/‖r‖)). |
| `uzawa.eps_tight` / `eps_loose` / `eps_max` / `relax_c` | 1e-10 / 1e-3 / 1e-2 / 1 | The constants above. |
| `uzawa.inner_max_iter` | 500 | Iteration cap of each velocity solve. Hitting it is counted (`uzawa_schur_cap_hits`). |
| `uzawa.max_cycles` | 3 | CG cycles per solve (step 6 above). A fixed `loose` inner tolerance leaves a true-residual gap of about `eps_loose` times each cycle's starting residual, so a tight `rtol` needs more cycles with it (CI, rtol 1e-10: 3 cycles fell short, 62 iterations against relaxed's 30). |
| `uzawa.history` | false | Record every iteration (see the log below). |
| `solver.log_interval` | 0 | Per-step solver log for either outer method (below). |

The `solver.max_iter` cap applies to the Schur iterations summed over the cycles.

## 3. Where it departs from the proposal

- **Flexible CG, not plain CG, and nothing else.** The proposal's plain CG needs a fixed
  linear preconditioner. Bessemer's Cahouet-Chabard applies its Poisson part by a
  fixed-count inner CG (`solver.n_inner`, case default 2), which is nonlinear in its
  input. Flexible CG (Notay 2000, the Polak-Ribière β = (z_{k+1}, r_{k+1} - r_k)/(r_k, z_k)
  computed as -γ (z_{k+1}, q_k)/(r_k, z_k)) needs no extra vector here, and it reduces to
  plain CG when the preconditioner is fixed and linear, so plain CG is not offered (human
  2026-10-16: "drop both"). A pressure-space FGMRES is not offered either: it would only
  serve the rotational form, where the coupled FGMRES already works.
- **The proposal's CahouetChabardSolver already exists** as `CahouetChabardSchurPC`, a
  pressure-space `mfem::Solver`. The Schur method uses it unchanged, through the
  `MonitoredSolver` wrapper, so its counts and time appear in `SolveStats`. On singular
  domains, the Schur method projects its output onto the mean-zero space, because CC
  projects its Poisson solve but not its mass term.
- **The initial guess** (human 2026-10-16, "ok ya that makes sense"). Under the coupled
  method, `initial_guess: project` keeps 2k system-sized vectors (16 at the default
  k = 8), more than the coupled Krylov basis Uzawa removes. Under `outer: schur` the
  projection is pressure-only: a basis of the pressure increments Δ = p - p_n of
  converged solves and their images SΔ = r_n - r. Both residuals are true residuals the
  solve computed anyway, so the basis costs no extra velocity solve. The guess is
  p_n + X Wᵀ r_n, with residual r_n - W Wᵀ r_n. That is 2k + 2 pressure vectors and no
  velocity vector. It is reset on every refresh (Δt or ν) and rotation update, as the
  coupled one is, and checkpointed (`projection_pressure: 1` in the checkpoint's meta). A
  coupled run's checkpoint restarted under Schur starts the pressure basis afresh, and
  the other way round. `previous` and `zero` work as before.
- **The rotational form** makes A, and so S, nonsymmetric, which CG cannot handle.
  `outer: schur` with `convective_form: rotational` stops at setup and points to
  `outer: coupled`.

## 4. Memory

Memory is the reason for this method (human 2026-10-16: "the whole point of moving to
this is to optimize memory usage"), so the Schur path allocates only what it applies.
Units as in `/mnt/project-files/memory/gpu_dof_ceiling.md`: B per velocity DOF, double
precision. At 64^3 Q3/Q2 (21.2 M velocity, 2.1 M pressure DOFs) a velocity vector is
8 B/vdof, a pressure vector 0.79, a system vector 8.8; 1 GB is 47 B/vdof.

What the Schur path holds besides the state, the operators and the preconditioners
themselves (case defaults: Cahouet-Chabard, `jacobi_pcg` with point-block Jacobi, the
projection at k = 8, `tolerance_reference: step`):

| Vectors | Count | B/vdof |
|---|---|---|
| Pressure: r, z, d, q of the flexible CG | 4 p | 3.2 |
| Pressure projection: 2k basis vectors, p_n, r_n | 18 p | 14.2 |
| Schur operator: B^T d (the CG's residual), v = A^-1 B^T d | 2 u | 16 |
| Velocity CG: direction, P r / A d (also Cahouet-Chabard's B M_v^-1 B^T scratch) | 2 u | 16 |
| **Total** | | **49** |

Before this was trimmed (5b55bf9) the same path held 115 B/vdof, and 139 for a moment at
every refresh. What went:

| Removed under `outer: schur` | B/vdof |
|---|---|
| The coupled velocity PCG (3 velocity vectors; the Schur method runs its own CG on the PCG's preconditioner) | 24 |
| The block preconditioner (a velocity and a pressure scratch vector) and the velocity-block monitor | 8.8 |
| The step reference's system-sized residual, and its operator application every step: under Schur with `previous` or `project` the reference is the Schur solver's own initial residual, computed in the vectors above | 8.8 |
| One of the velocity CG's three vectors: `VelocityCG` overwrites its right-hand side, the Schur operator's B^T d, with the residual (MFEM's CG copies it) | 8 |
| Cahouet-Chabard's own two velocity vectors for B M_v^-1 B^T: it runs between velocity solves, so it borrows the velocity CG's two (`CahouetChabardSchurPC::BorrowVelocityScratch`) | 16 |
| A new velocity CG at every refresh, built before the old one was freed: the CG is now re-pointed, not rebuilt | 24 at that moment |

Four velocity vectors is the floor for a nested preconditioned CG that carries u along
(the CG's residual, direction and P r, and v for the update of u). With
`initial_guess: extrapolate` the coupled path's two system vectors (x_{n-1}, and a
temporary each solve) and the step reference's residual remain; it is not the default.

For comparison, the coupled method holds for the same roles its FGMRES basis (105-282
B/vdof by `gpu_dof_ceiling.md`, from fast settings to the full restart length), the
projection's 2k system vectors (141 at k = 8), and 66 B/vdof of velocity PCG, block
preconditioner, Cahouet-Chabard scratch and the step reference's two system vectors
(`step_ax_`, `step_r_`). So the Schur method should save roughly 260-440 B/vdof against
the coupled method with the projection, about 18-29 % of today's measured 1,500 B/vdof
(5.6-9.3 GB at 64^3). That is arithmetic, not a measurement: `study.sh capacity` measures it.
Everything else (form data, the pressure AMG, the velocity PC, the history) is the same
for both methods, and is where the rest of the memory is.

## 5. The solver log

`solver.log_interval: N` writes `<output.name>_solver.csv` every N steps, for either outer
method. Each row has:

- the outer method, convergence, outer iterations, the final coupled residual and its
  tolerance;
- the coupled block PC's velocity-PC counts and time, and the Schur PC's applications and
  time;
- for Schur: cycles, velocity solves outside S and their iterations, S applications and
  the iterations inside them, cap hits, inner time, and the projection's start and guess
  residuals;
- the solve and step wall time;
- the device memory in use after the step (GB, the largest over the ranks; -1 on a host
  backend).

With `uzawa.history: true`, `<output.name>_uzawa_history.csv` gets one row per Schur
iteration with the recurrence residual, the inner tolerance and iterations, and the
coupled residual of the carried iterate. That last one costs one momentum application
each, so turn history off when timing.

A restart truncates both files (a new process starts them again). The coupled FGMRES's
Gram-Schmidt now has its own profiler region (`fgmres::orthogonalize`), shown as
`bench.py`'s `gram_schmidt` phase.

## 6. Tests (`test/schur_solver_test.cpp`; pass in CI at np 1, 2 and 4)

- **U1**: S is symmetric and its output is orthogonal to the constants.
- **U2**: Schur = coupled to 1e-6 in u and p. The domains are enclosed, periodic, and a
  channel with an outflow (no null space), each with grad-div 0 and 1, under CC and under
  `laplacian_legacy` (each against a coupled solve with the same Schur PC). Each solve's
  reported coupled residual must meet its tolerance.
- **U3**: tight, loose and relaxed inner tolerances give the same solution.
- **U4**: the pressure projection guesses a 1.5x-scaled system exactly.
- **U5**: a Taylor-Green NSE march through `Case` at case defaults, Schur against
  coupled.
- **U6**: a restart with the checkpointed pressure basis matches the uninterrupted run
  (`ReproTol`).
- **U7**: `VelocityCG` takes MFEM's `CGSolver`'s iterations and gives its solution, and
  leaves the residual in its right-hand side.
- **U8**: a Δt refresh of a Schur solver (CG re-pointed, point-block Jacobi updated in
  place) gives a fresh solver's answer, with point-block and scalar Jacobi.

## 7. The study (needs GPU runs; nothing measured yet)

`bench/uzawa/study.sh` runs the proposal's Phases 0, 2 and 3 as `bench/throughput/bench.py`
runs on one GPU at np 1. **Run it inside an allocation you hold: it submits no job.**
`bench/uzawa/compare.py A.json B.json` prints the step-time ratio, iterations, work inside
the solve, residual against tolerance, memory, energy and divergence for a pair, plus the
decision rule's verdict.

```sh
bench/uzawa/study.sh phase0     # coupled baseline (+ phase profile: Gram-Schmidt)
bench/uzawa/study.sh phase2     # correctness: rtol 1e-10, then case defaults
bench/uzawa/study.sh inner      # tight / loose / relaxed; relax_c; laplacian_legacy
bench/uzawa/study.sh guess      # previous vs project (coupled 2k system vs pressure-only)
bench/uzawa/study.sh dt         # dt x0.5 and x2 around the template's
bench/uzawa/study.sh order      # Q5/Q4 and Q7/Q6 at about the Q3 run's DOFs
bench/uzawa/study.sh channel    # Re_tau 180 replay from its frozen restart
bench/uzawa/study.sh capacity   # largest TGV mesh per method (88^3 and up)
```

**Decision rule (the proposal's):** Schur becomes the default only if its step is within
1.2x of coupled at a matched coupled residual and its peak memory is at least 25% lower.
The coupled path stays behind the switch either way. Not covered by the script: the
square cylinder with AMR (outflow, CC's weakest case: `cases/square_cylinder_re200.yaml`
with `solver.outer: schur`, CPU np 4, about 40 min) and Re_tau 550 (no deck yet). The
proposal's Olshanskii branch (rotational form, pressure FGMRES) is out of scope: the
rotational form stays on the coupled method.
