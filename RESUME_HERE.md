# RESUME HERE — read this first

Left 2026-07-24 ~22:30 by the user logging off with a job queued.

## The one thing to do

```bash
cd /ocean/projects/mth260019p/schneier/bessemer
cat build/nse_parallel/report.txt
```

That single file is the whole result. If it does not exist yet, the job has not
run — check `squeue -u $USER` and `sacct -j 42595862 --format=JobID,State,Elapsed,ExitCode`.

Everything else below is context for reading it.

## What was queued

- **Job `42595862`**, `scripts/nse_parallel.sbatch`, submitted 2026-07-24T22:23.
- `GPU-shared`, 4x h100-80, 4 tasks x 8 cores, 2h walltime.
- Branch **`sprint22-tgv-oracle`**, on top of `main` @ `770f024`.
- It was PENDING at log-off behind a deep queue (all 80 h100 cards allocated
  but two; 136 h100 jobs pending), so it may have started hours later.

Runs `convection_test`, `nse_mms_test`, `tgv_nse_test` at **np in {1,2,4}** =
9 runs. Per-run logs in `build/nse_parallel/<test>_np<N>.log`; Slurm stdout in
`nse-par-<jobid>.out`.

## Why: the actual question being answered

**Everything in the Sprint 2.2 sign-off ran at np1.** Convection does shared-face
assembly, so np2/np4 is the first real parallel exercise of that path — np1
cannot see a rank boundary at all. This is open item (1) from the roadmap.

Riding along: `tgv_nse_test` is **new and has never been compiled**. A compile
error on this run is a live possibility, not a surprise.

## How to read report.txt

One greppable line per run:

```
RESULT nse_mms_test np=4 rc=0 verdict=PASS ran=3 passed=3 failed=0 wall_s=210
```

and `OVERALL GREEN` / `OVERALL RED` at the bottom. Also check the physics lines
hoisted into the report — **a bare PASS is not enough**:

- `[ NSE MMS ]` lines must still show rates **~2.0** at every rank count. np1
  gave 1.978 -> 1.990 (2D) and 1.980 -> 1.990 (3D). A pass with degraded rates
  at np2/np4 is exactly the failure this job exists to catch.
- `[ TGV NSE ]` lines: `p_norm` ~1.42 with convection on, `u_err` small,
  `worst_ke`/`worst_eps` small, `worst_div` ~0.

## Expectation setting — do NOT assume a red means a real bug

- `convection_test` / `nse_mms_test` are proven code; a failure there IS
  interesting (likely a genuine rank-boundary issue).
- `tgv_nse_test` is unproven. Its thresholds (`u_err < 1e-4`, energy drift
  `< 5e-3`) are **reasoned estimates, never measured**. If one trips, read the
  printed numbers before touching the solver — it is more likely a bound set too
  tight than a solver defect. The numbers print unconditionally for this reason.
- Per CLAUDE.md: do not loosen a tolerance just to make it pass. Justify it or
  find the real cause.

## State at log-off

`sprint22-tgv-oracle` (5 commits, unpushed, off `main` @ `770f024`):

| commit | what |
|---|---|
| `e7cf846` | TGV NSE oracle: `test/tgv_nse_test.cpp` + analytic KE/dissipation in `src/exact/tgv2d.hpp` |
| `0419090` | `scripts/run_tgv_oracle.sh` (interactive runner, unused so far) |
| `6f8be7f` | `scripts/nse_parallel.sbatch` |
| `b15e0fd` | partition fix: `GPU` is whole-node only -> `GPU-shared` |
| `2cdbb01` | `scripts/nse_parallel_np12.sbatch` (2-GPU np{1,2} variant; user chose not to run it) |

`main` itself is still `770f024` and still **5 commits ahead of origin,
unpushed** — pre-existing, unrelated to this branch.

## After the report is read

Delete this file — it is a session breadcrumb, not project documentation. The
durable record belongs in the memory file `bessemer_gpu_verification_plan.md`.
