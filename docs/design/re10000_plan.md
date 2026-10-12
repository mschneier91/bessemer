# Re = 10 000 square cylinder (2D): plan

Status: drafted 2026-10-12 (agent, at the human's request), replacing the earlier
estimates in docs/status.md and docs/design/re1000_convergence_plan.md §7. Nothing here
has been run on Bridges-2; every job needs the human's yes (AGENTS.md rule 3).
Files: `benchmarks/square_cylinder_re10000/` (decks, job scripts, protocol).

## 1. The question, and what it is not

The flow past a square cylinder at Re = U D / ν = 10 000, **in 2D on purpose** (human):
how its statistics (C_D mean, C_L rms, the spectral peaks of C_L) converge with resolution,
continuing the Re 1000 study (re1000_convergence_plan.md) with the same method, decks
and analysis.

It is not a model of the physical flow: at this Re the real wake is three-dimensional
and turbulent. 2D simulations of bluff bodies over-predict drag and lift at such Re,
traced to the larger Reynolds stresses of a 2D wake (Mittal & Balachandar, Phys. Fluids 7
(1995) 1841). Physical references are 3D: the square-cylinder DNS at Re 22 000 (Trias,
Gorobets & Oliva, Comput. Fluids 123 (2015) 87; 323 M grid points, ~80 shedding cycles)
and experiments (St ≈ 0.13 for Re ≫ 1000). The closest physical 2D flows are soap films,
measured for cylinder drag at Re 35-560 and turbulence up to ~10^4.

**What has been done in 2D.** A quick literature search (2026-10-12) found resolved 2D
square-cylinder simulations up to Re ~1000 (e.g. Saha, Muralidhar & Biswas, Comput. Fluids (2000), at Re 100;
Ansumali et al. 2004, entropic lattice Boltzmann, Re ≤ 1000), and 2D simulations at
Re ~10^4-2×10^4 only with turbulence models (2D LES: Bouris & Bergeles, J. Wind Eng. 80
(1999) 31, Re 22 000; many 2D URANS studies). A resolved 2D simulation at Re 10^4 with a
resolution-convergence study and error-barred statistics was not found. That is not
proof that none exists: a real literature review (citations of the above, the 2D
turbulence literature) comes before claiming novelty.

## 2. Resources (human, 2026-10-12)

- **Bridges-2: 5 000 core-hours, CPU only** ("I don't want to use GPUs for the 2D stuff").
- The desktop (8 cores; 4 ranks without asking) and the MacBook Air: free, slow.

## 3. Inputs: measured, and what they extrapolate to

Measured (desktop, Re 1000 L2 at 4 ranks; Re 200 OIFS deck):

| Quantity | Value |
|---|---|
| unknowns per element (Q3/Q2, 2D) | 21.8 |
| cost per element per step (AMR mesh, OIFS CFL 2) | 0.37 ms at 4 ranks = ~1.5 core-ms |
| solver iterations per step | 13-17, flat from Re 200 to 1000 |
| dt at `min_size` 0.05 (CFL 2) | 0.010-0.014 |
| memory | ~44 KB per element (Mac, 12 k elements) |
| sampling error | 2-5 % at 70-100 time units of averaging (Re 1000, L1) |

Scaling to Re 10 000 (2D): boundary layers, shear layers and the enstrophy-dissipation
scale shrink like Re^-1/2, so the cells there shrink by sqrt(10) = 3.16: elements ×10,
dt /3.16. The AMR indicator is the velocity change across an element, so a level keeps
its Re 1000 **tolerance** (that already refines steeper gradients) and divides its
`min_size` floor by sqrt(10).

| Re 10 000 level | tolerance / `min_size` | elements | unknowns | dt | steps to t = 130 | core-hours (60% eff.) |
|---|---|---|---|---|---|---|
| l1eq | 0.25 / 0.032 | ~150 k | ~3 M | ~0.008 | ~16 k | ~1.5-2 k |
| l2eq | 0.125 / 0.016 | ~470 k | ~10 M | ~0.004 | ~32 k | ~8-11 k (~6 k at 100%) |
| l3eq | 0.0625 / 0.008 | ~1.4 M | ~30 M | ~0.002 | ~65 k | ~50-60 k |

All extrapolated from one Re 1000 level; the probe (Phase B) replaces the efficiency and
the per-step cost with measurements. If Re 1000 needs L3 to converge, every row moves up a
level (~5× the cost).

**With 5 000 core-hours, l1eq fits; l2eq does not**, even at perfect scaling (~6 k).

## 4. Phases and gates

**Phase A: finish Re 1000** (desktop and Mac; in progress).
- L2 runs on the desktop (started 2026-10-12, ~1.5 days); L0.5 and L1 on the Mac.
- **Q5 experiment** (`benchmarks/square_cylinder_re1000/l1_q5.yaml`, human 2026-10-12):
  l1's AMR with Q5/Q4 elements, about l2's unknowns. If Q5 matches l2 at a lower cost, the
  Re 10 000 levels move to Q5 and the table above shrinks (perhaps ~2×; not measured).
- **Gate A:** L2 and L1 agree within 2 % or the error bar. If not: L3 at Re 1000 (desktop
  ~2 weeks at 4 ranks, or ~1.4 k core-hours), and the Re 10 000 costs move up a level.

**Phase B: Bridges-2 readiness** (~150-250 core-hours; each job asked for).
- **B0, the environment (the human, on the cluster).** `psc_cpu` is resolved and mirrored
  but not installed, and it pins the GPU nodes' compiler build (`b2gpu`). RM nodes need
  the `b2rm` build: change the gcc external in `environments/psc_cpu/spack.yaml` to
  `/opt/packages/gcc/v13.3.1-p20240614/b2rm`, resolve on the login node, then run its
  install on an RM node (`install.sbatch` with `-p RM-shared` instead of GPU-shared).
  docs/install/bridges2.md, "The CPU-only environment".
- **B1, build and fast tier** (`jobs/build_test.sbatch`, RM-shared, 16 cores, ~10 core-hours).
- **B2, the scaling probe** (`jobs/probe.sbatch`, one RM node, ≤1.5 h, ~100-190
  core-hours): the fixed 142 k and 407 k meshes at 32, 64 and 128 ranks, at production's
  CFL 2. Output: s/step, parallel efficiency, iterations, memory per rank.
- **Gate B:** the measured cost of l1eq and l2eq, and the rank count to run at (the most
  core-hour-efficient one that fits the 48 h jobs).

**Phase C: Re 10 000 production.**
- **C1, l1eq** (~1.5-2 k core-hours; the pilot): the whole pipeline at about 1/5 of
  l2eq's cost: start, AMR to t = 60, chained jobs, restarts, `analyze.py`. Gives the first
  Re 10 000 statistics: one resolution, no convergence claim.
- **C2, l2eq**: only with more allocation (the remaining ~2.5-3 k core-hours do not cover
  it), or if Phase A/B cut its cost below the remainder (Q5; the measured efficiency).
- **Gate C:** l2eq and l1eq agree within 2 % or the error bar: the converged Re 10 000
  answer. Otherwise l3eq (~50 k core-hours) or a result with a stated uncertainty.

**Phase D: report**, and scope Re 100 000 from the measured Re 10 000 cost (~30× per
decade in 2D: ~300 k core-hours or more).

## 5. Cost levers, in order of promise

1. **Measure the efficiency** (Phase B): the table assumes 60 %; running at fewer ranks
   spends fewer core-hours for more wall time.
2. **Element order** (the Q5 experiment, Phase A).
3. **More allocation**: a supplement request backed by the probe's numbers.
4. **Start from a developed flow**: restarts don't check Re, so a Re 10 000 run could
   start from a developed Re 1000 state and skip most of the 30-unit spin-up (~20 %). A
   checkpoint restarts only at its own rank count, so that state must come from a Re 1000
   run at the production rank count; untested.
5. **A refinement region** (new code): stop refining the far wake (the domain runs 120 D
   downstream) if Re 1000 shows the forces don't depend on it.
6. **A shorter window**: averaging 60 instead of 100 time units cuts the fixed-mesh phase
   by 40 %, at a larger error bar (Re 1000: 2-5 % at 70-100).

## 6. What the job scripts do (and were tested for)

`benchmarks/square_cylinder_re10000/jobs/`: `build_test.sbatch`, `probe.sbatch`,
`production.sbatch` (sharing `common.sh`). The production script continues a level from
its checkpoint, refuses a rank count other than the checkpoint's, exits when the level is
done, and extends a finished level with `EXTEND_T_FINAL`; it never submits jobs itself
(a chain is submitted, and approved, as a whole). Tested on the desktop without Slurm
(2026-10-12) on tiny decks: the probe table; a fresh run, a second job that does nothing,
an extension past t_final from the checkpoint, a wrong rank count refused. Not tested:
under Slurm, the module loads, Bridges-2.

Two bugs that would have broken chained runs were found by those tests and fixed:
- a restart truncated the per-step history (`post/run_monitor`): it now appends, and the
  readers (`analyze.py`, `benchmarks/common.py`) drop the rows a later segment supersedes;
- a restart with forces aborted in the summary written before its first step
  (`Case::WriteSummary` asked the restored integrator for a force it doesn't have yet).

## 7. Decisions for the human

1. Phase B0 (the RM environment): the human's, on the cluster.
2. Each job, with its resources and cost (rule 3).
3. Whether to request more allocation (needed for l2eq unless the levers above close the gap).
4. Whether a Re 10 000 result at one resolution (l1eq) is worth reporting on its own.
