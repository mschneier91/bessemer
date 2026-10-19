# Taylor-Green vortex, Re = 1600

The International Workshop on High-Order CFD Methods case C3.5 (Wang et al., Int. J.
Numer. Meth. Fluids 72 (2013) 811-845): u = (sin x cos y cos z, -cos x sin y cos z, 0)
on a 2π-periodic box, ν = 1/1600, t ∈ [0, 20]. The vortex rolls up, stretches and breaks
down into turbulence; the dissipation rate peaks near t = 9.

| | |
|---|---|
| Reference | `reference/spectral_re1600_512.csv`: pseudo-spectral DNS on 512³: E(t), ε(t) = −dE/dt, enstrophy. ε_max = 0.01286 at t = 8.97 |
| Decks | `tgv_re1600_n21.yaml`, `_n32.yaml`, `_n42.yaml`: Q6/Q5, n³ elements (126, 192, 252 velocity nodes per direction); OIFS (BDF3) at CFL 2, the defaults (2026-10-19; the Q3/Q2 IMEX decks of the older results below were `_n16`, `_n32`, `_n64` at CFL 0.25) |
| Output | `<path>/tgv_diagnostics.csv` every step: t, ½∫\|u\|², ν∫\|∇u\|², ‖∇·u‖ |
| Compare | `python3 plot.py tgv_n32/tgv_diagnostics.csv [more runs]`: ε_max and its time, max \|ε − ε_ref\| over the run |

The viscous dissipation ν⟨|∇u|²⟩ and −dE/dt agree for a resolved run. Their gap is a
resolution indicator, and the plot shows both.

## Time step

**Check that the kinetic energy never increases** before reading anything else off a run:
in a periodic box without forcing it cannot. With the case defaults (IMEX, BDF2 with
second-order extrapolation of the convection term) this case is **unstable at CFL 0.5**.
The energy rises from t ≈ 1.3 at 32³ (t ≈ 0.7 at 64³), the dissipation and the divergence
norm jump several-fold, the step controller halves the step, and the run continues in a
noisy state to t = 20 without any error. The "CFL up to 0.7" guidance in
`docs/developing.md` was measured on the Re 100 cylinder; here there is too little
viscosity to damp the explicit term. 32³ to t = 2.5, one H100 (2026-10-13):

| Setting | Steps with rising energy | max ν⟨\|∇u\|²⟩ | max ‖∇·u‖/√\|Ω\| | Wall |
|---|---|---|---|---|
| IMEX BDF2, CFL 0.5 (γ = 2h) | 26, from t = 1.46 | 0.00276 | 0.028 | 263 s |
| the same, `grad_div_h: 0` | 96, from t = 1.10 | 0.00366 | 0.94 | 145 s |
| IMEX BDF3 / EXT3, CFL 0.5 | 1, at t = 1.39 | 0.00102 | 0.012 | 215 s |
| **IMEX BDF2, CFL 0.25** | **0** | 0.00088 | 0.0017 | 275 s |
| OIFS, CFL 2 | 0 | 0.00088 | 0.0015 | 73 s |

Grad-div is not the cause (without it the run is worse) and neither is the Schur
preconditioner's grad-div term (identical physics with it off). OIFS at CFL 2 is clean and
about 4× cheaper to t = 2.5, but it has not been taken through the turbulent phase, where
its larger step may cost accuracy. BDF3 probably needs CFL ≈ 0.35; not run.

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

## Results

**32³ Q3/Q2 (96 nodes per direction), IMEX BDF2 at CFL 0.25, γ = 2h** (2026-10-13;
Bridges-2, one H100, CUDA backend, 57 min, 1924 steps, 14 outer iterations per step; the
energy never increases). E and ε per unit volume, −dE/dt by a difference over ±0.25:

| t | E | E ref | −dE/dt | ν⟨\|∇u\|²⟩ | ε ref |
|---|---|---|---|---|---|
| 2 | 0.12392 | 0.12392 | 0.00071 | 0.00071 | 0.00071 |
| 4 | 0.12148 | 0.12151 | 0.00218 | 0.00207 | 0.00207 |
| 6 | 0.11276 | 0.11360 | 0.00620 | 0.00563 | 0.00553 |
| 8 | 0.09610 | 0.09831 | 0.01135 | 0.00980 | 0.01037 |
| 9 | 0.08395 | 0.08638 | 0.01216 | 0.01069 | 0.01285 |
| 10 | 0.07194 | 0.07450 | 0.01206 | 0.01070 | 0.01127 |
| 12 | 0.05016 | 0.05448 | 0.00862 | 0.00794 | 0.00836 |
| 16 | 0.02942 | 0.03249 | 0.00330 | 0.00319 | 0.00362 |

Exact to t ≈ 3; from t ≈ 4 the mesh dissipates about 10 % more than the reference
(−dE/dt above ε ref while the viscous part follows it), so the energy runs low, by 3 % at
the peak and 8–10 % from t = 12. The peak itself is flattened: max −dE/dt 0.0123 near
t = 8.7 and a plateau to t = 10, against 0.01286 at t = 8.97. Under-resolved, as expected
at 96 nodes against a 512³ reference. `plot.py` was not run (no matplotlib on the login
node); the table is from the diagnostics CSV directly.

**64³ Q3/Q2 (192 nodes per direction), IMEX BDF2 at CFL 0.3, γ = 2h** (2026-10-14;
Bridges-2, **one H100**, np 1, CUDA backend, after the GPU memory work of 2026-10-13,
[../../docs/gpu_memory.md](../../docs/gpu_memory.md); deck `tgv_re1600_n64.yaml` with
`cfl_target: 0.3`). t = 20 in 12 h 45 min, 3284 steps, 13.9 s per step on average (10–11 s
early, 17 s late, as the outer iterations rise from ~10 to 15); 43.7 GB of the GPU's 80;
**the energy never increases**. −dE/dt by a difference over ±0.25; errors are of E:

| t | E | E ref | E err | E err 32³ | −dE/dt | ν⟨\|∇u\|²⟩ | ε ref |
|---|---|---|---|---|---|---|---|
| 4 | 0.12151 | 0.12151 | 0.00 % | −0.02 % | 0.00210 | 0.00207 | 0.00207 |
| 8 | 0.09802 | 0.09829 | −0.28 % | −2.24 % | 0.01054 | 0.01030 | 0.01037 |
| 9 | 0.08591 | 0.08638 | −0.55 % | −2.82 % | 0.01283 | 0.01251 | 0.01285 |
| 10 | 0.07367 | 0.07446 | −1.06 % | −3.43 % | 0.01187 | 0.01163 | 0.01127 |
| 12 | 0.05317 | 0.05444 | −2.3 % | −7.9 % | 0.00814 | 0.00802 | 0.00836 |
| 16 | 0.03118 | 0.03251 | −4.1 % | −9.5 % | 0.00365 | 0.00362 | 0.00362 |
| 19.5 | 0.02142 | 0.02252 | −4.9 % | −7.8 % | 0.00190 | 0.00189 | 0.00204 |

**The dissipation peak is captured**: max −dE/dt 0.01294 at t = 8.95 against 0.01286 at
8.97 (+0.6 %); the resolved part ν⟨|∇u|²⟩ peaks at 0.01253 at t = 8.89 (97 % of the
reference maximum), so the mesh adds about 3 % at the peak (32³: about 12 %). E is within
0.6 % to the peak and drifts low after it, by 5 % at t = 20 (max |E − E_ref| = 0.0013):
the decay of the small scales is under-resolved at 192 nodes against 512³.

**CFL 0.3 gives the same answer as 0.25**: a 64³ run at CFL 0.25 on two H100s
(2026-10-13, cancelled at t = 10.8) agrees with this one in E to 0.02 percentage points
throughout, and was itself stable. Memory: 64³ needed two GPUs (62 GB each) before the
memory work; one H100 now holds up to 80³ (240 nodes per direction), roughly 30 h to t = 20
at this run's pace.

A 32³ run at CFL 0.5 (2026-10-13, 39 min) reached max −dE/dt 0.01292 at t = 10.1; it went
through the instability above and is not a result.


**Q6/Q5 with OIFS, the defaults** (2026-10-17, the 3D verification of the GPU memory work;
human: "run the 3d verification tests ... to really be sure we didn't introduce any bugs").
The n32 deck with `--set discretization.order_u=6 --set discretization.order_p=5
--set "mesh.elements=[n,n,n]" --set time.convection=oifs --set time.cfl_target=2` (substep
CFL 1.5 and grad-div 6 h_K / k_u by default; the deck's solver settings), each case run with
main of before the GPU memory work (aa6aa7e) and main after it (changes 17-27), side by side on Bridges-2, one H100 each. Dissipation error against the 512³ DNS (`plot.py`): at its peak, and the largest over
t <= 20 as a fraction of the peak. Peak memory: `nvidia-smi` over the whole run.

| Case | DOFs | Peak error old / new | Largest error old / new | Steps | Wall old / new | Peak memory old / new |
|---|---|---|---|---|---|---|
| 21³ (126 nodes), CFL 2 | 7.2M | 1.87 / 1.87 % | 3.32 / 3.32 % | 525 | 142 / 127 s | 10.5 / 8.2 GiB |
| 21³, CFL 1 | 7.2M | 3.36 / 3.37 % | 4.74 / 4.74 % | 1,025 | 170 / 144 s | 10.6 / 8.2 GiB |
| 21³, `solver.outer: schur` | 7.2M | 1.86 / 1.86 % | 3.29 / 3.30 % | 526 | 148 / 123 s | 9.2 / 6.7 GiB |
| 21³, IMEX at CFL 0.25 | 7.2M | 4.03 / 4.03 % | 5.31 / 5.31 % | 4,096 | 391 / 382 s | 9.9 / 7.2 GiB |
| 21³, np 2 (two H100s) | 7.2M | 1.87 / 1.86 % | 3.31 / 3.31 % | 525 | 104 / 97 s | 13.0 / 10.7 GiB |
| 21³, `time.oifs_on_the_fly` (new only) | 7.2M | 1.86 % | 3.31 % | 525 | 116 s | 7.5 GiB |
| 32³ (192 nodes) | 25.3M | 0.23 / 0.22 % | 2.42 / 2.42 % | 763 | 614 / 527 s | 32.1 / 23.6 GiB |
| 42³ (252 nodes) | 57.3M | 0.16 / 0.17 % | 1.80 / 1.79 % | 1,019 | 1,759 / 1,467 s | 70.1 / 51.0 GiB |
| 42³, Schur | 57.3M | 0.17 / 0.17 % | 1.75 / 1.75 % | 1,024 | 2,164 / 1,782 s | 58.8 / 39.7 GiB |
| 42³, Schur, on the fly (new only) | 57.3M | 0.17 % | 1.75 % | 1,023 | 1,874 s | 34.7 GiB |

- **The same answers.** The dissipation curves of the two mains differ by 1e-7 to 3e-6 of the
  peak up to t = 4 (roundoff passed through the solver tolerance), 2e-5 to 8e-5 up to the
  peak, and 1e-4 to 4e-4 over the turbulent decay, against gaps of 1.5e-2 and more to the DNS;
  the step counts match to one step.
- **252³ at Q6 is the first run under 2 %** in the largest dissipation error (1.79 %;
  the order study had none by 192³).
- The new main is 2-17 % faster in every pair. The two lanes shared a node, so the times
  are indicative; the small 3D-1Z case (dfg_3d) ran 3.5x slower paired than alone.
