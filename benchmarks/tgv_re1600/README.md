# Taylor-Green vortex, Re = 1600

The International Workshop on High-Order CFD Methods case C3.5 (Wang et al., Int. J.
Numer. Meth. Fluids 72 (2013) 811-845): u = (sin x cos y cos z, -cos x sin y cos z, 0)
on a 2π-periodic box, ν = 1/1600, t ∈ [0, 20]. The vortex rolls up, stretches and breaks
down into turbulence; the dissipation rate peaks near t = 9.

| | |
|---|---|
| Reference | `reference/spectral_re1600_512.csv`: pseudo-spectral DNS on 512³: E(t), ε(t) = −dE/dt, enstrophy. ε_max = 0.01286 at t = 8.97 |
| Decks | `tgv_re1600_n16.yaml`, `_n32.yaml`, `_n64.yaml`: Q3/Q2, n³ elements; IMEX convection (BDF2) at CFL 0.25 |
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

**64³** needs two 80 GB GPUs (62 GB each at np 2; out of memory on one). Not yet run at
CFL 0.25: about 7 s per step on two H100s, an estimated 9 hours to t = 20.

A 32³ run at CFL 0.5 (2026-10-13, 39 min) reached max −dE/dt 0.01292 at t = 10.1; it went
through the instability above and is not a result.

