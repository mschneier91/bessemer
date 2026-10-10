# Taylor-Green vortex, Re = 1600

The International Workshop on High-Order CFD Methods case C3.5 (Wang et al., Int. J.
Numer. Meth. Fluids 72 (2013) 811-845): u = (sin x cos y cos z, -cos x sin y cos z, 0)
on a 2π-periodic box, ν = 1/1600, t ∈ [0, 20]. The vortex rolls up, stretches and breaks
down into turbulence; the dissipation rate peaks near t = 9.

| | |
|---|---|
| Reference | `reference/spectral_re1600_512.csv`: pseudo-spectral DNS on 512³: E(t), ε(t) = −dE/dt, enstrophy. ε_max = 0.01286 at t = 8.97 |
| Decks | `tgv_re1600_n16.yaml`, `_n32.yaml`, `_n64.yaml`: Q3/Q2, n³ elements; OIFS at CFL 2 |
| Output | `<path>/tgv_diagnostics.csv` every step: t, ½∫\|u\|², ν∫\|∇u\|², ‖∇·u‖ |
| Compare | `python3 plot.py tgv_n32/tgv_diagnostics.csv [more runs]`: ε_max and its time, max \|ε − ε_ref\| over the run |

The viscous dissipation ν⟨|∇u|²⟩ and −dE/dt agree for a resolved run. Their gap is a
resolution indicator, and the plot shows both. For a time-step check, rerun at
`cfl_target: 1`.

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

Results: none yet.
