# Turek & Hron: cylinder with a flag (CFD1–3, FSI1–3)

The fluid tests of the Turek & Hron (2006) FSI benchmark, with the flag rigid:
the first step toward FSI1–3 (docs/ale_fluid.md, "Next"). A channel 2.5 × 0.41,
a cylinder of radius 0.05 at (0.2, 0.2) and a flag 0.35 × 0.02 attached behind it,
its right bottom corner at (0.6, 0.19). ν = 10⁻³, parabolic inflow of mean Ū,
1.5 Ū at its peak, ramped by (1 − cos(πt/2))/2 for t < 2; no-slip walls, cylinder
and flag; do-nothing outflow.

| Test | Ū | Re | Reference (Turek & Hron, finest level) |
|---|---|---|---|
| CFD1 | 0.2 | 20 | steady: drag 14.2929, lift 1.11905 |
| CFD2 | 1 | 100 | steady: drag 136.700, lift 10.5343 |
| CFD3 | 2 | 200 | periodic: drag 439.45 ± 5.6183, lift −11.893 ± 437.81, frequency 4.3956 (Δt 0.005) |

Forces are on the cylinder and the flag together, F = ∫ σ n, at density 1000 (the
decks' `reference_area: 0.002` turns the summary's cd and cl into these forces).
For CFD3, mean = (max + min)/2 and amplitude = (max − min)/2 over the last period:
`python3 forces.py cfd3/cfd3_forces.csv`. Source: the FeatFlow benchmark pages
(wwwold.mathematik.tu-dortmund.de/~featflow, "FSI benchmark", CFD tests);
S. Turek and J. Hron, in *Fluid-Structure Interaction*, LNCSE 53, Springer 2006.

**Mesh:** `mesh.geometry: cylinder_channel` with `flag_length: 0.35` (new): the DFG
O-grid with lines at y = 0.19, 0.21 and x = 0.6, the flag's cells left out and
boundary `flag` on its surface; 314 elements, the cylinder and the flag's ends
curved exactly. `part: solid` gives the flag's own mesh with the same nodes on the
interface (`cases/turek_hron_flag_linear.yaml`; the solid for FSI). `level: 1` doubles every count.

## Results

Bridges-2 (the Spack toolchain), CPU, main 12e3767 (the pressure LOR-AMG's strength
threshold 0.25 in 2D), 2026-10-24.

| Test | Discretization | Elements | Steps | Wall | Drag | Error | Lift | Error |
|---|---|---|---|---|---|---|---|---|
| CFD1 | Q4/Q3, IMEX, error control, np 4 | 314 | 6,395 | 2.7 min | 14.3373 | +0.31 % | 1.12695 | +0.71 % |
| CFD1 | same, `level: 1`, np 8 | 1,256 | 12,416 | 14.4 min | 14.3097 | +0.12 % | 1.12202 | +0.27 % |
| CFD2 | the defaults (Q6/Q5, OIFS at CFL 2), to t = 20, np 4 | 314 | 17,920 | 28.5 min | 136.791 | +0.07 % | 10.6883 | +1.46 % |
| CFD3 | the defaults (Q6/Q5, OIFS at CFL 2), to t = 12, np 4 | 314 | 20,843 | 26.8 min | 439.51 ± 5.702 | +0.01 % / +1.5 % | −11.69 ± 431.7 | +1.7 % / −1.4 % |
| CFD3 | same, `level: 1`, np 16 | 1,256 | 44,038 | 61.6 min | 439.57 ± 5.710 | +0.03 % / +1.6 % | −12.19 ± 436.7 | −2.5 % / −0.24 % |

Against the cloud scratch build's table at the 2D threshold 0.1 (2026-10-22): CFD1 and CFD2
agree to 10⁻⁴, CFD3's periodic values to 0.2 % of the drag amplitude and 0.1 of the lift
mean (a small difference of large values); the error-controlled CFD1 takes 6 % fewer steps.
CFD3's frequency: 4.439 (+1.0 %); at `level: 1` 4.444 (+1.1 %). Over the last four periods instead of the last: drag 439.48 ± 5.735, lift
−11.82 ± 431.9, frequency 4.440; at `level: 1` drag 439.59 ± 5.737, lift −12.23 ± 436.8.
Refinement brings the lift amplitude, the quantity that moves most, from −1.4 % to −0.24 %;
the drag mean and
amplitude, the frequency and the lift mean (a small difference of large values, ±3 % of
it is ±0.4) stay where they were: what is left there is not this mesh family's spatial error
(time step, or the reference's own error, not separated).

- CFD2 is steady here as in Turek & Hron (the flag holds the wake): over t ≥ 15 the
  drag varies by 6·10⁻³ and the lift by 10⁻⁵. Its lift, a small difference of large
  pressure forces, carries the larger error.
- CFD1 converges by a factor of ~2.6 per level, below the order's rate: the flag's
  corners (the re-entrant ones where it meets the cylinder, and its tip) are
  singular and the mesh is not graded toward them.
- `cases/turek_hron_flag_linear.yaml`: the flag alone under CSM1's load with the *linear* solid: tip
  u_y = −0.0680 against the large-deformation reference −0.0661, and u_x = 0 where
  the reference has −7.2·10⁻³ (the geometric effect a small-strain model has not).
  CSM1–3 proper need the St. Venant-Kirchhoff solid.
- Not run on a GPU.

## FSI1

The flag elastic (`fsi1.yaml`, [docs/fsi.md](../../docs/fsi.md)): the CFD1 flow
coupled with the flag of FSI1 (ρ_s = ρ_f = 1000, ν_s = 0.4, μ_s = 0.5·10⁶), steady;
Dirichlet-Neumann subiterations with IQN-ILS (two steps reused) to 10⁻⁴ at every
step, the flag solved statically, the **linear** solid (Turek & Hron's is St.
Venant-Kirchhoff). Reference: drag 14.295, lift 0.7638, tip (point A) displacement
u_x 2.270·10⁻⁵, u_y 8.209·10⁻⁴. Q4/Q3 fluid, Q4 solid, IMEX, Δt 0.01 to t = 15, the
flow solver to 10⁻⁸.

| Mesh | Elements (fluid / solid) | Subiterations per step | Wall | Drag | Lift | u_x(A) | u_y(A) |
|---|---|---|---|---|---|---|---|
| coarse, np 4 | 314 / 22 | 2.8 (≤ 15) | 4.3 min | 14.338 (+0.30 %) | 0.7595 (−0.56 %) | 2.418·10⁻⁵ (+6.5 %) | 8.203·10⁻⁴ (−0.07 %) |
| `level: 1`, np 8 | 1,256 / 88 | 2.8 (≤ 14) | 15.0 min | 14.311 (+0.11 %) | 0.7599 (−0.51 %) | 2.390·10⁻⁵ (+5.3 %) | 8.263·10⁻⁴ (+0.66 %) |

Bridges-2, CPU, main 12e3767, 2026-10-24. The lift is its mean over t ∈ [10, 15]: it keeps
a ±0.0014 wobble, so its value at t = 15 differs by up to 0.2 % (the cloud scratch build's
table of 2026-10-22, at t = 15: 0.7602 and 0.7600; the other columns are its digits). What
the errors are:

- **The flow solver must sit well below `fsi.rtol`.** At the default `solver.rtol` 10⁻⁶
  the flow's error fed the IQN-ILS columns: 13.8 subiterations per step over t ≤ 1 at
  `level: 1`, and with the 2D pressure LOR-AMG threshold 0.25 (2026-10-24) 14-19, which
  diverged at t = 0.28. At 10⁻⁸ (the decks since then): 5.8 there, 2.8 over the run, no cut.

- **The march must run past t ≈ 10.** The flag settles slowly after the inflow ramp
  (t ≤ 2): u_y was 4.0 % short at t = 5 (the first version of this deck stopped there,
  lift −2.4 %), 0.2 % at t = 8 and steady to 0.05 % from t = 10 (coarse mesh).
- **u_x is the linear solid's.** Bending shortens the flag by ½∫(w′)² ds ≈
  0.571 u_y²/L for its deflection shape, 1.1·10⁻⁶ = 4.9 % of the reference u_x, which a
  small-strain model does not have; after it the u_x error is +1.6 % (coarse) and
  +0.4 % (level 1).
- **Not the solid's mesh or the coupling tolerance.** The flag under CSM1's load moves
  u_y by under 0.03 % from 8 × 2 Q4 elements to 32 × 8 Q4 or Q6; `fsi.rtol` 10⁻⁴ and
  10⁻⁶ give the same digits (and 10⁻⁴ a third of the subiterations).
- The lift and u_y move together (the deflected flag lowers the lift, about −430 per
  metre of u_y in the reference values), so the level-1 run's higher u_y explains most of
  its lower lift. The remaining 0.5–0.7 % is not separated: the linear solid's missing
  stress stiffening (its axial tension, about 2.6 N, would lower u_y by of order 1 %:
  an estimate), the traction applied on the undeformed interface, the fluid mesh.
  The SVK solid will tell, and it does (below).

**With the St. Venant-Kirchhoff solid** (`fsi1_svk.yaml`, `fsi.solid: hyperelasticity`,
the flag dynamic: generalized-alpha, ρ_∞ 0.5; a static SVK flag does not couple,
[docs/fsi.md](../../docs/fsi.md) §1), coarse mesh, np 4, 6.6 min, 3.0 subiterations per
step (≤ 8), forces averaged over t ∈ [10, 15]:

| Solid | Drag | Lift | u_x(A) | u_y(A) |
|---|---|---|---|---|
| linear (above) | +0.30 % | −0.56 % | +6.5 % | −0.07 % |
| St. Venant-Kirchhoff | 14.338 (+0.30 %) | 0.7606 (−0.42 %) | 2.315·10⁻⁵ (+2.0 %) | 8.096·10⁻⁴ (−1.4 %) |

u_x loses the shortening (the +1.6 % left after it, +2.0 % measured) and u_y drops by
the stress stiffening (of order 1 %, −1.3 % measured), as estimated above. The lift should
rise with the lower deflection (−430 per metre of u_y: +0.6 %), but the dynamic flag's
lift oscillates by ±4 % (std 0.032, period 0.32-0.37: the flag's own mode, lightly
damped), and its mean over [10, 15] moves by up to ±0.6 % between the window's halves: the
+0.14 % measured here (+0.42 % on the cloud build at the default flow tolerance) does not
resolve it. The remaining u_y and u_x errors are the coarse mesh's, not separated yet (the
`level: 1` SVK run is not done).

## FSI3

The flag in a self-sustained oscillation (`fsi3.yaml`): Ū = 2 (Re 200), ρ_s = ρ_f =
1000 (the strongest added mass), μ_s = 2·10⁶, ν_s = 0.4, the St. Venant-Kirchhoff
solid (`fsi.solid: hyperelasticity`, plane strain) stepped by BDF2 like the fluid,
Dirichlet-Neumann with IQN-ILS (two steps reused) to `fsi.rtol` 10⁻⁶, the flow solver to
10⁻⁹. Q4/Q3 fluid, Q4 solid, IMEX, to t = 10, both runs from t = 0 on main 12e3767
(Bridges-2, CPU):
- the deck's mesh (314 fluid elements, solid 8 × 2) at Δt 2.5·10⁻⁴;
- `level: 1` (1,256; 16 × 4) at Δt 1.25·10⁻⁴.

IMEX's convective limit sets Δt. On the deck's mesh, 5·10⁻⁴ diverges (2.5·10⁻⁴ peaks at CFL
0.47 by the cylinder), so `level: 1`, with half the element size there, takes half the step.
The oscillation grows from rest, overshoots at t ≈ 4 and is periodic from t ≈ 7: `level: 1`'s
three-period windows ending at t = 8, 9 and 10 agree to 10⁻⁴.

Mean ± amplitude over the last three periods of u_y (`python3 fsi.py fsi3
fsi3/fsi3_forces.csv fsi3/fsi3_fsi.csv --periods 3 --smooth 0.00125`). The smoothing over
±0.00125 removes the forces' step-to-step ripple and takes 0.1 % off a 10.9 Hz amplitude;
FSI2's ±0.005 would take 2 % here. Displacements of point A in mm; in parentheses, the
amplitudes' errors against Turek & Hron's 2006 values:

| | deck (np 16) | `level: 1` (np 32) | Turek & Hron 2006 |
|---|---|---|---|
| u_x(A) | −2.59 ± 2.43 (−3.9 %) | −2.66 ± 2.50 (−1.0 %) | −2.69 ± 2.53 |
| u_y(A) | 1.42 ± 33.00 (−4.0 %) | 1.45 ± 33.57 (−2.4 %) | 1.48 ± 34.38 |
| frequency (u_y) | 5.40 | 5.47 | 5.3 |
| drag | 459.3 ± 26.07 (+15 %) | 458.9 ± 25.70 (+13 %) | 457.3 ± 22.66 |
| lift | 2.7 ± 134.9 (−10.0 %) | 2.3 ± 144.8 (−3.3 %) | 2.22 ± 149.78 |

The 2006 values are coarse, as in FSI2. FEAT's own refinement ([FEAT's FSI benchmark pages](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html), Δt 5·10⁻⁴):

| FEAT level (Q2/P1 unknowns) | u_x(A) | u_y(A) | Drag | Lift | Frequency |
|---|---|---|---|---|---|
| 2 (19,488) | −3.02 ± 2.85 | 1.42 ± 35.63 | 458.7 ± 28.78 | 2.23 ± 146.02 | 5.37 |
| 3 (76,672) | −2.78 ± 2.62 | 1.44 ± 34.35 | 459.1 ± 26.62 | 2.39 ± 150.68 | 5.46 |
| 4 (304,128) | −2.86 ± 2.70 | 1.45 ± 34.90 | 460.2 ± 27.47 | 2.37 ± 153.75 | 5.46 |

Their amplitudes are not monotone in the level (u_x 2.85, 2.62, 2.70). Their drag amplitude
is 26.6-28.8 at every level, so the 2006 value (22.66) is the outlier, which is what the
+13-15 % above reflects.

Against FEAT's levels 3 and 4, `level: 1`'s amplitudes are:
- u_x −4.4 / −7.2 %, u_y −2.3 / −3.8 %;
- drag −3.5 / −6.4 % (the mean −0.3 %), lift −3.9 / −5.8 %;
- the frequency +0.3 %.

From the deck's mesh to `level: 1` the amplitudes moved toward FEAT's finest: u_x +3.0 %, u_y
+1.7 %, lift +7.4 %. The drag's moved −1.4 %. `level: 1` sits at FEAT's level 3, as FSI2's
`level: 1` sat at their levels 2-3 before `level: 2` reached their level 4 to under 1 %.
`level: 2` (5,024 elements, Δt 6.25·10⁻⁵ by the same CFL argument, about 8× `level: 1`'s
cost: an estimate) is not run.

Cost: the deck's mesh took 4.5 subiterations per step, 40,000 steps, 86 min at np 16;
`level: 1` took 5.1 per step, 80,000 steps, 7.1 h at np 32. The cloud scratch build's run of
2026-10-23 (the deck's mesh, 5.7 h at np 4) gave the same digits but the lift's 2.7 mean.

Two settings matter, both found on this case (docs/fsi.md §1): with the solid on
generalized-alpha at ρ_∞ = 1 and `fsi.rtol` 10⁻⁴ the forces carried a step-to-step
ripple that grew with the flag's motion to ±25,000 in the lift by t = 4.5, while the
displacement stayed smooth. The coupling's residual error, times the added mass and
1/Δt², is the ripple; BDF2 cut it 17×, `fsi.rtol` 10⁻⁶ a further 100× for 4.4 → 5.0
subiterations per step. `solver.rtol` must then sit well below it (at the default the
subiterations stall in the third step).

## FSI2

`fsi2.yaml`: Ū = 1 (Re 100), ρ_s = 10⁴, μ_s = 0.5·10⁶, the large-amplitude
oscillation, the same settings as FSI3 at Δt 5·10⁻⁴ to t = 15. Reference: u_x(A)
−14.58 ± 12.44, u_y(A) 1.23 ± 80.6 (mm), frequency 2.0 (u_y), drag 208.83 ± 73.75, lift
0.88 ± 234.2.

**Verified (2026-10-24): `level: 2` agrees with the finest published results to under 1 %.**
With the neo-Hookean mesh motion (`mesh_motion.model: neo_hookean`, in the deck), every run
from t = 0 on main 12e3767 (the pressure LOR-AMG's 2D threshold 0.25), on Bridges-2's
CPUs. Mean ± amplitude over the last three periods of u_y (`python3 fsi.py fsi2
fsi2/fsi2_forces.csv fsi2/fsi2_fsi.csv --periods 3 --smooth 0.005`). In parentheses, the
amplitudes' errors against Turek & Hron's 2006 values above:

| Mesh | Δt | u_x(A) | u_y(A) | Frequency (u_y) |
|---|---|---|---|---|
| deck (314 elements), np 8 | 5·10⁻⁴ | −13.15 ± 11.39 (−8.4 %) | 1.25 ± 75.92 (−5.8 %) | 1.924 |
| deck | 2.5·10⁻⁴ | −13.14 ± 11.39 (−8.5 %) | 1.25 ± 75.92 (−5.8 %) | 1.924 |
| `level: 1` (1,256), np 32 | 5·10⁻⁴ | −14.34 ± 12.30 (−1.1 %) | 1.25 ± 79.80 (−1.0 %) | 1.927 |
| `level: 1` | 2.5·10⁻⁴ | −14.34 ± 12.30 (−1.2 %) | 1.25 ± 79.80 (−1.0 %) | 1.927 |
| `level: 2` (5,024), np 64 | 2.5·10⁻⁴ | −14.99 ± 12.80 (+2.9 %) | 1.26 ± 81.85 (+1.5 %) | 1.928 |

The displacements do not move with Δt (to 4 digits). They do move with the mesh, and so do
Turek & Hron's own: the 2006 values are not converged. FEAT's refinement of FSI2 ([FEAT's FSI benchmark pages](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html),
Δt 5·10⁻⁴, the same quantities):

| FEAT level (Q2/P1 unknowns) | u_x(A) | u_y(A) | Drag | Lift |
|---|---|---|---|---|
| 2 (19,488) | −14.01 ± 12.04 | 1.28 ± 79.2 | 210.14 ± 72.86 | 0.49 ± 228.7 |
| 3 (76,672) | −14.48 ± 12.45 | 1.24 ± 80.7 | 213.05 ± 75.74 | 0.84 ± 234.8 |
| 4 (304,128) | −14.85 ± 12.70 | 1.30 ± 81.6 | 215.06 ± 77.65 | 0.61 ± 237.8 |

Our `level: 2` is Q4/Q3 with 162,720 velocity and 45,948 pressure unknowns, plus 11,682
solid. Against FEAT's level 4:
- u_x amplitude +0.8 %, u_y +0.3 %;
- drag (smoothed, below) mean −0.1 % and amplitude +0.3 %;
- lift amplitude 0.0 %.

Our `level: 1` sits around FEAT's levels 2-3. The frequencies are 1.93 and 3.86 in both.

**The forces carry a step-to-step ripple.** It comes from the coupling's residual error:
its RMS doubles when Δt halves, and the max − min amplitude picks up its peaks. On `level: 1`,
for instance, the raw drag amplitude is +2.7 % at Δt 5·10⁻⁴ and +8.9 % at 2.5·10⁻⁴. The
displacements carry none. Smoothed by a moving average over ±0.005 (10-20 steps; it takes
0.08 % off u_y's amplitude), the force amplitudes do not move with Δt. In parentheses, the
errors against the 2006 values:

| Mesh | Δt | Drag | Lift |
|---|---|---|---|
| deck | 5·10⁻⁴ | 204.1 ± 66.4 (−9.9 %) | 1.0 ± 209.3 (−10.6 %) |
| deck | 2.5·10⁻⁴ | 204.2 ± 66.5 (−9.8 %) | 1.0 ± 209.9 (−10.4 %) |
| `level: 1` | 5·10⁻⁴ | 210.9 ± 73.7 (0.0 %) | 0.7 ± 226.7 (−3.2 %) |
| `level: 1` | 2.5·10⁻⁴ | 210.9 ± 73.8 (+0.1 %) | 0.5 ± 227.2 (−3.0 %) |
| `level: 2` | 2.5·10⁻⁴ | 214.8 ± 77.8 (+5.6 %) | 0.7 ± 237.7 (+1.5 %) |

A tighter `fsi.rtol` would shrink the ripple, at the cost of more subiterations; not tried.

`level: 2` stopped at t = 12.77 (the CPU budget; human 2026-10-24: verified as is), so its
window is t = 11.1-12.6.
- Its displacement amplitudes had been steady to 0.3 % since t ≈ 9. `level: 1`'s window
  ending at t = 12.6 matches its final one to 0.2 %.
- Its smoothed force amplitudes still moved by about 2 % from window to window: the lift's
  was +5.7, +4.0, +1.5 % (against 2006) over t = 9.0-10.6, 10.0-11.6, 11.1-12.6. So take
  `level: 2`'s force amplitudes to about ±2 %.

Before cb2c4a6 (2026-10-23) these runs stopped at t = 8.4-9.15 with a stalled flow solve. The
cause was folded LOR sub-elements in the pressure preconditioner near the tip
([docs/status.md](../../docs/status.md)). The runs are in bessemer-hyp3/runs/bench_12e3767 on
Bridges-2.

The linear-elastic mesh motion does not reach it. Coarse mesh, np 4, 2026-10-23: the
oscillation grows from rest at about 3 subiterations per step (the coupling is not the
limit), but at t = 8.6, with the tip's u_y at 58 mm of the reference 81 mm and still
growing, the linear-elastic mesh motion (`mesh_motion.stiffening: 1`) inverted the cells
next to the flag (min det F −0.72) and the run stopped. With stiffening 4 it reached
t = 8.86 (u_y 61 mm, u_x −17 mm) before the flow solve failed. Bending the flag boundary
alone in a `mesh_motion` deck (a cantilever shape in y, no tip rotation or shortening),
the coarse mesh survives tip deflections of 48 / 48 / 80 / 90 mm at stiffening 1 / 2 /
3 / 4; the real motion, which also shortens and rotates the tip, is harder. FSI2 needed
a mesh motion that follows large rotations: the neo-Hookean one above.

