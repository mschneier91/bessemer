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

Cloud scratch build (MFEM v4.10 from GitHub, apt hypre and Open MPI, not the Spack
toolchain), CPU, np 4, 2026-10-22.

| Test | Discretization | Elements | Steps | Wall | Drag | Error | Lift | Error |
|---|---|---|---|---|---|---|---|---|
| CFD1 | Q4/Q3, IMEX, error control | 314 | 6,783 | 5.3 min | 14.3373 | +0.31 % | 1.12702 | +0.71 % |
| CFD1 | same, `level: 1` | 1,256 | 13,148 | 40 min | 14.3097 | +0.12 % | 1.12210 | +0.27 % |
| CFD2 | the defaults (Q6/Q5, OIFS at CFL 2), to t = 20 | 314 | 17,920 | 38 min | 136.791 | +0.07 % | 10.6883 | +1.46 % |
| CFD3 | the defaults (Q6/Q5, OIFS at CFL 2), to t = 12 | 314 | 20,843 | 26 min | 439.52 ± 5.690 | +0.02 % / +1.3 % | −11.78 ± 431.7 | +1.0 % / −1.4 % |
| CFD3 | same, `level: 1` | 1,256 | 44,038 | not timed (shared cores, restarted twice) | 439.57 ± 5.713 | +0.03 % / +1.7 % | −12.27 ± 436.5 | −3.2 % / −0.29 % |

CFD3's frequency: 4.439 (+1.0 %); at `level: 1` 4.444 (+1.1 %). Over the last four
periods instead of the last: drag 439.58 ± 5.753, lift −11.88 ± 431.9, frequency 4.440;
at `level: 1` drag 439.57 ± 5.728, lift −12.29 ± 436.6. Refinement brings the lift
amplitude, the quantity that moves most, from −1.4 % to −0.29 %; the drag mean and
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
u_x 2.270·10⁻⁵, u_y 8.209·10⁻⁴. Q4/Q3 fluid, Q4 solid, IMEX, Δt 0.01 to t = 15.

| Mesh | Elements (fluid / solid) | Subiterations per step | Wall | Drag | Lift | u_x(A) | u_y(A) |
|---|---|---|---|---|---|---|---|
| coarse | 314 / 22 | 4.4 (≤ 40) | 7.3 min | 14.338 (+0.30 %) | 0.7602 (−0.48 %) | 2.418·10⁻⁵ (+6.5 %) | 8.204·10⁻⁴ (−0.06 %) |
| `level: 1` | 1,256 / 88 | 4.1 (≤ 48) | 32 min | 14.311 (+0.11 %) | 0.7600 (−0.50 %) | 2.390·10⁻⁵ (+5.3 %) | 8.264·10⁻⁴ (+0.67 %) |

Cloud scratch build, CPU, np 4, 2026-10-22. What the errors are:

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
[docs/fsi.md](../../docs/fsi.md) §1), coarse mesh, np 4, 7.1 min, 3-5 subiterations per
step, forces averaged over t ∈ [10, 15] (the dynamic flag leaves a small ripple on the
lift):

| Solid | Drag | Lift | u_x(A) | u_y(A) |
|---|---|---|---|---|
| linear (above) | +0.30 % | −0.48 % | +6.5 % | −0.06 % |
| St. Venant-Kirchhoff | 14.338 (+0.30 %) | 0.7634 (−0.06 %) | 2.315·10⁻⁵ (+2.0 %) | 8.095·10⁻⁴ (−1.4 %) |

All three changes are the ones estimated above: u_x loses the shortening (the +1.6 %
left after it, +2.0 % measured), u_y drops by the stress stiffening (of order 1 %,
−1.3 % measured), and the lift rises with the lower deflection (−430 per metre of u_y:
+0.6 %, +0.42 % measured). The remaining u_y and u_x errors are the coarse mesh's, not
separated yet (the `level: 1` SVK run is not done).

## FSI3

The flag in a self-sustained oscillation (`fsi3.yaml`): Ū = 2 (Re 200), ρ_s = ρ_f =
1000 (the strongest added mass), μ_s = 2·10⁶, ν_s = 0.4, the St. Venant-Kirchhoff
solid (`fsi.solid: hyperelasticity`, plane strain) stepped by BDF2 like the fluid,
Dirichlet-Neumann with IQN-ILS (two steps reused) to `fsi.rtol` 10⁻⁶, the flow solver to
10⁻⁹. Q4/Q3 fluid on the coarse mesh (314 elements), Q4 solid (8 × 2), IMEX, Δt
2.5·10⁻⁴ (IMEX's convective limit here: 5·10⁻⁴ diverges) to t = 10. The oscillation
grows from rest, overshoots at t ≈ 4 and is periodic from t ≈ 7; the last period
equals the mean of the last three to 2·10⁻⁴. Mean ± amplitude over the last period
(`python3 fsi.py fsi3 fsi3/fsi3_forces.csv fsi3/fsi3_fsi.csv`), displacements of point A
in mm:

| | bessemer | Turek & Hron | error |
|---|---|---|---|
| u_x(A) | −2.59 ± 2.44 | −2.69 ± 2.53 | amplitude −3.7 % |
| u_y(A) | 1.42 ± 33.02 | 1.48 ± 34.38 | amplitude −4.0 % |
| frequency | 5.40 | 5.3 | +2.0 % |
| drag | 459.1 ± 26.5 | 457.3 ± 22.66 | mean +0.4 %, amplitude +17 % |
| lift | 2.7 ± 137.0 | 2.22 ± 149.78 | amplitude −8.5 % |

4.5 subiterations per step, 40,000 steps, 5.7 h at np 4 (cloud scratch build, CPU,
2026-10-23, sharing the cores with other runs part of the time). Not separated yet: the
drag and lift amplitudes are the largest errors; CFD3 suggests the fluid mesh (refining
it moved CFD3's lift amplitude by 1.1 %), and neither the mesh nor Δt has been varied
here.

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

**Not reached yet: the fluid mesh folds first.** Coarse mesh, np 4, 2026-10-23: the
oscillation grows from rest at about 3 subiterations per step (the coupling is not the
limit), but at t = 8.6, with the tip's u_y at 58 mm of the reference 81 mm and still
growing, the linear-elastic mesh motion (`mesh_motion.stiffening: 1`) inverted the cells
next to the flag (min det F −0.72) and the run stopped. With stiffening 4 it reached
t = 8.86 (u_y 61 mm, u_x −17 mm) before the flow solve failed. Bending the flag boundary
alone in a `mesh_motion` deck (a cantilever shape in y, no tip rotation or shortening),
the coarse mesh survives tip deflections of 48 / 48 / 80 / 90 mm at stiffening 1 / 2 /
3 / 4; the real motion, which also shortens and rotates the tip, is harder. FSI2 needs a
mesh motion that follows large rotations (the hyperelastic mesh motion, or an
incremental update), not a coupling change.

