# Turek & Hron: cylinder with a flag (CFD1–3, FSI1)

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
  The SVK solid will tell.
