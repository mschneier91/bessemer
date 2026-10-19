# DFG flow around a cylinder in 3D (3D-1Z, 3D-3Z)

The 3D versions of the DFG benchmarks (Schäfer & Turek 1996): a channel 2.5 × 0.41 × 0.41
with a cylinder of diameter 0.1 spanning it along z at (0.5, 0.2). The inflow is
16 U_m y z (H − y)(H − z) / H⁴. Every channel wall is no-slip, including the z faces, and
the outflow is do-nothing.

| | |
|---|---|
| 3D-1Z | steady, U_m = 0.45 (Re 20). Reference c_D = 6.18533, c_L = 0.009401 (featflow.de) |
| 3D-3Z | U_m(t) = 2.25 sin(πt/8), t ∈ [0, 8]. FeatFlow's finest: c_D,max = 3.2978, c_L,min = −0.010999; histories in `reference/featflow_3d3z.csv` |
| Mesh | the 2D DFG mesh extruded in z (`mesh.dim: 3` with `geometry: cylinder_channel`), curved cylinder of the velocity order. Both decks Q6/Q5 with AMR since 2026-10-19 (the defaults): an event every time unit for t in [1, 5], relative marking θ = 0.5, at most 2,500 elements |
| Compare | `python3 plot.py 1z <summary.json>`, `python3 plot.py 3z <history.csv>` |

Coefficients: c = 2F / (ρ Ū² D H). Ū = 0.2 for 3D-1Z, and for 3D-3Z Ū = 1, the largest
mean inflow velocity. The z-force (`c_z` in the reference file) is not compared.

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

## Results

**2026-10-17, the 3D verification of the GPU memory work**: the decks as committed (Q3/Q2)
and at Q6/Q5 (`--set discretization.order_u=6 --set discretization.order_p=5`; the cylinder's
geometry order follows), run with main of before the GPU memory work (aa6aa7e) and main after it (changes 17-27), side by side on Bridges-2, one H100 each; 3D-3Z also with AMR (`amr.enabled`, an event every time
unit from t = 1 to 5, `threshold_mode: relative`, `theta: 0.5`, `max_elements: 2500`: the
first event refined 896 to 2,499 elements and the later ones changed nothing, so the runs
have hanging nodes on the curved cylinder throughout). Errors against FeatFlow's tabulated
values (c_D = 6.18533, c_L = 0.009401; c_D,max = 3.2978, c_L,min = -0.010999).

| Case | c_D or c_D,max | Error | c_L or c_L,min | Error | Old against new |
|---|---|---|---|---|---|
| 3D-1Z, Q3 | 6.078523 | 1.73 % | 0.009376 | 0.27 % | c_D to 7 digits |
| 3D-1Z, Q6 | 6.186173 | **0.014 %** | 0.009370 / 0.009371 | 0.3 % | c_D to 7 digits |
| 3D-3Z, Q3 | 3.21116 | 2.63 % | -0.010626 | 3.4 % | 3,557 steps both; c_D within 7e-5 |
| 3D-3Z, Q3, AMR (np 1 and 2) | 3.21312 | 2.57 % | -0.011167 | 1.5 % | equal at np 1 and 2 |
| 3D-3Z, Q6 | 3.28423 | **0.41 %** | -0.010923 / -0.010924 | 0.7 % | 11,433 steps both |
| 3D-3Z, Q6, AMR | 3.29204 | **0.17 %** | -0.011041 | 0.4 % | to t = 4.0 / 4.5 (the job's limit) |

- FeatFlow's own history (`reference/featflow_3d3z.csv`) has its extremes at c_D,max
  3.29777 (t = 3.99) and c_L,min -0.011026 (t = 3.89): the Q6 AMR run is 0.17 % and 0.14 %
  from them.
- Old against new: the forces agree within 4e-5 over the drag peak (t = 3 to 5) and the
  iteration counts within 0.5 %; the largest gap, 1e-3 in c_D, is in the first 0.2 time
  units of the start from rest, where the forces are small transients at the solver
  tolerance.
- **The Q6 decks first stopped at setup on the GPU, in both mains** (an illegal address in
  MFEM's interpolation kernels on curved meshes above 8 points per direction): changes 26
  and 27 ([../../docs/status.md](../../docs/status.md)).
