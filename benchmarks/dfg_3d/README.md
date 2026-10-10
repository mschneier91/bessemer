# DFG flow around a cylinder in 3D (3D-1Z, 3D-3Z)

The 3D versions of the DFG benchmarks (Schäfer & Turek 1996): a channel 2.5 × 0.41 × 0.41
with a cylinder of diameter 0.1 spanning it along z at (0.5, 0.2). The inflow is
16 U_m y z (H − y)(H − z) / H⁴. Every channel wall is no-slip, including the z faces, and
the outflow is do-nothing.

| | |
|---|---|
| 3D-1Z | steady, U_m = 0.45 (Re 20). Reference c_D = 6.18533, c_L = 0.009401 (featflow.de) |
| 3D-3Z | U_m(t) = 2.25 sin(πt/8), t ∈ [0, 8]. FeatFlow's finest: c_D,max = 3.2978, c_L,min = −0.010999; histories in `reference/featflow_3d3z.csv` |
| Mesh | the 2D DFG mesh extruded in z (`mesh.dim: 3` with `geometry: cylinder_channel`), curved cylinder of the velocity order |
| Compare | `python3 plot.py 1z <summary.json>`, `python3 plot.py 3z <history.csv>` |

Coefficients: c = 2F / (ρ Ū² D H). Ū = 0.2 for 3D-1Z, and for 3D-3Z Ū = 1, the largest
mean inflow velocity. The z-force (`c_z` in the reference file) is not compared.

The reference CSV is fetched, not committed: `python3 benchmarks/fetch_references.py`.

Results: none yet.
