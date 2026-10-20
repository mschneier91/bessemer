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

**The decks as committed, at the defaults** (2026-10-19; human: the verification
benchmarks "using the defaults we agreed on", AMR for the cylinder). Q6/Q5 (the cylinder's
geometry order follows) with AMR: an event every time unit for t in [1, 5], relative
marking θ = 0.5, at most 2,500 elements, one hanging-node level; no overrides. main a5d8ed0
(the decks) run by the CUDA build of 5da1553 (changes 28-56; the decks change no code);
Bridges-2, one H100 each (GPU-shared), np 1. Errors against FeatFlow's tabulated values;
peak memory by `nvidia-smi` over the run.

| Deck | Elements | DOFs | Steps | Wall | Peak memory | c_D or c_D,max | Error | c_L or c_L,min | Error |
|---|---|---|---|---|---|---|---|---|---|
| 3D-1Z | 896 → 2,506 | 1.97M | 21,245 | 48 min | 6.5 GiB | 6.184972 | **0.006 %** | 0.009398 | **0.03 %** |
| 3D-3Z | 896 → 2,506 | 1.97M | 19,898 | 27 min | 8.5 GiB | 3.29204 (t = 3.990) | **0.17 %** | −0.011042 (t = 3.891) | 0.39 % |

- **3D-1Z: AMR takes the lift error from 0.3 % to 0.03 %** and the drag's from 0.014 % to
  0.006 % (the 2026-10-17 Q6 run below, 896 elements, no AMR; 14,920 steps in 27 min). The
  first event (t = 1) refines the wake and the cylinder region to 2,506 elements; the later
  ones change nothing (`amr_events` 1 in the summary).
- **3D-3Z against FeatFlow's own history** (its extremes 3.29777 at t = 3.990 and −0.011026
  at t = 3.893): 0.17 % and 0.15 %, at the same times to 0.002. From t = 0.5 on, c_D stays
  within 5.9e-3 of that history (0.18 % of c_D,max) and c_L within 4e-4 (1.2e-4 from
  t = 1). `plot.py`'s largest gap over the run, 0.10 in c_D, is the first step from rest.
- **The same answers as the 2026-10-17 AMR run below** (c_D,max 3.29204, c_L,min
  −0.011041), now to t = 8 (that run stopped at t = 4.0 / 4.5 at its job's limit): the
  same steps (289, 1,897 and 6,982 in t = 0-1, 1-2, 2-4) and outer iterations (2.45, 3.00,
  5.28 per step against 2.50, 2.99, 5.34); 2,506 AMR elements against 2,499. 6.6 to 8.4×
  faster per step (0.068 to 0.092 s against 0.57 to 0.61 s): changes 28-56 between, and
  the 2026-10-17 lanes shared a node, which slowed the small DFG cases up to 3.5×.

Earlier results, Q3/Q2 (the decks' setting until 2026-10-19) and the first Q6 runs:

**2026-10-17, the 3D verification of the GPU memory work**: the decks as committed then (Q3/Q2)
and at Q6/Q5 (deck copies with `order_u: 6`, `order_p: 5`; the cylinder's
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
