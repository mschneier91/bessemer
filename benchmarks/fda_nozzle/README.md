# FDA benchmark nozzle

FDA's first computational round robin (Hariharan et al. 2011; Stewart et al. 2012): flow
through a nozzle with a 20° conical contraction, a throat of diameter d and a sudden
expansion, in the sudden-expansion direction. Laminar at throat Re 500, transitional around
2000 (where the jet breaks down depends on the inflow disturbances), turbulent at 6500.
FDA's PIV, pressure and blood-damage data:
https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks (not fetched by
`fetch_references.py` yet).

**Status (2026-11-11):** stage 1 of [HANDOFF.md](HANDOFF.md) has run at Re 500 on A100s, at
Q3/Q2 and Q4/Q3 (Results below). Not run: Re 500 at the defaults (Q6/Q5; a 24-step probe
only), the Re 2000 inflow sweep, the Re 6500 p-ladder, and any comparison with FDA's data.

## Files

| file | what |
|---|---|
| `fda_nozzle_mesh.py` | the structured hex mesh (gmsh): a butterfly O-grid, conforming, curved walls; every count and ratio an option |
| `fda_nozzle_tiny.msh` | a 148-hex order-2 version for tests and CPU tries (`--order 2 --nc 2 --nr 1 --no 1 --n-in 1 --n-cone 1 --n-throat 2 --n-jet 4 --n-far 1`) |
| `fda_nozzle_re500_smoke.yaml` | Re 500, Q3/Q2, t = 5: does it run, how much memory, how fast |
| `fda_nozzle_re500_q3.yaml` | Re 500, Q3/Q2, t = 1000: time to steady state, the coarse p-ladder member |
| `fda_nozzle_re500.yaml` | Re 500 at the defaults (Q6/Q5) |
| `fda_nozzle_re2000.yaml` | Re 2000, one member of the inflow-perturbation sweep |
| `run.sbatch` | one deck on GPUs (Delta header; Bridges-2 in its comments) |

## Setup

Lengths in throat diameters (d = 1), the throat's mean velocity 1, so Re = 1/ν and time is
in d/U_th. Inlet pipe D = 3 from z = −27.67 (12 long), the cone, the throat (10 long), the
expansion at z = 0, the outlet pipe to z = 30 (as Fehn, Wall & Kronbichler 2019). The
inflow is Poiseuille flow in the inlet pipe (`profile: pipe`, mean velocity 1/9), ramped up
over two time units, with optional travelling perturbations (`perturbation`). Walls
no-slip, the outlet do-nothing.

The level-0 mesh (`--level 0 --order 3`, 56,320 hexes, 126 MB) is generated, not committed:

```sh
mkdir -p runs/fda_nozzle
python3 benchmarks/fda_nozzle/fda_nozzle_mesh.py --level 0 --order 3 \
    -o runs/fda_nozzle/fda_nozzle_l0.msh
```

At Q3/Q2 it has 4.63M velocity and 0.46M pressure DOFs, at Q6/Q5 about 43.5M in all.
Level 1 has 8× the hexes.

**At Re 500 the outlet to z = 30 is too short** (Results): the recirculation behind the
expansion reattaches at z = 52. The Re 500 results use an outlet to z = 90 (87,040 hexes):

```sh
python3 benchmarks/fda_nozzle/fda_nozzle_mesh.py --level 0 --order 3 \
    --outlet-D 30 --n-far 60 --far-ratio 1.03 -o runs/fda_nozzle/fda_nozzle_l0_o30.msh
```

Each deck samples time averages (`flow_statistics`): the centreline, radial profiles at
FDA's axial stations, and wall pressure and shear stress in axial bins.
`benchmarks/flow_stats.py` reduces them (jet breakdown along the centreline, separation and
reattachment along the wall).

## Results

**Stage 1: Re 500 at Q3/Q2 and Q4/Q3** (recorded 2026-11-11; human: "Yes, record them").
NCSA Delta, A100 40 GB, one GPU per rank, the CUDA build of main 7aae328. The decks are
copies of `fda_nozzle_re500_q3.yaml` with the changes below; they are in
`runs/fda_nozzle/decks/` of the Delta checkout, not committed. OIFS, BDF3, CFL target 2 (the
deck's settings). Step counts, iterations and times are from the per-step histories, over
every restart segment; the long-outlet runs were restarted from checkpoints several times.

| Run | Mesh | Orders | DOFs (u + p) | GPUs | End time | Steps | dt | Outer iterations per step | Time per step | Stepping time |
|---|---|---|---|---|---|---|---|---|---|---|
| short outlet | 56,320 hexes, outlet to z = 30 | Q3/Q2 | 4.63M + 0.46M | 1 | 180 | 4,753 | 0.001 to 0.079 | 5.5 | 0.78 s | 1.0 h |
| long outlet | 87,040 hexes, outlet to z = 90 | Q3/Q2 | 7.14M + 0.71M | 1 | 350 | 8,773 | 0.001 to 0.079 | 6.2 | 1.40 s | 3.4 h |
| long outlet | the same | Q4/Q3 | 16.87M + 2.38M | 2 | 350 | 13,787 | 0.001 to 0.066 | 6.1 | 2.03 s | 7.8 h |

Every run ended with status ok and no rejected step. After t = 100 the steps settle at dt
0.047 (Q3) and 0.030 (Q4), with 3.5 to 4.9 outer iterations each. Peak device memory of the
Q4 run: 19.1 GiB per GPU (`output.profile: sync`, its last segment); not measured at Q3.

Changes to the committed deck: the short-outlet run stops at t = 180 and averages from
t = 100. The long-outlet runs use the mesh of "Setup", stop at t = 350, average over
t = 200 to 350, sample the centreline to z = 90, add profile stations at z = 25 to 80 and
bin the far wall from z = 15 to 90 in 75 bins. Below, velocities are in the throat's mean
velocity, pressures in its square, and z in throat diameters from the expansion.

- **The outlet to z = 30 is too short at Re 500.** The short-outlet run is steady by
  t = 180: its two half windows differ by 2.9e-5 in centreline velocity, relative. But its
  wall shear stress is negative through the last wall bin (z = 29.5): the recirculation
  leaves through the outlet. On the long outlet the flow reattaches at z = 52.4. Upstream of
  z = 25 the outlet length does not matter: the two Q3 runs' centreline velocities differ by
  at most 3e-5 for 0 < z < 25 and 2.3e-4 upstream of the expansion.
- **Q3 and Q4 agree downstream of the throat entrance** (long outlet, the same window):

  | Quantity | Q3/Q2 | Q4/Q3 | Difference |
  |---|---|---|---|
  | Peak centreline velocity (z = 0) | 1.8091 | 1.8082 | 0.05 % |
  | Centreline velocity, z > 0 | | | at most 1.6e-3 (0.09 % of the peak) |
  | Centreline pressure, z > 0 | | | at most 7.2e-4 |
  | Reattachment (wall shear changes sign) | z = 52.42 | z = 52.37 | 0.1 % |
  | Wall pressure, z > 0 | | | at most 7.0e-4 (its range there: 0.16) |
  | Wall shear stress, z > 0 | | | at most 9.2e-6 (its range there: 1.3e-3) |

  Centreline velocity, Q3 / Q4: z = 10: 1.5457 / 1.5441; z = 20: 1.2231 / 1.2217; z = 40:
  0.7087 / 0.7075; z = 60: 0.3670 / 0.3668; z = 80: 0.2300 / 0.2301.
- **Q3 is not resolved where the cone meets the throat (z = -10).** Wall shear stress in
  the bins from z = -11.1 to -9.1 differs by up to 4.5e-3, 11 % of its local peak (0.0370
  against 0.0414 in the peak's bin), with the sign alternating from bin to bin; wall
  pressure by 2.6e-2 (1.6 % of the pressure drop, 1.57); centreline velocity by 6.8e-3 and
  centreline pressure by 8.2e-3. One bin further downstream the wall shear agrees to 0.6 %.
- **Neither long-outlet run is steady beyond z = 65 at t = 350.** The Q4 run's two half
  windows (75 time units each) differ in centreline velocity by less than 5e-10 for z < 20,
  1.1e-5 for 20 < z < 45, and 2.7e-2 at z = 70 to 73, where the velocity is about 0.27. Q3
  and Q4 agree there to 1.3e-4, so both follow the same slow transient. The pressure level
  differs by 1.1e-3 to 1.6e-3 between the half windows at every z. Whether the far
  downstream settles is not known; the values beyond z = 65 are not a steady answer.

What these runs do not establish: one mesh at two orders is not a mesh-converged answer;
the radial profile sets were not compared; nothing was compared with FDA's data. Re 500 at
the defaults (Q6/Q5) has only a probe on the short-outlet mesh: 24 steps to t = 0.3 at
36.8M velocity DOFs, status ok. The smoke deck ran before these (Q3/Q2, short outlet, 302
steps to t = 5 in 5.4 min, status ok).
