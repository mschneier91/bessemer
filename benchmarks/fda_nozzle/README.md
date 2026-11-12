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
only), the Re 2000 inflow sweep and the Re 6500 p-ladder. Against FDA's Re 500 data the
velocities fall within or close to the five labs' spread; the wall pressure differs from the
experiment as every code's in FDA's study did (Results).

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
the two orders' radial profile sets were not compared with each other. Re 500 at
the defaults (Q6/Q5) has only a probe on the short-outlet mesh: 24 steps to t = 0.3 at
36.8M velocity DOFs, status ok. The smoke deck ran before these (Q3/Q2, short outlet, 302
steps to t = 5 in 5.4 min, status ok).

**Re 500 against FDA's experiments** (2026-11-11; human: "the ultimate goal is to make sure
we are matching the FDA results", "Yes, record it"). `SE_exp_0500.zip` of FDA's repository:
five labs' PIV (centreline and radial profiles of the axial velocity), four labs' wall
pressures; SI units. Its stated flow rate (5.20624e-6 m³/s), throat diameter (4 mm), density
(1056 kg/m³) and viscosity (0.0035 Pa s) give a throat mean velocity U = 0.4143 m/s, ρU² =
181.3 Pa and Re 500.0; the data are scaled with these. FDA's profile stations, in diameters:
z = -22, -16, -12, -5, -2, 0, 2, 4, 6, 8, 15, 20, so the data end at z = 20, where neither the
outlet length nor the unsteady far field matters. The long-outlet Q4/Q3 run is compared; Q3
gives the same numbers to three digits at every station. The comparison scripts are not
committed, and the data are not fetched by `fetch_references.py` yet.

Centreline axial velocity, in U:

| z | Labs, min to max | Labs, mean | Run |
|---|---|---|---|
| -22 | 0.201 to 0.224 | 0.210 | 0.222 |
| -16 | 0.205 to 0.216 | 0.212 | 0.224 |
| -12 | 0.456 to 0.571 | 0.482 | 0.454 |
| -10.5 (3 labs) | 0.911 to 0.919 | 0.914 | 0.894 |
| -5 | 1.589 to 1.725 | 1.642 | 1.632 |
| -2 | 1.693 to 1.854 | 1.770 | 1.755 |
| 0 | 1.712 to 1.818 | 1.757 | 1.808 |
| 2 | 1.673 to 1.786 | 1.752 | 1.773 |
| 4 (4 labs) | 1.620 to 1.741 | 1.697 | 1.726 |
| 6 (4 labs) | 1.572 to 1.635 | 1.610 | 1.671 |
| 8 | 1.515 to 1.579 | 1.539 | 1.609 |
| 10 (3 labs) | 1.473 to 1.538 | 1.508 | 1.544 |
| 12 (3 labs) | 1.445 to 1.518 | 1.483 | 1.478 |
| 15 | 1.072 to 1.417 | 1.295 | 1.379 |
| 20 (4 labs) | 1.006 to 1.239 | 1.108 | 1.222 |

- **Centreline:** the run is inside the labs' range at 9 of the 15 points and within 2.2 % of
  the nearest lab at the others, except z = -16 (3.7 % above the highest lab). In the inlet
  pipe the run has developed pipe flow's 2/9; the labs average 5 % below it.
- **Radial profiles** at the 11 stations the deck shares with FDA (each lab's two sides
  averaged; 20 radii per station; "inside" allows 0.01 beyond the labs' lowest and highest):
  the run is inside the labs' envelope at 20 of 20 radii at z = -22, -16, -5, -2 and 15, 18 at
  z = 0 and 2, 19 at z = 20, and 14 at z = 4. Its RMS distance from the labs' mean is 0.008 to
  0.061 there; the labs' own RMS spread about their mean is 0.004 to 0.063.
- **At z = 6 and 8 the run is 2 to 7 % above every lab across the jet** (inside at 9 of 20
  radii; RMS distance 0.040 and 0.044 against a lab spread of 0.009 and 0.011). The labs'
  profiles there integrate to 0.92 to 0.95 and 0.88 to 0.96 of the nominal flow rate; the
  run's to 1.001. With each lab's profile scaled to the nominal flow rate the run is inside
  at 18 and 20 of 20 radii. That reading (a flow-rate offset in the measurements, not a
  different jet) is the agent's; it is not established.
- **These runs sampled z = -10.5, which is not one of FDA's profile stations, and not FDA's
  z = -12 (in the cone).** The handoff's stations were from memory. The decks now have
  z = -12 (profile set `cone`, to r = 0.85; the wall is at 0.8527) in place of -10.5
  (2026-11-11). The changed decks have not been run or loaded by a test here: no build on
  the machine they were edited on.

Wall pressure relative to z = 0, in Pa at FDA's scales (ρU² = 181.3 Pa):

| z | Labs 243, 468, 763 (mean) | Lab 999 | Run |
|---|---|---|---|
| -22.6 | 484, 544, 551 (526) | 222 | 311 |
| -12.9 | 393, 469, 468 (443) | 142 | 304 |
| -10.0 | 262, 251, 234 (249) | 117 | 193 |
| -7.5 | 183, 185, 165 (178) | 72 | 113 |
| -5.0 | 131, 123, 110 (121) | 26 | 71 |
| -2.5 | 26, 38, 28 (30) | 9 | 33 |
| 0.5 | 131, 90, 83 (101) | 14 | -1 |
| 2.0 | 157, 117, 83 (119) | -16 | -1 |
| 8.0 | 131, 145, 138 (138) | -7 | 2 |

- **The run's pressure does not match the experiment, and neither did any code in FDA's
  study.** Three labs put the inlet about 530 Pa above z = 0 and show a rise of 100 to 140 Pa
  within half a diameter after the expansion; the run gives 311 Pa and no rise. Stewart et
  al. (2012), Figure 4a, shows the same for every code: the laminar solutions near 310 Pa
  upstream (all codes about 280 to 390 Pa, read from the figure) and 0 to 25 Pa downstream,
  none with the rise. The paper: "Experimental errors at the z = 0.0 m normalization point
  may partly explain the 100-250 N/m² ... offset at Ret = 500. Errors were traced to not
  using differential pressure transducers in some of the experiments." One lab's pressures
  are left out of that figure for measurement errors; by the numbers it is lab 999.
- So the run's pressure agrees with the study's laminar CFD, and the experiment cannot
  validate it more closely. About 100 Pa of the upstream difference is left unexplained by
  the paper ("partly").

Not compared: FDA's wall shear stress data (24 points per lab); the shear and Reynolds
stress profiles. Read of Stewart et al. (2012): the pressure passages and Figures 3 and 4,
not the whole paper.
