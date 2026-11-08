# FDA benchmark nozzle

FDA's first computational round robin (Hariharan et al. 2011; Stewart et al. 2012): flow
through a nozzle with a 20° conical contraction, a throat of diameter d and a sudden
expansion, in the sudden-expansion direction. Laminar at throat Re 500, transitional around
2000 (where the jet breaks down depends on the inflow disturbances), turbulent at 6500.
FDA's PIV, pressure and blood-damage data:
https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks (not fetched by
`fetch_references.py` yet).

**Status (2026-11-08):** set up, not yet run on a GPU. The decks load, and the Q3 smoke deck
runs on the level-0 mesh on a CPU (HANDOFF.md, last section). [HANDOFF.md](HANDOFF.md) is the
run plan: Re 500 smoke test, Re 500 at Q3 and Q6, then the Re 2000 inflow sweep and the
Re 6500 p-ladder.

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

Each deck samples time averages (`flow_statistics`): the centreline, radial profiles at
FDA's axial stations, and wall pressure and shear stress in axial bins.
`benchmarks/flow_stats.py` reduces them (jet breakdown along the centreline, separation and
reattachment along the wall).

## Results

None yet.
