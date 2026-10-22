# In-line oscillating cylinder (Dütsch et al. 1998)

The first validation of the moving-mesh (ALE) fluid ([docs/ale_fluid.md](../../docs/ale_fluid.md)).
A circular cylinder of diameter D oscillates along x in fluid at rest,
x(t) = −A sin(2πft). Re = U_max D / ν = 100, KC = U_max / (f D) = 5; with D = U_max = 1,
ν = 0.01, f = 0.2 and A = 5 / (2π). The flow stays symmetric (no lift), and the in-line
force follows Morison's form, F_x = −½ρD c_d ẋ|ẋ| − ¼πρD² c_i ẍ.

| | |
|---|---|
| Reference | Dütsch, Durst, Becker & Lienhart, J. Fluid Mech. 360 (1998) 249–271: c_d = 2.09, c_i = 1.45 (Fourier-averaged over a period). Other computations: 2.10 / 1.45 – 1.48 |
| Deck | `dutsch_re100_kc5.yaml`: a closed 40 D × 30 D box with no-slip walls (`cylinder_channel` geometry, cylinder at its centre, O-grid of half-width 2 D); 548 Q4/Q3 elements, 17,968 velocity DOFs; IMEX at the IMEX CFL target; the mesh moved every step by pseudo-elasticity with stiffening (the motion is rigid, so every move after the first reuses that solve) |
| Compare | `python3 morison.py <name>_forces.csv`: c_d and c_i from the fundamental of C_x over the whole periods from t = 25 on, and a least-squares fit of Morison's form for comparison |

The least-squares fit gives a lower c_d (2.03) because it projects on ẋ|ẋ|, whose
higher harmonics the force does not share; the reference values are the Fourier ones.

## Results

Cloud scratch build (MFEM v4.10 from GitHub, apt hypre and Open MPI, not the Spack
toolchain), CPU, 2026-10-22, branch `claude/project-thread-xnsz2c`. Coefficients over the
periodic state (the first five periods drift: c_d 2.49, 2.12, 2.10, 2.09, ...).

| Run | Elements | Steps | Wall | c_d | c_i |
|---|---|---|---|---|---|
| deck, to t = 60, np 4 | 548 | 8,362 | 7.7 min | **2.091** | **1.460** |
| per period, periods 6–12 | | | | 2.088 – 2.093 | 1.459 – 1.461 |
| 4× the elements (`level: 1`), to t = 20, np 4 | 2,192 | 5,669 | 17.8 min | period 4: 2.089 (deck 2.087) | period 4: 1.454 (deck 1.454) |
| box 40 D × 30 D → 80 D × 60 D, to t = 34.4, np 2 | 732 | | | periods 4–6: 2.086, 2.087, 2.082 (deck 2.087, 2.088, 2.088) | 1.452, 1.456, 1.458 (deck 1.454, 1.458, 1.459) |
| half the time step (`cfl_target: 0.25`), to t = 35, np 2 | 548 | 9,716 | 38 min | periods 4–7: 2.088, 2.089, 2.090, 2.089 | 1.454, 1.458, 1.459, 1.460 (= the deck's to 4 digits) |

- **c_i = 1.460 is converged to about 0.1 %** in mesh, step and domain size here, so its
  0.7 % above Dütsch's 1.45 is the model or the reference, not resolution; the other
  computations Dütsch cites give 1.45 to 1.48. c_d agrees to 0.1 %.
- The doubled box ran out of the scratch build's hypre at t = 34.4 ("Error during solve",
  the apt hypre failure that also hits a few fast-tier tests there); its first six
  periods are in the table.
- Not compared yet: Dütsch's velocity profiles at x = −0.6, 0, 0.6, 1.2 D for phases
  180°, 210° and 330° (they need a line probe in the output).
- Not run on a GPU.
