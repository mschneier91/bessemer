<!-- The human's proposal of 2026-10-16, kept as written. What was built and where it
     departs from this: docs/uzawa.md. -->

# Schur Complement Conjugate Gradient (Uzawa) vs Coupled Flexible GMRES

Status: proposal (2026-10-16). Implemented as an option, `solver.outer: schur`; see docs/uzawa.md.
Goal: cut Krylov memory by roughly 8x at equal or better solver accuracy, with wall time at parity to about 1.3x of the current coupled solver, so that resident problem size per H100 rises from about 50M to about 70M degrees of freedom with no change to the discretization.

## 1. What is being solved

Per time step (BDF2 with extrapolated convection, grad-div included in the velocity block):

```
[ A   B^T ] [ u ]   [ f ]
[ B    0  ] [ p ] = [ g ]        A = (alpha/dt) M + nu K
```

`A` is symmetric positive definite while convection stays explicit. `B` is the discrete divergence.

Eliminating `u` from the momentum row and substituting into continuity gives a single equation on the pressure space:

```
S p = B A^{-1} f - g,        S = B A^{-1} B^T
```

`S` is the Schur complement: symmetric positive semidefinite (null space is constant pressures on enclosed or periodic domains), dense, never assembled. Its action on a vector is gradient, velocity solve, divergence. Once `p` is known, `u = A^{-1} (f - B^T p)`.

Both the current coupled method and the proposed method solve this same pressure equation with the same preconditioner. They differ only in where the Krylov recurrence runs and in how inexact inner solves are tolerated.

## 2. Preconditioner (unchanged)

Cahouet-Chabard:

```
S~^{-1} = (alpha/dt) L_p^{-1} + nu M_p^{-1}
```

`L_p^{-1}` is a fixed number of low-order-refined BoomerAMG V-cycles. `M_p^{-1}` is a few Jacobi-conjugate-gradient iterations or a lumped inverse. It is exact in both the small-dt limit (S looks like a pressure Laplacian) and the large-dt limit (S looks like a pressure mass matrix) and interpolates between them. The spectrum of `S~^{-1} S` sits in an interval independent of h, so the outer iteration count `N_S` does not grow under refinement. Expect roughly 10 to 25 iterations, more with an outflow boundary.

The Olshanskii-tensor Schur preconditioner used for the stiff adaptive cases is not symmetric. See section 4.

## 3. Current method: coupled flexible GMRES

Flexible GMRES on the full coupled system of size `3 n_v + n_p`, preconditioned by the block-triangular factor with `S` replaced by `S~`. Each outer iteration:

1. apply the preconditioner to the coupled residual: one (1,1) solve with `A` (Jacobi-conjugate-gradient, loose tolerance), one Cahouet-Chabard application
2. apply the coupled operator `K` to the resulting correction (one `A` apply, one `B`, one `B^T`, no solves)
3. orthogonalize the new coupled-length vector against all previous ones (Gram-Schmidt)
4. solve the small least-squares problem, update the iterate, check the coupled residual

The preconditioned operator has identity on the velocity block and `S~^{-1} S` on the pressure block, so convergence is governed by the pressure spectrum alone. The velocity component of each Krylov vector is carried along but never iterated.

Properties:

- Inner (1,1) solve sits inside the preconditioner. Any inexactness is absorbed by flexible GMRES. Loose inner tolerances are free.
- Cahouet-Chabard may be a variable operator (iterate-to-tolerance is allowed).
- Storage: at restart 30, up to 60 coupled-length vectors (`Z` and `V` bases). On the current footprint this is roughly 480 bytes per degree of freedom out of about 1.5 kB total.
- Per-iteration orthogonalization streams up to `k` coupled-length vectors at iteration `k`. On a bandwidth-bound code this is comparable to several matrix-free operator applies late in a restart cycle.
- GMRES has no symmetric structure to exploit; restart truncation can cost iterations.

## 4. Proposed method: Schur complement conjugate gradient

The same elimination performed symbolically before iterating. The Krylov method runs on `S p = b` in the pressure space.

### Systems solved per time step

Before the loop:

```
System 1 (velocity space):   A u_0 = f              Jacobi-CG, warm start from previous velocity
                             r = B u_0 - g          project out mean
                             p = previous pressure; r = r - S p   (one System 2a)
                             z = S~^{-1} r;  d = z
```

Loop, k = 1, 2, ... until `||r|| &lt;= tol` (the count is `N_S`):

```
System 2a (velocity space):  A v = B^T d            Jacobi-CG, tolerance eps_k;  q = B v  (this is S d)
Update:                      gamma = (r,z)/(d,q);  p += gamma d;  r -= gamma q
Check:                       stop if ||r|| &lt;= tol
System 2b (pressure space):  L_p y1 = r             fixed BoomerAMG V-cycles
System 2c (pressure space):  M_p y2 = r             fixed Jacobi-CG iterations or lumped
Direction:                   z_new = (alpha/dt) y1 + nu y2;  beta = (r,z_new)/(r_old,z_old);  d = z_new + beta d
```

After the loop:

```
System 3 (velocity space):   A u = f - B^T p        Jacobi-CG, warm start from u_0
```

This is textbook preconditioned conjugate gradient on `S`. The operator application (normally a sparse matvec) is System 2a; the preconditioner application is 2b plus 2c. In MFEM terms: a `CGSolver` given an `Operator` whose `Mult` does &#34;solve A v = B^T d, return B v&#34; and a `Solver` whose `Mult` does Cahouet-Chabard.

### Cost per time step

- `N_S + 2` solves with `A` (System 1, `N_S` copies of 2a, System 3, plus one more 2a if the initial pressure guess is nonzero)
- `N_S + 1` multigrid applications and mass solves
- Storage in the loop: five pressure-length vectors (`p, r, z, d, q`) plus four velocity-length vectors for the inner conjugate gradient

### Where the roles flip

In the coupled method the (1,1) solve is inside the preconditioner. Here it is inside the operator `S`. Conjugate gradient assumes the operator is the same linear map every iteration; an inner solve stopped at a loose tolerance violates that. The symptom is stagnation or a recurrence residual that drops below tolerance while the true residual has not, giving a wrong pressure. This is the one genuine disadvantage of the Schur formulation and it has three fixes (section 5).

The Cahouet-Chabard preconditioner does not need to be accurate, same as in the coupled method. For plain conjugate gradient it does need to be a fixed linear operator: fixed V-cycle counts, fixed mass iteration counts, never iterate-to-tolerance.

## 5. Options for the outer pressure-space solver

| Option | Use when | Plus | Minus | Extra storage |
|---|---|---|---|---|
| Plain preconditioned conjugate gradient | `A` symmetric, Cahouet-Chabard preconditioner | Three-term recurrence, no orthogonalization, cleanest theory | Needs consistent inner (1,1) solve and fixed preconditioner | none beyond the five vectors |
| Flexible conjugate gradient (Notay) | Same, but with inexact inner solves or a variable preconditioner | Tolerates varying operator and preconditioner; about twenty lines on top of plain CG | Loses the strict optimality guarantee; one extra pressure vector | one pressure vector |
| Pressure-space flexible GMRES, restart 10 to 30 | `S` nonsymmetric: Olshanskii preconditioner, or the lagged-vorticity rotational term once it enters `A` | Handles everything; still pressure-length vectors | Orthogonalization returns, but over `n_p` not `3 n_v + n_p`; restart truncation | `2 k` pressure vectors |

Inner (1,1) tolerance strategies for the conjugate gradient options:

| Strategy | Plus | Minus |
|---|---|---|
| Tight fixed (1e-10 relative) | Simplest; `S` effectively exact | 1.5x to 2x more inner iterations than loose. In the small-dt regime `A` is mass-dominated and this is maybe 10 to 20 Jacobi-CG iterations versus 5 to 10, on the cheaper block |
| Loose fixed (1e-3 relative) | Cheapest per iteration | Breaks conjugacy; do not use with plain CG, only with flexible CG or GMRES |
| Relaxed schedule, `eps_k = min(eps_max, c * tol / ||r_{k-1}||)` | Near loose cost on average with plain CG | Tight early, loose late (inexact Krylov theory of van den Eshof and Sleijpen, Simoncini and Szyld: late corrections carry small weights). Needs a floor and a true-residual check |

Recommended default: plain conjugate gradient with the relaxed schedule for Cahouet-Chabard; flexible conjugate gradient as the fallback if the true-versus-recurrence residual log shows drift; pressure-space flexible GMRES at the smallest restart that does not raise the count for the Olshanskii branch.

## 6. Expected performance versus coupled flexible GMRES

Accuracy: identical. Both converge to the same saddle-point system; the discrete solution is set by the discretization and the stopping tolerance, not by which space the recurrence runs in. Any observed difference beyond tolerance is a bug.

Outer iterations: equal or slightly fewer. The two methods see the same spectrum of `S~^{-1} S`. Conjugate gradient on a symmetric positive definite operator does at least as well as GMRES on the nonsymmetric block-triangular-preconditioned system, and has no restart truncation.

Inner work: `N_S + 2` versus `N_S` solves with `A`, with the Schur inner solves tighter. Net roughly 1.0x to 1.5x more velocity-block work. The velocity block is the minority of the cost in the high-Reynolds-number cases, where the pressure Schur solve dominates.

Orthogonalization: removed for the conjugate gradient options, reduced to pressure-length for the GMRES option. Potential throughput gain not counted in the wall-time estimate below.

Wall time per step: parity to about 1.3x, with the relaxed schedule or flexible CG closing most of the gap. Measure; see section 7.

Memory: Krylov storage drops from about 60 coupled-length vectors to about 5 pressure-length plus 4 velocity-length vectors, roughly 8x. On the current footprint about 1.5 kB per degree of freedom becomes about 1.1 kB, so about 50M becomes about 70M per 80 GB H100.

When the coupled method would still be the better choice: a convection-dominated or large-dt regime where the (1,1) block is stiff and a very loose inner solve is a large saving, or a nonsymmetric `A` where the conjugate gradient advantage is gone and only the storage advantage remains. Neither is the regime of the current stability-limited bluff-body and channel runs. Once the semi-implicit rotational term goes into `A`, the Schur operator becomes nonsymmetric and the pressure-space GMRES option is the one that applies; decide the code structure with that in mind.

## 7. Implementation and measurement plan

Phase 0, baseline (half a day, 1 GPU): instrument the coupled solver per step for outer iterations, inner iterations per outer, multigrid cycles, mass solves, wall time per phase (velocity solve, pressure preconditioner, Gram-Schmidt, operator applies), peak device memory. Fix the stopping rule for both methods as the same coupled residual `||b - K x|| / ||b||`; the Schur method computes it once at the end of each solve during the study. Cases: Taylor-Green Re 1600 at p = 3, 5, 7; channel restart windows at Re_tau 180 and 550; periodic hill; square cylinder with adaptive refinement (outflow boundary, Cahouet-Chabard weakest). 20 to 50 steps per run.

Phase 1, implementation (one component per .hpp/.cpp pair):

1. `SchurOperator` (mfem::Operator on the pressure space): `Mult` solves `A v = B^T d` with the existing Jacobi-CG and returns `B v`. Settable inner tolerance; iteration counters.
2. `CahouetChabardSolver` (mfem::Solver on the pressure space): factored out of the current block preconditioner; fixed V-cycle and mass iteration counts; mean projection on input and output.
3. `SchurSolver` (top level): Systems 1, 2, 3 with warm starts; outer method selectable among `CGSolver`, flexible CG, `FGMRESSolver`.
4. Inner tolerance modes: tight fixed, loose fixed, relaxed schedule. Log recurrence residual and true residual every iteration.
5. Hook into the per-step diagnostic switch so coupled and Schur are runtime-selectable with identical tolerances.

Phase 2, correctness (1 GPU, short runs): Taylor-Green velocity, pressure, kinetic energy dissipation and divergence norm must agree between methods to the solver tolerance. Null space handling on periodic and wall-bounded cases.

Phase 3, performance:

- iteration counts versus dt (factors of 2 around the stability limit) and p (3, 5, 7) on Taylor-Green and channel 180; repeat on the square cylinder for the outflow penalty
- inner tolerance study: tight versus loose versus relaxed, total velocity operator applies per step
- orthogonalization cost of the coupled method measured separately
- Olshanskii branch: pressure-space FGMRES at restart 10, 20, 30 on the stiff adaptive case
- capacity: bisect mesh size on channel 550 for the largest problem fitting one H100 under each method (two or three bisection steps, the only full-size runs)

Decision rule: adopt Schur as default if wall time per step is within 1.2x of coupled at matched coupled residual and peak memory drops at least 25 percent. Keep the coupled path behind the switch regardless.

Risks: conjugate gradient stagnation from inexact inner solves (caught by the residual log, flexible CG is the fallback); an inner iterate-to-tolerance sneaking into the preconditioner; the outflow case degrading Cahouet-Chabard enough that the Olshanskii path has to carry the stiff runs either way.

## 8. One-paragraph summary

The coupled flexible GMRES solver and the Schur complement conjugate gradient solver are the same elimination of the velocity block, done numerically inside each iteration versus symbolically before iterating. They use the same (1,1) solver, the same Cahouet-Chabard preconditioner, see the same spectrum, and converge to the same answer in the same number of outer iterations. The Schur formulation moves the (1,1) solve from preconditioner to operator, which costs some inner-solve accuracy and gains symmetric conjugate gradient, no orthogonalization, and an 8x reduction in Krylov storage. For a capacity-limited code in the small-dt regime, that trade is favorable.
