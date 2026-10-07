# Spec: adaptive mesh refinement (refinement only, isotropic or anisotropic)

> **Status (2026-10-06): IMPLEMENTED** on branch `amr` as sub-sprints AMR.0–AMR.6 (plus
> AMR.4b for D6), with D1–D8 taken at the proposals in Section 12. Two deviations, both
> recorded in CLAUDE.md ("Adaptive mesh refinement"): (1) checkpoint/restart replays the
> logged refinement history instead of ParPrint-ing the mesh (5.6 — a re-read mesh need not
> number its dofs like the original); (2) the 2D periodic restart case and the TGV flow test
> skip on the debug device because of an MFEM debug-device false positive found along the
> way (pure-MFEM reproduction in CLAUDE.md). Section 8.3's open items were all verified by
> the tests (rebalance on 3D anisotropic NC meshes, LimitNCLevel after conflict resolution,
> restart of periodic adapted runs, the full solve on NC meshes, PBJ near hanging nodes).

Target: bessemer `main` at `af37b42`, built against MFEM `develop` at commit `17d1afc3`
(the desktop environment's installed MFEM; source in spack's cache as
`17d1afc3b78c55b17316dc262604a83bd91364d0.tar.gz`). Every MFEM file, line and behaviour cited
below was checked against that commit's source. Section 8.2 lists what was also checked by
running code on this machine. If MFEM moves, re-verify Section 8 before writing code.

Audience: an agent implementing this in bessemer, and the human reviewing it. Read the whole
document before writing code. Follow CLAUDE.md's sub-sprint protocol (Section 11): one
sub-sprint at a time, green at np ∈ {1, 2, 4}, then stop for review.

---

## 0. Summary

Add h-adaptive, **refinement-only** AMR to the unsteady solver (Stokes and NSE, convective and
rotational forms):

- Every `amr.interval` accepted steps (and optionally before the first step), compute a
  **directional velocity-gradient indicator** per element (Section 2), mark elements above a
  threshold (Section 3), and refine them either **isotropically** (split in every direction) or
  **anisotropically** (split only in the directions in which the velocity varies).
- Refined quad/hex meshes are **nonconforming** (hanging nodes). MFEM enforces continuity
  through the prolongation P. The existing PA operators are assembled on the refined mesh
  unchanged.
- **No derefinement.** The mesh only grows; `amr.max_elements` bounds it.
- Refinement nests the spaces, so **state transfer is exact**: the coarse velocity is exactly
  representable on the refined mesh. That is what makes this cheap to validate. The polynomial
  manufactured solutions stay exact through a refinement event.
- An adaptation event is structured as an **in-memory checkpoint and restart**: export the
  integrator state, refine, transfer, rebuild the integrator and every solver, then re-import the
  state. The restart path exists today and is tested to reproduce an uninterrupted run to 1e-13.
- **AMR off (the default) changes nothing**: no code path, result or baseline moves.

---

## 1. Background: what refinement does to the discretization

### 1.1 Nonconforming quads and hexes

Refining a quad or hex locally leaves hanging nodes on the faces and edges it shares with
unrefined neighbours. MFEM stores the mesh as a refinement tree (`NCMesh`). The H1 space keeps
every element's dofs in the L-vector, but hanging (slave) dofs are not true dofs; they are
interpolated from master dofs, $u_L = P u_T$. On a nonconforming mesh P is a general sparse
matrix (`HypreParMatrix`) rather than the Boolean conforming prolongation. The system the solver
sees is $P^T A_L P$, which is conforming. PA operators act on E- and L-vectors and need no
change; P and Pᵀ apply the constraints.

### 1.2 Refinement types

MFEM's `Refinement` (`mesh/ncmesh.hpp:40`) carries, per element, a mask `X = 1, Y = 2, Z = 4`
(`XY = 3`, `XYZ = 7`): the element is halved along each **reference** direction in the mask.
2D uses X, Y, XY; 3D uses any non-empty mask. "Direction d" always means the element's d-th
local coordinate $\xi_d$, not a physical axis. On bessemer's box meshes, stretched or not, the
two coincide, which is what makes anisotropic refinement meaningful there.

### 1.3 Exact transfer

Under refinement, $V_h^{\text{coarse}} \subset V_h^{\text{fine}}$ for both the velocity space
($Q_{k_u}$) and the pressure space ($Q_{k_p}$). MFEM's update path (`FiniteElementSpace::Update`
then `GridFunction::Update`) interpolates the coarse function at the fine nodes, which
reproduces it exactly. Section 8.2 records the check.

What is **not** preserved is discrete incompressibility with respect to the new pressure space;
Section 5.4 deals with it.

---

## 2. The indicator

### 2.1 Definition

For element K with reference map $x = F_K(\xi)$, $\xi \in [0,1]^{\dim}$, and the current
velocity $u_h$:

$$G_{K,d} = \Big(\frac{1}{|K|}\int_K \sum_{c=1}^{\dim}\Big(\frac{\partial u_{h,c}}{\partial \xi_d}\Big)^2 dx\Big)^{1/2},
\quad d = 1,\dots,\dim, \qquad \eta_K = \Big(\sum_d G_{K,d}^2\Big)^{1/2}.$$

$\partial u/\partial\xi_d = (\nabla_x u)\, J_{:,d}$ is the derivative of u along the element's
d-th edge direction, times that edge's length. So $G_{K,d}$ is the RMS change of the velocity
across the element in its d-th direction, in velocity units. On a cube of side h,
$\eta_K = h\,\|\nabla u\|_{\text{RMS},K}$: it reduces to the standard "h times gradient"
indicator.

Why this form:

- **Anisotropic without extra machinery.** On a cell stretched in x, $G_{K,x}$ sees the long
  edge. Halving the element in direction d halves $J_{:,d}$, so for smooth u it halves
  $G_{K,d}$ and leaves the other directions unchanged. Marking the directions with large
  $G_{K,d}$ (Section 3.2) therefore targets exactly the splits that reduce the indicator.
- **Velocity units**, so an absolute threshold can be stated relative to the velocity scale
  (U = 1 in bessemer's nondimensional runs).
- **Cheap and GPU-friendly**: reference derivatives at quadrature points come from MFEM's
  sum-factorized `QuadratureInterpolator`, on the device, followed by one small per-element
  reduction.

Properties the tests rely on (exact, any partition):

- **I1 (linear field).** For $u = Ax$ with A constant, on an affine element
  $\partial u/\partial\xi_d = A J_{:,d}$ is constant and $G_{K,d} = |A J_{:,d}|$ (Euclidean norm
  over components).
- **I2 (direction blindness).** If u varies only in x on an axis-aligned element,
  $G_{K,y} = G_{K,z} = 0$.
- **I3 (halving).** For linear u on affine elements, each child of a d-split of K has
  $G_{\text{child},d} = G_{K,d}/2$ and the same $G$ in the other directions.
- **I4 (locality).** $G_{K,d}$ depends only on K and u restricted to K, so it is identical at
  every rank count.

### 2.2 Implementation

New module `src/amr/gradient_indicator.{hpp,cpp}`:

- Input: the velocity `ParGridFunction`. Output: a device-resident `Vector g` of size dim·NE
  (layout `g[d + dim*e]`), plus `eta` of size NE.
- Get the E-vector from the velocity space's lexicographic element restriction. Use
  `QuadratureInterpolator` with `SetOutputLayout(QVectorLayout::byNODES)` and `Derivatives()`,
  which gives reference derivatives (`fem/quadinterpolator.hpp:144`; `PhysDerivatives` is the
  physical one). Take quadrature weights from the rule, and $\det J$ from
  `GetGeometricFactors(ir, GeometricFactors::DETERMINANTS)`.
- Rule: the RuleBook's diffusion rule (Gauss–Legendre, order $2k_u + \dim - 1$). It has at least
  $k_u + 1$ points per direction, which integrates the degree-$2k_u$ integrand exactly on affine
  elements.
- Do the per-element reduction in **one kernel** in a TU added to the nvcc list in
  `src/CMakeLists.txt`, opening with the `#error` guard (CLAUDE.md, "NEVER hand-write an
  mfem::forall"). Then copy the dim·NE result to the host: marking and `NCMesh` refinement are
  host operations in MFEM.
- Profiler scope `amr::indicator`.

---

## 3. Marking

New module `src/amr/refinement_marker.{hpp,cpp}`: host-side, operating on the indicator output.

### 3.1 Which elements

Two threshold modes (`amr.threshold_mode`):

- `relative` (default): mark K if $\eta_K \ge \theta \max_K \eta_K$, with the max a global
  `MPI_Allreduce` and $\theta$ = `amr.theta` (default 0.5). It always marks something unless the
  field is constant, so it suits tests and short studies.
- `absolute`: mark K if $\eta_K \ge \tau$, with $\tau$ = `amr.tolerance` in nondimensional
  velocity units. It stops refining once the flow is resolved, so it suits production runs.

Caps, applied in this order:

1. **Minimum size.** Do not split direction d of K if its extent $|J_{:,d}|$ at the element
   centre is below `amr.min_size`. Drop d from the mask; if the mask becomes empty, unmark K.
   Unlike an element-depth limit, this is direction-aware: `NCMesh::GetElementDepth` counts
   anisotropic splits in any direction.
2. **Maximum elements.** The projected global count
   $NE + \sum_{\text{marked}} (2^{|\text{mask}|} - 1)$ must not exceed `amr.max_elements`. If it
   would, raise the threshold by bisection on $\eta$, one `MPI_Allreduce` of the projected count
   per iteration, until it fits. The effect is "refine the largest indicators first". Forced
   refinements from the NC limit (3.3) can add a few more elements; log the overshoot rather
   than iterating on it.

### 3.2 Which directions

- `amr.anisotropic: false`: mask = every direction (XY in 2D, XYZ in 3D).
- `amr.anisotropic: true`: mask = $\{d : G_{K,d} \ge \rho \max_{d'} G_{K,d'}\}$, with
  $\rho$ = `amr.aniso_ratio` (default 0.5). $\rho = 1$ splits only the dominant direction, and
  $\rho \to 0$ approaches isotropic.

### 3.3 Mesh consistency

- Refine with `GeneralRefinement(refs, /*nonconforming=*/1, nc_limit)`, `nc_limit` =
  `amr.nc_limit` (default 1, 2:1 balance across faces). MFEM adds the forced refinements
  (`NCMesh::LimitNCLevel`).
- **3D, np > 1, any non-XYZ mask.** MFEM supports parallel anisotropic hex refinement only when
  face neighbours are not refined in conflicting directions (`CHANGELOG:163`,
  `mesh/pmesh.hpp:848`). Before refining, call `ParMesh::AnisotropicConflict(refs, conflicts)`
  (collective; returns a globally reduced bool) and set every entry in `conflicts` to XYZ.
  Repeat until it returns false, at most 5 times; then set every remaining non-XYZ 3D mask to
  XYZ and warn on rank 0.
- **Consequence:** in 3D anisotropic mode the refined mesh can depend on the rank count. Serial
  `NCMesh` resolves conflicts by forcing extra refinements internally
  (`mesh/ncmesh.hpp:227`), `AnisotropicConflict` returns false at np = 1
  (`mesh/pncmesh.cpp:1514`), and the parallel path upgrades conflicts to XYZ instead. Tests that
  compare results across rank counts must use 2D, isotropic 3D, or refinement patterns without
  conflicts.

Output: an `Array<Refinement>` with local element indices, plus statistics logged on rank 0
(marked count, per-direction counts, projected NE).

---

## 4. Schedule

- `amr.interval`: accepted steps between adaptation events (0 = no events during the run).
- `amr.initial_passes` (default 0): before the first step, up to N passes of: indicator on the
  initial condition, mark, refine, **re-project the analytic initial condition** on the refined
  mesh. Re-projecting gives the refined mesh's own interpolant. Stop early when a pass marks
  nothing.
- `amr.passes_per_event` (default 1): passes per event during the run. These transfer the state;
  there is no analytic field to re-project.
- **No events during the startup ramp.** `SetHistory` skips the ramp (trapezoidal starter, then
  BDF2; BDF3 history in adaptive mode), so an event while `completed_steps` is below the history
  depth would silently change the scheme. Postpone such events.
- **Adaptive time stepping:** events happen only between accepted steps, never inside the retry
  loop. The next attempted dt is unchanged; Section 9, trap 5 covers what that means for the
  convective CFL limit.

---

## 5. The adaptation event

### 5.1 Sequence (`Case::Adapt`, orchestrated by `src/amr/mesh_adapter.{hpp,cpp}`)

1. Indicator, then marks (Sections 2 and 3). If the global marked count is 0, return without
   rebuilding anything.
2. Export the integrator state into an `IntegratorState` (5.2).
3. Wrap each state vector in a `ParGridFunction` on the current spaces: each velocity history
   level, and the pressure.
4. `mesh.GeneralRefinement(refs, 1, nc_limit)`, then `vfes.Update(); pfes.Update();` and
   `Update()` on every wrapped GridFunction. If `amr.rebalance` and np > 1: `mesh.Rebalance()`,
   update both spaces, and update every GridFunction again. Finally `UpdatesFinished()` on both
   spaces. Every GridFunction must be updated after **each** space update and before
   `UpdatesFinished()`.
5. `MixedSpaces::Update()` (block offsets) and `BoundaryConditions::Update()` (essential true
   dofs; re-project the Dirichlet data at the current time).
6. Optionally project the history onto the discretely divergence-free subspace (5.4).
7. Destroy the integrator, and with it every operator, solver, preconditioner, LOR
   discretization, AMG hierarchy, CC object, convection operator and kinetic-head interpolator.
   Build a new one exactly as `Case::EnsureSetup` does, then `ImportState()` (5.2).
8. Rebuild `OutputWriter` with `ParaViewDataCollection::UseRestartMode(true)`
   (`fem/datacollection.hpp:580`) so the `.pvd` keeps earlier cycles. Optionally write $\eta$
   and per-direction refinement level as cell data (`amr.write_indicator`). `DiagnosticsLog` is
   unchanged.
9. Log on rank 0: event time, NE before and after, marked counts by direction, and wall time per
   phase (profiler scopes `amr::event`, `amr::mark`, `amr::refine`, `amr::transfer`,
   `amr::rebuild`).

**Why rebuild rather than update in place:** every operator, PA data set, LOR mesh, AMG
hierarchy and preconditioner depends on the mesh. Rebuilding them is exactly what construction
already does, events are rare, and the rebuild-from-state path is the restart path that already
reproduces uninterrupted runs. Updating each object in place would add an update path to every
module.

The `ParMesh` is refined **in place**: the object keeps its identity, so everything that holds a
reference to it (the spaces, the ParaView collection) stays valid. `Case` borrows the mesh today,
and that stays true.

### 5.2 `IntegratorState`

New struct in `src/time/integrator_state.hpp`:

- velocity history, true dofs, newest first, with times;
- completed steps and next dt;
- pressure true dofs, as `Pressure()` returns it (static pressure in the rotational form). It
  only seeds the Krylov warm start, exactly as in `Checkpoint`;
- the adaptive controller's PI memory (`PrevScaledError`) **and** its step-attempt record
  (`AdaptiveController::History()`). CLAUDE.md makes the step history part of the interface, so
  it must survive an event; this needs a small import method on `AdaptiveController`.

Add `StokesTimeIntegrator::ExportState()` and `ImportState()`. `ImportState` is `SetHistory`
plus the controller restore plus the record restore. **Refactor `Checkpoint::Write/Read` onto
`IntegratorState`**, giving one definition of the state with two transports, files and memory.
`Case::cycle_` carries over unchanged.

### 5.3 What is transferred and what is recomputed

| State | Treatment |
|---|---|
| Velocity history (2 levels; 3 in adaptive or BDF3 mode) | Transferred (exact) |
| Pressure (warm start) | Transferred (exact) |
| IMEX convection terms | Nothing to do: `SubtractConvection` re-evaluates $N(u^{n-j})$ from the history every step |
| Rotational $w^*$, static pressure, $\tfrac12\|u\|^2$ | Recomputed on the first step after the event |
| Dirichlet elimination data | `BoundaryConditions::Update()` |
| Grad-div $\gamma = c_{gd} h_K$ | Reassembled with the new element sizes |
| Adaptive controller | PI memory and step record restored |
| Time, dt, step and output counters | Restored |

### 5.4 Discrete divergence after transfer (decision D5)

The transferred $u^n$ is exactly the old function. It satisfies $(q, \nabla\cdot u^n) = 0$ for
the coarse pressures but not for the pressures the refinement adds. BDF puts the history on the
right-hand side with weight $1/\Delta t$. The solve still enforces incompressibility of
$u^{n+1}$, and the pressure absorbs the mismatch: expect a pressure transient of size about
$\|B_{\text{new}} u^n\|/\Delta t$ on the first steps after an event. The velocity is unaffected
to leading order. The polynomial manufactured solutions do not show this, because there $u^n$ is
divergence-free pointwise.

Option `amr.project_history` (proposed default `true`): replace each transferred history level
by its $M$-orthogonal projection onto the discretely divergence-free subspace of the new mesh,

$$\begin{pmatrix} M & B^T \\ B & 0 \end{pmatrix}\begin{pmatrix} v \\ \lambda \end{pmatrix} =
\begin{pmatrix} M u \\ 0 \end{pmatrix},$$

with that level's own Dirichlet data. The transferred u already satisfies it, so the correction
is homogeneous. Use the existing `StokesSolver` with `mass_coeff = 1`, `nu = 0` and the
**Cahouet–Chabard** Schur path regardless of the run's `schur` setting: the mass-path Schur block
$\nu^{-1} M_p$ is undefined at $\nu = 0$, while CC with $\nu = 0$ is the exact Schur model for a
pure mass block. Cost: 2–3 saddle solves per event.

### 5.5 Mesh construction when AMR is on

- The serial box mesh must be nonconforming **before** partitioning: `Mesh::EnsureNCMesh()`
  (`mesh/mesh.cpp:11670`). A conforming `ParMesh` cannot be refined nonconformingly
  (`mesh/pmesh.cpp:3945` aborts).
- **Pass METIS's partition explicitly.** A nonconforming `ParMesh` built without a partition
  uses `NCMesh`'s space-filling-curve `InitialPartition` (`mesh/pmesh.cpp:125–134`), not METIS.
  Merely enabling AMR would then change the partition and shift the per-rank-count AMG iteration
  baselines. Use `int *part = serial.GeneratePartitioning(np);` and
  `ParMesh mesh(comm, serial, part);`.
- New single entry point `MakeCaseMesh(const Parameters&) -> std::unique_ptr<mfem::ParMesh>` in
  `src/mesh/case_mesh.{hpp,cpp}`. `apps/run_case.cpp`, `apps/taylor_green.cpp` and
  `python/bindings.cpp` each build the mesh by hand today; all three move to it. It also loads
  the mesh from a checkpoint on restart (5.6).
- AMR off: construction is unchanged (no `EnsureNCMesh`), so nothing numerical changes.

### 5.6 Checkpoint and restart

> **As implemented (deviation):** the checkpoint stores the per-rank refinement history
> (`RefinementRecord`s) and a restart replays it on the initial mesh at the same np, which
> reproduces mesh, partition and dof numbering exactly; the ParPrint design below was not
> used. See CLAUDE.md.

- With AMR enabled, the checkpoint also writes the mesh, one file per rank, with
  `ParMesh::ParPrint` (`mesh/pmesh.cpp:6579`: nonconforming meshes use the NC format, which works
  in parallel). Name it `mesh.r<rank>.mesh`, and record `mesh: refined` in `meta.yaml`.
- Restart: `MakeCaseMesh` reads each rank's file with `ParMesh(comm, istream)`
  (`mesh/pmesh.hpp:362`) instead of building the box. The same-np contract is unchanged, and a
  rebalanced partition is preserved because each rank reads its own file.
- Write checkpoints **after** an event, never between refining and rebuilding.
- **Unverified:** the round trip of a *periodic* nonconforming mesh (periodic meshes carry an L2
  nodes GridFunction). Test C1 must cover it before relying on it.

---

## 6. Impact on existing modules

| Module | Change | Why |
|---|---|---|
| Drivers (`run_case`, `taylor_green`, Python) | Use `MakeCaseMesh` | `EnsureNCMesh` and the METIS partition must happen before partitioning; restart reads the saved mesh |
| `MixedSpaces` | `Update()` recomputes `block_true_offsets_` | True-dof counts change |
| `BoundaryConditions` | Public `Update()`: recompute `ess_tdofs_`, re-project at the current time | Essential dofs change |
| `StokesTimeIntegrator` | `ExportState()` / `ImportState()`; otherwise rebuilt wholesale | 5.1, 5.2 |
| `AdaptiveController` | Import of the step-attempt record | The record must survive events |
| `Checkpoint` | Via `IntegratorState`; writes and reads the mesh | 5.6 |
| `Case` | Owns the schedule and `MeshAdapter`; `Adapt()`; IC passes | 4, 5.1 |
| `OutputWriter` | Rebuilt per event in restart mode; optional indicator cell data | Fields are rebuilt |
| `KineticHeadInterpolator` | **Bug on any NC mesh**: after `Interpolate`, apply `GetTrueDofs` then `SetFromTrueDofs` | `MultLeftInverse` writes the nodal value of ½\|u\|² at hanging pressure nodes rather than the constrained interpolant, so the static pressure is not in the conforming space |
| `MeshSizeCoefficient` (grad-div $\gamma$) | None (decision D7) | `GetElementSize` type 0 is $\det(J)^{1/\dim}$, volume-equivalent; on a 4:1 cell it lies between the short and long edge |
| Jacobi, Chebyshev, point-block Jacobi | None; document | On NC meshes MFEM assembles the diagonal as $\|P^T\| d_L$ (`fem/pbilinearform.cpp:318`), an approximation of diag($P^TAP$), which is fine for smoothing. PBJ's skew vector goes through the same path; it is signed, and high-order hanging weights can be negative, so PBJ's blocks near hanging nodes are approximate. Still a valid preconditioner (test S3) |
| Collocated GLL mass | None; document | $P^TMP$ is not diagonal on NC meshes. `MassInverse` `Auto` detects that and uses Chebyshev; an explicit `DiagDirect` throws. Jacobi's near-exactness at small dt degrades near hanging nodes |
| LOR-AMG (velocity PC, CC's $L_p$ surrogate) | None | MFEM supports H1 LOR on AMR meshes (`miniapps/solvers/plor_solvers.cpp:127` refuses only ND/RT); the probe converged (8.2). `rotation_lor` relies on an identity LOR dof permutation, which `StokesOperator` already verifies at construction; if that fails on an NC mesh, `rotation_lor` with AMR is unsupported and must fail with a clear message |
| Diagnostics, `PressureMean`, null-space detection | None | Quadrature- or topology-based |
| PA kernels and their specializations | None | Mesh-independent |
| GPU | Add the indicator kernel to CLAUDE.md's PENDING GPU VALIDATION list | On NC meshes P is a `HypreParMatrix` (a hypre SpMV on the device per apply) rather than MFEM's conforming prolongation; measure. `NCMesh` refinement, marking and update operators are host-side, once per event |
| Python | Expose `Parameters.amr` | `PyCase` already owns its mesh |

---

## 7. Parameters and deck

New `AmrParameters` in `Parameters`, deck section `amr:`:

```yaml
amr:
  enabled: false
  interval: 50              # accepted steps between events; 0 = none during the run
  initial_passes: 0         # passes on the initial condition before step 1
  passes_per_event: 1
  anisotropic: false
  aniso_ratio: 0.5          # rho in Section 3.2
  threshold_mode: relative  # relative | absolute
  theta: 0.5                # relative mode
  tolerance: 0.05           # absolute mode, nondimensional velocity units
  min_size: 0.0             # nondimensional length; 0 = no limit
  max_elements: 0           # global cap; 0 = none (warn when enabled with no cap)
  nc_limit: 1
  rebalance: true
  project_history: true     # Section 5.4
  write_indicator: false    # eta and refinement level as ParaView cell data
```

Validate ranges with `MFEM_VERIFY`. In dimensional mode, normalize `min_size` by `L_ref` in
`Parameters::Normalize()`. Expose the struct to Python the same way as the other parameter
groups.

---

## 8. MFEM facts this depends on (commit `17d1afc3`)

### 8.1 Read from the source

| Fact | Where |
|---|---|
| `Refinement` masks X/Y/Z/XY/XZ/YZ/XYZ, per element | `mesh/ncmesh.hpp:40` |
| Isotropic and anisotropic refinement of quads/hexes supported | `mesh/ncmesh.hpp:176` |
| Anisotropic splits may force additional refinements (serial) | `mesh/ncmesh.hpp:227` |
| Conforming `ParMesh` cannot become nonconforming; call `EnsureNCMesh` on the serial mesh first | `mesh/pmesh.cpp:3945`, `mesh/mesh.cpp:11670` |
| NC `ParMesh` without a partition uses `InitialPartition` (SFC), not METIS | `mesh/pmesh.cpp:125–134` |
| Parallel anisotropic hex refinement requires non-conflicting face neighbours; `AnisotropicConflict` checks it | `CHANGELOG:163`, `mesh/pmesh.hpp:848`, `mesh/pncmesh.cpp:1514` |
| `AnisotropicConflict` returns false for dim < 3 or np = 1 | `mesh/pncmesh.cpp:1517` |
| Derefinement of 3D anisotropic meshes not implemented (irrelevant here: no derefinement) | `mesh/pncmesh.cpp:2145` |
| NC partial-assembly diagonal is $\|P^T\| d_L$ | `fem/pbilinearform.cpp:318–325` |
| `ParPrint` of an NC mesh uses the NC format, which works in parallel | `mesh/pmesh.cpp:6579` |
| `GetSerialMesh` / print-as-one unsupported for NC meshes (bessemer does not use them) | `mesh/pmesh.cpp:5546` |
| `QuadratureInterpolator::Derivatives` gives reference derivatives | `fem/quadinterpolator.hpp:144` |
| ParaView collection restart mode keeps earlier `.pvd` entries | `fem/datacollection.hpp:580` |
| H1 LOR on AMR meshes supported; ND/RT LOR not | `miniapps/solvers/plor_solvers.cpp:127` |
| The only built-in anisotropic estimator (ZZ) is serial-only, and `ThresholdRefiner` does not resolve 3D parallel conflicts, so bessemer needs its own indicator and marker | `fem/estimators.hpp:64,88`, `CHANGELOG:2653`, `mesh/mesh_operators.cpp:135` |

### 8.2 Checked by running code on this machine

A scratch probe used bessemer's `MakeBoxMesh`, then `EnsureNCMesh`, a `ParMesh`, and two
refinement passes: a band at the periodic seam in x, then a band in y. It ran with X-only then
Y-only refinements (anisotropic) or XYZ (isotropic), at p = 2, np = 1, 2, 4, in 2D and 3D. The
meshes were fully periodic, or stretched in y (two-sided tanh, walls) and periodic in x and z.
All 36 checks passed at each rank count:

- Transfer is exact: $\|u_h\|_{L^2}$, integrated with a rule exact for $|u_h|^2$, is unchanged
  to 1e-12 after both passes.
- Hanging nodes are consistent after transfer: $P R u = u$ to 1e-12.
- Re-projecting a smooth field on the refined mesh is no worse than on the coarse mesh.
- The pressure space still has an `ElementRestriction`, which `KineticHeadInterpolator`
  requires.
- The PA vector mass plus diffusion operator equals legacy assembly on the NC space, to 1e-12.
- LOR-AMG-preconditioned CG converges in 18–39 iterations.
- No 3D anisotropic conflicts arose for these band patterns.

### 8.3 Not yet verified: tests must cover these before anything relies on them

- `ParMesh::Rebalance` on 3D anisotropic NC meshes.
- `LimitNCLevel` forced refinements in 3D parallel anisotropic meshes: they could themselves
  create direction conflicts.
- The `ParPrint` / `ParMesh(istream)` round trip of a periodic NC mesh.
- The full Stokes and NSE solve on NC meshes (S1, S2).
- Point-block Jacobi quality near hanging nodes (S3).

---

## 9. Traps

1. **Collective calls inside rank-0 blocks deadlock.** `mesh.GetGlobalNE()` is an
   `MPI_Allreduce`; the probe hung at np = 2 because it was called inside
   `if (Mpi::Root())`. Compute reduced values on every rank, then print on rank 0.
2. **`EnsureNCMesh` must precede `ParMesh` construction** (5.5).
3. **Update every GridFunction after every space update, before `UpdatesFinished()`.** A
   GridFunction updated late interpolates with the wrong operator, or aborts.
4. **No events inside the startup ramp** (Section 4).
5. **Convective CFL.** With the IMEX convective form, refining shrinks the stable dt by the
   factor by which the smallest directional element size shrinks. The adaptive controller's CFL
   ceiling hook exists but is **not wired** (CLAUDE.md). A fixed-step run can become unstable
   after an event; an adaptive run sees only accuracy, not stability. Decision D6.
6. **3D anisotropic meshes can differ between np = 1 and np > 1** (3.3).
7. **Static pressure on NC meshes needs the conforming projection** (`KineticHeadInterpolator`,
   Section 6).
8. **The `.pvd` is overwritten** if `OutputWriter` is rebuilt without restart mode.
9. **Measure transfer exactness with exactly integrated quantities.** The probe's first version
   compared L2 errors against a non-polynomial field; refinement changes the quadrature, so those
   numbers move even for an exact transfer. Use $\|u_h\|$ with a rule exact for $|u_h|^2$, or
   polynomial fields.
10. **`reap.sh` only finds binaries under `build/`.** Scratch MPI programs elsewhere survive a
    timed-out `mpirun` and keep spinning; kill them by name.

---

## 10. Tests

All in the fast tier, registered at np ∈ {1, 2, 4}, with small meshes (≤ 8³ elements) and exact
assertions wherever an exact answer exists.

**`amr_indicator_test`**

- **N1:** I1 and I2 on affine meshes: a linear u on a sheared box ($G_{K,d} = |AJ_{:,d}|$ to
  1e-13), and u varying in x only (y and z indicators exactly 0). Run on both a conforming mesh
  and a refined NC mesh.
- **N2:** I3 halving across one refinement.
- **N3:** I4: the global sum of $\eta_K^2$ agrees across rank counts to 1e-13 relative.

**`amr_marker_test`**

- **K1:** On a synthetic $\eta$ field, the relative and absolute modes mark exactly the expected
  elements.
- **K2:** Masks follow $\rho$ (anisotropic mode); isotropic mode gives all directions.
- **K3:** `min_size` drops directions and unmarks elements; the projected count respects
  `max_elements` after bisection.
- **K4:** (3D, np ≥ 2) a forced conflict — face neighbours across a z-face marked X and Y — is
  resolved, the loop terminates, and `AnisotropicConflict` is false afterwards. Then refine with
  `nc_limit = 1` and assert the mesh is valid (8.3, second item).

**`amr_transfer_test`**

- **X1:** The probe as a unit test: exact transfer and $PRu = u$ for velocity and pressure, 2D
  and 3D, periodic and stretched, isotropic and anisotropic.
- **X2:** X1 with `Rebalance` at np 2 and 4 (8.3, first item).

**`nc_stokes_test`** (sub-sprint AMR.0: no adaptation, statically pre-refined meshes)

- **S1:** The steady Stokes polynomial manufactured solution on NC meshes with hanging nodes in
  the interior and on Dirichlet walls, 2D and 3D, isotropic and anisotropic. Reproduce it to
  solver tolerance on the Mass and CC Schur paths, with Jacobi and LOR-AMG velocity
  preconditioners. This is the single most important test: it exercises hanging nodes across the
  whole solver stack.
- **S2:** The rotational form's static pressure on NC meshes is conforming ($PRp = p$), and the
  NSE manufactured solution's static pressure still converges in h.
- **S3:** PBJ and LOR-AMG iteration counts on an NC mesh stay within 1.5× of the conforming
  mesh's counts for the same problem.
- **S4:** AMR enabled with no refinement versus AMR disabled: identical step counts, velocity
  within solver tolerance, Jacobi iteration counts within ±1 (METIS partition passed
  explicitly).

**`amr_event_test`**

- **E1:** The unsteady polynomial manufactured solution (exact in space and time, time-dependent
  Dirichlet), 2D and 3D, with an anisotropic event mid-run. Exactness holds at every step,
  before and after the event, in fixed-step and adaptive modes.
- **E2:** A no-op event (threshold marks nothing, but the export / rebuild / import path forced)
  reproduces the uninterrupted run to `ReproTol(1e-13)` with an identical adaptive step
  sequence. This is the same bar as `checkpoint_test`.
- **E3:** The adaptive controller's step record and PI memory survive an event.
- **E4:** `project_history`: after an event on the 2D TGV, $\|B u^n\|$ on the new mesh is at
  solver tolerance with projection and nonzero without. Record the first-step pressure error
  both ways.
- **E5:** Initial passes re-project the analytic initial condition: after N passes the IC error
  equals a direct projection on the final mesh.

**`amr_checkpoint_test`**

- **C1:** Write a checkpoint after events on a periodic, anisotropic, rebalanced mesh; restart at
  the same np; the run matches the uninterrupted one to `ReproTol(1e-13)`. Covers 8.3, third
  item.

**`amr_aniso_test`**

- **A1:** A boundary-layer field $u = (\tanh((y - y_0)/\delta), 0, 0)$ on a channel box. After N
  anisotropic passes there are no X or Z refinements. The anisotropic run reaches the same
  maximum $\eta$ with fewer than half the isotropic run's elements (2D) and fewer than a quarter
  (3D).

**Extensions to existing tests**

- `deck_test`: parse and validate the `amr:` section.
- `tgv_nse_test`: one convective and one rotational run with events stays inside the existing
  energy harness bounds.
- Debug device: every new test passes under `scripts/debug_device.sh`.

**Slow tier / bench:** time per event (indicator, marking, refinement, transfer, rebuild), and
Krylov iteration counts on NC meshes versus conforming ones.

---

## 11. Sub-sprints (each: green at np 1/2/4, full fast tier green, commit with evidence, stop)

| Sub-sprint | Delivers | Green when |
|---|---|---|
| AMR.0 NC readiness | `MakeCaseMesh` (`EnsureNCMesh` and METIS partition when AMR is on); `KineticHeadInterpolator` fix | S1–S4. Hidden hanging-node bugs surface here, before any adaptation code exists |
| AMR.1 Indicator | `gradient_indicator` (device kernel, nvcc list) | N1–N3, on the debug device too |
| AMR.2 Marker | `refinement_marker` (thresholds, directions, caps, 3D conflicts, NC limit) | K1–K4 |
| AMR.3 Transfer | `mesh_adapter` refine/rebalance/update registry; `MixedSpaces::Update`; `BoundaryConditions::Update` | X1, X2 |
| AMR.4 Events | `IntegratorState` (with `Checkpoint` refactored onto it); `Case::Adapt`; schedule; IC passes; output restart mode; `project_history` | E1–E5; `checkpoint_test` unchanged |
| AMR.5 Checkpoint | Mesh written and read; restart through `MakeCaseMesh` | C1 |
| AMR.6 Surface | Deck, Python, docs, `write_indicator`, profiler scopes, CLAUDE.md and the GPU list | A1, the `deck_test` and `tgv_nse_test` additions |

None of these changes numerical results with AMR off, so no existing baseline needs
re-blessing. AMR-on runs get their own baselines, per np for AMG.

---

## 12. Decisions for the human

| | Question | Proposal |
|---|---|---|
| D1 | Indicator field | Full velocity gradient, directional (Section 2). Vorticity magnitude is the common alternative but has no natural per-direction split |
| D2 | Marking | Both modes; `relative` as the default for studies, `absolute` for production. Dörfler (bulk) marking needs a global sort: later, if wanted |
| D3 | Anisotropy ratio $\rho$ | 0.5 |
| D4 | `nc_limit` | 1 (2:1 balance) |
| D5 | Project history onto divergence-free after transfer | On by default; E4 measures what it buys |
| D6 | Convective CFL after refinement (trap 5) | Wire the CFL ceiling in AMR.4, with the directional form $\Delta t \le C \min_K \min_q \big(k_u^2 \sum_d \|(J^{-1}u)_d\|\big)^{-1}$; in fixed-step mode, abort at an event that would exceed it rather than change dt silently |
| D7 | $h_K$ for grad-div $\gamma$ on anisotropic cells | Keep volume-equivalent $\det(J)^{1/\dim}$ |
| D8 | Rebalance after events | On for np > 1 |

**Out of scope:** derefinement; p-refinement; goal-oriented or residual estimators; Dörfler
marking; cost-weighted load balancing; restart at a different np.
