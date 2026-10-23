# Project status

The living handoff log: where the project stands, what was decided (dated, with who
decided), and what is open. The maintainer and their agents update it as work lands.
Rules and design live in [developing.md](developing.md); this file is the *state*.

## Where development left off (2026-10-23)

Keep this section short and current: whoever stops working updates it (AGENTS.md rule 8).
The dated log below keeps the history.

**FSI stage F5: the St. Venant-Kirchhoff solid in the coupling, FSI1-FSI3 (2026-10-23,
branch `claude/project-thread-xnsz2c`; human: "Ok ya pull down the latest and do F5").**
`fsi.solid: linear_elasticity | hyperelasticity` (`FsiSolid`, fsi/fsi_solid.hpp: an adapter
over either class); the SVK solid takes the nominal traction (the fluid's moments through the
reference interface's mass, `FsiCase::NominalTractionFromLoad`), Dirichlet-Neumann only, and
dynamic only: a static SVK solid is refused (human 2026-10-23, "that sounds right"; a massless
solid makes the subiterations try 0.1-0.5 m on FSI1, where SVK folds; the linear one survives
by staying linear; docs/fsi.md §1). Tests `fsi_test` F8 (SVK = linear at small strain),
`run_case_rejects_static_svk_fsi`. Decks `benchmarks/turek_hron/fsi1_svk.yaml`, `fsi2.yaml`,
`fsi3.yaml` (BDF2 solid, `fsi.rtol` 1e-6, `solver.rtol` 1e-9: at 1e-4 and undamped
generalized-alpha FSI3's forces carried a growing step-to-step ripple), `fsi.py`. Coarse
mesh, np 4: FSI1 SVK u_x +2.0 % (linear +6.5 %), u_y -1.4 %, drag +0.30 %, lift -0.06 %;
FSI3 displacement amplitudes -3.7/-4.0 %, frequency +2.0 %, drag mean +0.4 %, drag and lift
amplitudes +17 / -8.5 % (not separated; mesh and dt not varied), 5.7 h. FSI2 is not
reached: at t = 8.6 (tip u_y 58 of 81 mm) the linear-elastic mesh motion folds the cells
by the flag (stiffening 4: t = 8.86), so it needs the hyperelastic mesh motion
(benchmarks/turek_hron/README.md, FSI2).
Found in Hyperelasticity (not changed): Newton cannot stop at tiny loads, its residual's
roundoff floor (~4.5e-11 on the flag) is above both relative tests; the decks set
`solid.newton_atol` 1e-9. Cloud scratch build, CPU; fast tier as on main there (the 10 known
scratch-build failures). Not on a GPU. **Next:** a mesh motion for large rotations (FSI2), the meshes for FSI3's force
amplitudes, an absolute Newton floor in Hyperelasticity, SVK under the other conditions.

**Hyperelasticity milestone 4: neo-Hookean mesh motion (2026-10-23; human: "Oh we want
M4", then neo-Hookean of the options offered).** A compressible neo-Hookean material
(`NeoHookean`, `solid.model: neo_hookean`) behind the hyperelastic kernels, and
`HyperelasticMotion` with it as `mesh_motion.model: neo_hookean`, for the mesh-motion
case and the moving-mesh flows (`DeckMeshMotionFactory`). On the DFG cylinder turned about
its centre with the walls held (V3, np 4): neo-Hookean stays valid to 180 degrees (min
det F 0.65 at 90, 0.25 at 180; 2-5 Newton iterations per 10-degree move; back at rest to
2e-12), where the linear model reaches 90 (0.21) and inverts at 100, and St.
Venant-Kirchhoff fails after 40 ([hyperelasticity.md](hyperelasticity.md) §3). The linear
model stays the default (the standard for moderate motions; neo-Hookean costs a Newton
solve per move). FSI decks reach it through `Case` too, and **FSI2 is reached with it**
(`benchmarks/turek_hron/fsi2.yaml` plus `mesh_motion.model: neo_hookean`, np 8, jobs
49696518 and 49697143). The deck as it is stops at t = 8.60 (tip u_y 57.5 mm, as the other
thread found); with neo-Hookean it runs to t = 15, det F in [0.40, 1.54], periodic from
t = 11 (u_y amplitude 75.92 mm over the last eight periods). Last period against Turek &
Hron: u_x -13.15 +- 11.39 (-14.58 +- 12.44), u_y 1.25 +- 75.92 (1.23 +- 80.6) mm, drag
205.1 +- 67.8 (208.83 +- 73.75), lift 0.95 +- 220.8 (0.88 +- 234.2), u_y frequency 1.924
(2.0): amplitudes 5.7-8.4 % low, means within 2 %, on the coarse mesh (not separated, as
FSI3's). 3.18 subiterations per step, 2.17 h at np 8, about 1.7x the linear run's time per
step. Its first move needed the near-rest Newton test (`294a29e`). CPU np 1/2/4, the debug device
and the verify tier (jobs 49695435, 49695655, 49696320); the GPU suite was still queued at
landing (human: "Push into main ... still run the GPU stuff too"). The parked SVK branch
`hyper-m4` is superseded.

**The GPU suite is green again: diagonals of matrices with empty rows (2026-10-23,
`1b5e44e`).** MFEM's `HypreParMatrix::GetDiag` and hypre's Jacobi read each row's first
entry as its diagonal; a boundary or interface mass has empty rows off its faces (a rank
without the faces, no entry at all). On a host that was garbage in two Jacobi diagonals (the
elastic support's, the Robin term's); on a GPU an illegal access (`linear_elasticity_test`
E6, `robin_test` R1 at np 2) or a NaN now and then (`fsi_test` F3, the FSI flag deck: hypre's
Jacobi in `FsiCase`'s interface mass solve; compute-sanitizer initcheck). `HypreDiagonal`
(precond/hypre_device) searches each row; the interface mass solve preconditions with Jacobi
on it. Main's GPU suite: 92/92 at np 1 and 2 (job 49679538), the first clean one since the
FSI work; CPU fast tier np 1/2/4. `fsi_test` F2's cross-run check allows the GPU's 1e-7.

**The structured hanging-node prolongation (2026-10-23, branch `hanging-p`; human: "ok you
can do this after").** On nonconforming meshes the true-dof applies take
`HangingProlongation` (copies for the unconstrained dofs, weighted sums for the hanging
ones, a fixed-order transpose, ghost masters through `TrueDofExchange`) in place of MFEM's
assembled P and its stored transpose. One H100, curved adaptive cylinder (~25M DOFs): OIFS
Schur 848.8 -> 771.3 ms/step (-9.1 %), IMEX Schur 679.1 -> 584.4 (-13.9 %)
([throughput.md](throughput.md)). Results change at roundoff on hanging-node meshes only.
CPU fast tier np 1/2/4, debug device (AMR, OIFS), GPU suite np 1/2: no failures beyond main's.

**Linear elasticity on our kernels: 3D Q6 on a GPU (2026-10-22, branch `elastic-own`;
human: "ya go ahead with the elasticity fix").** `LinearElasticity`'s stiffness is
`HyperelasticOperator` with a linear material at orders 1 to 7 (MFEM's partial assembly
above): MFEM's elasticity kernel could not launch in 3D from Q6 on a GPU (1,029 threads
per block), which stopped the solid, ALE mesh motion and the FSI solid there. A
sum-factorized diagonal for the Jacobi smoother. Results change at roundoff (decks to the
printed digits; Jacobi CG counts within 1 %); `Form()` gave way to `ApplyStiffness()` and
`StiffnessDiagonal()`. CPU, np 2, 3D cantilevers: Jacobi Q6 141.9 -> 8.1 s, LOR-AMG Q6
20.7 -> 13.0 s ([elasticity.md](elasticity.md) §3). CPU np 1/2/4 and the debug device
pass. Two H100s (job 49656532): every 3D Q6 cantilever runs (they stopped in MFEM's kernel
before) and matches the CPU to 1e-11; `verify_elasticity_test` passes whole at np 1 and 2
(V2's 3D level included). The suite's GPU failures there are main's: `fsi_test`, the FSI
flag deck, `linear_elasticity_test` E6 at np 2 (cudaFree; fixed since, `1b5e44e`).

**FSI stage F4, part 2: Neumann-Dirichlet and Robin-Robin (2026-10-22, branch
`claude/project-thread-xnsz2c` after PR #15 merged; human: "when this is all done and
passes please merge").** `fsi.coupling: neumann_dirichlet | robin_robin` ([fsi.md](fsi.md)
§3-4); all four conditions now exist. Neumann-Dirichlet: the solid's interface prescribed
(its reaction from `LinearElasticity::TrackReaction`, unweighted by the time scheme), the
fluid's a traction boundary (a Robin boundary with alpha 0), iterated on d with
d~ = d + (u - w) / c0 (`Case::MeshVelocityLead`). Robin-Robin: Robin-Neumann's fluid side
and an elastic support alpha_s c0 on the solid (`LinearElasticity::AddSupport`), iterated
on the solid's Robin data; `fsi.alpha_solid` has no default. Both reach Dirichlet-Neumann's
coupled steps (`fsi_test` F6, F7; the solid pieces `linear_elasticity_test` E6). FSI1
coarse: Robin-Robin (alpha_s 1,000) 4.3 subiterations per step, the same answer, 2.8x
Dirichlet-Neumann's wall time; Neumann-Dirichlet stops in its first steps (the flag's
squeeze mode; on a flag 1,000x softer it needs 5x fewer subiterations than
Dirichlet-Neumann). Cloud scratch build, CPU, np 1/2/4; not on a GPU. **Next:** FSI2/FSI3
(F5), and the hyperelastic solid in the coupling.

**FSI stage F4, part 1: Robin-Neumann, and Dirichlet-Neumann + IQN-ILS by default
(2026-10-22, branch `claude/project-thread-xnsz2c` after PR #14 merged; human: "Ya merge
and start F4", then the default on a decision card).** A fluid Robin condition
(sigma_f n + alpha u = tau + alpha w; `BoundaryConditions::AddRobin`, a Robin block in
`StokesOperator`, `StokesTimeIntegrator::SetBoundaryTraction`; test `robin_test` R1-R3)
and `fsi.coupling: robin_neumann | dirichlet_neumann` ([fsi.md](fsi.md) §2). Robin-Neumann
iterates on the interface traction (on d, the load was hidden state and IQN-ILS
stalled); alpha_f = rho_s min(h, c_P dt) / dt face by face, h the solid's thickness
behind the face (`fsi.alpha_fluid` / `fsi.structure_thickness` override). It reaches
Dirichlet-Neumann's coupled steps to 1e-6 (`fsi_test` F6) but is slower on the two-sided
flag: FSI1 coarse to t = 15, 6.6 subiterations per step against 4.4, each ~3x costlier,
the same answer; solids4foam reports the same (fsi.md). Aitken fails FSI1's first step
under both, hence the defaults `dirichlet_neumann` + `iqn_ils`. `fsi_test` F5 (static
solid) now solves the flow to 1e-12: the static solid amplifies the flow's error ~1e5 and
the test had passed on main at np 2 only by luck. Cloud scratch build, CPU, np 1/2/4; not
on a GPU. (Part 2 above.)

**FSI stage F3, the coupling framework (2026-10-22, branch
`claude/project-thread-xnsz2c` after PR #13 merged; human: "Ya merge once it passes and
then do the coupling").** Partitioned Dirichlet-Neumann coupling of the ALE flow and the
linear solid, iterated in every step ([fsi.md](fsi.md)): deck section `fsi`, boundary
type `interface` (a wall moving with the solid, u = w), the solid on the geometry's solid
part (`cylinder_channel` with a flag only, so far). New: `Case::TrialStep / UndoTrialStep
/ CommitTrialStep` and `StokesTimeIntegrator::UndoStep`, `LinearElasticity::ExportState /
ImportState`, `Case::InterfaceDisplacement`, `Case::SetSummaryHook`, `Case::FinishRun`;
`src/fsi/`: `InterfaceTransfer` (point-search interpolation between the reference
meshes, exact on the matching Turek-Hron interface), `InterfaceAccelerator` (constant,
dynamic Aitken (the default, human 2026-10-21), IQN-ILS with `reuse`), `FsiCase` (the
traction from the momentum residual through the interface mass matrix, rho_f times the
fluid's force exactly). On `fsi_test`'s impulsive start (rho_s = rho_f) IQN-ILS took 27
subiterations in two steps against Aitken's 54-62; the constant relaxation at 0.5
diverges. The flow solver's `rtol` must sit well below `fsi.rtol` (else the residual
stalls at the solver's noise and IQN-ILS wanders). Tests `fsi_test` F1-F5 and
`run_case_fsi_flag` (`cases/fsi_flag.yaml`), np 1/2/4, cloud scratch build, CPU; not on
a GPU. Not yet: Robin conditions and Neumann-Dirichlet (F4), gslib's map, restarts, the
SVK solid (F1, the maintainer's). FSI1 (`benchmarks/turek_hron/fsi1.yaml`, Q4, linear solid,
np 4, to t = 15): coarse mesh (7 min) drag +0.30 %, lift -0.48 %, tip u_y -0.06 %, u_x
+6.5 %; `level: 1` (32 min) +0.11 %, -0.50 %, +0.67 %, +5.3 %; 4.1-4.4 subiterations per
step. The first version stopped at t = 5, before the flag had settled (u_y -4.0 %, lift
-2.4 %; human 2026-10-22: "explore the error first"); u_x's error is mostly the linear
solid's missing bending shortening (+4.9 %); solid mesh and `fsi.rtol` ruled out
(benchmarks/turek_hron/README.md). **Next:** F4 (Robin-Neumann
default, Robin-Robin, Neumann-Dirichlet), then FSI2/FSI3.

**ALE stage 3: the solvers refreshed in place after a move (2026-10-22, branch
`claude/project-thread-xnsz2c` after PR #11 merged; human: "Ok merge please and go to
stage 3").** `mesh_motion.refresh: in_place | rebuild` (case default in_place; the
low-level `TimeIntegratorOptions::ale_refresh_in_place` is off) and
`mesh_motion.amg_interval` (10). In place, `StokesSolver::UpdateGeometry` re-assembles
what depends on the geometry (masses and their inverses, divergence, gamma, the momentum
block, the pressure mean, the guess state's images; `StokesOperator::UpdateGeometry`,
`CahouetChabardSchurPC::UpdateGeometry`, `MassInverse::Refresh`) and keeps the AMG
hierarchies and Chebyshev estimates for `amg_interval` moves, by the move count. Not with
the rotational form (refused already on moving meshes). Dütsch deck to t = 1, np 4: the
per-move refresh 1.42 -> 0.35 s, the run 11.2 -> 9.6 s, iterations unchanged. Tests
`ale_fluid_test` A10 (in place = rebuild to 2e-16 with the AMG rebuilt every move,
< 1e-12 kept for 3 moves, also with a LOR-AMG velocity block and the Schur method), A9 for
both paths (a restart in place matches to 1e-10, not bitwise: its AMG is built on the
restart's mesh). Cloud scratch build, CPU, np 1/2/4; not on a GPU. **Next:** the
coupling framework (F3).

**Turek-Hron geometry and CFD1-3 (2026-10-22, branch `claude/project-thread-xnsz2c`
after PR #10 merged; human: "make sure that you keep going").** FSI stage F2 of the
ALE scope. `mesh.cylinder_channel` takes a flag (`flag_length`, `flag_thickness`,
`n_flag`, `n_flag_y`): grid lines at the flag's edges and tip, the ring paired by
angle on its right side so the flag's edges are straight through it, the flag's cells
left out, boundary `flag`; `part: solid` gives the flag's own mesh with the same nodes
on the interface (the solid side of FSI). DFG meshes unchanged. Test
`cylinder_mesh_test` M4. benchmarks/turek_hron: CFD1 (Q4, 314 / 1,256 elements) drag
+0.31 / +0.12 %, lift +0.71 / +0.27 %; CFD2 at the defaults (Q6 OIFS) drag +0.07 %,
lift +1.46 %; CFD3 at the defaults (Q6 OIFS, 314 elements,
26 min at np 4) drag 439.52 +- 5.69, lift -11.78 +- 431.7, frequency 4.439 against
439.45 +- 5.62, -11.89 +- 437.8, 4.396; the flag alone under CSM1's load with the
linear solid (CSM proper needs the SVK solid). Cloud scratch build, CPU; not run on a
GPU. CFD3 on the refined mesh (`level: 1`, 1,256 elements): lift amplitude -1.4 -> -0.29 %, the rest unchanged (drag +0.03 % / +1.7 %, frequency +1.1 %). **Next:** the coupling framework (F3).

**ALE fluid, IMEX, on the CPU (2026-10-21, branch `claude/project-thread-xnsz2c`;
human: "can you go scope out what needs to be done to implement ALE now?", then
"Alright can you go get started on this on CPUs").** Stages 0-1 of the scope
(decisions: IMEX first; mesh moved every step by pseudo-elasticity with stiffening;
cached geometry kept for fixed meshes plus a refresh path for moving ones; then the
partitioned FSI to all nine Turek-Hron problems). Landed ([ale_fluid.md](ale_fluid.md)):
`Case::SetMeshMotion(MeshMotion&)` and `StokesTimeIntegrator::SetMeshMotion`, the
non-conservative ALE form on the current mesh: each step moves the mesh, takes w as
the BDF derivative of the nodes (`MeshVelocityField`), rebuilds every solver and the
convection on the moved mesh, and adds the explicit, extrapolated mesh convection
C(w) u* (`src/ale/mesh_convection`); u = w on moving walls
(`Case::MeshVelocityCoefficient`); a cut step moves the mesh back
(`MeshMotion::Revert`, new); the CFL uses u - w; `time.convection: auto` picks IMEX
on a moving mesh; OIFS, the rotational form, error control and AMR are refused.
Test `ale_fluid_test` A1-A7 (fast): the free stream stays uniform (GCL) in 2D and 3D,
a rigid box carries the fluid, Taylor-Green on an oscillating mesh converges at
order 4.3 in h (Q3) and 2.2 in dt. Run in the cloud on a scratch build (MFEM v4.10
from GitHub, apt hypre and Open MPI, not the Spack toolchain) at np 1/2/4; its fast
tier fails 10 tests that fail the same way on main there (the unpatched MFEM 4.10
AnisotropicConflict segfault, apt hypre's "Error during solve"); CI is the gate.
Never run on a GPU. **Stage 2, the deck surface (2026-10-22; human: "Rather than
merge keep going"):** a boundary group's `motion` (expressions in the reference
x, y, z, t with `parameters`) moves the mesh; groups without it stay fixed; a moving
`no_slip` group takes u = w; the interior follows section `mesh_motion`'s elastic
model (`Case::BuildDeckMotion`); checkpoints carry the node history
(`MeshMotion::Save/Load`, per rank), the summary a `mesh_motion` section. Tests A8
(deck rigid box), A9 (restart = uninterrupted to 1e-13), np 1/2/4. Deck
`cases/ale_oscillating_cylinder.yaml` (DFG channel, Re 100, cylinder moved across at
0.2 D: t = 2 in 55 s at np 4, det F in [0.89, 1.11]; not validated). Refused:
dimensional decks, periodic meshes. Then: the outer solver's initial-guess state carried
across the per-step rebuild (outer iterations 1,336 -> 545 on the cylinder deck to
t = 0.3; zero motion now costs what the fixed mesh does), the mesh solve from the
extrapolated node history (3D 8^3 Q3: 5.3 -> 3.1 s over 10 moves), and an abort when
moving walls change a closed domain's volume (it used to iterate to the solver's cap;
test `run_case_rejects_volume_change`). Stage 4: `LinearElasticMotion` reuses earlier
solves when the boundary data lie in their span (`mesh_motion.basis_size`, 12; rigid or
modal motion needs one solve per shape; checkpointed; test `mesh_motion_test` M6): the
cylinder deck to t = 0.3 makes 1 mesh solve in 116 moves, run 3.26 s -> 1.46 s at np 4
(the fixed mesh: 0.82 s in 80 steps). **Slow by design for now (stage 3 above replaced it):** every solver, the
pressure LOR-AMG included, is rebuilt after each move (bessemer's divergence, grad-div
and OIFS kernels keep raw pointers to MFEM's geometric factors, which a move frees).
Stage 5, validation: the in-line oscillating cylinder of Dütsch et al. (1998), Re 100,
KC 5 (benchmarks/oscillating_cylinder/; new `mesh.cylinder_channel` keys for any channel
and cylinder size): c_d 2.091, c_i 1.460 against 2.09 / 1.45, converged to ~0.1 % in mesh,
dt and domain; 8 min at np 4 on the CPU. **Next:** the in-place refresh (stage 3; the
per-move rebuild is 5-9 % of a step in the 2D and 3D profiles), then the FSI stages.

**Hyperelasticity, milestone 1 and a memory step (2026-10-21; human: "start specking on
adding hyperelasticity and at venant Kirkhoff model", "go with your recommendations,
commit the spec and start M1", "after this is a good time to also start optimizing memory
and then do performance engineering").** St. Venant-Kirchhoff material and matrix-free
residual and tangent operators, one fused sum-factorized kernel per apply
(`src/solid/hyperelastic_*`; [hyperelasticity.md](hyperelasticity.md); spec
design/hyperelasticity_spec.md). `hyperelasticity_test` checks them against MFEM's
`HyperelasticNLFIntegrator` (CPU np 1/2/4, H100 np 1/2). The performance step rewrote
the 3D GPU kernels by lines, after Nsight Compute found their register tensors spilling
to DRAM, then fused the gather and scatter into them and collocated the derivatives
(human: "Keep optimizing"; hyperelasticity.md section 4). It then exchanged the shared dofs
directly instead of through MFEM's prolongation (`TrueDofExchange`) and fit 7 blocks per
SM, then overlapped the exchange with the interior elements. 3D Q6 on one H100: 93 B per
vector dof (the default storage, F), residual 20.2 and tangent 17.2 Gdof/s (5.3x and
4.7x the first kernels). On two H100s it reaches 89-93 % of one (it was 76-79 %). MFEM's
linear elasticity PA cannot launch there and is 11-25x slower at Q3-Q5. Geometry on the
fly is shelved (human 2026-10-21: under 2 % of the solver's memory). Milestone 2 began
on 2026-10-22 with the static solver, `Hyperelasticity` (Newton, load steps, line search,
the reference LOR-AMG; tests H4-H6: an exact homogeneous finite deformation, a strain-free
180-degree rotation, and the linear limit at O(load^2)). Its deck,
`equation: hyperelasticity` through `ElasticityCase` (linear decks bitwise unchanged), runs
`cases/turek_hron_csm{1,2}.yaml`. Point A converges to within 1-2e-4 of the benchmark's
values (FeatFlow table, checked 2026-10-22), about the spread of the benchmark's own
levels. Verify tier: `verify_hyperelasticity_test` (a finite manufactured deformation's
rates, CSM1/CSM2). Newton also stops on a small correction, since the residual's roundoff
floor reached 1e-7 of the load on fine flag meshes. Milestone 3 (dynamics, the same
day): generalized-alpha and BDF2 on the nonlinear system as one Newton solve per step,
second order in time on a finite separable motion (H7), the dynamic deck, and
`cases/turek_hron_csm3.yaml`. CSM3 is within 0.5 % (u_y, frequency) and 1.7 % (u_x mean)
of the benchmark at its dt 0.005, about as close as the benchmark's own dt dependence
(2.4 %) allows. **Next:** the hyperelastic mesh motion (milestone 4; done 2026-10-23
with neo-Hookean, above).
The landing's GPU suite (job 49592163, np 1 and 2, H100s) passed everything here but
found three failures in the FSI F3 and ALE stage 3 code merged the same day, which had
not run on a GPU: `fsi_test` F2 compares a device result read on the host (2e-319, the
pattern of the earlier `ale_fluid_test` fix); `fsi_test` and `run_case_fsi_flag`
diverge (`IsFinite(nom)` fails); `ale_fluid_test` A9's restart differs by 1.3e-12
against its 1e-13 tolerance. The CPU suite passes them at np 1, 2 and 4.

**Mesh deformation from a deck (2026-10-21, branch `claude/project-thread-hq7y8y`;
human: "add the capability to deform a given mesh using the structural solver ...
specify like a function on the boundary which will then deform the mesh using jacobian
stiffening"; demos: a rectangle with its bottom deformed by a parabola, and a 3D cube).**
`equation: mesh_motion` makes `run_case` (and `incns.run_deck`) deform the deck's mesh
with the existing `MeshMotion` + `LinearElasticMotion` (docs/mesh_motion.md §3):
boundary groups `displacement` (a constant `value` or, new, an `expression` per
component in x, y, z, t with named `parameters`; `config/expression`, a small parser
of our own) and `fixed`; section `mesh_motion` (model, poisson_ratio, stiffening,
order, steps, CG settings); moves at t_k = k t_final / steps. Output: ParaView of the
deformed high-order mesh with `displacement` and `det_F`, a per-move history CSV, and
summary `results` (det F range, the smallest element Jacobian ratio, max displacement,
the boundary error against the expression); an inverted mesh stops the run as
`diverged`. Displacement expressions work in `linear_elasticity` decks too. Choices
made by the agent: expressions rather than named profiles, moves solved from the
reference mesh (the existing total-displacement design), no free or sliding mesh
boundaries. Decks `cases/mesh_deform_2d.yaml`, `cases/mesh_deform_cube.yaml`; tests
`mesh_motion_case_test` C1-C3, `expression_test` E1-E4, `run_case_mesh_deform` (fast).
The two decks at np 4 (cloud scratch build, MFEM v4.10 from GitHub, apt hypre/Open
MPI, not the Spack toolchain): rectangle 16 x 8 Q4, parabola peak 0.3, min det F 0.68,
0.07 s; cube 8^3 Q3, bump peak 0.2, min det F 0.45, 2.5 s; boundary error <= 1e-16 in
both. Rendered with headless ParaView. Not run on a GPU; the expression evaluation and
the boundary-error check are host loops (setup and once per move). Stiffening and
Poisson-ratio guidance, measured on the DFG cylinder and a graded rectangle (chi 0/1/2):
docs/mesh_motion.md §1, "Choosing the settings" (human: "put in comments about the
stiffening ... when it should be used and recommendation for settings"). **Next:** test it
on a GPU (human 2026-10-21: "next step is to have it tested on GPU"), then the ALE fluid
driven by the same section.

**README rewrite (2026-10-21, PR #8, merged).** The README now describes the fluid,
solid, mesh-motion, LES and Schur capabilities on CPUs and GPUs, the solver options,
elasticity, mesh motion and Python, names no default values (it points at
docs/deck_reference.md), and has a doc index and a Testing section.

**Python bindings brought up to date, and built in CI (2026-10-21, branch
`claude/project-thread-048nzj`; human: "go do the python stuff", the bindings item the
LES plan left open).** The reading taken: Python access to drive a case and read its
results, covering what landed since 2026-10-10, not every C++ class
(developing.md §5.7). The module had never been built in CI, so nothing guarded it.
Landed: `incns.run_deck(deck)` (path, dict or Parameters; `run_case`'s sequence for flows
and `linear_elasticity` solids; returns the summary dict); `Parameters.from_dict` /
`from_yaml_string` (every deck key through the deck loader; unknown keys raise
`ValueError` instead of aborting) and `incns.deck_keys()`; attribute bindings for the
LES (`subgrid`), the Schur method (`outer`, `uzawa`), `initial_guess`,
`tolerance_reference`, `on_solver_failure` + retries, `output.progress/history/profile`,
`initial`, `forcing`, `channel_statistics`, `cc.a_pcg_pc` / `schur_grad_div` and more;
`Case.subgrid_statistics()`, `cfl_number()`, `summary()` (dict), `write_profile()`.
**Behaviour change:** `incns.Case` now applies the deck's named `initial_velocity`, as
`run_case` does (it started from zero; `set_initial_velocity` still replaces it). New
tests `py_run_deck`, `py_run_les` (np 1/2/4). CI (§7.6) now builds `incns_pymodule` in
`build/cpu-python` after the fast tier and runs the `py_*` tests; pybind11 and Python from
apt, numba from pip, so the Spack stack and its cache key are unchanged. Pre-checked in
the cloud session on a scratch MFEM (1c19aba, apt hypre 2.28 and Open MPI 4.1, not the
Spack toolchain): the module builds under `-Werror`, all 18 `py_*` tests pass at np 1/2/4,
astyle 3.4.11 clean. Not exposed: geometry specs other than the box (use a deck), AMR
history, channel profiles (all in the summary or its files). **Next:** the Python side of
the ALE work when it gets deck keys.

**Large-eddy simulation: the sigma and WALE subgrid models (2026-10-17, branch
`claude/project-thread-jjlvl5`, PR #3, rebased onto main 2026-10-21; human: "ok you can
merge this", 2026-10-19). Written in a cloud session that cannot build; CI is its only build: style,
`-Werror` build, Doxygen and the fast tier pass there (CPU, np 1/2/4). Never run on a
GPU, nothing timed.** Human (2026-10-16): "alright why don't you go put in the LES model you
reccomended", "don't forget the symmetric gradient!"; in the project chat: wall-resolved
LES makes sense to have, and for a high-order code that is probably the top of the
fidelity stack. The memo it follows recommended holding off on LES for the DNS roadmap
and, when needed, one explicit subgrid term with the sigma model first and WALE behind
the same key. Landed ([les.md](les.md), developing.md §5.8): `physics.subgrid.model:
none | constant | wale | sigma` (+ `coefficient`, `filter_width`), the explicit stress
term `(2 ν_t S(u), S(v))` with the symmetric gradient on the convection's explicit path
(IMEX: with N; OIFS: in the BDF step with the outflow term, `SubtractOifsExplicitTerms`,
which replaces `SubtractOifsDirectionalDoNothing`), `operators/subgrid_model.hpp`
(pointwise models), `operators/subgrid_stress.{hpp,cpp}` (two device kernels, an nvcc
TU; no per-step state, ~150 B per velocity dof of scratch inside an apply, 8 B per element
kept), `summary.json` `subgrid` section. Found while writing it: Cardano's formula for
the singular values (Nicoud et al.'s appendix) loses σ₃ near walls (0 at y = 1e-3 where
the truth is 2e-9), so `SingularValues3` uses one-sided Jacobi. Tests `subgrid_test`
S1-S5 (models, operator = elasticity(λ = 0, μ = ν_t) to 1e-12, dissipation identity,
explicit constant ν_t → implicit at order 2 under IMEX and OIFS, sigma through `Case`).
**Not done:** a turbulent validation (TGV Re 1600 on a coarse mesh against the DNS
dissipation curve is the first candidate), any GPU run or timing (the kernels interpolate
directly, O(k⁶) per element; sum factorization if a profile asks). (The Python
bindings followed on 2026-10-21, below.)
Defaults unchanged (model off); bitwise neutral with the model off. Its apply allocates
its own scratch rather than borrowing the time integrator's `ElementScratch` (change 41's
pattern): a follow-up if the allocations show in a profile.

**Verification at the defaults (2026-10-19/20; human: "run the full Verification
benchmarks ... using the defaults we agreed on", "write up basically each of them
individually").** The benchmark decks are the defaults now (a5d8ed0; human: "these options
... should just become the defaults", AMR for the cylinder, not the channel or
Taylor-Green): Q6/Q5, OIFS at CFL 2; Taylor-Green `_n21`, `_n32`, `_n42`; the channel at Q6
on half the elements; DFG 3D-1Z and 3D-3Z with AMR. Run as committed, one H100 each (the
CUDA build of 5da1553): Taylor-Green dissipation peak error 1.86 / 0.22 / 0.17 % and
largest error 3.31 / 2.42 / 1.80 % at 126³ / 192³ / 252³ nodes (87 s, 5.5 min, 16 min to
t = 20); the channel U⁺ within 1.47 % at Re_τ 180 and 0.66 % at Re_τ 395 (the deck's first
run), Re_τ within 0.13 % (17 min, 2 h 34 min); DFG 3D-1Z c_D 0.006 %, c_L 0.03 % (0.3 %
without AMR), 3D-3Z c_D,max 0.17 %, c_L,min 0.39 % (0.15 % against FeatFlow's history) to
t = 8 in 27 min (the 2026-10-17 AMR run had stopped at t = 4.5). Write-ups:
benchmarks/*/README.md, "Results". `run_case` takes no `--set` overrides: the 2026-10-17 runs
used deck copies, and the READMEs no longer say otherwise.

**Linear elasticity on a GPU (2026-10-19; human: "check that all the linear elasticity
stuff added can run on GPUs").** main a5d8ed0 (before the ALE merge), two H100s: the whole
GPU suite passes at np 1 and 2 (78/78, the first GPU suite since the elasticity merge), and
the 2D example decks run and match the CPU to 5e-9 in every result (CG iterations 40-70 %
higher on the GPU: hypre's device AMG). **Every 3D run at the default order (Q6) stops at
its first operator apply**: MFEM 4.10's `ElasticityAddMultPA_`
(fem/integ/bilininteg_elasticity_kernels.hpp) launches dim × (p+1)^dim threads per block,
3 × 343 = 1,029 > 1,024 ("invalid configuration argument"); its diagonal and EA kernels
have the same shape, and MFEM's master is unchanged. In 3D only orders up to Q5 fit
(3 × 216 = 648; the Q5 runs were still queued at this writing); `verify_elasticity_test` V2
(3D) stops at its default-order level, after Q2 and Q3 pass. The ALE mesh motion
uses the same operator. Reported, not changed (the elasticity thread's code); the fix is
open (an own sum-factorized elasticity operator, or upstream block limits; MFEM's kernel is
not sum-factorized either).

**ALE mesh motion (2026-10-20, branch `claude/mesh-motion-tc3hlq`, PR #5;
human: "add the capability to deform a mesh using the linear elasticity solver
... make it general that a different type of solver (hyper elasticity etc) can
be used").** `src/ale/`: `MeshMotionModel` (the interface: fixed and displaced
boundaries, `Solve(t)`, a displacement on the reference mesh; built by a
factory), `LinearElasticMotion` (LinearElasticity with Jacobian stiffening
(V_mean / V_K)^chi, Stein-Tezduyar-Benney 2003), `MeshMotion` (copies the mesh
as the reference, moves the nodes to X + d at the nodal order, NodesUpdated,
BDF mesh velocity from the kept node displacements, det F range).
`LinearElasticity` gained `Options::element_stiffness` and `Solve(t)`.
Choices made by the agent and stated to the human, not yet reviewed: the solve
is total displacement on the reference mesh (no drift, one AMG setup; large
rotations are the hyperelastic model's job), any boundary not named is fixed
(no sliding yet), nu 0.3 and chi 1, no deck keys until the ALE fluid. Test
`mesh_motion_test` (fast, M1-M5). Written in a cloud session without the
toolchain: type-checked against MFEM 4.10's headers, never linked or run
there; CI is the gate. Never run on a GPU. Docs: docs/mesh_motion.md.
**Next:** the ALE fluid (u - w in the convection, GCL with BDF, the flow
operators refreshed after each move, a `mesh_motion:` deck section,
a moving-domain MMS), then a hyperelastic model.

**Linear elasticity, static and dynamic (2026-10-19, branch
`claude/linear-elasticity-qqgghm`, PR #4, merged 2026-10-19; human: "scope and then add a linear
elasticity solver", the first step toward FSI with ALE; "I want A and C
implemented" for the time schemes).** `src/solid/` (`LinearElasticity`,
`ElasticityCase`), `equation: linear_elasticity` in `run_case`, deck section `solid`
(`analysis: static | dynamic`, `time_scheme: generalized_alpha | bdf`, `bdf_order`,
`rho_inf`, `density`), boundary types `fixed`, `displacement`, `traction`, `free`;
decks `cases/cantilever_2d.yaml`, `cases/cook_membrane.yaml`. Defaults picked by the
agent: generalized-alpha with rho_inf 0.8, BDF order 2. **BDF3 is not A-stable for
undamped structures** (a mode with omega dt near 1 grows 4 % per step): fine for the
fluid, short or damped solid runs only (docs/elasticity.md §1). Tests
`linear_elasticity_test`, `elastodynamics_test` (fast) and `verify_elasticity_test`
(verify: MMS 2D/3D, Lame cylinder on a curved mesh, Cook's membrane in plane stress
and strain, the default scheme's time order, a cantilever's first frequency against
Euler-Bernoulli). Cook's membrane: the literature's 23.96 is the loaded edge's
midpoint (48, 52), not the top corner (which converges to ~25.2; checked with an
independent Q2 code). Pre-checked in the cloud session with a scratch MFEM build
(1c19aba, pip Open MPI, not the Spack toolchain): the three test programs pass at
np 1/2/4, and that build's fast tier passed 231 of 237 (the failures flow tests this
work does not touch: `persistent_fgmres_test`, the known `-march` mismatch, and hypre
"Error during solve" in three np-1 tests; not compared with main there); CI is the real
gate (PR #4). Never run on a GPU; memory per DOF estimated only
(docs/elasticity.md §3). **Next:** large deformation (St. Venant-Kirchhoff, Turek-Hron
CSM), then the ALE fluid and the coupling.

**Capacity on one H100 (2026-10-18, gpu_memory.md):** 100M+ DOFs with every Schur
configuration (Taylor–Green OIFS 128.6M, IMEX 158.8M; curved cylinder OIFS 101.0M, IMEX
134.5M; with hanging nodes (Taylor–Green, one AMR pass) OIFS 106.6M, IMEX 123.0M) and IMEX coupled (straight 115.1M, curved 101.0M);
OIFS coupled reaches 96.6M straight, 78.6M curved.

**Changes 28–41 (2026-10-17/18).** Changes 38–41: change 36's kernel at Q7 (38), the
outflow term without whole-vector host round trips (39), the PA blocks' AddMult without
a temporary (40), the diagnostics borrowing the element scratch (41: runs no longer die
in the end-of-run summary). A GPU-only bug of 28–29 (the curved self apply's chunk
views) is fixed in place. Changes 28–37: memory: curved elements keep no convection data per
point (28–29), the CFL rate allocates nothing on straight-sided elements (30), OIFS's
Dirichlet data per stage without projections or kept boundary vectors (32), point-block
Jacobi without stored inverse blocks (35). The collocated mass is lumped at hanging nodes
by default (31; human: "this should be the default option and we shouldn't allow a regime
that fails"). Throughput: the Stokes step's Dirichlet values without a projection per
solve (33; 230 ms of the cylinder's step), the Q6 momentum apply with a thread per point
(36; 11–24 % per step), GPU-aware MPI by default in both profiles (37; human 2026-10-17;
19–20 % per step at np 2 on Taylor–Green). Fix: the Schur method's pressure target (34;
curved IMEX Schur stalled). The solver matrix at np 1 and 2 and what it found:
throughput.md ("The solver matrix at Q6"); each memory change, the gaps and the capacities:
gpu_memory.md; the multi-GPU plan: multi_gpu.md (human: 4–8 GPUs for scaling studies; no
hard cap on concurrent H100s, the queue is the limit). Held by the human: geometry on the
fly (+26–28 % per step for ~74 B/DOF). The planned `performance.priority: throughput |
memory` key for trade-offs (human 2026-10-17) was dropped (human 2026-10-18): only
`time.oifs_on_the_fly` is such a trade-off, and it stays its own opt-in, renamed
`memory.max_dofs` (change 51, human 2026-10-18; the old name still read), which a run near
a full GPU suggests. Changes 42–45
(2026-10-18): UCX's cuda_ipc off by default (42; the human chose the memory), hypre's own
SpMV in the pressure AMG (43; the cylinder's np-2 step −18 %), the AMR history projection
without a velocity AMG or a Krylov basis (44) and the 3D indicator without a quadrature-point
buffer (45): the curved adaptive run's peak 65.4 → 39.8 GB (IMEX Schur, 56.5M DOFs), its
AMR events 387 → 36 s. The CFL rate on curved elements allocates nothing (46: a 2.3 GB
spike every step that the gauge could not see; OIFS Schur 65.7 → 48.9 GB on that case).
The p = 6 comparison with one binary after 42–46 (gpu_memory.md): five meshes × four solvers,
B/DOF at ~57M and the capacity on one H100; 100M fits with every Schur configuration but OIFS
Schur on the curved adaptive mesh (84.4M), IMEX Schur there 112.5M. **The memory campaign
ended with change 47 (human 2026-10-18)**: one pair of work vectors for all the P^T A P
operators (IMEX Schur −22 B/DOF on the curved adaptive mesh); there OIFS Schur reaches 93.7M
(~96.8M out of memory), IMEX Schur 118.7M. The summary (start and end, what was dropped, what
is left, what MFEM changes would have helped) is gpu_memory.md's first section.
**Throughput restarted (2026-10-18):** the solver matrix at np 1, 2 and 4 on B47
(throughput.md): Taylor-Green 85-92 % efficient at np 2, 72-73 % at np 4. The cylinder's
outflow term ran on the host on one rank: changes 48 (OIFS's classification flags on the
device) and 49 (the outflow term from cached face data), cylinder OIFS np 2 0.524 ->
0.313 s per step, IMEX -30 %, np 1 -12 to -20 %. Change 50: OIFS's wall-node convection
on the wall elements only (cylinder -4 % at np 1, -8 % at np 2). Hot spots per solver path
(2026-10-19, throughput.md): the GPU is 94-98 % busy; MFEM's consistent vector mass apply was
17-22 % of IMEX steps -> change 52 (a thread per point, IMEX -5 to -19 % per step); hypre's
SpMV is 21 % of curved adaptive steps (the assembled hanging-node P). Kernel headroom
(Nsight Compute, 2026-10-19, throughput.md): a quarter of a step is at the DRAM limit, the
momentum kernel is bound by the L1 pipe and the OIFS kernel by latency; ~1.2-1.5x per step
is left in kernel work, not 2x. Change 54: the gradient without shared-memory bank
conflicts on a device (-14 % per call, -1.2 % per TGV OIFS step). Change 55: the OIFS
column kernels by direction, four blocks per SM, on a device (-38 % per call, -7.7 % per TGV
OIFS step). Change 56: the fused momentum apply by lines on a device (-33 % per apply, -10.9 %
per TGV OIFS step). The day (52-56, throughput.md): 1.17-1.39x per step on one H100. Weak scaling on full GPUs
(~67-75M DOFs per GPU): np 4 at 68-92 %, the losses beyond ~10-14 % from iterations growing
with the problem. Open for the human: a UCX `+gdrcopy` build (gdrdrv is on the GPU
nodes; the spack attempts failed), the coupled method's FGMRES and projection bases (what
keeps OIFS coupled under 100M), the hanging-node cost in the steady state (+97–146 B/DOF
at p = 6, not traced).

**Defaults: Q6/Q5, grad-div gamma = 6 h_K / k_u, OIFS at CFL 2 (2026-10-16, human: "go with
p=6 and your grad-div scaling recommendation", "make oifs the default it honestly seems to
always be better").** The study is in throughput.md ("The default
polynomial order"): Taylor-Green Re 1600 with OIFS, 96^3 to 192^3, outer CFL 0.5-4. Q6 is the
cheapest order to a 5 % largest dissipation error (126^3: 212 s at CFL 1, 13.7 GiB); Q3 and
Q4 do not reach it by 192^3. With gamma = 2 h_K the velocity PCG's iterations grew 17 -> 38
from Q3 to Q7 at equal DOFs; 6 h_K / k_u keeps them at 17-25 and leaves Q3 unchanged.
`discretization.order_u` defaults to 6 and `order_p` to 5 (a deck that sets only `order_u`
gets k_u - 1); `physics.grad_div_h` defaults to < 0 = 6 / k_u (`Parameters::GradDivH()`).
`time.convection` defaults to `auto` (`Parameters::Convection()`): OIFS where it applies
(Navier-Stokes, the convective form, fixed or CFL steps), IMEX for Stokes, the rotational
form and error-controlled steps; `time.cfl_target` defaults to auto (`CflTarget()`: 2 under
OIFS, 0.5 under IMEX) and `time.oifs_cfl` to 1.5, the study's settings. The verify tier
follows the defaults by design, so it now verifies Q6 with OIFS. Q6 reaches 2 % at 252^3
(1.79 %) and the channel is validated at Q6 (2026-10-17, below).

**GPU memory toward 100M DOFs on one H100 (2026-10-16, branch memory-audit).** The pressure
LOR-AMG was 39 % of a Q7 run (752 B/DOF). Freeing MFEM's LOR scaffolding after assembly
(6d90d0c) saves 460 MiB; capping hypre's device pool (it rounds every allocation up to a
power of 8 and caches every freed block) saves 2.7 GB at Q7 128^3 with no time cost
(ConfigureDevice; docs/gpu_memory.md). Q7 1,946 -> 1,488 B/DOF. Measured capacity at Q6
(Taylor-Green, one H100): 39.2M DOFs before (out of memory at 57.3M), 57.3M with the fixes
(out at 70.4M), 70.4M with `solver.outer: schur` (out at 85.5M). The working list (the
measured ledger at 57.3M, the phases, the held items): gpu_memory.md, "Q6: toward 100M DOFs".
Change 17 (no RAP over identity prolongations, three kept copies dropped) saves 3.6 GiB at
57.3M in both methods, bitwise unchanged; change 18 (the collocated mass as its diagonal)
3.2 GiB more, at roundoff and 10–14 % faster; change 19 (our own PA blocks from MFEM's
public pieces, shared element vectors) 2.8 GiB more, bitwise unchanged. Peak at 57.3M:
coupled 61,955 MiB, Schur 48,127 MiB (881 B/DOF); change 20 (the divergence from the
cached Jacobians, phase 2) 2.6 GiB more at 0.5–1.4 % per step: coupled 57,603 MiB, Schur
46,031 MiB (842 B/DOF); changes 21-25 (OIFS keeps only its combined wind data,
the on-the-fly OIFS apply as an opt-in, IMEX's convection on the fly, the lean vector
mass): OIFS coupled 50,465 MiB, Schur 40,351 MiB (738 B/DOF); IMEX coupled 44,745 MiB,
Schur 32,165 MiB (589 B/DOF).

**3D verification of the GPU memory work (2026-10-17).** Human: "run the 3d verification
tests (TG/channel flow/3d cylinder/... curved boundary ARM example) to really be sure we
didn't introduce any bugs". Main of before the memory work (aa6aa7e) against main after it,
one H100 each: Taylor-Green Re 1600 at Q6 (126^3 with OIFS at CFL 1 and 2, Schur, IMEX, the
on-the-fly opt-in, np 2; 192^3; 252^3 = 57.3M DOFs coupled and Schur), the channel at Re_tau
180 at Q6, DFG 3D-1Z and 3D-3Z at Q3 and Q6, and 3D-3Z with AMR on the curved cylinder (np 1
and 2). **The same answers**: dissipation curves within 3e-6 of the peak to t = 4, forces
within 4e-5 over the drag peak, channel statistics within the sampling noise; the new main
2-17 % faster, and at 57.3M 70.1 -> 51.0 GiB (coupled) and 58.8 -> 39.7 GiB (Schur) over a
whole run. Accuracy at Q6: 252^3 under 2 % in the largest dissipation error (1.79 %); the
channel within 1.7 % in U+ (Re_tau 178.04); DFG c_D within 0.014 % (3D-1Z) and c_D,max within
0.41 % (3D-3Z; 0.17 % with AMR). Results in benchmarks/*/README.md; where each memory change
applies and the open gaps in gpu_memory.md. It also found the bug below.

**Q6 on curved meshes on a GPU (2026-10-17, change 26).** The 3D verification (human: "run
the 3d verification tests ... to really be sure we didn't introduce any bugs") found the Q6
DFG 3D decks stopping at setup on the H100 with an illegal address -- in main and in the main
of before the memory work alike; the CPU runs them. The cylinder meshes take the velocity
order as their geometry order (7 nodes per direction at Q6), and the dealiased convection's
10-point rule (MFEM's form on curved elements: IMEX's explicit term, OIFS's wall nodes) asks
MFEM's quadrature interpolator for the Jacobians, whose runtime-size kernels hold 8 per
direction on a GPU, unchecked. `AddInterpolationKernels` registered compile-time kernels for
linear nodes only; it now covers node orders 1-7 at 9 and 10 points (3D; 9-11 in 2D). Curved
meshes at Q7 (11 points, where MFEM's 3D kernel cannot launch 11^3 threads per element): the
dealiased convection takes the OIFS kernel's stored path instead of MFEM's form (change 27,
human: "ya do that small change for Q7").

**Branch `verification` (2026-10-15, PR #1, merged 2026-10-16): CI and the verification
tier (Phase 1 of the verification audit).** Human: "start phase 1", "the defaults will be
changing over time", rate threshold theory - 0.3 approved, grad-div "an open question",
"can you set up the CI ... are you able to handle the caching?".
- **CI** (`.github/workflows/ci.yml`, `environments/ci/spack.yaml`; developing.md §7.6):
  style, `cpu` build, Doxygen, fast tier on every push to `main` and every PR; the verify
  tier every third day (human: "it's free right?": yes, public repo), by dispatch, or
  with the PR label `verify`. Spack stack cached by the Spack release + the env's hash:
  the cold build took ~1 h (2026-10-15); with the stack and ccache warm a run takes
  ~11 min (run 38807839901: build 33 s, fast tier 198/198 in 5.3 min, verify 3/3 in
  3.7 min).
- **Verify tier** (label `verify`, np 4; developing.md §7.2): `verify_spatial_test`,
  `verify_kovasznay_test`, `verify_temporal_test`, helpers in `test/verify_util.hpp`,
  levels and the 0.3 margin in `baselines.yaml` `verification`. Written in a sandbox
  with no toolchain: compiled and run only in CI. **All three pass in CI** (run
  38805467828; 90 / 73 / 58 s at np 4); the rates are in the run summary and PR #1.
- **Next (audit Phases 2-4):** provenance (commit, defaults) in `summary.json`, splitting
  `nse_mms_test` / `tgv_nse_test` (they time out on a GPU), then the solution-
  verification and validation studies on Bridges-2.

**Ranks that own no elements run (2026-10-16, branch `throughput`).** `benchmark_setup_test`
M1 (a 3-element Gmsh mesh) segfaulted at np 4 on rank 3, which owns no elements (job
49276869; with the pinned MFEM and 4.10 alike). The CPU test jobs ran np 1 and 2 only, so
it went unnoticed. `StokesSolver::SolveTrue` built its kept block vectors only when their
total size changed. An empty rank's offsets are {0, 0, 0} from the start, so it never built
them, and `GetBlock(0)` dereferenced a BlockVector with no blocks. It now compares the
block layout. Same family in the checkpointed guess state (32287da): whether x_prev / A
x_prev existed was read off the vectors' sizes. An empty rank wrote no `guess_*_prev` /
`guess_*_ax` files, so a restart aborted there, and the import would have marked A x_prev
absent on that rank alone. `SolverGuessState::has_previous_x` / `has_step_ax` carry it now,
the same on every rank. New `empty_partition_test` puts every element on rank 0, so the
other ranks are empty at np 2 and 4. A Stokes cavity and Navier-Stokes Taylor-Green (IMEX,
OIFS, CFL control) match the default partition to 1e-8, and a restart with the guess state
(project, extrapolate) is bitwise. Bitwise-neutral otherwise: the old and new `run_case`
write byte-identical checkpoints (cavity, np 2, both guesses). smoke|fast green at np 1, 2
and 4 on the CPU (job 49277774, 72 tests each). Next: the CPU test jobs should also run
np 4 (`-R "_np4$"`).

**Schur complement (Uzawa) outer method, as an option (2026-10-16, branch
`claude/schur-cg-solver-9sf1r5`, PR #2, merged 2026-10-16, human: "merge to main and
push"). Written in a cloud session that cannot build; CI is its only build: style,
`-Werror` build, Doxygen and the fast tier pass there (CPU, np 1/2/4, 219/219). Never run
on a GPU, nothing timed.**
Human: "implement the uzawa stuff here as an option for the solver" (the proposal is
[design/uzawa_schur_cg_spec.md](design/uzawa_schur_cg_spec.md)); on the initial guess,
"ok ya that makes sense" (pressure-only projection under Schur). `solver.outer: coupled |
schur` (default coupled), `solver.uzawa.*` (flexible CG only, human: "drop both" for plain CG
and pressure FGMRES; inner tolerance `relaxed` | `tight` | `loose`, ...; not with the
rotational form), `solver.log_interval` (per-step solver log,
either method, with device memory). New `SchurComplementOperator`, `SchurSolver`,
`VelocityCG`; `StokesSolver` dispatches on `outer`; the pressure projection is
checkpointed. Memory (human: "hyper focused on ... minimizing memory usage"): the Schur
path builds nothing coupled-only and holds 4 velocity + 2k + 6 pressure vectors, 49
B/vdof against 115 before the trim (counted, not measured; uzawa.md §4). Departures
from the proposal and the study plan: [uzawa.md](uzawa.md). **Next:**
`bench/uzawa/study.sh` inside a GPU allocation (Phases 0, 2, 3, and `capacity` for the
memory; submits nothing). Gaps: the Python bindings do not expose the new keys.

**CI builds bessemer for the Spack stack's target (2026-10-16, branch
`claude/schur-cg-solver-9sf1r5`).** `persistent_fgmres_test` failed in CI at np 1 and
4 (on `main` too): FGMRES(4) differed from MFEM's by up to 4e-13 relative against a
1e-14 limit, and even FGMRES(50) was not bitwise (1-2e-15). The arithmetic outside
libmfem is MFEM's own Givens rotations and back substitution, copied; MFEM is built
by Spack with `-march=x86-64-v3` (FMA contraction), bessemer was built for the
baseline x86-64. The build step now sets `CXXFLAGS=-march=x86-64-v3`, and the fast tier passes
219/219 (CI run 38963922762). The differences a passing test prints are not in the CI
log, so "bitwise" is not confirmed, only within the test's 1e-14. A desktop or cluster
build without the stack's `-march` may show the same failure.

**Duplicate deck keys are an error; bench.py's `--set` replaces template keys
(2026-10-16, branch `throughput`).** `bench.py run --set discretization.order_u=5` had
appended `order_u: 5` after the template's `order_u: 3`, and the loader read the first
(job 49270645 ran at order 3). Now `write_deck` replaces a key the template sets (any
section, nested keys too) and adds the rest at the end of their section; the loader aborts
on a key a map sets twice, naming the key, the section and the lines
(`DeckReader::DuplicateKeys`; tests `Deck.DuplicateKeysAreReported`,
`Deck.EveryCaseDeckLoads`, `run_case_rejects_duplicate_key`). Before this, a `--set` had
no effect when the template set that `discretization` or `time` key, or when it named any
other section the template has (`physics`, `mesh`, `output`, ...: a second section);
`solver.*` and new `discretization` / `time` keys worked.
Verified on the CPU (job 49276422: smoke|fast at np 1 and 2) and on two H100s (job 49277617:
`deck_test` and `run_case_rejects_duplicate_key` at np 1 and 2; `bench.py` runs with
`--set discretization.order_u=5` and `--set mesh.elements=...` ran at the overridden order
and mesh). `scripts/docs.sh` not run (no doxygen on Bridges-2).

**Throughput merged into main; new solver defaults (2026-10-16).** The `throughput` branch
(2026-10-14 to 16; record and decision log: [throughput.md](throughput.md)) is in `main`.
One H100 against main of 2026-10-14: Taylor-Green 64^3 Q3/Q2 to t = 20 in **17.8 min instead
of 12 h 44 min** (43x), the channel at Re_tau 180 from t = 0 to 120 in **13.8 min instead of
7 h 15 min** (31.6x), 37-46x per step; compile-time kernels for velocity orders 2 to 6; an
order study (Q4 and Q5 reach a given accuracy fastest; Q3 is the fastest per DOF). **Case-level
defaults** (human 2026-10-16, option (a) in throughput.md's "Restarts with the projected
guess, and the defaults"): `solver.initial_guess: project` (k 8), `tolerance_reference: step`,
`rtol` 1e-6, `n_inner` 2; checkpoints keep the solvers' guess state, so restarts stay bitwise
exact. `rtol` is per problem: the Taylor-Green and channel decks set 1e-3 (CFL-limited
steps); a small fixed dt or an error controller wants 1e-6. Next (human): MFEM 4.10 with the
PR #5464 patch, then a single-kernel dFEM probe.

**Throughput plan, overnight 2026-10-14/15 (branch `throughput`, not merged).** Human:
up to 80 GPU SU, push only the branch, "as far as possible", projection "absolutely ...
as an option", single precision / libCEED / Tier 5 / new cases deferred. Record and
decision log: [throughput.md](throughput.md). Kept: the benchmark harness (`bench/`), NVTX
ranges, MFEM kernel specializations (+24 %), the fused viscous + grad-div kernel (+7 %),
and the opt-in `solver.initial_guess` / `solver.tolerance_reference`; Tier 3 found
`solver.n_inner` 3 (+54 %) and, with the projection, `rtol` 1e-6. With all of it: TGV
32^3 14.2x, 64^3 14.6x per step, and the 64^3 run to t = 20 in **1 h 01 min instead of 12 h
44 min** with the same dissipation curve (2.4e-4). Defaults unchanged pending the human
(decided 2026-10-16, above).

**GPU memory, third round (2026-10-14): the single-H100 limit is 88^3 Q3/Q2
(55.2 M velocity dofs); about 85^3 at full speed; 64^3 steps 8 % faster.** Human: "do all
of these but keep a path for the rotational form ... do the vector diffusion thing too
... except keep a path to the built in mfem way", then "do 1", "Yes, do that for sure",
"yes", "go ahead and make that CFL change". Changes 11-16 in
[gpu_memory.md](gpu_memory.md):
- **Compact vector-diffusion storage**, the default (`discretization.vector_diffusion:
  compact | mfem`): 6 numbers per point instead of MFEM's 27, apply bit-identical to
  MFEM's in all 16 tested cases. 64^3: 42.8 -> 37.6 GB, 2 % faster.
- **Nine small items** (shared P^T A P wrappers, symmetric point-block inverses, `keep_rhs`,
  w* only for the rotational form, the plain pressure mass freed, gamma per element, ...):
  0.7 GB at the Taylor-Green peak; more with walls and forces.
- **Grad-div stores alpha only** (adj(J) recomputed from MFEM's cached Jacobians): 64^3
  37.6 -> 35.3 GB, the velocity-block apply 5 % slower.
- **Delta-t-independent refresh work kept** (momentum diagonal split as c0 M_diag + a kept
  nu K + grad-div diagonal; point-block Jacobi and its CG updated in place): refresh
  1.22 -> 0.009 s per step, 64^3 step 11.11 -> 9.84 s, for 0.5 GB.
- **Restart-length guard measures the first solve** (`StokesSolver::WarmUp`, once per
  solver, 1.7 s at 64^3): the 20 % allowance missed the 9.2 GB workspace and 86^3 ran out
  of memory; now 86^3 and 87^3 run, at restart lengths 3-6 and 1 (87^3: 29.9 s per step,
  ~20 % slower than proportional).
- **CFL Jacobians per call** (MFEM's routine, not its cache; bit-identical): 64^3
  35.8 -> 34.7 GB; 88^3 runs (restart length 1, 32.4 s per step), 89^3 does not.
- **Tests:** new `StokesOperator.MomentumDiagonalAfterRefreshes2D/3D`,
  `PbjGradDiv.P4_NewDiagonalMatchesAFreshBuild`, `SchurQuality.WarmUpLeavesTheSolveUnchanged`,
  `Cfl.C4_PerCallJacobiansEqualTheCachedOnes`, R3 rewritten. Before the CFL change:
  fast tier CUDA np 1 64/64 (one H100), CPU np 1 and np 2 66/66. With it: CPU np 1 66/66.
  With the CFL change (after the commit, at the human's request): CUDA np 1 64/64, host
  device 10/10, CPU np 1 and np 2 66/66 (jobs 49216290, 49216292); CUDA np 2 (two H100s,
  job 49216291) still queued at the time of writing.

**CPU environment works on RM nodes (2026-10-14).** `environments/psc_cpu` re-pinned to
the b2rm gcc (`hosts` selects it on `r*` nodes, `preset cpu`), `reuse: false`, numpy
pinned below 2.5 for numba; Open MPI built without CUDA (the install check reads its
`opal_built_with_cuda_support` flag). CPU build plus fast tier on 32 RM-shared cores:
np 1 and np 2 66/66. The point-block Jacobi abort seen with the CUDA build on the host
device did not recur (CUDA build, host device, 2026-10-14) and never appears in the CPU
build.

**A failed solve cuts the step, by default (2026-10-13, on `main`).** Human: "if the
outer solver fails ... cut the timestep and then let it try to increase again", "a hold of
5 steps", "just option 1 and 2". `time.on_solver_failure: cut_step` (default) | `abort`;
`retry_factor` 0.5, `retry_hold` 5, `retry_max` 8, `retry_dt_min` 0. With `cut_step` the
FGMRES restart length is capped before each solve to the free GPU memory. A "max-DOF"
mode (size the mesh to the memory) was dropped: with AMR the mesh grows during the run.
Tests: `step_retry_test` R1 (fixed steps: two cuts, hold, growth back; error 0.002705
against 0.002696 undisturbed), R2 (CFL), R3 (the restart-length formula). Fast tier
CUDA np 1 64/64 (one H100) and np 2 64/64 (two H100s, one per rank), 2026-10-14; the new
`SchurQuality.RefreshWithViscosity...` first hung at np 2 (a rank-0-only collective in
the test's print, found with stack traces; fixed). The cap near the memory limit was
first tested 2026-10-14 and failed (86^3 out of memory), fixed by the warm-up above.
**Not tested:** a real (not injected) solver failure; `nse_mms_test`, `tgv_nse_test`.
Memory record: [gpu_memory.md](gpu_memory.md).

**Taylor-Green 64^3 at CFL 0.3 on one H100** (job 49142049, 2026-10-14;
human: "back off the 75^3 run to 64^3 ... increase the CFL target a little bit to .3"):
**Finished**: t = 20 in 12 h 45 min (3284 steps, 13.9 s per step, ~25 SU); energy never
increased, so CFL 0.3 is stable here and matches CFL 0.25 in E to 0.02 points. Max
−dE/dt 0.01294 at t = 8.95 (reference 0.01286 at 8.97); E within 0.6 % to the peak, 5 %
low at t = 20 (under-resolved decay). Table in
[../benchmarks/tgv_re1600/README.md](../benchmarks/tgv_re1600/README.md). The
restart-length cap stayed at 119 (of 200), far above the ~15 iterations used. Output
`runs/tgv_re1600/tgv_n64_1gpu` beside the repo on Bridges-2.

**GPU memory, second round (2026-10-13, on `main`): the single-H100 limit is 80^3 Q3/Q2
(41.5 M velocity dofs).** Human: "tackle the first 3", "go ahead with the single solver
change", "make that simple change to the CFL stuff too".
- **Measured** (Taylor-Green defaults, ~11 steps, one H100): 64^3 43.5 GB (38 GB after the
  first step), 72^3 59.0, 76^3 68.0, **80^3 78.9 GB**, 81^3 out of memory; 1.95 kB per
  velocity dof. 64^3: 10.7 s per step, 10.4 outer iterations. Headroom: FGMRES holds two
  system-sized vectors per outer iteration (0.37 GB each pair at 64^3, 0.73 GB at 80^3);
  these runs peaked at 14 iterations.
- **Diagnostics hold nothing**: `FlowDiagnostics` sums the integrands in one device pass
  (`post/flow_sums`, nvcc) instead of three assembled forms (10 GB at 64^3, built even
  with the CSV off: the summary asks for the energy). `All()` gives the three values.
- **One solver for the starter and the BDF steps**: `StokesSolver::Refresh(c0, nu)` and
  `StokesTimeIntegrator::AdoptSolver` hand the trapezoidal starter (built at nu / 2) to
  BDF2, and BDF2's to BDF3. Not with the rotational form or error control (old path).
  Test `SchurQuality.RefreshWithViscosityMatchesAFreshSolver`.
- **gamma uses the fluid's viscosity in every solver** (`grad_div_nu_ref`): the starter
  computed its nu-proportional part from nu / 2. Changes step 1 only, and only when
  `physics.grad_div_nu` is not 0.
- **The unconstrained viscous operator** (6.75 GB at 64^3) is built on demand and freed
  right after the starter's right-hand side; the BDF solver never builds it.
- **CFL estimator**: its two velocity copies live per call, and with the GLL nodal basis
  nothing is interpolated (2.0 -> 1.2 GB at 64^3; the rest is Jacobians at the nodes).
- **Pressure AMG knobs**, defaults unchanged: `solver.lp_strong_threshold`,
  `solver.lp_agg_levels`, `solver.lp_print_level`. At 64^3 the threshold changes nothing;
  one level of aggressive coarsening saves 2.1 GB with the same iterations over 11
  steps (not tried through the turbulent phase). Human 2026-10-13: AMG options are "an
  optimization for another day".
- **Looked at and left**: the convection operator (half of its 4.5 GB is the Jacobians it
  shares with the other 5-point forms, half MFEM's own data); the Schur setup (77 % AMG);
  the momentum form's 27-per-point diffusion data (human: do not replace MFEM's core
  operator; reversed 2026-10-14 with MFEM's path kept, above).
- **Tests**: fast tier CUDA np 1 63/63 before the last three small changes, then 14
  selected np 1 tests (time stepping, CFL, Schur, checkpoint, AMR event) pass. np 2 (two
  H100s, one per rank): fast tier 64/64 with the step-retry work on top (2026-10-14). `nse_mms_test`,
  `tgv_nse_test` and a CPU build not run.
- **Taylor-Green 64^3 at CFL 0.25** (two H100s, cancelled at t = 10.8 to rerun on one GPU
  with this code): energy never increases; max -dE/dt 0.01294 at t = 8.95 (reference
  0.01286 at 8.97); viscous dissipation within 1 % of the reference to the peak; energy
  0.5 % low at the peak, 1.5 % at t = 10.5. `runs/tgv_re1600/tgv_n64` beside the repo.

**GPU memory, first round (2026-10-13): 64^3 Q3/Q2 fits one 80 GB H100; the limit is 69^3.**
Human: "find out how many dofs I can fit on a GPU", then "implement both".
- **Measured** (Taylor-Green defaults, ~10 steps, one H100): before, 56^3 (14.2 M velocity
  dofs) at 74.5 GB and 60^3 out of memory; after the two fixes below 64^3 at 63.9 GB,
  **69^3 (26.6 M velocity + 2.6 M pressure dofs) at 78.4 GB, 70^3 out of memory**; about
  3.0 kB per velocity dof.
- **The profiler has a gauge** (`Profiler::SetGauge`; `output.profile: sync` on a GPU
  installs `DeviceMemoryInUse`): three more CSV columns per region, what its first call
  left allocated, the same over all calls, and the peak while it was open.
- **Fix 1: 11 GB of GPU stack.** MFEM's `VectorDiffusionIntegrator::AssembleDiagonalPA`
  reserves 44 kB of stack per GPU thread on its first call (work arrays sized by
  `DofQuadLimits::MAX_*`), 11 GB on an H100 for the life of the process, independent of
  the problem. `LeanVectorDiffusionIntegrator` overrides only that diagonal. Written up
  for upstream in [upstream/mfem_vector_diffusion_diagonal.md](upstream/mfem_vector_diffusion_diagonal.md);
  **not reported to MFEM yet**.
- **Fix 2: the trapezoidal starter solver** is built on first use and released before
  the first BDF solver is built (not under OIFS or error control), so two whole operators
  and preconditioners never coexist; a resumed march never builds it.
- **Tests:** fast tier CUDA np 1 63/63 (without `nse_mms_test`, `tgv_nse_test`);
  `lean_vector_diffusion_test`, `profiler_test`, `pbj_grad_div_test` also at np 2. **Not
  run:** the fast tier at np 2, a CPU build, OIFS or adaptive stepping after fix 2 beyond
  the np 1 tier.
- **Where the rest goes at 64^3 (63.9 GB)**, candidates, none acted on: 14.9 GB in
  `case::setup` outside the time integrator, with no regions yet; vector-diffusion PA data
  13.5 GB (27 numbers per point where 6 would do, in the momentum form and again in
  `viscous_unconstrained`); the pressure LOR-AMG about 9 GB for 2.1 M unknowns (3 GB at
  setup, 5.8 GB at its first apply); 4.9 GB unattributed in `time_integrator::setup`; each
  momentum reassembly allocates 9.4 GB before the old form is freed.
- Feature costs at 32^3 (23.7 GB before the fixes): grad-div 1.1 GB; OIFS +4.4 GB;
  `a_pc: loramg` +21.7 GB; PBJ, the Schur grad-div term and diagnostics within 0.3 GB.

**Branch `profile-output` (2026-10-12, not merged): the profiler's results are saved.**
Human: "can we save out the profiler results so we know where the compute is being
spent?", ahead of the 3D Taylor-Green study.
- `apps/run_case` writes `<path>/<name>_profile.csv` next to the summary (every region:
  nesting path, calls, inclusive and exclusive wall seconds, max and min over the ranks),
  and the summary gets a `profile` section (top three levels). Deck key
  `output.profile: off | wall | sync`, default `wall`. `sync` waits for the GPU at every
  region boundary (`incns::DeviceSync`, a new nvcc TU `util/device_sync.cpp`), so GPU time
  is charged where it is spent. `Case::WriteProfile`, `Profiler::WriteCsv`/`SetSync`.
- Tests: `profiler_test` (the hook; the exact CSV), new `profile_output_test` W1 (a short
  run's file: nesting, one step region per step, children fit in their parent). Bridges-2,
  CUDA backend, np 1 and 2: pass, with `deck_test` and `run_case`; style clean. **Not
  run:** a CPU build; the Python bindings do not expose `output.profile`.
- **First 3D profile** (`tgv_re1600_n16` to t = 0.5, one H100, np 1; 18 steps, 112 s):
  87 % of the run is `inner_cg`, the inner CG of the Cahouet-Chabard Schur
  preconditioner (106 applies, 0.93 s each, for 32,768 pressure unknowns); 12 % is in
  `case::run` outside the steps (not broken down: the OIFS substeps have no region);
  assembly and the outer FGMRES are negligible. `wall` and `sync` agree to 0.3 % here and
  cost the same. **Next:** regions inside the inner solve (the LOR-AMG cycle, B M^-1 B^T)
  and around the OIFS substeps, then the same at 32^3: is 0.93 s per apply a fixed cost of
  a small problem on a GPU, or does it scale?
- **Answered (same day): the AMG smoother.** With regions inside the Schur preconditioner
  (`cc::lor_amg`, `cc::bmb`, `cc::gradient`, `cc::divergence`,
  `cc::velocity_mass_inverse`, `cc::pressure_mass_inverse`; `velocity_block::apply`,
  `schur_block::apply`; the time integrator's and Case's pieces; `ProfiledOperator`),
  `cc::lor_amg` was 88.6 % of an IMEX run: 86 ms per V-cycle at 16^3, 670 ms at 32^3. It
  was on the GPU, but smoother 8 (l1 symmetric Gauss-Seidel) runs there as triangular
  solves. With l1-Jacobi (18): 0.74 ms per cycle, the 16^3 run 16 s instead of 145 s, same
  7 outer iterations. Now `SymmetricAmgSmoother()` (18 on a GPU, 8 on a host) and
  `VerifyHypreAmgOnDevice` (aborts on any option outside hypre's GPU list), test
  `hypre_device_test`. After that fix the diagnostics were 77 % of the run (host element
  loops): rewritten as `FlowDiagnostics` (PA forms, device), test
  `Diagnostics.AssembledFormsMatchTheFreeFunctions`.
- **Grad-div on by default, point-block Jacobi under the velocity PCG** (human, same day).
  Deck: `physics.grad_div_h` (default 2) and `physics.grad_div_nu` (default 0),
  γ = c_h·h_K + c_ν·ν; the old `grad_div` / `grad_div_scale` deck keys are rejected.
  `solver.a_pcg_pc: auto | jacobi | pbj`, `auto` = PBJ with the grad-div nodal blocks when
  there is a grad-div term. New test `pbj_grad_div_test`. The TGV decks are IMEX at CFL 0.5
  (human: "I'd rather not use OIFS for this just imex").
- **First build and runs (jobs 49098413, 49098829; CUDA, one H100, np 1).** Built first
  time. Fast tier np 1: 59/64; `nse_mms_test` and `tgv_nse_test` time out at 900 s as on
  2026-10-10; `deck_test` failed on the stale deck reference; **`oifs_test` O2 fails**
  (OIFS error 0.0175 / 0.0453 against 1.2 x the IMEX error 0.0039: moved by today's
  defaults, cause not found); `pbj_grad_div_test` P1 failed (the test's host probing is
  suspected, P2 passes: PBJ 665 / 1699 CG iterations against Jacobi 744 / 2021). np 2 not
  run. Taylor-Green 16³ to t = 0.5: no grad-div 7 outer iterations, 3.0 s; γ = 2h 28
  iterations, 11.8 s, PBJ = scalar Jacobi (uniform mesh: the nodal coupling cancels);
  momentum reassembly 150 ms per Δt change against 3 ms. 32³: 3.6 s per step (55 s
  before the smoother fix). Hence, human 2026-10-13: γ in the Schur PC's mass term
  (`solver.schur_grad_div`) and γ cached per mesh -- written, see below for their state.
- **Second round (2026-10-13, jobs 49099073, 49099090, 49099146): built, tested, NOT
  merged.** Fast tier, CUDA, one / two H100s, without `nse_mms_test` and `tgv_nse_test`
  (they time out on a GPU; untested on this code): **np 1 62/62; np 2 61/62**.
  `nc_stokes_test` S4 fails at np 2: Cahouet-Chabard iterations 25 against 21 on the two
  meshes at step 0, limit 3 (human 2026-10-10), errors equal to 1e-8. Same kind of spread
  as the hypre GPU random-stream effect that limit was set for, but wider than seen
  before (the AMG smoother on a GPU is now l1-Jacobi). **Limit set to 4 (human
  2026-10-13); the test was NOT rerun with it, and the edit is not astyle-checked.**
  Taylor-Green 64^3 (job 49099148) ran out of GPU memory on one 80 GB H100 after the
  first step: it needs two GPUs or a smaller mesh. `oifs_test` O2 now pins `grad_div_h = 0` (with gamma = 2h the 8x8
  mesh spatial error falls 30x, IMEX 0.097 -> 0.0033, and OIFS at 4-8x the step loses on
  time error: 0.018 / 0.045). `pbj_grad_div_test` passes with host-pointer probing.
  Taylor-Green 32^3 to t = 0.5 (25 steps): defaults 12 outer iterations, 36.3 s;
  `schur_grad_div: false` 23, 73.5 s; `a_pcg_pc: jacobi` 12, 35.8 s; `grad_div_h: 0` 7,
  17.5 s. Momentum reassembly 15 ms (was 1.17 s).
- **Taylor-Green Re 1600, 32^3 Q3/Q2 to t = 20** (one H100, 39 min, 1087 steps, 15-16
  outer iterations; `runs/tgv_re1600/tgv_n32` beside the repo on Bridges-2): max -dE/dt
  0.01292 at t = 10.1, max viscous dissipation 0.01150 at t = 10.1 (reference 0.01286
  at 8.97): the peak is about 1.1 late, under-resolved as expected at 96 nodes. Not yet
  compared point by point (`plot.py`; the reference CSV is not fetched). 64^3 queued
  (job 49099148, 6 h limit, checkpointing; estimated ~10 h to t = 20).
- **STATE (superseded where the items above say so): written, NOT compiled, NOT run** (human: no runs until all of it is written).
  Only the profiling regions were built and tested (CUDA, np 1 and 2, 20/20) before the
  later edits. Uncompiled: `hypre_device`, `FlowDiagnostics`, the grad-div options,
  `AddNodalOffDiagonalPA`, the PBJ extension, `a_pcg_pc`, both new tests. Unformatted
  (`scripts/style.sh format`); `docs/deck_reference.md` not regenerated. **Expect Case-level
  results to move** with γ = 2h on by default (iteration counts, forces; pin
  `grad_div_h: 0` to reproduce old numbers). List what moves; do not re-bless `test/baselines.yaml`. The Python
  bindings do not expose `output.profile` or `a_pcg_pc`. Not pushed (human: wait until the
  grad-div work is done).

**Branch `mesh-census` (2026-10-12, merged): `apps/mesh_census`**, to check whether an
`amr.end_time` freeze of a refinement-only AMR run is safe. `amr/mesh_census` counts
elements per streamwise band; `ReplayMeshCensus` rebuilds a checkpointed run's mesh pass by
pass (as a restart does). Test `mesh_census_test` C1 (exact band counts on a box), C2
(replay = the run's final mesh).

**Restart fixes (2026-10-12):** a restart replaced the per-step history instead of
continuing it (`post/run_monitor` appends now; `benchmarks/common.py` drops rows a later
segment supersedes, using the step's start t - dt); a restart with forces aborted in the
summary before its first step (`Case::WriteSummary` gated forces on StepCount instead of an
available force). Tests `run_monitor_test` R5, R6 (R6 aborts without the fix: checked).

**NOT YET TESTED on the changes below (human 2026-10-10: merge and push anyway, test
after). Do these before relying on them:**
- **The CPU fast tier on the desktop** (`scripts/test.sh cpu -L fast`, `cpu` preset). The
  changes were only run on Bridges-2, where the one build is the CUDA one. Watch the AMR
  tests: the marker's element cap now treats equal indicators as a group (below), which
  can change which elements are refined near `amr.max_elements`.
- **`scripts/docs.sh`**: Bridges-2 has no Doxygen.
- **The Python tests and the debug-device sweep**: queued on Bridges-2 when this was
  written, not yet seen to pass.
- **The GPU fast tier at np 4 and the slow tier.** With two H100s (two ranks per card) the
  np 4 pass got through 16 of 60 tests in 28 minutes (15 passed, `nse_mms_test` timed out)
  and was stopped; the slow tier never ran. They need one GPU per rank: 4 H100s.
- **The CPU backend of the CUDA build is not a CPU reference.** `INCNS_DEVICE=cpu` in
  `build/cuda` passes 175 of 183; the 8 failures are inside hypre ("Error during solve",
  `amr_checkpoint_test`, `amr_event_test`, `unsteady_mms_test` at np 1 and 2; they pass at
  np 4) and MFEM's cuSPARSE path (`point_block_jacobi_test` np 2 and 4). All eight pass on
  the CUDA backend; the hypre ones match the July note about a CUDA-built hypre on the
  host. Not confirmed against a CPU build: that needs the desktop, or `psc_cpu`.

**In flight (branch `macos-gcc`, 2026-10-10, first cold-start run of `setup.sh` on a
fresh machine, an Apple Silicon MacBook Air, macOS 15.6.1): not merged.**
- **`setup.sh --yes` failed building `gcc@14.3.0`** at configure: `Configuration
  aarch64-apple-darwin24.6.0 not supported`. Spack's gcc recipe carries Iain Sandoe's
  aarch64-darwin patches only up to 14.2.0. Fix on the branch: `setup.sh` uses
  `gcc@14.2.0` on Apple Silicon (`docs/install/spack.md`, "The compiler"). The rest of the
  setup on macOS is **not yet verified**: resolve, install, build and the fast tier are
  still ahead on this machine; update this entry with what they find.
- **`gcc@14.2.0` then failed in libsanitizer** (~50 min in; configure and everything
  before it passed): linking `libasan.8.dylib` crashes Apple's linker, `ld: Assertion
  failed: (entry.toAddend < 0x400000), function encodeSectionReloc` (CLT 16.3.0,
  ld-1167.4.1, macOS 15.6.1, arm64). Spack's gcc has a `libsanitizer` variant (default on).
  Fix on the branch (human 2026-10-10): `gcc@14.2.0~libsanitizer` on macOS. Cost: no GCC
  `libasan`. That matters less than first thought: bessemer is compiled by Apple clang
  (below), whose own sanitizer runtime `scripts/asan.sh` should use; not yet run there.
  `-ld_classic` still exists in this ld but is deprecated.
- **`spack concretize` then failed** (human 2026-10-10: "we just need to use system
  clang"): `python: CPython does not compile with GCC on macOS yet, use clang` (a Spack
  conflict `platform=darwin %gcc`; it reaches Python through numba/llvmlite, matplotlib
  and `ninja`). Fix on the branch: `c`/`cxx` require `apple-clang@<detected>` (external,
  prefix `/usr`), `fortran` stays the Spack-built `gcc@14.2.0~libsanitizer`; `setup.sh`
  writes both on Darwin and `env.sh` whitelists the `apple-clang` external. Concretizes
  in ~20 s with that (hypre resolved 3.1.0 there, `desktop` has 3.2.0). **Not yet
  verified:** the install, the build, and `env.sh`'s assertion with `gcc`/`g++` resolving
  to `/usr/bin` (its banner also prints clang's version as `gcc=`).
- **`spack install` hung at 33/69 on `python@3.12.13`** (human's Terminal panel in the
  Claude desktop app; stopped, 0% CPU, for an hour; the progress line kept counting).
  Observed: the build worker, `make` and `python.exe check_extension_modules.py` were in
  `ps` state `T` (job-control stop), in a process group that was not the terminal's
  foreground group. **Cause not established**: the theory is a terminal-mode call
  (`readline` import) from a background group; the signal was not captured, and the gcc
  build ran fine from the same kind of tab. Rerunning with stdin detached
  (`spack -e ENV install < /dev/null`) was started; whether that is the fix is not
  confirmed until that Python build passes the extension check. Not tried in Terminal.app.
- **`setup.sh`'s install check was wrong after an interrupted install.** It reported
  step 6 "done" when the environment's view contained MFEM; Spack refreshes the view on
  Ctrl-C with whatever is built so far (MFEM was, python/numpy/ninja were not), so a
  rerun went straight to the build, which failed with "unable to find ... Ninja".
  **Fixed (2026-10-11):** it asks Spack (`spack -e ENV find -L`: every root spec marked
  `[+]`, none ` -  `), only when the lock is current (so `--plan` still changes nothing).
  Checked: this environment reads "installed (all 12 root specs)"; the marker format of an
  uninstalled root was confirmed on a scratch environment; a half-installed real
  environment was not reproduced (the parsing was tested on canned output).
- **Re 200 OIFS deck reproduced on the Mac (2026-10-11, M3 MacBook Air, clang, np 4):**
  `cases/square_cylinder_re200_oifs.yaml`, status ok, 2,191 steps, 3,000 elements (the
  deck's cap, reached at t ~ 50; 47 AMR events), 53,526 velocity + 11,842 pressure dofs,
  wall 1,936 s (32.3 min; the deck says ~13 min on the desktop, so ~2.5x here: a
  fanless Air with four performance cores). Periodic statistics, 10 periods,
  period spread 0.04%:

  | | this run | desktop (GCC, validated 2026-10-09) | paper | vs paper |
  |---|---|---|---|---|
  | C_D mean | 1.4265 | 1.4265 | 1.44 | -0.9% |
  | C_L,rms | 0.3924 | 0.3923 | 0.42 | -6.6% |
  | St | 0.1565 | 0.1565 | 0.151 | +3.6% |

  So the clang build agrees with the GCC desktop run to the digits quoted (same step
  count), and the C_L,rms gap to the paper is the desktop's too, not a platform effect.
  One run on one AMR level: not a mesh-converged answer.
- **Fast tier on the Mac (2026-10-11, `scripts/test.sh cpu -L fast`, clang, M3 Air):**
  180 of 180 pass (0 failed), 269 s of test time (4 min 31 s wall), after the three
  clang fixes. 3 of the 180 were *skipped*, as designed: `benchmark_plot_tgv_re1600`,
  `_channel` and `_dfg_3d` need `python3 benchmarks/fetch_references.py` (exit 77, not
  fetched on this machine). This closes the macOS bring-up of `scripts/setup.sh` ->
  build -> doctor -> case -> fast tier. **Not run on the Mac:** the slow tier,
  `scripts/asan.sh`, the Python tests (`cpu-python` preset), `scripts/docs.sh` (Doxygen),
  `scripts/debug_device.sh`.
- **First build on clang (`scripts/build.sh cpu`), 2026-10-11:** configure passed (AppleClang
  17.0.0 through the Spack `mpicxx`, MFEM and GTest found). Then compile stopped at
  `src/precond/mass_inverse.hpp:187:13: error: private field 'comm_' is not used
  [-Werror,-Wunused-private-field]` (clang-only warning, GCC doesn't have it; header
  included from `cahouet_chabard.hpp`). Not fixed: it is a `src/` change (ask). A
  keep-going pass (`ninja -k 0`) compiled 106 of 173 targets and found three distinct
  causes, all clang-only (GCC accepts them), all in `src/`; the other 67 targets depend
  on the failed ones, so more may appear once these are fixed:
  1. `src/precond/mass_inverse.hpp:187` `comm_` is set (line 69) and never read:
     `-Wunused-private-field`; 57 of the errors, via every includer of the header.
  2. `src/config/parameters.cpp:747` `DeckReader r(YAML::Node(YAML::NodeType::Map));` is a
     most-vexing parse for clang (`-Wvexing-parse`, then three follow-on errors).
  3. `src/config/deck_reader.cpp:119-120` a lambda captures the structured bindings
     `parent`/`leaf`: a C++20 feature, the build is C++17 (`-Wc++20-extensions`).
  **Fixed (2026-10-11, human approved):** commit `2870c38` (remove the unused member;
  brace-initialize the `DeckReader`; hold the `SplitPath` pair and take references). The
  whole `cpu` preset then built, 125/125, with no further clang errors (only the
  linker's "ignoring duplicate libraries" warnings); `scripts/style.sh check` clean;
  `scripts/doctor.sh --run` all good (cavity smoke run at 2 ranks, 3 s).
- **`scripts/env.sh` failed when sourced from zsh** (the macOS login shell, and what
  AGENTS.md's `. scripts/env.sh` runs in): `'gcc' -> '/usr/bin/gcc' is NOT inside
  spack`. Cause: the declared-external prefixes are one newline-separated string
  iterated with an unquoted `for _p in $_spack_ext_prefixes`; zsh does not word-split
  unquoted expansions, so with TWO declared externals (the Spack gcc and `apple-clang`,
  /usr) the pattern match never saw `/usr`. One external worked, which is why Linux
  never saw it. **Fixed (2026-10-11, human asked):** a `while read` loop over a here-doc
  (and an empty line is skipped, since the pattern would be `/*`). Checked: `. scripts/
  env.sh` returns 0 under `sh`, `bash` and `zsh` on this Mac. Not run on Linux.
- **`ccache` is not installed** on this machine: the build prints "compiler launcher
  'ccache' not found on PATH -- building without it". AGENTS.md rule 4 (never
  `rm -rf build/`) assumes the cache; here every rebuild is a full one.
- **Setup layer restructured (human 2026-10-11, "keep one setup.sh as the single entry
  point ... move platform facts into small sourced files"):** `scripts/platform/darwin.sh`
  and `linux.sh` hold the GCC spec, the `c`/`cxx` compiler, the job count and the
  `apple-clang` external; `setup.sh` sources the one `uname -s` picks (`INCNS_PLATFORM`
  overrides). Checked on this Mac: the generated `spack.yaml` machine settings are
  identical to before, `--plan` shows `gcc@14.2.0~libsanitizer` here and `gcc@14.3.0` under
  `INCNS_PLATFORM=linux`, an unknown platform exits 2 with a message, and `nproc` -> `sysctl
  -n hw.ncpu` fixes the `-j 4` fallback. **Not run on Linux.** `env.sh`'s zsh bug, the
  install check and the `src/` warnings are NOT part of this change (shared bugs, below).
- **Compiler version in the docs (fixed 2026-10-11):** `docs/install/spack.md` and the
  `setup.sh` header, `desktop.md` and the YAML examples no longer name a version: the
  GCC version is written once, in `scripts/platform/<uname -s>.sh` (human 2026-10-11: a
  different gcc is coming).

**Just landed (`gpu-validation`, merged to `main` and pushed 2026-10-10): the Bridges-2
GPU environment rebuilt, the setup machinery made machine-neutral, and the first GPU run
of the fast tier.**
- **GPU fast tier (H100, CUDA backend, MFEM `1c19aba`, hypre 2.33.0 +cuda):** np 2 60/60.
  np 1 61/63: `nse_mms_test` and `tgv_nse_test` timed out at 15 min while sharing one card
  (every np 1 test uses GPU 0); both pass at np 2 (428 s, 224 s), and `nse_mms_test` passed
  alone at np 1 (13/13, 291 s) before the last changes. Run GPU tests two at a time at
  most, and with Open MPI's core binding off (`PRTE_MCA_hwloc_default_binding_policy=none`):
  concurrent `mpirun`s otherwise pin to the same cores.
- **AMR marker, element cap** (`refinement_marker.cpp`): indicators equal to within 1e-8
  relative are marked together. When the cap would split such a group, the whole group is
  marked if that exceeds `amr.max_elements` by at most 10% (human: a small overshoot is
  fine), and none of it otherwise. Why: the bisection used exact comparisons, so in a
  symmetric flow roundoff chose which members of a tied group were refined. A CPU restart
  is bitwise, a GPU one reproduces the state to ~1e-11: `amr_checkpoint_test` 2D differed
  by 74% on the GPU. It now matches to 1e-13 (112 elements; 118 before, cap 120).
- **`amr.threshold_mode` defaults to `absolute`** (human: refine to the tolerance you
  need, not to an element count; no cap by default). It was `relative`. Every test, app
  and example that sets `theta` now sets `relative` explicitly, so their results are
  unchanged. A deck that sets only `amr.theta` must add `threshold_mode: relative`.
- **Tolerances (human-approved):** `amr_event_test` E2 compares attempted steps to 1e-9
  on every backend (GPU: ~2e-11). `nc_stokes_test` S4 allows iteration counts to differ by
  3 (was 1): on a GPU hypre's coarsening draws from one random stream per process, so an
  AMG hierarchy depends on every setup before it (the same problem took 21 to 24
  iterations depending on what had run; the two meshes differed by up to 2; counts are
  identical on a host backend). `HYPRE_SetUseGpuRand(0)` removes the effect but made the
  GPU tests 3 to 11 times slower, so it is not used. **Open:** reseed the GPU stream per
  AMG setup instead; until then a restarted GPU run builds different hierarchies from the
  uninterrupted one.
- **Build:** the CUDA compute capability comes from the environment's `cuda_arch`
  (`scripts/env.sh` exports `INCNS_CUDA_ARCH`), no longer a hardcoded 90.
- **Setup machinery:** `scripts/setup.sh --plan` is the one entry point. An environment
  can carry `INSTALL` (its own procedure; `setup.sh` reports and runs nothing), `hosts`
  (full-hostname patterns), `preset` and `spack.yaml.sha256` (written by
  `scripts/envlock.sh`: a lock is stale when its `spack.yaml` changed, whatever the file
  times say). `scripts/doctor.sh` only checks; `--run` builds and smoke-runs.
  `scripts/test.sh` takes the preset as its first argument. `setup.sh --yes` refuses on a
  cluster login node without `--on-login-node`.
- **Agents:** rule 3 asks before any job or allocation and says nothing in the repo is
  permission; the committed `.claude/settings.json` only denies or asks. Allow rules are
  personal (`.claude/settings.local.json`, git-ignored).
- **Bridges-2:** `psc_gpu` re-resolved (MUMPS, the MFEM pin, `py-pip`) and installed with
  `environments/psc_gpu/install.sbatch` (one H100, 10 min for the five new packages);
  numba 0.66.0 from the wheel folder. `psc_cpu` (the same stack, no CUDA) is resolved and
  mirrored, not installed; it resolved hypre 3.1.0 where `psc_gpu` has 2.33.0, and is
  pinned to the GPU nodes' compiler until there is an RM allocation. Neither has
  `py-matplotlib`.

**Just landed (`benchmark-infra`, merged to `main` and pushed 2026-10-10): the benchmark
infrastructure** (human
2026-10-10: "do not run any of the actual test problems, I will do that at a later time, I
just want the test infrastructure set up"). Re_tau pair 180/395, constant pressure
gradient and `benchmarks/<case>/` were the agent's choices (the human didn't pick).
- Deck `forcing.body_force`; initial conditions `taylor_green_3d` and `channel` (Reichardt
  + curl perturbation, exactly divergence-free, zero with its normal derivative at the
  walls); `mesh.geometry: file` (`mesh.file`, Gmsh `$PhysicalNames` -> boundary names,
  `mesh.boundary_names`); the DFG channel in 3D (`Extrude2D`, curved nodes kept, z faces
  join `walls`; product parabolic inflow); `output.diagnostics` without field output,
  appended on restart; `reference.re_tau`.
- `post/channel_statistics`: plane and time averages at every velocity-node height
  (U, V, W, u'u', v'v', w'w', u'v'), tau_w, u_tau, Re_tau, bulk velocity; Save/Load exact,
  carried by Case's checkpoints; summary section `channel`, `<name>_profiles.csv`.
- `benchmarks/`: tgv_re1600 (n16/32/64), channel (Re_tau 180/395), dfg_3d (3D-1Z/3Z),
  sphere_re300 (README: needs a hex mesh). Reference data is NOT committed (human: the
  sources state no redistribution terms): `python3 benchmarks/fetch_references.py`
  downloads it from the original sources into git-ignored `*/reference/*.csv`;
  `plot.py --selftest` is in the fast tier and skips (exit 77) until it's fetched.
- Tests: `channel_statistics_test` S1-S5 (exact plane averages, wall shear, trapezoid,
  Save/Load, restart through a checkpoint == uninterrupted), `benchmark_setup_test`
  F1-F2, I1-I2, M1-M2, B1. None of the benchmarks has been run.
- Found while testing: `solver.atol` defaults to 0, so a step that starts from an exact
  steady state aborts ("implicit solve did not converge"): its residual cannot drop by
  rtol. Worth a small absolute floor by default (changes baselines: the human's call).

**Just landed (`oifs-mass`, merged to `main` and pushed 2026-10-10):** OIFS's substeps invert the BDF step's own mass
operator, so the two can no longer differ (human: "I literally want it to error out if
the two aren't consistent, using the diagonal for both of them as the default is fine").
Pointwise for a diagonal mass (collocated, conforming: the default, unchanged results),
Jacobi-CG per RK stage otherwise (consistent mass; collocated at hanging nodes, ~6
its/stage), always the FULL inverse (a constrained solve left 3.6e-4 vs IMEX). Setup
checks M^-1(Mz) = z and aborts otherwise. OIFS now matches IMEX with either mass (wall MMS
3.0e-6 both; was 1.6e-3 mismatched). AMR runs change slightly (exact P^T D P instead of
row-sum lumping). Deck option `discretization.mass: auto | collocated | consistent`
(replaces `collocated_mass`; auto = collocated under OIFS, consistent under IMEX).

**To evaluate later (human 2026-10-10: "we'll do a bunch of numerical experiments at a
later date"): collocated vs consistent mass in general**, for IMEX and OIFS, as the
default. Questions: accuracy per DOF on resolved benchmarks (TGV, channel, DFG), cost
(the consistent mass under OIFS adds a CG solve per substep stage: ~2x wall on the 6x6
cavity, ~6 its/stage at hanging nodes), and robustness. Data so far: on the Re 200 square
cylinder the collocated mass moves IMEX by C_D -0.4% / C_L,rms -1.8% (a resolution
effect: both should converge to the same answer); on the 6x6 cavity, KE differs 1.1%.
Benchmark plan (3D TGV, channel Re_tau 180/395, sphere, DFG 3D,
Ethier-Steinman) proposed; waiting on the human's answers (Re_tau pair, forcing,
directory layout, GPU-hours).

**Just landed (branch `license`, merged to `main` and pushed 2026-10-09, human: "merge and
push"):** BSD-3-Clause `LICENSE` (copyright Michael
Schneier; human: "it should be me"). README rewritten so the agent is the way in: what to
ask, what the agent asks first, what comes back. Human: the test cases are not the
standard CFD benchmarks to showcase (literature table removed; `cases/` decks are now
"example decks"), and the "production DNS/LES on DOE systems" line removed. AGENTS.md
gains "Working with the user" and rule 9 (licensing); user runs go under `runs/`.
Spack for anyone's machine (human: "someone points their agent at the repo ... it can do
everything from installing with spack (with users permissions of course)"):
`environments/stack.yaml` (portable stack, + py-matplotlib), `scripts/setup.sh` (plan,
then `--yes`; builds gcc@14.3.0 and declares it external), `scripts/machine.sh` (no default
machine any more; this checkout has `environments/.machine` = desktop),
`docs/install/spack.md` (how it all fits, for agents). Not yet run: `setup.sh --yes` on a
fresh machine. Next: that test (host terminal, clean VM or container); the TGV example
(deferred by the human).

**Just landed (branch `agent-first`, merged to `main` and pushed 2026-10-09, human:
"commit merge and push"):**
- Environment refresh: MFEM 4.10.1-dev with MUMPS, hypre 3.2, libCEED 1.0. MFEM pinned
  to `1c19aba` because of an upstream regression (PR #5297, reported 2026-10-09).
- Agent-first front door: `AGENTS.md`, short README, `docs/using.md`,
  `docs/install/desktop.md`, CLAUDE.md split into `developing.md` + this file.
- Generated deck reference (`docs/deck_reference.md`); unknown deck keys are errors.
- `scripts/doctor.sh`: setup check; `--run` adds the build-if-missing and smoke run (seconds).
- Benchmarks as decks: `mesh.geometry` (square cylinder, DFG channel), deck velocity
  BCs (constant / parabolic, ramp / sine), boundary coverage check, `uniform` initial
  flow, time-based AMR, force statistics, progress / history, pressure probe,
  reference values, and `summary.json` from every `run_case` run. Decks:
  `cases/square_cylinder_re200[_oifs].yaml`, `dfg_2d1.yaml`, `dfg_2d3.yaml`,
  `cavity.yaml`. The OIFS deck reproduced the validated result (C_D 1.4275, C_L,rms
  0.3940, St 0.1568).

**Validated:** `dfg_2d1.yaml` (relative errors C_D 1.05e-4, C_L 2.79e-3, dp 4.06e-3 = the
slow-tier values, 70 s), `dfg_2d3.yaml` (dt 0.001; C_D,max 0.08%, 172 s),
`square_cylinder_re200_oifs.yaml` (C_D 1.4265, C_L,rms 0.3923, St 0.1565 in 2,191 steps,
13 min: the driver's run to every digit). **IMEX deck run:** `square_cylinder_re200.yaml` validated 2026-10-09: C_D 1.4436,
C_L,rms 0.4034, St 0.1568 in 11,130 steps, 42 min at 4 ranks. (An earlier run that seemed
to stop at t = 90 had been killed by the agent itself, to restart it on the corrected deck;
the restart never launched.) Against the OIFS deck: C_D 1.2% and C_L,rms 2.8% higher, St
equal; the "within 1%" figure is IMEX with OIFS's collocated mass, not this deck.
Debug-device sweep: green, 114/114. The full sweep after the merge failed only
`run_monitor_test` R3 (AMR on a periodic Taylor-Green box), the known MFEM periodic-NC
debug-device false positive; R3 now skips there like `amr_flow_test.cpp:88`. Bridges-2: `psc_gpu` was
re-resolved on the cluster on 2026-10-10 (MUMPS and the pin).

**Next, in order:**
1. A cold-start test: a fresh agent with no context, given real tasks (install check,
   reproduce a benchmark from its deck, set up a new case); fix where it stumbles.
2. When MFEM fixes PR #5297's regression: drop the pin in both `spack.yaml` files.

**Open decisions for the human:** OIFS CFL 2 as the default for production runs.

**DFG 2D-3's 2026-10-09 "shift": resolved 2026-10-23 (human: "look into this").** Not
the stack and not a code change: a different grad-div. The 10-07 study ran with grad-div
gamma = 1 h_K (`apps/dfg_cylinder -gd 1`); the deck of 10-09 (`2394084`) set none, and
`physics.grad_div` defaulted to 0 then. Both reproduce exactly on today's CPU stack (MFEM
4.10 + PR 5464, hypre 3.1), the old commits built there: the 10-09 deck as it was gives
C_D,max 2.94857870 at t 3.93632 and diverges at dt 0.0015 at t = 3.9105; the same deck
with `physics.grad_div: 1` gives the study's 2.94451202, stable at dt 0.0015 (2.94451768);
the study's code (`55509f1`) and `530f93d` give 2.94451202 too (jobs 49658244, 49659444).
Grad-div off is closer to John's drag peak on this coarse mesh (0.08% against 0.22%) but
lowers the explicit convection's stability edge below dt 0.0015. Today's defaults (OIFS,
BDF3, collocated mass, gamma = (6 / k_u) h_K) give C_D,max 2.94319 (0.26%), C_L,max 9.2%,
dp(8) 3.4%, stable at 0.0015 (job 49657642); the deck's comment now says so.

## Log: state and decisions (last full update 2026-10-08)

**Branches.** `main` includes `rotational-schur` (merged and pushed 2026-10-07:
rotation-aware Schur PC, DFG 2D-3 driver, `jacobi_pcg` default + grad-div in LOR-AMG, step
control `fixed|error|cfl` with Nek's CFL and the BDF3/EXT3 estimator, case-level CFL control
at 0.5, the debug-device ess-list fix developing.md §6) and **`dfg-scheme-comparison`** (fast-forward
merged and pushed 2026-10-08, human: "merge and push everything"). New work goes on a new
branch; merge/push only when the human says so. dfg-scheme-comparison: DFG app `-mref` nested meshes, per-step
history CSV and `step_wall`; `bench/dfg3/` (pinned two-slot queue runner, sweep generator,
analysis, HTML report generator, raw results of 2026-10-08); the work-precision write-up
(`docs/imex_vs_semi_implicit.md` §7 + summary at its top); **case-level default back to
BDF2/EXT2** (human 2026-10-08; developing.md §5.2). The charts were published as a private claude.ai
artifact (regenerate with `bench/dfg3/report.py OUT.html
bench/dfg3/results_2026-10-08/findings.html bench/dfg3/results_2026-10-08/sweep.txt`).

**Scheme comparison concluded (human 2026-10-08): for DFG 2D-3 there is no benefit to the
semi-implicit rotational form** — same CFL limit as IMEX, 7–9% more per step, comparable
lift, 50–70× worse drag; both fail silently past the limit (developing.md §5.6, the doc §7). The finest
mesh (M2) was started and stopped unfinished at the human's call.

**`directional-do-nothing`** (fast-forward merged and pushed 2026-10-08, human): Braack &
Mucha's directional do-nothing outflow condition for the IMEX convective form, default on,
explicit (on the RHS with the convection; no operator or preconditioner change — a
semi-implicit LHS version is the fallback if outlet backflow ever destabilizes below the CFL
limit) (developing.md §4 BCs; `docs/outflow_conditions.md`); `directional_do_nothing_test`; DFG app
`-cdn` for the classical condition.

**Square cylinder (Joly, Etienne & Pelletier, J. Fluids Struct. 28 (2012) 232–243;
human 2026-10-08) — branch `square-cylinder`, fast-forward merged and pushed 2026-10-08
(human: "commit merge and push this stuff")**: `src/mesh/square_cylinder.{hpp,cpp}`,
`apps/square_cylinder.cpp`,
`ConvectiveCfl::RateAndLocation` + `Case::ConvectiveCflNumber(Vector* where)` (where the
CFL rate peaks), `cylinder_mesh_test` M3. Details developing.md §5.6. The paper PDF `big_domain.pdf` sits
untracked in the repo root (do not commit it; copyright).
- **Re = 200, α = 0 — DONE, all within the human's 5%** (IMEX, Q3/Q2, CFL 0.5, EXT2, AMR):
  C_D 1.4434 (paper 1.44, +0.2%), mean C_L 0.0003 (0.001), C_L,rms 0.4034 (0.42, −4.0%),
  St 0.1568 (0.151, +3.8%), 10 periods, period spread 0.03%. Fine AMR (tol 0.12, 5,003
  cells vs 3,001): C_D 1.4425, C_L −0.0003, C_L,rms 0.4025, St 0.1569 (11,518 steps, dt
  5.3e-3–2.5e-2) — every coefficient within 0.2% of the base run → wake-converged; corner resolution (smallest cell 0.125 in
  both) and dt were NOT varied. dt 6.2e-3–2.5e-2 (mean 1.4e-2, ~440 steps/period vs the
  paper's ~32 at UΔt/D = 0.2); 11,138 steps, 69 min at np 4. Remaining −4% / +3.8% may be the
  paper's large implicit step (untested). Literature scatter (their Table 1): C_L,rms
  0.32–0.55, St 0.142–0.170.
- **dt sawtooth on AMR meshes (open):** once shedding starts, every ~0.13 time units the CFL
  rate jumps ~1.5× in ONE step → the controller cuts dt ~1.9e-2 → ~1.0e-2 and regrows ×1.2.
  Forces unaffected (C_D jitter ~1e-4); ~10–20% extra steps. Likely a fast mode at its
  stability limit at CFL 0.5 on the AMR (hanging-node) mesh — conforming DFG meshes were
  stable to 0.7. Diagnose with the CFL location now printed; a 0.4 target likely removes it.
- **AMR stays refinement-only (human 2026-10-08):** derefinement breaks the multistep
  history (refinement transfers every BDF/EXT level exactly; coarsening must project them),
  and single-step schemes are order-bound. For a statistically stationary turbulent wake,
  refine-only converges to resolving the wake region anyway; bound the startup waste with
  `-amr-start` (spin up on the coarse mesh — spin-up only has to reach the state, not
  accurately) and `-amr-end`; a remesh-and-restart (multistep startup ramp) is the fallback.

**`oifs` (branch; human 2026-10-08 "give a shot at implementing the OIFS stuff", 2026-10-09
"you should be running everything with BDF3 with the OIFS stuff"; fast-forward merged and
pushed 2026-10-09, human: "merge and push"):** OIFS (developing.md §5.2), BDF3 by default under OIFS, collocated GLL mass always under
OIFS. Square cylinder Re 200 (same AMR as the base IMEX run, np 4; vs IMEX CFL 0.5 with
the SAME collocated mass: C_D 1.4381, C_L,rms 0.3957, St 0.1564, 9,103 steps):
```
OIFS BDF3, collocated   C_D            C_L,rms         St             steps
  CFL 1                 1.4346 -0.2%   0.3931 -0.7%    0.1565 +0.1%   4,379
  CFL 2                 1.4265 -0.8%   0.3923 -0.9%    0.1565 +0.1%   2,191
  CFL 4                 1.4772 +2.7%   0.4778 +21%     0.1507 -3.6%   1,120
```
**CFL 2 is the sweet spot: within 1% of IMEX in 4.2× fewer steps (~3× less wall time;
an OIFS step costs ~1.3–1.5× an IMEX step).** CFL 4 loses the lift amplitude. History of
the study (each step measured): BDF2 + consistent BDF mass converged in dt to the WRONG
answer (C_D 1.49, C_L,rms 0.45: +3% / +12% at CFL 4 → 0.5) — the mass mismatch (developing.md §5.2),
not the wall treatment (pointwise wall acceleration: no change, reverted), the force
evaluation (wall MMS: = IMEX to 1.5e-3), hanging nodes (`oifs_test` O3) or the outflow
(IMEX classical = directional BITWISE on this domain: no outlet backflow). BDF2's
trajectory time error at CFL 4 was ~4× BDF3's (C_D). OIFS also has no dt sawtooth (steady
dt at every CFL). The collocated mass itself moves IMEX by −0.4% C_D / −1.8% C_L,rms.
DDN under OIFS done (explicit, extrapolated in the BDF step: Braack & Mucha's Table 5.1
to IMEX's values at CFL 2, stable to CFL 8). **Error control under OIFS: not pursued
(human 2026-10-09: "we are always going to need to do some multiple of CFL basically,
so maybe we just skip the error estimator")** — rejected at validation. Open: OIFS cost
per step (substep CFL 0.5 — try 1.0); the fixed-step BDF3-OIFS startup (O(dt²) starter);
whether the case-level default for HPC runs should become OIFS CFL 2.

**Environment refresh (2026-10-09, human):** full desktop refresh with current spack:
MFEM 4.10.1-dev **+MUMPS**, hypre 3.2.0, libCEED 1.0.0, SuiteSparse 7.14 (developing.md
§3). MFEM's head broke 3D parallel anisotropic AMR (PR #5297; reported upstream), so MFEM
is **pinned to `1c19aba`** in both `spack.yaml` files until it is fixed. On the pinned
stack: fast tier 159/159, Python 12/12. The Bridges-2 lock (`psc_gpu`) was re-resolved on that
machine on 2026-10-10 with the same MUMPS + pin edit.

**`agent-first` (branch, 2026-10-09, human: "an agent first repo where someone can just
point their agent at it"):** `AGENTS.md` front door, short README, `docs/using.md`,
`docs/install/desktop.md`, this file and `developing.md` split out of CLAUDE.md (which now
imports them), specs moved to `docs/design/`. Next on that track: `scripts/doctor.sh`,
validated decks for the benchmarks (needs the drivers' AMR scheduling / averaging in the
library), `summary.json` per run, a generated deck reference, a cold-start agent test.

**Open follow-ups** (none started):
- Outflow condition for the rotational form: do-nothing acts on the Bernoulli head
  ("notoriously bad" for this form, human); the DFG studies used a Dirichlet outflow instead.
- A velocity PC whose iteration count does not grow as σ = β₀/Δt shrinks (Jacobi-PCG's
  does, so CFL 0.5 → 0.7 saves 29% of steps but only ~10–15% of wall time) — e.g. the
  p-multigrid velocity block (spec Part D).
- Warn when the measured CFL creeps above `cfl_target` (the IMEX/EXT3 failure symptom on
  DFG); the rotational form's silent failure shows no such signal.
- If the rotational form is pursued: find the source of its DFG drag error (the form's
  spatial error vs the force evaluation from the Bernoulli head).
- `auto` Schur: compute μ every N steps (it costs ≤1% per step now; low priority).
- Fast tier ~177–197 s, near the 3-min budget; critical path `tgv_nse_test_np2`.
- LOR-AMG with strong grad-div in 3D is still 4–5× its no-grad-div count; BoomerAMG's
  elasticity mode is the standard fix but needs `Ordering::byVDIM` (velocity is byNODES).
- Rotation-aware Schur `auto` thresholds (spec's 20/10) uncalibrated in 3D; default `cc`.
- p-multigrid velocity block (spec Part D Level 1) not built (human: PBJ only for now).
- Skew-symmetric convective form: gated on upstream MFEM (developing.md §5.4 TRAP).
- Not pursued (human 2026-10-07: "not interested in implicit methods at the moment"): a
  CFL-free option (Oseen linearization (w*·∇)u^{n+1}, or converged Newton/Picard steps).
- PENDING GPU VALIDATION list (developing.md §7.5).
- Upstream candidates: the MFEM debug-device BlockVector false positive (developing.md §6).
- 2D NSE TGV pressure-rate oracle never built (human accepted the 3D MMS, 2026-07-24).
- `VectorRotationalConvectionComponentIntegrator::AssembleEA` is untested.
- Convective-form TGV at ν = 0.05 is pre-asymptotic on coarse Q3 meshes (4×4 → 8×8 raises
  the error 0.012 → 0.024 before 16×16 drops it to 0.0027); oracles run at ν = 1.

**Design specs** in `docs/design/` (authoritative design records; developing.md wins where they
disagree): `docs/design/SPEC_cahouet_chabard_mfem.md` (+ `docs/precond_cc.md`),
`docs/design/rotational_convection_pa_spec.md`, `docs/design/rotational_schur_velocity_mg_spec.md`,
`docs/design/amr_spec.md`, `docs/design/vecdivdiv_spec.md`, `docs/install/bridges2.md`.
