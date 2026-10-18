# bessemer on many GPUs: the plan (2026-10-17)

Goal: take the single-H100 work (100M+ DOFs per 80 GB card for most solver
configurations, after changes 28–35 land) to many GPUs. Targets:

1. **Strong scaling**: ≥ 80 % parallel efficiency from 1 to 8 H100s on one
   Bridges-2 node at ≥ 20M DOFs per GPU.
2. **Weak scaling**: ~80–100M DOFs per GPU, ≥ 85 % efficiency to 8 GPUs and then to
   2–4 nodes. That is ~1B DOFs on one node and a few billion on a handful of nodes.
3. **No regressions**: run-to-run bitwise results at every rank count, and
   restarts across rank counts.

## Where it stands (measured 2026-10-17, change-32 code, profile: sync)

Two H100s on one node, np 2 (one rank per GPU), the same problem as np 1:

| Case (Q6) | DOFs | np 1 s/step | np 2 s/step | Efficiency |
|---|---|---|---|---|
| TGV OIFS coupled | 21.6M | 0.511 | 0.363 | 70 % |
| TGV OIFS Schur | 21.6M | 0.616 | 0.395 | 78 % |
| TGV IMEX coupled | 21.6M | 0.308 | 0.225 | 69 % |
| TGV IMEX Schur | 21.6M | 0.341 | 0.216 | 79 % |
| Curved cylinder OIFS coupled | ~22M | 1.118 | 0.730 | 77 % |
| Curved cylinder OIFS Schur | ~22M | 1.209 | 0.735 | 82 % |
| Curved cylinder IMEX coupled | ~22M | 0.832 | 0.582 | 72 % |
| DFG Q3 AMR, channel (small) | ~0.3–1M | 0.03 | slower | latency-bound |

What scales worst:

- **The pressure LOR-AMG.** Per V-cycle it is 2x *slower* at np 2 on Taylor–Green
  and 4x slower on the cylinder, with half the work per GPU. It goes from 4 % of
  the step to 11–22 %.
- **Transport** (fixed by change 37, see Phase 1). Open MPI 5.0.10 and UCX 1.20
  are built with CUDA, but hypre 2.33 is built with `gpu-aware-mpi: False`, and
  bessemer never told MFEM that MPI is GPU-aware. Every halo exchange in hypre
  (each AMG level) and in MFEM's parallel prolongation (each operator apply)
  therefore went device → host → MPI → device.
- **Small problems** (< ~2M DOFs per GPU) get slower: launches and latency
  dominate.
- **Memory per rank at np > 1.** MFEM's device prolongation is built from R,
  so P and R exist per rank (~17 B/DOF). The RAP wrappers' vector pairs (~1.9 GiB
  at 57.3M) only appear when the prolongation is not the identity. That is
  exactly np > 1, so the tabled work-vector pool matters here.

## Phase 0: instrumentation (before changing anything)

- **Communication regions** in the profiler: time in MPI waits for MFEM's
  prolongation and restriction exchanges, hypre's AMG cycles, and global
  reductions (Krylov dot products, norms, projection). Keep it per rank:
  report the max and the min over ranks (the CSV already reduces both).
- **Load balance**: elements, DOFs and shared DOFs per rank; the max/mean
  imbalance printed at setup.
- **Nsight Systems** traces at np 2 and 4 (one per rank): kernel/communication
  overlap, idle gaps, host staging copies.
- **Memory per rank** with the gauge (profile: sync) at np 1, 2 and 4 on the
  same problem: what is duplicated or grows with np (P, R, halo buffers,
  face-neighbour data, LOR replicas, AMG on every rank).

Deliverable: a per-region "where the np-2/np-4 time goes" table, like the
memory ledger.

**First measurements (2026-10-18, job 49404241; changes through 41, np 2 on
two H100s, Nsight Systems with CUDA, NVTX and MPI tracing):**

- **Taylor–Green Q6 OIFS, 21.6M DOFs.** The GPUs are busy 81 % of the step
  (242 of 298 ms of kernels per GPU) and the work is balanced (241.7 / 240.9
  ms). Rank 0's MPI waits (~180 ms per step, against ~20 on rank 1) are its
  host running ahead of queued GPU work, mostly hidden. The loss against np 1
  is extra GPU work: 398 ms of kernels per step at np 1, 483 ms summed over the
  two GPUs at np 2 (+21 %):
  - the AMG's SpMVs (cuSPARSE csrmv) 18 -> 52 ms, launches 154 -> 324 per
    step per GPU (diagonal and off-diagonal blocks on every level);
  - MFEM's prolongation exchanges' gather and scatter (ExtractSubArray,
    SetSubArray) +22 ms, kernels that do not exist at one rank;
  - the assembled diagonal +7 ms.
  Halo overlap (Phase 4) would buy little here; the AMG at np > 1 (Phase 2)
  and the RAP wrappers' extra vector passes are where the time is.
- **The curved cylinder OIFS, ~22M DOFs.** The GPUs are busy only ~70 % of
  the step (486 / 502 ms of 716): rank 0 waits ~430 ms per step on rank 1's
  serial host work (`oifs::classify`'s host loop, 43 ms per step; the outflow
  term's face loop on the rank that owns the outflow). Moving the
  classification onto the device and balancing the outflow faces come before
  any transport work for walled cases.

## Phase 1: transport (cheap, no trade-offs)

1. **GPU-aware MPI: done, on by default (change 37).** MFEM
   `Device::SetGPUAwareMPI(true)` plus hypre `HYPRE_SetGpuAwareMPI(1)`, set at
   startup whenever the MPI library takes CUDA device pointers (Open MPI's
   `MPIX_Query_cuda_support`); `INCNS_GPU_AWARE_MPI=0` or `1` overrides the
   detection.
   - Measured at np 2 on two H100s (job 49397771, TGV Q6 32³ and the curved
     cylinder, 21.6M DOFs, off against on):

     | Case | s/step off | on | Change | Pressure AMG (s, whole run) |
     |---|---|---|---|---|
     | TGV OIFS coupled | 0.371 | 0.297 | −20 % | 1.69 → 1.27 |
     | TGV IMEX coupled | 0.225 | 0.182 | −19 % | 1.35 → 1.10 |
     | Curved cylinder OIFS | 0.743 | 0.719 | −3 % | 3.52 → 3.31 |

   - Memory: 0.1–0.4 GiB more per GPU at ~11M DOFs per GPU, 2.5 GB per GPU
     at ~68M (Taylor–Green IMEX Schur, 136.6M DOFs on two H100s: 39.46 ->
     41.92 GB on the larger rank; job 49400578). All of it is UCX's cuda_ipc
     transport (job 49401722): MFEM's GPU-aware mode alone adds nothing, and
     with `UCX_TLS=^cuda_ipc` the GPU-aware run takes 39.46 GB at the same
     step time (1.470 s; host-memory rendezvous fragments change nothing). The
     human made GPU-aware MPI the default in both the throughput and the
     memory profile (2026-10-17); whether to drop cuda_ipc (job scripts or
     scripts/env.sh) is theirs to decide -- at 4-8 GPUs on a node the NVLink
     path may matter more than at 2.
   - Results are not bitwise equal off against on, nor run to run on the GPU
     (library-level nondeterminism in hypre/cuSPARSE, known and accepted since
     2026-10-15). On the CPU build the switch does nothing.
   - **No hypre rebuild is needed.** Checked 2026-10-17 against the hypre
     2.33.0 source:
     - the configure option `--enable-gpu-aware-mpi`
       (`HYPRE_USING_GPU_AWARE_MPI`) only sets the default of the handle's
       `use_gpu_aware_mpi` flag (`utilities/general.c`);
     - every exchange reads the flag at run time through
       `hypre_GetGpuAwareMPI()`: `par_csr_communication.c` (halo exchanges),
       `par_csr_matvec_device.c` (the AMG cycle's matvecs),
       `par_csr_triplemat_device.c`, `par_csr_matop_device.c`,
       `par_csr_fffc_device.c`;
     - so `HYPRE_SetGpuAwareMPI(1)` before the first hypre object is enough,
       given an MPI that takes device pointers.
   - **The spack side.** `environments/psc_gpu/spack.yaml` already builds the
     MPI CUDA-aware (`openmpi+cuda fabrics=ucx` on `ucx+cuda`; the installed
     `mpi-ext.h` has `MPIX_CUDA_AWARE_SUPPORT 1`). hypre's spack variant
     `+gpu-aware-mpi` (default off) only flips the default of the runtime flag
     change 37 sets anyway; MFEM has no build option (runtime only:
     `Device::SetGPUAwareMPI` or `MFEM_GPU_AWARE_MPI`). Adding
     `^hypre+cuda+gpu-aware-mpi` is harmless at the next environment rebuild;
     not worth a rebuild by itself.
2. **Intra-node path**: confirm UCX uses `cuda_ipc` between GPUs on a node
   (NVLink) and `rc`/`dc` with GPUDirect RDMA across nodes. Tune `UCX_TLS` and
   the rendezvous thresholds only if traces show staging.
   - **OSU micro-benchmarks with device buffers** (`osu_latency -d cuda D D`,
     `osu_bw`, `osu_allreduce`) on 2 GPUs of one node and on 2 nodes: the
     transport's latency and bandwidth by message size, against which the AMG
     coarse levels (small messages) and the operator halos (large) can be
     judged.
   - **GDRCopy**: UCX is built `~gdrcopy`. GDRCopy cuts small-message latency
     from device buffers (the AMG coarse levels). The GPU nodes have the
     `gdrdrv` module and `nvidia_peermem` (GPUDirect RDMA) loaded (job
     49400578, w008), so a UCX `+gdrcopy` rebuild (and Open MPI and everything
     above it) is possible: the human's call, best after the multi-node traces.
   - **Topology** (w008): the two GPUs of a GPU-shared allocation are joined by
     NVLink (NV18); each sits on its own PCIe switch with an InfiniBand HCA
     (GPU0 with NIC4, GPU1 with NIC6 in `nvidia-smi topo -m`'s numbering; its
     legend maps them to mlx5 devices for `UCX_NET_DEVICES`), CPU cores
     75, 79-89, 96-103 on NUMA node 1.
3. **Binding**: one rank per GPU (already enforced: `ResolveDeviceId`), each
   rank's CPU cores and memory on the GPU's NUMA domain (`--cpu-bind`, PRTE
   mapping), so the host-side work (setup, the residual host loops) doesn't
   cross sockets. Across nodes, set `UCX_NET_DEVICES` to the HCA nearest each
   GPU (`nvidia-smi topo -m`) rather than letting UCX pick.

Measured: the step 19–20 % shorter at np 2 on Taylor–Green, mostly in the AMG
(above).

## Phase 2: the pressure solve at scale

The LOR-AMG coarse levels become latency-bound as ranks grow (few rows per rank,
many messages per level). AMG settings are the human's call, so these are
options to measure and bring back, not to switch on:

- **Coarse-grid agglomeration / redistribution**: gather the coarsest levels
  onto fewer ranks (hypre's `agg_num_levels`, or a redundant coarse solve).
- **Fewer levels plus a direct or Chebyshev coarse solve.**
- **Aggressive coarsening on the first level** (already on the open list for
  memory: 2.1 GB at Q3 64³). Measure iterations and time per step at 1, 2, 4
  and 8 GPUs before proposing it.
- **Keep the transpose** (`HYPRE_BoomerAMGSetKeepTranspose(1)`): hypre's GPU
  guidance; the restriction is stored explicitly instead of applied as the
  transposed interpolation, a faster matvec for more memory. A candidate for the
  throughput profile only (`performance.priority: throughput`).
- **Already at MFEM's GPU BoomerAMG defaults**: PMIS coarsening, ext+i
  interpolation with at most 4 entries per row, l1-Jacobi smoothing (relax type
  18), θ 0.25 in 3D, no aggressive levels, hypre's own SpGEMM.
- **Smoother**: Chebyshev or l1-Jacobi instead of Gauss–Seidel-type smoothers
  (no sequential sweeps, fewer reductions).
- **Longer term**: a p-multigrid pressure preconditioner (Schwarz- or
  Chebyshev-smoothed on the high-order levels, AMG only on the Q1 level). Far
  less communication per cycle. A real project; decide after Phase 1's numbers.

## Phase 3: global reductions

Every Krylov iteration pays several `MPI_Allreduce`s:
- FGMRES orthogonalization: modified Gram–Schmidt, one reduction per basis
  vector;
- the solution projection;
- the velocity CG;
- the pressure FCG.

At 8+ GPUs these latencies add up.

- **Batched dot products**: classical Gram–Schmidt with
  reorthogonalization (CGS2), giving one fused reduction per iteration instead
  of k. The same for the projection's coefficients.
- **Fuse norm and dot products** where several are computed back to back (CG's
  (r, z) and ||r||).
- **Pipelined or s-step variants**: only if the traces show reductions still
  dominate after batching.
- **Bitwise constraint**: a different summation order changes results at
  roundoff once. Any reduction change must stay deterministic for a fixed rank
  count (no atomics, a fixed tree).

## Phase 4: overlap of halo exchange and computation

MFEM's parallel prolongation exchanges shared DOFs before the element kernels
run. With interior and boundary elements split (boundary = touches a shared
DOF), the interior kernels can run while the halo travels. This applies to the
momentum apply, divergence/gradient, OIFS convection and the mass. It needs our
own split restriction or a wrapper around MFEM's (no protected-member hacks).
Worth it only if Phase 0 shows exposed exchange time after Phase 1.

## Phase 5: partitioning and load balance

- **Weights**: METIS by element count today. Curved elements (Jacobians on the
  fly per chunk) and AMR-refined regions cost more, so weight by measured kernel
  cost.
- **AMR**: rebalance after each adaptation (`ParMesh::Rebalance`), with the
  history projection following the new partition. Check that restarts and
  checkpoints survive it.
- **Shared-DOF surface**: report it per rank. Box partitions of TGV and channel
  meshes should prefer slabs or cubes with small surfaces.

## Phase 6: memory per GPU at np > 1

- **The work-vector pool** (tabled 2026-10-17 for one GPU): at np > 1 the RAP
  wrappers' pairs (~1.9 GiB at 57.3M), the OIFS mass CG and the Chebyshev mass
  inverse all hold full-size vectors that are never live at once. Bring it back
  to the human with the np-4 memory numbers.
- **P and R per rank**: required by MFEM's device prolongation. Check their
  size per rank and whether R alone would do (MFEM copies R's column indices
  and keeps P).
- **Capacity per GPU at np 2, 4 and 8**: the same probe method as the
  single-GPU capacity table, so the weak-scaling size per GPU is measured, not
  assumed.

## Phase 7: scaling studies

- **Strong scaling**: TGV Re 1600 Q6 at ~86M DOFs (fits one GPU with OIFS
  Schur or IMEX) on 1, 2, 4 and 8 GPUs; the curved cylinder at ~70M; the
  channel at Re_τ 550 (the deck exists).
- **Weak scaling**: ~80M DOFs per GPU, the element count scaled by 2^(1/3) per
  doubling, on 1, 2, 4 and 8 GPUs, then 16 and 32 (2–4 nodes).
- **Metrics**: s/step, DOF-steps/s/GPU, efficiency, peak memory per GPU,
  iterations per step (to separate solver and algorithmic scaling from
  implementation scaling), time to solution for a fixed physical time.
- **Each solver configuration**: OIFS and IMEX, coupled and Schur (Schur with
  change 34's pressure target).

## Correctness at scale

- **Bitwise**: every run repeated twice at the same np gives identical output
  (GPU-aware MPI on and off, every phase's change).
- **Across np**: results agree to solver tolerance between np 1, 2, 4 and 8
  (not bitwise: partition-dependent sums).
- **Restart across np**: a checkpoint written at np 4 restarts at np 8 and
  continues within tolerance.
- **Tests**: GPU tests stay at np 1 and 2 (standing rule). Scaling runs are
  studies, not tests.

## Human decisions needed

1. **GPU budget**: decided (human 2026-10-17). 4–8 H100s are fine for scaling
   studies; the 2-GPU limit stays for everything else. Multi-node (16–32) is to
   be asked when it comes up.
2. **Rebuilding hypre**: not needed (hypre's GPU-aware MPI is a runtime switch;
   see Phase 1). GPU-aware MPI is on by default in both profiles (decided
   2026-10-17).
3. **Pressure AMG changes** (Phase 2).
4. **The work-vector pool** for np > 1 (Phase 6).
5. **Whether to start the p-multigrid pressure preconditioner** as a project.

## Order and rough effort

| Step | Effort | Needs |
|---|---|---|
| Phase 0 instrumentation + np-2/np-4 ledger | 1–2 days | 4 GPUs (approved) |
| Phase 1 GPU-aware MPI (done, change 37), binding, UCX/OSU checks | 1 day | — |
| Phase 7 first pass: strong + weak to 8 GPUs | 1 day of runs | 8 GPUs (approved) |
| Phase 3 batched reductions | 2–3 days | — |
| Phase 2 AMG options measured | 2–3 days | your call on settings |
| Phase 5 weights + AMR rebalance | 2–4 days | — |
| Phase 6 pool + per-GPU capacity at np > 1 | 2–3 days | your call on the pool |
| Phase 4 overlap | ~1 week | only if Phase 0 shows exposed exchange |
| Multi-node runs (2–4 nodes) | after the above | allocation |
