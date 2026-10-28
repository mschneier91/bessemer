# NCSA Delta

How bessemer is set up on [Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
(environment `environments/delta`, first done 2026-10-27). The general procedure is
[spack.md](spack.md); this page is what is particular to Delta.

## The machine

- **Nodes used:** `gpuA100x4`: one AMD EPYC 7763 (64 cores), 256 GB, four A100-SXM4 **40 GB**
  with NVLink. Later `gpuH200x8`: eight H200 141 GB per node, charged 3x an A100-hour.
  The `gpuA100x8` nodes charge 1.5x.
- **OS:** RHEL 9.6 on the login and compute nodes. **Network:** HPE Slingshot 11 (`cxi`
  NICs), so the MPI is the site's Cray MPICH; Open MPI over UCX does not run on it.
- **Compute nodes have internet:** Spack fetches its sources inside the job; no mirror.
- **Login:** password and Duo on every new connection, no SSH keys. A shared connection
  (`ControlMaster`/`ControlPersist` in `~/.ssh/config`) lets later `ssh` commands reuse one
  login.
- **Storage:** `$HOME` (100 GB) holds Spack (`~/spack`, the `env.sh` default);
  the checkout and runs live in the project space, `/projects/<project>/<user>/` (500 GB per
  project).

## The environment

`environments/delta/spack.yaml` is `psc_gpu` with Delta's externals:

| | |
|---|---|
| compiler | the site's GCC 14.2.1 (module `gcc-native/14`, Red Hat's `gcc-toolset-14`) |
| MPI | `cray-mpich@9.1.0+wrappers+cuda`, the gnu build at `/opt/cray/pe/mpich/9.1.0/ofi/gnu/11.2`; `+wrappers` makes Spack's dependents compile through its `mpicc`/`mpicxx` |
| CUDA | 13.2, the standalone toolkit at `/opt/nvidia/cuda-13.2` (the same `nvcc` V13.2.78 as the HPC SDK's module `cudatoolkit/26.5_13.2`, loaded by default). Not the SDK's own prefix: it keeps cuBLAS and cuSPARSE in a separate `math_libs` tree, and libCEED, hypre and MFEM failed to link `-lcublas` against it. Cray MPICH's GPU transport library links `libcudart.so.13`, so the stack uses the same runtime |
| `cuda_arch` | 80 (A100). The H200 is 90: a second environment, otherwise the same |
| numba | from pip's wheel, as on Bridges-2 (Spack's `py-numba` builds LLVM) |
| MFEM | v4.10 with mfem/mfem#5464 from `/projects/bjdd/mhs13c/mfem-src/mfem-4.10` (`scripts/mfem_source.sh`) |

The default login and job environment already loads `PrgEnv-gnu`, `gcc-native/14`,
`cray-mpich/9.1.0` and `cudatoolkit/26.5_13.2`; nothing else needs loading.

**Two build details:**
- MFEM's `config.mk` asks for `-lcusparse -lcublas -lcurand -lcusolver` with no `-L`.
  Bridges-2's CUDA module resolved them through `LIBRARY_PATH`; Delta's leaves it empty,
  so the top-level CMakeLists puts the toolkit's library directory (`CUDAToolkit`) on
  every link line.
- bessemer is compiled with `CXXFLAGS=-march=x86-64-v3`, the target Spack built MFEM for,
  as in CI (developing.md §7.6). Without it `persistent_fgmres_test` differs from MFEM's
  FGMRES by 5e-11 against its 1e-14 limit (job 23250720). `test.sbatch` sets it and drops
  a CMake cache configured without it.

**The launcher is `srun`** (`environments/delta/mpiexec`): Cray MPICH has no `mpirun`.
CTest and the doctor run `srun --overlap -c 1 --cpu-bind=none -n <ranks>` inside a job.
Delta's default PMI plugin (PMIx) works; `--mpi=pmi2` starts N separate one-rank jobs.
`--overlap` lets concurrent test steps share the job's cores and GPU.

**GPU-aware MPI is on by default** (2026-10-27), as with CUDA-aware Open MPI on Bridges-2,
where it took 20 % off the step at np 2. Three pieces:
- CMake sees Cray MPICH as the compiler wrapper (`.../pe/mpich/<version>/ofi/...`) and
  links its GPU Transport Layer, `gtl/lib/libmpi_gtl_cuda.so`, into every target. Cray's
  own `cc`/`CC` wrappers do that; its plain `mpicxx` does not.
- `ConfigureMpiTransport` sets `MPICH_GPU_SUPPORT_ENABLED=1` before `MPI_Init` unless the
  job sets it. With the variable at 1 and no GTL linked, `MPI_Init` aborts.
- `ConfigureDevice` asks MPICH (`MPIX_GPU_query_support`) and turns MFEM's and hypre's
  GPU-aware exchanges on; the run prints `[incns] GPU-aware MPI: on`.
- `ConfigureMpiTransport` also makes each rank's own GPU current before `MPI_Init`. The
  GTL records the device current at `MPI_Init`, and a rank that moves to another GPU
  afterwards gets "This process is not using the same device it did during gtl_init ...
  Use of IPC is disabled" (every rank but the first on a node, job 23251132), and its
  transfers fall back from CUDA IPC.

**CUDA IPC stays on** (human 2026-10-27): Cray MPICH moves GPU buffers between the ranks on a
node by CUDA IPC unless the job sets `MPICH_GPU_IPC_ENABLED=0`, and bessemer leaves it alone.
Two A100s, Taylor-Green 32³ Q6/Q5 (21M velocity DOFs), 23 steps: the solves took 8.46 s with
IPC and 20.66 s without (job 23261041). Its device memory was not measured; on Bridges-2 UCX's
`cuda_ipc` held 1–2.5 GB per GPU near capacity ([gpu_memory.md](../gpu_memory.md), change 42),
so set `MPICH_GPU_IPC_ENABLED=0` in a job that runs out of memory at np > 1.

`MPICH_GPU_SUPPORT_ENABLED=0 INCNS_GPU_AWARE_MPI=0` in the job stages the exchanges through
the host instead. Each rank binds to the GPU `SLURM_LOCALID` names (rank on the node modulo
the GPUs it sees), so a production run gives every rank all the node's GPUs:

```sh
srun -n 4 -c 16 --cpu-bind=cores build/cuda/apps/run_case deck.yaml   # 4 ranks, 4 A100s
```

`environments/delta/gpu_aware.sbatch` compares the two on two A100s (Taylor-Green Q6, 21M
velocity DOFs) and runs the fast tier at np 2. Results: status.md.

## Install and test

From a login node, with the checkout at `/projects/<project>/<user>/bessemer`:

```sh
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ~/spack
scripts/mfem_source.sh /projects/<project>/<user>/mfem-src/mfem-4.10
SPACK_DISABLE_LOCAL_CONFIG=1 ~/spack/bin/spack -e environments/delta concretize
scripts/envlock.sh record environments/delta
cd environments/delta
sbatch install.sbatch                                   # the stack (asks first: AGENTS.md rule 3)
sbatch --dependency=afterok:<install job> test.sbatch   # build, smoke run, fast tier at np 1
sbatch --dependency=afterok:<test job> gpu_aware.sbatch # two A100s: GPU-aware check, fast tier at np 2
```

The first two take a quarter of a `gpuA100x4` node (1 A100, 16 cores, 60 GB), billed as one
A100-hour per hour used; `gpu_aware.sbatch` takes half (two A100-hours per hour). Edit `-A`
in them for another allocation. Submit them from `environments/delta`: they find the
checkout as `$SLURM_SUBMIT_DIR/../..`. `test.sbatch` passes extra arguments to `ctest`
(`sbatch test.sbatch -I 1,30` runs the first 30 tests).

On A100s `fsi_test` at np 2 takes longer than ctest's default 1500 s (2267 s alone; at np
1, 815 s): run it with `--timeout 3300`, apart from the rest of the fast tier, which fits an
hour's interactive job at np 1 or np 2.

## The queue

Delta's GPU partitions ran full on 2026-10-27 (every `gpuA100x4` and `gpuA40x4` node
allocated). The batch partitions' start estimates were days out (A100 a week, H200 three
weeks; the preempt partitions one to two days), and 4-GPU jobs waited a median of about 22 h
on A100s and 60 h on H200s over the past months (XDMoD). What moved:

- **The interactive partitions** (`gpuA100x4-interactive`, `gpuA40x4-interactive`): at
  most 1 h, one queued job per user per partition, started within minutes to a few hours.
  Enough for the build and the test tiers in pieces.
- **A40s** (`gpuA40x4*`): sm_86, 48 GB, half an A100's charge; the `cuda_arch=80` build runs
  on them unchanged. Their FP64 is a small fraction of an A100's: correctness checks, not
  timings.
- An idle job holds its place: check that a submitted job starts, and move it if its
  estimate (`squeue --start`) slips.
- **The interactive partitions bill twice:** `TRESBillingWeights` GPU=2000 against the batch
  partition's 1000, so an hour on two A100s there costs 4 SU, not 2.
- One job per user per interactive partition, running or pending: the next is refused
  (`QOSMaxSubmitJobPerUserLimit`) until the first ends. `srun --jobid=<id> --overlap`
  runs more steps inside a running allocation.

## What the A100's 40 GB changes

MFEM's runtime-size kernels keep per-thread arrays sized for 14 points per direction, and
CUDA reserves that local memory for every thread the card can hold. MFEM's 3D
`ConvectionIntegrator` above order 3 at OIFS's rule needs ~58 GB of it on a 40 GB A100
(~71 GB on an 80 GB H100, which fits), and its launch fails "out of memory" (job 23251030).
The solver does not use it, but `oifs_convection_test` compares against it: the test
registers MFEM's compile-time kernels for those sizes
(`AddConvectionReferenceKernels`). Any other path that lands in an MFEM runtime-size
kernel at a high order can hit the same wall on a 40 GB card.

## hypre 3.2, not Bridges-2's 2.33

Bridges-2's GPU stack, on which the GPU results were validated, had hypre 2.33.0. Delta's
cannot: Spack's recipe allows no hypre 2.x with CUDA 13 and none below 3.2.0 with CUDA 13.2,
and Cray MPICH 9.1.0's GPU transport links the CUDA 13 runtime (CUDA 12.9 is installed, but
without GPU-aware MPI). hypre 3's CUDA builds also require Umpire. Four consequences:

- **hypre runs cuSPARSE's SpMV kernels here** (`ConfigureHypreSpMV`: hypre >= 3; hypre 2
  keeps its own, change 43). With hypre's own kernels, coarsening at np 2
  (`amr_transfer_test` X4) gave one mesh node a wrong coordinate in 14 of 14 plain runs,
  with cuSPARSE's in none of 9 (jobs 23273228, 23274156, 23275289); the cause is not found
  (status.md). Cost: +3 % on Taylor-Green 32³ Q6 at np 2. `INCNS_HYPRE_SPMV_VENDOR=0`
  restores hypre's kernels.
- **`INCNS_HYPRE_POOL` does nothing:** hypre allocates through Umpire, not its own pool, so
  the cap `ConfigureHyprePool` puts on that pool ([gpu_memory.md](../gpu_memory.md)) does
  not apply. hypre's device memory near capacity has not been measured here.
- **MFEM 4.10's `HypreParMatrix::AbsMult` segfaults** on a rank whose block has a `rownnz`
  cache: hypre 3 builds that cache on the device and MFEM's host loop reads it
  (mfem/mfem#5595; fix in mfem/mfem#5610). No bessemer path calls it;
  `hanging_prolongation_test` HP1 takes |P| x as |P^T|^T x.
- Iteration counts and roundoff differ from Bridges-2's (another hypre, another AMG
  setup); tests compare against Bridges-2's numbers only through their tolerances.
