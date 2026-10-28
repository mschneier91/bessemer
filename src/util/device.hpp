/**
 * @file device.hpp
 * @brief Process-wide MFEM device (backend) configuration.
 */
#pragma once

#include <string>

namespace incns
{

/**
 * @brief Configure the MFEM device backend once per process.
 *
 * Constructs the single mfem::Device with @p backend (e.g. @c "cpu",
 * @c "cuda", @c "hip", @c "cuda:cpu") the first time it is called and is a
 * no-op on later calls with the SAME backend; a different backend is an error
 * (MFEM allows only one Device per process). Call it during case setup, before
 * any mesh / finite element space / true-dof vector is allocated, so all
 * partial-assembly operators and their vectors live in the right memory space.
 *
 * The environment variable @c INCNS_DEVICE, when set, overrides @p backend for
 * the whole process -- used to re-run the suite on MFEM's mprotect-guarded
 * @c "debug" device (catches silent host fallbacks) without editing callers.
 *
 * On a GPU backend each rank is bound to its **own** device
 * (@c node_local_rank @c % @c device_count). mfem::Device would otherwise put
 * every rank on device 0, so a multi-rank job would serialize on a single GPU
 * (and can exhaust its memory) however many GPUs were allocated. The node-local
 * rank comes from the launcher (@c OMPI_COMM_WORLD_LOCAL_RANK, @c SLURM_LOCALID,
 * ...), falling back to the MPI_COMM_WORLD rank. Allocate at least as many GPUs
 * as there are ranks per node, or ranks will double up on a device.
 *
 * On a GPU backend it also caps hypre's device memory pool (when hypre was
 * built with it): only blocks up to 1 MiB are pooled, at most 64 MiB cached.
 * The environment variable @c INCNS_HYPRE_POOL
 * (@c "growth,min_bin,max_bin,cached_MiB"; @c "8,1,-1,-1" is hypre's default)
 * overrides it. hypre's SpMVs use its own kernels, not cuSPARSE's (several
 * times faster in the pressure AMG at more than one rank);
 * @c INCNS_HYPRE_SPMV_VENDOR=1 restores cuSPARSE.
 *
 * @param backend The mfem::Device configuration string (default when
 *                @c INCNS_DEVICE is unset).
 */
void ConfigureDevice(const std::string& backend);

/**
 * @brief MPI transport settings that must precede MPI_Init: call it first
 *        in main, before mfem::Mpi::Init.
 *
 * In CUDA builds, unless the environment sets @c UCX_TLS already, UCX's
 * cuda_ipc transport is left out (@c UCX_TLS=^cuda_ipc; GPU-aware MPI then
 * goes through the other CUDA paths). With bessemer's device buffers cuda_ipc
 * cost 2.5 GB per GPU near capacity (np 2) and at np 4 doubled the step of
 * Taylor-Green Q6 IMEX Schur at 136.6M DOFs (2.00 s against 0.79 s without
 * it); at 21.6M it was 8 % faster (0.139 against 0.151 s; jobs 49401722,
 * 49408610). UCX reads the variable during MPI_Init.
 *
 * With Cray MPICH in a CUDA build (CMake links its GPU Transport Layer and
 * defines @c INCNS_MPI_GTL), it also sets @c MPICH_GPU_SUPPORT_ENABLED=1
 * unless the environment sets it: Cray MPICH then takes device buffers, and
 * ConfigureDevice turns GPU-aware MPI on. @c MPICH_GPU_SUPPORT_ENABLED=0 in
 * the job keeps exchanges staged through the host. It also makes the rank's
 * own GPU current (ConfigureDevice's choice, from the launcher's local rank):
 * Cray's GTL keeps the device current at MPI_Init and turns its CUDA IPC off
 * on a rank that moves to another one.
 */
void ConfigureMpiTransport();

/**
 * @brief Wait until the GPU has finished everything launched so far.
 *
 * A no-op on host backends. On a GPU it blocks until the device is idle, so a
 * wall-clock reading taken afterwards includes the kernels launched before it
 * (the profiler's `sync` mode, Profiler::SetSync).
 */
void DeviceSync();

/**
 * @brief Memory in use on this process's GPU, in bytes: everything allocated
 *        there (MFEM, hypre, the CUDA context), as the driver reports it.
 *
 * For the profiler's gauge (Profiler::SetGauge). 0 on host backends and on
 * HIP. On a GPU shared with other processes it counts theirs too.
 * @return Bytes in use.
 */
double DeviceMemoryInUse();

/**
 * @brief Open a named NVTX range (Nsight Systems timelines), for the
 *        profiler's markers (Profiler::SetMarkers). A no-op without CUDA, and
 *        nearly free when no tool is attached.
 * @param name Range label.
 */
void NvtxPush(const char* name);

/// @brief Close the innermost NVTX range opened by NvtxPush.
void NvtxPop();

/**
 * @brief Free and total memory of this process's GPU, in bytes, as the
 *        driver reports them.
 * @param free  Output: free bytes.
 * @param total Output: total bytes.
 * @return False (and zeros) on host backends and on HIP.
 */
bool DeviceMemoryInfo(double& free, double& total);

} // namespace incns
