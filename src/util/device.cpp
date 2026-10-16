#include "util/device.hpp"

#include "mfem.hpp"

#include <cstdio>
#include <cstdlib>

namespace incns
{

namespace
{

/// @return This process's rank *within its node*, from the launcher's
///         environment, or -1 if no launcher variable is set.
/// Only the launcher knows how ranks were packed onto nodes -- the
/// MPI_COMM_WORLD rank does not (rank 4 of 8 may be the first rank on the
/// second node), so this is preferred over the world rank when available.
int LauncherLocalRank()
{
   // OpenMPI, MVAPICH/MPICH, Intel MPI, and Slurm respectively.
   for (const char* var :
        {"OMPI_COMM_WORLD_LOCAL_RANK",
         "MV2_COMM_WORLD_LOCAL_RANK",
         "MPI_LOCALRANKID",
         "SLURM_LOCALID"
        })
   {
      const char* v = std::getenv(var);
      if (v && v[0] != '\0') { return std::atoi(v); }
   }
   return -1;
}

/**
 * @brief Which GPU this rank should bind to (0 for host backends).
 *
 * mfem::Device defaults to device 0 for EVERY process (`Device::Configure`'s
 * @c device_id defaults to 0 and MFEM does no rank mapping). Without this,
 * every rank of a multi-rank job piles onto GPU 0 no matter how many GPUs were
 * allocated: the ranks serialize on one card's contexts and can exhaust its
 * memory, while the other GPUs sit idle. Observed on an H100 node as a ~11x
 * slowdown at np2 and an outright `CUDA error: out of memory` at np4.
 */
int ResolveDeviceId(const std::string& backend)
{
   // Backend strings may be lists ("cuda:cpu", "ceed-cuda"), so match on
   // substring rather than equality. Host-only backends always use id 0.
   if (backend.find("cuda") == std::string::npos &&
       backend.find("hip") == std::string::npos)
   {
      return 0;
   }

   const int ngpu = mfem::Device::GetDeviceCount();
   if (ngpu <= 1) { return 0; } // nothing to spread across

   int local = LauncherLocalRank();
   if (local < 0)
   {
      // No launcher hint. The world rank is exact for a single-node job and is
      // the best available guess otherwise.
      local = mfem::Mpi::IsInitialized() ? mfem::Mpi::WorldRank() : 0;
   }
   return local % ngpu;
}

/**
 * @brief Cap hypre's GPU memory pool before its first device allocation.
 *
 * Spack builds hypre with its device memory pool for CUDA (without Umpire),
 * and the pool's defaults round every allocation up to a power of 8 and keep
 * every freed block: on Taylor-Green Q7 at ~128^3 the pressure LOR-AMG held
 * 3.3 GB for ~0.6 GB of hierarchy. Pooling only blocks up to 1 MiB (bin growth
 * 2) with at most 64 MiB cached cut the run's peak by 21 % with no measured
 * cost per step (docs/gpu_memory.md). The pool's allocator is created at
 * hypre's first device allocation, so this runs before any hypre object exists.
 * INCNS_HYPRE_POOL="growth,min_bin,max_bin,cached_MiB" overrides it;
 * "8,1,-1,-1" restores hypre's defaults.
 */
void ConfigureHyprePool(const std::string& backend)
{
#if defined(HYPRE_USING_DEVICE_POOL)
   if (backend.find("cuda") == std::string::npos &&
       backend.find("hip") == std::string::npos)
   {
      return;
   }
   mfem::Hypre::Init(); // idempotent; the pool lives in hypre's handle
   int growth = 2, min_bin = 1, max_bin = 20;
   long cached_mib = 64;
   if (const char* env = std::getenv("INCNS_HYPRE_POOL"))
   {
      MFEM_VERIFY(std::sscanf(env, "%d,%d,%d,%ld", &growth, &min_bin, &max_bin,
                              &cached_mib) == 4,
                  "INCNS_HYPRE_POOL must be growth,min_bin,max_bin,cached_MiB");
   }
   const size_t cached = cached_mib < 0 ? static_cast<size_t>(-1)
                         : static_cast<size_t>(cached_mib) << 20;
   HYPRE_SetGPUMemoryPoolSize(growth, min_bin, max_bin, cached);
#else
   (void)backend;
#endif
}

} // namespace

void ConfigureDevice(const std::string& backend)
{
   // Environment override: INCNS_DEVICE, when set, wins over the requested
   // backend. This lets the whole test suite (or any driver) be re-run on a
   // different backend -- e.g. INCNS_DEVICE=debug forces MFEM's mprotect-guarded
   // "debug" device, which faults on any un-annotated host access of device
   // memory (a silent CPU fallback on a real GPU). See scripts/debug_device.sh.
   const char* env = std::getenv("INCNS_DEVICE");
   const std::string want = (env && env[0] != '\0') ? std::string(env) : backend;

   // One mfem::Device per process, for the program's lifetime. Held in a
   // function-local static so any driver (C++ or Python) can request it without
   // threading a Device object through the call tree.
   static mfem::Device device;
   static bool configured = false;
   static std::string current;

   if (!configured)
   {
      // Bind this rank to its own GPU; see ResolveDeviceId. On host backends,
      // and whenever there is at most one visible device, this is plain 0.
      device.Configure(want, ResolveDeviceId(want));
      ConfigureHyprePool(want);
      configured = true;
      current = want;
   }
   else
   {
      // Idempotent for the resolved backend. With INCNS_DEVICE set, every call
      // resolves to the same override, so a per-Case ConfigureDevice("cpu")
      // after the harness already selected "debug" is a no-op, not a conflict.
      MFEM_VERIFY(current == want,
                  "ConfigureDevice: the device is already configured as '"
                  << current << "' and cannot be reconfigured to '" << want
                  << "' (one mfem::Device per process)");
   }
}

} // namespace incns
