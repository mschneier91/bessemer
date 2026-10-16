#include "util/device.hpp"

#include "mfem.hpp"

#if defined(MFEM_USE_CUDA)
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#endif

// MFEM_DEVICE_SYNC is cudaDeviceSynchronize() only under nvcc (mfem
// general/cuda.hpp); compiled by a host compiler in a CUDA build it is empty,
// and DeviceSync() would silently do nothing. See src/CMakeLists.txt's nvcc
// list: this line turns a lost entry into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "device_sync.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

void DeviceSync()
{
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK))
   {
      MFEM_DEVICE_SYNC;
   }
}

bool DeviceMemoryInfo(double& free, double& total)
{
   free = total = 0.0;
#if defined(MFEM_USE_CUDA)
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
   {
      size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
      {
         free = static_cast<double>(free_bytes);
         total = static_cast<double>(total_bytes);
         return true;
      }
   }
#endif
   return false;
}

double DeviceMemoryInUse()
{
#if defined(MFEM_USE_CUDA)
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
   {
      size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
      {
         return static_cast<double>(total_bytes - free_bytes);
      }
   }
#endif
   return 0.0;
}

void NvtxPush(const char* name)
{
#if defined(MFEM_USE_CUDA)
   nvtxRangePushA(name);
#else
   (void)name;
#endif
}

void NvtxPop()
{
#if defined(MFEM_USE_CUDA)
   nvtxRangePop();
#endif
}

} // namespace incns
