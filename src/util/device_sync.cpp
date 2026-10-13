#include "util/device.hpp"

#include "mfem.hpp"

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

} // namespace incns
