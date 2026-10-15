#include "operators/kernel_specializations.hpp"

#include "mfem.hpp"
#include "mfem/fem/integ/bilininteg_vecdiv_pa.hpp"
#include "mfem/fem/integ/bilininteg_vecmass_pa.hpp"

// The specializations instantiate device kernels here: nvcc must compile this
// file (src/CMakeLists.txt's LANGUAGE CUDA list), or a CUDA build would get
// host loops over device pointers.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "kernel_specializations.cpp instantiates device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

namespace
{

// Velocity order K: K+1 velocity and K pressure dofs per direction; K+1
// mass points (2K rule, or GLL), K+2 divergence points (2K+2 rule).
template <int K>
void AddOrder()
{
   using mfem::VectorDivergenceIntegrator;
   using mfem::VectorMassIntegrator;
   VectorMassIntegrator::VectorMassAddMultPA::Specialization < 2, K + 1,
                        K + 1 >::Add();
   VectorMassIntegrator::VectorMassAddMultPA::Specialization < 3, K + 1,
                        K + 1 >::Add();
   VectorDivergenceIntegrator::VectorDivergenceAddMultPA::
   Specialization < 2, K + 1, K, K + 2 >::Add();
   VectorDivergenceIntegrator::VectorDivergenceAddMultPA::
   Specialization < 3, K + 1, K, K + 2 >::Add();
   VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::
   Specialization < 2, K + 1, K, K + 2 >::Add();
   VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::
   Specialization < 3, K + 1, K, K + 2 >::Add();
}

} // namespace

void AddKernelSpecializations()
{
   static const bool added = []
   {
      AddOrder<2>();
      AddOrder<3>();
      AddOrder<4>();
      return true;
   }();
   (void)added;
}

} // namespace incns
