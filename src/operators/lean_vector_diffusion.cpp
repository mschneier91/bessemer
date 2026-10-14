#include "operators/lean_vector_diffusion.hpp"

#include <mfem/general/forall.hpp>

// The kernels below must be compiled by nvcc in a CUDA build, or they run on
// the host while the data is on the device. See src/CMakeLists.txt's nvcc
// list: this line turns a lost entry into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "lean_vector_diffusion.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Device kernels live in free functions in a NAMED namespace: nvcc rejects
// extended lambdas in functions with internal linkage or in private members.
namespace leandiff
{

// pa_data as VectorDiffusionIntegrator::AssemblePA writes it for sdim == dim
// and a scalar coefficient: (quadrature point, dim * dim entries, component,
// element). 2D: entries 0, 1, 3 are D_xx, D_xy, D_yy. 3D: entries 0..5 are
// D_xx, D_xy, D_xz, D_yy, D_yz, D_zz. Component 0 is read (they are equal).
void Diagonal2D(int ne, int D1D, int Q1D, const Array<real_t>& b,
                const Array<real_t>& g, const Vector& pa_data, Vector& diag)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, 4, 2, ne);
   auto Y = Reshape(diag.ReadWrite(), D1D, D1D, 2, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      for (int dy = 0; dy < D1D; ++dy)
         for (int dx = 0; dx < D1D; ++dx)
         {
            real_t acc = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t g0 = G(qx, dx) * B(qy, dy);
                  const real_t g1 = B(qx, dx) * G(qy, dy);
                  acc += D(qx, qy, 0, 0, e) * g0 * g0 +
                         2.0 * D(qx, qy, 1, 0, e) * g0 * g1 +
                         D(qx, qy, 3, 0, e) * g1 * g1;
               }
            Y(dx, dy, 0, e) += acc;
            Y(dx, dy, 1, e) += acc;
         }
   });
}

void Diagonal3D(int ne, int D1D, int Q1D, const Array<real_t>& b,
                const Array<real_t>& g, const Vector& pa_data, Vector& diag)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, 9, 3, ne);
   auto Y = Reshape(diag.ReadWrite(), D1D, D1D, D1D, 3, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      for (int dz = 0; dz < D1D; ++dz)
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               real_t acc = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
                  for (int qy = 0; qy < Q1D; ++qy)
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        const real_t g0 = G(qx, dx) * B(qy, dy) * B(qz, dz);
                        const real_t g1 = B(qx, dx) * G(qy, dy) * B(qz, dz);
                        const real_t g2 = B(qx, dx) * B(qy, dy) * G(qz, dz);
                        acc += D(qx, qy, qz, 0, 0, e) * g0 * g0 +
                               D(qx, qy, qz, 3, 0, e) * g1 * g1 +
                               D(qx, qy, qz, 5, 0, e) * g2 * g2 +
                               2.0 * (D(qx, qy, qz, 1, 0, e) * g0 * g1 +
                                      D(qx, qy, qz, 2, 0, e) * g0 * g2 +
                                      D(qx, qy, qz, 4, 0, e) * g1 * g2);
                     }
               Y(dx, dy, dz, 0, e) += acc;
               Y(dx, dy, dz, 1, e) += acc;
               Y(dx, dy, dz, 2, e) += acc;
            }
   });
}

} // namespace leandiff

void LeanVectorDiffusionIntegrator::AssembleDiagonalPA(Vector& diag)
{
   // The layouts above hold for a scalar coefficient with sdim == dim == vdim
   // and MFEM's own PA data; anything else is MFEM's business.
   // (vdim is private to the base class: read it off the size of pa_data.)
   long long nq = quad1D;
   for (int d = 1; d < dim; ++d) { nq *= quad1D; }
   const bool ours = !DeviceCanUseCeed() && !VQ && !MQ && coeff_vdim == 1 &&
                     sdim == dim && (dim == 2 || dim == 3) &&
                     pa_data.Size() == nq * dim * dim * dim * ne;
   if (!ours)
   {
      VectorDiffusionIntegrator::AssembleDiagonalPA(diag);
      return;
   }
   if (dim == 2)
   {
      leandiff::Diagonal2D(ne, dofs1D, quad1D, maps->B, maps->G, pa_data, diag);
   }
   else
   {
      leandiff::Diagonal3D(ne, dofs1D, quad1D, maps->B, maps->G, pa_data, diag);
   }
}

} // namespace incns
