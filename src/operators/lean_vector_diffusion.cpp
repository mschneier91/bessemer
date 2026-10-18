#include "operators/lean_vector_diffusion.hpp"

#include "operators/component_kernels.hpp"
#include "operators/grad_div_integrator.hpp"

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

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

using kernels::internal::SetMaxOf;

// ---------------------------------------------------------------------------
// Compact setup: the symmetric geometric matrix per point, times the
// coefficient. The expressions are those of VectorDiffusionIntegrator::
// AssemblePA (dim == sdim), so the stored values are MFEM's.
// ---------------------------------------------------------------------------
void Setup(int dim, int ne, int nq, const Array<real_t>& w, const Vector& jac,
           const real_t* coeff, real_t coeff_const, Vector& d)
{
   const auto W = Reshape(w.Read(), nq);
   const int ns = (dim == 3) ? 6 : 3;
   auto D = Reshape(d.Write(), nq, ns, ne);
   if (dim == 2)
   {
      const auto J = Reshape(jac.Read(), nq, 2, 2, ne);
      mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int idx)
      {
         const int q = idx % nq, e = idx / nq;
         const real_t c = coeff ? coeff[q + nq * e] : coeff_const;
         const real_t J11 = J(q, 0, 0, e), J21 = J(q, 1, 0, e);
         const real_t J12 = J(q, 0, 1, e), J22 = J(q, 1, 1, e);
         const real_t w_detJ = W(q) / ((J11 * J22) - (J21 * J12));
         D(q, 0, e) = (w_detJ * (J12 * J12 + J22 * J22)) * c;
         D(q, 1, e) = (-w_detJ * (J12 * J11 + J22 * J21)) * c;
         D(q, 2, e) = (w_detJ * (J11 * J11 + J21 * J21)) * c;
      });
   }
   else
   {
      const auto J = Reshape(jac.Read(), nq, 3, 3, ne);
      mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int idx)
      {
         const int q = idx % nq, e = idx / nq;
         const real_t c = coeff ? coeff[q + nq * e] : coeff_const;
         const real_t J11 = J(q, 0, 0, e), J21 = J(q, 1, 0, e), J31 = J(q, 2, 0, e);
         const real_t J12 = J(q, 0, 1, e), J22 = J(q, 1, 1, e), J32 = J(q, 2, 1, e);
         const real_t J13 = J(q, 0, 2, e), J23 = J(q, 1, 2, e), J33 = J(q, 2, 2, e);
         const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         const real_t c_detJ = W(q) / detJ;
         const real_t A11 = (J22 * J33) - (J23 * J32);
         const real_t A12 = (J32 * J13) - (J12 * J33);
         const real_t A13 = (J12 * J23) - (J22 * J13);
         const real_t A21 = (J31 * J23) - (J21 * J33);
         const real_t A22 = (J11 * J33) - (J13 * J31);
         const real_t A23 = (J21 * J13) - (J11 * J23);
         const real_t A31 = (J21 * J32) - (J31 * J22);
         const real_t A32 = (J31 * J12) - (J11 * J32);
         const real_t A33 = (J11 * J22) - (J12 * J21);
         D(q, 0, e) = (c_detJ * (A11 * A11 + A12 * A12 + A13 * A13)) * c;
         D(q, 1, e) = (c_detJ * (A11 * A21 + A12 * A22 + A13 * A23)) * c;
         D(q, 2, e) = (c_detJ * (A11 * A31 + A12 * A32 + A13 * A33)) * c;
         D(q, 3, e) = (c_detJ * (A21 * A21 + A22 * A22 + A23 * A23)) * c;
         D(q, 4, e) = (c_detJ * (A21 * A31 + A22 * A32 + A23 * A33)) * c;
         D(q, 5, e) = (c_detJ * (A31 * A31 + A32 * A32 + A33 * A33)) * c;
      });
   }
}

// ---------------------------------------------------------------------------
// Compact apply: MFEM's SmemPAVectorDiffusionApply2D/3D (fem/integ/
// bilininteg_vecdiffusion_pa.hpp, scalar-coefficient branch), each component
// reading the one shared symmetric matrix.
// ---------------------------------------------------------------------------
template <int T_D1D = 0, int T_Q1D = 0>
void CompactApply2D(const int NE, const Array<real_t>& b,
                    const Array<real_t>& g,
                    const Vector& d, const Vector& x, Vector& y,
                    const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, 3, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      for (int i = 0; i < VDIM; i++)
      {
         kernels::internal::LoadDofs2d(e, D1D, i, XE, r0);
         kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t gradX = r1[i][0][qy][qx];
               const real_t gradY = r1[i][1][qy][qx];
               const real_t O11 = DE(qx, qy, 0, e), O12 = DE(qx, qy, 1, e);
               const real_t O22 = DE(qx, qy, 2, e);
               r0[i][0][qy][qx] = (O11 * gradX) + (O12 * gradY);
               r0[i][1][qy][qx] = (O12 * gradX) + (O22 * gradY);
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1, i);
         kernels::internal::WriteDofs2d(e, D1D, i, i, r1, YE);
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void CompactApply3D(const int NE, const Array<real_t>& b,
                    const Array<real_t>& g,
                    const Vector& d, const Vector& x, Vector& y,
                    const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, 6, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;

      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      // The component loops are unrolled: r0 and r1 hold VDIM x DIM x Q1D
      // values per thread and stay in registers only when every index is a
      // compile-time constant (with a runtime component index they went to
      // the stack: 720-792 bytes of local memory per thread at Q1D = 5, most
      // of the kernel's DRAM traffic; Nsight Compute, 2026-10-15).
      MFEM_UNROLL(VDIM)
      for (int i = 0; i < VDIM; i++)
      {
         kernels::internal::LoadDofs3d(e, D1D, i, XE, r0);
         kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, i);
         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t gradX = r1[i][0][qz][qy][qx];
                  const real_t gradY = r1[i][1][qz][qy][qx];
                  const real_t gradZ = r1[i][2][qz][qy][qx];
                  const real_t O11 = DE(qx, qy, qz, 0, e), O12 = DE(qx, qy, qz, 1, e),
                               O13 = DE(qx, qy, qz, 2, e);
                  const real_t O22 = DE(qx, qy, qz, 3, e), O23 = DE(qx, qy, qz, 4, e);
                  const real_t O33 = DE(qx, qy, qz, 5, e);
                  r0[i][0][qz][qy][qx] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
                  r0[i][1][qz][qy][qx] = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ);
                  r0[i][2][qz][qy][qx] = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ);
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, i);
         kernels::internal::WriteDofs3d(e, D1D, i, i, r1, YE);
      }
   });
}

// CompactApply3D at Q1D >= 7, one component at a time
// (operators/component_kernels.hpp): there r0 and r1 went to the stack even
// with the component loop unrolled, 1008 and 1152 bytes per thread at (6, 7)
// and (7, 8) (ptxas, 2026-10-16). The components are independent; the
// arithmetic is CompactApply3D's.
template <int D1D, int Q1D>
void CompactApply3DByComponent(const int NE, const Array<real_t>& b,
                               const Array<real_t>& g, const Vector& d,
                               const Vector& x, Vector& y, const int,
                               const int)
{
   static constexpr int VDIM = 3;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, 6, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = SetMaxOf(D1D), MQ1 = SetMaxOf(Q1D);
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<1, 3, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);
      MFEM_UNROLL(VDIM)
      for (int i = 0; i < VDIM; i++)
      {
         kernels3d::LoadComponent(e, D1D, i, XE, r0);
         kernels3d::GradComponent<false>(D1D, Q1D, smem, sB, sG, r0, r1);
         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t gradX = r1[0][0][qz][qy][qx];
                  const real_t gradY = r1[0][1][qz][qy][qx];
                  const real_t gradZ = r1[0][2][qz][qy][qx];
                  const real_t O11 = DE(qx, qy, qz, 0, e), O12 = DE(qx, qy, qz, 1, e),
                               O13 = DE(qx, qy, qz, 2, e);
                  const real_t O22 = DE(qx, qy, qz, 3, e), O23 = DE(qx, qy, qz, 4, e);
                  const real_t O33 = DE(qx, qy, qz, 5, e);
                  r0[0][0][qz][qy][qx] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
                  r0[0][1][qz][qy][qx] = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ);
                  r0[0][2][qz][qy][qx] = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ);
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels3d::GradComponent<true>(D1D, Q1D, smem, sB, sG, r0, r1);
         kernels::internal::WriteDofs3d(e, D1D, 0, i, r1, YE);
      }
   });
}

// ---------------------------------------------------------------------------
// The diagonal, both layouts: diag(a, c) = sum_q ghat_a^T D_q ghat_a, ghat_a
// the reference gradient of dof a's basis function rebuilt from the 1D tables;
// the same for every component (a scalar coefficient). No local arrays.
// MFEM layout: (Q1D^dim, dim*dim, vdim, ne), component 0 read; 3D entries
// 0..5 = xx, xy, xz, yy, yz, zz, 2D entries 0, 1, 3 = xx, xy, yy. Compact:
// (Q1D^dim, ns, ne) as documented in the header.
// ---------------------------------------------------------------------------
void Diagonal2D(int ne, int D1D, int Q1D, bool compact, const Array<real_t>& b,
                const Array<real_t>& g, const Vector& pa_data, Vector& diag)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const int stride = compact ? 3 : 8; // entries per point per element
   const int iyy = compact ? 2 : 3;
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, stride, ne);
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
                  acc += D(qx, qy, 0, e) * g0 * g0 +
                         2.0 * D(qx, qy, 1, e) * g0 * g1 +
                         D(qx, qy, iyy, e) * g1 * g1;
               }
            Y(dx, dy, 0, e) += acc;
            Y(dx, dy, 1, e) += acc;
         }
   });
}

void Diagonal3D(int ne, int D1D, int Q1D, bool compact, const Array<real_t>& b,
                const Array<real_t>& g, const Vector& pa_data, Vector& diag)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const int stride = compact ? 6 : 27; // entries per point per element
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, stride, ne);
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
                        acc += D(qx, qy, qz, 0, e) * g0 * g0 +
                               D(qx, qy, qz, 3, e) * g1 * g1 +
                               D(qx, qy, qz, 5, e) * g2 * g2 +
                               2.0 * (D(qx, qy, qz, 1, e) * g0 * g1 +
                                      D(qx, qy, qz, 2, e) * g0 * g2 +
                                      D(qx, qy, qz, 4, e) * g1 * g2);
                     }
               Y(dx, dy, dz, 0, e) += acc;
               Y(dx, dy, dz, 1, e) += acc;
               Y(dx, dy, dz, 2, e) += acc;
            }
   });
}


// adj(J) at a point, A[k][i] (row k: reference direction, column i: physical
// component), with the expressions graddiv's kernels use (grad_div_integrator.cpp
// View3D/View2D), so the fused apply forms the same values.
template <typename TJ>
MFEM_HOST_DEVICE inline void Adjugate3D(const TJ& J, int qx, int qy, int qz,
                                        int e, real_t A[3][3])
{
   const real_t j00 = J(qx, qy, qz, 0, 0, e), j01 = J(qx, qy, qz, 0, 1, e);
   const real_t j02 = J(qx, qy, qz, 0, 2, e), j10 = J(qx, qy, qz, 1, 0, e);
   const real_t j11 = J(qx, qy, qz, 1, 1, e), j12 = J(qx, qy, qz, 1, 2, e);
   const real_t j20 = J(qx, qy, qz, 2, 0, e), j21 = J(qx, qy, qz, 2, 1, e);
   const real_t j22 = J(qx, qy, qz, 2, 2, e);
   A[0][0] = (j11 * j22) - (j12 * j21);
   A[1][0] = (j20 * j12) - (j10 * j22);
   A[2][0] = (j10 * j21) - (j20 * j11);
   A[0][1] = (j21 * j02) - (j01 * j22);
   A[1][1] = (j00 * j22) - (j02 * j20);
   A[2][1] = (j20 * j01) - (j00 * j21);
   A[0][2] = (j01 * j12) - (j11 * j02);
   A[1][2] = (j10 * j02) - (j00 * j12);
   A[2][2] = (j00 * j11) - (j01 * j10);
}

template <typename TJ>
MFEM_HOST_DEVICE inline void Adjugate2D(const TJ& J, int qx, int qy, int e,
                                        real_t A[2][2])
{
   A[0][0] = J(qx, qy, 1, 1, e);
   A[1][0] = -J(qx, qy, 1, 0, e);
   A[0][1] = -J(qx, qy, 0, 1, e);
   A[1][1] = J(qx, qy, 0, 0, e);
}

// The compact apply with grad-div's term (LeanVectorDiffusionIntegrator::
// FuseGradDiv). Pass 1: every component's reference gradients (r1 keeps them
// all) and, per point, t = alpha S, S = sum_c sum_k grad_c[k] A[k][c] summed
// as graddiv's kernel sums it. Pass 2, per component: the flux O grad_c +
// t A[:,c], contracted back and written once.
template <int T_D1D = 0, int T_Q1D = 0>
void FusedApply2D(const int NE, const Array<real_t>& b, const Array<real_t>& g,
                  const Vector& d, const Vector& j, const Vector& alpha,
                  const Vector& x, Vector& y, const int d1d = 0,
                  const int q1d = 0)
{
   static constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, 3, NE);
   const auto J = Reshape(j.Read(), Q1D, Q1D, 2, 2, NE);
   const auto AL = Reshape(alpha.Read(), Q1D, Q1D, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::vd_regs2d_t<1, 1, MQ1> T;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs2d(e, D1D, c, XE, r0);
         kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1, c);
      }
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            real_t A[2][2];
            Adjugate2D(J, qx, qy, e, A);
            real_t s = 0.0;
            for (int c = 0; c < VDIM; ++c)
            {
               real_t acc = 0.0;
               for (int k = 0; k < DIM; ++k) { acc += r1[c][k][qy][qx] * A[k][c]; }
               s += acc;
            }
            T[0][0][qy][qx] = AL(qx, qy, e) * s;
         }
      }
      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               real_t A[2][2];
               Adjugate2D(J, qx, qy, e, A);
               const real_t t = T[0][0][qy][qx];
               const real_t gradX = r1[c][0][qy][qx];
               const real_t gradY = r1[c][1][qy][qx];
               const real_t O11 = DE(qx, qy, 0, e), O12 = DE(qx, qy, 1, e);
               const real_t O22 = DE(qx, qy, 2, e);
               r0[c][0][qy][qx] = (O11 * gradX) + (O12 * gradY) + t * A[0][c];
               r0[c][1][qy][qx] = (O12 * gradX) + (O22 * gradY) + t * A[1][c];
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs2d(e, D1D, c, c, r1, YE);
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void FusedApply3D(const int NE, const Array<real_t>& b, const Array<real_t>& g,
                  const Vector& d, const Vector& j, const Vector& alpha,
                  const Vector& x, Vector& y, const int d1d = 0,
                  const int q1d = 0)
{
   static constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, 6, NE);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto AL = Reshape(alpha.Read(), Q1D, Q1D, Q1D, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::vd_regs3d_t<1, 1, MQ1> T;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);
      // Unrolled component loops: see CompactApply3D.
      MFEM_UNROLL(VDIM)
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs3d(e, D1D, c, XE, r0);
         kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               real_t A[3][3];
               Adjugate3D(J, qx, qy, qz, e, A);
               real_t s = 0.0;
               MFEM_UNROLL(VDIM)
               for (int c = 0; c < VDIM; ++c)
               {
                  real_t acc = 0.0;
                  for (int k = 0; k < DIM; ++k)
                  {
                     acc += r1[c][k][qz][qy][qx] * A[k][c];
                  }
                  s += acc;
               }
               T[0][0][qz][qy][qx] = AL(qx, qy, qz, e) * s;
            }
         }
      }
      MFEM_UNROLL(VDIM)
      for (int c = 0; c < VDIM; ++c)
      {
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t A[3][3];
                  Adjugate3D(J, qx, qy, qz, e, A);
                  const real_t t = T[0][0][qz][qy][qx];
                  const real_t gradX = r1[c][0][qz][qy][qx];
                  const real_t gradY = r1[c][1][qz][qy][qx];
                  const real_t gradZ = r1[c][2][qz][qy][qx];
                  const real_t O11 = DE(qx, qy, qz, 0, e), O12 = DE(qx, qy, qz, 1, e),
                               O13 = DE(qx, qy, qz, 2, e);
                  const real_t O22 = DE(qx, qy, qz, 3, e), O23 = DE(qx, qy, qz, 4, e);
                  const real_t O33 = DE(qx, qy, qz, 5, e);
                  r0[c][0][qz][qy][qx] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ) +
                                         t * A[0][c];
                  r0[c][1][qz][qy][qx] = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ) +
                                         t * A[1][c];
                  r0[c][2][qz][qy][qx] = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ) +
                                         t * A[2][c];
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs3d(e, D1D, c, c, r1, YE);
      }
   });
}

// The fused apply in 3D at compile-time sizes: one thread per quadrature
// point (Q1D^3 per element), the shape of MFEM's SmemPAConvectionApply3D.
// Every value a thread holds is its own point's; the gradients of all
// components sit in shared memory (VDIM x DIM x Q1D^3) and the contractions
// go through shared memory. FusedApply3D (the runtime-size fallback) gives
// each thread a column of Q1D points of every component's gradient: 248
// registers and 12 % occupancy at (4, 5), against 56 and 55 % here; 1.09 ->
// 0.65 ms per apply on an H100 at TGV 32^3 (Nsight Compute, 2026-10-15). The
// same terms; the contractions sum in a different order (roundoff).
//
// One element's shared arrays, in reals: every component's gradients,
// grad-div's term per point and six scratch arrays of max(D1D, Q1D)^3.
template <int D1D, int Q1D>
struct FusedPointsShared
{
   static constexpr int DQ = (Q1D > D1D) ? Q1D : D1D;
   static constexpr int size = 3 * 3 * Q1D * Q1D * Q1D + Q1D * Q1D * Q1D +
                               6 * DQ * DQ * DQ;
};

// One element of the thread-per-point apply, its shared arrays carved from
// smem (FusedPointsShared<D1D, Q1D>::size reals).
template <int D1D, int Q1D, typename TM, typename TD, typename TJ,
          typename TA, typename TX, typename TY>
MFEM_HOST_DEVICE inline void FusedPointsElement(const int e, real_t* smem,
      const TM& B, const TM& G,
      const TD& DE, const TJ& J,
      const TA& AL, const TX& XE,
      const TY& YE)
{
   static constexpr int DIM = 3, VDIM = 3, NQ = Q1D * Q1D * Q1D;
   static constexpr int DQ = FusedPointsShared<D1D, Q1D>::DQ, NM = DQ * DQ * DQ;
   real_t (*sGr)[DIM][Q1D][Q1D][Q1D] = (real_t (*)[DIM][Q1D][Q1D][Q1D]) smem;
   real_t (*sT)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D])(smem + VDIM * DIM * NQ);
   real_t* sm0 = smem + (VDIM * DIM + 1) * NQ;
   real_t* sm1 = sm0 + NM;
   real_t* sm2 = sm1 + NM;
   real_t* sm3 = sm2 + NM;
   real_t* sm4 = sm3 + NM;
   real_t* sm5 = sm4 + NM;
   // Every component's reference gradient at the points.
   for (int c = 0; c < VDIM; ++c)
   {
      real_t (*u)[D1D][D1D] = (real_t (*)[D1D][D1D]) sm0;
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D) { u[dz][dy][dx] = XE(dx, dy, dz, c, e); }
         }
      }
      MFEM_SYNC_THREAD;
      real_t (*Bu)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm1;
      real_t (*Gu)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm2;
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t bu = 0.0, gu = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  bu += B(qx, dx) * u[dz][dy][dx];
                  gu += G(qx, dx) * u[dz][dy][dx];
               }
               Bu[dz][dy][qx] = bu;
               Gu[dz][dy][qx] = gu;
            }
         }
      }
      MFEM_SYNC_THREAD;
      real_t (*BBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm3;
      real_t (*GBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm4;
      real_t (*BGu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm5;
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t bbu = 0.0, gbu = 0.0, bgu = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  bbu += B(qy, dy) * Bu[dz][dy][qx];
                  gbu += G(qy, dy) * Bu[dz][dy][qx];
                  bgu += B(qy, dy) * Gu[dz][dy][qx];
               }
               BBu[dz][qy][qx] = bbu;
               GBu[dz][qy][qx] = gbu;
               BGu[dz][qy][qx] = bgu;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t gx = 0.0, gy = 0.0, gz = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  gx += B(qz, dz) * BGu[dz][qy][qx];
                  gy += B(qz, dz) * GBu[dz][qy][qx];
                  gz += G(qz, dz) * BBu[dz][qy][qx];
               }
               sGr[c][0][qz][qy][qx] = gx;
               sGr[c][1][qz][qy][qx] = gy;
               sGr[c][2][qz][qy][qx] = gz;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
   // grad-div's term at each point: alpha sum_c sum_k grad_c[k] A[k][c].
   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            real_t A[3][3];
            Adjugate3D(J, qx, qy, qz, e, A);
            real_t s = 0.0;
            MFEM_UNROLL(VDIM)
            for (int c = 0; c < VDIM; ++c)
            {
               real_t acc = 0.0;
               MFEM_UNROLL(DIM)
               for (int k = 0; k < DIM; ++k) { acc += sGr[c][k][qz][qy][qx] * A[k][c]; }
               s += acc;
            }
            sT[qz][qy][qx] = AL(qx, qy, qz, e) * s;
         }
      }
   }
   MFEM_SYNC_THREAD;
   // Per component: the flux O grad_c + t adj(J)[:, c], contracted back.
   // Unrolled: adj(J)[:, c] then has compile-time indices.
   MFEM_UNROLL(VDIM)
   for (int c = 0; c < VDIM; ++c)
   {
      real_t (*F0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm0;
      real_t (*F1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm1;
      real_t (*F2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm2;
      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t A[3][3];
               Adjugate3D(J, qx, qy, qz, e, A);
               const real_t t = sT[qz][qy][qx];
               const real_t gradX = sGr[c][0][qz][qy][qx];
               const real_t gradY = sGr[c][1][qz][qy][qx];
               const real_t gradZ = sGr[c][2][qz][qy][qx];
               const real_t O11 = DE(qx, qy, qz, 0, e), O12 = DE(qx, qy, qz, 1, e),
                            O13 = DE(qx, qy, qz, 2, e);
               const real_t O22 = DE(qx, qy, qz, 3, e), O23 = DE(qx, qy, qz, 4, e);
               const real_t O33 = DE(qx, qy, qz, 5, e);
               F0[qz][qy][qx] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ) + t * A[0][c];
               F1[qz][qy][qx] = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ) + t * A[1][c];
               F2[qz][qy][qx] = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ) + t * A[2][c];
            }
         }
      }
      MFEM_SYNC_THREAD;
      // z: B for F0 and F1, G for F2.
      real_t (*Z0)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm3;
      real_t (*Z1)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm4;
      real_t (*Z2)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm5;
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t z0 = 0.0, z1 = 0.0, z2 = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  z0 += B(qz, dz) * F0[qz][qy][qx];
                  z1 += B(qz, dz) * F1[qz][qy][qx];
                  z2 += G(qz, dz) * F2[qz][qy][qx];
               }
               Z0[dz][qy][qx] = z0;
               Z1[dz][qy][qx] = z1;
               Z2[dz][qy][qx] = z2;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // y: B for F0, G for F1, B for F2; F1 and F2 both take B in x.
      real_t (*W0)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm0;
      real_t (*W1)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm1;
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t w0 = 0.0, w1 = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  w0 += B(qy, dy) * Z0[dz][qy][qx];
                  w1 += G(qy, dy) * Z1[dz][qy][qx] + B(qy, dy) * Z2[dz][qy][qx];
               }
               W0[dz][dy][qx] = w0;
               W1[dz][dy][qx] = w1;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // x: G for F0, B for F1 and F2.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               real_t v = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  v += G(qx, dx) * W0[dz][dy][qx] + B(qx, dx) * W1[dz][dy][qx];
               }
               YE(dx, dy, dz, c, e) += v;
            }
         }
      }
      MFEM_SYNC_THREAD;
   }
}

// The thread-per-point apply with static shared memory: up to 48 KB, so Q1D
// up to 7 (44 KB at (7, 7)).
template <int D1D, int Q1D>
void FusedApply3DPoints(const int NE, const Array<real_t>& b,
                        const Array<real_t>& g, const Vector& d,
                        const Vector& j, const Vector& alpha, const Vector& x,
                        Vector& y, const int, const int)
{
   static constexpr int VDIM = 3;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, 6, NE);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto AL = Reshape(alpha.Read(), Q1D, Q1D, Q1D, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t smem[FusedPointsShared<D1D, Q1D>::size];
      FusedPointsElement<D1D, Q1D>(e, smem, B, G, DE, J, AL, XE, YE);
   });
}

// A forall_3D whose kernel takes its shared memory dynamically, for more than
// the 48 KB of static shared memory: opted in once per kernel (the H100 allows
// 227 KB per block). An X x Y x Z thread block per element, at most 512
// threads. On the host: the elements in turn, the arrays in one buffer.
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
template <typename BODY>
__global__ void __launch_bounds__(512) DynamicSharedKernel3D(const int N,
      BODY body)
{
   extern __shared__ real_t dyn_smem[];
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k, dyn_smem); }
}
#endif

template <typename BODY>
void ForallDynamicShared3D(const int N, const int X, const int Y, const int Z,
                           const int reals, BODY&& body)
{
   if (N == 0) { return; }
   MFEM_VERIFY(X * Y * Z <= 512, "ForallDynamicShared3D: at most 512 threads");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
   if (Device::Allows(Backend::CUDA_MASK))
   {
      using K = typename std::decay<BODY>::type;
      const std::size_t bytes = sizeof(real_t) * reals;
      static bool opted_in = false;
      if (!opted_in)
      {
         MFEM_GPU_CHECK(cudaFuncSetAttribute(DynamicSharedKernel3D<K>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(bytes)));
         opted_in = true;
      }
      DynamicSharedKernel3D<K> <<< N, dim3(X, Y, Z), bytes>>>(N, body);
      MFEM_GPU_CHECK(cudaGetLastError());
      return;
   }
#endif
   std::vector<real_t> buf(reals);
   for (int e = 0; e < N; ++e) { body(e, buf.data()); }
}

// The thread-per-point apply with dynamic shared memory: (7, 8), 64 KB.
template <int D1D, int Q1D>
void FusedApply3DPointsDynamic(const int NE, const Array<real_t>& b,
                               const Array<real_t>& g, const Vector& d,
                               const Vector& j, const Vector& alpha,
                               const Vector& x, Vector& y, const int, const int)
{
   static_assert(Q1D * Q1D * Q1D <= 512, "a thread per point: at most 512");
   static constexpr int VDIM = 3;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, 6, NE);
   const auto J = Reshape(j.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto AL = Reshape(alpha.Read(), Q1D, Q1D, Q1D, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   ForallDynamicShared3D(NE, Q1D, Q1D, Q1D, FusedPointsShared<D1D, Q1D>::size,
                         [ = ] MFEM_HOST_DEVICE(int e, real_t* smem)
   {
      FusedPointsElement<D1D, Q1D>(e, smem, B, G, DE, J, AL, XE, YE);
   });
}

} // namespace leandiff

/// \cond DO_NOT_DOCUMENT
template <int DIM, int D1D, int Q1D>
LeanVectorDiffusionIntegrator::ApplyKernelType
LeanVectorDiffusionIntegrator::CompactApplyKernels::Kernel()
{
   if constexpr(DIM == 2) { return leandiff::CompactApply2D<D1D, Q1D>; }
   else if constexpr(DIM == 3 && Q1D >= 7)
   {
      return leandiff::CompactApply3DByComponent<D1D, Q1D>;
   }
   else if constexpr(DIM == 3) { return leandiff::CompactApply3D<D1D, Q1D>; }
   MFEM_ABORT("LeanVectorDiffusionIntegrator: unsupported kernel dim");
}

LeanVectorDiffusionIntegrator::ApplyKernelType
LeanVectorDiffusionIntegrator::CompactApplyKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return leandiff::CompactApply2D; }
   else if (dim == 3) { return leandiff::CompactApply3D; }
   MFEM_ABORT("LeanVectorDiffusionIntegrator: unsupported kernel dim");
}

template <int DIM, int D1D, int Q1D>
LeanVectorDiffusionIntegrator::FusedKernelType
LeanVectorDiffusionIntegrator::FusedApplyKernels::Kernel()
{
   if constexpr(DIM == 2) { return leandiff::FusedApply2D<D1D, Q1D>; }
   // A thread per point. Its shared memory is static up to 48 KB (Q1D up to
   // 7: 44 KB at (7, 7)) and dynamic beyond: (7, 8) takes 64 KB, 0.94 -> 0.57
   // ms per apply against FusedApply3D's column of points (96 -> 64
   // registers, occupancy 29 -> 49 %; TGV 16^3 Q6, one H100, job 49397767),
   // 11-24 % off the step at 21.6M DOFs. Q1D = 9 (729 threads) keeps
   // FusedApply3D.
   else if constexpr(DIM == 3 && Q1D <= 7)
   {
      return leandiff::FusedApply3DPoints<D1D, Q1D>;
   }
   else if constexpr(DIM == 3 && Q1D == 8)
   {
      return leandiff::FusedApply3DPointsDynamic<D1D, Q1D>;
   }
   else if constexpr(DIM == 3) { return leandiff::FusedApply3D<D1D, Q1D>; }
   MFEM_ABORT("LeanVectorDiffusionIntegrator: unsupported kernel dim");
}

LeanVectorDiffusionIntegrator::FusedKernelType
LeanVectorDiffusionIntegrator::FusedApplyKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return leandiff::FusedApply2D; }
   else if (dim == 3) { return leandiff::FusedApply3D; }
   MFEM_ABORT("LeanVectorDiffusionIntegrator: unsupported kernel dim");
}

// The (DIM, D1D, Q1D) specializations GradDivIntegrator registers (through Q7
// velocity; GL rules give Q1D ~ D1D + 1, collocated GLL Q1D == D1D): the
// untemplated fallback sizes its registers at MAX_T1D.
namespace
{
struct LeanVectorDiffusionRegistrar
{
   LeanVectorDiffusionRegistrar()
   {
      using I = LeanVectorDiffusionIntegrator;
      I::AddSpecialization<2, 3, 3>(); I::AddSpecialization<2, 3, 4>();
      I::AddSpecialization<2, 4, 4>(); I::AddSpecialization<2, 4, 5>();
      I::AddSpecialization<2, 5, 5>(); I::AddSpecialization<2, 5, 6>();
      I::AddSpecialization<2, 6, 6>(); I::AddSpecialization<2, 6, 7>();
      I::AddSpecialization<2, 7, 7>(); I::AddSpecialization<2, 7, 8>();
      I::AddSpecialization<3, 3, 3>(); I::AddSpecialization<3, 3, 4>();
      I::AddSpecialization<3, 4, 4>(); I::AddSpecialization<3, 4, 5>();
      I::AddSpecialization<3, 5, 5>(); I::AddSpecialization<3, 5, 6>();
      I::AddSpecialization<3, 6, 6>(); I::AddSpecialization<3, 6, 7>();
      I::AddSpecialization<3, 7, 7>(); I::AddSpecialization<3, 7, 8>();
      I::AddSpecialization<2, 8, 8>(); I::AddSpecialization<2, 8, 9>();
      I::AddSpecialization<3, 8, 8>(); I::AddSpecialization<3, 8, 9>();
   }
};
const LeanVectorDiffusionRegistrar lean_vector_diffusion_registrar;
} // namespace
/// \endcond

void LeanVectorDiffusionIntegrator::AssemblePA(const FiniteElementSpace& fes)
{
   Mesh* mesh = fes.GetMesh();
   const FiniteElement& el = *fes.GetTypicalFE();
   const int dim = mesh->Dimension();
   const bool tensor = dynamic_cast<const TensorBasisElement*>(&el) != nullptr;
   const bool eligible =
      storage_ == Storage::Compact && !DeviceCanUseCeed() && !VQ && !MQ &&
      (dim == 2 || dim == 3) && mesh->SpaceDimension() == dim &&
      fes.GetVDim() == dim && tensor && !fes.IsVariableOrder() &&
      mesh->GetNumGeometries(dim) <= 1;
   if (!eligible)
   {
      compact_ = false;
      cpa_.Destroy();
      VectorDiffusionIntegrator::AssemblePA(fes);
      return;
   }
   compact_ = true;
   pa_data.Destroy(); // in case an earlier assembly used MFEM's storage

   const IntegrationRule* ir =
      IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                         ? Device::GetDeviceMemoryType() : pa_mt;
   cdim_ = dim;
   cne_ = fes.GetNE();
   const GeometricFactors* geom =
      mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   cmaps_ = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   cd1d_ = cmaps_->ndof;
   cq1d_ = cmaps_->nqpt;
   const int nq = ir->GetNPoints();

   // A constant coefficient (the viscosity, here) needs no per-point array.
   const auto* qc = dynamic_cast<const ConstantCoefficient*>(Q);
   std::unique_ptr<CoefficientVector> coeff;
   const real_t* cp = nullptr;
   real_t cconst = qc ? qc->constant : 1.0;
   if (Q && !qc)
   {
      QuadratureSpace qs(*mesh, *ir);
      coeff = std::make_unique<CoefficientVector>(qs, CoefficientStorage::FULL);
      coeff->Project(*Q);
      MFEM_VERIFY(coeff->GetVDim() == 1, "LeanVectorDiffusionIntegrator: "
                  "scalar coefficient required");
      cp = coeff->Read();
   }
   const int ns = (dim == 3) ? 6 : 3;
   cpa_.SetSize(nq * ns * cne_, mt);
   cpa_.UseDevice(true);
   if (cne_ > 0)
   {
      leandiff::Setup(dim, cne_, nq, ir->GetWeights(), geom->J, cp, cconst, cpa_);
   }
}

void LeanVectorDiffusionIntegrator::AddMultPA(const Vector& x, Vector& y) const
{
   if (!compact_)
   {
      VectorDiffusionIntegrator::AddMultPA(x, y);
      return;
   }
   if (cne_ == 0) { return; }
   if (fused_gd_)
   {
      FusedApplyKernels::Run(cdim_, cd1d_, cq1d_, cne_, cmaps_->B, cmaps_->G,
                             cpa_, fused_gd_->PAJacobians(),
                             fused_gd_->PAAlpha(), x, y, cd1d_, cq1d_);
      return;
   }
   CompactApplyKernels::Run(cdim_, cd1d_, cq1d_, cne_, cmaps_->B, cmaps_->G,
                            cpa_, x, y, cd1d_, cq1d_);
}

bool LeanVectorDiffusionIntegrator::FuseGradDiv(GradDivIntegrator& gd)
{
   fused_gd_ = nullptr;
   gd.SetApplyFused(false);
   if (!compact_ || cne_ == 0 || gd.PAMaps() == nullptr) { return false; }
   if (gd.PAMaps() != cmaps_ || gd.PADim() != cdim_ ||
       gd.PANumElements() != cne_ || gd.PADofs1D() != cd1d_ ||
       gd.PAQuad1D() != cq1d_)
   {
      return false;
   }
   fused_gd_ = &gd;
   gd.SetApplyFused(true);
   return true;
}

void LeanVectorDiffusionIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (compact_)
   {
      if (cne_ == 0) { return; }
      if (cdim_ == 2)
      {
         leandiff::Diagonal2D(cne_, cd1d_, cq1d_, true, cmaps_->B, cmaps_->G,
                              cpa_, diag);
      }
      else
      {
         leandiff::Diagonal3D(cne_, cd1d_, cq1d_, true, cmaps_->B, cmaps_->G,
                              cpa_, diag);
      }
      return;
   }
   // MFEM's storage: its layout holds for a scalar coefficient with
   // sdim == dim == vdim (vdim is private to the base class: read it off the
   // size of pa_data); anything else is MFEM's business.
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
      leandiff::Diagonal2D(ne, dofs1D, quad1D, false, maps->B, maps->G, pa_data,
                           diag);
   }
   else
   {
      leandiff::Diagonal3D(ne, dofs1D, quad1D, false, maps->B, maps->G, pa_data,
                           diag);
   }
}

} // namespace incns
