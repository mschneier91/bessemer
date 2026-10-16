#include "operators/kernel_specializations.hpp"

#include "operators/component_kernels.hpp"

#include "mfem.hpp"
#include "mfem/fem/integ/bilininteg_vecdiv_pa.hpp"
#include "mfem/fem/integ/bilininteg_vecmass_pa.hpp"
#include "mfem/fem/integ/nonlininteg_vecconvection_pa.hpp"

#include <tuple>
#include <type_traits>

// The specializations instantiate device kernels here: nvcc must compile this
// file (src/CMakeLists.txt's LANGUAGE CUDA list), or a CUDA build would get
// host loops over device pointers.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "kernel_specializations.cpp instantiates device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

// MFEM's 3D vector divergence apply and its transpose (internal::
// SmemPADivergenceApply3D, SmemPADivergenceApplyTranspose3D) with the
// component loops unrolled: the same arithmetic per component, so the same
// results, but the 3 x 3 x Q1D gradient registers keep compile-time indices.
// MFEM's loops over the components (in LoadDofs3d, Grad3d, GradTranspose3d,
// WriteDofs3d) are not unrolled, which put those arrays on the stack: 720
// bytes of local memory per thread and most of the kernels' DRAM traffic
// (Nsight Compute, 2026-10-15). Named namespace: nvcc rejects extended
// lambdas in functions with internal linkage.
namespace divkernels
{

template <int TR_D1D, int TE_D1D, int Q1D>
void Apply3D(const int NE, const mfem::Array<mfem::real_t>& b_,
             const mfem::Array<mfem::real_t>& g_,
             const mfem::Array<mfem::real_t>& bt_, const mfem::Vector& q_,
             const mfem::Vector& x_, mfem::Vector& y_, const int, const int,
             const int)
{
   using mfem::real_t;
   namespace ki = mfem::kernels::internal;
   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = mfem::Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = mfem::Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = mfem::Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      ki::vd_regs3d_t<3, 3, MQ1> g0, g1;
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::LoadMatrix(TR_D1D, Q1D, B, sB);
      ki::LoadMatrix(TR_D1D, Q1D, G, sG);
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         ki::LoadDofs3d(e, TR_D1D, c, X, g0);
         ki::Grad3d(TR_D1D, Q1D, smem, sB, sG, g0, g1, c);
      }
      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               r0[0][qz][qy][qx] =
                  g1[0][0][qz][qy][qx] * Q(qx, qy, qz, 0, 0, e) +
                  g1[0][1][qz][qy][qx] * Q(qx, qy, qz, 1, 0, e) +
                  g1[0][2][qz][qy][qx] * Q(qx, qy, qz, 2, 0, e) +
                  g1[1][0][qz][qy][qx] * Q(qx, qy, qz, 0, 1, e) +
                  g1[1][1][qz][qy][qx] * Q(qx, qy, qz, 1, 1, e) +
                  g1[1][2][qz][qy][qx] * Q(qx, qy, qz, 2, 1, e) +
                  g1[2][0][qz][qy][qx] * Q(qx, qy, qz, 0, 2, e) +
                  g1[2][1][qz][qy][qx] * Q(qx, qy, qz, 1, 2, e) +
                  g1[2][2][qz][qy][qx] * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      ki::LoadMatrix<MQ1, true>(TE_D1D, Q1D, Bt, sB);
      ki::EvalTranspose3d(TE_D1D, Q1D, smem, sB, r0, r1);
      ki::WriteDofs3d(e, TE_D1D, r1, Y);
   });
}

template <int TR_D1D, int TE_D1D, int Q1D>
void ApplyTranspose3D(const int NE, const mfem::Array<mfem::real_t>& bt,
                      const mfem::Array<mfem::real_t>& gt,
                      const mfem::Array<mfem::real_t>& b, const mfem::Vector& q_,
                      const mfem::Vector& x_, mfem::Vector& y_, const int,
                      const int, const int)
{
   using mfem::real_t;
   namespace ki = mfem::kernels::internal;
   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = mfem::Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = mfem::Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = mfem::Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::vd_regs3d_t<3, 3, MQ1> g0, g1;
      ki::LoadMatrix(TE_D1D, Q1D, B, sB);
      ki::LoadDofs3d(e, TE_D1D, X, r0);
      ki::Eval3d(TE_D1D, Q1D, smem, sB, r0, r1);
      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t r = r1[0][qz][qy][qx];
               g0[0][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 0, e);
               g0[0][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 0, e);
               g0[0][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 0, e);
               g0[1][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 1, e);
               g0[1][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 1, e);
               g0[1][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 1, e);
               g0[2][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, 2, e);
               g0[2][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, 2, e);
               g0[2][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, 2, e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      ki::LoadMatrix<MQ1, true>(TR_D1D, Q1D, Bt, sB);
      ki::LoadMatrix<MQ1, true>(TR_D1D, Q1D, Gt, sG);
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         ki::GradTranspose3d(TR_D1D, Q1D, smem, sB, sG, g0, g1, c);
         ki::WriteDofs3d(e, TR_D1D, c, c, g1, Y);
      }
   });
}

// Velocity orders 5 and up (Q1D >= 7): the kernels above one velocity
// component at a time (operators/component_kernels.hpp). There ptxas put the
// kernels above's two 3 x 3 x Q1D gradient arrays on the stack, 1008 and 1152
// bytes per thread at Q1D 7 (transpose) and 8 (both); one component's arrays
// are 6 Q1D doubles, in registers (2026-10-16). Each point's sum keeps the
// order of the kernels above (component, then direction).
template <int TR_D1D, int TE_D1D, int Q1D>
void Apply3DByComponent(const int NE, const mfem::Array<mfem::real_t>& b_,
                        const mfem::Array<mfem::real_t>& g_,
                        const mfem::Array<mfem::real_t>& bt_,
                        const mfem::Vector& q_, const mfem::Vector& x_,
                        mfem::Vector& y_, const int, const int, const int)
{
   using mfem::real_t;
   namespace ki = mfem::kernels::internal;
   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto Q = mfem::Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = mfem::Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = mfem::Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      ki::vd_regs3d_t<1, 3, MQ1> g0, g1;
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::LoadMatrix(TR_D1D, Q1D, B, sB);
      ki::LoadMatrix(TR_D1D, Q1D, G, sG);
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         kernels3d::LoadComponent(e, TR_D1D, c, X, g0);
         kernels3d::GradComponent<false>(TR_D1D, Q1D, smem, sB, sG, g0, g1);
         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t r = g1[0][0][qz][qy][qx] * Q(qx, qy, qz, 0, c, e);
                  if (c > 0) { r = r0[0][qz][qy][qx] + r; }
                  r += g1[0][1][qz][qy][qx] * Q(qx, qy, qz, 1, c, e);
                  r += g1[0][2][qz][qy][qx] * Q(qx, qy, qz, 2, c, e);
                  r0[0][qz][qy][qx] = r;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      ki::LoadMatrix<MQ1, true>(TE_D1D, Q1D, Bt, sB);
      ki::EvalTranspose3d(TE_D1D, Q1D, smem, sB, r0, r1);
      ki::WriteDofs3d(e, TE_D1D, r1, Y);
   });
}

template <int TR_D1D, int TE_D1D, int Q1D>
void ApplyTranspose3DByComponent(const int NE,
                                 const mfem::Array<mfem::real_t>& bt,
                                 const mfem::Array<mfem::real_t>& gt,
                                 const mfem::Array<mfem::real_t>& b,
                                 const mfem::Vector& q_, const mfem::Vector& x_,
                                 mfem::Vector& y_, const int, const int,
                                 const int)
{
   using mfem::real_t;
   namespace ki = mfem::kernels::internal;
   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto Q = mfem::Reshape(q_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = mfem::Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = mfem::Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::vd_regs3d_t<1, 3, MQ1> g0, g1;
      ki::LoadMatrix(TE_D1D, Q1D, B, sB);
      ki::LoadDofs3d(e, TE_D1D, X, r0);
      ki::Eval3d(TE_D1D, Q1D, smem, sB, r0, r1);
      MFEM_SYNC_THREAD;
      ki::LoadMatrix<MQ1, true>(TR_D1D, Q1D, Bt, sB);
      ki::LoadMatrix<MQ1, true>(TR_D1D, Q1D, Gt, sG);
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         for (int qz = 0; qz < Q1D; qz++)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t r = r1[0][qz][qy][qx];
                  g0[0][0][qz][qy][qx] = r * Q(qx, qy, qz, 0, c, e);
                  g0[0][1][qz][qy][qx] = r * Q(qx, qy, qz, 1, c, e);
                  g0[0][2][qz][qy][qx] = r * Q(qx, qy, qz, 2, c, e);
               }
            }
         }
         kernels3d::GradComponent<true>(TR_D1D, Q1D, smem, sB, sG, g0, g1);
         ki::WriteDofs3d(e, TR_D1D, 0, c, g1, Y);
      }
   });
}

} // namespace divkernels

namespace
{

// MFEM's dispatch table for a kernel registry, writable: MFEM exposes it
// read-only (GetDispatchTable); the table itself is an ordinary static.
template <typename Kernels>
auto& Table()
{
   using T = std::remove_const_t<std::remove_reference_t<decltype(
                Kernels::GetDispatchTable())>>;
   return const_cast<T&>(Kernels::GetDispatchTable());
}

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
   VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA::
   Specialization < 2, K + 1, K, K + 2 >::Add();
   // 3D: ours, in MFEM's tables under MFEM's keys (dim, trial d1d, test d1d,
   // q1d), so VectorDivergenceIntegrator dispatches to them.
   auto& apply =
      Table<VectorDivergenceIntegrator::VectorDivergenceAddMultPA>()
      [std::make_tuple(3, K + 1, K, K + 2)];
   auto& apply_t =
      Table<VectorDivergenceIntegrator::VectorDivergenceAddMultTransposePA>()
      [std::make_tuple(3, K + 1, K, K + 2)];
   if constexpr(K + 2 >= 7)
   {
      apply = divkernels::Apply3DByComponent < K + 1, K, K + 2 >;
      apply_t = divkernels::ApplyTranspose3DByComponent < K + 1, K, K + 2 >;
   }
   else
   {
      apply = divkernels::Apply3D < K + 1, K, K + 2 >;
      apply_t = divkernels::ApplyTranspose3D < K + 1, K, K + 2 >;
   }
}

} // namespace

void AddKernelSpecializations()
{
   static const bool added = []
   {
      AddOrder<2>();
      AddOrder<3>();
      AddOrder<4>();
      // Orders 5 and 6 (human 2026-10-16, before the polynomial-order study):
      // without these, MFEM's runtime-size kernels (arrays sized for 14
      // points per direction) ran there.
      AddOrder<5>();
      AddOrder<6>();
      // The IMEX nonlinear convection at the dealiased order 3K
      // (Convection::DealiasedOrder: K + 1 dofs, ceil((3K + 1) / 2) Gauss
      // points per direction); MFEM's table stops at 5 dofs in 3D. The apply
      // only: AddSpecialization would also add the Newton gradient's kernels,
      // which the solver does not use (and whose templates MFEM defines in
      // other headers).
      using Convection = mfem::VectorConvectionNLFIntegrator;
      Convection::AddMultPAKernels::Specialization<3, 6, 8>::Add();
      Convection::AddMultPAKernels::Specialization<3, 7, 10>::Add();
      return true;
   }();
   (void)added;
}

} // namespace incns
