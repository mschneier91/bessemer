#include "operators/lean_vector_divergence.hpp"

#include "operators/component_kernels.hpp"

#include "mfem.hpp"
#include "mfem/fem/integ/bilininteg_vecdiv_pa.hpp"

// Device kernels: nvcc must compile this file (src/CMakeLists.txt's LANGUAGE
// CUDA list).
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "lean_vector_divergence.cpp has device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

using namespace mfem;

namespace incns
{

// The 3D divergence kernels of kernel_specializations.cpp (MFEM's, with the
// component loops unrolled, and one component at a time from 7 points per
// direction), reading W, the coefficient and the Jacobians instead of the
// stored Q(d, c) = W coeff adj(J)(d, c). Named namespace: nvcc rejects
// extended lambdas in functions with internal linkage.
namespace leandiv
{

/// Column @p c of W coeff adj(J) at one point: (Q(0, c), Q(1, c), Q(2, c)) of
/// MFEM's PADivergenceSetup3D, with its expressions (J(r, s) = dx_r / dxi_s).
template <typename Tj>
MFEM_HOST_DEVICE inline void AdjColumn(const int c, const Tj& J, const int qx,
                                       const int qy, const int qz, const int e,
                                       const real_t cw, real_t& a0, real_t& a1,
                                       real_t& a2)
{
   if (c == 0)
   {
      const real_t J21 = J(qx, qy, qz, 1, 0, e), J31 = J(qx, qy, qz, 2, 0, e);
      const real_t J22 = J(qx, qy, qz, 1, 1, e), J32 = J(qx, qy, qz, 2, 1, e);
      const real_t J23 = J(qx, qy, qz, 1, 2, e), J33 = J(qx, qy, qz, 2, 2, e);
      a0 = cw * ((J22 * J33) - (J23 * J32)); // A11
      a1 = cw * ((J31 * J23) - (J21 * J33)); // A21
      a2 = cw * ((J21 * J32) - (J31 * J22)); // A31
   }
   else if (c == 1)
   {
      const real_t J11 = J(qx, qy, qz, 0, 0, e), J31 = J(qx, qy, qz, 2, 0, e);
      const real_t J12 = J(qx, qy, qz, 0, 1, e), J32 = J(qx, qy, qz, 2, 1, e);
      const real_t J13 = J(qx, qy, qz, 0, 2, e), J33 = J(qx, qy, qz, 2, 2, e);
      a0 = cw * ((J32 * J13) - (J12 * J33)); // A12
      a1 = cw * ((J11 * J33) - (J13 * J31)); // A22
      a2 = cw * ((J31 * J12) - (J11 * J32)); // A32
   }
   else
   {
      const real_t J11 = J(qx, qy, qz, 0, 0, e), J21 = J(qx, qy, qz, 1, 0, e);
      const real_t J12 = J(qx, qy, qz, 0, 1, e), J22 = J(qx, qy, qz, 1, 1, e);
      const real_t J13 = J(qx, qy, qz, 0, 2, e), J23 = J(qx, qy, qz, 1, 2, e);
      a0 = cw * ((J12 * J23) - (J22 * J13)); // A13
      a1 = cw * ((J21 * J13) - (J11 * J23)); // A23
      a2 = cw * ((J11 * J22) - (J12 * J21)); // A33
   }
}

template <int TR_D1D, int TE_D1D, int Q1D>
void Apply3D(const int NE, const Array<real_t>& b_, const Array<real_t>& g_,
             const Array<real_t>& bt_, const Array<real_t>& w_,
             const real_t coeff, const Vector& j_, const Vector& x_, Vector& y_)
{
   namespace ki = mfem::kernels::internal;
   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

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
               const real_t cw = W(qx, qy, qz) * coeff;
               real_t a[3][3]; // a[c][d] = Q(d, c)
               AdjColumn(0, J, qx, qy, qz, e, cw, a[0][0], a[0][1], a[0][2]);
               AdjColumn(1, J, qx, qy, qz, e, cw, a[1][0], a[1][1], a[1][2]);
               AdjColumn(2, J, qx, qy, qz, e, cw, a[2][0], a[2][1], a[2][2]);
               r0[0][qz][qy][qx] =
                  g1[0][0][qz][qy][qx] * a[0][0] +
                  g1[0][1][qz][qy][qx] * a[0][1] +
                  g1[0][2][qz][qy][qx] * a[0][2] +
                  g1[1][0][qz][qy][qx] * a[1][0] +
                  g1[1][1][qz][qy][qx] * a[1][1] +
                  g1[1][2][qz][qy][qx] * a[1][2] +
                  g1[2][0][qz][qy][qx] * a[2][0] +
                  g1[2][1][qz][qy][qx] * a[2][1] +
                  g1[2][2][qz][qy][qx] * a[2][2];
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
void ApplyTranspose3D(const int NE, const Array<real_t>& bt,
                      const Array<real_t>& gt, const Array<real_t>& b,
                      const Array<real_t>& w_, const real_t coeff,
                      const Vector& j_, const Vector& x_, Vector& y_)
{
   namespace ki = mfem::kernels::internal;
   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

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
               const real_t cw = W(qx, qy, qz) * coeff;
               const real_t r = r1[0][qz][qy][qx];
               real_t a[3][3]; // a[c][d] = Q(d, c)
               AdjColumn(0, J, qx, qy, qz, e, cw, a[0][0], a[0][1], a[0][2]);
               AdjColumn(1, J, qx, qy, qz, e, cw, a[1][0], a[1][1], a[1][2]);
               AdjColumn(2, J, qx, qy, qz, e, cw, a[2][0], a[2][1], a[2][2]);
               g0[0][0][qz][qy][qx] = r * a[0][0];
               g0[0][1][qz][qy][qx] = r * a[0][1];
               g0[0][2][qz][qy][qx] = r * a[0][2];
               g0[1][0][qz][qy][qx] = r * a[1][0];
               g0[1][1][qz][qy][qx] = r * a[1][1];
               g0[1][2][qz][qy][qx] = r * a[1][2];
               g0[2][0][qz][qy][qx] = r * a[2][0];
               g0[2][1][qz][qy][qx] = r * a[2][1];
               g0[2][2][qz][qy][qx] = r * a[2][2];
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

template <int TR_D1D, int TE_D1D, int Q1D>
void Apply3DByComponent(const int NE, const Array<real_t>& b_,
                        const Array<real_t>& g_, const Array<real_t>& bt_,
                        const Array<real_t>& w_, const real_t coeff,
                        const Vector& j_, const Vector& x_, Vector& y_)
{
   namespace ki = mfem::kernels::internal;
   const auto B = b_.Read(), G = g_.Read(), Bt = bt_.Read();
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TR_D1D, TR_D1D, TR_D1D, 3, NE);
   auto Y = Reshape(y_.ReadWrite(), TE_D1D, TE_D1D, TE_D1D, 1, NE);

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
                  real_t a0, a1, a2;
                  AdjColumn(c, J, qx, qy, qz, e, W(qx, qy, qz) * coeff, a0, a1,
                            a2);
                  real_t r = g1[0][0][qz][qy][qx] * a0;
                  if (c > 0) { r = r0[0][qz][qy][qx] + r; }
                  r += g1[0][1][qz][qy][qx] * a1;
                  r += g1[0][2][qz][qy][qx] * a2;
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
void ApplyTranspose3DByComponent(const int NE, const Array<real_t>& bt,
                                 const Array<real_t>& gt,
                                 const Array<real_t>& b,
                                 const Array<real_t>& w_, const real_t coeff,
                                 const Vector& j_, const Vector& x_,
                                 Vector& y_)
{
   namespace ki = mfem::kernels::internal;
   const auto Bt = bt.Read(), Gt = gt.Read(), B = b.Read();
   const auto W = Reshape(w_.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(j_.Read(), Q1D, Q1D, Q1D, 3, 3, NE);
   const auto X = Reshape(x_.Read(), TE_D1D, TE_D1D, TE_D1D, 1, NE);
   auto Y = Reshape(y_.ReadWrite(), TR_D1D, TR_D1D, TR_D1D, 3, NE);

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
                  real_t a0, a1, a2;
                  AdjColumn(c, J, qx, qy, qz, e, W(qx, qy, qz) * coeff, a0, a1,
                            a2);
                  g0[0][0][qz][qy][qx] = r * a0;
                  g0[0][1][qz][qy][qx] = r * a1;
                  g0[0][2][qz][qy][qx] = r * a2;
               }
            }
         }
         kernels3d::GradComponent<true>(TR_D1D, Q1D, smem, sB, sG, g0, g1);
         ki::WriteDofs3d(e, TR_D1D, 0, c, g1, Y);
      }
   });
}

} // namespace leandiv

namespace
{

using Kernel = void (*)(const int, const Array<real_t>&, const Array<real_t>&,
                        const Array<real_t>&, const Array<real_t>&,
                        const real_t, const Vector&, const Vector&, Vector&);

// The kernels for velocity order K (K + 1 trial and K test dofs, K + 2
// points per direction), as kernel_specializations.cpp picks them.
template <int K>
Kernel Pick(const bool transpose)
{
   if constexpr(K + 2 >= 7)
   {
      return transpose
             ? leandiv::ApplyTranspose3DByComponent < K + 1, K, K + 2 >
             : leandiv::Apply3DByComponent < K + 1, K, K + 2 >;
   }
   else
   {
      return transpose ? leandiv::ApplyTranspose3D < K + 1, K, K + 2 >
             : leandiv::Apply3D < K + 1, K, K + 2 >;
   }
}

Kernel Select(const int k, const bool transpose)
{
   switch (k)
   {
      case 2: return Pick<2>(transpose);
      case 3: return Pick<3>(transpose);
      case 4: return Pick<4>(transpose);
      case 5: return Pick<5>(transpose);
      case 6: return Pick<6>(transpose);
      case 7: return Pick<7>(transpose);
      default: return nullptr;
   }
}

} // namespace

void LeanVectorDivergenceIntegrator::AssemblePA(
   const FiniteElementSpace& trial_fes, const FiniteElementSpace& test_fes)
{
   lean_ = false;
   Mesh* mesh = trial_fes.GetMesh();
   const FiniteElement& trial_fe = *trial_fes.GetTypicalFE();
   const FiniteElement& test_fe = *test_fes.GetTypicalFE();
   const int k = trial_fe.GetOrder();
   const auto* cq = dynamic_cast<const ConstantCoefficient*>(Q);
   bool ok = mesh->Dimension() == 3 && mesh->SpaceDimension() == 3 &&
             IntRule != nullptr && (Q == nullptr || cq != nullptr) &&
             trial_fes.GetVDim() == 3 && test_fes.GetVDim() == 1 &&
             trial_fes.GetOrdering() == Ordering::byNODES &&
             UsesTensorBasis(trial_fes) && UsesTensorBasis(test_fes) &&
             test_fe.GetOrder() == k - 1 && Select(k, false) != nullptr;
   if (ok)
   {
      trial_maps_ = &trial_fe.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
      test_maps_ = &test_fe.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
      ok = trial_maps_->ndof == k + 1 && test_maps_->ndof == k &&
           trial_maps_->nqpt == k + 2 && test_maps_->nqpt == k + 2;
   }
   if (!ok)
   {
      mfem::VectorDivergenceIntegrator::AssemblePA(trial_fes, test_fes);
      return;
   }
   ne_ = trial_fes.GetNE();
   k_ = k;
   coeff_ = cq ? cq->constant : 1.0;
   ir_ = IntRule;
   // The mesh's cache, the entry the grad-div form uses (same rule, flags and
   // memory type): no second copy of the Jacobians.
   geom_ = mesh->GetGeometricFactors(*ir_, GeometricFactors::JACOBIANS,
                                     Device::GetDeviceMemoryType());
   lean_ = true;
}

void LeanVectorDivergenceIntegrator::AddMultPA(const Vector& x, Vector& y) const
{
   if (!lean_)
   {
      mfem::VectorDivergenceIntegrator::AddMultPA(x, y);
      return;
   }
   Select(k_, false)(ne_, trial_maps_->B, trial_maps_->G, test_maps_->Bt,
                     ir_->GetWeights(), coeff_, geom_->J, x, y);
}

void LeanVectorDivergenceIntegrator::AddMultTransposePA(const Vector& x,
      Vector& y) const
{
   if (!lean_)
   {
      mfem::VectorDivergenceIntegrator::AddMultTransposePA(x, y);
      return;
   }
   Select(k_, true)(ne_, trial_maps_->Bt, trial_maps_->Gt, test_maps_->B,
                    ir_->GetWeights(), coeff_, geom_->J, x, y);
}

} // namespace incns
