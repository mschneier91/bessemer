/**
 * @file hyperelastic_operator.cpp
 * @brief HyperelasticOperator: the fused sum-factorized residual, energy and
 *        tangent kernels (2D and 3D, 2 to 8 points per direction).
 */
#include "solid/hyperelastic_operator.hpp"

// This TU defines device kernels (mfem::forall_2D). A host compiler would
// build them as host loops over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "hyperelastic_operator.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

#include <type_traits>
#include <utility>

namespace incns
{

using namespace mfem;
using kernels::internal::SetMaxOf;

namespace
{

/// J^-1 and det J of a column-major DIM x DIM Jacobian.
template <int DIM>
MFEM_HOST_DEVICE inline real_t Invert(const real_t* J, real_t* Ji)
{
   if constexpr(DIM == 2)
   {
      const real_t det = J[0] * J[3] - J[1] * J[2];
      Ji[0] = J[3] / det; Ji[1] = -J[1] / det;
      Ji[2] = -J[2] / det; Ji[3] = J[0] / det;
      return det;
   }
   else
   {
      auto a = [&](int i, int j) { return J[i + 3 * j]; };
      const real_t c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
      const real_t c01 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
      const real_t c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
      const real_t det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
      Ji[0 + 3 * 0] = c00 / det;
      Ji[0 + 3 * 1] = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) / det;
      Ji[0 + 3 * 2] = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) / det;
      Ji[1 + 3 * 0] = c01 / det;
      Ji[1 + 3 * 1] = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) / det;
      Ji[1 + 3 * 2] = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) / det;
      Ji[2 + 3 * 0] = c02 / det;
      Ji[2 + 3 * 1] = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) / det;
      Ji[2 + 3 * 2] = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) / det;
      return det;
   }
}

/// J^-1 (nq, dim, dim, ne) and w det J (nq, ne) from the mesh's Jacobians.
template <int DIM>
void SetupGeometry(int nq, int ne, const IntegrationRule& ir, const Vector& jac,
                   Vector& jinv, Vector& wdet)
{
   const auto W = Reshape(ir.GetWeights().Read(), nq);
   const auto J = Reshape(jac.Read(), nq, DIM, DIM, ne);
   auto JI = Reshape(jinv.Write(), nq, DIM, DIM, ne);
   auto WD = Reshape(wdet.Write(), nq, ne);
   mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % nq, e = i / nq;
      real_t Jm[DIM * DIM] = {}, Ji[DIM * DIM] = {};
      for (int b = 0; b < DIM; b++)
      {
         for (int a = 0; a < DIM; a++) { Jm[a + DIM * b] = J(q, a, b, e); }
      }
      const real_t det = Invert<DIM>(Jm, Ji);
      for (int d = 0; d < DIM; d++)
      {
         for (int k = 0; k < DIM; k++) { JI(q, k, d, e) = Ji[k + DIM * d]; }
      }
      WD(q, e) = W(q) * det;
   });
}

// ---------------------------------------------------------------------------
// The kernels. Per element: the reference gradients r1[c][k] of every
// component (k the reference direction), the material law at every point with
// F = I + sum_k r1[c][k] J^-1(k, d), the flux back in reference directions,
// r0[c][k] = w det J sum_d P(c, d) J^-1(k, d), and the transposed gradient
// added into the output. N = dofs = points per direction.
//
// STORE: 0 nothing, 1 F, 2 F and S, into (points, STORE dim^2, ne).
// ENERGY: sum over the thread's points of W w det J into (N, N, ne) -- per
// point in 2D, per z-column in 3D.
// The tangent's MODE: 0 F and S stored, 1 F stored, 2 the displacement's
// E-vector (F from one more gradient pass); S recomputed unless stored.
// ---------------------------------------------------------------------------

template <class MAT, int N, bool RESIDUAL, int STORE, bool ENERGY>
void Pointwise2D(int NE, const Array<real_t>& b, const Array<real_t>& g,
                 const Vector& jinv, const Vector& wdet, const Vector& lam,
                 const Vector& mu, const Vector& x, Vector* y, Vector* fs,
                 Vector* we)
{
   constexpr int DIM = 2, VDIM = 2, D2 = DIM * DIM;
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto XE = Reshape(x.Read(), N, N, VDIM, NE);
   auto YE = Reshape(RESIDUAL ? y->ReadWrite() : nullptr, N, N, VDIM, NE);
   auto FS = Reshape(STORE ? fs->Write() : nullptr, N, N,
                     STORE > 0 ? STORE* D2 : 1, NE);
   auto WE = Reshape(ENERGY ? we->Write() : nullptr, N, N, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N);
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(N, N, B, sB);
      kernels::internal::LoadMatrix(N, N, G, sG);
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs2d(e, N, c, XE, r0);
         kernels::internal::Grad2d(N, N, smem, sB, sG, r0, r1, c);
      }
      const real_t lame = L[e], mue = MU[e];
      MFEM_FOREACH_THREAD_DIRECT(qy, y, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, N)
         {
            real_t F[D2] = {}, S[D2] = {}, P[D2] = {};
            for (int d = 0; d < DIM; d++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  real_t h = 0.0;
                  for (int k = 0; k < DIM; k++) { h += r1[c][k][qy][qx] * JI(qx, qy, k, d, e); }
                  F[c + DIM * d] = h + (c == d ? 1.0 : 0.0);
               }
            }
            MAT::template Stress<DIM>(F, lame, mue, S, P);
            if constexpr(STORE >= 1)
            {
               for (int i = 0; i < D2; i++) { FS(qx, qy, i, e) = F[i]; }
            }
            if constexpr(STORE == 2)
            {
               for (int i = 0; i < D2; i++) { FS(qx, qy, D2 + i, e) = S[i]; }
            }
            if constexpr(ENERGY)
            {
               WE(qx, qy, e) = WD(qx, qy, e) * MAT::template Energy<DIM>(F, lame, mue);
            }
            if constexpr(RESIDUAL)
            {
               const real_t w = WD(qx, qy, e);
               for (int c = 0; c < DIM; c++)
               {
                  for (int k = 0; k < DIM; k++)
                  {
                     real_t q = 0.0;
                     for (int d = 0; d < DIM; d++) { q += P[c + DIM * d] * JI(qx, qy, k, d, e); }
                     r0[c][k][qy][qx] = w * q;
                  }
               }
            }
         }
      }
      if constexpr(RESIDUAL)
      {
         MFEM_SYNC_THREAD;
         for (int c = 0; c < VDIM; ++c)
         {
            kernels::internal::GradTranspose2d(N, N, smem, sB, sG, r0, r1, c);
            kernels::internal::WriteDofs2d(e, N, c, c, r1, YE);
            MFEM_SYNC_THREAD; // smem reused by the next GradTranspose
         }
      }
   });
}

template <class MAT, int N, int MODE>
void Tangent2D(int NE, const Array<real_t>& b, const Array<real_t>& g,
               const Vector& jinv, const Vector& wdet, const Vector& lam,
               const Vector& mu, const Vector& lin, const Vector& x, Vector& y)
{
   constexpr int DIM = 2, VDIM = 2, D2 = DIM * DIM;
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto FS = Reshape(lin.Read(), N, N,
                           MODE == 0 ? 2 * D2 : (MODE == 1 ? D2 : VDIM), NE);
   const auto UL = Reshape(lin.Read(), N, N, VDIM, NE);
   const auto XE = Reshape(x.Read(), N, N, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), N, N, VDIM, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N);
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1, r2;
      kernels::internal::LoadMatrix(N, N, B, sB);
      kernels::internal::LoadMatrix(N, N, G, sG);
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs2d(e, N, c, XE, r0);
         kernels::internal::Grad2d(N, N, smem, sB, sG, r0, r1, c);
      }
      if constexpr(MODE == 2)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            kernels::internal::LoadDofs2d(e, N, c, UL, r0);
            kernels::internal::Grad2d(N, N, smem, sB, sG, r0, r2, c);
         }
      }
      const real_t lame = L[e], mue = MU[e];
      MFEM_FOREACH_THREAD_DIRECT(qy, y, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, N)
         {
            real_t F[D2] = {}, S[D2] = {}, dF[D2] = {}, dP[D2] = {};
            for (int d = 0; d < DIM; d++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  real_t h = 0.0, hl = 0.0;
                  for (int k = 0; k < DIM; k++)
                  {
                     h += r1[c][k][qy][qx] * JI(qx, qy, k, d, e);
                     if constexpr(MODE == 2) { hl += r2[c][k][qy][qx] * JI(qx, qy, k, d, e); }
                  }
                  dF[c + DIM * d] = h;
                  if constexpr(MODE == 2) { F[c + DIM * d] = hl + (c == d ? 1.0 : 0.0); }
               }
            }
            if constexpr(MODE <= 1)
            {
               for (int i = 0; i < D2; i++) { F[i] = FS(qx, qy, i, e); }
            }
            if constexpr(MODE == 0)
            {
               for (int i = 0; i < D2; i++) { S[i] = FS(qx, qy, D2 + i, e); }
            }
            else
            {
               real_t P[D2] = {};
               MAT::template Stress<DIM>(F, lame, mue, S, P);
            }
            MAT::template Tangent<DIM>(F, S, dF, lame, mue, dP);
            const real_t w = WD(qx, qy, e);
            for (int c = 0; c < DIM; c++)
            {
               for (int k = 0; k < DIM; k++)
               {
                  real_t q = 0.0;
                  for (int d = 0; d < DIM; d++) { q += dP[c + DIM * d] * JI(qx, qy, k, d, e); }
                  r0[c][k][qy][qx] = w * q;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::GradTranspose2d(N, N, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs2d(e, N, c, c, r1, YE);
         MFEM_SYNC_THREAD;
      }
   });
}

template <class MAT, int N, bool RESIDUAL, int STORE, bool ENERGY>
void Pointwise3D(int NE, const Array<real_t>& b, const Array<real_t>& g,
                 const Vector& jinv, const Vector& wdet, const Vector& lam,
                 const Vector& mu, const Vector& x, Vector* y, Vector* fs,
                 Vector* we)
{
   constexpr int DIM = 3, VDIM = 3, D2 = DIM * DIM;
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto XE = Reshape(x.Read(), N, N, N, VDIM, NE);
   auto YE = Reshape(RESIDUAL ? y->ReadWrite() : nullptr, N, N, N, VDIM, NE);
   auto FS = Reshape(STORE ? fs->Write() : nullptr, N, N, N,
                     STORE > 0 ? STORE* D2 : 1,
                     NE);
   auto WE = Reshape(ENERGY ? we->Write() : nullptr, N, N, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N);
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(N, N, B, sB);
      kernels::internal::LoadMatrix(N, N, G, sG);
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs3d(e, N, c, XE, r0);
         kernels::internal::Grad3d(N, N, smem, sB, sG, r0, r1, c);
      }
      const real_t lame = L[e], mue = MU[e];
      MFEM_FOREACH_THREAD_DIRECT(qy, y, N)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, N)
         {
            real_t esum = 0.0;
            for (int qz = 0; qz < N; ++qz)
            {
               real_t F[D2] = {}, S[D2] = {}, P[D2] = {};
               for (int d = 0; d < DIM; d++)
               {
                  for (int c = 0; c < DIM; c++)
                  {
                     real_t h = 0.0;
                     for (int k = 0; k < DIM; k++)
                     {
                        h += r1[c][k][qz][qy][qx] * JI(qx, qy, qz, k, d, e);
                     }
                     F[c + DIM * d] = h + (c == d ? 1.0 : 0.0);
                  }
               }
               MAT::template Stress<DIM>(F, lame, mue, S, P);
               if constexpr(STORE >= 1)
               {
                  for (int i = 0; i < D2; i++) { FS(qx, qy, qz, i, e) = F[i]; }
               }
               if constexpr(STORE == 2)
               {
                  for (int i = 0; i < D2; i++) { FS(qx, qy, qz, D2 + i, e) = S[i]; }
               }
               if constexpr(ENERGY)
               {
                  esum += WD(qx, qy, qz, e) * MAT::template Energy<DIM>(F, lame, mue);
               }
               if constexpr(RESIDUAL)
               {
                  const real_t w = WD(qx, qy, qz, e);
                  for (int c = 0; c < DIM; c++)
                  {
                     for (int k = 0; k < DIM; k++)
                     {
                        real_t q = 0.0;
                        for (int d = 0; d < DIM; d++)
                        {
                           q += P[c + DIM * d] * JI(qx, qy, qz, k, d, e);
                        }
                        r0[c][k][qz][qy][qx] = w * q;
                     }
                  }
               }
            }
            if constexpr(ENERGY) { WE(qx, qy, e) = esum; }
         }
      }
      if constexpr(RESIDUAL)
      {
         MFEM_SYNC_THREAD;
         for (int c = 0; c < VDIM; ++c)
         {
            kernels::internal::GradTranspose3d(N, N, smem, sB, sG, r0, r1, c);
            kernels::internal::WriteDofs3d(e, N, c, c, r1, YE);
            MFEM_SYNC_THREAD; // smem reused by the next GradTranspose
         }
      }
   });
}

template <class MAT, int N, int MODE>
void Tangent3D(int NE, const Array<real_t>& b, const Array<real_t>& g,
               const Vector& jinv, const Vector& wdet, const Vector& lam,
               const Vector& mu, const Vector& lin, const Vector& x, Vector& y)
{
   constexpr int DIM = 3, VDIM = 3, D2 = DIM * DIM;
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto FS = Reshape(lin.Read(), N, N, N,
                           MODE == 0 ? 2 * D2 : (MODE == 1 ? D2 : VDIM),
                           NE);
   const auto UL = Reshape(lin.Read(), N, N, N, VDIM, NE);
   const auto XE = Reshape(x.Read(), N, N, N, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), N, N, N, VDIM, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N);
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1, r2;
      kernels::internal::LoadMatrix(N, N, B, sB);
      kernels::internal::LoadMatrix(N, N, G, sG);
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs3d(e, N, c, XE, r0);
         kernels::internal::Grad3d(N, N, smem, sB, sG, r0, r1, c);
      }
      if constexpr(MODE == 2)
      {
         for (int c = 0; c < VDIM; ++c)
         {
            kernels::internal::LoadDofs3d(e, N, c, UL, r0);
            kernels::internal::Grad3d(N, N, smem, sB, sG, r0, r2, c);
         }
      }
      const real_t lame = L[e], mue = MU[e];
      for (int qz = 0; qz < N; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, N)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, N)
            {
               real_t F[D2] = {}, S[D2] = {}, dF[D2] = {}, dP[D2] = {};
               for (int d = 0; d < DIM; d++)
               {
                  for (int c = 0; c < DIM; c++)
                  {
                     real_t h = 0.0, hl = 0.0;
                     for (int k = 0; k < DIM; k++)
                     {
                        const real_t ji = JI(qx, qy, qz, k, d, e);
                        h += r1[c][k][qz][qy][qx] * ji;
                        if constexpr(MODE == 2) { hl += r2[c][k][qz][qy][qx] * ji; }
                     }
                     dF[c + DIM * d] = h;
                     if constexpr(MODE == 2) { F[c + DIM * d] = hl + (c == d ? 1.0 : 0.0); }
                  }
               }
               if constexpr(MODE <= 1)
               {
                  for (int i = 0; i < D2; i++) { F[i] = FS(qx, qy, qz, i, e); }
               }
               if constexpr(MODE == 0)
               {
                  for (int i = 0; i < D2; i++) { S[i] = FS(qx, qy, qz, D2 + i, e); }
               }
               else
               {
                  real_t P[D2] = {};
                  MAT::template Stress<DIM>(F, lame, mue, S, P);
               }
               MAT::template Tangent<DIM>(F, S, dF, lame, mue, dP);
               const real_t w = WD(qx, qy, qz, e);
               for (int c = 0; c < DIM; c++)
               {
                  for (int k = 0; k < DIM; k++)
                  {
                     real_t q = 0.0;
                     for (int d = 0; d < DIM; d++)
                     {
                        q += dP[c + DIM * d] * JI(qx, qy, qz, k, d, e);
                     }
                     r0[c][k][qz][qy][qx] = w * q;
                  }
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::GradTranspose3d(N, N, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs3d(e, N, c, c, r1, YE);
         MFEM_SYNC_THREAD;
      }
   });
}

/// f(std::integral_constant<int, N>) for N = n in 2..8 (compile-time sizes).
template <class F>
void WithPoints(int n, F&& f)
{
   switch (n)
   {
      case 2: f(std::integral_constant<int, 2> {}); break;
      case 3: f(std::integral_constant<int, 3> {}); break;
      case 4: f(std::integral_constant<int, 4> {}); break;
      case 5: f(std::integral_constant<int, 5> {}); break;
      case 6: f(std::integral_constant<int, 6> {}); break;
      case 7: f(std::integral_constant<int, 7> {}); break;
      case 8: f(std::integral_constant<int, 8> {}); break;
      default:
         MFEM_ABORT("HyperelasticOperator: " << n << " points per direction "
                    "(orders 1 to 7 are compiled)");
   }
}

/// @return What the linearization stores per point, in dim^2 blocks: 2 F
///         and S, 1 F, 0 nothing (the displacement's E-vector instead).
int StoreFor(TangentStorage s)
{
   return s == TangentStorage::StressAndDeformation ? 2 :
          (s == TangentStorage::Deformation ? 1 : 0);
}

} // namespace

class HyperelasticOperator::Tangent : public Operator
{
public:
   explicit Tangent(const HyperelasticOperator& op)
      : Operator(op.Height()), op_(op) {}

   void Mult(const Vector& w, Vector& y) const override
   {
      op_.ToElements(w);
      op_.ye_ = 0.0;
      const auto& m = *op_.maps_;
      const int mode = 2 - StoreFor(op_.storage_); // 0 F,S; 1 F; 2 displacement
      WithPoints(op_.n1d_, [&](auto nc)
      {
         constexpr int N = decltype(nc)::value;
         auto run = [&](auto mc)
         {
            constexpr int MODE = decltype(mc)::value;
            if (op_.dim_ == 2)
            {
               Tangent2D<StVenantKirchhoff, N, MODE>(op_.ne_, m.B, m.G, op_.jinv_,
                                                     op_.wdet_, op_.lambda_, op_.mu_,
                                                     op_.lin_, op_.ue_, op_.ye_);
            }
            else
            {
               Tangent3D<StVenantKirchhoff, N, MODE>(op_.ne_, m.B, m.G, op_.jinv_,
                                                     op_.wdet_, op_.lambda_, op_.mu_,
                                                     op_.lin_, op_.ue_, op_.ye_);
            }
         };
         if (mode == 0) { run(std::integral_constant<int, 0> {}); }
         else if (mode == 1) { run(std::integral_constant<int, 1> {}); }
         else { run(std::integral_constant<int, 2> {}); }
      });
      op_.FromElements(y);
   }

private:
   const HyperelasticOperator& op_;
};

HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace& fes,
      const IntegrationRule& ir,
      HyperelasticMaterial material,
      const Vector& lambda, const Vector& mu,
      TangentStorage storage,
      ElementScratch* scratch)
   : Operator(fes.GetTrueVSize()), fes_(fes), ir_(ir), material_(material),
     storage_(storage)
{
   ParMesh& mesh = *fes.GetParMesh();
   dim_ = mesh.Dimension();
   ne_ = mesh.GetNE();
   MFEM_VERIFY(material_ == HyperelasticMaterial::StVenantKirchhoff,
               "HyperelasticOperator: unknown material");
   MFEM_VERIFY(fes.GetVDim() == dim_ && fes.GetOrdering() == Ordering::byNODES &&
               mesh.SpaceDimension() == dim_,
               "HyperelasticOperator: a vector space (vdim = dim = sdim) ordered by "
               "nodes");
   MFEM_VERIFY(lambda.Size() == ne_ && mu.Size() == ne_,
               "HyperelasticOperator: lambda and mu per local element");
   const FiniteElement& el = *fes.GetTypicalFE();
   MFEM_VERIFY(dynamic_cast<const TensorBasisElement*>(&el),
               "HyperelasticOperator: tensor-product elements (quads, hexes)");
   maps_ = &el.GetDofToQuad(ir_, DofToQuad::TENSOR);
   n1d_ = maps_->ndof;
   MFEM_VERIFY(maps_->nqpt == n1d_, "HyperelasticOperator: the rule needs as many "
               "points per direction as dofs (" << n1d_ << "), has " << maps_->nqpt);
   MFEM_VERIFY(n1d_ >= 2 && n1d_ <= 8, "HyperelasticOperator: orders 1 to 7");
   restriction_ = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   lambda_ = lambda;
   mu_ = mu;
   lambda_.UseDevice(true);
   mu_.UseDevice(true);

   const int nq = ir_.GetNPoints();
   jinv_.SetSize(nq * dim_ * dim_ * ne_);
   wdet_.SetSize(nq * ne_);
   jinv_.UseDevice(true);
   wdet_.UseDevice(true);
   if (ne_ > 0)
   {
      const GeometricFactors* geom =
         mesh.GetGeometricFactors(ir_, GeometricFactors::JACOBIANS);
      if (dim_ == 2) { SetupGeometry<2>(nq, ne_, ir_, geom->J, jinv_, wdet_); }
      else { SetupGeometry<3>(nq, ne_, ir_, geom->J, jinv_, wdet_); }
   }
   const int nstore = StoreFor(storage_);
   lin_.SetSize(nstore > 0 ? nq * nstore * dim_ * dim_ * ne_ :
                restriction_->Height());
   lin_.UseDevice(true);

   const int ne_size = restriction_->Height(), nl = fes.GetVSize();
   if (scratch && scratch->Size() >= ne_size)
   {
      scratch->Lend(0, ne_size, ue_);
      scratch->Lend(1, ne_size, ye_);
      own_e_ = false;
   }
   else
   {
      for (Vector* v : {&ue_, &ye_}) { v->SetSize(ne_size); v->UseDevice(true); }
   }
   if (!fes.GetProlongationMatrix())
   {
      own_l_ = false; // true dofs are the local dofs: no L-vectors at all
   }
   else if (scratch && scratch->LocalSize() >= nl)
   {
      scratch->LendLocal(0, nl, ul_);
      scratch->LendLocal(1, nl, yl_);
      own_l_ = false;
   }
   else
   {
      for (Vector* v : {&ul_, &yl_}) { v->SetSize(nl); v->UseDevice(true); }
   }
   tangent_ = std::make_unique<Tangent>(*this);
}

HyperelasticOperator::~HyperelasticOperator() = default;

std::size_t HyperelasticOperator::MemoryBytes() const
{
   std::size_t n = jinv_.Size() + wdet_.Size() + lin_.Size() + qe_.Size() +
                   lambda_.Size() + mu_.Size();
   if (own_e_) { n += ue_.Size() + ye_.Size(); }
   if (own_l_) { n += ul_.Size() + yl_.Size(); }
   return n * sizeof(real_t);
}

void HyperelasticOperator::ToElements(const Vector& u) const
{
   const Operator* P = fes_.GetProlongationMatrix();
   if (P)
   {
      P->Mult(u, ul_);
      restriction_->Mult(ul_, ue_);
   }
   else { restriction_->Mult(u, ue_); }
}

void HyperelasticOperator::FromElements(Vector& y) const
{
   const Operator* P = fes_.GetProlongationMatrix();
   y.SetSize(Height());
   if (P)
   {
      restriction_->MultTranspose(ye_, yl_);
      P->MultTranspose(yl_, y);
   }
   else { restriction_->MultTranspose(ye_, y); }
}

void HyperelasticOperator::Mult(const Vector& u, Vector& r) const
{
   ToElements(u);
   ye_ = 0.0;
   WithPoints(n1d_, [&](auto nc)
   {
      constexpr int N = decltype(nc)::value;
      if (dim_ == 2)
      {
         Pointwise2D<StVenantKirchhoff, N, true, 0, false>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, &ye_, nullptr,
            nullptr);
      }
      else
      {
         Pointwise3D<StVenantKirchhoff, N, true, 0, false>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, &ye_, nullptr,
            nullptr);
      }
   });
   FromElements(r);
}

Operator& HyperelasticOperator::GetGradient(const Vector& u) const
{
   ToElements(u);
   const int nstore = StoreFor(storage_);
   if (nstore == 0)
   {
      lin_ = ue_; // the displacement's E-vector: the tangent recomputes F, S
      return *tangent_;
   }
   WithPoints(n1d_, [&](auto nc)
   {
      constexpr int N = decltype(nc)::value;
      auto run = [&](auto sc)
      {
         constexpr int STORE = decltype(sc)::value;
         if (dim_ == 2)
         {
            Pointwise2D<StVenantKirchhoff, N, false, STORE, false>(
               ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, nullptr, &lin_,
               nullptr);
         }
         else
         {
            Pointwise3D<StVenantKirchhoff, N, false, STORE, false>(
               ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, nullptr, &lin_,
               nullptr);
         }
      };
      if (nstore == 1) { run(std::integral_constant<int, 1> {}); }
      else { run(std::integral_constant<int, 2> {}); }
   });
   return *tangent_;
}

double HyperelasticOperator::Energy(const Vector& u) const
{
   // Partial sums per point (2D) or per z-column (3D): N^2 per element,
   // allocated on the first call.
   if (qe_.Size() != n1d_ * n1d_ * ne_)
   {
      qe_.SetSize(n1d_ * n1d_ * ne_);
      qe_.UseDevice(true);
   }
   ToElements(u);
   WithPoints(n1d_, [&](auto nc)
   {
      constexpr int N = decltype(nc)::value;
      if (dim_ == 2)
      {
         Pointwise2D<StVenantKirchhoff, N, false, 0, true>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, nullptr, nullptr,
            &qe_);
      }
      else
      {
         Pointwise3D<StVenantKirchhoff, N, false, 0, true>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, ue_, nullptr, nullptr,
            &qe_);
      }
   });
   // Summed on the host in a fixed order: deterministic.
   const real_t* q = qe_.HostRead();
   double local = 0.0;
   for (int i = 0; i < qe_.Size(); i++) { local += q[i]; }
   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, fes_.GetComm());
   return global;
}

} // namespace incns
