/**
 * @file hyperelastic_operator.cpp
 * @brief HyperelasticOperator: the fused sum-factorized residual, energy and
 *        tangent kernels (2D and 3D, 2 to 8 points per direction; 3D on a
 *        device by lines).
 */
#include "solid/hyperelastic_operator.hpp"

#include "operators/true_dof_exchange.hpp"

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
#include <vector>

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
   constexpr int DIM = 2, VDIM = 2, NF = DIM * DIM; // entries of F
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto XE = Reshape(x.Read(), N, N, VDIM, NE);
   auto YE = Reshape(RESIDUAL ? y->ReadWrite() : nullptr, N, N, VDIM, NE);
   auto FS = Reshape(STORE ? fs->Write() : nullptr, N, N,
                     STORE > 0 ? STORE* NF : 1, NE);
   auto WE = Reshape(ENERGY ? we->Write() : nullptr, N, N, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N), D2 = DIM * DIM;
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      // nvcc rejects capturing an outer variable first inside an if
      // constexpr: these are captured here, and D2 is the lambda's own.
      MFEM_CONTRACT_VAR(YE);
      MFEM_CONTRACT_VAR(FS);
      MFEM_CONTRACT_VAR(WE);
      MFEM_CONTRACT_VAR(WD);
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
   constexpr int DIM = 2, VDIM = 2, NF = DIM * DIM; // entries of F
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto FS = Reshape(lin.Read(), N, N,
                           MODE == 0 ? 2 * NF : (MODE == 1 ? NF : VDIM), NE);
   const auto UL = Reshape(lin.Read(), N, N, VDIM, NE);
   const auto XE = Reshape(x.Read(), N, N, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), N, N, VDIM, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N), D2 = DIM * DIM;
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      // nvcc rejects capturing an outer variable first inside an if
      // constexpr: these are captured here, and D2 is the lambda's own.
      MFEM_CONTRACT_VAR(FS);
      MFEM_CONTRACT_VAR(UL);
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
   constexpr int DIM = 3, VDIM = 3, NF = DIM * DIM; // entries of F
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto XE = Reshape(x.Read(), N, N, N, VDIM, NE);
   auto YE = Reshape(RESIDUAL ? y->ReadWrite() : nullptr, N, N, N, VDIM, NE);
   auto FS = Reshape(STORE ? fs->Write() : nullptr, N, N, N,
                     STORE > 0 ? STORE* NF : 1,
                     NE);
   auto WE = Reshape(ENERGY ? we->Write() : nullptr, N, N, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N), D2 = DIM * DIM;
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      // nvcc rejects capturing an outer variable first inside an if
      // constexpr: these are captured here, and D2 is the lambda's own.
      MFEM_CONTRACT_VAR(YE);
      MFEM_CONTRACT_VAR(FS);
      MFEM_CONTRACT_VAR(WE);
      MFEM_CONTRACT_VAR(WD);
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
   constexpr int DIM = 3, VDIM = 3, NF = DIM * DIM; // entries of F
   const auto B = b.Read(), G = g.Read();
   const auto JI = Reshape(jinv.Read(), N, N, N, DIM, DIM, NE);
   const auto WD = Reshape(wdet.Read(), N, N, N, NE);
   const auto L = lam.Read(), MU = mu.Read();
   const auto FS = Reshape(lin.Read(), N, N, N,
                           MODE == 0 ? 2 * NF : (MODE == 1 ? NF : VDIM),
                           NE);
   const auto UL = Reshape(lin.Read(), N, N, N, VDIM, NE);
   const auto XE = Reshape(x.Read(), N, N, N, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), N, N, N, VDIM, NE);

   mfem::forall_2D<N* N>(NE, N, N, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = SetMaxOf(N), D2 = DIM * DIM;
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], smem[MQ1][MQ1];
      // nvcc rejects capturing an outer variable first inside an if
      // constexpr: these are captured here, and D2 is the lambda's own.
      MFEM_CONTRACT_VAR(FS);
      MFEM_CONTRACT_VAR(UL);
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

// ---------------------------------------------------------------------------
// 3D on a device, by lines (the performance step, 2026-10-21). The kernels
// above keep the reference gradients of all three components in MFEM's
// register tensors, 9 N values per thread and as many for the fluxes; nvcc
// puts them in local memory, and at Q6 on an H100 that overflowed L1 and L2:
// the tangent moved 8.7 GB of DRAM per apply for 2.3 GB of data (its writes
// 18 times its output), at 25 % occupancy (Nsight Compute, job 49569116).
// Here, as in the momentum apply by lines (operators/lean_vector_diffusion.cpp,
// change 56): an N x N block per element; in each stage of the sum
// factorization a thread owns a line of outputs; the 1D matrices are kernel
// parameters (constant-bank operands); and the reference gradients, then in
// place the fluxes, are kept in shared memory, 9 stacks of N z-planes (18 when
// the tangent forms F as well, MODE 2; dynamic shared memory, up to 64 KB per
// block). On a device only: the host runs the kernels above, and the CPU
// build does not change.
//
// Round 2 (2026-10-21; human: "Keep optimizing"). The gather and scatter are
// fused: the kernels read the L-vector through the restriction's gather map
// and write a dof one element holds straight into the output L-vector, a
// shared dof's contribution into its slot, which ScatterSlots sums in element
// order (no E-vectors, no atomics). At Q6 that cut the restriction's 0.47 ms
// per apply to 0.16 ms, but the index loads cost 12-22 registers and the
// tangent fell from 6 blocks per SM to 4 (job 49571253). The sum
// factorization is therefore collocated: B along each direction to the values
// at the points, then D = G B^-1 there, 6 N^4 multiply-adds per component and
// pass against 8 N^4. Q6 residual 206 registers -> 128, both kernels back at
// 6 blocks per SM (job 49572810); a register cap instead gave the same times
// (job 49573235), so the gain is mostly occupancy.
// ---------------------------------------------------------------------------
namespace lines
{

template <int N>
struct Basis
{
   real_t B[N][N]; ///< B[q][d] = B(q, d), as maps->B.
   real_t D[N][N]; ///< D = G B^-1: the derivative at the points from values there.
};

template <int N>
Basis<N> MakeBasis(const Array<real_t>& b, const Array<real_t>& g)
{
   const real_t* hb = b.HostRead();
   const real_t* hg = g.HostRead();
   DenseMatrix Bm(N), Gm(N), Bi(N), Dm(N);
   for (int q = 0; q < N; ++q)
   {
      for (int d = 0; d < N; ++d)
      {
         Bm(q, d) = hb[q + N * d];
         Gm(q, d) = hg[q + N * d];
      }
   }
   DenseMatrixInverse(Bm).GetInverseMatrix(Bi); // dofs and points unisolvent
   Mult(Gm, Bi, Dm);
   Basis<N> bs;
   for (int q = 0; q < N; ++q)
   {
      for (int r = 0; r < N; ++r)
      {
         bs.B[q][r] = Bm(q, r);
         bs.D[q][r] = Dm(q, r);
      }
   }
   return bs;
}

// Strides in reals, change 56's LineStrides with D1D = Q1D = N: the x stage's
// [dz][dy][qx] arrays with an odd dy stride and a dz stride 8 mod 16, the
// stacks' [qz][qy][qx] planes with a z stride 8 mod 16, so that a half-warp's
// lanes fall in distinct banks.
template <int N>
struct Strides
{
   static constexpr int XQ = (N % 2 == 0) ? N + 1 : N + 2;
   static constexpr int XZ = ((N * XQ + 7) / 16) * 16 + 8;
   static constexpr int YZ = ((N * N + 7) / 16) * 16 + 8;
   static constexpr int XS = N * XZ, YS = N * YZ;
   /// Bytes of shared memory per block with @p stacks stacks.
   static constexpr int Bytes(int stacks)
   {
      return (2 * XS + stacks * YS) * static_cast<int>(sizeof(real_t));
   }
};

/// A kernel's data (device pointers; null where unused).
template <int N>
struct Args
{
   Basis<N> bs;
   const real_t* jinv = nullptr;   ///< J^-1, (N^3, 3, 3, ne)
   const real_t* wdet = nullptr;   ///< w det J, (N^3, ne)
   const real_t* lambda = nullptr; ///< Per element.
   const real_t* mu = nullptr;     ///< Per element.
   /// Input (3 stride): u or the direction, true dofs or the L-vector.
   const real_t* x = nullptr;
   const real_t* ghosts = nullptr; ///< The input's ghosts, (3, ghosts).
   const real_t* lin = nullptr;    ///< The linearization (MODE 2: as x).
   const real_t* linghosts = nullptr; ///< MODE 2: the linearization's ghosts.
   real_t* y = nullptr;            ///< Output, as x: its single-element dofs.
   real_t* fs = nullptr;           ///< Stored F (and S), (N^3, 9 STORE, ne).
   real_t* we = nullptr;           ///< Energy partial sums, (N^2, ne).
   const int* gmap = nullptr;      ///< E-dof to its input index or -1 - ghost.
   const int* dmap = nullptr;      ///< E-dof to its output index or -1 - slot.
   real_t* slots = nullptr;        ///< Shared dofs' contributions, (ns, 3).
   int stride = 0;                 ///< The component stride of x, lin and y.
   int ns = 0;                     ///< Slots.
};

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
/// Component c's reference gradient on the thread's column (qx, qy) = (tx,
/// ty) of the stacks s0, s1, s2 (k = x, y, z), its dofs read at the thread's
/// x line @p gi from the component's input @p xc (gi >= 0) or its ghosts
/// @p gc (stride 3, at -1 - gi): the gather fused. Collocated: B along x, y and z to the values U at the points
/// (in sXG, as a stack), then D = G B^-1 along each direction there: 6 N^4
/// multiply-adds per component against change 56's 8 N^4 (B and G in every
/// stage). Ends with each thread's own column written (no sync after it).
template <int N>
__device__ inline void GradLines(const real_t* __restrict__ xc,
                                 const real_t* __restrict__ gc, const int* gi,
                                 const int tx, const int ty, const Basis<N>& bs,
                                 real_t* sXB, real_t* sXG, real_t* s0, real_t* s1,
                                 real_t* s2)
{
   using L = Strides<N>;
   // x: thread (dy, dz) = (tx, ty), its line of dofs: B along x to sXB.
   {
      real_t u[N];
      MFEM_UNROLL(N)
      for (int dx = 0; dx < N; ++dx)
      {
         const int s = gi[dx];
         u[dx] = s >= 0 ? xc[s] : gc[3 * (-1 - s)];
      }
      MFEM_UNROLL(N)
      for (int qx = 0; qx < N; ++qx)
      {
         real_t bu = 0.0;
         MFEM_UNROLL(N)
         for (int dx = 0; dx < N; ++dx) { bu += bs.B[qx][dx] * u[dx]; }
         sXB[ty * L::XZ + tx * L::XQ + qx] = bu;
      }
   }
   __syncthreads();
   // y: thread (qx, dz) = (tx, ty): B along y to s0 ([dz][qy][qx]).
   {
      real_t bu[N];
      MFEM_UNROLL(N)
      for (int dy = 0; dy < N; ++dy) { bu[dy] = sXB[ty * L::XZ + dy * L::XQ + tx]; }
      MFEM_UNROLL(N)
      for (int qy = 0; qy < N; ++qy)
      {
         real_t bbu = 0.0;
         MFEM_UNROLL(N)
         for (int dy = 0; dy < N; ++dy) { bbu += bs.B[qy][dy] * bu[dy]; }
         s0[ty * L::YZ + qy * N + tx] = bbu;
      }
   }
   __syncthreads();
   // z: thread (qx, qy) = (tx, ty), its own column: B along z to U (sXG, its
   // column also kept for the z derivative).
   const int col = ty * N + tx;
   real_t uq[N];
   {
      real_t bb[N];
      MFEM_UNROLL(N)
      for (int dz = 0; dz < N; ++dz) { bb[dz] = s0[dz * L::YZ + col]; }
      MFEM_UNROLL(N)
      for (int qz = 0; qz < N; ++qz)
      {
         real_t v = 0.0;
         MFEM_UNROLL(N)
         for (int dz = 0; dz < N; ++dz) { v += bs.B[qz][dz] * bb[dz]; }
         uq[qz] = v;
         sXG[qz * L::YZ + col] = v;
      }
   }
   __syncthreads();
   // The derivatives at the points: x and y from U's rows, z from the column.
   MFEM_UNROLL(N)
   for (int qz = 0; qz < N; ++qz)
   {
      real_t gx = 0.0, gy = 0.0, gz = 0.0;
      MFEM_UNROLL(N)
      for (int i = 0; i < N; ++i)
      {
         gx += bs.D[tx][i] * sXG[qz * L::YZ + ty * N + i];
         gy += bs.D[ty][i] * sXG[qz * L::YZ + i * N + tx];
         gz += bs.D[qz][i] * uq[i];
      }
      s0[qz * L::YZ + col] = gx;
      s1[qz * L::YZ + col] = gy;
      s2[qz * L::YZ + col] = gz;
   }
}

/// The transposed gradient of one component's flux (stacks f0, f1, f2, k = x,
/// y, z; each thread's own column written by the point loop) into its dofs,
/// collocated as GradLines: D^T along each direction at the points, summed,
/// then B^T along z, y and x. The thread's x line of dofs @p di goes to the
/// component's output @p yc where a single element holds the dof and no other
/// rank shares it (di >= 0), to its slot in @p sc otherwise (-1 - di; summed
/// by ScatterSlots).
template <int N>
__device__ inline void GradTransposeLines(const int tx, const int ty,
      const Basis<N>& bs, real_t* sXB,
      real_t* sXG, const real_t* f0,
      const real_t* f1, const real_t* f2,
      const int* di, real_t* __restrict__ yc,
      real_t* __restrict__ sc)
{
   using L = Strides<N>;
   __syncthreads(); // every thread's fluxes (and sXG free)
   const int col = ty * N + tx;
   // D^T at the points (x and y from the flux rows, z from the column), then
   // B^T along z: thread (qx, qy) = (tx, ty), to sXG ([dz][qy][qx]).
   {
      real_t a2[N], g[N];
      MFEM_UNROLL(N)
      for (int qz = 0; qz < N; ++qz) { a2[qz] = f2[qz * L::YZ + col]; }
      MFEM_UNROLL(N)
      for (int qz = 0; qz < N; ++qz)
      {
         real_t v = 0.0;
         MFEM_UNROLL(N)
         for (int i = 0; i < N; ++i)
         {
            v += bs.D[i][tx] * f0[qz * L::YZ + ty * N + i] +
                 bs.D[i][ty] * f1[qz * L::YZ + i * N + tx] + bs.D[i][qz] * a2[i];
         }
         g[qz] = v;
      }
      MFEM_UNROLL(N)
      for (int dz = 0; dz < N; ++dz)
      {
         real_t v = 0.0;
         MFEM_UNROLL(N)
         for (int qz = 0; qz < N; ++qz) { v += bs.B[qz][dz] * g[qz]; }
         sXG[dz * L::YZ + col] = v;
      }
   }
   __syncthreads();
   // y: thread (qx, dz) = (tx, ty): B^T along y to sXB ([dz][dy][qx]).
   {
      real_t z[N];
      MFEM_UNROLL(N)
      for (int qy = 0; qy < N; ++qy) { z[qy] = sXG[ty * L::YZ + qy * N + tx]; }
      MFEM_UNROLL(N)
      for (int dy = 0; dy < N; ++dy)
      {
         real_t w = 0.0;
         MFEM_UNROLL(N)
         for (int qy = 0; qy < N; ++qy) { w += bs.B[qy][dy] * z[qy]; }
         sXB[ty * L::XZ + dy * L::XQ + tx] = w;
      }
   }
   __syncthreads();
   // x: thread (dy, dz) = (tx, ty): B^T along x, to the dofs. The next call's
   // first sync orders its writes of sXG after these reads of sXB.
   {
      real_t w[N];
      MFEM_UNROLL(N)
      for (int qx = 0; qx < N; ++qx) { w[qx] = sXB[ty * L::XZ + tx * L::XQ + qx]; }
      MFEM_UNROLL(N)
      for (int dx = 0; dx < N; ++dx)
      {
         real_t v = 0.0;
         MFEM_UNROLL(N)
         for (int qx = 0; qx < N; ++qx) { v += bs.B[qx][dx] * w[qx]; }
         const int d = di[dx];
         if (d >= 0) { yc[d] = v; }
         else { sc[-1 - d] = v; }
      }
   }
}

/// J^-1 at point q of element e: JI[k + 3 d].
template <int N>
__device__ inline void LoadJinv(const real_t* __restrict__ jinv, const int q,
                                const int e, real_t* JI)
{
   constexpr int NQ = N * N * N;
   const real_t* p = jinv + q + NQ * 9 * e;
   MFEM_UNROLL(9)
   for (int m = 0; m < 9; ++m) { JI[m] = p[NQ * m]; }
}

/// G[c + 3 d] = sum_k r(c, k) J^-1(k, d), r(c, k) the reference gradient k
/// of component c in stack 3 c + k at offset i.
template <int N>
__device__ inline void PhysicalGradient(const real_t* st, const int i,
                                        const real_t* JI, real_t* G)
{
   constexpr int YS = Strides<N>::YS;
   real_t r[9];
   MFEM_UNROLL(9)
   for (int j = 0; j < 9; ++j) { r[j] = st[j * YS + i]; }
   MFEM_UNROLL(3)
   for (int d = 0; d < 3; ++d)
   {
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         G[c + 3 * d] = r[3 * c] * JI[3 * d] + r[3 * c + 1] * JI[1 + 3 * d] +
                        r[3 * c + 2] * JI[2 + 3 * d];
      }
   }
}

/// Stack 3 c + k at offset i gets w sum_d P(c, d) J^-1(k, d): the flux in
/// reference directions.
template <int N>
__device__ inline void WriteFlux(real_t* st, const int i, const real_t w,
                                 const real_t* JI, const real_t* P)
{
   constexpr int YS = Strides<N>::YS;
   MFEM_UNROLL(3)
   for (int c = 0; c < 3; ++c)
   {
      MFEM_UNROLL(3)
      for (int k = 0; k < 3; ++k)
      {
         st[(3 * c + k) * YS + i] = w * (P[c] * JI[k] + P[c + 3] * JI[k + 3] +
                                         P[c + 6] * JI[k + 6]);
      }
   }
}

// The minimum blocks per SM is a template argument, as change 56's
// LinesKernel: nvcc 12.6 capped that kernel's registers when the bound was a
// literal.
template <class MAT, int N, bool RESIDUAL, int STORE, bool ENERGY, int MINB>
__global__ void __launch_bounds__(N* N, MINB)
PointwiseKernel(const __grid_constant__ Args<N> a)
{
   using L = Strides<N>;
   constexpr int NQ = N * N * N;
   extern __shared__ real_t smem[];
   real_t* sXB = smem;
   real_t* sXG = smem + L::XS;
   real_t* st = smem + 2 * L::XS; // stack 3 c + k at st + (3 c + k) YS
   const int e = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
   const int col = ty * N + tx, line = NQ * e + N * (tx + N * ty);
   {
      int gi[N]; // the thread's x line of L-dofs
      MFEM_UNROLL(N)
      for (int dx = 0; dx < N; ++dx) { gi[dx] = a.gmap[line + dx]; }
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         real_t* s = st + 3 * c * L::YS;
         GradLines<N>(a.x + a.stride * c, a.ghosts + c, gi, tx, ty, a.bs, sXB, sXG, s,
                      s + L::YS,
                      s + 2 * L::YS);
      }
   }
   const real_t lam = a.lambda[e], mu = a.mu[e];
   [[maybe_unused]] real_t esum = 0.0;
#pragma unroll 1
   for (int qz = 0; qz < N; ++qz)
   {
      const int i = qz * L::YZ + col, q = col + N * N * qz;
      real_t JI[9], F[9], S[9], P[9];
      LoadJinv<N>(a.jinv, q, e, JI);
      PhysicalGradient<N>(st, i, JI, F);
      F[0] += 1.0; F[4] += 1.0; F[8] += 1.0;
      MAT::template Stress<3>(F, lam, mu, S, P);
      if constexpr(STORE >= 1)
      {
         MFEM_UNROLL(9)
         for (int m = 0; m < 9; ++m) { a.fs[q + NQ * (m + 9 * STORE * e)] = F[m]; }
      }
      if constexpr(STORE == 2)
      {
         MFEM_UNROLL(9)
         for (int m = 0; m < 9; ++m) { a.fs[q + NQ * (9 + m + 18 * e)] = S[m]; }
      }
      if constexpr(ENERGY)
      {
         esum += a.wdet[q + NQ * e] * MAT::template Energy<3>(F, lam, mu);
      }
      if constexpr(RESIDUAL) { WriteFlux<N>(st, i, a.wdet[q + NQ * e], JI, P); }
   }
   if constexpr(ENERGY) { a.we[col + N * N * e] = esum; }
   if constexpr(RESIDUAL)
   {
      int di[N]; // the thread's x line of destinations
      MFEM_UNROLL(N)
      for (int dx = 0; dx < N; ++dx) { di[dx] = a.dmap[line + dx]; }
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         real_t* s = st + 3 * c * L::YS;
         GradTransposeLines<N>(tx, ty, a.bs, sXB, sXG, s, s + L::YS, s + 2 * L::YS,
                               di, a.y + a.stride * c, a.slots + a.ns * c);
      }
   }
}

template <class MAT, int N, int MODE, int MINB>
__global__ void __launch_bounds__(N* N, MINB)
TangentKernel(const __grid_constant__ Args<N> a)
{
   using L = Strides<N>;
   constexpr int NQ = N * N * N, NS = (MODE == 0) ? 18 : 9;
   extern __shared__ real_t smem[];
   real_t* sXB = smem;
   real_t* sXG = smem + L::XS;
   real_t* st = smem + 2 * L::XS; // dF's stacks 0-8, F's 9-17 (MODE 2)
   const int e = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
   const int col = ty * N + tx, line = NQ * e + N * (tx + N * ty);
   {
      int gi[N]; // the thread's x line of L-dofs
      MFEM_UNROLL(N)
      for (int dx = 0; dx < N; ++dx) { gi[dx] = a.gmap[line + dx]; }
      if constexpr(MODE == 2)
      {
         MFEM_UNROLL(3)
         for (int c = 0; c < 3; ++c)
         {
            real_t* s = st + (9 + 3 * c) * L::YS;
            GradLines<N>(a.lin + a.stride * c, a.linghosts + c, gi, tx, ty, a.bs, sXB, sXG,
                         s,
                         s + L::YS, s + 2 * L::YS);
         }
      }
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         real_t* s = st + 3 * c * L::YS;
         GradLines<N>(a.x + a.stride * c, a.ghosts + c, gi, tx, ty, a.bs, sXB, sXG, s,
                      s + L::YS,
                      s + 2 * L::YS);
      }
   }
   const real_t lam = a.lambda[e], mu = a.mu[e];
#pragma unroll 1
   for (int qz = 0; qz < N; ++qz)
   {
      const int i = qz * L::YZ + col, q = col + N * N * qz;
      real_t JI[9], F[9], S[9], dF[9], dP[9];
      LoadJinv<N>(a.jinv, q, e, JI);
      PhysicalGradient<N>(st, i, JI, dF);
      if constexpr(MODE == 2)
      {
         PhysicalGradient<N>(st + 9 * L::YS, i, JI, F);
         F[0] += 1.0; F[4] += 1.0; F[8] += 1.0;
      }
      else
      {
         MFEM_UNROLL(9)
         for (int m = 0; m < 9; ++m) { F[m] = a.lin[q + NQ * (m + NS * e)]; }
      }
      if constexpr(MODE == 0)
      {
         MFEM_UNROLL(9)
         for (int m = 0; m < 9; ++m) { S[m] = a.lin[q + NQ * (9 + m + NS * e)]; }
      }
      else
      {
         real_t P[9];
         MAT::template Stress<3>(F, lam, mu, S, P);
      }
      MAT::template Tangent<3>(F, S, dF, lam, mu, dP);
      WriteFlux<N>(st, i, a.wdet[q + NQ * e], JI, dP);
   }
   int di[N]; // the thread's x line of destinations
   MFEM_UNROLL(N)
   for (int dx = 0; dx < N; ++dx) { di[dx] = a.dmap[line + dx]; }
   MFEM_UNROLL(3)
   for (int c = 0; c < 3; ++c)
   {
      real_t* s = st + 3 * c * L::YS;
      GradTransposeLines<N>(tx, ty, a.bs, sXB, sXG, s, s + L::YS, s + 2 * L::YS,
                            di, a.y + a.stride * c, a.slots + a.ns * c);
   }
}

/// One block of N x N threads per element, @p stacks stacks of shared memory.
template <int N, class K>
void Launch(K kernel, const Args<N>& a, const int ne, const int stacks)
{
   const int bytes = Strides<N>::Bytes(stacks);
   MFEM_GPU_CHECK(cudaFuncSetAttribute(kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       bytes));
   kernel <<< ne, dim3(N, N, 1), bytes>>>(a);
   MFEM_GPU_CHECK(cudaGetLastError());
}
#endif // MFEM_USE_CUDA && __CUDACC__

} // namespace lines

/// Whether 3D applies run the by-lines device kernels (a CUDA device).
bool Lines3D(int dim)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
   return dim == 3 && Device::Allows(Backend::CUDA_MASK);
#else
   MFEM_CONTRACT_VAR(dim);
   return false;
#endif
}

/// The by-lines kernels' gather and scatter maps
/// (HyperelasticOperator::SetupLines).
struct LinesMaps
{
   const Array<int>* gmap = nullptr; ///< E-dof to input index, or -1 - ghost.
   const Array<int>* dmap = nullptr; ///< E-dof to output index, or -1 - slot.
   Vector* slots = nullptr;          ///< (ns, 3).
   int stride = 0, ns = 0;           ///< Component stride, slots.
   const Vector* ghosts = nullptr;   ///< The input's ghosts (the exchange's).
   int linghosts = -1; ///< MODE 2: where lin's ghosts start (-1: none).
};

/// Pointwise3D's by-lines counterpart: x and y are true-dof vectors (a
/// conforming space; x's ghosts in m.ghosts) or L-vectors; y gets its
/// single-element dofs, the slots their shared dofs' contributions
/// (ScatterSlots finishes it).
template <class MAT, int N, bool RESIDUAL, int STORE, bool ENERGY>
void PointwiseLines3D(int NE, const Array<real_t>& b, const Array<real_t>& g,
                      const Vector& jinv, const Vector& wdet, const Vector& lam,
                      const Vector& mu, const Vector& x, Vector* y, Vector* fs,
                      Vector* we, const LinesMaps& m)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
   if (NE == 0) { return; }
   lines::Args<N> a;
   a.bs = lines::MakeBasis<N>(b, g);
   a.jinv = jinv.Read();
   a.wdet = wdet.Read();
   a.lambda = lam.Read();
   a.mu = mu.Read();
   a.x = x.Read();
   a.ghosts = (m.ghosts && m.ghosts->Size() > 0) ? m.ghosts->Read() : a.x;
   a.gmap = m.gmap->Read();
   a.stride = m.stride;
   a.ns = m.ns;
   if constexpr(RESIDUAL)
   {
      a.y = y->Write();
      a.dmap = m.dmap->Read();
      a.slots = m.slots->Write();
   }
   if constexpr(STORE > 0) { a.fs = fs->Write(); }
   if constexpr(ENERGY) { a.we = we->Write(); }
   lines::Launch<N>(lines::PointwiseKernel<MAT, N, RESIDUAL, STORE, ENERGY, 1>, a,
                    NE, 9);
#else
   MFEM_CONTRACT_VAR(NE); MFEM_CONTRACT_VAR(b); MFEM_CONTRACT_VAR(g);
   MFEM_CONTRACT_VAR(jinv); MFEM_CONTRACT_VAR(wdet); MFEM_CONTRACT_VAR(lam);
   MFEM_CONTRACT_VAR(mu); MFEM_CONTRACT_VAR(x); MFEM_CONTRACT_VAR(y);
   MFEM_CONTRACT_VAR(fs); MFEM_CONTRACT_VAR(we); MFEM_CONTRACT_VAR(m);
   MFEM_ABORT("HyperelasticOperator: the by-lines kernels need a CUDA build");
#endif
}

/// Tangent3D's by-lines counterpart: x, y (and lin for MODE 2, its ghosts at
/// m.linghosts) as PointwiseLines3D's.
template <class MAT, int N, int MODE>
void TangentLines3D(int NE, const Array<real_t>& b, const Array<real_t>& g,
                    const Vector& jinv, const Vector& wdet, const Vector& lam,
                    const Vector& mu, const Vector& lin, const Vector& x,
                    Vector& y, const LinesMaps& m)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
   if (NE == 0) { return; }
   lines::Args<N> a;
   a.bs = lines::MakeBasis<N>(b, g);
   a.jinv = jinv.Read();
   a.wdet = wdet.Read();
   a.lambda = lam.Read();
   a.mu = mu.Read();
   a.lin = lin.Read();
   a.linghosts = (MODE == 2 && m.linghosts >= 0) ? a.lin + m.linghosts : a.lin;
   a.x = x.Read();
   a.ghosts = (m.ghosts && m.ghosts->Size() > 0) ? m.ghosts->Read() : a.x;
   a.y = y.Write();
   a.gmap = m.gmap->Read();
   a.dmap = m.dmap->Read();
   a.slots = m.slots->Write();
   a.stride = m.stride;
   a.ns = m.ns;
   lines::Launch<N>(lines::TangentKernel<MAT, N, MODE, 1>, a, NE,
                    MODE == 2 ? 18 : 9);
#else
   MFEM_CONTRACT_VAR(NE); MFEM_CONTRACT_VAR(b); MFEM_CONTRACT_VAR(g);
   MFEM_CONTRACT_VAR(jinv); MFEM_CONTRACT_VAR(wdet); MFEM_CONTRACT_VAR(lam);
   MFEM_CONTRACT_VAR(mu); MFEM_CONTRACT_VAR(lin); MFEM_CONTRACT_VAR(x);
   MFEM_CONTRACT_VAR(y); MFEM_CONTRACT_VAR(m);
   MFEM_ABORT("HyperelasticOperator: the by-lines kernels need a CUDA build");
#endif
}

/// Each slotted dof's slots summed per component, in slot (element) order:
/// the by-lines scatter's second half. Destination sdof[k] >= 0: y's entry
/// (component stride @p stride); < 0: ghost -1 - sdof[k] of @p g, (3, ghosts),
/// the exchange's sums for the owners.
void ScatterSlots(const Array<int>& sdof, const Array<int>& soff,
                  const Array<int>& sidx, const Vector& slots, const int stride,
                  const int ns, Vector& y, Vector* g)
{
   const int n = sdof.Size();
   if (n == 0) { return; }
   const auto J = sdof.Read(), O = soff.Read(), I = sidx.Read();
   const auto S = slots.Read();
   auto Y = y.ReadWrite();
   real_t* G = (g && g->Size() > 0) ? g->Write() : Y; // Y: never written then
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int k)
   {
      const int j = J[k], k0 = O[k], k1 = O[k + 1];
      for (int c = 0; c < 3; ++c)
      {
         real_t v = 0.0;
         for (int i = k0; i < k1; ++i) { v += S[I[i] + ns * c]; }
         if (j >= 0) { Y[j + stride * c] = v; }
         else { G[3 * (-1 - j) + c] = v; }
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
      const auto& m = *op_.maps_;
      const int mode = 2 - StoreFor(op_.storage_); // 0 F,S; 1 F; 2 displacement
      const LinesMaps lm{op_.gmap_, &op_.dmap_, &op_.slots_, op_.stride_, op_.nslots_,
                         op_.ex_ ? &op_.ex_->Ghosts() : nullptr, op_.lin_ghosts_};
      const Vector* xl = nullptr;
      Vector* yl = nullptr;
      if (op_.lines_)
      {
         xl = &op_.LocalInput(w);
         yl = &op_.LocalOutput(y);
      }
      else
      {
         op_.ToElements(w);
         op_.ye_ = 0.0;
      }
      WithPoints(op_.n1d_, [&](auto nc)
      {
         constexpr int N = decltype(nc)::value;
         auto run = [&](auto mc)
         {
            constexpr int MODE = decltype(mc)::value;
            if (op_.lines_)
            {
               TangentLines3D<StVenantKirchhoff, N, MODE>(op_.ne_, m.B, m.G, op_.jinv_,
                     op_.wdet_, op_.lambda_, op_.mu_,
                     op_.lin_, *xl, *yl, lm);
            }
            else if (op_.dim_ == 2)
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
      if (op_.lines_) { op_.FinishOutput(*yl, y); }
      else { op_.FromElements(y); }
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
   lines_ = Lines3D(dim_);
   if (lines_) { SetupLines(scratch); }
   const int nstore = StoreFor(storage_);
   // Displacement storage: the E-vector, or what the by-lines kernels read: the
   // L-vector, or the true dofs with the ghosts after them.
   const int nlu = ex_ ? Height() + 3 * ex_->NumGhosts() : fes.GetVSize();
   lin_.SetSize(nstore > 0 ? nq * nstore * dim_ * dim_ * ne_ :
                (lines_ ? nlu : restriction_->Height()));
   lin_ghosts_ = ex_ ? Height() : -1;
   lin_.UseDevice(true);

   const int ne_size = restriction_->Height(), nl = fes.GetVSize();
   if (lines_)
   {
      own_e_ = false; // no E-vectors
   }
   else if (scratch && scratch->Size() >= ne_size)
   {
      scratch->Lend(0, ne_size, ue_);
      scratch->Lend(1, ne_size, ye_);
      own_e_ = false;
   }
   else
   {
      for (Vector* v : {&ue_, &ye_}) { v->SetSize(ne_size); v->UseDevice(true); }
   }
   // One rank, conforming: MFEM's prolongation is an IdentityOperator, and
   // the true dofs are the local dofs -- no L-vectors at all.
   const Operator* P = fes.GetProlongationMatrix();
   p_identity_ = !P || dynamic_cast<const IdentityOperator*>(P) != nullptr;
   if (p_identity_ || ex_) // the exchange: true dofs in and out
   {
      own_l_ = false;
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
   if (own_slots_) { n += slots_.Size(); }
   const std::size_t ni = dmap_.Size() + sdof_.Size() + soff_.Size() +
                          sidx_.Size() + gsrc_.Size();
   return n * sizeof(real_t) + ni * sizeof(int) +
          (ex_ ? ex_->MemoryBytes() : 0);
}

void HyperelasticOperator::SetupLines(ElementScratch* scratch)
{
   const auto* er = dynamic_cast<const ElementRestriction*>(restriction_);
   if (!er)
   {
      lines_ = false;
      return;
   }
   const int nq = n1d_ * n1d_ * n1d_, nd = fes_.GetNDofs();
   const Array<int>& g = er->GatherMap();
   MFEM_VERIFY(g.Size() == nq * ne_, "HyperelasticOperator: gather map size");
   const int* h = g.HostRead();
   std::vector<int> count(nd, 0);
   for (int i = 0; i < g.Size(); i++)
   {
      MFEM_VERIFY(h[i] >= 0 && h[i] < nd, "HyperelasticOperator: an H1 gather map");
      count[h[i]]++;
   }
   // A conforming space: the kernels read and write true dofs, the shared dofs
   // other ranks own (ghosts) through the exchange. Otherwise the L-vector.
   std::vector<int> io(nd);   // per L-dof: its input/output index, or -1 - ghost
   std::vector<char> remote(nd, 0); // another rank needs or adds to its result
   if (fes_.Conforming())
   {
      ex_ = std::make_unique<TrueDofExchange>(fes_);
      const int* ls = ex_->LocalSource().HostRead();
      std::vector<int> tl(ex_->TrueStride(), -1);
      for (int j = 0; j < nd; j++)
      {
         io[j] = ls[j];
         if (ls[j] < 0) { remote[j] = 1; }
         else { tl[ls[j]] = j; }
      }
      const Array<int>& os = ex_->OwnedShared();
      const int* o = os.HostRead();
      for (int k = 0; k < os.Size(); k++) { remote[tl[o[k]]] = 1; }
      stride_ = ex_->TrueStride();
      gsrc_.SetSize(g.Size());
      int* gs = gsrc_.HostWrite();
      for (int i = 0; i < g.Size(); i++) { gs[i] = io[h[i]]; }
      gmap_ = &gsrc_;
   }
   else
   {
      for (int j = 0; j < nd; j++) { io[j] = j; }
      stride_ = nd;
      gmap_ = &g;
   }
   // A dof one element holds and no other rank shares is written directly;
   // the others get a slot per element, in element (E-dof) order.
   auto direct = [&](int j) { return count[j] == 1 && !remote[j]; };
   dmap_.SetSize(g.Size());
   int* dm = dmap_.HostWrite();
   int ns = 0;
   for (int i = 0; i < g.Size(); i++) { dm[i] = direct(h[i]) ? io[h[i]] : -1 - ns++; }
   int nsh = 0;
   for (int j = 0; j < nd; j++)
   {
      MFEM_VERIFY(count[j] > 0, "HyperelasticOperator: a dof in no local element");
      if (!direct(j)) { nsh++; }
   }
   sdof_.SetSize(nsh);
   soff_.SetSize(nsh + 1);
   int* sd = sdof_.HostWrite();
   int* so = soff_.HostWrite();
   std::vector<int> next(nd, -1);
   so[0] = 0;
   for (int j = 0, k = 0; j < nd; j++)
   {
      if (direct(j)) { continue; }
      sd[k] = io[j]; // the destination: an output index, or -1 - ghost
      next[j] = so[k];
      so[k + 1] = so[k] + count[j];
      k++;
   }
   sidx_.SetSize(ns);
   int* si = sidx_.HostWrite();
   for (int i = 0; i < g.Size(); i++)
   {
      if (dm[i] < 0) { si[next[h[i]]++] = -1 - dm[i]; } // ascending slots
   }
   nslots_ = ns;
   if (scratch && scratch->Size() >= 3 * ns)
   {
      scratch->Lend(0, 3 * ns, slots_);
      own_slots_ = false;
   }
   else
   {
      slots_.SetSize(3 * ns);
      slots_.UseDevice(true);
   }
}

const Vector& HyperelasticOperator::LocalInput(const Vector& u) const
{
   if (ex_)
   {
      ex_->GatherBegin(u);
      ex_->GatherEnd();
      return u;
   }
   if (p_identity_) { return u; }
   fes_.GetProlongationMatrix()->Mult(u, ul_);
   return ul_;
}

Vector& HyperelasticOperator::LocalOutput(Vector& y) const
{
   y.SetSize(Height());
   return (p_identity_ || ex_) ? y : yl_;
}

void HyperelasticOperator::FinishOutput(Vector& yl, Vector& y) const
{
   if (ex_)
   {
      ScatterSlots(sdof_, soff_, sidx_, slots_, stride_, nslots_, y,
                   &ex_->GhostSums());
      ex_->ReduceBegin();
      ex_->ReduceEnd(y);
      return;
   }
   ScatterSlots(sdof_, soff_, sidx_, slots_, stride_, nslots_, yl, nullptr);
   if (!p_identity_) { fes_.GetProlongationMatrix()->MultTranspose(yl_, y); }
}

void HyperelasticOperator::ToElements(const Vector& u) const
{
   if (!p_identity_)
   {
      const Operator* P = fes_.GetProlongationMatrix();
      P->Mult(u, ul_);
      restriction_->Mult(ul_, ue_);
   }
   else { restriction_->Mult(u, ue_); }
}

void HyperelasticOperator::FromElements(Vector& y) const
{
   y.SetSize(Height());
   if (!p_identity_)
   {
      const Operator* P = fes_.GetProlongationMatrix();
      restriction_->MultTranspose(ye_, yl_);
      P->MultTranspose(yl_, y);
   }
   else { restriction_->MultTranspose(ye_, y); }
}

void HyperelasticOperator::Mult(const Vector& u, Vector& r) const
{
   if (lines_)
   {
      const Vector& xl = LocalInput(u);
      Vector& yl = LocalOutput(r);
      const LinesMaps lm{gmap_, &dmap_, &slots_, stride_, nslots_, ex_ ? &ex_->Ghosts() : nullptr, lin_ghosts_};
      WithPoints(n1d_, [&](auto nc)
      {
         constexpr int N = decltype(nc)::value;
         PointwiseLines3D<StVenantKirchhoff, N, true, 0, false>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, xl, &yl, nullptr,
            nullptr, lm);
      });
      FinishOutput(yl, r);
      return;
   }
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
   const int nstore = StoreFor(storage_);
   const Vector* xl = nullptr;
   if (lines_) { xl = &LocalInput(u); }
   else { ToElements(u); }
   if (nstore == 0)
   {
      // The displacement (true dofs and ghosts, the L-vector or the E-vector):
      // the tangent recomputes F and S.
      if (ex_)
      {
         // Copied into lin_ itself: written through sub-vector aliases on a
         // device, lin_'s device copy is not marked valid, and its next Read()
         // brings the stale host data back (hyperelasticity_test HM, job
         // 49574179).
         const int n = Height(), ng = 3 * ex_->NumGhosts();
         const auto U = u.Read();
         auto L = lin_.Write();
         mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int i) { L[i] = U[i]; });
         if (ng > 0)
         {
            const auto G = ex_->Ghosts().Read();
            mfem::forall(ng, [ = ] MFEM_HOST_DEVICE(int i) { L[n + i] = G[i]; });
         }
      }
      else { lin_ = lines_ ? *xl : ue_; }
      return *tangent_;
   }
   const LinesMaps lm{gmap_, &dmap_, &slots_, stride_, nslots_, ex_ ? &ex_->Ghosts() : nullptr, lin_ghosts_};
   WithPoints(n1d_, [&](auto nc)
   {
      constexpr int N = decltype(nc)::value;
      auto run = [&](auto sc)
      {
         constexpr int STORE = decltype(sc)::value;
         if (lines_)
         {
            PointwiseLines3D<StVenantKirchhoff, N, false, STORE, false>(
               ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, *xl, nullptr, &lin_,
               nullptr, lm);
         }
         else if (dim_ == 2)
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
   const Vector* xl = nullptr;
   if (lines_) { xl = &LocalInput(u); }
   else { ToElements(u); }
   const LinesMaps lm{gmap_, &dmap_, &slots_, stride_, nslots_, ex_ ? &ex_->Ghosts() : nullptr, lin_ghosts_};
   WithPoints(n1d_, [&](auto nc)
   {
      constexpr int N = decltype(nc)::value;
      if (lines_)
      {
         PointwiseLines3D<StVenantKirchhoff, N, false, 0, true>(
            ne_, maps_->B, maps_->G, jinv_, wdet_, lambda_, mu_, *xl, nullptr, nullptr,
            &qe_, lm);
      }
      else if (dim_ == 2)
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
