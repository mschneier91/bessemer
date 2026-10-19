#pragma once

// One velocity component's register arrays for MFEM's 3D sum-factorized
// kernel helpers (mfem::kernels::internal): loading a component's dofs and
// its gradient (or the gradient's transpose) with each direction's
// contraction written out. At Q1D >= 7 nvcc no longer unrolls the loop over
// directions in MFEM's Grad3d, whose index then selects the register arrays
// at run time and puts them on the stack (ptxas, 2026-10-16); the kernels
// that hold every component's gradient (VDIM x DIM x Q1D doubles per array)
// are also past the 255-register limit at Q1D 8. Used by the divergence
// (operators/kernel_specializations.cpp, operators/lean_vector_divergence.cpp)
// and viscous (operators/lean_vector_diffusion.cpp) kernels at those sizes.

#include "mfem/fem/kernels.hpp"

namespace incns::kernels3d
{

/// Y[0][d] = component c of element e's dofs, for each direction d (the
/// contractions overwrite their input, as in MFEM's LoadDofs3d).
template <int MQ1>
MFEM_HOST_DEVICE inline void LoadComponent(
   const int e, const int d1d, const int c,
   const mfem::DeviceTensor<5, const mfem::real_t>& X,
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& Y)
{
   for (int dz = 0; dz < d1d; ++dz)
   {
      MFEM_FOREACH_THREAD_DIRECT(dy, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(dx, x, d1d)
         {
            const mfem::real_t v = X(dx, dy, dz, c, e);
            Y[0][0][dz][dy][dx] = v;
            Y[0][1][dz][dy][dx] = v;
            Y[0][2][dz][dy][dx] = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// MFEM's Grad3d (Transpose false) or GradTranspose3d (true) of one
/// component: the same contractions, each direction's with a compile-time
/// index.
template <bool Transpose, int MQ1>
MFEM_HOST_DEVICE inline void GradComponent(
   const int d1d, const int q1d, mfem::real_t (&smem)[MQ1][MQ1],
   const mfem::real_t (*B)[MQ1], const mfem::real_t (*G)[MQ1],
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& X,
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& Y)
{
   namespace ki = mfem::kernels::internal;
   ki::Contract3d<Transpose>(d1d, q1d, smem, G, B, B, X[0][0], Y[0][0]);
   ki::Contract3d<Transpose>(d1d, q1d, smem, B, G, B, X[0][1], Y[0][1]);
   ki::Contract3d<Transpose>(d1d, q1d, smem, B, B, G, X[0][2], Y[0][2]);
}

/// MFEM's ContractZ3d, ContractY3d and ContractX3d with Transpose true, in
/// Contract3d's order, with the 1D matrices stored by point: B[q][d] (the
/// transposed maps as they are in memory, ki::LoadMatrix(q1d, d1d, Bt, sB))
/// where MFEM reads B[d][q]. MFEM's X and Y contractions read B[d][q] with d
/// the thread index, a row (MQ1 doubles) apart: at Q1D 8 four of a warp's
/// dofs share a bank, and ~40 % of the gradient kernel's shared loads were
/// bank conflicts (Nsight Compute, 2026-10-19). Here a warp reads consecutive
/// doubles. The same products, summed in the same order. Used on a device
/// (GradTransposeComponent below).
template <int MQ1>
MFEM_HOST_DEVICE inline void ContractTranspose3dByPoint(
   const int d1d, const int q1d, mfem::real_t (&smem)[MQ1][MQ1],
   const mfem::real_t (*Bx)[MQ1], const mfem::real_t (*By)[MQ1],
   const mfem::real_t (*Bz)[MQ1],
   mfem::kernels::internal::s_regs3d_t<MQ1>& X,
   mfem::kernels::internal::s_regs3d_t<MQ1>& Y)
{
   // Z: X -> Y.
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d)
         {
            mfem::real_t u = 0.0;
            for (int k = 0; k < q1d; ++k) { u += Bz[k][z] * X[k][y][x]; }
            Y[z][y][x] = u;
         }
      }
   }
   // Y: Y -> X.
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, q1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { smem[y][x] = Y[z][y][x]; }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d)
         {
            mfem::real_t u = 0.0;
            for (int k = 0; k < q1d; ++k) { u += By[k][y] * smem[k][x]; }
            X[z][y][x] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
   // X: X -> Y.
   for (int z = 0; z < d1d; ++z)
   {
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, q1d) { smem[y][x] = X[z][y][x]; }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD_DIRECT(y, y, d1d)
      {
         MFEM_FOREACH_THREAD_DIRECT(x, x, d1d)
         {
            mfem::real_t u = 0.0;
            for (int k = 0; k < q1d; ++k) { u += Bx[k][x] * smem[y][k]; }
            Y[z][y][x] = u;
         }
      }
      MFEM_SYNC_THREAD;
   }
}

/// GradComponent<true> with the 1D matrices stored by point
/// (ContractTranspose3dByPoint): B[q][d] = B(q, d), G[q][d] = G(q, d).
template <int MQ1>
MFEM_HOST_DEVICE inline void GradTransposeComponentByPoint(
   const int d1d, const int q1d, mfem::real_t (&smem)[MQ1][MQ1],
   const mfem::real_t (*B)[MQ1], const mfem::real_t (*G)[MQ1],
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& X,
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& Y)
{
   ContractTranspose3dByPoint(d1d, q1d, smem, G, B, B, X[0][0], Y[0][0]);
   ContractTranspose3dByPoint(d1d, q1d, smem, B, G, B, X[0][1], Y[0][1]);
   ContractTranspose3dByPoint(d1d, q1d, smem, B, B, G, X[0][2], Y[0][2]);
}

/// The gradient's 1D matrices from the transposed maps (Bt, Gt), as
/// GradTransposeComponent reads them: by point on a device, MFEM's layout
/// (B[d][q]) on the host. The CPU build's GCC targets znver2 (FMA, AVX2) and
/// contracts a * b + c wherever its vectorizer leaves one: the by-point loops,
/// vectorized otherwise, changed CPU results at roundoff (2026-10-19). The
/// host keeps MFEM's loops, and its results bit for bit.
template <int MQ1>
MFEM_HOST_DEVICE inline void LoadGradTranspose(
   const int d1d, const int q1d, const mfem::real_t* Bt,
   const mfem::real_t* Gt, mfem::real_t (*B)[MQ1], mfem::real_t (*G)[MQ1])
{
   namespace ki = mfem::kernels::internal;
#if defined(__CUDA_ARCH__)
   ki::LoadMatrix(q1d, d1d, Bt, B);
   ki::LoadMatrix(q1d, d1d, Gt, G);
#else
   ki::LoadMatrix<MQ1, true>(d1d, q1d, Bt, B);
   ki::LoadMatrix<MQ1, true>(d1d, q1d, Gt, G);
#endif
}

/// The gradient's transpose of one component from LoadGradTranspose's
/// matrices: GradTransposeComponentByPoint on a device, GradComponent<true>
/// on the host.
template <int MQ1>
MFEM_HOST_DEVICE inline void GradTransposeComponent(
   const int d1d, const int q1d, mfem::real_t (&smem)[MQ1][MQ1],
   const mfem::real_t (*B)[MQ1], const mfem::real_t (*G)[MQ1],
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& X,
   mfem::kernels::internal::vd_regs3d_t<1, 3, MQ1>& Y)
{
#if defined(__CUDA_ARCH__)
   GradTransposeComponentByPoint(d1d, q1d, smem, B, G, X, Y);
#else
   GradComponent<true>(d1d, q1d, smem, B, G, X, Y);
#endif
}

} // namespace incns::kernels3d
