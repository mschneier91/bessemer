#pragma once

// One velocity component's register arrays for MFEM's 3D sum-factorized
// kernel helpers (mfem::kernels::internal): loading a component's dofs and
// its gradient (or the gradient's transpose) with each direction's
// contraction written out. At Q1D >= 7 nvcc no longer unrolls the loop over
// directions in MFEM's Grad3d, whose index then selects the register arrays
// at run time and puts them on the stack (ptxas, 2026-10-16); the kernels
// that hold every component's gradient (VDIM x DIM x Q1D doubles per array)
// are also past the 255-register limit at Q1D 8. Used by the divergence
// (operators/kernel_specializations.cpp) and viscous
// (operators/lean_vector_diffusion.cpp) kernels at those sizes.

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

} // namespace incns::kernels3d
