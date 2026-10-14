/**
 * @file lean_vector_diffusion.hpp
 * @brief mfem::VectorDiffusionIntegrator with a partial-assembly diagonal that
 *        needs no per-thread stack on a GPU.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief VectorDiffusionIntegrator whose AssembleDiagonalPA is a direct sum
 *        over the quadrature points, with no local arrays.
 *
 * MFEM's 3D kernel (PAVectorDiffusionDiagonal3D, called without its template
 * sizes) declares two work arrays per thread at the device's largest supported
 * sizes, [MAX_Q1D][MAX_Q1D][MAX_D1D] and [MAX_Q1D][MAX_D1D][MAX_D1D] with
 * MAX = 14 under CUDA: 44 kB of stack per thread. The CUDA runtime reserves
 * stack for every thread the device can run at once, so the first call leaves
 * about 11 GB of an H100 allocated for the life of the process, whatever the
 * mesh or the order (measured 2026-10-13: 11,088 MiB at 24^3 Q3, the same at
 * Q2 and without grad-div; docs/upstream/mfem_vector_diffusion_diagonal.md).
 *
 * Here the diagonal entry of dof a is sum_q ghat_a^T D_q ghat_a, with ghat_a
 * the reference gradient of the dof's basis function rebuilt from the 1D
 * tables: O(p^6) per element in 3D instead of the sum-factorized O(p^4), which
 * does not matter for a setup operation. Everything else is MFEM's.
 *
 * Scalar coefficient, quads and hexes with sdim == dim; any other case
 * falls back to MFEM's kernel.
 */
class LeanVectorDiffusionIntegrator : public mfem::VectorDiffusionIntegrator
{
public:
   /**
    * @brief Vector diffusion with a scalar coefficient.
    * @param q The coefficient (borrowed).
    */
   explicit LeanVectorDiffusionIntegrator(mfem::Coefficient& q)
      : mfem::VectorDiffusionIntegrator(q) { }

   /// Vector diffusion with unit coefficient.
   LeanVectorDiffusionIntegrator() = default;

   /**
    * @brief Add the diagonal of the partially assembled operator.
    * @param diag E-vector in the PA diagonal's layout (accumulated into).
    */
   void AssembleDiagonalPA(mfem::Vector& diag) override;
};

} // namespace incns
