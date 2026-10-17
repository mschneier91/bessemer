/**
 * @file lean_vector_divergence.hpp
 * @brief The vector divergence (div u, q) whose partial assembly keeps no
 *        geometric data of its own: its kernels read the mesh's cached
 *        Jacobians, which the grad-div form on the same rule keeps anyway.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief mfem::VectorDivergenceIntegrator whose partial assembly stores
 *        nothing per quadrature point.
 *
 * MFEM's partial assembly stores W adj(J) at every point (9 numbers: 2.5 GiB
 * at Q6 on 42^3 elements) next to the mesh's cached Jacobians at the same
 * rule (9 numbers), which the grad-div form reads at every apply (change 13).
 * This one keeps a pointer to those Jacobians and forms W adj(J) in its
 * kernels with MFEM's expressions (PADivergenceSetup3D): the same operator to
 * roundoff, from the same number of values per point.
 *
 * The lean path (Lean()): 3D; H1 trial (vdim 3, byNODES) and test spaces of
 * orders k and k - 1 on tensor elements; the 2k + 2 rule (k + 2 points per
 * direction); k = 2 to 7; no coefficient or a constant one. Anything else is
 * MFEM's partial assembly, unchanged.
 */
class LeanVectorDivergenceIntegrator : public mfem::VectorDivergenceIntegrator
{
public:
   LeanVectorDivergenceIntegrator() = default;

   /// @param q The coefficient (borrowed); the lean path takes constants.
   explicit LeanVectorDivergenceIntegrator(mfem::Coefficient& q)
      : mfem::VectorDivergenceIntegrator(q) {}

   using mfem::BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const mfem::FiniteElementSpace& trial_fes,
                   const mfem::FiniteElementSpace& test_fes) override;
   void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const override;
   void AddMultTransposePA(const mfem::Vector& x,
                           mfem::Vector& y) const override;

   /// @return True when the last AssemblePA() took the lean path.
   bool Lean() const { return lean_; }

private:
   bool lean_ = false;         ///< The lean path is set up.
   int ne_ = 0;                ///< Elements.
   int k_ = 0;                 ///< Trial (velocity) order.
   mfem::real_t coeff_ = 1.0;  ///< The constant coefficient.
   const mfem::IntegrationRule* ir_ = nullptr;    ///< The rule (borrowed).
   const mfem::DofToQuad* trial_maps_ = nullptr;  ///< Not owned.
   const mfem::DofToQuad* test_maps_ = nullptr;   ///< Not owned.
   const mfem::GeometricFactors* geom_ = nullptr; ///< The mesh's cache.
};

} // namespace incns
