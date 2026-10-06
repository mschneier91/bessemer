/**
 * @file gradient_indicator.hpp
 * @brief Directional velocity-gradient refinement indicator (amr_spec.md,
 *        Section 2), computed on the device.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief Per element K and reference direction d, the RMS change of the
 *        velocity across K along d:
 *        @f$ G_{K,d} = \big(\tfrac{1}{|K|}\int_K \sum_c (\partial u_c/\partial\xi_d)^2\,dx\big)^{1/2} @f$,
 *        and @f$ \eta_K = |G_K| @f$.
 *
 * @f$ \partial u/\partial\xi_d = (\nabla_x u)\,J_{:,d} @f$ is the derivative
 * along the element's d-th edge direction times that edge's length, so
 * @f$ G_{K,d} @f$ has velocity units, reduces to @f$ h|\nabla u| @f$ on a cube,
 * and halves when K is split in direction d (for smooth u) while the other
 * directions are unchanged -- which is what makes it an anisotropic
 * criterion: refining the directions with large @f$ G_{K,d} @f$ is exactly
 * what reduces the indicator. Element-local, so independent of the partition.
 *
 * Reference derivatives at quadrature points come from MFEM's sum-factorized
 * QuadratureInterpolator; one device kernel reduces them per element. The rule
 * is the RuleBook's diffusion rule (Gauss-Legendre, order 2k + dim - 1), exact
 * for the integrand on affine elements. Built per adaptation event (it holds
 * mesh-dependent data).
 */
class GradientIndicator
{
public:
   /**
    * @brief Set up on the velocity space.
    * @param vfes  Velocity space (vector H1, vdim = dim, tensor elements).
    * @param rules Quadrature source; must outlive this object and the mesh's
    *              use of its rules.
    */
   GradientIndicator(const mfem::ParFiniteElementSpace& vfes,
                     const RuleBook& rules);

   /**
    * @brief Compute @f$ G_{K,d} @f$ for every local element.
    * @param u Velocity on @p vfes (a conforming L-vector).
    * @param g Output, size dim * NE, layout g[d + dim * e]; device-resident.
    */
   void Compute(const mfem::ParGridFunction& u, mfem::Vector& g) const;

   /**
    * @brief @f$ \eta_K = (\sum_d G_{K,d}^2)^{1/2} @f$ from Compute()'s output.
    * @param g   Directional indicator, layout g[d + dim * e].
    * @param dim Spatial dimension.
    * @param eta Output, size NE.
    */
   static void Eta(const mfem::Vector& g, int dim, mfem::Vector& eta);

private:
   const mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int dim_ = 0;                             ///< Spatial dimension.
   int ne_ = 0;                              ///< Local element count.
   const mfem::IntegrationRule* ir_ = nullptr; ///< Indicator rule (RuleBook).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic restriction.
   const mfem::QuadratureInterpolator* qi_ = nullptr; ///< Reference grads.
   const mfem::GeometricFactors* geom_ = nullptr;     ///< det J at points.
   mutable mfem::Vector ue_;                 ///< Velocity E-vector buffer.
   mutable mfem::Vector qd_;                 ///< Reference-gradient buffer.
};

} // namespace incns
