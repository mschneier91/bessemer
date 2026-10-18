/**
 * @file gradient_indicator.hpp
 * @brief Directional velocity-gradient refinement indicator (docs/design/amr_spec.md,
 *        Section 2), computed on the device.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

class ElementScratch;

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
 * Reference derivatives at quadrature points: in 3D one sum-factorized device
 * kernel makes them per element and reduces them as it goes, with no
 * quadrature-point buffer (9 values per point: 2.65 GB at 72k Q6 elements,
 * on top of the live time integrator at an adaptation event's first pass).
 * In 2D, and past the device's 1D size limits, MFEM's QuadratureInterpolator
 * makes them into such a buffer and a second kernel reduces it. The rule is
 * the RuleBook's diffusion rule (Gauss-Legendre, order 2k + dim - 1), exact
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
    * @brief Restrict the velocity into a buffer of a time integrator's
    *        operator scratch instead of one of this object's own (0.59 GB at
    *        72k Q6 elements). The scratch's operators must not run between
    *        this call and the last Compute().
    * @param scratch Scratch over the same velocity space; must outlive this.
    */
   void UseScratch(ElementScratch& scratch);

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
   const mfem::GeometricFactors* geom_ = nullptr;     ///< det J at points.
   /// The 1D basis and its derivative at the rule's 1D points (the fused
   /// kernel's; null on the QuadratureInterpolator path).
   const mfem::DofToQuad* maps_ = nullptr;
   /// Reference gradients (the QuadratureInterpolator path; else null).
   const mfem::QuadratureInterpolator* qi_ = nullptr;
   mutable mfem::Vector ue_;                 ///< Velocity E-vector buffer.
   /// Reference-gradient buffer (the QuadratureInterpolator path; else empty).
   mutable mfem::Vector qd_;
};

} // namespace incns
