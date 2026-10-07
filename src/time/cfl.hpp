/**
 * @file cfl.hpp
 * @brief Directional convective CFL number of a velocity field on its mesh
 *        (the stability ceiling for explicit convection; amr_spec.md D6).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief The convective CFL rate
 *        @f$ c = \max_K \max_q\, k_u^2 \sum_d |(J_q^{-1} u_q)_d| @f$,
 *        so that the CFL number of a step is @f$ c\,\Delta t @f$.
 *
 * @f$ J^{-1}u @f$ is the velocity in reference coordinates per unit time
 * (elements are [0,1]^dim): its d-th entry is how many element widths per
 * unit time the flow crosses along the element's d-th direction. That makes
 * the estimate directional -- a cell stretched along the flow allows a larger
 * step than one stretched across it -- which matters on anisotropically
 * refined meshes. The k_u^2 factor is the usual high-order scaling (node
 * spacing ~ h/k^2 near element edges). Global over ranks.
 *
 * Built per mesh (it caches the mesh's Jacobians); rebuild after refinement.
 */
class ConvectiveCfl
{
public:
   /**
    * @param vfes  Velocity space (vector H1, vdim = dim, tensor elements).
    * @param rules Quadrature source (the diffusion rule is used); must
    *              outlive this object and the mesh.
    */
   ConvectiveCfl(const mfem::ParFiniteElementSpace& vfes, const RuleBook& rules);

   /**
    * @brief The rate c above (collective).
    * @param u Velocity on the space given at construction.
    * @return c >= 0; the step's CFL number is c * dt.
    */
   double Rate(const mfem::ParGridFunction& u) const;

private:
   const mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int dim_ = 0;                             ///< Spatial dimension.
   int ne_ = 0;                              ///< Local element count.
   int order_ = 0;                           ///< Velocity order k_u.
   const mfem::IntegrationRule* ir_ = nullptr; ///< Evaluation rule.
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic restriction.
   const mfem::QuadratureInterpolator* qi_ = nullptr; ///< Values at points.
   const mfem::GeometricFactors* geom_ = nullptr;     ///< Jacobians.
   mutable mfem::Vector ue_; ///< Velocity E-vector buffer.
   mutable mfem::Vector uq_; ///< Velocity at quadrature points.
   mutable mfem::Vector rate_; ///< Per-element rate.
};

} // namespace incns
