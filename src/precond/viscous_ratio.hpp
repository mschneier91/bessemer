/**
 * @file viscous_ratio.hpp
 * @brief The viscous-ratio diagnostic @f$ \hat v_{\max} @f$ of the velocity
 *        block (rotational_schur_velocity_mg_spec.md 7.3): how far viscosity
 *        dominates the time derivative at the finest grid scale -- what tells
 *        you when point-block Jacobi is running out of steam.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief @f$ \hat v_{\max} = \frac{\nu}{\sigma}\,\frac{\max_a K_{aa}/
 *        M_{aa}}{c_p} @f$ with K, M the scalar stiffness and mass of the
 *        velocity order, and @f$ c_p = \max_a K_{aa}/M_{aa} / p^2 @f$ on ONE
 *        unit-square/cube element of order p.
 *
 * The normalization makes @f$ \hat v_{\max} = \nu / (\sigma (h/p)^2) @f$
 * exactly on a uniform mesh of element size h -- the viscous ratio
 * @f$ v = \nu/(\sigma h^2) @f$ at the nodal spacing -- and on a locally
 * refined mesh it reports the most refined region. The diagonals are
 * computed once per mesh (construct anew after AMR); per step it is a scalar
 * multiplication by nu/sigma. 2D and 3D (the spec is 3D only).
 */
class ViscousRatioDiagnostic
{
public:
   /**
    * @brief Assemble the diagonals on @p vfes's mesh and calibrate c_p
    *        (collective).
    * @param vfes  Velocity space (any vdim; a scalar space of the same order
    *              on the same mesh is used).
    * @param rules Quadrature source (mass rule 2p, diffusion rule
    *              2p + dim - 1 -- the solver's).
    */
   ViscousRatioDiagnostic(const mfem::ParFiniteElementSpace& vfes,
                          const RuleBook& rules);

   /**
    * @brief The diagnostic for the current coefficients.
    * @param sigma Leading mass coefficient beta0/dt (> 0).
    * @param nu    Viscosity.
    * @return @f$ \hat v_{\max} @f$.
    */
   double VHatMax(double sigma, double nu) const
   {
      return nu / sigma * r_max_ / c_p_;
   }

   /// @return The calibrated c_p of the velocity order.
   double Cp() const { return c_p_; }

private:
   double r_max_ = 0.0; ///< Global max of K_aa / M_aa on the mesh.
   double c_p_ = 0.0;   ///< Unit-element normalization.
};

} // namespace incns
