/**
 * @file kinetic_head.hpp
 * @brief Nodal interpolant of the kinetic head 1/2 |u|^2 on the pressure
 *        space, computed on the device (static pressure for the rotational
 *        form).
 */
#ifndef INCNS_POST_KINETIC_HEAD_HPP
#define INCNS_POST_KINETIC_HEAD_HPP

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Computes @f$ I_p(\tfrac12|u_h|^2) @f$: half the squared speed
 *        evaluated at the pressure space's nodes (nodal interpolation) -- the
 *        term that turns the rotational form's Bernoulli head
 *        @f$ P = p + \tfrac12|u|^2 @f$ back into static pressure.
 *
 * Device path: the velocity E-vector is evaluated at the pressure nodes by a
 * sum-factorized x -> y -> z contraction (forall_2D, max(D1D, Q1D)^2 threads
 * per element, shared tiles sized by DofQuadLimits), squared and halved
 * pointwise, and mapped to the pressure L-vector with the element
 * restriction's left inverse (u_h is continuous, so every element sharing a
 * node yields the same value). O(p^4) per element in 3D. Beyond the device's
 * tile limits it falls back to InterpolateHost, which is also the reference
 * the device path is tested against.
 *
 * Requirements: one mesh, quads/hexes, velocity vdim == dim, tensor H1 on
 * both spaces, and a NODAL pressure basis (GLL -- MFEM's default -- or any
 * other nodal 1D point set).
 */
class KineticHeadInterpolator
{
public:
   /**
    * @brief Set up the 1D velocity-at-pressure-node tables and buffers.
    * @param vfes Velocity space (vdim == dim).
    * @param pfes Pressure space (scalar, nodal tensor H1, same mesh).
    */
   KineticHeadInterpolator(const mfem::ParFiniteElementSpace& vfes,
                           const mfem::ParFiniteElementSpace& pfes);

   /**
    * @brief @p ke = I_p(1/2 |u|^2) as a pressure L-vector, on the device.
    * @param u  Velocity field; its L-vector must be up to date (distributed).
    * @param ke Output, on the pressure space given at construction.
    */
   void Interpolate(const mfem::ParGridFunction& u,
                    mfem::ParGridFunction& ke) const;

   /**
    * @brief The host reference: ProjectCoefficient of 1/2 |u_h|^2 (element by
    *        element, GridFunction::GetVectorValue). Fallback beyond the tile
    *        limits.
    * @param u  Velocity field (distributed).
    * @param ke Output, on a pressure space with a nodal basis.
    */
   static void InterpolateHost(const mfem::ParGridFunction& u,
                               mfem::ParGridFunction& ke);

   /// @return Whether Interpolate uses the device kernel (else the host path).
   bool UsesDevicePath() const { return device_path_; }

private:
   int dim_ = 0;                ///< Spatial dimension (2 or 3).
   int ne_ = 0;                 ///< Number of local elements.
   int d1d_ = 0;                ///< Velocity dofs per direction.
   int q1d_ = 0;                ///< Pressure nodes per direction.
   bool device_path_ = false;   ///< Sizes within the device tile limits.
   /// 1D velocity basis at the pressure nodes: (Q1D, D1D), column-major.
   mfem::Vector B_;
   /// Lexicographic element restriction of the velocity space (not owned).
   const mfem::Operator* u_restr_ = nullptr;
   /// Lexicographic element restriction of the pressure space (not owned).
   const mfem::ElementRestriction* p_restr_ = nullptr;
   mutable mfem::Vector u_e_;  ///< Velocity E-vector buffer (reused).
   mutable mfem::Vector ke_e_; ///< Kinetic-head E-vector buffer (reused).
};

} // namespace incns

#endif // INCNS_POST_KINETIC_HEAD_HPP
