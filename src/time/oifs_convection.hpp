/**
 * @file oifs_convection.hpp
 * @brief The OIFS substeps' convection, sum_l w_l (u_l . grad) x_c for every
 *        velocity component c at once, with the advecting velocities u_l
 *        combined at each quadrature point (3D).
 */
#pragma once

#include "operators/element_scratch.hpp"
#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief y_c = sum_l w_l C(u_l) x_c for every component c of a velocity-space
 *        vector, where C(u) is the partially assembled form (u . grad phi, v)
 *        of MFEM's ConvectionIntegrator (alpha = 1) on the scalar companion
 *        space, and up to three advecting velocities u_l are set by SetWind()
 *        and weighted per Mult().
 *
 * The form is linear in the wind, so the quadrature data of the u_l
 * (W adj(J) u_l at each point, as MFEM's PAConvectionSetup3D computes it) is
 * combined with the weights at each point inside one kernel, which then
 * applies it to all three components: one kernel and one pass over the wind
 * data per apply, where OifsAdvector ran one MFEM apply per component and wind
 * (nine per RK4 stage with three winds). The results agree with that sum to
 * roundoff (oifs_convection_test).
 *
 * 3D only, for the (dofs, points) per direction the solver uses with velocity
 * orders 2 to 4 under OIFS's 3k rule (Supported()); OifsAdvector keeps MFEM's
 * forms otherwise.
 */
class OifsConvection
{
public:
   /**
    * @brief Whether this space and rule have a kernel: 3D, vdim 3, byNODES,
    *        tensor elements with (dofs, points) per direction (3, 4), (4, 5) or
    *        (5, 7).
    * @param vfes Velocity space.
    * @param ir   Quadrature rule of the convection forms.
    * @return True when OifsConvection can be built for them.
    */
   static bool Supported(const mfem::ParFiniteElementSpace& vfes,
                         const mfem::IntegrationRule& ir);

   /**
    * @brief Bind to the velocity space and the rule (Supported() must hold).
    * @param vfes    Velocity space (borrowed).
    * @param ir      Quadrature rule (borrowed).
    * @param scratch Element-vector buffers for the wind, input and output
    *                element vectors (borrowed; must outlive this); null:
    *                its own.
    */
   OifsConvection(mfem::ParFiniteElementSpace& vfes,
                  const mfem::IntegrationRule& ir,
                  ElementScratch* scratch = nullptr);

   /**
    * @brief Set advecting velocity @p l: its quadrature data.
    * @param l Index, 0 to 2.
    * @param u The velocity (on the bound space, local dofs valid).
    */
   void SetWind(int l, const mfem::ParGridFunction& u);

   /**
    * @brief y := sum_{l < w.size()} w[l] C(u_l) x, component by component.
    * @param w Weights, at most 3, of winds already set.
    * @param x Input, velocity true dofs.
    * @param y Output, velocity true dofs (overwritten).
    */
   void Mult(const std::vector<double>& w, const mfem::Vector& x,
             mfem::Vector& y);

private:
   mfem::ParFiniteElementSpace& vfes_;       ///< Velocity space (borrowed).
   const mfem::IntegrationRule& ir_;         ///< Rule (borrowed).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic E-restriction.
   /// True -> local dofs (null when it is the identity: one rank).
   const mfem::Operator* prolong_ = nullptr;
   const mfem::DofToQuad* maps_ = nullptr;   ///< 1D maps at ir_ (not owned).
   const mfem::GeometricFactors* geom_ = nullptr; ///< Jacobians (mesh cache).
   /// Jacobians computed here (nq x 3 x 3 x ne) when MFEM's quadrature
   /// interpolator cannot run (Columns()), and which ones SetWind reads.
   mfem::Vector jac_;
   const mfem::Vector* j_ = nullptr;
   /// Q1D > 8 (velocity orders 6 and 7 at the 3k rule): MFEM's quadrature
   /// interpolator has no compile-time kernel for the velocity there, its
   /// runtime-size ones hold 8 points per direction (DofQuadLimits::
   /// MAX_INTERP_1D on CUDA) and launch a thread per point (more than a block
   /// holds at Q1D 11). The Jacobians and the winds' values are then computed
   /// here, a thread per (x, y) column.
   bool Columns() const { return q1_ > 8; }
   int ne_ = 0;  ///< Local elements.
   int d1_ = 0;  ///< Dofs per direction.
   int q1_ = 0;  ///< Points per direction.
   int set_ = 0; ///< Winds set (bit l: SetWind(l) called).
   /// Quadrature data of the three winds, (Q1D^3, 3, NE, 3).
   mfem::Vector op_;
   mfem::Vector ue_; ///< Wind, element vector (scratch).
   mfem::Vector uq_; ///< Wind at the points, (3, NQ, NE) (scratch).
   mfem::Vector xl_; ///< Input, local dofs (scratch).
   mfem::Vector xe_; ///< Input, element vector (scratch).
   mfem::Vector ye_; ///< Output, element vector (scratch).
   mfem::Vector yl_; ///< Output, local dofs (scratch).
};

} // namespace incns
