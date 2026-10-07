/**
 * @file body_force.hpp
 * @brief Force exerted by the fluid on a body (lift and drag) by the
 *        volume-integral ("weak") formulation of V. John.
 */
#pragma once

#include "mfem.hpp"

#include <vector>

namespace incns
{

/// Deck section `forces:` -- which boundary is the body, and the reference
/// scales of the force coefficients.
struct ForceParameters
{
   /// Compute and log the body force (needs at least one attribute).
   bool enabled = false;
   /// Boundary attributes forming the body S (no-slip walls, e.g. a cylinder).
   std::vector<int> attributes;
   /// Reference velocity U of C = 2 F / (rho U^2 A) (nondimensional; the
   /// DFG cylinder benchmark uses the mean inflow velocity).
   double reference_velocity = 1.0;
   /// Reference area A (2D: a length -- the cylinder diameter in the DFG
   /// benchmark; 3D: an area). Nondimensional.
   double reference_area = 1.0;
   /// Log every this many accepted steps to
   /// `<output.path>/<output.name>_forces.csv`; 0 = no log (the force is
   /// still available through the API).
   int interval = 1;
};

/**
 * @brief Force F on the body S from the discrete momentum residual, by the
 *        volume-integral formulation of John (Int. J. Numer. Meth. Fluids 44,
 *        2004; following John & Matthies 2001):
 *        @f$ F_i = -\big[(\partial_t u, v_i) + \nu(\nabla u, \nabla v_i)
 *        + n(u; v_i) - (p, \nabla\cdot v_i) - (f, v_i)\big] @f$,
 *        with @f$ v_i @f$ any function equal to @f$ e_i @f$ on S and zero on
 *        the rest of the boundary.
 *
 * Integrating the momentum equation against v_i by parts shows this equals
 * @f$ \int_S \sigma n\,ds @f$ (n pointing from the body into the fluid,
 * @f$ \sigma = -pI + \nu\nabla u @f$) for the exact solution, but the volume
 * form is far more accurate for a finite element solution than integrating the
 * traction over S -- it is superconvergent, and it is what the DFG benchmark
 * reference values were computed with.
 *
 * Discretely, v_i is John's choice: the finite element function equal to e_i
 * at the velocity nodes on S and zero at every other node. The bracket is
 * evaluated as the residual of the time step that produced (u, p) --
 * StokesTimeIntegrator::MomentumResidual, with the scheme's own time
 * derivative, viscous, grad-div, convection (extrapolated or rotational) and
 * forcing terms -- so F_i = -r . v_i. Because that residual vanishes at every
 * free dof (to Krylov tolerance), the result depends only on v_i's values on
 * the boundary, exactly as in the continuous identity. The pressure in the
 * rotational form is the Bernoulli head P = p + 1/2|u|^2, which equals p on a
 * no-slip body.
 *
 * Coefficients: C_i = 2 F_i / (rho U^2 A) with rho = 1 (nondimensional).
 * Drag is the x-component, lift the y-component.
 */
class BodyForce
{
public:
   /**
    * @brief Build the test functions v_i on @p vfes.
    * @param vfes        Velocity space (vector H1, vdim = dim).
    * @param attributes  Boundary attributes forming the body.
    */
   BodyForce(mfem::ParFiniteElementSpace& vfes,
             const std::vector<int>& attributes);

   /**
    * @brief The force from a momentum residual (collective).
    * @param r Residual on velocity true dofs (MomentumResidual()).
    * @return F, size dim: F_i = -r . v_i (global).
    */
   mfem::Vector Force(const mfem::Vector& r) const;

   /**
    * @brief -r . v for an arbitrary test vector (collective) -- for tests of
    *        the choice-independence of v away from the boundary.
    * @param r Residual on velocity true dofs.
    * @param v Test function on velocity true dofs.
    * @return -r . v.
    */
   double ForceWith(const mfem::Vector& r, const mfem::Vector& v) const;

   /**
    * @brief The test function v_i on true dofs.
    * @param i Force component (0 <= i < dim).
    * @return v_i.
    */
   const mfem::Vector& TestFunction(int i) const { return v_[i]; }

   /**
    * @brief Force coefficients C = 2 F / (U^2 A).
    * @param F Force (from Force()).
    * @param U Reference velocity.
    * @param A Reference area (2D: length).
    * @return C, same size as F.
    */
   static mfem::Vector Coefficients(const mfem::Vector& F, double U, double A);

private:
   MPI_Comm comm_;               ///< Velocity space communicator.
   std::vector<mfem::Vector> v_; ///< v_i on true dofs, i < dim.
};

} // namespace incns
