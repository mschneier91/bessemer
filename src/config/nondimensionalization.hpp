/**
 * @file nondimensionalization.hpp
 * @brief Reference scales, the convective (Reynolds) scaling, and wrappers
 *        that map dimensional user data into nondimensional coefficients.
 */
#ifndef INCNS_CONFIG_NONDIMENSIONALIZATION_HPP
#define INCNS_CONFIG_NONDIMENSIONALIZATION_HPP

#include "mfem.hpp"

#include <functional>
#include <memory>

namespace incns
{

/// How the case inputs are scaled.
enum class ScalingMode
{
   Dimensionless, ///< Inputs are already nondimensional (nu is read as 1/Re).
   Dimensional    ///< Inputs carry units; Parameters::Normalize() rescales them.
};

/**
 * @brief Reference scales for the textbook convective nondimensionalization:
 *        x* = x/L_ref, u* = u/U_ref, t* = t U_ref/L_ref, p* = p/(rho U_ref^2),
 *        turning the momentum equation into
 *        u*_t + (u*.grad*)u* = -grad* p* + (1/Re) lap* u*,  Re = U_ref L_ref/nu.
 *
 * The solver core always works in the starred variables (it is form-identical
 * to the nondimensional equations); these scales exist so the
 * Parameters/Case boundary can accept dimensional inputs and rescale
 * them in ONE place. Beyond exactness, the scaling is what makes the solver's
 * decisions meaningful: the FGMRES stopping test mixes momentum and continuity
 * rows (different units), and absolute tolerances (adaptive atol) carry
 * velocity units -- both only behave as designed on an O(1) problem.
 */
struct Nondimensionalization
{
   ScalingMode mode = ScalingMode::Dimensionless; ///< Input scaling mode.
   double L_ref = 1.0; ///< Reference length (dimensional mode).
   double U_ref = 1.0; ///< Reference velocity (dimensional mode).
   /// Density; used only to re-dimensionalize pressure on output
   /// (the solve itself is kinematic: p is p/rho throughout).
   double rho = 1.0;
   double Re = 0.0; ///< Reynolds number; filled by Parameters::Normalize().
   bool normalized = false; ///< Set by Normalize(); guards idempotence.

   /// @return The reference time T_ref = L_ref / U_ref (1 when dimensionless).
   double TRef() const { return L_ref / U_ref; }
};

/// Dimensional vector field callback: f(x_dim, t_dim) -> value in its units.
using VectorFieldFn =
   std::function<void(const mfem::Vector& x_dim, double t_dim,
                      mfem::Vector& value)>;

/**
 * @brief Wrap a DIMENSIONAL velocity field (IC or Dirichlet data) as a
 *        nondimensional coefficient.
 *
 * The mesh is built in starred coordinates, so MFEM evaluates coefficients at
 * (x*, t*); the wrapper feeds the user function dimensional arguments and
 * scales the result: @c u*(x*, t*) = f(L_ref x*, T_ref t*) / U_ref. This is
 * why dimensional-mode analytic data enters as a std::function rather than a
 * prebuilt mfem::VectorCoefficient -- an opaque coefficient computes its
 * evaluation point from the (scaled) mesh internally and cannot be re-based.
 *
 * @param f   Velocity in dimensional units of dimensional (x, t).
 * @param nd  Reference scales (Dimensional mode).
 * @param dim Spatial dimension.
 * @return The owning nondimensional coefficient.
 */
std::unique_ptr<mfem::VectorCoefficient>
WrapDimensionalVelocity(VectorFieldFn f, const Nondimensionalization& nd,
                        int dim);

/**
 * @brief Wrap a DIMENSIONAL momentum forcing as a nondimensional coefficient:
 *        @c f*(x*, t*) = f(L_ref x*, T_ref t*) * L_ref / U_ref^2.
 * @param f   Forcing (acceleration units) of dimensional (x, t).
 * @param nd  Reference scales (Dimensional mode).
 * @param dim Spatial dimension.
 * @return The owning nondimensional coefficient.
 */
std::unique_ptr<mfem::VectorCoefficient>
WrapDimensionalForcing(VectorFieldFn f, const Nondimensionalization& nd,
                       int dim);

} // namespace incns

#endif // INCNS_CONFIG_NONDIMENSIONALIZATION_HPP
