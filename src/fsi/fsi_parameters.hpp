/**
 * @file fsi_parameters.hpp
 * @brief Settings of a fluid-structure interaction run (deck section `fsi`):
 *        the partitioned coupling of the ALE fluid with the solid, iterated
 *        in every step (docs/fsi.md).
 */
#pragma once

#include <vector>
#include <string>

namespace incns
{

/// The interface conditions of the partitioned coupling (`fsi.coupling`).
enum class FsiCoupling
{
   /// Dirichlet on the fluid (the mesh and u = w follow the solid's
   /// interface displacement), Neumann on the solid (the fluid's traction).
   DirichletNeumann
};

/// How the interface displacement is relaxed between subiterations.
enum class FsiAcceleration
{
   Aitken,   ///< Dynamic Aitken relaxation (Kuttler and Wall 2008).
   Constant, ///< A fixed relaxation factor (`fsi.relaxation`).
   /// Interface quasi-Newton, least-squares inverse Jacobian (Degroote et
   /// al. 2009).
   IqnIls
};

/**
 * @brief A fluid-structure interaction run: the flow of the deck (its mesh,
 *        boundary groups, physics and time steps; the groups of type
 *        `interface` are the wetted surface) coupled with the solid of
 *        section `solid` on its own mesh.
 */
struct FsiParameters
{
   bool enabled = false; ///< Whether the deck is an FSI run.
   /// Boundary names (or attributes) of the solid mesh held fixed.
   std::vector<std::string> solid_fixed;
   /// The interface's boundary names on the solid mesh; empty: the names
   /// of the fluid's interface groups.
   std::vector<std::string> solid_interface;
   double fluid_density = 1.0; ///< rho_f: the fluid's traction is rho_f times
   ///< the kinematic one.
   FsiCoupling coupling = FsiCoupling::DirichletNeumann; ///< Conditions.
   FsiAcceleration acceleration = FsiAcceleration::Aitken; ///< Relaxation.
   /// The first subiteration's relaxation factor (Aitken) or every one's
   /// (constant), in (0, 1].
   double relaxation = 0.5;
   double omega_max = 1.0; ///< Aitken's factor is kept in [-omega_max, omega_max].
   double rtol = 1e-6; ///< Stop when |r| <= rtol |d| + atol (r: the interface
   ///< displacement residual, d: the displacement).
   double atol = 1e-12; ///< Absolute part of the stopping test.
   int max_iterations = 50; ///< Subiterations per step before giving up.
   /// 1: print every subiteration's residual on rank 0; 0: nothing.
   int print_level = 0;
   /// IQN-ILS: earlier steps whose columns are kept.
   int reuse = 0;
   /// Order of the interface displacement's predictor: 0 the last step's,
   /// 1 linear, 2 quadratic extrapolation of the committed steps.
   int predictor = 2;
};

} // namespace incns
