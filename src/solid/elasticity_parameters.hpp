/**
 * @file elasticity_parameters.hpp
 * @brief Settings of the elasticity solvers (deck section `solid`):
 *        material, plane model, displacement order, linear solver, time
 *        integration, and the hyperelastic model with its Newton settings.
 */
#pragma once

#include <array>

namespace incns
{

/// What a 2D elasticity case models (3D ignores it).
enum class PlaneModel
{
   Strain, ///< Plane strain: the 2D Lame parameters are the 3D ones.
   Stress  ///< Plane stress: lambda* = 2 lambda mu / (lambda + 2 mu).
};

/// Preconditioner of the elasticity CG.
enum class ElasticityPC
{
   /// One BoomerAMG V-cycle (systems mode, by nodes) on the low-order-refined
   /// (Q1 on the GLL nodes) rediscretization: mesh-independent iterations.
   LorAmg,
   /// The operator's diagonal: no setup and no assembled matrix, iterations
   /// growing with the resolution.
   Jacobi
};

/// Static equilibrium or elastodynamics (deck `solid.analysis`).
enum class SolidAnalysis
{
   Static, ///< One solve of K u = f.
   Dynamic ///< rho u_tt + K u = f(t), marched from t = 0 to time.t_final.
};

/// Time integrator of elastodynamics (deck `solid.time_scheme`).
enum class SolidTimeScheme
{
   /// Chung-Hulbert generalized-alpha on (u, v, a): second order,
   /// unconditionally stable, high-frequency damping set by rho_inf.
   GeneralizedAlpha,
   /// The fluid's variable-step BDF2/BDF3 on the first-order system (u, v),
   /// started by one trapezoidal step.
   Bdf
};

/// The hyperelastic material (deck `solid.model`, equation hyperelasticity).
enum class SolidModel
{
   /// St. Venant-Kirchhoff: W = lambda/2 (tr E)^2 + mu E : E.
   StVenantKirchhoff
};

/**
 * @brief Chung-Hulbert generalized-alpha parameters (J. Appl. Mech. 60
 *        (1993) 371-375).
 */
struct GeneralizedAlphaCoefficients
{
   double alpha_m; ///< Weight of a_n in the inertia term.
   double alpha_f; ///< Weight of t_n in the stiffness and load terms.
   double beta;    ///< Newmark beta.
   double gamma;   ///< Newmark gamma.
};

/**
 * @param rho_inf Spectral radius at infinite frequency, in [0, 1]: 1 is the
 *        trapezoidal rule (no numerical damping), 0 annihilates the highest
 *        modes in one step.
 * @return alpha_m = (2 rho_inf - 1)/(rho_inf + 1), alpha_f = rho_inf/(rho_inf
 *         + 1), gamma = 1/2 - alpha_m + alpha_f, beta = (1 - alpha_m +
 *         alpha_f)^2 / 4: second order with minimal low-frequency damping.
 */
inline GeneralizedAlphaCoefficients GeneralizedAlpha(double rho_inf)
{
   GeneralizedAlphaCoefficients c;
   c.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0);
   c.alpha_f = rho_inf / (rho_inf + 1.0);
   c.gamma = 0.5 - c.alpha_m + c.alpha_f;
   const double s = 1.0 - c.alpha_m + c.alpha_f;
   c.beta = 0.25 * s * s;
   return c;
}

/**
 * @brief Elasticity settings: an isotropic, homogeneous material (E,
 *        Poisson's ratio), displacement-only; small strains (equation
 *        linear_elasticity) or a hyperelastic model (equation hyperelasticity).
 */
struct ElasticityParameters
{
   double youngs_modulus = 1.0; ///< Young's modulus E (> 0).
   double poisson_ratio = 0.3;  ///< Poisson's ratio (-1 < nu < 1/2).
   PlaneModel plane = PlaneModel::Strain; ///< 2D model.
   /// Displacement polynomial order; <= 0: the fluid's velocity order
   /// (discretization.order_u), so the two match at a future interface.
   int order = 0;
   /// A constant body force per unit volume (deck `solid.body_force`).
   std::array<double, 3> body_force = {0.0, 0.0, 0.0};
   double rtol = 1e-10;   ///< CG relative tolerance.
   double atol = 0.0;     ///< CG absolute tolerance.
   int max_iter = 2000;   ///< CG iteration cap.
   ElasticityPC preconditioner = ElasticityPC::LorAmg; ///< CG preconditioner.
   /// Point (deck `solid.probe`) whose displacement the summary reports;
   /// unused unless has_probe.
   std::array<double, 3> probe = {0.0, 0.0, 0.0};
   bool has_probe = false; ///< Whether solid.probe was given.

   SolidAnalysis analysis = SolidAnalysis::Static; ///< Static or dynamic.
   double density = 1.0; ///< Mass density rho (> 0; dynamics only).
   /// Time integrator of a dynamic analysis.
   SolidTimeScheme time_scheme = SolidTimeScheme::GeneralizedAlpha;
   int bdf_order = 2;    ///< BDF order (2 or 3) of time_scheme bdf.
   double rho_inf = 0.8; ///< Generalized-alpha high-frequency radius, [0, 1].

   /// Equation hyperelasticity: the material.
   SolidModel model = SolidModel::StVenantKirchhoff;
   double newton_rtol = 1e-8; ///< Newton residual reduction per load step.
   double newton_atol = 0.0;  ///< Newton absolute residual tolerance.
   /// Newton: converged also when ||dx|| <= newton_dx_rtol ||u|| (0: off).
   double newton_dx_rtol = 1e-10;
   int newton_max_iter = 25;  ///< Newton iterations per load step.
   int load_steps = 1;        ///< Load steps of a static solve.
   bool line_search = false;  ///< Newton's backtracking line search.

   /// @param order_u The fluid's velocity order. @return The order in effect.
   int Order(int order_u) const { return order > 0 ? order : order_u; }

   /// @return The shear modulus mu = E / (2 (1 + nu)).
   double Mu() const { return youngs_modulus / (2.0 * (1.0 + poisson_ratio)); }

   /**
    * @param dim Spatial dimension.
    * @return The first Lame parameter of the model in effect: E nu / ((1 +
    *         nu)(1 - 2 nu)), or in 2D plane stress E nu / (1 - nu^2).
    */
   double Lambda(int dim) const
   {
      const double E = youngs_modulus, nu = poisson_ratio;
      if (dim == 2 && plane == PlaneModel::Stress)
      {
         return E * nu / (1.0 - nu * nu);
      }
      return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
   }
};

} // namespace incns
