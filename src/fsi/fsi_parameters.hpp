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
   /// Robin on the fluid (sigma_f n + alpha_f u = the last subiteration's
   /// traction + alpha_f w, the mesh following the solid's interface),
   /// Neumann on the solid (the fluid's traction), iterated on the traction.
   /// alpha_f ~ rho_s H / dt takes the solid's inertia into the fluid's
   /// solve (Badia, Nobile and Vergara 2008); on a wall wetted on both
   /// sides its through-thickness squeeze mode converges slowly (docs/fsi.md).
   RobinNeumann,
   /// Dirichlet on the fluid (the mesh and u = w follow the solid's
   /// interface displacement), Neumann on the solid (the fluid's traction).
   DirichletNeumann,
   /// Neumann on the fluid (the solid's interface reaction as a traction,
   /// the mesh following the iterate), Dirichlet on the solid (the
   /// displacement the fluid's interface velocity integrates to), iterated
   /// on the interface displacement.
   NeumannDirichlet,
   /// Robin on both: the fluid's as Robin-Neumann's, the solid's
   /// sigma_s n_s + alpha_s w = -sigma_f n + alpha_s u (an elastic support
   /// alpha_s c0 in its stiffness, c0 the mesh velocity's BDF weight),
   /// iterated on the solid's Robin data. alpha_s = 0 is Robin-Neumann.
   RobinRobin
};

/// The solid's model (`fsi.solid`).
enum class FsiSolidModel
{
   /// Small-strain linear elasticity (LinearElasticity): the load on the
   /// reference configuration, no geometric stiffening or shortening.
   LinearElasticity,
   /// Large deformation, St. Venant-Kirchhoff by default (Hyperelasticity,
   /// `solid.model`): the Turek-Hron benchmarks' solid.
   Hyperelasticity
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
   /// The solid's model.
   FsiSolidModel solid = FsiSolidModel::LinearElasticity;
   /// The fluid's Robin coefficient alpha_f (a density times a velocity);
   /// <= 0: Badia, Nobile and Vergara's rho_s H / dt (Parameters::
   /// FsiFluidRobin()).
   double alpha_fluid = 0.0;
   /// The structure's thickness H in the default alpha_f; <= 0: the
   /// geometry's flag thickness.
   double structure_thickness = 0.0;
   /// Robin-Robin: the solid's Robin coefficient alpha_s (a density times a
   /// velocity), > 0.
   double alpha_solid = 0.0;
   /// Relaxation: IQN-ILS by default (Aitken fails FSI1's first step under
   /// either condition; docs/fsi.md).
   FsiAcceleration acceleration = FsiAcceleration::IqnIls;
   /// The first subiteration's relaxation factor (Aitken) or every one's
   /// (constant), in (0, 1].
   double relaxation = 0.5;
   double omega_max = 1.0; ///< Aitken's factor is kept in [-omega_max, omega_max].
   double rtol = 1e-6; ///< Stop when |r| <= rtol |d| + atol (r: the interface
   ///< displacement residual, d: the displacement).
   double atol = 1e-12; ///< Absolute part of the stopping test.
   int max_iterations = 50; ///< Subiterations per step before giving up.
   /// A failed solid solve on a subiteration after the first halves its
   /// input toward the last one whose subiteration completed, at most this
   /// many times, before the attempt fails (0: at once).
   int backtracks = 4;
   /// 1: print every subiteration's residual on rank 0; 0: nothing.
   int print_level = 0;
   /// IQN-ILS: earlier steps whose columns are kept.
   int reuse = 0;
   /// Order of the interface displacement's predictor: 0 the last step's,
   /// 1 linear, 2 quadratic extrapolation of the committed steps.
   int predictor = 2;
};

} // namespace incns
