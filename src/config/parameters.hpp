/**
 * @file parameters.hpp
 * @brief Case parameters (physics, discretization, mesh, time, solver, output)
 *        and the YAML deck loader.
 */
#ifndef INCNS_CONFIG_PARAMETERS_HPP
#define INCNS_CONFIG_PARAMETERS_HPP

#include "mesh/periodic_box.hpp"
#include "time/adaptive_controller.hpp"

#include <string>

namespace incns
{

/// ParaView output settings (see post/output).
struct OutputParameters
{
   bool enabled = false;      ///< Write ParaView output during Run().
   std::string path = ".";    ///< Prefix directory for the collection.
   std::string name = "case"; ///< Collection name.
   int interval = 1;          ///< Save every this many accepted steps.
};

/**
 * @brief Everything a case needs: physics, discretization, mesh, time
 *        integration, Krylov, initial condition, and output settings.
 *
 * A case is set up by a thin driver -- a .cpp program or a YAML deck via
 * LoadYAML() -- and both go through the same library surface (StokesCase); the
 * solver core is never edited to run a new case. Every field has a sensible
 * default, so a deck only states what differs.
 */
struct Parameters
{
   // --- physics -------------------------------------------------------------
   double nu = 1.0;       ///< Kinematic viscosity.
   double grad_div = 0.0; ///< Grad-div scale c_gd (gamma = c_gd*h); 0 = off.

   // --- discretization (default Q3/Q2 Taylor-Hood) ---------------------------
   int order_u = 3;              ///< Velocity polynomial order k_u.
   int order_p = 2;              ///< Pressure polynomial order k_p.
   bool collocated_mass = false; ///< GLL collocated (diagonal) mass option.

   /// Box mesh specification (quads/hexes; per-direction periodicity).
   BoxSpec mesh;

   // --- time integration ------------------------------------------------------
   double dt = 1e-2;     ///< Fixed step size / adaptive initial guess.
   double t_final = 1.0; ///< End time.
   int time_order = 2;   ///< BDF order: 2 production, 3 test-only.
   bool adaptive = false; ///< Adaptive stepping (BDF2 advance, BDF3 estimator).
   AdaptiveControllerOptions controller; ///< Adaptive tolerances/constants.

   // --- Krylov ----------------------------------------------------------------
   double krylov_rtol = 1e-10; ///< FGMRES relative tolerance.
   double krylov_atol = 0.0;   ///< FGMRES absolute tolerance.
   int max_iter = 2000;        ///< FGMRES iteration cap.
   int kdim = 200;             ///< FGMRES restart size.
   int print_level = -1;       ///< Solver print level.

   // --- case data ---------------------------------------------------------------
   /// Named initial velocity: "zero" or "taylor_green_2d" (uses nu). Decks
   /// select analytic ICs by name so no recompilation is needed per case.
   std::string initial_velocity = "zero";

   OutputParameters output; ///< ParaView output settings.

   /**
    * @brief Load a YAML deck, overriding the defaults field by field.
    * @param path Path to the YAML file.
    * @return The populated parameters (validated; throws on malformed input).
    */
   static Parameters LoadYAML(const std::string& path);
};

} // namespace incns

#endif // INCNS_CONFIG_PARAMETERS_HPP
