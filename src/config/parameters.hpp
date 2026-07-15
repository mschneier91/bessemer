/**
 * @file parameters.hpp
 * @brief Case parameters (physics, discretization, mesh, time, solver, output)
 *        and the YAML deck loader.
 */
#ifndef INCNS_CONFIG_PARAMETERS_HPP
#define INCNS_CONFIG_PARAMETERS_HPP

#include "config/nondimensionalization.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/velocity_preconditioner.hpp"
#include "time/adaptive_controller.hpp"

#include <string>
#include <vector>

namespace incns
{

/// The equation set a case solves -- a runtime option; the unified Case
/// dispatches on it (never a class per physics).
enum class Equation
{
   Stokes,      ///< Unsteady Stokes (Sprint 1).
   NavierStokes ///< Incompressible Navier-Stokes (Sprint 2; not yet available).
};

/// The kind of boundary condition a deck-declared group carries.
enum class BcType
{
   VelocityDirichlet, ///< Prescribed velocity (the field is bound in the driver).
   NoSlip,            ///< Zero velocity wall (no field needed).
   Outflow            ///< Do-nothing / natural traction (no field).
};

/**
 * @brief A boundary-condition group declared in the deck: a label, a boundary
 *        selection, and a type. The field for a Dirichlet group is bound in the
 *        driver by group name (fields never live in the deck).
 *
 * The selection mixes box face names (@c "xmin" ... resolved geometrically),
 * explicit boundary attribute integers, or @c select_all for every real
 * boundary -- MFEM attribute-marker style, by name or number.
 */
struct BcSpec
{
   std::string group;              ///< Label for binding a Dirichlet field.
   BcType type = BcType::VelocityDirichlet; ///< Dirichlet or outflow.
   bool select_all = false;        ///< Select every real (non-periodic) face.
   std::vector<std::string> faces; ///< Face-name selectors (box convenience).
   std::vector<int> attributes;    ///< Explicit boundary attribute selectors.
};

/// Rolling checkpoint settings (see post/checkpoint).
struct CheckpointParameters
{
   bool enabled = false;     ///< Write a rolling checkpoint during Run().
   std::string path = "chk"; ///< Checkpoint directory (overwritten each write).
   int interval = 10;        ///< Write every this many accepted steps.
};

/// ParaView output settings (see post/output).
struct OutputParameters
{
   bool enabled = false;      ///< Write ParaView output during Run().
   std::string path = ".";    ///< Prefix directory for the collection.
   std::string name = "case"; ///< Collection name.
   int interval = 1;          ///< Save every this many accepted steps.
   bool diagnostics = false;  ///< Also log KE/dissipation/||div u|| to a CSV.
};

/**
 * @brief Everything a case needs: physics, discretization, mesh, time
 *        integration, Krylov, initial condition, and output settings.
 *
 * A case is set up by a thin driver -- a .cpp program or a YAML deck via
 * LoadYAML() -- and both go through the same library surface (Case); the
 * solver core is never edited to run a new case. Every field has a sensible
 * default, so a deck only states what differs.
 */
struct Parameters
{
   // --- backend -------------------------------------------------------------
   /// MFEM device backend ("cpu", "cuda", "hip", ...); configured once at case
   /// setup before any mesh/space is built. Default CPU.
   std::string device = "cpu";

   // --- physics -------------------------------------------------------------
   /// Equation set to solve. NavierStokes is not available until Sprint 2; the
   /// Case rejects it cleanly for now (interface lands, wiring comes later).
   Equation equation = Equation::Stokes;
   /// Kinematic viscosity. In Dimensionless mode this is read as 1/Re (a deck
   /// may equivalently give `physics: Re`); in Dimensional mode it carries
   /// units and Normalize() converts it to 1/Re.
   double nu = 1.0;
   double grad_div = 0.0; ///< Grad-div scale c_gd (gamma = c_gd*h); 0 = off.

   /// Input scaling: mode + reference scales (see nondimensionalization.hpp).
   Nondimensionalization nondim;

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
   /// Velocity-block preconditioner (deck `solver.preconditioner: jacobi|amg`).
   VelocityPreconditioner velocity_prec = VelocityPreconditioner::Jacobi;

   // --- case data ---------------------------------------------------------------
   /// Named initial velocity: "zero" or "taylor_green_2d" (uses nu). Decks
   /// select analytic ICs by name so no recompilation is needed per case.
   std::string initial_velocity = "zero";

   /// Boundary-condition groups (topology only; Dirichlet fields are bound in
   /// the driver by group name). Empty means the driver sets BCs explicitly (or
   /// a fully periodic mesh needs none).
   std::vector<BcSpec> boundary_conditions;

   OutputParameters output; ///< ParaView output settings.

   CheckpointParameters checkpoint; ///< Rolling checkpoint settings.
   /// When non-empty, restore the marching state from this checkpoint
   /// directory before stepping (same-np restart; see post/checkpoint).
   std::string restart_from;

   /**
    * @brief Apply the convective nondimensionalization in place (idempotent).
    *
    * Dimensionless mode: records Re = 1/nu, nothing else changes. Dimensional
    * mode: rescales mesh lengths (/L_ref), dt and t_final (*U_ref/L_ref),
    * nu -> 1/Re with Re = U_ref*L_ref/nu, and the adaptive atol (/U_ref --
    * the LTE is velocity-normed); named initial conditions other than "zero"
    * are rejected (registry entries are inherently nondimensional -- provide
    * dimensional data through WrapDimensionalVelocity instead).
    *
    * MUST run after the parameters are filled and BEFORE the mesh is built
    * (Case verifies this). LoadYAML() calls it automatically; in-code
    * drivers call it themselves.
    */
   void Normalize();

   /**
    * @brief Load a YAML deck, overriding the defaults field by field, and
    *        Normalize() the result.
    * @param path Path to the YAML file.
    * @return The populated, normalized parameters (throws on malformed input).
    */
   static Parameters LoadYAML(const std::string& path);
};

} // namespace incns

#endif // INCNS_CONFIG_PARAMETERS_HPP
