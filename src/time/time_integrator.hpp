/**
 * @file time_integrator.hpp
 * @brief In-repo fixed-step time integrator for unsteady Stokes
 *        (trapezoidal starter -> BDF2 -> BDF3-in-test-mode ramp).
 */
#ifndef INCNS_TIME_TIME_INTEGRATOR_HPP
#define INCNS_TIME_TIME_INTEGRATOR_HPP

#include "bc/boundary_conditions.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <deque>
#include <memory>

namespace incns
{

/// Options for the unsteady Stokes march.
struct TimeIntegratorOptions
{
   double nu = 1.0;              ///< Kinematic viscosity.
   double dt = 1e-2;             ///< Fixed time step (adaptive is 1.9).
   double t_final = 1.0;         ///< End time for Run().
   /// BDF order: 2 = production; 3 = TEST-ONLY (marched directly solely to
   /// validate the order-3 path that drives the 1.9 LTE estimator -- never a
   /// production advancing scheme).
   int order = 2;
   bool collocated_mass = false; ///< GLL collocated mass option.
   double rtol = 1e-10;          ///< FGMRES relative tolerance.
   double atol = 0.0;            ///< FGMRES absolute tolerance.
   int max_iter = 2000;          ///< FGMRES iteration cap.
   int kdim = 200;               ///< FGMRES restart size.
   int print_level = -1;         ///< Solver print level.
};

/**
 * @brief Fixed-step implicit time integrator for unsteady Stokes -- written
 *        in-repo (deliberately NOT mfem::ODESolver).
 *
 * Each step solves the coupled saddle-point system
 * @code
 *   [ (beta0/dt) M + nu K   -B^T ] [u^{n+1}]   [ F^{n+1} - M * (history) ]
 *   [        B                0  ] [p^{n+1}] = [           0             ]
 * @endcode
 * via StokesSolver::SolveTrue, with the Dirichlet data re-eliminated at
 * t^{n+1} EVERY step (stale BC elimination silently drops temporal order; the
 * periodic TGV oracle can never catch it -- the unsteady MMS does).
 *
 * @b Startup ramp. BDF2 needs two history levels but only u^0 exists, so the
 * first step uses the trapezoidal (Crank-Nicolson) rule: LTE O(dt^3) and,
 * critically, EXACT for solutions quadratic in time -- which the unsteady
 * polynomial MMS demands of the whole ramp. A backward-Euler first step (any
 * substep size) has error ~ dt_1^2 * g'' != 0 on quadratics and can never pass
 * that gate; full-size BE would also visibly pollute the BDF3 order study.
 * Known property: the trapezoidal step's pressure is the time-average
 * (p^0 + p^1)/2, not p(t^1) -- pressure is checked from step 2 on. In BDF3
 * (test) mode the ramp extends: trapezoidal -> BDF2 -> BDF3, each startup step
 * with LTE O(dt^3), preserving the global order 3.
 *
 * BDF weights are recomputed from the ACTUAL stored node times each step via
 * multistep_coeffs (uniform here, but variable-step-ready for 1.9). The AB/EXT
 * half is dormant for Stokes -- nothing to extrapolate (pure BDF); convection
 * uses it in Sprint 2.
 */
class StokesTimeIntegrator
{
public:
   /**
    * @brief Set up the integrator (assembles one solver per active scheme).
    * @param spaces  Mixed velocity/pressure spaces (borrowed).
    * @param rules   Quadrature source (borrowed).
    * @param bc      Boundary conditions; SetTime is driven by the integrator.
    * @param forcing Momentum forcing f(x, t); SetTime is driven per step.
    * @param opts    Physics, step, and Krylov options.
    */
   StokesTimeIntegrator(MixedSpaces& spaces, const RuleBook& rules,
                        BoundaryConditions& bc,
                        mfem::VectorCoefficient& forcing,
                        const TimeIntegratorOptions& opts);

   /**
    * @brief Project the initial velocity (t = 0). Pressure has no independent
    *        initial condition (index-2 DAE: it is algebraic).
    * @param u0 Initial velocity coefficient.
    */
   void SetInitialVelocity(mfem::VectorCoefficient& u0);

   /// Advance one step of the ramp/scheme; updates Velocity()/Pressure().
   void Step();

   /// March Step() until Done().
   void Run();

   /// @return Current time.
   double Time() const { return t_; }

   /// @return True once t_final is reached (within half a step).
   bool Done() const { return t_ >= opts_.t_final - 0.5 * opts_.dt; }

   /// @return FGMRES iterations of the most recent step.
   int LastIterations() const { return last_iterations_; }

   /// @return The velocity field (updated by Step()).
   mfem::ParGridFunction& Velocity() { return u_; }

   /// @return The pressure field (updated by Step()).
   mfem::ParGridFunction& Pressure() { return p_; }

private:
   /// Assemble the forcing functional F(t) on velocity true dofs.
   void AssembleForcing(double t, mfem::Vector& F);

   MixedSpaces& spaces_;            ///< Mixed spaces (borrowed).
   const RuleBook& rules_;          ///< Quadrature source (borrowed).
   BoundaryConditions& bc_;         ///< Boundary conditions (borrowed).
   mfem::VectorCoefficient& forcing_; ///< Momentum forcing (borrowed).
   TimeIntegratorOptions opts_;     ///< Options.

   std::unique_ptr<StokesSolver> trap_;  ///< Trapezoidal starter solver.
   std::unique_ptr<StokesSolver> bdf2_;  ///< BDF2 solver.
   std::unique_ptr<StokesSolver> bdf3_;  ///< BDF3 solver (test mode only).

   mfem::ParGridFunction u_;  ///< Velocity field.
   mfem::ParGridFunction p_;  ///< Pressure field.

   std::deque<mfem::Vector> hist_;   ///< Velocity true-dof history, newest first.
   std::deque<double> hist_times_;   ///< Times of the history entries.

   double t_ = 0.0;           ///< Current time.
   int step_count_ = 0;       ///< Completed steps.
   int last_iterations_ = 0;  ///< Iterations of the most recent solve.
};

} // namespace incns

#endif // INCNS_TIME_TIME_INTEGRATOR_HPP
