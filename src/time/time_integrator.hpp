/**
 * @file time_integrator.hpp
 * @brief In-repo time integrator for unsteady Stokes: fixed-step
 *        (trapezoidal starter -> BDF2 -> BDF3-in-test-mode) and adaptive
 *        (BDF2 advancing, BDF3 LTE estimator, PI controller).
 */
#ifndef INCNS_TIME_TIME_INTEGRATOR_HPP
#define INCNS_TIME_TIME_INTEGRATOR_HPP

#include "bc/boundary_conditions.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/adaptive_controller.hpp"
#include "mfem.hpp"

#include <deque>
#include <memory>

namespace incns
{

/// Options for the unsteady Stokes march.
struct TimeIntegratorOptions
{
   double nu = 1.0;              ///< Kinematic viscosity.
   double dt = 1e-2;             ///< Fixed step size / adaptive initial guess.
   double t_final = 1.0;         ///< End time for Run().
   /// BDF order: 2 = production; 3 = TEST-ONLY (marched directly solely to
   /// validate the order-3 path that drives the adaptive LTE estimator --
   /// never a production advancing scheme). Ignored when adaptive is on
   /// (the solution always advances with the order-2 step).
   int order = 2;
   /// Adaptive stepping (default stays fixed-step BDF2). The LTE is estimated
   /// from the difference between the BDF2 and BDF3 solutions each step;
   /// control acts on VELOCITY only (index-2 DAE: pressure is algebraic).
   bool adaptive = false;
   /// Controller tolerances and constants (used when adaptive is on).
   AdaptiveControllerOptions controller;
   bool collocated_mass = false; ///< GLL collocated mass option.
   double grad_div =
      0.0;        ///< Grad-div scale c_gd (gamma = c_gd*h); 0 = off.
   /// Velocity-block preconditioner (Jacobi default; BoomerAMG option).
   VelocityPreconditioner velocity_prec = VelocityPreconditioner::Jacobi;
   double rtol = 1e-10;          ///< FGMRES relative tolerance.
   double atol = 0.0;            ///< FGMRES absolute tolerance.
   int max_iter = 2000;          ///< FGMRES iteration cap.
   int kdim = 200;               ///< FGMRES restart size.
   int print_level = -1;         ///< Solver print level.
};

/**
 * @brief Time integrator for unsteady Stokes -- written in-repo (deliberately
 *        NOT mfem::ODESolver).
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
 * @b Adaptive mode. From the third step on (three history levels), each
 * attempt solves BOTH the BDF2 and BDF3 candidates from the same accepted
 * history; the solution ALWAYS advances with the order-2 candidate (no local
 * extrapolation), the order-3 one exists only for the LTE estimate
 * ||u2 - u3||. The AdaptiveController accepts/rejects (retrying rejected
 * steps at its proposed dt) and proposes the next step through the mixed
 * atol/rtol test and the PI law, capped by the pluggable dt ceiling (inert in
 * Sprint 1; Sprint 2 wires the convective CFL). The step history is recorded
 * and exposed via Controller().
 *
 * The momentum operator depends on beta0/dt, so solvers are cached per
 * leading BDF weight and rebuilt whenever it changes (cheap with the Jacobi
 * default -- a design reason AMG is not the default; the Sprint-1 Schur block
 * is dt-independent). BDF weights are recomputed from the ACTUAL stored node
 * times each step via multistep_coeffs -- never uniform-step values under a
 * varying dt. The AB/EXT half is dormant for Stokes -- nothing to extrapolate
 * (pure BDF); convection uses it in Sprint 2.
 */
class StokesTimeIntegrator
{
public:
   /**
    * @brief Set up the integrator.
    * @param spaces  Mixed velocity/pressure spaces (borrowed).
    * @param rules   Quadrature source (borrowed).
    * @param bc      Boundary conditions; SetTime is driven by the integrator.
    * @param forcing Momentum forcing f(x, t); SetTime is driven per step.
    * @param opts    Physics, stepping, and Krylov options.
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

   /**
    * @brief Restore a marching state (checkpoint/restart): replaces any
    *        initial condition and SKIPS the startup ramp.
    *
    * The BDF weights are recomputed from the restored times each step, so a
    * history with non-uniform spacing (adaptive runs) continues exactly; the
    * solver caches rebuild automatically from the leading weight.
    *
    * @param states          Velocity true-dof history, NEWEST FIRST (>= 1).
    * @param times           Matching times, strictly decreasing (t = times[0]).
    * @param completed_steps Steps completed before the checkpoint (drives the
    *                        startup-vs-BDF dispatch exactly as live stepping).
    * @param next_dt         Step size to attempt next (adaptive continuation).
    * @param pressure        Optional pressure true dofs (Krylov warm start).
    */
   void SetHistory(const std::vector<mfem::Vector>& states,
                   const std::vector<double>& times, int completed_steps,
                   double next_dt, const mfem::Vector* pressure = nullptr);

   /**
    * @brief Install the adaptive dt ceiling hook (stability, not accuracy).
    *        Inert by default; Sprint 2 wires the convective CFL bound.
    * @param ceiling Callback returning the max admissible dt at a time.
    */
   void SetDtCeiling(AdaptiveController::DtCeilingFn ceiling);

   /// Advance one (accepted) step; in adaptive mode this may retry internally.
   void Step();

   /// March Step() until Done().
   void Run();

   /// @return Current time.
   double Time() const { return t_; }

   /// @return True once t_final is reached.
   bool Done() const;

   /// @return FGMRES iterations of the most recent implicit solve.
   int LastIterations() const { return last_iterations_; }

   /// @return Completed (accepted) steps.
   int StepCount() const { return step_count_; }

   /// @return The current step size (varies in adaptive mode).
   double CurrentDt() const { return dt_; }

   /// @return Velocity true-dof history, newest first (checkpointing).
   const std::deque<mfem::Vector>& History() const { return hist_; }

   /// @return Times of the history entries (checkpointing).
   const std::deque<double>& HistoryTimes() const { return hist_times_; }

   /// @return The adaptive controller (step history, rejection counts);
   ///         null in fixed-step mode.
   const AdaptiveController* Controller() const { return controller_.get(); }

   /// @return Mutable adaptive controller (restart restores its PI memory);
   ///         null in fixed-step mode.
   AdaptiveController* Controller() { return controller_.get(); }

   /// @return The velocity field (updated by Step()).
   mfem::ParGridFunction& Velocity() { return u_; }

   /// @return The pressure field (updated by Step()).
   mfem::ParGridFunction& Pressure() { return p_; }

private:
   /// A solver cached per leading BDF weight (the momentum mass factor).
   struct SolverCache
   {
      std::unique_ptr<StokesSolver> solver; ///< The cached solver.
      double c0 = -1.0;                     ///< Leading weight it was built with.
   };

   /// @return The cached solver rebuilt if @p c0 differs from the cached one.
   StokesSolver& EnsureBdfSolver(SolverCache& cache, double c0);

   /// Assemble the forcing functional F(t) on velocity true dofs.
   void AssembleForcing(double t, mfem::Vector& F);

   /// Build the BDF right-hand side for weights @p c at time @p t_new.
   void AssembleBdfRhs(const std::vector<double>& c, double t_new,
                       mfem::Vector& b);

   void StepStartup();  ///< Trapezoidal starter / early BDF2 steps.
   void StepFixed();    ///< Fixed-step BDF2/BDF3 step.
   void StepAdaptive(); ///< Adaptive attempt loop (BDF2 + BDF3 candidates).

   /// Commit an accepted velocity/pressure into fields and history.
   void Commit(double t_new, const mfem::Vector& u_true,
               mfem::ParGridFunction& u_gf, mfem::ParGridFunction& p_gf);

   MixedSpaces& spaces_;              ///< Mixed spaces (borrowed).
   const RuleBook& rules_;            ///< Quadrature source (borrowed).
   BoundaryConditions& bc_;           ///< Boundary conditions (borrowed).
   mfem::VectorCoefficient& forcing_; ///< Momentum forcing (borrowed).
   TimeIntegratorOptions opts_;       ///< Options.

   std::unique_ptr<StokesSolver> trap_; ///< Trapezoidal starter solver.
   SolverCache bdf2_;                   ///< BDF2 solver cache.
   SolverCache bdf3_;                   ///< BDF3 solver cache.

   std::unique_ptr<AdaptiveController> controller_; ///< Adaptive mode only.

   mfem::ParGridFunction u_;  ///< Velocity field.
   mfem::ParGridFunction p_;  ///< Pressure field.
   mfem::ParGridFunction u2_scratch_; ///< BDF2 candidate (adaptive).
   mfem::ParGridFunction p2_scratch_; ///< BDF2 candidate pressure (adaptive).
   mfem::ParGridFunction u3_scratch_; ///< BDF3 candidate (adaptive).
   mfem::ParGridFunction p3_scratch_; ///< BDF3 candidate pressure (adaptive).

   std::deque<mfem::Vector> hist_; ///< Velocity true-dof history, newest first.
   std::deque<double> hist_times_; ///< Times of the history entries.

   double t_ = 0.0;           ///< Current time.
   double dt_ = 0.0;          ///< Current step size (varies in adaptive mode).
   int step_count_ = 0;       ///< Completed (accepted) steps.
   int last_iterations_ = 0;  ///< Iterations of the most recent solve.
};

} // namespace incns

#endif // INCNS_TIME_TIME_INTEGRATOR_HPP
