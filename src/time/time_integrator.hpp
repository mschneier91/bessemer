/**
 * @file time_integrator.hpp
 * @brief In-repo time integrator for unsteady Stokes: fixed-step
 *        (trapezoidal starter -> BDF2 -> BDF3-in-test-mode) and adaptive
 *        (BDF2 advancing, BDF3 LTE estimator, PI controller).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "operators/convection.hpp"
#include "post/kinetic_head.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/adaptive_controller.hpp"
#include "time/integrator_state.hpp"
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
   /// Solve Navier-Stokes rather than unsteady Stokes (Sprint 2.2): adds the
   /// dealiased convection term N(u) = (u.grad)u, extrapolated to t^{n+1} by
   /// AB/EXT and carried on the RIGHT-hand side. The implicit block solve is
   /// UNCHANGED -- this is IMEX, so convection never enters the matrix.
   bool convection = false;
   /// Treatment of the nonlinear term when @ref convection is on. Rotational:
   /// the lagged-vorticity term (curl w*) x u moves INTO the implicit velocity
   /// block (w* = EXT-extrapolated velocity, updated every step), nothing is
   /// explicit, and Pressure() reports static pressure recovered from the
   /// Bernoulli head the solve produces (P - 1/2|u|^2, interpolated).
   ConvectiveForm convective_form = ConvectiveForm::Convective;
   /// Velocity-block PC with the rotational form (see RotationVelocityPC).
   RotationVelocityPC rotation_pc = RotationVelocityPC::Symmetric;
   /// Rotational form + LOR-AMG velocity PC: include the rotation term in the
   /// LOR operator, re-set-up every step (StokesSolverOptions::rotation_in_lor).
   bool rotation_in_lor = false;
   bool collocated_mass = false; ///< GLL collocated mass option.
   double grad_div = 0.0;   ///< Grad-div scale c_gd; 0 = off.
   /// Grad-div scaling mode (OrderH default; OrderNu = c_gd*nu).
   GradDivScale grad_div_scale = GradDivScale::OrderH;
   /// Velocity-block preconditioner (Jacobi default; LORAMG option).
   VelocityPreconditioner velocity_prec = VelocityPreconditioner::Jacobi;
   /// AMG only: reuse (freeze) the LOR hierarchy across Delta-t changes.
   bool amg_reuse = false;
   /// Pressure Schur block: Mass (default) or CahouetChabard.
   SchurBlockType schur = SchurBlockType::Mass;
   /// CC configuration (sigma/nu auto-filled by the solver each build/refresh).
   CahouetChabardConfig cc;
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
    * @param pressure        Optional true dofs of the solver's pressure
    *                        variable (the Bernoulli head in the rotational
    *                        form) -- the Krylov warm start. Pressure() is
    *                        recomputed from it.
    */
   void SetHistory(const std::vector<mfem::Vector>& states,
                   const std::vector<double>& times, int completed_steps,
                   double next_dt, const mfem::Vector* pressure = nullptr);

   /**
    * @brief The full marching state (history, times, counters, next dt, the
    *        solver pressure, adaptive controller memory).
    * @return A copy; ImportState() on a fresh integrator continues the march
    *         exactly as this one would.
    */
   IntegratorState ExportState() const;

   /**
    * @brief Restore a state from ExportState() (or a checkpoint): SetHistory
    *        plus the adaptive controller's memory. The history vectors must
    *        live on this integrator's spaces (an AMR event transfers them).
    * @param state State to continue from.
    */
   void ImportState(const IntegratorState& state);

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

   /**
    * @brief The discrete momentum residual of the last accepted step, on
    *        velocity true dofs, before Dirichlet elimination:
    *        r = A u + N u - B^T p - b, with A, N, B the UNCONSTRAINED blocks of
    *        the solver that took the step and b its raw right-hand side (BDF
    *        history, forcing, explicit convection). Zero at the free dofs (to
    *        Krylov tolerance); at Dirichlet dofs it is minus the reaction
    *        force -- what post/body_force turns into lift and drag.
    * @param r Output (resized).
    * @pre At least one step since construction or the last ImportState().
    */
   void MomentumResidual(mfem::Vector& r) const;

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

   /// @return The (static) pressure field, updated by Step(). With the
   ///         rotational form this is P - 1/2|u|^2, not the solved Bernoulli
   ///         head P.
   mfem::ParGridFunction& Pressure() { return rotational_ ? p_static_ : p_; }

private:
   /// A solver cached per leading BDF weight (the momentum mass factor).
   struct SolverCache
   {
      std::unique_ptr<StokesSolver> solver; ///< The cached solver.
      double c0 = -1.0;                     ///< Leading weight it was built with.
   };

   /**
    * @brief The cached solver for a leading BDF weight, built on first use
    *        and refreshed when @p c0 differs from the cached one.
    * @param cache Solver cache (BDF2 or BDF3).
    * @param c0    Leading BDF weight beta0/dt.
    * @return The solver.
    */
   StokesSolver& EnsureBdfSolver(SolverCache& cache, double c0);

   /**
    * @brief Assemble the forcing functional F(t) on velocity true dofs.
    * @param t Time to evaluate the forcing at.
    * @param F Output: true-dof functional.
    */
   void AssembleForcing(double t, mfem::Vector& F);

   /**
    * @brief Build the BDF right-hand side for weights @p c at time @p t_new.
    * @param c     BDF weights from the actual node times (c[0] leading).
    * @param t_new Time the step solves for.
    * @param b     Output: momentum right-hand side (true dofs, before
    *              Dirichlet elimination).
    */
   void AssembleBdfRhs(const std::vector<double>& c, double t_new,
                       mfem::Vector& b);

   /**
    * @brief Subtract the AB/EXT-extrapolated convection term from @p b.
    *
    * IMEX: the nonlinear term is explicit, so N(u) is evaluated on the stored
    * velocity history and extrapolated to @p t_new -- it never enters the
    * matrix. Convection sits on the LHS of the momentum equation, so it is
    * SUBTRACTED from the right-hand side (Convection::Mult returns +(u.grad)u).
    *
    * The extrapolation order is matched to the number of available history
    * entries, so the startup ramp degrades gracefully (EXT1 on the first step)
    * exactly as the BDF side does -- a fixed EXT2 on step 0 would read a
    * history entry that does not exist yet.
    *
    * No-op when TimeIntegratorOptions::convection is false.
    * @param t_new Time the step solves for.
    * @param b     Momentum right-hand side (true dofs), modified in place.
    */
   void SubtractConvection(double t_new, mfem::Vector& b);

   /**
    * @brief Rotational form: set the lagged velocity w* to the EXT
    *        extrapolation of the history to @p t_new (same order matching as
    *        SubtractConvection: EXT1 on the first step, EXT2 after). The
    *        caller then calls UpdateRotation() on each solver it uses.
    * @param t_new Time the step solves for.
    */
   void UpdateLaggedVelocity(double t_new);

   /// Rotational form: p_static_ = p_ - I(1/2|u_|^2), mean-normalized when
   /// the pressure null space exists.
   void UpdateStaticPressure();

   void StepStartup();  ///< Trapezoidal starter / early BDF2 steps.
   void StepFixed();    ///< Fixed-step BDF2/BDF3 step.
   void StepAdaptive(); ///< Adaptive attempt loop (BDF2 + BDF3 candidates).

   /**
    * @brief Commit an accepted velocity/pressure into fields and history.
    * @param t_new  Time of the accepted state.
    * @param u_true Accepted velocity, true dofs (pushed onto the history).
    * @param u_gf   Accepted velocity field (copied into Velocity()).
    * @param p_gf   Accepted pressure field (copied into the pressure).
    */
   void Commit(double t_new, const mfem::Vector& u_true,
               mfem::ParGridFunction& u_gf, mfem::ParGridFunction& p_gf);

   MixedSpaces& spaces_;              ///< Mixed spaces (borrowed).
   const RuleBook& rules_;            ///< Quadrature source (borrowed).
   BoundaryConditions& bc_;           ///< Boundary conditions (borrowed).
   mfem::VectorCoefficient& forcing_; ///< Momentum forcing (borrowed).
   TimeIntegratorOptions opts_;       ///< Options.

   std::unique_ptr<StokesSolver> trap_; ///< Trapezoidal starter solver.
   double trap_c0_ = 0.0;               ///< Mass factor trap_ is built for (1/dt).
   SolverCache bdf2_;                   ///< BDF2 solver cache.
   SolverCache bdf3_;                   ///< BDF3 solver cache.

   std::unique_ptr<AdaptiveController> controller_; ///< Adaptive mode only.
   /// Dealiased convection operator; built only when opts_.convection is set.
   std::unique_ptr<Convection> convection_;
   /// Rotational form active (convection on, ConvectiveForm::Rotational).
   bool rotational_ = false;

   mfem::ParGridFunction u_;  ///< Velocity field.
   mfem::ParGridFunction p_;  ///< Pressure field.
   mfem::ParGridFunction u2_scratch_; ///< BDF2 candidate (adaptive).
   mfem::ParGridFunction p2_scratch_; ///< BDF2 candidate pressure (adaptive).
   mfem::ParGridFunction u3_scratch_; ///< BDF3 candidate (adaptive).
   mfem::ParGridFunction p3_scratch_; ///< BDF3 candidate pressure (adaptive).
   /// Rotational form: the lagged velocity w* every solver's rotation term
   /// reads (updated in place before each solve).
   mfem::ParGridFunction w_star_;
   /// Rotational form: static pressure P - 1/2|u|^2 (what Pressure() returns).
   mfem::ParGridFunction p_static_;
   /// Rotational form: I_p(1/2|u|^2) scratch on the pressure space (reused).
   mfem::ParGridFunction ke_;
   /// Rotational form: device interpolant of 1/2|u|^2 at the pressure nodes.
   std::unique_ptr<KineticHeadInterpolator> kinetic_head_;

   std::deque<mfem::Vector> hist_; ///< Velocity true-dof history, newest first.
   std::deque<double> hist_times_; ///< Times of the history entries.

   /// Solver of the last accepted step (MomentumResidual); not owned.
   StokesSolver* last_solver_ = nullptr;
   mfem::Vector last_b_;      ///< Its raw momentum right-hand side.

   double t_ = 0.0;           ///< Current time.
   double dt_ = 0.0;          ///< Current step size (varies in adaptive mode).
   int step_count_ = 0;       ///< Completed (accepted) steps.
   int last_iterations_ = 0;  ///< Iterations of the most recent solve.
};

} // namespace incns
