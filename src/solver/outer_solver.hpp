/**
 * @file outer_solver.hpp
 * @brief Which outer iteration solves the saddle-point system, and the
 *        options of the Schur complement (Uzawa) one (solver/schur_solver.hpp).
 *
 * Kept free of solver headers so that the Parameters layer can hold them.
 */
#pragma once

namespace incns
{

/**
 * @brief The outer iteration of a StokesSolver (deck `solver.outer`).
 *
 * Both solve the same eliminated system [A B^T; B 0] with the same velocity
 * preconditioner and the same Cahouet-Chabard Schur preconditioner, to the
 * same coupled residual ||b - K x||; they differ in where the Krylov
 * recurrence runs.
 */
enum class OuterMethod
{
   /// FGMRES on the whole coupled system, block-triangular preconditioner
   /// (the default). Its basis holds two system-sized vectors per iteration.
   Coupled,
   /// Flexible CG on the pressure Schur complement S = B A^-1 B^T, each
   /// operator application a velocity solve (SchurSolver). A few pressure-
   /// and velocity-sized vectors in place of the coupled basis. Symmetric
   /// momentum block only: not with the rotational form.
   Schur
};

/// How tightly the velocity solve inside each Schur operator application is
/// converged (deck `solver.uzawa.inner_tolerance`).
enum class InnerTolerance
{
   Tight,  ///< eps_tight relative, every application: S is effectively exact.
   Loose,  ///< eps_loose relative, every application.
   /// eps_k = min(eps_max, max(eps_tight, relax_c * tol / ||r_{k-1}||)):
   /// tight early, loose late (van den Eshof and Sleijpen 2004; Simoncini and
   /// Szyld 2003). The default.
   Relaxed
};

/**
 * @brief Options of the Schur complement solver (deck `solver.uzawa`).
 *
 * The outer tolerance, iteration cap and initial guess are the StokesSolver's
 * own (solver.rtol, solver.atol, solver.max_iter, solver.tolerance_reference,
 * solver.initial_guess): both outer methods stop on the same coupled residual.
 */
struct SchurSolveOptions
{
   InnerTolerance inner = InnerTolerance::Relaxed; ///< Inner tolerance mode.
   double eps_tight = 1e-10; ///< Tight inner relative tolerance; Relaxed's floor.
   double eps_loose = 1e-3;  ///< Loose inner relative tolerance.
   double eps_max = 1e-2;    ///< Relaxed: the loosest inner tolerance.
   double relax_c = 1.0;     ///< Relaxed: the schedule's constant c.
   /// Iteration cap of each velocity solve (the System 1 and System 3 solves
   /// and every Schur operator application).
   int inner_max_iter = 500;
   /// Outer cycles: after each, the velocity is re-solved from the pressure
   /// and the TRUE pressure residual checked; a residual that drifted above
   /// the tolerance starts another cycle from the current iterate.
   int max_cycles = 3;
   /// Record every outer iteration (recurrence residual, inner tolerance and
   /// iterations) and the coupled residual of the iterate, which costs one
   /// momentum-block application per iteration (no solve).
   bool history = false;
};

} // namespace incns
