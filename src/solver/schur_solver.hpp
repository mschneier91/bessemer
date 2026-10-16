/**
 * @file schur_solver.hpp
 * @brief The Schur complement (Uzawa) outer iteration: Krylov on the
 *        pressure equation S p = B A^-1 f - g, S = B A^-1 B^T, in place of
 *        FGMRES on the coupled system (deck `solver.outer: schur`).
 */
#pragma once

#include "precond/nullspace.hpp"
#include "solver/outer_solver.hpp"
#include "solver/persistent_fgmres.hpp"
#include "solver/schur_complement.hpp"
#include "solver/solution_projection.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/// One outer iteration of a SchurSolver solve (SchurSolveOptions::history).
struct SchurIterationRecord
{
   int cycle = 0;              ///< Outer cycle (0 first; see max_cycles).
   int iteration = 0;          ///< Outer iteration, counted over all cycles.
   double residual = 0.0;      ///< ||r||: the recurrence's pressure residual.
   /// The coupled residual ||b - K x|| of the iterate (u, p), u carried
   /// along with p; -1 where not computed (FGMRES, which does not form the
   /// iterate until it stops).
   double coupled_residual = -1.0;
   double inner_tolerance = 0.0; ///< Relative tolerance of its velocity solve.
   int inner_iterations = 0;     ///< Iterations of its velocity solve.
};

/// What the last SchurSolver::Mult did.
struct SchurSolveStats
{
   int iterations = 0;           ///< Outer iterations, all cycles.
   int cycles = 0;               ///< Outer cycles run (Krylov restarts + 1).
   /// Velocity solves outside the Schur operator (System 1, System 3 and
   /// their corrections), their iterations, and their wall time [s].
   int velocity_solves = 0;
   int velocity_iterations = 0;  ///< See velocity_solves.
   double velocity_time = 0.0;   ///< See velocity_solves.
   int schur_applications = 0;   ///< Schur operator applications.
   int schur_inner_iterations = 0; ///< Their velocity-solve iterations.
   int schur_cap_hits = 0;       ///< Of them, those that hit the cap.
   double schur_inner_time = 0.0; ///< Their wall time [s].
   double tolerance = 0.0;       ///< The coupled-residual target tau.
   double momentum_residual = 0.0; ///< ||f - A u - B^T p|| at exit.
   double pressure_residual = 0.0; ///< ||g - B u|| at exit.
   double final_residual = 0.0;  ///< The coupled residual at exit.
   bool breakdown = false;       ///< CG met (d, S d) <= 0.
   /// Pressure projection: whether it made a guess, and the pressure
   /// residual before it (after System 1) and after it.
   bool projected = false;
   double start_residual = 0.0;  ///< See projected.
   double guess_residual = 0.0;  ///< See projected.
};

/**
 * @brief Solves the eliminated saddle-point system [A B^T; B 0] [u; p~] =
 *        [f; g] (StokesSolver's Cahouet-Chabard path, p~ = -p) by a Krylov
 *        iteration on the pressure Schur complement.
 *
 * Per solve, with the incoming (u, p) as the warm start:
 *  - System 1: u = A^-1 (f - B^T p) (the spec's System 1 with the warm-start
 *    pressure folded in: no separate S p application);
 *  - the pressure residual r = B u - g, the preconditioner z = P_S^-1 r;
 *  - per iteration one Schur application q = S d (a velocity solve to the
 *    inner tolerance), p += gamma d, u -= gamma A^-1 B^T d (so that u stays
 *    A^-1 (f - B^T p): one velocity vector, no extra solve), r -= gamma q;
 *  - System 3: u = A^-1 (f - B^T p), warm-started from the carried u (cheap:
 *    only the inner solves' errors remain);
 *  - the TRUE pressure residual B u - g from that u: above the tolerance, a
 *    new cycle starts from (u, p) (SchurSolveOptions::max_cycles).
 *
 * The stopping rule is the coupled method's: ||b - K x|| <= tau, tau =
 * max(atol, rtol ||b - K x_0||) as FGMRES's. Split as ||g - B u|| <= tau/sqrt2
 * and ||f - A u - B^T p|| <= tau/sqrt2, both checked on true residuals, so
 * Converged() means the coupled residual is below tau.
 *
 * Velocity solves use CG (symmetric A) or GMRES (with the rotation term) on
 * the eliminated momentum block, preconditioned by the coupled path's own
 * velocity preconditioner (Jacobi, point-block Jacobi, Chebyshev or a LOR-AMG
 * V-cycle: fixed linear operators). Outside the Schur operator each solve is
 * a correction to an explicit l2 residual target, not a relative tolerance.
 *
 * Memory: pressure vectors r, z, d, q (+ 2 kdim for FGMRES), velocity vectors
 * two in the Schur operator plus the velocity Krylov solver's own (3 for CG).
 * No system-sized vector.
 */
class SchurSolver
{
public:
   /**
    * @brief Wire the solver (operators and preconditioners borrowed).
    * @param comm     Communicator.
    * @param A        Eliminated momentum block (constrained: identity rows on
    *                 the essential velocity dofs); with a rotation term the
    *                 full nonsymmetric one.
    * @param B        Eliminated divergence; B^T through MultTranspose.
    * @param a_pc     Preconditioner of the velocity solves (a fixed linear
    *                 operator: not an iterate-to-tolerance solver).
    * @param symmetric_a A is symmetric (CG velocity solves; else GMRES).
    * @param schur_pc Positive Schur approximate inverse P_S^-1 (CC).
    * @param proj     Constant-mode projector when the pressure null space
    *                 exists, else null.
    * @param opts     Method and inner-tolerance options.
    */
   SchurSolver(MPI_Comm comm, const mfem::Operator& A, const mfem::Operator& B,
               mfem::Solver& a_pc, bool symmetric_a, mfem::Solver& schur_pc,
               const ConstantPressureProjector* proj,
               const SchurSolveOptions& opts);

   /**
    * @brief Use a new momentum block and new preconditioners (after a
    *        Delta-t or viscosity refresh rebuilt them).
    * @param A        New eliminated momentum block (borrowed).
    * @param a_pc     New velocity preconditioner (borrowed).
    * @param schur_pc New Schur approximate inverse (borrowed).
    */
   void Rewire(const mfem::Operator& A, mfem::Solver& a_pc,
               mfem::Solver& schur_pc);

   /**
    * @brief The coupled tolerance of the next solves, as FGMRES's:
    *        tau = max(atol, rtol ||b - K x_0||).
    * @param rtol Relative to the initial coupled residual (0 = none: then
    *             no initial residual is computed).
    * @param atol Absolute.
    */
   void SetTolerance(double rtol, double atol)
   {
      rtol_ = rtol;
      atol_ = atol;
   }

   /// @param n Cap on the outer iterations of a solve, all cycles.
   void SetMaxIter(int n) { max_iter_ = n; }

   /// @param level >= 1 prints every outer iteration on rank 0; -1 quiet.
   void SetPrintLevel(int level) { print_level_ = level; }

   /**
    * @brief Start each solve from a projected pressure (the Schur method's
    *        `solver.initial_guess: project`): the least-squares projection
    *        of Fischer (SolutionProjection) on S p = b_S, over the pressure
    *        increments of the last @p k solves.
    *
    * After System 1 at the warm-start pressure p_n, with residual r_n, the
    * guess is p_n + X W^T r_n and its residual r_n - W W^T r_n: no operator
    * application, and the velocity offset it leaves is removed by System 3.
    * Each solve adds its increment p - p_n with image S (p - p_n) = r_n - r
    * (from true residuals: no velocity solve). Memory: 2k + 2 pressure
    * vectors. The images belong to one S: ResetGuess() whenever the
    * momentum block changes (dt or nu).
    * @param k The most increments kept (>= 1).
    */
   void EnablePressureProjection(int k);

   /// Drop the projection's basis (S changed).
   void ResetGuess()
   {
      if (projection_) { projection_->Reset(); }
   }

   /// @return The pressure projection (null unless enabled), for a
   ///         checkpoint.
   SolutionProjection* Projection() const { return projection_.get(); }

   /**
    * @brief Solve; @p x is the warm start (with the Dirichlet values in its
    *        velocity block, as FGMRES's).
    * @param b Eliminated right-hand side [f; g] (true dofs).
    * @param x In: warm start; out: solution [u; p~].
    */
   void Mult(const mfem::Vector& b, mfem::Vector& x);

   /// @return Whether the last solve met the tolerance (true residuals).
   bool Converged() const { return converged_; }

   /// @return What the last solve did.
   const SchurSolveStats& Stats() const { return stats_; }

   /// @return The last solve's per-iteration record (history option only).
   const std::vector<SchurIterationRecord>& History() const { return history_; }

private:
   /// (Re)build the velocity Krylov solver for A_ and a_pc_.
   void BuildVelocitySolver();
   /**
    * @brief u <- A^-1 (f - B^T p) to ||f - B^T p - A u|| <= target, by
    *        corrections from the warm start u (at most a few).
    * @param f      Momentum right-hand side.
    * @param p      Pressure.
    * @param u      In: warm start; out: the velocity.
    * @param target l2 residual target.
    * @param res    Out: the final residual norm.
    * @return Whether the target was met.
    */
   bool SolveVelocity(const mfem::Vector& f, const mfem::Vector& p,
                      mfem::Vector& u, double target, double& res);
   /// ||f - A u - B^T p|| (into the operator's work vectors).
   double MomentumResidual(const mfem::Vector& f, const mfem::Vector& p,
                           const mfem::Vector& u) const;
   /// r = B u - g, projected; @return ||r||.
   double PressureResidual(const mfem::Vector& g, const mfem::Vector& u,
                           mfem::Vector& r) const;
   /// The Relaxed / Tight / Loose inner tolerance after residual @p rnorm.
   double InnerRelTol(double rnorm, double tau_p) const;
   /// One CG / flexible CG cycle from (u, p) with pressure residual r_.
   void CgCycle(mfem::Vector& u, mfem::Vector& p, double rnorm,
                double tau_p, const mfem::Vector& f, int cycle);
   /// One pressure-space FGMRES cycle: p += S^-1 r_ (u is not carried).
   void FgmresCycle(mfem::Vector& p, double tau_p, int cycle);
   /// Dot product over the communicator.
   double Dot(const mfem::Vector& a, const mfem::Vector& b) const
   {
      return mfem::InnerProduct(comm_, a, b);
   }

   MPI_Comm comm_;                       ///< Communicator.
   const mfem::Operator* A_;             ///< Momentum block (borrowed).
   const mfem::Operator& B_;             ///< Divergence (borrowed).
   mfem::Solver* a_pc_;                  ///< Velocity PC (borrowed).
   bool symmetric_a_;                    ///< CG (true) or GMRES velocity solves.
   mfem::Solver* schur_pc_;              ///< P_S^-1 (borrowed).
   const ConstantPressureProjector* proj_; ///< Projector or null (borrowed).
   SchurSolveOptions opts_;              ///< Options.
   double rtol_ = 1e-10;                 ///< See SetTolerance.
   double atol_ = 0.0;                   ///< See SetTolerance.
   int max_iter_ = 2000;                 ///< See SetMaxIter.
   int print_level_ = -1;                ///< See SetPrintLevel.

   /// The velocity Krylov solver (CG or GMRES), shared by the Schur
   /// operator and System 1 / 3. Declared before sop_, which borrows it.
   std::unique_ptr<mfem::IterativeSolver> vel_;
   std::unique_ptr<SchurComplementOperator> sop_; ///< S = B A^-1 B^T.
   /// Pressure-space FGMRES (SchurKrylov::FGMRES only; built on first use).
   std::unique_ptr<PersistentFGMRESSolver> fgmres_;
   /// FGMRES's monitor: records the history and sets the next application's
   /// inner tolerance from the residual FGMRES reports.
   std::unique_ptr<mfem::IterativeSolverMonitor> monitor_;
   int fg_cycle_ = 0;           ///< FGMRES: the cycle being run.
   int fg_base_ = 0;            ///< FGMRES: outer iterations before it.
   double fg_tau_p_ = 0.0;      ///< FGMRES: the pressure-residual target.
   double fg_inner_tol_ = 0.0;  ///< FGMRES: the inner tolerance in use.
   /// Pressure vectors (device): residual, preconditioned residual,
   /// direction, S times direction.
   mfem::Vector r_, z_, d_, q_;
   /// Pressure projection (EnablePressureProjection; null otherwise), and
   /// the solve's starting pressure p_n and its residual r_n (device).
   std::unique_ptr<SolutionProjection> projection_;
   mfem::Vector p_start_, r_start_; ///< See projection_.

   SchurSolveStats stats_;                     ///< See Stats().
   std::vector<SchurIterationRecord> history_; ///< See History().
   bool converged_ = false;                    ///< See Converged().
};

} // namespace incns
