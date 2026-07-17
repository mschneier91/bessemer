/**
 * @file stokes_solver.hpp
 * @brief One implicit (steady) Stokes saddle-point solve.
 */
#ifndef INCNS_SOLVER_STOKES_SOLVER_HPP
#define INCNS_SOLVER_STOKES_SOLVER_HPP

#include "bc/boundary_conditions.hpp"
#include "operators/block_preconditioner.hpp"
#include "operators/pressure_schur.hpp"
#include "operators/stokes_operator.hpp"
#include "precond/block_stokes_pc.hpp"
#include "precond/cahouet_chabard.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/velocity_preconditioner.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Options for the Stokes solve.
struct StokesSolverOptions
{
   double nu = 1.0;              ///< Kinematic viscosity.
   bool collocated_mass = false; ///< GLL collocated mass option (see 1.4).
   /// Momentum-block mass factor (beta0/dt); 0 = steady.
   double mass_coeff = 0.0;
   /// Grad-div scale c_gd (gamma(x) = c_gd * h_K); 0 = off. Never enters the
   /// Schur block -- it stays nu * M_p^{-1} with gamma on or off.
   double grad_div = 0.0;
   /// Velocity-block preconditioner (default Jacobi; BoomerAMG for stiffer
   /// regimes -- it triggers a full assembly of the momentum block).
   VelocityPreconditioner velocity_prec = VelocityPreconditioner::Jacobi;
   /// AMG only: freeze the LOR hierarchy at nu*K and reuse it across Delta-t
   /// changes instead of rebuilding it each refresh (see StokesOperatorOptions::
   /// lor_frozen). Cheap adaptive stepping; valid in the viscous-dominated
   /// regime AMG is chosen for.
   bool amg_reuse = false;
   /// Pressure Schur block: Mass (Sprint-1 default) or CahouetChabard. With
   /// CC the system is assembled in the canonical SYMMETRIC convention
   /// [A B^T; B 0] on the internal pressure p~ = -p_physical (one sign flip
   /// at output), the block PC shape comes from cc.block_shape (UpperTri
   /// default), and the velocity block PC from cc.a_pc (velocity_prec above
   /// applies to the Mass path only).
   SchurBlockType schur = SchurBlockType::Mass;
   /// CC configuration (consulted when schur == CahouetChabard). sigma and nu
   /// are OVERWRITTEN from mass_coeff and nu above -- single source of truth;
   /// likewise the outer-solver knobs (rtol/atol/max_iter/kdim) come from this
   /// struct, and cc.outer_* exist only for spec parity.
   CahouetChabardConfig cc;
   double rtol = 1e-10;          ///< FGMRES relative tolerance.
   double atol = 0.0;            ///< FGMRES absolute tolerance.
   int max_iter = 2000;          ///< FGMRES iteration cap.
   int kdim = 200;               ///< FGMRES restart (Krylov subspace) size.
   int print_level = -1;         ///< mfem::IterativeSolver print level.
};

/**
 * @brief Solves the steady Stokes saddle-point system
 *        @c [A  -B^T; B  0] [u; p] = [f; 0], A = mass_coeff*M + nu*K,
 *        with velocity Dirichlet elimination.
 *
 * Outer solver: FGMRES (right-preconditioned) on the full block system with the
 * block-diagonal preconditioner diag(Jacobi(nu*K), nu*M_p^{-1}) -- the Sprint-1
 * pressure-mass Schur block. When the BC set implies the constant pressure
 * null space (no outflow -- fully periodic or fully enclosed), the Schur block
 * is wrapped in an mfem::OrthoSolver (P * S^{-1} * P with P the zero-sum
 * projection) and the output pressure is normalized to zero mass-weighted
 * mean by the post-processor. The pressure is never pinned. (The constraint
 * RHS needs no projection: it is compatible to roundoff by construction, and
 * truly incompatible data should fail loudly, not be masked.)
 *
 * The unsteady stepper (sub-sprint 1.8) reuses this same implicit solve with
 * the BDF mass term added to the velocity block.
 */
class StokesSolver
{
public:
   /**
    * @brief Assemble operators and set up the solver.
    * @param spaces Mixed velocity/pressure spaces (borrowed).
    * @param rules  Quadrature source (borrowed, outlives the operators).
    * @param bc     Boundary conditions; supplies the essential dofs, the
    *               Dirichlet data, and the null-space detection (borrowed).
    * @param opts   Physics and Krylov options.
    */
   StokesSolver(MixedSpaces& spaces, const RuleBook& rules,
                BoundaryConditions& bc,
                const StokesSolverOptions& opts = StokesSolverOptions());

   /**
    * @brief Solve the steady Stokes system for a given forcing.
    *
    * Projects the Dirichlet data into @p u, assembles the forcing functional
    * (fast-assembly path), eliminates the Dirichlet contribution from both
    * right-hand-side blocks, runs FGMRES, and distributes the solution. With a
    * pressure null space, @p p is normalized to zero mass-weighted mean.
    *
    * @param forcing Momentum forcing @c f (vector coefficient).
    * @param u       In: any content; out: the velocity solution (with the
    *                Dirichlet boundary values imposed).
    * @param p       Out: the pressure solution.
    */
   void Solve(mfem::VectorCoefficient& forcing, mfem::ParGridFunction& u,
              mfem::ParGridFunction& p);

   /**
    * @brief Lower-level solve with a caller-assembled momentum right-hand side
    *        (true dofs, BEFORE Dirichlet elimination) -- the time stepper's
    *        entry point: it adds BDF history / explicit starter terms itself.
    *
    * Projects the Dirichlet data at the BC's CURRENT time onto @p u (set
    * BoundaryConditions::SetTime first -- re-elimination every step), uses the
    * incoming @p u / @p p as the Krylov warm start, eliminates the Dirichlet
    * contribution from both RHS blocks, solves, distributes, and mean-
    * normalizes @p p when the null space exists.
    *
    * @param b_mom Momentum RHS on velocity true dofs (not yet eliminated).
    * @param u     In: warm start; out: velocity solution (Dirichlet imposed).
    * @param p     In: warm start; out: pressure solution.
    */
   void SolveTrue(const mfem::Vector& b_mom, mfem::ParGridFunction& u,
                  mfem::ParGridFunction& p);

   /**
    * @brief Refresh the solver for a new BDF mass factor @p c0 (= beta0/dt),
    *        reusing all Delta-t-independent state.
    *
    * Reassembles only the momentum block A = c0*M + nu*K (+ grad-div) and
    * refreshes the velocity preconditioner; B, B^T, the pressure Schur block,
    * and the FGMRES object are untouched. Far cheaper than reconstructing the
    * solver -- the time integrator calls this on every Delta-t change instead
    * of rebuilding. With amg_reuse the frozen AMG hierarchy is not rebuilt.
    *
    * @param c0 New leading BDF weight beta0/dt (>= 0).
    */
   void Refresh(double c0);

   /// @return FGMRES iterations of the last Solve().
   int Iterations() const { return iterations_; }

   /// @return Whether the last Solve() converged to tolerance.
   bool Converged() const { return converged_; }

   /// @return The assembled operator blocks (constrained with the BC's dofs).
   StokesOperator& Blocks() { return op_; }

private:
   /// (Re)build the velocity preconditioner + block preconditioner and hand it
   /// to FGMRES. Called at construction and on a non-frozen Refresh.
   void BuildVelocityPreconditioner();

   MixedSpaces& spaces_;        ///< Mixed spaces (borrowed).
   const RuleBook& rules_;      ///< Quadrature source (borrowed).
   BoundaryConditions& bc_;     ///< Boundary conditions (borrowed).
   StokesSolverOptions opts_;   ///< Options.
   bool nullspace_;             ///< Constant pressure mode present?
   bool cc_mode_;               ///< CahouetChabard Schur path active?

   StokesOperator op_;          ///< Constrained blocks.
   std::unique_ptr<mfem::TransposeOperator> BT_; ///< -B^T wrapper (block 0,1).
   mfem::BlockOperator block_op_;                ///< The saddle-point operator.
   PressureMassSchur schur_;                     ///< Pressure Schur block.
   /// Wraps schur_ as P*S^{-1}*P (P = zero-sum projection) when the constant
   /// pressure null space exists -- orthogonalization, never pinning.
   mfem::OrthoSolver ortho_schur_;
   /// Active pressure block (schur_ or ortho_schur_); set once at construction.
   mfem::Solver* pressure_block_ = nullptr;
   /// Velocity block preconditioner (Jacobi smoother or BoomerAMG). Declared
   /// after op_ so it is destroyed before the momentum matrix it may reference.
   std::unique_ptr<mfem::Solver> vel_prec_;
   /// CC Schur PC (CahouetChabard mode only; null on the Mass path). Declared
   /// before prec_ (the block wrapper borrows it).
   std::unique_ptr<CahouetChabardSchurPC> cc_pc_;
   /// Block preconditioner: StokesBlockPreconditioner (Mass) or BlockStokesPC
   /// (CC shapes) behind the common Solver interface.
   std::unique_ptr<mfem::Solver> prec_;
   mfem::FGMRESSolver fgmres_;                   ///< Outer Krylov solver.

   int iterations_ = 0;      ///< Iterations of the last solve.
   bool converged_ = false;  ///< Convergence flag of the last solve.
};

} // namespace incns

#endif // INCNS_SOLVER_STOKES_SOLVER_HPP
