/**
 * @file velocity_cg.hpp
 * @brief Preconditioned CG for the Schur method's velocity solves that
 *        works in its right-hand side: one velocity vector fewer than
 *        mfem::CGSolver, and nothing reallocated when the operator changes.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Preconditioned CG on a symmetric positive definite operator, from a
 *        zero start, that overwrites its right-hand side with the residual.
 *
 * The algorithm and the stopping test are mfem::CGSolver's with
 * iterative_mode false (the same operations in the same order): stop when
 * (r, P r) <= max(rel_tol^2 (b, P b), abs_tol^2). Where it differs:
 *  - the right-hand side is the residual vector (mfem::CGSolver copies it
 *    into one of its own): the work vectors are the direction d and z (P r
 *    and A d), two velocity vectors instead of three;
 *  - SetOperator and SetPreconditioner only re-point: no allocation, and the
 *    preconditioner's own SetOperator is never called (mfem::CGSolver
 *    forwards it, which would rebuild a LOR-AMG or a Jacobi diagonal).
 *
 * Between solves the two work vectors hold nothing: WorkD() and WorkZ() lend
 * them out (the Cahouet-Chabard preconditioner's velocity scratch under the
 * Schur method, StokesSolver).
 */
class VelocityCG
{
public:
   /**
    * @brief Allocate the two work vectors.
    * @param comm Communicator of the dot products.
    * @param n    Local size (velocity true dofs).
    */
   VelocityCG(MPI_Comm comm, int n);

   /// @param A The operator from now on (borrowed; symmetric positive
   ///          definite, Height() = n).
   void SetOperator(const mfem::Operator& A) { A_ = &A; }

   /// @param pc The preconditioner from now on (borrowed; its iterative_mode
   ///           is set to false).
   void SetPreconditioner(mfem::Solver& pc)
   {
      pc_ = &pc;
      pc_->iterative_mode = false;
   }

   /// @param rel Relative tolerance (of the preconditioned residual norm).
   void SetRelTol(double rel) { rel_tol_ = rel; }

   /// @param abs Absolute tolerance (of the preconditioned residual norm).
   void SetAbsTol(double abs) { abs_tol_ = abs; }

   /// @param n Most iterations per solve.
   void SetMaxIter(int n) { max_iter_ = n; }

   /**
    * @brief Solve A x = b from x = 0.
    * @param b In: the right-hand side; out: the final residual b - A x.
    * @param x Out: the solution (its input is ignored).
    */
   void Solve(mfem::Vector& b, mfem::Vector& x);

   /// @return Iterations of the last solve.
   int GetNumIterations() const { return iterations_; }

   /// @return Whether the last solve met its tolerance.
   bool GetConverged() const { return converged_; }

   /// @return The direction work vector: free between solves.
   mfem::Vector& WorkD() { return d_; }

   /// @return The second work vector (P r, A d): free between solves.
   mfem::Vector& WorkZ() { return z_; }

private:
   /// Dot product over the communicator.
   double Dot(const mfem::Vector& a, const mfem::Vector& b) const
   {
      return mfem::InnerProduct(comm_, a, b);
   }

   MPI_Comm comm_;                    ///< Communicator.
   const mfem::Operator* A_ = nullptr; ///< Operator (borrowed).
   mfem::Solver* pc_ = nullptr;       ///< Preconditioner (borrowed).
   double rel_tol_ = 1e-10;           ///< See SetRelTol.
   double abs_tol_ = 0.0;             ///< See SetAbsTol.
   int max_iter_ = 500;               ///< See SetMaxIter.
   int iterations_ = 0;               ///< See GetNumIterations.
   bool converged_ = false;           ///< See GetConverged.
   mfem::Vector d_;                   ///< Direction (device).
   mfem::Vector z_;                   ///< P r, and A d (device).
};

} // namespace incns
