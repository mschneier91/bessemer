/**
 * @file solution_projection.hpp
 * @brief Initial guesses for the step's block solve from earlier solutions,
 *        and what the outer tolerance is relative to.
 */
#pragma once

#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/// What the outer FGMRES tolerance is relative to (deck
/// `solver.tolerance_reference`).
enum class ToleranceReference
{
   /// ||r|| <= rtol ||r_0|| (MFEM's rel_tol): a better initial guess only
   /// makes the answer more accurate, it saves no iterations.
   InitialResidual,
   /// ||r|| <= rtol ||b||: a better initial guess stops the solve sooner.
   Rhs
};

/// The outer solve's initial guess (deck `solver.initial_guess`).
enum class InitialGuess
{
   Previous,    ///< The previous step's solution (the warm start).
   Extrapolate, ///< Its linear extrapolation in time from the last two.
   Project      ///< Least squares over the span of the last K solutions.
};

/**
 * @brief Least-squares initial guesses from the span of earlier solutions
 *        (the projection of Fischer, CMAME 163, 1998, for a nonsymmetric or
 *        indefinite operator: the residual is minimized in the 2-norm).
 *
 * Keeps solutions X = [x_1 .. x_m] and their images W = A X, transformed so
 * that W is orthonormal. For a right-hand side b the guess is X W^T b, the
 * x in span(X) with the smallest ||b - A x||. Each added solution costs one
 * operator application and 2m inner products; the guess costs m. When the
 * basis is full it restarts from the newest solution. The images are those
 * of the operator they were computed with: after the operator changes, the
 * guess is still a valid initial guess, only a less good one, until
 * Rebuild() recomputes them. Memory: 2K vectors of the system's size.
 */
class SolutionProjection
{
public:
   /**
    * @param k    The most solutions kept (>= 1).
    * @param comm Communicator of the distributed vectors.
    */
   SolutionProjection(int k, MPI_Comm comm);

   /// @return The number of solutions in the basis.
   int Size() const { return static_cast<int>(x_.size()); }

   /// @return The largest number of vectors held (2K).
   int Capacity() const { return 2 * k_; }

   /// Forget all solutions.
   void Reset();

   /**
    * @brief The guess for @p b.
    * @param b  Right-hand side.
    * @param x0 Output: X W^T b; untouched when the basis is empty.
    * @return Whether a guess was made.
    */
   bool Guess(const mfem::Vector& b, mfem::Vector& x0) const;

   /**
    * @brief Add a solution of the system with operator @p A.
    * @param A The operator (applied once).
    * @param x The solution.
    */
   void Add(const mfem::Operator& A, const mfem::Vector& x);

   /**
    * @brief Recompute the images with a changed operator @p A (m
    *        applications) and orthonormalize them again.
    * @param A The new operator.
    */
   void Rebuild(const mfem::Operator& A);

private:
   /// Orthonormalize w (and carry x along) against the basis, twice
   /// (classical Gram-Schmidt reorthogonalized); append unless w vanishes.
   /// Appended vectors are swapped into the basis (x and w come back empty).
   void Append(mfem::Vector& x, mfem::Vector& w);

   /// A device work vector of size @p n into @p v: a spare one if there is
   /// one of that size, else a new one.
   void Take(int n, mfem::Vector& v);
   /// Keep @p v (unless empty) as a spare.
   void Give(mfem::Vector& v);

   int k_;                       ///< Basis capacity.
   MPI_Comm comm_;               ///< Communicator.
   std::vector<mfem::Vector> x_; ///< Solutions (transformed with w_).
   std::vector<mfem::Vector> w_; ///< Their images, orthonormal.
   /// Vectors of a reset basis and unused work vectors, reused instead of
   /// freed and allocated again (each free synchronizes a GPU). mfem::Vector
   /// has no move: x_ and w_ are reserved for k_ so they never reallocate
   /// (which would copy every vector), and vectors enter them by Swap.
   std::vector<std::unique_ptr<mfem::Vector>> spare_;
};

} // namespace incns
