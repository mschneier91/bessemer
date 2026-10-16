/**
 * @file schur_complement.hpp
 * @brief The pressure Schur complement S = B A^-1 B^T as an operator whose
 *        every application is a velocity solve (the Uzawa / Schur CG
 *        method's operator; solver/schur_solver.hpp).
 */
#pragma once

#include "precond/nullspace.hpp"
#include "solver/velocity_cg.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief @f$ q = S d = B A^{-1} B^T d @f$ on pressure true dofs, matrix-free:
 *        one B^T, one velocity solve, one B.
 *
 * A, B and B^T are the ELIMINATED blocks of the symmetric system
 * [A B^T; B 0] on the internal pressure p~ = -p (StokesSolver's
 * Cahouet-Chabard path): B^T d is zero on the essential velocity rows and the
 * constrained A is the identity there, so v = A^-1 B^T d is zero on them too.
 * S is symmetric positive semidefinite when A is symmetric (no rotation
 * term), with the constant pressures as its null space on enclosed and
 * periodic domains.
 *
 * The velocity solve is a VelocityCG started from zero to the inner
 * tolerance of SetInnerTolerance(); a loose one makes S an inexact, slightly
 * varying operator, which plain CG does not tolerate (SchurSolver uses
 * flexible CG). The velocity of the last application is kept
 * (LastVelocity()): the outer iteration updates u with it alongside p, so
 * that u stays A^-1 (f - B^T p). Memory: two velocity vectors, B^T d (which
 * the CG overwrites with its residual) and v.
 */
class SchurComplementOperator : public mfem::Operator
{
public:
   /**
    * @brief Wire the operator (everything borrowed).
    * @param B     Eliminated divergence (pressure rows, velocity columns).
    * @param a_inv Velocity CG on the eliminated momentum block (zero
    *              start).
    * @param proj  Constant-mode projector when the pressure null space
    *              exists (the output is projected), else null.
    */
   SchurComplementOperator(const mfem::Operator& B,
                           VelocityCG& a_inv,
                           const ConstantPressureProjector* proj);

   /**
    * @brief q = B A^-1 B^T d; the intermediate v = A^-1 B^T d is kept.
    * @param d Pressure direction.
    * @param q Its image.
    */
   void Mult(const mfem::Vector& d, mfem::Vector& q) const override;

   /// @param rel Relative tolerance of the next applications' velocity solve.
   void SetInnerTolerance(double rel)
   {
      rel_ = rel;
      a_inv_->SetRelTol(rel);
   }

   /// @return v = A^-1 B^T d of the last application.
   const mfem::Vector& LastVelocity() const { return v_; }

   /// @return The velocity-sized right-hand-side work vector, free between
   ///         applications: SchurSolver borrows it for its velocity solves,
   ///         so that the method keeps no velocity vectors of its own.
   mfem::Vector& RhsWork() const { return w_; }

   /// @return The vector LastVelocity() refers to, writable: free once the
   ///         caller has used LastVelocity() (see RhsWork()).
   mfem::Vector& VelocityWork() const { return v_; }

   /// @return Velocity-solver iterations of the last application.
   int LastIterations() const { return last_iterations_; }

   /// Zero the counters below.
   void ResetCounters() const;

   /// @return Applications since ResetCounters().
   int Applications() const { return applications_; }

   /// @return Velocity-solver iterations summed since ResetCounters().
   int InnerIterations() const { return inner_iterations_; }

   /// @return Applications whose velocity solve hit its iteration cap.
   int CapHits() const { return cap_hits_; }

   /// @return Wall time in the velocity solves since ResetCounters() [s].
   double InnerTime() const { return inner_time_; }

private:
   const mfem::Operator& B_;                 ///< Divergence (borrowed).
   VelocityCG* a_inv_;                  ///< Velocity CG (borrowed).
   const ConstantPressureProjector* proj_;   ///< Projector or null (borrowed).
   double rel_ = 1e-10;                      ///< Inner relative tolerance.
   /// B^T d, then the velocity CG's residual (device).
   mutable mfem::Vector w_;
   mutable mfem::Vector v_;                  ///< A^-1 B^T d (device).
   mutable int last_iterations_ = 0;         ///< See LastIterations().
   mutable int applications_ = 0;            ///< See Applications().
   mutable int inner_iterations_ = 0;        ///< See InnerIterations().
   mutable int cap_hits_ = 0;                ///< See CapHits().
   mutable double inner_time_ = 0.0;         ///< See InnerTime().
};

} // namespace incns
