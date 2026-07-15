/**
 * @file block_preconditioner.hpp
 * @brief Block-diagonal preconditioner for the Stokes saddle-point system.
 */
#ifndef INCNS_OPERATORS_BLOCK_PRECONDITIONER_HPP
#define INCNS_OPERATORS_BLOCK_PRECONDITIONER_HPP

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Block-diagonal preconditioner for @c [A B^T; B 0]:
 *        @c diag(velocity^{-1}, S_hat^{-1}).
 *
 * Both blocks are borrowed Solvers, so the choice of velocity preconditioner is
 * the caller's (see VelocityPreconditioner): the default is a matrix-free
 * mfem::OperatorJacobiSmoother on the momentum diagonal (the block stays
 * mass-dominated at DNS time steps), with mfem::HypreBoomerAMG as an option for
 * stiffer regimes. The pressure block is PressureMassSchur, wrapped in an
 * mfem::OrthoSolver by the caller when the pressure constant null space exists
 * (orthogonalized, never pinned -- see StokesSolver).
 */
class StokesBlockPreconditioner : public mfem::Solver
{
public:
   /**
    * @brief Assemble the block preconditioner from two block solvers.
    * @param offsets  True-dof block offsets [0, n_u, n_u + n_p].
    * @param velocity Velocity block solver (borrowed; Jacobi or AMG).
    * @param pressure Pressure block solver (borrowed).
    */
   StokesBlockPreconditioner(const mfem::Array<int>& offsets,
                             mfem::Solver& velocity,
                             mfem::Solver& pressure);

   /**
    * @brief Apply the block-diagonal preconditioner.
    * @param x Input block true-dof vector (velocity; pressure).
    * @param y Output block true-dof vector.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   /// Required by mfem::Solver; the blocks are fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

private:
   mfem::Array<int> offsets_;   ///< Block offsets (copied).
   mfem::Solver& velocity_;     ///< Velocity block (borrowed).
   mfem::Solver& pressure_;     ///< Pressure block (borrowed).
};

} // namespace incns

#endif // INCNS_OPERATORS_BLOCK_PRECONDITIONER_HPP
