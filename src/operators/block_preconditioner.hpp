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
 *        @c diag(Jacobi(A), S_hat^{-1}).
 *
 * The velocity block is an mfem::OperatorJacobiSmoother built from the
 * assembled diagonal of @c A (deliberate default -- see CLAUDE.md; the block
 * stays mass-dominated at DNS time steps). The pressure block is any Solver:
 * Sprint 1 passes PressureMassSchur, wrapped in an mfem::OrthoSolver by the
 * caller when the pressure constant null space exists (the null space is
 * orthogonalized, never pinned -- see StokesSolver).
 */
class StokesBlockPreconditioner : public mfem::Solver
{
public:
   /**
    * @brief Assemble the block preconditioner.
    * @param offsets       True-dof block offsets [0, n_u, n_u + n_p].
    * @param velocity_diag Assembled diagonal of the velocity block @c A.
    * @param ess_tdofs     Essential velocity true dofs (identity rows in @c A).
    * @param pressure      Pressure block solver (borrowed).
    */
   StokesBlockPreconditioner(const mfem::Array<int>& offsets,
                             const mfem::Vector& velocity_diag,
                             const mfem::Array<int>& ess_tdofs,
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
   mfem::Array<int> offsets_;                 ///< Block offsets (copied).
   mfem::OperatorJacobiSmoother vel_jacobi_;  ///< Velocity Jacobi block.
   mfem::Solver& pressure_;                   ///< Pressure block (borrowed).
};

} // namespace incns

#endif // INCNS_OPERATORS_BLOCK_PRECONDITIONER_HPP
