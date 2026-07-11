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
 * @brief Remove the component along the constant vector: @c v <- v - mean(v).
 *
 * Plain (l2) mean over the GLOBAL vector -- the discrete form of removing the
 * constant pressure null-space mode inside the Krylov solve. (The physical
 * mean-zero output normalization is the separate mass-weighted post-processor
 * in post/pressure_mean.) Collective over @p comm.
 *
 * @param v    True-dof vector, modified in place.
 * @param comm Communicator of the underlying space.
 */
void SubtractGlobalMean(mfem::Vector& v, MPI_Comm comm);

/**
 * @brief Block-diagonal preconditioner for @c [A B^T; B 0]:
 *        @c diag(Jacobi(A), S_hat^{-1}).
 *
 * The velocity block is an mfem::OperatorJacobiSmoother built from the
 * assembled diagonal of @c A (deliberate default -- see CLAUDE.md; the block
 * stays mass-dominated at DNS time steps). The pressure block is any Solver,
 * Sprint 1: PressureMassSchur. When the pressure constant null space exists
 * (detected from the BC set), each application removes the constant mode from
 * the pressure output so the Krylov iterates never accumulate a null-space
 * component -- the null space is orthogonalized, never pinned.
 */
class StokesBlockPreconditioner : public mfem::Solver
{
public:
   /**
    * @brief Assemble the block preconditioner.
    * @param offsets       True-dof block offsets [0, n_u, n_u + n_p].
    * @param velocity_diag Assembled diagonal of the velocity block @c A.
    * @param ess_tdofs     Essential velocity true dofs (identity rows in @c A).
    * @param schur         Pressure Schur approximation (borrowed).
    * @param orthogonalize Remove the constant pressure mode on each apply
    *                      (enable exactly when the null space exists).
    * @param comm          Communicator for the orthogonalization reduction.
    */
   StokesBlockPreconditioner(const mfem::Array<int>& offsets,
                             const mfem::Vector& velocity_diag,
                             const mfem::Array<int>& ess_tdofs,
                             mfem::Solver& schur,
                             bool orthogonalize,
                             MPI_Comm comm);

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
   mfem::Solver& schur_;                      ///< Pressure Schur block (borrowed).
   bool orthogonalize_;                       ///< Null-space removal toggle.
   MPI_Comm comm_;                            ///< For the mean reduction.
};

} // namespace incns

#endif // INCNS_OPERATORS_BLOCK_PRECONDITIONER_HPP
