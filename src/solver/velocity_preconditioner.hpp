/**
 * @file velocity_preconditioner.hpp
 * @brief Choice of preconditioner for the velocity (momentum) block.
 */
#ifndef INCNS_SOLVER_VELOCITY_PRECONDITIONER_HPP
#define INCNS_SOLVER_VELOCITY_PRECONDITIONER_HPP

namespace incns
{

/**
 * @brief Which preconditioner to use on the velocity block @c A of the
 *        block-diagonal Stokes preconditioner.
 *
 * The block system, its FGMRES driver, and the pressure Schur block are
 * identical for both choices; only the velocity block differs.
 */
enum class VelocityPreconditioner
{
   /// Matrix-free Jacobi (mfem::OperatorJacobiSmoother) from the assembled
   /// momentum diagonal. The default -- the block stays mass-dominated at DNS
   /// time steps, and it needs no assembled matrix.
   Jacobi,
   /// hypre BoomerAMG on the assembled momentum matrix (for stiffer regimes).
   /// Requires a full assembly of the momentum block (a different assembly
   /// route than the matrix-free default) and a heavier per-refresh setup.
   BoomerAMG
};

} // namespace incns

#endif // INCNS_SOLVER_VELOCITY_PRECONDITIONER_HPP
