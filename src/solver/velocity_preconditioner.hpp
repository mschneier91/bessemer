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
   /// hypre BoomerAMG on a LOW-ORDER-REFINED rediscretization of the momentum
   /// block (for stiffer regimes). AMG coarsens the dense high-order operator
   /// poorly, so it is built on a spectrally-equivalent Q1-on-GLL-nodes LOR
   /// operator instead. A different assembly route than the matrix-free default,
   /// with a heavier per-refresh setup. (grad-div is omitted from the LOR
   /// operator -- gamma ~ h is negligible in the preconditioner.)
   ///
   /// NOTE the name: this is ALWAYS LOR-AMG, never plain BoomerAMG on the
   /// high-order operator -- there is no such path in this codebase. Mirrors
   /// APC::LORAMG on the Cahouet-Chabard side, which names the same scheme.
   LORAMG
};

} // namespace incns

#endif // INCNS_SOLVER_VELOCITY_PRECONDITIONER_HPP
