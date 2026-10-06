/**
 * @file velocity_preconditioner.hpp
 * @brief Choice of preconditioner for the velocity (momentum) block.
 */
#pragma once

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

/**
 * @brief Velocity-block preconditioner when the semi-implicit rotational term
 *        N is in the momentum block (rotational_convection_pa_spec.md par.7.5;
 *        deck `solver.rotation_pc: symmetric|pbj_only|pbj_krylov`).
 *
 * N is skew with an exactly zero diagonal, so scalar Jacobi -- and LOR-AMG,
 * built on sigma M + nu K -- cannot see it; the outer FGMRES always applies
 * the true operator, so every choice is correct, they differ in iterations.
 * The two point-block Jacobi choices replace the velocity PC on BOTH Schur
 * paths (VelocityPreconditioner / cc.a_pc are then not used).
 */
enum class RotationVelocityPC
{
   /// The usual velocity PC (Jacobi / Chebyshev / LOR-AMG) on the symmetric
   /// part only; N enters through the outer operator. The default.
   Symmetric,
   /// Point-block Jacobi (the nodal dim x dim blocks of the full block,
   /// rotation included) applied once. Cheapest.
   PbjOnly,
   /// GMRES on the full velocity block preconditioned by point-block Jacobi,
   /// loose tolerance -- for runs where |omega| dt exceeds about 1 over a
   /// sizable part of the domain.
   PbjKrylov
};

} // namespace incns
