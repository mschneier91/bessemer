/**
 * @file mixed_poisson_op.hpp
 * @brief The consistent mixed Poisson operator B M_v^-1 B^T -- matrix-free
 *        composition (docs/design/SPEC_cahouet_chabard_mfem.md par.2.3/par.3;
 *        docs/precond_cc.md).
 *
 * INVARIANT (spec par.8 item 16): this operator is NEVER assembled into a
 * matrix -- no code path for that may exist, in this class or elsewhere. It is
 * exactly the composition B ( M_v^-1 ( B^T x ) ) of three fixed applications.
 */
#pragma once

#include "precond/mass_inverse.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief @f$ y = B\, M_v^{-1} B^T x @f$ on pressure true dofs, matrix-free.
 *
 * The consistent replacement for the assembled pressure Laplacian in the
 * Cahouet-Chabard Schur block (Creff & Guermond): in the reaction limit
 * (sigma large) the true Schur complement S = B A^-1 B^T tends to
 * sigma^-1 B M_v^-1 B^T, so this operator captures exactly the part the
 * legacy L_p surrogate gets wrong on fine meshes.
 *
 * Requirements enforced by construction:
 *  - @p B must be the ELIMINATED divergence operator (columns at essential
 *    velocity dofs zeroed -- what FormRectangularSystemMatrix produces). Using
 *    a raw B here while the outer system uses the eliminated one silently
 *    degrades convergence near boundaries (spec par.3; test T1c).
 *  - @p mv_inv must be a FIXED linear operator (MassInverse: diagonal multiply
 *    or fixed-order Chebyshev) -- never a tolerance-based solve; that is what
 *    legalizes CG on this operator (spec par.2.4). MassInverse cannot express a
 *    tolerance solve, so the requirement holds by type.
 *
 * The operator is symmetric positive (semi-)definite on pressure true dofs:
 * semi-definite exactly when the constant pressure mode is present (enclosed /
 * periodic -- B^T 1 = 0); pair it with ConstantPressureProjector there.
 * With M_v^-1 = diag(M_v)^-1 (collocated GLL), B^T x lands only on free
 * velocity dofs (eliminated rows are zero), the diagonal multiply preserves
 * that, and B maps back -- no special essential-dof handling is needed here.
 */
class MixedPoissonOperator : public mfem::Operator
{
public:
   /**
    * @brief Compose the operator from its three fixed parts (all borrowed).
    * @param B      Eliminated divergence operator, velocity -> pressure true
    *               dofs (rectangular; MultTranspose applies B^T).
    * @param mv_inv Fixed velocity mass inverse (MassInverse).
    */
   MixedPoissonOperator(const mfem::Operator& B, const MassInverse& mv_inv)
      : mfem::Operator(B.Height()), B_(B), mv_inv_(mv_inv),
        vu_(B.Width()), vu2_(B.Width())
   {
      vu_.UseDevice(true);
      vu2_.UseDevice(true);
   }

   /**
    * @brief Apply @f$ y = B (M_v^{-1} (B^T x)) @f$.
    * @param x Pressure true-dof input.
    * @param y Pressure true-dof output.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      B_.MultTranspose(x, vu_);   // B^T: pressure -> velocity (PA gradient)
      mv_inv_.Mult(vu_, vu2_);    // M_v^-1: fixed diagonal / Chebyshev
      B_.Mult(vu2_, y);           // B: velocity -> pressure (PA divergence)
   }

private:
   const mfem::Operator& B_;      ///< Eliminated divergence block (borrowed).
   const MassInverse& mv_inv_;    ///< Fixed velocity mass inverse (borrowed).
   mutable mfem::Vector vu_;      ///< Velocity-sized scratch (device).
   mutable mfem::Vector vu2_;     ///< Velocity-sized scratch (device).
};

} // namespace incns
