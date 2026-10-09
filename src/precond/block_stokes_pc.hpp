/**
 * @file block_stokes_pc.hpp
 * @brief Block Diag / LowerTri / UpperTri preconditioner shapes on the
 *        canonical symmetric system [A B^T; B 0] with internal p~ = -p_phys
 *        (docs/design/SPEC_cahouet_chabard_mfem.md par.2.4/par.6.5; docs/precond_cc.md).
 */
#pragma once

#include "precond/cahouet_chabard.hpp" // BlockPCShape
#include "mfem.hpp"

namespace incns
{

/**
 * @brief Block preconditioner @f$ P = [\hat A\; B^T; 0\; -\hat S] @f$
 *        (UpperTri default; Diag and LowerTri via config) for the symmetric
 *        saddle-point system on @f$ \tilde p = -p @f$.
 *
 * The MINUS of the pressure row lives HERE and only here (SPEC par.6.5 P1/P1c):
 * the Schur block solver applies the positive @f$ P_S^{-1} @f$; every shape
 * negates its output exactly once. Application per shape, given r = [r_u; r_p]:
 *
 *  - UpperTri: z_p = -Shat^-1 r_p;  z_u = Ahat^-1 (r_u - B^T z_p)   [spec P1-P3]
 *  - LowerTri: z_u = Ahat^-1 r_u;   z_p = -Shat^-1 (r_p - B z_u)
 *  - Diag:     z_u = Ahat^-1 r_u;   z_p = -Shat^-1 r_p
 *
 * Ahat^-1 and Shat^-1 are borrowed fixed-cost Solvers (V-cycles, the CC Schur
 * PC); the Schur block is NONLINEAR (inner CG), so the outer Krylov must be
 * FGMRES. B/B^T are the ELIMINATED blocks of the same system being solved.
 */
class BlockStokesPC : public mfem::Solver
{
public:
   /**
    * @brief Wire the block preconditioner (everything borrowed).
    * @param offsets   True-dof block offsets [0, n_u, n_u + n_p].
    * @param a_inv     Velocity block preconditioner Ahat^-1.
    * @param schur_inv Positive Schur approximate inverse P_S^-1.
    * @param B         Eliminated divergence operator (LowerTri needs it).
    * @param shape     Diag / LowerTri / UpperTri.
    */
   BlockStokesPC(const mfem::Array<int>& offsets, mfem::Solver& a_inv,
                 mfem::Solver& schur_inv, const mfem::Operator& B,
                 BlockPCShape shape)
      : mfem::Solver(offsets.Last()), offsets_(offsets), a_inv_(a_inv),
        schur_inv_(schur_inv), B_(B), shape_(shape),
        scratch_u_(offsets[1] - offsets[0]), scratch_p_(offsets[2] - offsets[1])
   {
      MFEM_VERIFY(offsets_.Size() == 3, "block_stokes_pc: need 3 offsets");
      scratch_u_.UseDevice(true);
      scratch_p_.UseDevice(true);
   }

   /**
    * @brief Apply the block preconditioner (one segregated sweep).
    * @param x Input block residual [r_u; r_p].
    * @param y Output block correction [z_u; z_p].
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      const mfem::BlockVector xb(const_cast<mfem::Vector&>(x), offsets_);
      mfem::BlockVector yb(y, offsets_);
      const mfem::Vector& r_u = xb.GetBlock(0);
      const mfem::Vector& r_p = xb.GetBlock(1);
      mfem::Vector& z_u = yb.GetBlock(0);
      mfem::Vector& z_p = yb.GetBlock(1);

      switch (shape_)
      {
         case BlockPCShape::UpperTri:
            // P1: z_p <- -P_S^-1 r_p (the one minus of the pressure row).
            schur_inv_.Mult(r_p, z_p);
            z_p.Neg();
            // P2: r_u' <- r_u - B^T z_p (one B^T action).
            B_.MultTranspose(z_p, scratch_u_);
            scratch_u_.Neg();
            scratch_u_ += r_u;
            // P3: z_u <- Ahat^-1 r_u'.
            a_inv_.Mult(scratch_u_, z_u);
            break;
         case BlockPCShape::LowerTri:
            a_inv_.Mult(r_u, z_u);
            B_.Mult(z_u, scratch_p_);
            scratch_p_.Neg();
            scratch_p_ += r_p;          // r_p - B z_u
            schur_inv_.Mult(scratch_p_, z_p);
            z_p.Neg();
            break;
         case BlockPCShape::Diag:
            a_inv_.Mult(r_u, z_u);
            schur_inv_.Mult(r_p, z_p);
            z_p.Neg();
            break;
      }

      // BlockVector views may hold stale data pointers after block writes.
      yb.SyncFromBlocks();
      y.SyncMemory(yb);
   }

   /// Required by mfem::Solver; the blocks are fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

private:
   mfem::Array<int> offsets_;      ///< Block offsets (copied).
   mfem::Solver& a_inv_;           ///< Velocity block PC (borrowed).
   mfem::Solver& schur_inv_;       ///< Positive Schur inverse (borrowed).
   const mfem::Operator& B_;       ///< Eliminated divergence (borrowed).
   BlockPCShape shape_;            ///< Diag / LowerTri / UpperTri.
   mutable mfem::Vector scratch_u_; ///< Velocity-sized scratch (device).
   mutable mfem::Vector scratch_p_; ///< Pressure-sized scratch (device).
};

} // namespace incns
