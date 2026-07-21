/**
 * @file mass_inverse.hpp
 * @brief Fixed-linear-operator mass-inverse strategies for the Cahouet-Chabard
 *        preconditioner (SPEC_cahouet_chabard_mfem.md par.6.1; docs/precond_cc.md).
 */
#ifndef INCNS_PRECOND_MASS_INVERSE_HPP
#define INCNS_PRECOND_MASS_INVERSE_HPP

#include "mfem.hpp"

#include <memory>
#include <stdexcept>

namespace incns
{

/// How a mass matrix is (approximately) inverted inside the preconditioner.
/// Every variant is a FIXED linear operator per application -- never a
/// tolerance-based solve (Krylov legality: the inner CG's operator must be
/// fixed, see docs/precond_cc.md).
enum class MassInvType
{
   Auto,       ///< DiagDirect iff the mass operator IS its diagonal, else Chebyshev.
   DiagDirect, ///< Direct reciprocal-diagonal multiply (collocated GLL mass).
   Chebyshev,  ///< Fixed-order Chebyshev with Jacobi (consistent, non-diagonal mass).
   AbsLumped   ///< Simplex fallback -- REJECTED here (bessemer is quad/hex only).
};

/**
 * @brief Applies @f$ M^{-1} @f$ as a fixed linear operator (mfem::Solver).
 *
 * Two live strategies:
 *  - @b DiagDirect (collocated GLL mass, exactly diagonal): the reciprocal
 *    diagonal is precomputed once at setup and applied as a device-aware
 *    elementwise multiply @c y = x .* d_inv. This path constructs NO solver,
 *    smoother, or iteration object of any kind (asserted by T1f via
 *    HasSolverObject()) -- it is a vector multiply, nothing more.
 *  - @b Chebyshev(k): fixed-order Chebyshev iteration with the Jacobi
 *    (reciprocal-diagonal) preconditioner; the largest eigenvalue of the
 *    diagonally preconditioned operator is estimated by power iteration once
 *    at setup and cached. Symmetric by construction, so it is legal inside CG.
 *
 * @c Auto resolves by probing whether the operator IS its diagonal (one
 * seeded-random matvec compared against the diagonal multiply, globally
 * reduced -- rank-consistent). Requesting @c DiagDirect on a non-diagonal mass
 * throws std::invalid_argument (spec par.7 validation). @c AbsLumped throws:
 * it exists for simplices at p >= 2, and bessemer rejects simplex meshes at
 * load, so the fallback has no callers.
 *
 * Works for any mass operator on true dofs (velocity vector mass, pressure
 * mass); the caller supplies the operator and its assembled diagonal.
 * Borrowed references: @p mass and @p diag must outlive this object.
 */
class MassInverse : public mfem::Solver
{
public:
   /**
    * @brief Resolve the strategy and do the one-time setup.
    * @param mass       Mass operator on true dofs (borrowed; PA or assembled).
    * @param diag       Assembled diagonal of @p mass (borrowed).
    * @param comm       Communicator for the diagonality probe / power iteration.
    * @param type       Strategy (Auto resolves via the diagonality probe).
    * @param cheb_order Chebyshev polynomial order (Chebyshev path only).
    * @throws std::invalid_argument on DiagDirect-with-non-diagonal-mass or
    *         AbsLumped (config validation -- identical on every rank).
    */
   MassInverse(const mfem::Operator& mass, const mfem::Vector& diag,
               MPI_Comm comm, MassInvType type = MassInvType::Auto,
               int cheb_order = 4)
      : mfem::Solver(diag.Size()), comm_(comm)
   {
      iterative_mode = false;

      const bool is_diagonal = ProbeDiagonal(mass, diag, comm);
      if (type == MassInvType::Auto)
      {
         type = is_diagonal ? MassInvType::DiagDirect : MassInvType::Chebyshev;
      }

      switch (type)
      {
         case MassInvType::DiagDirect:
         {
            if (!is_diagonal)
            {
               throw std::invalid_argument(
                  "mass_inverse: DiagDirect requested but the mass operator is "
                  "not diagonal (needs the collocated GLL basis/rule); use "
                  "Chebyshev or Auto");
            }
            // Reciprocal diagonal, device-resident: the application is an
            // elementwise multiply -- no solver object on this path.
            //
            // Spelled with mfem::Vector operators rather than a hand-written
            // forall on purpose: bessemer is compiled by mpicxx, never nvcc, so
            // MFEM_HOST_DEVICE expands to nothing here and forall's CUDA
            // dispatch is preprocessed out (general/forall.hpp, guarded on
            // __CUDACC__). The lambda would then run on the HOST over the device
            // pointers Read()/Write() hand back -- a segfault with "Invalid
            // permissions" on a real GPU. These operators live inside libmfem,
            // which IS nvcc-built, so they hit genuine device kernels.
            d_inv_.SetSize(diag.Size());
            d_inv_.UseDevice(true);
            d_inv_ = 1.0;
            d_inv_ /= diag;
            break;
         }
         case MassInvType::Chebyshev:
         {
            // Fixed-order Chebyshev, Jacobi-preconditioned; eigenvalue bound
            // from power iteration at setup (deterministic seed), cached
            // inside the smoother. A fixed symmetric linear operator. MFEM
            // hardcodes the polynomial coefficients up to order 5.
            if (cheb_order < 1 || cheb_order > 5)
            {
               throw std::invalid_argument(
                  "mass_inverse: Chebyshev order must be in [1, 5] (MFEM's "
                  "OperatorChebyshevSmoother implements those orders)");
            }
            cheb_ = std::make_unique<mfem::OperatorChebyshevSmoother>(
                       mass, diag, empty_ess_, cheb_order, comm);
            break;
         }
         case MassInvType::AbsLumped:
            throw std::invalid_argument(
               "mass_inverse: AbsLumped is a simplex-mesh fallback; bessemer "
               "rejects simplices at mesh load, so it has no callers -- use "
               "DiagDirect (collocated GLL) or Chebyshev");
         case MassInvType::Auto: break; // unreachable (resolved above)
      }
      resolved_ = type;
   }

   /**
    * @brief Apply @f$ y = M^{-1} x @f$ (fixed cost; never iterates to tolerance).
    * @param x Input true-dof vector.
    * @param y Output true-dof vector.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      if (resolved_ == MassInvType::DiagDirect)
      {
         // Two device-aware mfem::Vector kernels (see the setup comment above
         // for why this is not a hand-written forall).
         y = x;
         y *= d_inv_;
         return;
      }
      cheb_->Mult(x, y);
   }

   /// Required by mfem::Solver; the strategy is fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

   /// @return The strategy actually in use (after Auto resolution).
   MassInvType Resolved() const { return resolved_; }

   /// @return True iff a solver/smoother object exists -- MUST be false on the
   ///         DiagDirect path (T1f: that path is a bare vector multiply).
   bool HasSolverObject() const { return static_cast<bool>(cheb_); }

private:
   /// One seeded-random matvec vs. the diagonal multiply, globally reduced --
   /// identical verdict on every rank.
   static bool ProbeDiagonal(const mfem::Operator& mass,
                             const mfem::Vector& diag, MPI_Comm comm)
   {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      mfem::Vector x(diag.Size()), y_op(diag.Size()), y_diag(diag.Size());
      x.Randomize(12345 + rank); // deterministic, distinct per rank
      mass.Mult(x, y_op);
      y_diag = x;
      y_diag *= diag;
      y_diag -= y_op;
      const double err2 = mfem::InnerProduct(comm, y_diag, y_diag);
      const double ref2 = mfem::InnerProduct(comm, y_op, y_op);
      return err2 <= 1e-24 * ref2; // relative ~1e-12: exact-diagonal or not
   }

   MPI_Comm comm_;                 ///< Communicator (probe / power iteration).
   MassInvType resolved_ = MassInvType::Auto; ///< Post-Auto strategy.
   mfem::Vector d_inv_;            ///< Reciprocal diagonal (DiagDirect only).
   mfem::Array<int> empty_ess_;    ///< No essential dofs on mass solves.
   /// Chebyshev smoother (Chebyshev path only; null on DiagDirect -- T1f).
   std::unique_ptr<mfem::OperatorChebyshevSmoother> cheb_;
};

} // namespace incns

#endif // INCNS_PRECOND_MASS_INVERSE_HPP
