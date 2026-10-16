/**
 * @file persistent_fgmres.hpp
 * @brief MFEM's FGMRES with its work vectors kept between solves.
 */
#pragma once

#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief mfem::FGMRESSolver whose residual and Krylov vectors outlive a
 *        solve.
 *
 * mfem::FGMRESSolver::Mult allocates the residual and every Krylov vector it
 * uses (v_i and z_i, full system size, device memory on a GPU) at each call
 * and frees them at its end: about 2 k + 2 allocations and frees per solve of
 * k iterations, and each cudaFree waits for the device. Mult() here is the
 * same algorithm line for line (the same results, bitwise; persistent_fgmres
 * _test) with the vectors kept: allocated on first use, and after each solve
 * those beyond what it used plus two are freed, so a long solve (the first
 * steps) does not pin its memory for the run.
 */
class PersistentFGMRESSolver : public mfem::FGMRESSolver
{
public:
   /**
    * @brief As mfem::FGMRESSolver(comm).
    * @param comm Communicator of the vectors.
    */
   explicit PersistentFGMRESSolver(MPI_Comm comm) : mfem::FGMRESSolver(comm) {}

   /**
    * @brief mfem::FGMRESSolver::Mult with kept work vectors.
    * @param b Right-hand side.
    * @param x Solution (initial guess if iterative_mode).
    */
   void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

   /// Bytes of the vectors kept between solves (device memory on a GPU).
   double KeptBytes() const;

   /// Frees the kept vectors.
   void ReleaseWork() const;

private:
   /// Kept vector @p i of @p pool, allocated with size @p n if needed.
   mfem::Vector* Work(std::vector<std::unique_ptr<mfem::Vector>>& pool,
                      int i, int n) const;
   /// Frees the vectors of @p pool from index @p keep on.
   static void Trim(std::vector<std::unique_ptr<mfem::Vector>>& pool,
                    int keep);

   mutable mfem::Vector r_;                                 ///< Residual.
   mutable std::vector<std::unique_ptr<mfem::Vector>> v_;   ///< Krylov basis.
   mutable std::vector<std::unique_ptr<mfem::Vector>> z_;   ///< Preconditioned.
};

} // namespace incns
