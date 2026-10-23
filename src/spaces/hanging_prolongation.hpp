/**
 * @file hanging_prolongation.hpp
 * @brief The prolongation of a nonconforming (hanging-node) parallel space as
 *        a device operator: copies for the unconstrained dofs, weighted sums
 *        for the hanging ones, in place of MFEM's assembled HypreParMatrix and
 *        its stored transpose.
 */
#pragma once

#include "mfem.hpp"

#include <cstddef>
#include <memory>

namespace incns
{

class TrueDofExchange;

/**
 * @brief P (true dofs -> local dofs) of a nonconforming space and its
 *        transpose, built once from MFEM's assembled P.
 *
 * On a nonconforming space MFEM's prolongation is a HypreParMatrix: an
 * identity row for every unconstrained local dof and the interpolation
 * weights of each hanging one, with its transpose stored as well
 * (EnsureMultTranspose). Every true-dof apply ran both through hypre's SpMV:
 * 21 % of a curved adaptive step on an H100, 25 % at np 4 (throughput.md).
 * Here:
 * - Mult: a local dof with an identity row copies its true dof (or ghost); a
 *   hanging dof sums its weighted masters, in the order of P's row (its local
 *   columns, then the others, as hypre's SpMV);
 * - MultTranspose: per true dof and per ghost, the weighted sum of the local
 *   dofs that refer to it, in the order of the local dofs; the ghosts' sums are
 *   then added at their owners in a fixed order (TrueDofExchange): no atomics,
 *   no stored transpose;
 * - the masters other ranks own (P's off-processor columns) arrive by the
 *   exchange's gather, the only communication.
 * Mult reproduces hypre's sums; the transpose's order differs from hypre's
 * stored transpose (roundoff).
 */
class HangingProlongation : public mfem::Operator
{
public:
   /// @param fes A nonconforming space (borrowed; its P is read here only).
   ///            Collective.
   explicit HangingProlongation(const mfem::ParFiniteElementSpace& fes);
   ~HangingProlongation() override;

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;
   void AbsMult(const mfem::Vector& x, mfem::Vector& y) const override;
   void AbsMultTranspose(const mfem::Vector& x,
                         mfem::Vector& y) const override;
   /// The device's memory class: the applies run there.
   mfem::MemoryClass GetMemoryClass() const override;

   /// @return The local dofs whose row is not the identity (hanging ones).
   int NumHanging() const { return nh_; }
   /// @return Bytes of device data held (maps, weights, exchange buffers).
   std::size_t MemoryBytes() const;

private:
   template <bool ABS>
   void Apply(const mfem::Vector& x, mfem::Vector& y) const;
   template <bool ABS>
   void ApplyTranspose(const mfem::Vector& x, mfem::Vector& y) const;

   int nt_ = 0; ///< True dofs.
   int ng_ = 0; ///< Ghosts: P's off-processor columns.
   int nh_ = 0; ///< Hanging rows.
   /// Per local dof: its column (a true dof, or nt_ + a ghost) where its row
   /// is the identity, else -1 - its hanging row.
   mfem::Array<int> src_;
   mfem::Array<int> hoff_, hcol_; ///< The hanging rows' columns (CSR).
   mfem::Vector hw_;              ///< Their weights.
   /// Per column (the true dofs, then the ghosts): its entries' local dofs,
   /// ascending (CSR).
   mfem::Array<int> toff_, trow_;
   mfem::Vector tw_; ///< Their weights.
   /// The ghosts' gather and reduce (more than one rank; else null).
   std::unique_ptr<TrueDofExchange> ex_;
};

} // namespace incns
