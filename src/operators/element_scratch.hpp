/**
 * @file element_scratch.hpp
 * @brief One pair of element-vector (E-vector) buffers lent to the partially
 *        assembled operators of a time integrator, which never run inside one
 *        another's apply.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Two device buffers of the velocity space's element-vector size.
 *
 * A partially assembled operator restricts its input into one element
 * vector, accumulates into another and scatters that back. MFEM's forms keep
 * such a pair each for the run (2 x 582 MiB per velocity form at Q6 on 42^3
 * elements); the integrators are kernels and the composed operators call
 * their parts one after another, so no apply runs inside another, and one
 * pair serves them all (PaOperator, PaMixedOperator, the OIFS convection
 * kernel). Lend() makes a borrower's vector an alias of a buffer
 * (Vector::MakeRef); the buffers must outlive every borrower.
 *
 * Also a pair of the velocity space's local-dof (L-vector) size for the
 * true-dof operators P^T A P, whose work vectors P x and A P x MFEM's
 * RAPOperator keeps per operator; they too never run inside one another
 * (LendLocal; allocated on the first call: where the prolongation is the
 * identity -- one rank, no hanging nodes -- there are no such operators).
 */
class ElementScratch
{
public:
   /// @param vfes The velocity space: its lexicographic element vector is the
   ///             largest a borrower holds.
   explicit ElementScratch(const mfem::ParFiniteElementSpace& vfes);

   /**
    * @brief Make @p v an alias of the first @p n entries of a buffer.
    * @param which 0 (an apply's input side) or 1 (its output side).
    * @param n     Entries (at most Size()).
    * @param v     The vector (its own data, if any, is freed).
    */
   void Lend(int which, int n, mfem::Vector& v);

   /// @return Entries per buffer.
   int Size() const { return size_; }

   /**
    * @brief Lend() for the local-dof buffers (allocated on the first call).
    * @param which 0 (P x) or 1 (A P x).
    * @param n     Entries (at most LocalSize()).
    * @param v     The vector (its own data, if any, is freed).
    */
   void LendLocal(int which, int n, mfem::Vector& v);

   /// @return Entries per local-dof buffer (the velocity's local dofs).
   int LocalSize() const { return lsize_; }

private:
   int size_ = 0;        ///< Entries per buffer.
   mfem::Vector buf_[2]; ///< The buffers (device).
   int lsize_ = 0;        ///< Entries per local-dof buffer.
   mfem::Vector lbuf_[2]; ///< The local-dof buffers (device; on first use).
};

} // namespace incns
