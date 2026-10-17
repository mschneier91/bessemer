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

private:
   int size_ = 0;        ///< Entries per buffer.
   mfem::Vector buf_[2]; ///< The buffers (device).
};

} // namespace incns
