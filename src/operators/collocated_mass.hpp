/**
 * @file collocated_mass.hpp
 * @brief The collocated (GLL) velocity mass as what it is, a diagonal: its
 *        diagonal assembled without caching mesh Jacobians, the operator that
 *        multiplies by it, and a block plus a multiple of it.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Assemble the diagonal of the collocated vector mass on true dofs:
 *        W det J at each GLL node, summed over the elements that share it.
 *
 * MFEM's partial assembly (VectorMassIntegrator on the GLL rule, then
 * AssembleDiagonal) gives the same values to roundoff, but through the mesh's
 * cache of geometric factors, which keeps the Jacobians at the GLL points (9
 * numbers per point: 1.7 GiB at Q6 on 42^3 elements) for the run. Here they
 * are computed for the call and freed (only det J is requested).
 * @param vfes Velocity space: the GLL nodal H1 basis on tensor elements.
 * @param ir   The collocated GLL rule (RuleBook::CollocatedMass), one point
 *             per node.
 * @param diag Output, true dofs (resized, device-enabled).
 */
void AssembleCollocatedMassDiagonal(mfem::ParFiniteElementSpace& vfes,
                                    const mfem::IntegrationRule& ir,
                                    mfem::Vector& diag);

/// y = d .* x, element by element: the collocated mass on true dofs.
class DiagonalOperator : public mfem::Operator
{
public:
   /// @param d The diagonal (borrowed: must outlive this operator).
   explicit DiagonalOperator(const mfem::Vector& d);

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
   {
      Mult(x, y);
   }

private:
   const mfem::Vector& d_; ///< The diagonal (borrowed).
};

/**
 * @brief y = A x + c (d .* x), with @p c read at every apply: the momentum
 *        block c0 M + nu K (+ grad-div) when M is the collocated diagonal, so
 *        a new mass factor changes nothing that has to be re-assembled.
 *
 * @p x and @p y must be different vectors (A writes y before d .* x is added).
 */
class DiagonalShiftOperator : public mfem::Operator
{
public:
   /**
    * @param A The block (borrowed).
    * @param d The diagonal (borrowed).
    * @param c The factor (borrowed, read at each apply; 0 adds nothing).
    */
   DiagonalShiftOperator(const mfem::Operator& A, const mfem::Vector& d,
                         const double& c);

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

private:
   const mfem::Operator& A_; ///< The block (borrowed).
   const mfem::Vector& d_;   ///< The diagonal (borrowed).
   const double& c_;         ///< The factor (borrowed).
};

} // namespace incns
