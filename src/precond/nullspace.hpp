/**
 * @file nullspace.hpp
 * @brief Constant-pressure nullspace projector for the Cahouet-Chabard
 *        preconditioner (SPEC_cahouet_chabard_mfem.md par.4; docs/precond_cc.md).
 *
 * Detection is NOT here: whether the constant pressure mode exists is decided
 * by BoundaryConditions::PressureNullspaceExists() (singular iff no
 * traction/outflow BC on any boundary attribute that actually exists in the
 * mesh -- a phrasing that survives periodic meshes, including the fully
 * periodic empty-boundary case). This header supplies the treatment: the l2
 * projector applied to RHS and iterates of singular solves.
 */
#ifndef INCNS_PRECOND_NULLSPACE_HPP
#define INCNS_PRECOND_NULLSPACE_HPP

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Euclidean (l2) projector onto the complement of the constant
 *        pressure mode: @f$ q \leftarrow q - \frac{1^T q}{1^T 1}\,1 @f$.
 *
 * The nullspace vector of @f$ B M_v^{-1} B^T @f$ (an SPD operator on
 * COEFFICIENT vectors) is the coefficient vector of the constant-1 pressure
 * function -- all-ones for the nodal H1 Lagrange basis used here -- and the
 * correct projection is plain Euclidean on true dofs. Do NOT
 * mass-orthogonalize here; the mass-weighted mean is only for reporting the
 * physical mean-zero pressure at output (post/pressure_mean).
 *
 * The 1-vector is stored explicitly (rather than hardcoding "subtract the
 * average") so a future non-nodal pressure basis (L2/DG, spec par.12) only has
 * to change how 1 is built -- the separable-functions accommodation the spec
 * asks for. Device-resident; all reductions global (true dofs partition
 * cleanly across ranks, including under periodic identification).
 */
class ConstantPressureProjector
{
public:
   /**
    * @brief Build the constant-mode coefficient vector for a pressure space.
    * @param pfes Pressure finite element space (nodal H1 in v1).
    */
   explicit ConstantPressureProjector(mfem::ParFiniteElementSpace& pfes)
      : comm_(pfes.GetComm()), ones_(pfes.GetTrueVSize())
   {
      // Nodal H1 Lagrange: the constant-1 function's coefficient vector is
      // all-ones on true dofs (identification-safe: a true dof represents the
      // identified node once). A non-nodal basis must project the constant
      // instead -- v1 rejects those spaces upstream (config validation).
      ones_.UseDevice(true);
      ones_ = 1.0;
      ones_norm2_ = mfem::InnerProduct(comm_, ones_, ones_);
   }

   /**
    * @brief Remove the constant component in place.
    * @param q Pressure-space true-dof vector.
    */
   void Project(mfem::Vector& q) const
   {
      const double c = mfem::InnerProduct(comm_, ones_, q) / ones_norm2_;
      q.Add(-c, ones_);
   }

   /**
    * @brief The constant component @f$ (1^T q)/(1^T 1) @f$ (diagnostic/tests).
    * @param q Pressure-space true-dof vector.
    * @return The l2 coefficient of the constant mode in @p q.
    */
   double ConstantComponent(const mfem::Vector& q) const
   {
      return mfem::InnerProduct(comm_, ones_, q) / ones_norm2_;
   }

   /// @return The constant-mode coefficient vector (borrowed; e.g. for tests).
   const mfem::Vector& One() const { return ones_; }

private:
   MPI_Comm comm_;        ///< Pressure-space communicator.
   mfem::Vector ones_;    ///< Coefficient vector of the constant-1 function.
   double ones_norm2_;    ///< Global @f$ 1^T 1 @f$ (cached).
};

} // namespace incns

#endif // INCNS_PRECOND_NULLSPACE_HPP
