/**
 * @file nullspace.hpp
 * @brief Constant-pressure nullspace projector for the Cahouet-Chabard
 *        preconditioner (docs/design/SPEC_cahouet_chabard_mfem.md par.4; docs/precond_cc.md).
 *
 * Detection is NOT here: whether the constant pressure mode exists is decided
 * by BoundaryConditions::PressureNullspaceExists() (singular iff no
 * traction/outflow BC on any boundary attribute that actually exists in the
 * mesh -- a phrasing that survives periodic meshes, including the fully
 * periodic empty-boundary case). This header supplies the treatment: the l2
 * projector applied to RHS and iterates of singular solves.
 */
#pragma once

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

/**
 * @brief Wraps an SPD operator as @f$ P \circ S @f$ (project the output onto
 *        the constant-mode complement).
 *
 * Used as the inner-CG operator on singular domains: projecting every
 * application controls the floating-point drift of the constant mode more
 * strongly than the spec's every-k_reproj iterate re-orthogonalization (mfem's
 * CG exposes no iterate hook; the projection costs one dot + axpy against an
 * application containing B, B^T, and an AMG V-cycle -- negligible).
 */
class ProjectedOperator : public mfem::Operator
{
public:
   /**
    * @brief Compose the projected operator (both borrowed).
    * @param S Underlying SPD operator.
    * @param P Constant-mode projector.
    */
   ProjectedOperator(const mfem::Operator& S, const ConstantPressureProjector& P)
      : mfem::Operator(S.Height()), S_(S), P_(P) {}

   /**
    * @brief Apply @f$ y = P(Sx) @f$.
    * @param x Input true-dof vector.
    * @param y Projected output.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      S_.Mult(x, y);
      P_.Project(y);
   }

private:
   const mfem::Operator& S_;               ///< Underlying operator (borrowed).
   const ConstantPressureProjector& P_;    ///< Projector (borrowed).
};

/**
 * @brief mfem::OrthoSolver with the projection on the device.
 *
 * mfem::OrthoSolver (MFEM 1c19aba, linalg/solvers.cpp, marked "TODO:
 * GPU/device implementation") subtracts the mean of its input and of its
 * output in a host loop: two device-host round trips of the vector around
 * every solve. Around the Cahouet-Chabard pressure AMG that was 32 per
 * channel step, the GPU idle ~0.16 ms each (2026-10-15). This is the same
 * algorithm -- the sum by Vector::Sum, summed over the ranks, then
 * v - sum / global size -- with the subtraction as a device-aware vector
 * operation: the same IEEE operation on every entry, so the same values
 * (nullspace_test). Drop-in for the uses here: SetSolver, SetOperator, Mult,
 * iterative_mode passed on to the wrapped solver.
 */
class DeviceOrthoSolver : public mfem::Solver
{
public:
   /// @param comm Communicator of the vectors (the global size and sums).
   explicit DeviceOrthoSolver(MPI_Comm comm) : mfem::Solver(0, false),
      comm_(comm) {}

   /// Wrap @p s (borrowed; square).
   void SetSolver(mfem::Solver& s)
   {
      solver_ = &s;
      height = s.Height();
      width = s.Width();
      MFEM_VERIFY(height == width, "DeviceOrthoSolver: the solver must be "
                  "square");
      global_size_ = -1;
   }

   /// Forwarded to the wrapped solver.
   void SetOperator(const mfem::Operator& op) override
   {
      MFEM_VERIFY(solver_, "DeviceOrthoSolver: call SetSolver first");
      solver_->SetOperator(op);
      height = solver_->Height();
      width = solver_->Width();
      MFEM_VERIFY(height == width, "DeviceOrthoSolver: the solver must be "
                  "square");
      global_size_ = -1;
   }

   /// x = P S^{-1} P b, P the projection off the constant vector.
   void Mult(const mfem::Vector& b, mfem::Vector& x) const override
   {
      MFEM_VERIFY(solver_, "DeviceOrthoSolver: call SetSolver first");
      MFEM_VERIFY(height == solver_->Height(), "DeviceOrthoSolver: the solver "
                  "was modified externally; call SetSolver again");
      MFEM_VERIFY(height == b.Size() && height == x.Size(),
                  "DeviceOrthoSolver: incompatible vector sizes");
      Orthogonalize(b, b_ortho_);
      solver_->iterative_mode = iterative_mode;
      solver_->Mult(b_ortho_, x);
      Orthogonalize(x, x);
   }

private:
   /// v_ortho = v - (sum of v) / (global size); v_ortho may be v.
   void Orthogonalize(const mfem::Vector& v, mfem::Vector& v_ortho) const
   {
      if (global_size_ == -1)
      {
         global_size_ = height;
         MPI_Allreduce(MPI_IN_PLACE, &global_size_, 1, HYPRE_MPI_BIG_INT,
                       MPI_SUM, comm_);
      }
      mfem::real_t global_sum = v.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &global_sum, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm_);
      const mfem::real_t ratio =
         global_sum / static_cast<mfem::real_t>(global_size_);
      if (&v_ortho != &v)
      {
         v_ortho.SetSize(v.Size());
         v_ortho.UseDevice(true);
         v_ortho = v;
      }
      v_ortho -= ratio;
   }

   mfem::Solver* solver_ = nullptr;         ///< Wrapped solver (borrowed).
   MPI_Comm comm_;                          ///< Communicator.
   mutable HYPRE_BigInt global_size_ = -1;  ///< Global size (lazy).
   mutable mfem::Vector b_ortho_;           ///< The projected input.
};

} // namespace incns
