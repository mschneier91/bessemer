/**
 * @file oseen_convection.hpp
 * @brief The linearized (Oseen) convection of the semi-implicit and implicit
 *        BDF2 schemes (docs/design/oseen_bdf2.md), per velocity component:
 *        skew-symmetric convection by a given wind plus orthogonal-subscale
 *        (OSS) stabilization, partially assembled.
 */
#pragma once

#include "oseen/oseen_wind.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief The Oseen convection of one scalar field u by a wind w, on true dofs:
 * @f[ N(w) = C(w) + \tfrac12 (\nabla\cdot w) M + S(w), \qquad
 *     S = K_\tau - C_\tau^T W_\tau^{-1} C_\tau , @f]
 * C(w) = ((w.grad) phi_j, phi_i) (Gauss, order 3k), the skew term's mass
 * with 1/2 div w at the same rule; S the OSS stabilization (Codina's
 * term-by-term form): K_tau = (tau_c w.grad phi_j, w.grad phi_i) and C_tau =
 * (tau_c w.grad phi_j, phi_l) at the same Gauss rule of order 2k + 2, W_tau
 * the diagonal (tau_c phi_l, 1) at the GLL rule -- the tau-weighted L2
 * projection onto the space itself, so S = sum_K tau_K (P^perp (w.grad u),
 * w.grad v)_K, symmetric positive semidefinite with tau_c one value per
 * element (OseenWindData).
 *
 * On a vector field ordered by nodes, ComponentwiseOperator applies it to
 * each component. Assemble() re-reads the wind data (after
 * OseenWindData::Update) into the partially assembled forms.
 */
class ScalarOseenOperator : public mfem::Operator
{
public:
   /**
    * @param sfes The scalar space of one velocity component (the velocity's
    *             collection, vdim 1; borrowed).
    * @param wind The wind data (borrowed; Update() it, then Assemble()).
    * @param oss  Include the OSS stabilization S.
    */
   ScalarOseenOperator(mfem::ParFiniteElementSpace& sfes, OseenWindData& wind,
                       bool oss = true);

   /// (Re)assemble the forms from the wind data's current values; needed
   /// before the first apply.
   void Assemble();

   /// y = N(w) x.
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   /// y = S(w) x, the stabilization alone (tests).
   void MultStabilization(const mfem::Vector& x, mfem::Vector& y) const;

   /// y = [C(w) + 1/2 (div w) M] x, the skew-symmetric convection alone.
   void MultConvection(const mfem::Vector& x, mfem::Vector& y) const;

private:
   mfem::ParFiniteElementSpace& sfes_; ///< Scalar space (borrowed).
   OseenWindData& wind_;               ///< Wind data (borrowed).
   bool oss_;                          ///< With the OSS term.
   std::unique_ptr<mfem::ParBilinearForm> conv_; ///< C(w) + 1/2 div w M.
   std::unique_ptr<mfem::ParBilinearForm> ktau_; ///< K_tau.
   std::unique_ptr<mfem::ParBilinearForm> ctau_; ///< C_tau.
   mfem::OperatorHandle Conv_, Ktau_, Ctau_;     ///< Their true-dof operators.
   mfem::Vector winv_;          ///< 1 / W_tau (0 where W_tau = 0).
   mutable mfem::Vector t1_, t2_; ///< Work vectors.
};

/**
 * @brief A scalar operator applied to each block of a vector by nodes (the
 *        components of a byNODES vector space's true dofs).
 */
class ComponentwiseOperator : public mfem::Operator
{
public:
   /// @param op   The scalar operator (borrowed), square.
   /// @param vdim Number of components.
   ComponentwiseOperator(const mfem::Operator& op, int vdim)
      : mfem::Operator(vdim * op.Height()), op_(op), vdim_(vdim) {}

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

private:
   const mfem::Operator& op_; ///< Scalar operator (borrowed).
   int vdim_;                 ///< Components.
};

} // namespace incns
