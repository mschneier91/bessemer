/**
 * @file stokes_operator.hpp
 * @brief The Stokes saddle-point blocks (mass, viscous, divergence), partially
 *        assembled.
 */
#ifndef INCNS_OPERATORS_STOKES_OPERATOR_HPP
#define INCNS_OPERATORS_STOKES_OPERATOR_HPP

#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Options controlling the Stokes block assembly.
struct StokesOperatorOptions
{
   /// Kinematic viscosity (constant).
   double nu = 1.0;
   /// Use the collocated Gauss-Lobatto mass rule (diagonal, SEM lumping)
   /// instead of the default Gauss-Legendre rule.
   bool collocated_mass = false;
};

/**
 * @brief Assembles and owns the operator blocks of the Stokes saddle-point
 *        system on the mixed velocity/pressure spaces.
 *
 * Blocks (all partial assembly, AssemblyLevel::PARTIAL, acting on true dofs):
 *  - @b Mass:       velocity vector mass @c M (GL rule, or collocated GLL when
 *    StokesOperatorOptions::collocated_mass is set -- then @c M is diagonal);
 *  - @b Viscous:    @c nu*K with @c K the vector Laplacian (VectorDiffusion);
 *  - @b Divergence: @c B = (div u, q) mapping velocity to pressure;
 *    @c B^T (the pressure gradient block) is its MultTranspose.
 *
 * All quadrature comes from the RuleBook, which must outlive this object (the
 * integrators hold non-owning rule pointers). This class only assembles blocks;
 * composing them into @c [A B^T; B 0] with BC elimination is the solver's job
 * (sub-sprint 1.5).
 *
 * @pre collocated_mass requires the GLL nodal H1 basis
 *      (mfem::BasisType::GaussLobatto, MFEM's H1 default): GLL points against a
 *      non-collocated basis silently give a full, under-integrated mass matrix,
 *      so the basis is verified at construction.
 */
class StokesOperator
{
public:
   /**
    * @brief Assemble the blocks.
    * @param spaces Mixed velocity/pressure spaces (borrowed, must outlive this).
    * @param rules  Quadrature source (borrowed, must outlive this).
    * @param opts   Viscosity and mass-rule options.
    */
   StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                  const StokesOperatorOptions& opts = StokesOperatorOptions());

   /// @return The velocity mass operator @c M on true dofs.
   mfem::Operator& Mass() { return *M_.Ptr(); }

   /// @return The viscous operator @c nu*K on true dofs.
   mfem::Operator& Viscous() { return *K_.Ptr(); }

   /**
    * @brief The divergence block @c B (velocity true dofs -> pressure true
    *        dofs), with the convention @c (B u)_i = (div u, q_i).
    * @return Rectangular operator; MultTranspose applies @c B^T.
    */
   mfem::Operator& Divergence() { return *B_.Ptr(); }

   /**
    * @brief Assembled diagonal of the mass operator (the Jacobi diagonal).
    * @return Velocity true-dof vector; with collocated_mass on, @c M is exactly
    *         this diagonal.
    */
   const mfem::Vector& MassDiagonal() const { return mass_diag_; }

   /// @return The options this operator was assembled with.
   const StokesOperatorOptions& Options() const { return opts_; }

private:
   MixedSpaces& spaces_;             ///< Mixed spaces (borrowed).
   StokesOperatorOptions opts_;      ///< Assembly options.
   mfem::ConstantCoefficient nu_;    ///< Viscosity coefficient (owned).

   mfem::ParBilinearForm mass_form_;      ///< Velocity vector mass form.
   mfem::ParBilinearForm viscous_form_;   ///< Velocity vector diffusion form.
   mfem::ParMixedBilinearForm div_form_;  ///< Mixed divergence form.

   mfem::OperatorPtr M_; ///< True-dof mass operator.
   mfem::OperatorPtr K_; ///< True-dof viscous operator.
   mfem::OperatorPtr B_; ///< True-dof divergence operator.
   mfem::Vector mass_diag_; ///< Assembled mass diagonal (true dofs).
};

} // namespace incns

#endif // INCNS_OPERATORS_STOKES_OPERATOR_HPP
