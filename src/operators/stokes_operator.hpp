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
   /// Coefficient of the mass term in the momentum block:
   /// @c A = mass_coeff * M + nu * K. Zero (the default) gives the steady
   /// block; the unsteady stepper passes the BDF factor @c beta0/dt.
   double mass_coeff = 0.0;
   /// Grad-div augmentation scale @c c_gd: adds gamma (div u, div v) to the
   /// momentum block with the ORDER-H, spatially varying coefficient
   /// gamma(x) = c_gd * h_K per element. Zero (the default) disables it.
   /// gamma never enters the Schur complement or the preconditioner -- the
   /// Schur block is nu * M_p^{-1} with gamma on or off.
   double grad_div = 0.0;
   /// Build the low-order-refined (LOR) source form for the BoomerAMG velocity
   /// preconditioner (see MomentumLORForm()). Off by default -- the matrix-free
   /// Jacobi path needs it not; the solver sets it when AMG is chosen. The
   /// matrix-free Momentum() operator (what the Krylov apply uses) is unchanged.
   bool lor_momentum = false;
   /// Freeze the LOR source at the REFERENCE operator c0_ref*M + nu*K (c0_ref =
   /// the construction-time mass_coeff), so its BoomerAMG hierarchy is built once
   /// and NEVER rebuilt on a Delta-t refresh. The mass term is kept: it keeps the
   /// frozen operator SPD (nu*K alone is singular on a fully periodic domain) and
   /// bounds the preconditioned condition number ~ max(c0/c0_ref, c0_ref/c0), so
   /// it stays tight while the adaptive dt hovers near its reference. Off =
   /// rebuild the LOR hierarchy on each mass-factor change. Only meaningful with
   /// @ref lor_momentum.
   bool lor_frozen = false;
};

/**
 * @brief Assembles and owns the operator blocks of the Stokes saddle-point
 *        system on the mixed velocity/pressure spaces.
 *
 * Blocks (all partial assembly, AssemblyLevel::PARTIAL, acting on true dofs):
 *  - @b Mass:       velocity vector mass @c M (GL rule, or collocated GLL when
 *    StokesOperatorOptions::collocated_mass is set -- then @c M is diagonal);
 *  - @b Momentum:   @c A = mass_coeff*M + nu*K (VectorMass + VectorDiffusion;
 *    the steady block when @c mass_coeff = 0);
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
    * @param spaces    Mixed velocity/pressure spaces (borrowed, must outlive this).
    * @param rules     Quadrature source (borrowed, must outlive this).
    * @param opts      Viscosity and mass-rule options.
    * @param ess_tdofs Optional essential (Dirichlet) velocity true dofs. When
    *        given, Momentum() is a ConstrainedOperator (identity on the
    *        eliminated rows/columns) and Divergence() has the corresponding
    *        trial columns eliminated -- both expose EliminateRHS for the
    *        Dirichlet contribution to the right-hand side. When null (the
    *        default), the operators are unconstrained.
    */
   StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                  const StokesOperatorOptions& opts = StokesOperatorOptions(),
                  const mfem::Array<int>* ess_tdofs = nullptr);

   /**
    * @brief Reassemble the momentum block for a new BDF mass factor @p c0.
    *
    * Rebuilds ONLY the Delta-t-dependent momentum block A = c0*M + nu*K
    * (+ grad-div) and its diagonal (and, for the non-frozen AMG path, the LOR
    * source) -- the Delta-t-independent M, nu*K, B are left untouched. Cheap:
    * the mesh geometric factors are cached, so only the coefficient*geometry PA
    * data is recomputed. Note Momentum()'s underlying pointer changes, so any
    * BlockOperator referencing it must re-point (StokesSolver::Refresh does).
    *
    * @param c0 New leading BDF weight beta0/dt (>= 0).
    */
   void SetMassCoeff(double c0);

   /// @return The velocity mass operator @c M on true dofs (unconstrained;
   ///         used for the BDF history right-hand side in the unsteady solve).
   mfem::Operator& Mass() { return *M_.Ptr(); }

   /**
    * @brief The momentum block @c A = mass_coeff * M + nu * K on true dofs
    *        (equal to @c nu*K when @c mass_coeff is zero).
    * @return Constrained when essential dofs were given at construction.
    */
   mfem::Operator& Momentum() { return *K_.Ptr(); }

   /**
    * @brief The pure viscous operator @c nu*K on true dofs, UNCONSTRAINED
    *        (no Dirichlet elimination) -- for explicit right-hand-side terms
    *        of time steppers (e.g. the trapezoidal starter's K u^0 term).
    * @return Unconstrained operator regardless of @c ess_tdofs.
    */
   mfem::Operator& ViscousUnconstrained() { return *Kunc_.Ptr(); }

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

   /**
    * @brief Assembled diagonal of the (unconstrained) momentum block.
    * @return Velocity true-dof vector, used for the Jacobi velocity block of
    *         the preconditioner (essential dofs are handled by the smoother).
    */
   const mfem::Vector& MomentumDiagonal() const { return momentum_diag_; }

   /**
    * @brief High-order source form for the LOR BoomerAMG velocity block
    *        (mass + diffusion; grad-div omitted -- see the cpp for why).
    *
    * The caller (StokesSolver) hands this to mfem::ParLORDiscretization /
    * LORSolver, which rediscretizes it at low order on the GLL-node LOR mesh and
    * builds AMG on that spectrally-equivalent operator.
    *
    * @return The momentum bilinear form (not assembled here).
    * @pre StokesOperatorOptions::lor_momentum was set at construction.
    */
   mfem::ParBilinearForm& MomentumLORForm() const;

   /// @return The options this operator was assembled with.
   const StokesOperatorOptions& Options() const { return opts_; }

private:
   /// Add the momentum integrators (diffusion, optional c0*mass, optional
   /// grad-div) to @p form, using the current mass_coeff_.
   void AddMomentumIntegrators(mfem::ParBilinearForm& form, bool include_mass,
                               bool include_grad_div);
   /// (Re)build the momentum block A = c0*M + nu*K (+ grad-div), its diagonal,
   /// and the non-frozen LOR source, from the current mass_coeff_.
   void BuildMomentum();

   MixedSpaces& spaces_;             ///< Mixed spaces (borrowed).
   const RuleBook& rules_;           ///< Quadrature source (borrowed).
   StokesOperatorOptions opts_;      ///< Assembly options (mass_coeff mutable).
   mfem::ConstantCoefficient nu_;    ///< Viscosity coefficient (owned).
   mfem::ConstantCoefficient mass_coeff_; ///< Momentum-block mass coefficient.
   /// Grad-div coefficient gamma(x) = c_gd * h_K (owned; null when disabled).
   std::unique_ptr<mfem::Coefficient> gamma_;
   const mfem::IntegrationRule* mass_rule_ = nullptr; ///< Mass quadrature rule.
   int dim_ = 0;                     ///< Spatial dimension.
   int ku_ = 0;                      ///< Velocity order.
   mfem::Geometry::Type geom_ = mfem::Geometry::INVALID; ///< Element geometry.
   /// Essential velocity true dofs. Must outlive the constrained operators:
   /// MFEM's (Rectangular)ConstrainedOperator MakeRef's the list, it does not
   /// copy it, so this is a member rather than a constructor local.
   mfem::Array<int> ess_tdofs_;
   /// Essential pressure true dofs -- always empty (the pressure level is never
   /// pinned); a member for the same MakeRef lifetime reason.
   mfem::Array<int> ess_p_tdofs_;

   mfem::ParBilinearForm mass_form_;      ///< Velocity vector mass form.
   /// Momentum block form (mass + diffusion [+ grad-div]); rebuilt by
   /// BuildMomentum on each mass-factor change, hence held by pointer.
   std::unique_ptr<mfem::ParBilinearForm> momentum_form_;
   /// HO source form for the LOR AMG velocity block (built only for AMG).
   /// Frozen: nu*K only, built once. Non-frozen: c0*M + nu*K, rebuilt with the
   /// momentum block. Grad-div is always omitted (see the cpp).
   std::unique_ptr<mfem::ParBilinearForm> lor_form_;
   mfem::ParBilinearForm viscous_form_;   ///< Pure viscous form (unconstrained).
   mfem::ParMixedBilinearForm div_form_;  ///< Mixed divergence form.

   mfem::OperatorPtr M_; ///< True-dof mass operator.
   mfem::OperatorPtr K_; ///< True-dof momentum operator.
   mfem::OperatorPtr Kunc_; ///< True-dof unconstrained viscous operator.
   mfem::OperatorPtr B_; ///< True-dof divergence operator.
   mfem::Vector mass_diag_;     ///< Assembled mass diagonal (true dofs).
   mfem::Vector momentum_diag_; ///< Assembled momentum diagonal (true dofs).
};

} // namespace incns

#endif // INCNS_OPERATORS_STOKES_OPERATOR_HPP
