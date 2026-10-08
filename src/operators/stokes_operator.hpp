/**
 * @file stokes_operator.hpp
 * @brief The Stokes saddle-point blocks (mass, viscous, divergence), partially
 *        assembled.
 */
#pragma once

#include "operators/grad_div_scale.hpp"
#include "operators/rotational_convection.hpp"
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
   /// momentum block. Zero (the default) disables it. gamma never enters the
   /// Schur complement or the preconditioner (see grad_div_scale for the
   /// caveat when scaling by nu).
   double grad_div = 0.0;
   /// How @c grad_div is scaled: OrderH -> gamma = c_gd * h_K per element
   /// (default); OrderNu -> gamma = c_gd * nu (constant). See GradDivScale.
   GradDivScale grad_div_scale = GradDivScale::OrderH;
   /// Build the low-order-refined (LOR) source form for the BoomerAMG velocity
   /// preconditioner (see MomentumLORForm()). Off by default -- the matrix-free
   /// Jacobi path needs it not; the solver sets it when AMG is chosen. The
   /// matrix-free Momentum() operator (what the Krylov apply uses) is unchanged.
   bool lor_momentum = false;
   /// Freeze the LOR source at the REFERENCE operator c0_ref*M + nu*K (+ grad-div,
   /// which is Delta-t independent) (c0_ref =
   /// the construction-time mass_coeff), so its BoomerAMG hierarchy is built once
   /// and NEVER rebuilt on a Delta-t refresh. The mass term is kept: it keeps the
   /// frozen operator SPD (nu*K alone is singular on a fully periodic domain) and
   /// bounds the preconditioned condition number ~ max(c0/c0_ref, c0_ref/c0), so
   /// it stays tight while the adaptive dt hovers near its reference. Off =
   /// rebuild the LOR hierarchy on each mass-factor change. Only meaningful with
   /// @ref lor_momentum.
   bool lor_frozen = false;
   /// Lagged velocity w* of the semi-implicit rotational term
   /// alpha ((curl w*) x u, v) (ConvectiveForm::Rotational). Null (the
   /// default) = no rotation term. Borrowed: the caller updates it in place
   /// and calls Rotation()->UpdateVorticity(); must outlive this object.
   const mfem::ParGridFunction* lagged_velocity = nullptr;
   /// The rotation term's alpha: 1 for BDF, 1/2 for the trapezoidal starter.
   double rotation_alpha = 1.0;
   /// Include the rotation term in the LOR source (needs lagged_velocity and
   /// a non-frozen lor_momentum): w is copied to the LOR space by true dofs
   /// and the LOR operator re-assembled by AssembleLorMomentum() each step.
   bool rotation_in_lor = false;
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
    * @brief The momentum block the outer solver applies: Momentum() plus, with
    *        a lagged velocity, the rotation term N (constrained with zero
    *        diagonal on essential dofs, so the sum keeps identity rows there).
    * @return Momentum() itself when there is no rotation term.
    */
   mfem::Operator& FullMomentum()
   {
      return full_momentum_ ? *full_momentum_ : *K_.Ptr();
   }

   /// @return The rotation integrator (null without a lagged velocity).
   VectorRotationalConvectionIntegrator* Rotation() { return rot_; }

   /**
    * @brief The rotation term N on true dofs, UNCONSTRAINED (explicit
    *        right-hand-side terms, e.g. the trapezoidal starter's N u^0).
    * @pre A lagged velocity was given at construction.
    * @return The operator.
    */
   mfem::Operator& RotationUnconstrained();

   /**
    * @brief The rotation term with essential rows/columns zeroed -- for the
    *        Dirichlet right-hand-side elimination (b -= N u_D).
    * @pre A lagged velocity was given at construction.
    * @return The constrained operator.
    */
   mfem::ConstrainedOperator& RotationConstrained();

   /// @return Whether the LOR source carries the rotation term.
   bool RotationInLor() const { return lor_disc_ != nullptr; }

   /**
    * @brief Copy the lagged velocity onto the LOR space (by true dofs -- H1
    *        LOR shares them) and assemble the LOR momentum operator
    *        c0 M + nu K + N on it (MFEM legacy LOR assembly, CPU).
    * @pre StokesOperatorOptions::rotation_in_lor.
    * @return The assembled LOR operator, owned by LorDiscretization(); the
    *         next call replaces it.
    */
   mfem::HypreParMatrix& AssembleLorMomentum();

   /// @return The LOR discretization the rotation-in-LOR operator lives on.
   mfem::ParLORDiscretization& LorDiscretization();

   /**
    * @brief The essential velocity true dofs this operator was built with
    *        (its own copy). Preconditioners that keep the list -- MFEM's
    *        Jacobi/Chebyshev smoothers hold a pointer and read it on the
    *        device -- borrow THIS one, never the BoundaryConditions' list:
    *        that one is rebuilt in place on AMR, and a device read of it
    *        leaves its host mirror protected on MFEM's debug device, where
    *        the in-place rebuild (MarkerToList's size-0 HostWrite) faults.
    * @return The list (lives as long as this operator).
    */
   const mfem::Array<int>& EssentialTrueDofs() const { return ess_tdofs_; }

   /**
    * @brief The pure viscous operator @c nu*K on true dofs, UNCONSTRAINED
    *        (no Dirichlet elimination) -- for explicit right-hand-side terms
    *        of time steppers (e.g. the trapezoidal starter's K u^0 term).
    * @return Unconstrained operator regardless of @c ess_tdofs.
    */
   mfem::Operator& ViscousUnconstrained() { return *Kunc_.Ptr(); }

   /**
    * @brief The momentum block A = mass_coeff*M + nu*K (+ grad-div) on true
    *        dofs, UNCONSTRAINED -- its Dirichlet rows are the reaction forces
    *        (body-force evaluation, post/body_force).
    * @return Unconstrained operator (rebuilt with the momentum block).
    */
   mfem::Operator& MomentumUnconstrained() { return *Aunc_.Ptr(); }

   /**
    * @brief The divergence block B (velocity -> pressure true dofs) with NO
    *        Dirichlet columns eliminated; MultTranspose applies B^T.
    * @return Unconstrained rectangular operator.
    */
   mfem::Operator& DivergenceUnconstrained() { return *Bunc_.Ptr(); }

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
    *        (mass + diffusion + grad-div, the latter with the PARENT
    *        element's gamma = c_gd h_K evaluated on the LOR mesh).
    *
    * Grad-div was omitted until 2026-10-07 on the grounds that gamma ~ h is
    * negligible; it is not when nu << h (gamma/nu ~ 60 at h = 1/16, nu =
    * 1e-3), and LOR-AMG without it needed 10-30x more outer iterations
    * (bench/bench_velocity_pc).
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
   /**
    * @brief Add the momentum integrators (diffusion, optional c0*mass,
    *        optional grad-div) to @p form, using the current mass_coeff_.
    * @param form             Form to add to (it takes ownership).
    * @param include_mass     Add c0*M (when mass_coeff > 0).
    * @param include_grad_div Add the grad-div term (when grad_div > 0).
    * @param gamma            Grad-div coefficient to use instead of gamma_
    *                         (the LOR source passes gamma_lor_); null =
    *                         gamma_.
    */
   void AddMomentumIntegrators(mfem::ParBilinearForm& form, bool include_mass,
                               bool include_grad_div,
                               mfem::Coefficient* gamma = nullptr);
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
   /// The same gamma for the LOR source: evaluated on the order-k_u LOR mesh,
   /// it returns the parent high-order element's value (owned; null unless
   /// grad-div and lor_momentum).
   std::unique_ptr<mfem::Coefficient> gamma_lor_;
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
   /// Frozen: c0_ref*M + nu*K (+ grad-div), built once. Non-frozen: c0*M + nu*K
   /// (+ grad-div), rebuilt with the momentum block.
   std::unique_ptr<mfem::ParBilinearForm> lor_form_;
   mfem::ParBilinearForm viscous_form_;   ///< Pure viscous form (unconstrained).
   mfem::ParMixedBilinearForm div_form_;  ///< Mixed divergence form.

   mfem::OperatorPtr M_; ///< True-dof mass operator.
   mfem::OperatorPtr K_; ///< True-dof momentum operator.
   mfem::OperatorPtr Kunc_; ///< True-dof unconstrained viscous operator.
   mfem::OperatorPtr B_; ///< True-dof divergence operator.
   mfem::OperatorPtr Aunc_; ///< Momentum block, unconstrained (reactions).
   mfem::OperatorPtr Bunc_; ///< Divergence block, unconstrained.
   /// Rotation-term PA form (lagged velocity only); Delta-t independent, so
   /// built once -- the vorticity is updated in place every step.
   std::unique_ptr<mfem::ParBilinearForm> rot_form_;
   /// The rotation integrator (owned by rot_form_; null without rotation).
   VectorRotationalConvectionIntegrator* rot_ = nullptr;
   mfem::OperatorPtr N_; ///< True-dof rotation operator P^T N P (unconstrained).
   /// N with DIAG_ZERO on essential dofs: summed with the DIAG_ONE momentum
   /// block, the essential rows stay identity (two DIAG_ONE operators would
   /// put 2 there and halve inhomogeneous Dirichlet data).
   std::unique_ptr<mfem::ConstrainedOperator> Nc_;
   /// Momentum() + Nc_, rebuilt with the momentum block (rotation only).
   std::unique_ptr<mfem::SumOperator> full_momentum_;
   /// Rotation in LOR: the LOR discretization of the velocity space (owned
   /// here so the LOR-space w below and the assembled operator share it).
   std::unique_ptr<mfem::ParLORDiscretization> lor_disc_;
   /// Rotation in LOR: the lagged velocity on the LOR space.
   std::unique_ptr<mfem::ParGridFunction> w_lor_;
   mfem::Vector mass_diag_;     ///< Assembled mass diagonal (true dofs).
   mfem::Vector momentum_diag_; ///< Assembled momentum diagonal (true dofs).
};

} // namespace incns
