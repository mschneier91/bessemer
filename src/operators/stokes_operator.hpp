/**
 * @file stokes_operator.hpp
 * @brief The Stokes saddle-point blocks (mass, viscous, divergence), partially
 *        assembled.
 */
#pragma once

#include "operators/element_scratch.hpp"
#include "operators/grad_div_integrator.hpp"
#include "operators/lean_vector_diffusion.hpp"
#include "operators/pa_operator.hpp"
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
   /// With hanging nodes, lump the collocated mass by its row sums
   /// (b = P^T d, Nek5000's non-conforming lumping) so it stays diagonal; false
   /// keeps the exact P^T D P (a mass solve wherever its inverse is needed).
   /// The lump is checked at assembly (LumpedMassRatio).
   bool lumped_hanging_mass = true;
   /// Element-vector buffers lent to the partially assembled blocks
   /// (borrowed; operators/element_scratch.hpp); null: the operator makes
   /// its own pair.
   ElementScratch* scratch = nullptr;
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
   /// A second, constant part of the grad-div coefficient, proportional to
   /// the viscosity and added to what @c grad_div gives:
   /// gamma = [c_gd * h_K or c_gd * nu] + grad_div_nu * nu. Grad-div is on
   /// when either is positive.
   double grad_div_nu = 0.0;
   /// The viscosity gamma's nu-proportional part is computed from; <= 0: nu.
   /// A solver whose nu is not the fluid's (the trapezoidal starter is
   /// built with nu / 2, the factor on ITS viscous term) passes the fluid's
   /// here: grad-div is a stabilization, the same in every step.
   double grad_div_nu_ref = -1.0;
   /// Vector diffusion with compact partial-assembly storage: one symmetric
   /// matrix per quadrature point shared by the components (6 numbers in
   /// 3D) instead of MFEM's one per component (27). False = MFEM's own
   /// AssemblePA and apply (LeanVectorDiffusionIntegrator::Storage).
   bool compact_diffusion = true;
   /// Apply grad-div with the viscous term in one kernel (compact storage;
   /// LeanVectorDiffusionIntegrator::FuseGradDiv).
   bool fuse_grad_div = true;
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
 *    StokesOperatorOptions::collocated_mass is set -- then @c M is diagonal,
 *    and on a conforming space applied as its diagonal:
 *    operators/collocated_mass.hpp);
 *  - @b Momentum:   @c A = mass_coeff*M + nu*K (VectorMass + VectorDiffusion;
 *    the steady block when @c mass_coeff = 0; with the collocated mass,
 *    mass_coeff times the diagonal is added at each apply instead);
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
   mfem::Operator& ViscousUnconstrained();

   /// Free the unconstrained viscous operator (its PA data is as large as
   /// the momentum block's diffusion data); the next ViscousUnconstrained()
   /// builds it again.
   void ReleaseViscousUnconstrained();

   /**
    * @brief Change the viscosity. The momentum block is NOT rebuilt here:
    *        follow with SetMassCoeff() (StokesSolver::Refresh does both).
    *
    * Not for a frozen LOR source (built once at the old viscosity) or a
    * grad-div coefficient with a part proportional to nu: the caller
    * checks (StokesSolver::CanChangeViscosity).
    * @param nu New viscosity (> 0).
    */
   void SetViscosity(double nu);

   /// @return Whether the velocity LOR source is frozen at its reference
   ///         mass factor and viscosity.
   bool LorFrozen() const { return opts_.lor_momentum && opts_.lor_frozen; }

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
    * @brief 1 / MassDiagonal() when the mass is applied as its diagonal: one
    *        copy, which the OIFS substeps and the Schur PC's mass inverse
    *        borrow (each kept its own).
    * @return The reciprocal (true dofs), or null when the mass is not
    *         applied as its diagonal.
    */
   const mfem::Vector* MassDiagonalInverse() const
   {
      return DiagonalMass() ? &mass_diag_inv_ : nullptr;
   }


   /**
    * @brief Assembled diagonal of the (unconstrained) momentum block.
    *
    * Formed as mass_coeff * MassDiagonal() + the diagonal of nu*K (+
    * grad-div), the latter assembled once (again after SetViscosity): only
    * the mass factor changes with Delta-t. Equal to the momentum form's own
    * AssembleDiagonal to roundoff.
    * @return Velocity true-dof vector, used for the Jacobi velocity block of
    *         the preconditioner (essential dofs are handled by the smoother).
    */
   const mfem::Vector& MomentumDiagonal() const { return momentum_diag_; }

   /**
    * @brief The grad-div integrator inside the current momentum block.
    *
    * Point-block Jacobi reads the nodal coupling of the velocity components
    * from it (GradDivIntegrator::AddNodalOffDiagonalPA). The momentum form is
    * rebuilt by every SetMassCoeff(), which replaces this object: ask again
    * after it.
    * @return The integrator (partially assembled), or null when grad-div is
    *         off.
    */
   const GradDivIntegrator* GradDivTerm() const { return grad_div_integ_; }

   /// @return The grad-div coefficient gamma(x), or null when grad-div is off
   ///         (for the Schur preconditioner's mass term).
   mfem::Coefficient* GradDivCoefficient() { return gamma_.get(); }

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
    * @param diffusion        If not null, receives the diffusion integrator
    *                         added to @p form (owned by the form).
    * @param mass             If not null, receives the mass integrator
    *                         added to @p form (owned by the form), or null.
    * @return The grad-div integrator added to @p form (owned by the form), or
    *         null when none was.
    */
   template <class Form>
   GradDivIntegrator* AddMomentumIntegrators(Form& form,
         bool include_mass,
         bool include_grad_div,
         mfem::Coefficient* gamma = nullptr,
         mfem::BilinearFormIntegrator** diffusion = nullptr,
         mfem::BilinearFormIntegrator** mass = nullptr);
   /// (Re)build the momentum block A = c0*M + nu*K (+ grad-div), its diagonal,
   /// and the non-frozen LOR source, from the current mass_coeff_.
   void BuildMomentum();
   /// momentum_diag_ = c0 mass_diag_ + rest_diag_ (rest_diag_ assembled).
   void UpdateMomentumDiagonal();
   /// The non-frozen LOR source, from the current mass_coeff_ (nothing
   /// otherwise).
   void BuildLorSource();
   /// Assemble rest_diag_ from the current momentum form's diffusion and
   /// grad-div integrators (no assembly of their data: a proxy form).
   void AssembleRestDiagonal(mfem::BilinearFormIntegrator& diffusion);

   MixedSpaces& spaces_;             ///< Mixed spaces (borrowed).
   const RuleBook& rules_;           ///< Quadrature source (borrowed).
   StokesOperatorOptions opts_;      ///< Assembly options (mass_coeff mutable).
   mfem::ConstantCoefficient nu_;    ///< Viscosity coefficient (owned).
   mfem::ConstantCoefficient mass_coeff_; ///< Momentum-block mass coefficient.
   /// Grad-div coefficient gamma(x) = c_gd * h_K (+ grad_div_nu * nu) (owned;
   /// null when disabled).
   std::unique_ptr<mfem::Coefficient> gamma_;
   /// gamma per element, evaluated once (one number per element, device):
   /// the momentum form is reassembled at every Delta-t change and gamma,
   /// constant on each element, does not depend on Delta-t. Empty when gamma
   /// is a constant or grad-div is off. (It replaced a copy at every
   /// quadrature point: 125 times larger at Q3.)
   mfem::Vector gamma_elem_;
   /// The grad-div integrator of the current momentum form (owned by that
   /// form; null when grad-div is off). Replaced by every BuildMomentum().
   GradDivIntegrator* grad_div_integ_ = nullptr;
   /// The mass integrator of the current momentum form (owned by that form;
   /// null when c0 = 0 or the mass is collocated). A new c0 re-assembles only
   /// its data (SetMassCoeff).
   mfem::BilinearFormIntegrator* mass_integ_ = nullptr;
   /// The viscosity changed since BuildMomentum: the next SetMassCoeff
   /// rebuilds the whole form.
   bool momentum_stale_ = false;
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

   /// Its own element-vector buffers when opts_.scratch is null. Declared
   /// before the blocks that borrow them: destroyed after.
   std::unique_ptr<ElementScratch> own_scratch_;
   ElementScratch* scratch_ = nullptr; ///< The buffers the blocks borrow.
   /// Velocity vector mass (null when the mass is applied as its diagonal).
   std::unique_ptr<PaOperator> mass_form_;
   /// The collocated mass applied as its diagonal: exactly diagonal on true
   /// dofs of a conforming space; with hanging nodes (P^T D P couples them to
   /// the nodes they are constrained by) its row sums, b = P^T d, unless
   /// lumped_hanging_mass is off (then the form stays).
   bool DiagonalMass() const
   {
      return opts_.collocated_mass &&
             (spaces_.Velocity().Conforming() || opts_.lumped_hanging_mass);
   }
   /// The diffusion storage (opts_.compact_diffusion).
   LeanVectorDiffusionIntegrator::Storage DiffusionStorage() const;
   /**
    * @brief P^T A P on true dofs, as MFEM's FormSystemMatrix builds it but
    *        without the ConstrainedOperator wrapper (and its two work
    *        vectors): the caller wraps it when it needs constraints.
    *        Identity prolongations (one rank) give @p A itself.
    * @param Pin  Prolongation of the input space (null: identity).
    * @param A    The L-vector operator (a form, or its extension).
    * @param Pout Prolongation of the output space (null: identity).
    * @param out  Receives the operator (owned unless it is @p A itself).
    */
   static void TrueDofOperator(const mfem::Operator* Pin, mfem::Operator& A,
                               const mfem::Operator* Pout,
                               mfem::OperatorPtr& out);
   /// Momentum block form (mass + diffusion [+ grad-div]); rebuilt by
   /// BuildMomentum on each mass-factor change, hence held by pointer.
   std::unique_ptr<PaOperator> momentum_form_;
   /// HO source form for the LOR AMG velocity block (built only for AMG).
   /// Frozen: c0_ref*M + nu*K (+ grad-div), built once. Non-frozen: c0*M + nu*K
   /// (+ grad-div), rebuilt with the momentum block.
   std::unique_ptr<mfem::ParBilinearForm> lor_form_;
   /// Pure viscous form (unconstrained); null until ViscousUnconstrained()
   /// builds it, and again after ReleaseViscousUnconstrained().
   std::unique_ptr<PaOperator> viscous_form_;
   std::unique_ptr<PaMixedOperator> div_form_; ///< Mixed divergence block.

   mfem::OperatorPtr M_; ///< True-dof mass operator.
   /// True-dof momentum operator: a ConstrainedOperator over Aunc_ (the
   /// two share one P^T A P and its work vectors).
   mfem::OperatorPtr K_;
   mfem::OperatorPtr Kunc_; ///< True-dof unconstrained viscous operator.
   /// True-dof divergence operator: a RectangularConstrainedOperator over
   /// Bunc_ (one shared P^T B P).
   mfem::OperatorPtr B_;
   /// Collocated mass only: nu K (+ grad-div) on true dofs, under Aunc_'s
   /// diagonal shift.
   mfem::OperatorPtr Arest_;
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
   mfem::Vector mass_diag_inv_; ///< 1 / mass_diag_ (diagonal mass only).
   /// W det J at the mass rule, (nq, ne), for the mass integrators of a
   /// non-diagonal mass (LeanVectorMassIntegrator); empty otherwise.
   mfem::Vector mass_wdet_;
   mfem::Vector momentum_diag_; ///< Assembled momentum diagonal (true dofs).
   /// Diagonal of the Delta-t independent rest of the momentum block, nu*K
   /// (+ grad-div), true dofs: momentum_diag_ = mass_coeff * mass_diag_ +
   /// rest_diag_. Its direct-sum kernels cost ~0.4 s at 64^3 Q3 on an H100,
   /// paid at every Delta-t change before it was kept.
   mfem::Vector rest_diag_;
   bool rest_diag_valid_ =
      false; ///< False until assembled and after SetViscosity.
};

} // namespace incns
