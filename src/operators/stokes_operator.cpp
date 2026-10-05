#include "operators/stokes_operator.hpp"

#include "operators/convection.hpp" // DealiasedOrder
#include "operators/grad_div_integrator.hpp"

#include "mesh/mesh_size_coefficient.hpp"
#include "mesh/periodic_box.hpp" // AssertTensorProductGeometry
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesOperator::StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                               const StokesOperatorOptions& opts,
                               const Array<int>* ess_tdofs)
   : spaces_(spaces), rules_(rules), opts_(opts), nu_(opts.nu),
     mass_coeff_(opts.mass_coeff),
     mass_form_(&spaces.Velocity()),
     viscous_form_(&spaces.Velocity()),
     div_form_(&spaces.Velocity(), &spaces.Pressure())
{
   INCNS_PROFILE("stokes_operator::assemble");

   MFEM_VERIFY(opts_.nu > 0.0, "stokes_operator: viscosity must be positive");
   MFEM_VERIFY(opts_.mass_coeff >= 0.0,
               "stokes_operator: mass_coeff must be non-negative");
   MFEM_VERIFY(opts_.grad_div >= 0.0,
               "stokes_operator: grad_div scale must be non-negative");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   AssertTensorProductGeometry(mesh);

   dim_ = spaces_.Dim();
   geom_ = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   ku_ = spaces_.OrderU();

   // Essential velocity dofs eliminated from the viscous block and the
   // divergence trial columns; empty when no Dirichlet BCs are given. Stored as
   // a member: the constrained operators MakeRef this list.
   if (ess_tdofs) { ess_tdofs_ = *ess_tdofs; }

   // --- velocity mass: GL 2k default, or the collocated GLL (diagonal) rule --
   if (opts_.collocated_mass)
   {
      // Diagonality requires the GLL *nodal basis*: GLL points against a
      // non-collocated basis silently give a full, under-integrated mass.
      const auto* h1 =
         dynamic_cast<const H1_FECollection*>(spaces_.Velocity().FEColl());
      MFEM_VERIFY(h1 && h1->GetBasisType() == BasisType::GaussLobatto,
                  "stokes_operator: collocated_mass requires the GLL nodal "
                  "H1 basis (BasisType::GaussLobatto)");
      mass_rule_ = &rules.CollocatedMass(geom_, ku_);
   }
   else
   {
      mass_rule_ = &rules.Get(geom_, 2 * ku_);
   }

   // grad-div coefficient (shared by every momentum form). OrderH: c_gd * h_K
   // per element (spatially varying). OrderNu: c_gd * nu (constant) -- note
   // this is NOT negligible in the Schur block; gamma still does not enter it
   // automatically (see the header), set cc.nu_pc to compensate if wanted.
   if (opts_.grad_div > 0.0)
   {
      if (opts_.grad_div_scale == GradDivScale::OrderNu)
      {
         gamma_ = std::make_unique<ConstantCoefficient>(opts_.grad_div * opts_.nu);
      }
      else
      {
         gamma_ = std::make_unique<MeshSizeCoefficient>(mesh, opts_.grad_div);
      }
   }

   // ==== Delta-t INDEPENDENT blocks: assembled ONCE (a refresh never touches
   // these -- M, nu*K, B are pure mesh/space, only their COMBINATION scales with
   // the BDF factor via the momentum block below) ============================

   // --- velocity mass M ------------------------------------------------------
   {
      INCNS_PROFILE("mass");
      auto* mi = new VectorMassIntegrator;
      mi->SetIntRule(mass_rule_);
      mass_form_.AddDomainIntegrator(mi);
      mass_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mass_form_.Assemble();
      Array<int> empty;
      mass_form_.FormSystemMatrix(empty, M_);
      mass_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
      mass_form_.AssembleDiagonal(mass_diag_);
   }

   // --- pure viscous nu*K, UNCONSTRAINED: explicit RHS terms in time steppers -
   {
      INCNS_PROFILE("viscous_unconstrained");
      auto* kui = new VectorDiffusionIntegrator(nu_);
      kui->SetIntRule(&rules.Get(geom_, 2 * ku_ + dim_ - 1));
      viscous_form_.AddDomainIntegrator(kui);
      viscous_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      viscous_form_.Assemble();
      Array<int> no_ess;
      viscous_form_.FormSystemMatrix(no_ess, Kunc_);
   }

   // --- divergence block B = (div u, q): default exactness order
   // 2*(trial order) + 2 = 2*k_u + 2 -------------------------------------------
   {
      INCNS_PROFILE("divergence");
      auto* di = new VectorDivergenceIntegrator;
      di->SetIntRule(&rules.Get(geom_, 2 * ku_ + 2));
      div_form_.AddDomainIntegrator(di);
      div_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      div_form_.Assemble();
      div_form_.FormRectangularSystemMatrix(ess_tdofs_, ess_p_tdofs_, B_);
   }

   // --- FROZEN LOR source, built ONCE at the reference mass factor c0_ref*M +
   // nu*K (c0_ref = the construction-time mass_coeff) so its BoomerAMG hierarchy
   // is never rebuilt on a Delta-t refresh. The mass term is KEPT so the frozen
   // operator is SPD even on a fully periodic domain (nu*K alone is singular
   // there -- the constant velocity mode); it also bounds the preconditioned
   // condition number ~ max(c0/c0_ref, c0_ref/c0), i.e. tight while the adaptive
   // dt stays near its reference (see the header note).
   if (opts_.lor_momentum && opts_.lor_frozen)
   {
      lor_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
      AddMomentumIntegrators(*lor_form_, /*include_mass=*/true,
                             /*include_grad_div=*/false);
   }

   // --- rotation in the LOR source: an LOR discretization owned HERE, so the
   // LOR-space copy of w* lives on the very mesh MFEM's legacy LOR assembly
   // hands the integrator (it reads curl w through that mesh's elements).
   if (opts_.rotation_in_lor)
   {
      MFEM_VERIFY(opts_.lagged_velocity && opts_.lor_momentum &&
                  !opts_.lor_frozen, "stokes_operator: rotation_in_lor needs "
                  "a lagged velocity and a non-frozen LOR source");
      lor_disc_ = std::make_unique<ParLORDiscretization>(spaces_.Velocity());
      // The true-dof copy below relies on H1 LOR's identity dof permutation.
      const Array<int>& perm = lor_disc_->GetDofPermutation();
      for (int i = 0; i < perm.Size(); ++i)
      {
         MFEM_VERIFY(perm[i] == i, "stokes_operator: LOR dof permutation is not "
                     "the identity -- the true-dof copy of w would be wrong");
      }
      w_lor_ = std::make_unique<ParGridFunction>(&lor_disc_->GetParFESpace());
      *w_lor_ = 0.0;
   }

   // --- semi-implicit rotational term N = alpha ((curl w*) x u, v): Delta-t
   // independent (the vorticity is updated in place each step), so built once.
   // Same dealiased 3k rule as the convective form -- over-integration and the
   // choice of form are orthogonal. Kept OUT of the momentum form so that
   // Momentum() stays the symmetric block the Chebyshev/LOR-AMG velocity PCs
   // need; the outer solver applies FullMomentum() = Momentum() + N.
   if (opts_.lagged_velocity)
   {
      INCNS_PROFILE("rotation");
      rot_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
      rot_ = new VectorRotationalConvectionIntegrator(*opts_.lagged_velocity,
         opts_.rotation_alpha);
      rot_->SetIntRule(&rules_.Get(geom_, Convection::DealiasedOrder(ku_)));
      rot_form_->AddDomainIntegrator(rot_);
      rot_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      rot_form_->Assemble();
      Array<int> no_ess;
      rot_form_->FormSystemMatrix(no_ess, N_);
      Nc_ = std::make_unique<ConstrainedOperator>(N_.Ptr(), ess_tdofs_, false,
            Operator::DIAG_ZERO);
   }

   // ==== Delta-t DEPENDENT momentum block A = c0*M + nu*K (+ grad-div) and, for
   // the non-frozen AMG path, its LOR source -- (re)built by SetMassCoeff ======
   BuildMomentum();
}

void StokesOperator::AddMomentumIntegrators(ParBilinearForm& form,
      bool include_mass, bool include_grad_div)
{
   // Diffusion at the default exactness order 2k + dim - 1 (covers the metric
   // factors on deformed elements). The mass term reuses the mass rule (incl.
   // the collocated-GLL option). Fresh integrator objects -- each form owns
   // its own; the coefficients and rules are shared members.
   auto* ki = new VectorDiffusionIntegrator(nu_);
   ki->SetIntRule(&rules_.Get(geom_, 2 * ku_ + dim_ - 1));
   form.AddDomainIntegrator(ki);
   if (include_mass && opts_.mass_coeff > 0.0)
   {
      auto* mmi = new VectorMassIntegrator(mass_coeff_);
      mmi->SetIntRule(mass_rule_);
      form.AddDomainIntegrator(mmi);
   }
   if (include_grad_div && opts_.grad_div > 0.0)
   {
      // gamma (div u, div v) via the in-repo sum-factorized
      // GradDivIntegrator -- replaces ElasticityIntegrator(lambda, mu=0),
      // elmat-identical (pinned by grad_div_integrator_test) but with fused O(p^4)
      // tensor PA kernels instead of elasticity's dense non-tensor path.
      // Stiffness-type integrand -> the 2k + dim - 1 default rule. gamma never
      // enters the Schur block (pressure_schur stays nu*M_p^{-1}).
      auto* gdi = new GradDivIntegrator(*gamma_);
      gdi->SetIntRule(&rules_.Get(geom_, 2 * ku_ + dim_ - 1));
      form.AddDomainIntegrator(gdi);
   }
}

void StokesOperator::BuildMomentum()
{
   INCNS_PROFILE("stokes_operator::momentum");

   // Fused matrix-free momentum block A = c0*M + nu*K (+ grad-div). Rebuilding
   // recomputes only the coefficient*geometry PA data (mesh geometric factors
   // are cached), so it is cheap -- much cheaper than a full solver rebuild --
   // and it hands us the new diagonal for free.
   momentum_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
   AddMomentumIntegrators(*momentum_form_, /*include_mass=*/true,
                          /*include_grad_div=*/true);
   momentum_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   momentum_form_->Assemble();
   momentum_form_->FormSystemMatrix(ess_tdofs_, K_);
   momentum_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
   momentum_form_->AssembleDiagonal(momentum_diag_);

   // Non-frozen LOR source (c0*M + nu*K) tracks the current mass factor, so the
   // BoomerAMG hierarchy is rebuilt to match on each refresh (see lor_frozen for
   // the alternative that reuses a single nu*K hierarchy).
   if (opts_.lor_momentum && !opts_.lor_frozen)
   {
      lor_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
      AddMomentumIntegrators(*lor_form_, /*include_mass=*/true,
                             /*include_grad_div=*/false);
      if (w_lor_)
      {
         // Legacy element matrices on the LOR elements, reading the LOR-space
         // w (MFEM sets the collocated rule during LOR assembly).
         lor_form_->AddDomainIntegrator(new VectorRotationalConvectionIntegrator(
                                           *w_lor_, opts_.rotation_alpha));
      }
   }

   // The outer operator A + N re-points at the rebuilt momentum block.
   if (Nc_)
   {
      full_momentum_ = std::make_unique<SumOperator>(K_.Ptr(), 1.0, Nc_.get(),
                       1.0, false, false);
   }
}

void StokesOperator::SetMassCoeff(double c0)
{
   MFEM_VERIFY(c0 >= 0.0, "stokes_operator: mass_coeff must be non-negative");
   mass_coeff_.constant = c0;
   opts_.mass_coeff = c0;
   BuildMomentum();
}

Operator& StokesOperator::RotationUnconstrained()
{
   MFEM_VERIFY(N_.Ptr(), "stokes_operator: no rotation term (no lagged "
               "velocity at construction)");
   return *N_.Ptr();
}

ConstrainedOperator& StokesOperator::RotationConstrained()
{
   MFEM_VERIFY(Nc_, "stokes_operator: no rotation term (no lagged velocity "
               "at construction)");
   return *Nc_;
}

HypreParMatrix& StokesOperator::AssembleLorMomentum()
{
   MFEM_VERIFY(lor_disc_, "stokes_operator: AssembleLorMomentum() requires "
               "rotation_in_lor");
   INCNS_PROFILE("stokes_operator::lor_rotation");
   Vector w_true(spaces_.Velocity().GetTrueVSize());
   opts_.lagged_velocity->GetTrueDofs(w_true);
   w_lor_->SetFromTrueDofs(w_true);
   lor_disc_->AssembleSystem(*lor_form_, ess_tdofs_);
   return lor_disc_->GetAssembledMatrix();
}

ParLORDiscretization& StokesOperator::LorDiscretization()
{
   MFEM_VERIFY(lor_disc_, "stokes_operator: no LOR discretization (requires "
               "rotation_in_lor)");
   return *lor_disc_;
}

ParBilinearForm& StokesOperator::MomentumLORForm() const
{
   MFEM_VERIFY(lor_form_, "stokes_operator: MomentumLORForm() requires "
               "lor_momentum = true at construction");
   return *lor_form_;
}

} // namespace incns
