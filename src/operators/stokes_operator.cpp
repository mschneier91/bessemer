#include "operators/stokes_operator.hpp"

#include "operators/collocated_mass.hpp"
#include "operators/lean_vector_divergence.hpp"
#include "operators/lean_vector_mass.hpp"

#include "operators/convection.hpp" // DealiasedOrder
#include "operators/grad_div_integrator.hpp"
#include "operators/kernel_specializations.hpp"
#include "operators/lean_vector_diffusion.hpp"

#include "mesh/mesh_size_coefficient.hpp"
#include "mesh/periodic_box.hpp" // AssertTensorProductGeometry
#include "util/profiler.hpp"

#include <vector>

namespace incns
{

using namespace mfem;

namespace
{

/**
 * @brief An integrator whose PA diagonal is the sum of other integrators'
 *        (partially assembled in another form): AssembleDiagonal of a form
 *        holding it reduces their diagonals to true dofs with MFEM's own
 *        reduction, without assembling their data again (the device of
 *        PointBlockJacobi's proxies).
 */
class DiagonalSumProxy : public BilinearFormIntegrator
{
   std::vector<BilinearFormIntegrator*> parts_; ///< Not owned.

public:
   /// @param parts Assembled integrators; must outlive the assembly.
   explicit DiagonalSumProxy(std::vector<BilinearFormIntegrator*> parts)
      : parts_(std::move(parts)) { }
   using BilinearFormIntegrator::AssemblePA;
   /// Nothing to set up: the data lives in the integrators.
   void AssemblePA(const FiniteElementSpace&) override { }
   /// @param diag_e E-vector (accumulated into).
   void AssembleDiagonalPA(Vector& diag_e) override
   {
      for (BilinearFormIntegrator* p : parts_) { p->AssembleDiagonalPA(diag_e); }
   }
};

/**
 * @brief MFEM's RAPOperator (Rt^T A P, the same arithmetic) with its two work
 *        vectors lent by an ElementScratch: one pair for all the true-dof
 *        operators of an integrator, which never run inside one another,
 *        instead of a pair each for the run. On hanging-node meshes, and at
 *        more than one rank, the momentum, mass and divergence blocks held
 *        five velocity and one pressure local-dof vector (1.3 GB at 56.5M
 *        DOFs, Q6, the curved cylinder with hanging nodes); now two.
 */
class SharedRAPOperator : public Operator
{
   const Operator& Rt_;
   const Operator& A_;
   const Operator& P_;
   mutable Vector Px_;
   mutable Vector APx_;
   MemoryClass mem_class_;

public:
   SharedRAPOperator(const Operator& Rt, const Operator& A, const Operator& P,
                     ElementScratch& scratch)
      : Operator(Rt.Width(), P.Width()), Rt_(Rt), A_(A), P_(P)
   {
      MFEM_VERIFY(Rt.Height() == A.Height() && A.Width() == P.Height(),
                  "stokes_operator: incompatible operators for P^T A P");
      mem_class_ = Rt.GetMemoryClass() * P.GetMemoryClass();
      scratch.LendLocal(0, P.Height(), Px_);
      scratch.LendLocal(1, A.Height(), APx_);
   }
   MemoryClass GetMemoryClass() const override { return mem_class_; }
   void Mult(const Vector& x, Vector& y) const override
   { P_.Mult(x, Px_); A_.Mult(Px_, APx_); Rt_.MultTranspose(APx_, y); }
   void AbsMult(const Vector& x, Vector& y) const override
   { P_.AbsMult(x, Px_); A_.AbsMult(Px_, APx_); Rt_.AbsMultTranspose(APx_, y); }
   void AssembleDiagonal(Vector& diag) const override
   {
      A_.AssembleDiagonal(APx_);
      P_.MultTranspose(APx_, diag);
   }
   void MultTranspose(const Vector& x, Vector& y) const override
   { Rt_.Mult(x, APx_); A_.MultTranspose(APx_, Px_); P_.MultTranspose(Px_, y); }
   void AbsMultTranspose(const Vector& x, Vector& y) const override
   {
      Rt_.AbsMult(x, APx_);
      A_.AbsMultTranspose(APx_, Px_);
      P_.AbsMultTranspose(Px_, y);
   }
};

} // namespace

StokesOperator::StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                               const StokesOperatorOptions& opts,
                               const Array<int>* ess_tdofs)
   : spaces_(spaces), rules_(rules), opts_(opts), nu_(opts.nu),
     mass_coeff_(opts.mass_coeff)
{
   INCNS_PROFILE("stokes_operator::assemble");
   AddKernelSpecializations(); // MFEM's kernels at our sizes, before any apply

   MFEM_VERIFY(opts_.nu > 0.0, "stokes_operator: viscosity must be positive");
   MFEM_VERIFY(opts_.mass_coeff >= 0.0,
               "stokes_operator: mass_coeff must be non-negative");
   MFEM_VERIFY(opts_.grad_div >= 0.0 && opts_.grad_div_nu >= 0.0,
               "stokes_operator: the grad_div scales must be non-negative");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   AssertTensorProductGeometry(mesh);
   // The element vectors of every partially assembled block: lent by one pair
   // of buffers, the time integrator's when it shares them.
   if (opts_.scratch) { scratch_ = opts_.scratch; }
   else
   {
      own_scratch_ = std::make_unique<ElementScratch>(spaces_.Velocity());
      scratch_ = own_scratch_.get();
   }

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
   if (opts_.grad_div > 0.0 || opts_.grad_div_nu > 0.0)
   {
      // gamma = h_part * h_K + nu_part: grad_div goes to one or the other by
      // its scale, grad_div_nu * nu is always in the constant part.
      const bool by_h = (opts_.grad_div_scale == GradDivScale::OrderH);
      const double h_part = by_h ? opts_.grad_div : 0.0;
      const double nu_part =
         (opts_.grad_div_nu + (by_h ? 0.0 : opts_.grad_div)) *
         (opts_.grad_div_nu_ref > 0.0 ? opts_.grad_div_nu_ref : opts_.nu);
      if (h_part > 0.0)
      {
         auto size_coeff =
            std::make_unique<MeshSizeCoefficient>(mesh, h_part, 1, nu_part);
         // Its value on each element, once: what the momentum form's
         // grad-div reads at every (re)assembly.
         size_coeff->ElementValues(gamma_elem_);
         gamma_elem_.UseDevice(true);
         gamma_ = std::move(size_coeff);
      }
      else
      {
         gamma_ = std::make_unique<ConstantCoefficient>(nu_part);
      }
      // The LOR source's gamma: MFEM's LOR assembly evaluates the
      // coefficient on the order-k_u LOR mesh (k_u^dim children per element,
      // in order), where gamma must be the PARENT element's value.
      if (opts_.lor_momentum)
      {
         int children = 1;
         for (int d = 0; d < dim_; ++d) { children *= ku_; }
         if (h_part > 0.0)
         {
            gamma_lor_ = std::make_unique<MeshSizeCoefficient>(
                            mesh, h_part, children, nu_part);
         }
         else
         {
            gamma_lor_ = std::make_unique<ConstantCoefficient>(nu_part);
         }
      }
   }

   // ==== Delta-t INDEPENDENT blocks: assembled ONCE (a refresh never touches
   // these -- M, nu*K, B are pure mesh/space, only their COMBINATION scales with
   // the BDF factor via the momentum block below) ============================

   // --- velocity mass M ------------------------------------------------------
   {
      INCNS_PROFILE("mass");
      if (DiagonalMass())
      {
         // Exactly diagonal (GLL nodes and rule, conforming) or lumped at
         // hanging nodes, so the operator is its diagonal, assembled without
         // the mesh's cached Jacobians at the GLL points
         // (operators/collocated_mass.hpp); mass_form_ stays empty.
         AssembleCollocatedMassDiagonal(spaces_.Velocity(), *mass_rule_,
                                        mass_diag_);
         if (!spaces_.Velocity().Conforming())
         {
            // Hanging nodes: the row sums b = P^T d, checked against the
            // conforming weight. b >= ref with one hanging level on elements
            // whose |J| is a low-degree polynomial (the solver enforces
            // amr.nc_limit 1); this rejects whatever is not covered.
            const double ratio =
               LumpedMassRatio(spaces_.Velocity(), *mass_rule_, mass_diag_);
            MFEM_VERIFY(ratio >= kMinLumpedMassRatio, "stokes_operator: the "
                        "collocated mass lumped at hanging nodes is too small "
                        "or negative somewhere (min b / ref = " << ratio
                        << " < " << kMinLumpedMassRatio << "): the mesh has "
                        "hanging-node patches the lumping does not cover -- "
                        "more than one hanging level with very different "
                        "element thicknesses across a hanging face, or "
                        "strongly distorted refined elements. Use amr.nc_limit "
                        "1, improve the mesh, or set "
                        "discretization.hanging_mass: exact");
            if (ratio < kWarnLumpedMassRatio && Mpi::Root())
            {
               mfem::out << "[incns] warning: the collocated mass lumped at "
                         "hanging nodes is below its conforming weight "
                         "somewhere (min b / ref = " << ratio << "): a "
                         "hanging-node patch outside the proved regime" << std::endl;
            }
         }
         // Its reciprocal, once, for the OIFS substeps and the Schur PC.
         mass_diag_inv_.UseDevice(true);
         mass_diag_inv_.SetSize(mass_diag_.Size());
         mass_diag_inv_ = 1.0;
         mass_diag_inv_ /= mass_diag_;
         M_.Reset(new DiagonalOperator(mass_diag_), true);
      }
      else
      {
         // W det J at the mass rule, kept (one number per point) for every
         // mass integrator here, instead of the mesh's cached Jacobians.
         MassWeightDet(spaces_.Velocity(), *mass_rule_, mass_wdet_);
         // The diagonal through MFEM's form (its parallel and hanging-node
         // assembly), which is freed after; the operator from the same
         // integrator in a PaOperator.
         {
            ParBilinearForm diag_form(&spaces_.Velocity());
            auto* mi = new LeanVectorMassIntegrator(mass_wdet_);
            mi->SetIntRule(mass_rule_);
            diag_form.AddDomainIntegrator(mi);
            diag_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            diag_form.Assemble();
            mass_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
            diag_form.AssembleDiagonal(mass_diag_);
         }
         mass_form_ = std::make_unique<PaOperator>(spaces_.Velocity(), *scratch_);
         auto* mi = new LeanVectorMassIntegrator(mass_wdet_);
         mi->SetIntRule(mass_rule_);
         mass_form_->AddDomainIntegrator(mi);
         mass_form_->Assemble();
         // Unconstrained: no ConstrainedOperator wrapper (two vectors saved).
         TrueDofOperator(mass_form_->GetProlongation(), *mass_form_,
                         mass_form_->GetProlongation(), M_);
      }
   }

   // The pure viscous nu*K (unconstrained) is assembled on first use
   // (ViscousUnconstrained): only a trapezoidal starter's right-hand side
   // needs it, and its PA data is as large as the momentum block's.

   // --- divergence block B = (div u, q): default exactness order
   // 2*(trial order) + 2 = 2*k_u + 2 -------------------------------------------
   {
      INCNS_PROFILE("divergence");
      // Its kernels read the mesh's cached Jacobians at this rule (the
      // grad-div form's), not a W adj(J) of its own per point.
      auto* di = new LeanVectorDivergenceIntegrator;
      di->SetIntRule(&rules.Get(geom_, 2 * ku_ + 2));
      div_form_ = std::make_unique<PaMixedOperator>(spaces_.Velocity(),
                  spaces_.Pressure(), *scratch_);
      div_form_->AddDomainIntegrator(di);
      div_form_->Assemble();
      // One P_p^T B P_u for both: unconstrained as it is (reactions), and
      // wrapped with the constraints -- what FormRectangularSystemMatrix
      // builds, without a second RAP and its work vectors.
      TrueDofOperator(div_form_->GetProlongation(), *div_form_,
                      div_form_->GetOutputProlongation(), Bunc_);
      B_.Reset(new RectangularConstrainedOperator(Bunc_.Ptr(), ess_tdofs_,
               ess_p_tdofs_, false), true);
   }

   // --- FROZEN LOR source, built ONCE at the reference mass factor c0_ref*M +
   // nu*K + grad-div (c0_ref = the construction-time mass_coeff; grad-div is
   // Delta-t independent and, with gamma = c h, can dominate nu*K -- leaving it
   // out wrecked LOR-AMG, measured 2026-10-07) so its BoomerAMG hierarchy
   // is never rebuilt on a Delta-t refresh. The mass term is KEPT so the frozen
   // operator is SPD even on a fully periodic domain (nu*K alone is singular
   // there -- the constant velocity mode); it also bounds the preconditioned
   // condition number ~ max(c0/c0_ref, c0_ref/c0), i.e. tight while the adaptive
   // dt stays near its reference (see the header note).
   if (opts_.lor_momentum && opts_.lor_frozen)
   {
      lor_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
      AddMomentumIntegrators(*lor_form_, /*include_mass=*/true,
                             /*include_grad_div=*/true, gamma_lor_.get());
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

template <class Form>
GradDivIntegrator* StokesOperator::AddMomentumIntegrators(
   Form& form, bool include_mass, bool include_grad_div,
   Coefficient* gamma, BilinearFormIntegrator** diffusion,
   BilinearFormIntegrator** mass)
{
   GradDivIntegrator* added = nullptr;
   // Diffusion at the default exactness order 2k + dim - 1 (covers the metric
   // factors on deformed elements). The mass term reuses the mass rule (incl.
   // the collocated-GLL option). Fresh integrator objects -- each form owns
   // its own; the coefficients and rules are shared members.
   // The Lean variant: MFEM's own PA diagonal reserves ~11 GB of GPU stack on
   // its first call (operators/lean_vector_diffusion.hpp).
   auto* ki = new LeanVectorDiffusionIntegrator(nu_, DiffusionStorage());
   ki->SetIntRule(&rules_.Get(geom_, 2 * ku_ + dim_ - 1));
   form.AddDomainIntegrator(ki);
   if (diffusion) { *diffusion = ki; }
   if (include_mass && opts_.mass_coeff > 0.0)
   {
      // The kept W det J when there is one (a non-diagonal mass): no cached
      // Jacobians at the mass rule, here or at a new mass factor.
      BilinearFormIntegrator* mmi = (mass_wdet_.Size() > 0)
                                    ? static_cast<BilinearFormIntegrator*>(
                                       new LeanVectorMassIntegrator(mass_wdet_, mass_coeff_))
                                    : new VectorMassIntegrator(mass_coeff_);
      mmi->SetIntRule(mass_rule_);
      form.AddDomainIntegrator(mmi);
      if (mass) { *mass = mmi; }
   }
   if (include_grad_div && gamma_)
   {
      // gamma (div u, div v) via the in-repo sum-factorized
      // GradDivIntegrator -- replaces ElasticityIntegrator(lambda, mu=0),
      // elmat-identical (pinned by grad_div_integrator_test) but with fused O(p^4)
      // tensor PA kernels instead of elasticity's dense non-tensor path.
      // Stiffness-type integrand -> the 2k + dim - 1 default rule. The
      // high-order forms take gamma from its cached per-element values
      // (the coefficient object still serves full and LOR assembly).
      auto* gdi = new GradDivIntegrator(gamma ? *gamma : *gamma_);
      if (!gamma && gamma_elem_.Size() > 0)
      {
         gdi->SetElementCoefficient(&gamma_elem_);
      }
      gdi->SetIntRule(&rules_.Get(geom_, 2 * ku_ + dim_ - 1));
      form.AddDomainIntegrator(gdi);
      added = gdi;
   }
   return added;
}

Operator& StokesOperator::ViscousUnconstrained()
{
   if (!Kunc_.Ptr())
   {
      INCNS_PROFILE("viscous_unconstrained");
      auto* kui = new LeanVectorDiffusionIntegrator(nu_, DiffusionStorage());
      kui->SetIntRule(&rules_.Get(geom_, 2 * ku_ + dim_ - 1));
      viscous_form_ = std::make_unique<PaOperator>(spaces_.Velocity(),
                      *scratch_);
      viscous_form_->AddDomainIntegrator(kui);
      viscous_form_->Assemble();
      TrueDofOperator(viscous_form_->GetProlongation(), *viscous_form_,
                      viscous_form_->GetProlongation(), Kunc_);
   }
   return *Kunc_.Ptr();
}

LeanVectorDiffusionIntegrator::Storage StokesOperator::DiffusionStorage() const
{
   return opts_.compact_diffusion ? LeanVectorDiffusionIntegrator::Storage::Compact
          : LeanVectorDiffusionIntegrator::Storage::Mfem;
}

void StokesOperator::TrueDofOperator(const Operator* Pin, Operator& A,
                                     const Operator* Pout, OperatorPtr& out)
{
   if (IsIdentityProlongation(Pin) && IsIdentityProlongation(Pout))
   {
      // One rank, conforming: MFEM's prolongation is an IdentityOperator, and
      // a RAPOperator over it would only copy, through two work vectors it
      // keeps on the device for the run (a velocity-sized pair per operator).
      // MFEM's own FormSystemMatrix skips the identity the same way.
      out.Reset(&A, false);
   }
   else if (Pin && Pout)
   {
      // Pout^T A Pin, as MFEM's SetupRAP builds it (a RAPOperator), its work
      // vectors shared by all of them.
      out.Reset(new SharedRAPOperator(*Pout, A, *Pin, *scratch_), true);
   }
   else
   {
      MFEM_ABORT("stokes_operator: one-sided prolongation is not supported");
   }
}

void StokesOperator::ReleaseViscousUnconstrained()
{
   Kunc_.Clear();
   viscous_form_.reset();
}

void StokesOperator::SetViscosity(double nu)
{
   MFEM_VERIFY(nu > 0.0, "stokes_operator: viscosity must be positive");
   nu_.constant = nu;
   opts_.nu = nu;
   ReleaseViscousUnconstrained(); // it was nu_old * K
   rest_diag_valid_ = false;      // and so was this: nu_old K + grad-div
   momentum_stale_ = true;        // and the momentum form's viscous data
}

void StokesOperator::UpdateGeometry()
{
   INCNS_PROFILE("stokes_operator::update_geometry");
   MFEM_VERIFY(!rot_form_, "stokes_operator: UpdateGeometry() with a rotation "
               "term is not supported (a moving mesh needs the convective "
               "form)");
   // gamma = c h_K: the element sizes changed with the nodes.
   if (auto* h = dynamic_cast<MeshSizeCoefficient*>(gamma_.get()))
   {
      h->Invalidate();
      if (gamma_elem_.Size() > 0) { h->ElementValues(gamma_elem_); }
   }
   if (auto* h = dynamic_cast<MeshSizeCoefficient*>(gamma_lor_.get()))
   {
      h->Invalidate();
   }
   {
      INCNS_PROFILE("mass");
      if (DiagonalMass())
      {
         // Into the same vectors: M_, the Schur PC's mass inverse and the
         // momentum shift hold references to them.
         AssembleCollocatedMassDiagonal(spaces_.Velocity(), *mass_rule_,
                                        mass_diag_);
         if (!spaces_.Velocity().Conforming())
         {
            const double ratio =
               LumpedMassRatio(spaces_.Velocity(), *mass_rule_, mass_diag_);
            MFEM_VERIFY(ratio >= kMinLumpedMassRatio, "stokes_operator: the "
                        "collocated mass lumped at hanging nodes is too small "
                        "or negative somewhere after a mesh move (min b / ref "
                        "= " << ratio << ")");
         }
         mass_diag_inv_ = 1.0;
         mass_diag_inv_ /= mass_diag_;
      }
      else
      {
         MassWeightDet(spaces_.Velocity(), *mass_rule_, mass_wdet_);
         {
            ParBilinearForm diag_form(&spaces_.Velocity());
            auto* mi = new LeanVectorMassIntegrator(mass_wdet_);
            mi->SetIntRule(mass_rule_);
            diag_form.AddDomainIntegrator(mi);
            diag_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            diag_form.Assemble();
            diag_form.AssembleDiagonal(mass_diag_);
         }
         mass_form_->Assemble();
      }
   }
   {
      INCNS_PROFILE("divergence");
      div_form_->Assemble();
   }
   ReleaseViscousUnconstrained();
   rest_diag_valid_ = false;
   momentum_stale_ = true;
}

void StokesOperator::BuildMomentum()
{
   INCNS_PROFILE("stokes_operator::momentum");

   // Fused matrix-free momentum block A = c0*M + nu*K (+ grad-div). Rebuilding
   // recomputes only the coefficient*geometry PA data (mesh geometric factors
   // are cached), so it is cheap -- much cheaper than a full solver rebuild --
   // and it hands us the new diagonal for free.
   momentum_form_ = std::make_unique<PaOperator>(spaces_.Velocity(),
                    *scratch_);
   BilinearFormIntegrator* diffusion = nullptr;
   mass_integ_ = nullptr;
   // The diagonal mass enters as a diagonal shift instead (below).
   grad_div_integ_ = AddMomentumIntegrators(*momentum_form_,
                     /*include_mass=*/!DiagonalMass(),
                     /*include_grad_div=*/true, nullptr, &diffusion,
                     &mass_integ_);
   momentum_stale_ = false;
   {
      INCNS_PROFILE("momentum::assemble");
      momentum_form_->Assemble();
   }
   // nu K and grad-div in one kernel: the velocity gradients once per apply.
   if (opts_.fuse_grad_div && grad_div_integ_ && diffusion)
   {
      static_cast<LeanVectorDiffusionIntegrator*>(diffusion)->FuseGradDiv(
         *grad_div_integ_);
   }
   {
      INCNS_PROFILE("momentum::form_system");
      // One P^T A P serves the unconstrained block (reactions) and, wrapped
      // with the Dirichlet constraints (DIAG_ONE, as FormSystemMatrix does),
      // the one the solver applies: one RAP and its work vectors, not two.
      // The old K_ wraps the old Aunc_: release it first.
      K_.Clear();
      if (DiagonalMass())
      {
         // c0 M is c0 times the mass diagonal, added at every apply with c0
         // read then: a new mass factor re-assembles nothing.
         TrueDofOperator(momentum_form_->GetProlongation(), *momentum_form_,
                         momentum_form_->GetProlongation(), Arest_);
         Aunc_.Reset(new DiagonalShiftOperator(*Arest_.Ptr(), mass_diag_,
                                               opts_.mass_coeff), true);
      }
      else
      {
         TrueDofOperator(momentum_form_->GetProlongation(), *momentum_form_,
                         momentum_form_->GetProlongation(), Aunc_);
      }
      K_.Reset(new ConstrainedOperator(Aunc_.Ptr(), ess_tdofs_, false), true);
   }
   {
      INCNS_PROFILE("momentum::diagonal");
      // Only c0 changes with Delta-t: c0 M's diagonal is mass_diag_ scaled
      // (same rule), and that of nu K (+ grad-div) is assembled once. The
      // mass integrator is in the form only when c0 > 0.
      if (!rest_diag_valid_) { AssembleRestDiagonal(*diffusion); }
      UpdateMomentumDiagonal();
   }

   BuildLorSource();

   // The outer operator A + N re-points at the rebuilt momentum block.
   if (Nc_)
   {
      full_momentum_ = std::make_unique<SumOperator>(K_.Ptr(), 1.0, Nc_.get(),
                       1.0, false, false);
   }
}

void StokesOperator::UpdateMomentumDiagonal()
{
   momentum_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
   momentum_diag_.UseDevice(true);
   const double c0 = (opts_.mass_coeff > 0.0) ? opts_.mass_coeff : 0.0;
   add(rest_diag_, c0, mass_diag_, momentum_diag_);
}

void StokesOperator::BuildLorSource()
{
   // Non-frozen LOR source (c0*M + nu*K + grad-div, the latter with the
   // parent-element gamma) tracks the current mass factor, so the
   // BoomerAMG hierarchy is rebuilt to match on each refresh (see lor_frozen for
   // the alternative that reuses a single nu*K hierarchy).
   if (opts_.lor_momentum && !opts_.lor_frozen)
   {
      lor_form_ = std::make_unique<ParBilinearForm>(&spaces_.Velocity());
      AddMomentumIntegrators(*lor_form_, /*include_mass=*/true,
                             /*include_grad_div=*/true, gamma_lor_.get());
      if (w_lor_)
      {
         // Legacy element matrices on the LOR elements, reading the LOR-space
         // w (MFEM sets the collocated rule during LOR assembly).
         lor_form_->AddDomainIntegrator(new VectorRotationalConvectionIntegrator(
                                           *w_lor_, opts_.rotation_alpha));
      }
   }
}

void StokesOperator::AssembleRestDiagonal(BilinearFormIntegrator& diffusion)
{
   INCNS_PROFILE("momentum::rest_diagonal");
   std::vector<BilinearFormIntegrator*> parts{&diffusion};
   if (grad_div_integ_) { parts.push_back(grad_div_integ_); }
   ParBilinearForm proxy(&spaces_.Velocity());
   proxy.AddDomainIntegrator(new DiagonalSumProxy(std::move(parts)));
   proxy.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   proxy.Assemble();
   rest_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
   rest_diag_.UseDevice(true);
   proxy.AssembleDiagonal(rest_diag_);
   rest_diag_valid_ = true;
}

void StokesOperator::SetMassCoeff(double c0)
{
   MFEM_VERIFY(c0 >= 0.0, "stokes_operator: mass_coeff must be non-negative");
   const double c0_old = opts_.mass_coeff;
   mass_coeff_.constant = c0;
   opts_.mass_coeff = c0;
   if (DiagonalMass() && momentum_form_ && !momentum_stale_)
   {
      // The diagonal c0 M is the momentum operator's diagonal shift, which
      // reads c0 at each apply: only the diagonal (and an LOR source) change.
      INCNS_PROFILE("momentum::mass_coeff");
      UpdateMomentumDiagonal();
      BuildLorSource();
      return;
   }
   if (momentum_form_ && mass_integ_ && !momentum_stale_ && c0_old > 0.0 &&
       c0 > 0.0)
   {
      // Only c0 M depends on dt: re-assemble the mass integrator's data alone
      // (c0 W det J from the new mass_coeff_: the values a new form computes,
      // so the same operator bitwise). The viscous and grad-div data, their
      // fusion, the true-dof and constrained operators stay -- rebuilding
      // the whole form cost ~1.3 ms per step-size change on the Re_tau 180
      // channel, where CFL-controlled steps change dt at ~75 % of the steps
      // (2026-10-15).
      INCNS_PROFILE("momentum::mass_coeff");
      mass_integ_->AssemblePA(spaces_.Velocity());
      UpdateMomentumDiagonal();
      BuildLorSource();
      return;
   }
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
