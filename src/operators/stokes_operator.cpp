#include "operators/stokes_operator.hpp"

#include "mesh/periodic_box.hpp" // AssertTensorProductGeometry
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesOperator::StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                               const StokesOperatorOptions& opts,
                               const Array<int>* ess_tdofs)
   : spaces_(spaces), opts_(opts), nu_(opts.nu), mass_coeff_(opts.mass_coeff),
     mass_form_(&spaces.Velocity()),
     momentum_form_(&spaces.Velocity()),
     viscous_form_(&spaces.Velocity()),
     div_form_(&spaces.Velocity(), &spaces.Pressure())
{
   INCNS_PROFILE("stokes_operator::assemble");

   MFEM_VERIFY(opts_.nu > 0.0, "stokes_operator: viscosity must be positive");
   MFEM_VERIFY(opts_.mass_coeff >= 0.0,
               "stokes_operator: mass_coeff must be non-negative");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   AssertTensorProductGeometry(mesh);

   const int dim = spaces_.Dim();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int ku = spaces_.OrderU();

   // Essential velocity dofs eliminated from the viscous block and the
   // divergence trial columns; empty when no Dirichlet BCs are given. Stored as
   // a member: the constrained operators MakeRef this list.
   if (ess_tdofs) { ess_tdofs_ = *ess_tdofs; }

   // --- velocity mass: GL 2k default, or the collocated GLL (diagonal) rule --
   const IntegrationRule* mass_rule;
   if (opts_.collocated_mass)
   {
      // Diagonality requires the GLL *nodal basis*: GLL points against a
      // non-collocated basis silently give a full, under-integrated mass.
      const auto* h1 =
         dynamic_cast<const H1_FECollection*>(spaces_.Velocity().FEColl());
      MFEM_VERIFY(h1 && h1->GetBasisType() == BasisType::GaussLobatto,
                  "stokes_operator: collocated_mass requires the GLL nodal "
                  "H1 basis (BasisType::GaussLobatto)");
      mass_rule = &rules.CollocatedMass(geom, ku);
   }
   else
   {
      mass_rule = &rules.Get(geom, 2 * ku);
   }
   {
      INCNS_PROFILE("mass");
      auto* mi = new VectorMassIntegrator;
      mi->SetIntRule(mass_rule);
      mass_form_.AddDomainIntegrator(mi);
      mass_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mass_form_.Assemble();
      Array<int> empty;
      mass_form_.FormSystemMatrix(empty, M_);
      mass_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
      mass_form_.AssembleDiagonal(mass_diag_);
   }

   // --- momentum block A = mass_coeff*M + nu*K. Diffusion at the default
   // exactness order 2k + dim - 1 (covers the metric factors on deformed
   // elements, not just the affine minimum); the mass term reuses the mass rule
   // (incl. the collocated-GLL option -- the SEM payoff is exactly a diagonal
   // mass contribution in this block at small dt) ------------------------------
   {
      INCNS_PROFILE("momentum");
      auto* ki = new VectorDiffusionIntegrator(nu_);
      ki->SetIntRule(&rules.Get(geom, 2 * ku + dim - 1));
      momentum_form_.AddDomainIntegrator(ki);
      if (opts_.mass_coeff > 0.0)
      {
         auto* mmi = new VectorMassIntegrator(mass_coeff_);
         mmi->SetIntRule(mass_rule);
         momentum_form_.AddDomainIntegrator(mmi);
      }
      momentum_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      momentum_form_.Assemble();
      momentum_form_.FormSystemMatrix(ess_tdofs_, K_);
      momentum_diag_.SetSize(spaces_.Velocity().GetTrueVSize());
      momentum_form_.AssembleDiagonal(momentum_diag_);
   }

   // --- pure viscous nu*K, UNCONSTRAINED: explicit RHS terms in time steppers -
   {
      INCNS_PROFILE("viscous_unconstrained");
      auto* kui = new VectorDiffusionIntegrator(nu_);
      kui->SetIntRule(&rules.Get(geom, 2 * ku + dim - 1));
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
      di->SetIntRule(&rules.Get(geom, 2 * ku + 2));
      div_form_.AddDomainIntegrator(di);
      div_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      div_form_.Assemble();
      div_form_.FormRectangularSystemMatrix(ess_tdofs_, ess_p_tdofs_, B_);
   }
}

} // namespace incns
