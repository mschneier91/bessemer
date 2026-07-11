#include "operators/stokes_operator.hpp"

#include "mesh/periodic_box.hpp" // AssertTensorProductGeometry
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesOperator::StokesOperator(MixedSpaces& spaces, const RuleBook& rules,
                               const StokesOperatorOptions& opts)
   : spaces_(spaces), opts_(opts), nu_(opts.nu),
     mass_form_(&spaces.Velocity()),
     viscous_form_(&spaces.Velocity()),
     div_form_(&spaces.Velocity(), &spaces.Pressure())
{
   INCNS_PROFILE("stokes_operator::assemble");

   MFEM_VERIFY(opts_.nu > 0.0, "stokes_operator: viscosity must be positive");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   AssertTensorProductGeometry(mesh);

   const int dim = spaces_.Dim();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int ku = spaces_.OrderU();
   const int kp = spaces_.OrderP();

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

   // --- viscous block nu*K: grad(u):grad(v), integrand degree 2(k-1) + metric;
   // exact on the affine tensor-product meshes we accept ------------------------
   {
      INCNS_PROFILE("viscous");
      auto* ki = new VectorDiffusionIntegrator(nu_);
      ki->SetIntRule(&rules.Get(geom, std::max(2 * ku - 2, 0)));
      viscous_form_.AddDomainIntegrator(ki);
      viscous_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      viscous_form_.Assemble();
      Array<int> empty;
      viscous_form_.FormSystemMatrix(empty, K_);
   }

   // --- divergence block B = (div u, q): integrand degree (k_u - 1) + k_p ----
   {
      INCNS_PROFILE("divergence");
      auto* di = new VectorDivergenceIntegrator;
      di->SetIntRule(&rules.Get(geom, (ku - 1) + kp));
      div_form_.AddDomainIntegrator(di);
      div_form_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      div_form_.Assemble();
      Array<int> empty_u, empty_p;
      div_form_.FormRectangularSystemMatrix(empty_u, empty_p, B_);
   }
}

} // namespace incns
