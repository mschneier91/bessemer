#include "operators/convection.hpp"

#include "operators/directional_do_nothing.hpp"

namespace incns
{

using namespace mfem;

Convection::Convection(MixedSpaces& spaces, const RuleBook& rules)
   : spaces_(spaces), rules_(rules)
{
   // Quad/hex only, same as every other operator here (StokesOperator does the
   // identical mapping) -- the tensor-product structure the PA kernels assume.
   const Geometry::Type geom =
      (spaces_.Dim() == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // THE dealiasing choice. Elevated from the linear blocks' ~2k to 3k so the
   // cubic (u.grad)u . v integrand is integrated exactly -- see the header for
   // why, and do not lower it. Verified against MFEM's PA path, which selects
   // `IntRule ? IntRule : &GetRule(...)` (fem/integ/nonlininteg_vecconvection_pa.cpp),
   // so this rule is honored under partial assembly rather than silently
   // replaced by the integrator's default.
   rule_ = &rules_.Get(geom, DealiasedOrder(spaces_.OrderU()));

   form_ = std::make_unique<ParNonlinearForm>(&spaces_.Velocity());
   // No coefficient: the integrator returns +(u.grad)u, and the sign stays with
   // the caller (see Mult's doc comment). MFEM's navier miniapp folds a -1 in
   // here instead; keeping it out makes the operator directly testable against
   // the analytic term.
   auto* nlfi = new VectorConvectionNLFIntegrator();
   nlfi->SetIntRule(rule_);
   form_->AddDomainIntegrator(nlfi);

   // Partial assembly, matching every other operator in this codebase (the
   // nonlinear form is applied every step and never needs its matrix).
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->Setup();
}

void Convection::EnableDirectionalDoNothing(const Array<int>& outflow_attrs)
{
   MFEM_VERIFY(!ddn_form_, "convection: directional do-nothing already enabled");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   const int n_attr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   ddn_marker_.SetSize(n_attr);
   ddn_marker_ = 0;
   for (int a : outflow_attrs)
   {
      MFEM_VERIFY(a >= 1 && a <= n_attr, "convection: outflow attribute " << a
                  << " not on the mesh");
      ddn_marker_[a - 1] = 1;
   }
   ddn_u_ = std::make_unique<ParGridFunction>(&spaces_.Velocity());
   *ddn_u_ = 0.0;
   ddn_form_ = std::make_unique<ParLinearForm>(&spaces_.Velocity());
   auto* integ = new DirectionalDoNothingIntegrator(*ddn_u_, 0.5);
   // Same exactness as the interior term: (u.n)_- (u.phi) is cubic in u.
   const Geometry::Type face =
      (spaces_.Dim() == 3) ? Geometry::SQUARE : Geometry::SEGMENT;
   integ->SetIntRule(&rules_.Get(face, DealiasedOrder(spaces_.OrderU())));
   ddn_form_->AddBdrFaceIntegrator(integ, ddn_marker_);
   ddn_true_.SetSize(spaces_.Velocity().GetTrueVSize());
   ddn_true_.UseDevice(true);
}

void Convection::Mult(const Vector& u, Vector& y) const
{
   form_->Mult(u, y);
   if (!ddn_form_) { return; }
   // The boundary term: legacy host assembly over the marked faces (few),
   // reading u's element values on the host.
   ddn_u_->SetFromTrueDofs(u);
   ddn_u_->HostRead();
   ddn_form_->Assemble();
   ddn_form_->ParallelAssemble(ddn_true_); // MFEM-owned buffer (CLAUDE.md §6)
   y += ddn_true_;
}

} // namespace incns
