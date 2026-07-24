#include "operators/convection.hpp"

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

void Convection::Mult(const Vector& u, Vector& y) const
{
   form_->Mult(u, y);
}

} // namespace incns
