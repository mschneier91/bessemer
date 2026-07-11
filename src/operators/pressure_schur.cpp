#include "operators/pressure_schur.hpp"

namespace incns
{

using namespace mfem;

PressureMassSchur::PressureMassSchur(ParFiniteElementSpace& pfes,
                                     const RuleBook& rules, double scale)
   : Solver(pfes.GetTrueVSize()), scale_(scale), mass_(&pfes)
{
   MFEM_VERIFY(scale_ > 0.0, "pressure_schur: scale must be positive");

   const int dim = pfes.GetParMesh()->Dimension();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int kp = pfes.FEColl()->GetOrder();

   auto* mi = new MassIntegrator;
   mi->SetIntRule(&rules.Get(geom, 2 * kp));
   mass_.AddDomainIntegrator(mi);
   mass_.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_.Assemble();

   Vector diag(pfes.GetTrueVSize());
   mass_.AssembleDiagonal(diag);
   inv_diag_.SetSize(diag.Size());
   for (int i = 0; i < diag.Size(); ++i)
   {
      MFEM_VERIFY(diag(i) > 0.0, "pressure_schur: non-positive mass diagonal");
      inv_diag_(i) = scale_ / diag(i);
   }
}

void PressureMassSchur::Mult(const Vector& x, Vector& y) const
{
   y = x;
   y *= inv_diag_;
}

} // namespace incns
