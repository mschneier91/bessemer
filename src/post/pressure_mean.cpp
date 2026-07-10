#include "post/pressure_mean.hpp"

#include <memory>

namespace incns
{

using namespace mfem;

double MassWeightedMean(const ParGridFunction& p, const RuleBook& rules)
{
   ParFiniteElementSpace& fes = *p.ParFESpace();
   ParMesh& mesh = *fes.GetParMesh();
   const MPI_Comm comm = mesh.GetComm();

   const int dim = mesh.Dimension();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // Global polynomial order -- robust to ranks that own no local elements
   // (GetFE(0) would be invalid there), taken as a max reduction.
   const int local_order = (mesh.GetNE() > 0) ? fes.GetFE(0)->GetOrder() : 0;
   int order = 0;
   MPI_Allreduce(&local_order, &order, 1, MPI_INT, MPI_MAX, comm);

   // Assemble the "integrate against each basis function" functional
   // ones(i) = ∫ phi_i, then reduce it to TRUE dofs (ParallelAssemble sums the
   // shared-dof contributions) so every dof is counted exactly once. Pairing
   // with the field's true dofs gives ∫ p; pairing with the all-ones vector
   // gives |Omega| (the basis is a partition of unity). Doing this on vdofs and
   // MPI-summing would double-count shared dofs.
   ParLinearForm ones(&fes);
   ConstantCoefficient one(1.0);
   auto* integ = new DomainLFIntegrator(one);
   integ->SetIntRule(&rules.Get(geom, 2 * order));
   ones.AddDomainIntegrator(integ);
   ones.Assemble();

   std::unique_ptr<HypreParVector> ones_true(ones.ParallelAssemble());
   Vector p_true;
   p.GetTrueDofs(p_true);
   Vector unit(p_true.Size());
   unit = 1.0;

   const double integral = InnerProduct(comm, *ones_true, p_true);
   const double volume = InnerProduct(comm, *ones_true, unit);
   MFEM_VERIFY(volume > 0.0, "pressure_mean: non-positive domain volume");
   return integral / volume;
}

void SubtractMean(ParGridFunction& p, const RuleBook& rules)
{
   p -= MassWeightedMean(p, rules);
}

} // namespace incns
