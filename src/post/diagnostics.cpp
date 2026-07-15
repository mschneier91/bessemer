#include "post/diagnostics.hpp"

namespace incns
{

using namespace mfem;

namespace
{
// Fill an elevated error/measurement rule table for the quad/hex geometries.
// Elevated (2*k_u + 2) so smooth non-polynomial integrands (|u|^2, |grad u|^2)
// are integrated well past the default rule -- the same hygiene the convergence
// oracles use.
void ElevatedRules(const RuleBook& rules, int order,
                   const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, order);
   irs[Geometry::CUBE] = &rules.Get(Geometry::CUBE, order);
}

int MeasureOrder(const ParGridFunction& u)
{
   return 2 * u.ParFESpace()->GetMaxElementOrder() + 2;
}
} // namespace

double KineticEnergy(const ParGridFunction& u, const RuleBook& rules)
{
   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, MeasureOrder(u), irs);
   Vector zero(u.ParFESpace()->GetVDim());
   zero = 0.0;
   VectorConstantCoefficient zero_vec(zero);
   const double l2 = u.ComputeL2Error(zero_vec, irs); // sqrt(int |u|^2)
   return 0.5 * l2 * l2;
}

double DissipationRate(const ParGridFunction& u, double nu,
                       const RuleBook& rules)
{
   ParFiniteElementSpace& vfes = *u.ParFESpace();
   const Geometry::Type geom =
      (vfes.GetParMesh()->Dimension() == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // (grad u, grad v) over the vector space, matrix-free; nu * (K u, u) on the
   // true dofs = nu * int |grad u|^2, reduced across ranks.
   ParBilinearForm k(&vfes);
   k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto* integ = new VectorDiffusionIntegrator;
   integ->SetIntRule(&rules.Get(geom, MeasureOrder(u)));
   k.AddDomainIntegrator(integ);
   k.Assemble();

   OperatorPtr K;
   Array<int> no_ess;
   k.FormSystemMatrix(no_ess, K);

   Vector U, KU;
   u.GetTrueDofs(U);
   KU.SetSize(U.Size());
   K->Mult(U, KU);
   const double grad_sq = InnerProduct(vfes.GetComm(), U, KU);
   return nu * grad_sq;
}

double DivergenceNorm(const ParGridFunction& u, const RuleBook& rules)
{
   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, MeasureOrder(u), irs);
   ConstantCoefficient zero(0.0);
   // ComputeDivError of a vector H1 field = sqrt(int (div u)^2); same measure
   // as the divergence fast-tier check.
   return u.ComputeDivError(&zero, irs);
}

DiagnosticsLog::DiagnosticsLog(const std::string& path, MPI_Comm comm)
{
   MPI_Comm_rank(comm, &rank_);
   if (rank_ != 0) { return; }
   os_.open(path);
   MFEM_VERIFY(os_.is_open(), "DiagnosticsLog: cannot open '" << path << "'");
   os_ << "t,kinetic_energy,dissipation,divergence\n";
}

void DiagnosticsLog::Write(double t, double energy, double dissipation,
                           double divergence)
{
   if (rank_ != 0) { return; }
   os_.precision(15);
   os_ << t << ',' << energy << ',' << dissipation << ',' << divergence << '\n';
   os_.flush();
}

} // namespace incns
