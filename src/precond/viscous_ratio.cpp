#include "precond/viscous_ratio.hpp"

namespace incns
{

using namespace mfem;

namespace
{

// max_a K_aa / M_aa on a scalar space (local; device-aware Vector ops).
double LocalMaxRatio(FiniteElementSpace& fes, BilinearForm& k, BilinearForm& m)
{
   Vector kd(fes.GetTrueVSize()), md(fes.GetTrueVSize());
   kd.UseDevice(true);
   md.UseDevice(true);
   k.AssembleDiagonal(kd);
   m.AssembleDiagonal(md);
   kd /= md;
   return kd.Size() ? kd.Max() : 0.0;
}

// The scalar mass and diffusion forms at the solver's rules, PA.
void AddForms(BilinearForm& k, BilinearForm& m, const RuleBook& rules,
              Geometry::Type geom, int p, int dim)
{
   auto* di = new DiffusionIntegrator;
   di->SetIntRule(&rules.Get(geom, 2 * p + dim - 1));
   k.AddDomainIntegrator(di);
   auto* mi = new MassIntegrator;
   mi->SetIntRule(&rules.Get(geom, 2 * p));
   m.AddDomainIntegrator(mi);
   for (BilinearForm* f : {&k, &m})
   {
      f->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      f->Assemble();
   }
}

} // namespace

ViscousRatioDiagnostic::ViscousRatioDiagnostic(
   const ParFiniteElementSpace& vfes, const RuleBook& rules)
{
   ParMesh& mesh = *vfes.GetParMesh();
   const int dim = mesh.Dimension();
   const int p = vfes.GetMaxElementOrder();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // The mesh's ratio, on true dofs (AssembleDiagonal of a ParBilinearForm
   // returns the assembled -- summed over elements and ranks -- diagonal).
   {
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace sfes(&mesh, &fec);
      ParBilinearForm k(&sfes), m(&sfes);
      AddForms(k, m, rules, geom, p, dim);
      r_max_ = LocalMaxRatio(sfes, k, m);
      MPI_Allreduce(MPI_IN_PLACE, &r_max_, 1, MPI_DOUBLE, MPI_MAX,
                    mesh.GetComm());
   }

   // c_p: the same ratio on one unit element of order p, over p^2. Serial
   // and identical on every rank.
   {
      Mesh unit = (dim == 3)
                  ? Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON)
                  : Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
      H1_FECollection fec(p, dim);
      FiniteElementSpace ufes(&unit, &fec);
      BilinearForm k(&ufes), m(&ufes);
      AddForms(k, m, rules, geom, p, dim);
      c_p_ = LocalMaxRatio(ufes, k, m) / (static_cast<double>(p) * p);
   }
}

} // namespace incns
