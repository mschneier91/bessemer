// LeanVectorDiffusionIntegrator: the partial-assembly diagonal without
// per-thread stack (operators/lean_vector_diffusion).
//  L1 Its diagonal is MFEM's VectorDiffusionIntegrator diagonal: 2D and 3D,
//     orders 2 and 3, a deformed mesh and a variable coefficient, byNODES and
//     byVDIM, to roundoff (5e-15) on the true dofs.
//  L2 The operator itself is untouched: Mult is MFEM's, bit for bit.
// Self-contained: MFEM and the integrator.

#include <gtest/gtest.h>

#include "operators/lean_vector_diffusion.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::LeanVectorDiffusionIntegrator;

namespace
{

void Deform(const Vector& x, Vector& y)
{
   y = x;
   y(0) += 0.08 * std::sin(2.3 * x(1));
   y(1) += 0.06 * x(0) * x(0);
   if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
}

double NuFn(const Vector& x)
{
   return 0.7 + 0.5 * x(0) + 0.3 * x(1) * x(1);
}

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

void Check(int dim, int order, Ordering::Type ordering)
{
   Mesh serial = (dim == 2)
                 ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                 : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
   serial.Transform(Deform);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim, ordering);
   FunctionCoefficient nu(NuFn);
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = IntRules.Get(geom, 2 * order + dim - 1);

   ParBilinearForm lean(&fes), ref(&fes);
   auto* li = new LeanVectorDiffusionIntegrator(nu);
   li->SetIntRule(&ir);
   lean.AddDomainIntegrator(li);
   auto* ri = new VectorDiffusionIntegrator(nu);
   ri->SetIntRule(&ir);
   ref.AddDomainIntegrator(ri);
   lean.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   ref.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   lean.Assemble();
   ref.Assemble();

   const int n = fes.GetTrueVSize();
   Vector d_lean(n), d_ref(n);
   lean.AssembleDiagonal(d_lean);
   ref.AssembleDiagonal(d_ref);
   EXPECT_GT(Norm(d_ref), 0.0);
   // The same sum in another order (direct against sum-factorized): equal
   // to roundoff, not bit for bit. The largest entrywise difference is
   // printed so the bound below can be judged against what is measured.
   d_lean -= d_ref;
   const double d_rel = Norm(d_lean) / Norm(d_ref);
   double d_max = d_lean.Normlinf(), r_max = d_ref.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &r_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      mfem::out << "[lean diffusion] dim " << dim << " order " << order
                << ": diagonal difference, relative l2 " << d_rel
                << ", largest entry / largest diagonal entry "
                << d_max / r_max << std::endl;
   }
   EXPECT_LT(d_rel, 5e-15); // measured 1.5e-16 to 3.7e-16 (H100, 2026-10-13)
   EXPECT_LT(d_max, 5e-15 * r_max); // measured 3.2e-16 to 8.2e-16

   Array<int> no_ess;
   OperatorPtr A_lean, A_ref;
   lean.FormSystemMatrix(no_ess, A_lean);
   ref.FormSystemMatrix(no_ess, A_ref);
   Vector x(n), y_lean(n), y_ref(n);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   x.Randomize(3 + rank);
   A_lean->Mult(x, y_lean);
   A_ref->Mult(x, y_ref);
   // The apply is MFEM's own kernel on MFEM's own data in both: bit for bit.
   y_lean -= y_ref;
   EXPECT_EQ(Norm(y_lean), 0.0);
}

} // namespace

TEST(LeanVectorDiffusion, L1L2_DiagonalAndApplyAreMfems)
{
   for (const int dim : {2, 3})
   {
      for (const int order : {2, 3})
      {
         for (const Ordering::Type ordering : {Ordering::byNODES, Ordering::byVDIM})
         {
            SCOPED_TRACE(testing::Message() << "dim " << dim << " order " << order
                         << (ordering == Ordering::byVDIM ? " byVDIM" : " byNODES"));
            Check(dim, order, ordering);
         }
      }
   }
}
