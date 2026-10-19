// LeanVectorMassIntegrator against MFEM's VectorMassIntegrator (both
// partially assembled): the same apply and diagonal, to roundoff, on a curved
// mesh in 2D and 3D, with and without a constant coefficient (on a device the
// 3D apply is VectorMassApply3DPoints). VectorMassApply3DPoints itself, on
// every device, against MFEM's apply at orders 2 to 7.

#include <gtest/gtest.h>

#include "operators/lean_vector_mass.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::LeanVectorMassIntegrator;
using incns::RuleBook;

namespace
{

Mesh CurvedBox(int dim)
{
   Mesh mesh = (dim == 3)
               ? Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0,
                                       1.0)
               : Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, false,
                                       1.0, 1.0);
   mesh.SetCurvature(3);
   mesh.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      for (int d = 0; d < x.Size(); ++d)
      {
         y(d) += 0.05 * std::sin(M_PI * x((d + 1) % x.Size())) * x(d);
      }
   });
   return mesh;
}

void Check(int dim, int k, double c)
{
   SCOPED_TRACE(testing::Message() << "dim " << dim << ", k " << k
                << ", coefficient " << c);
   Mesh serial = CurvedBox(dim);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(k, dim);
   ParFiniteElementSpace V(&mesh, &fec, dim);
   RuleBook rules;
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = rules.Get(geom, 2 * k);
   ConstantCoefficient q(c);
   Vector wdet;
   incns::MassWeightDet(V, ir, wdet);

   auto build = [&](BilinearFormIntegrator * integ)
   {
      integ->SetIntRule(&ir);
      auto form = std::make_unique<ParBilinearForm>(&V);
      form->AddDomainIntegrator(integ);
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form->Assemble();
      return form;
   };
   auto ref = build(c == 1.0 ? new VectorMassIntegrator
                    : new VectorMassIntegrator(q));
   auto lean = build(c == 1.0 ? new LeanVectorMassIntegrator(wdet)
                     : new LeanVectorMassIntegrator(wdet, q));

   Vector x(V.GetVSize()), y_ref(V.GetVSize()), y(V.GetVSize());
   for (Vector* v : {&x, &y_ref, &y}) { v->UseDevice(true); }
   x.Randomize(3 + mesh.GetMyRank());
   ref->Mult(x, y_ref);
   lean->Mult(x, y);
   Vector d_ref(V.GetTrueVSize()), d(V.GetTrueVSize());
   d_ref.UseDevice(true);
   d.UseDevice(true);
   ref->AssembleDiagonal(d_ref);
   lean->AssembleDiagonal(d);
   double diff = 0.0;
   const double* a = y.HostRead();
   const double* b = y_ref.HostRead();
   for (int i = 0; i < y.Size(); ++i) { diff = std::max(diff, std::abs(a[i] - b[i])); }
   const double* da = d.HostRead();
   const double* db = d_ref.HostRead();
   for (int i = 0; i < d.Size(); ++i) { diff = std::max(diff, std::abs(da[i] - db[i])); }
   MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LE(diff, 1e-14 * y_ref.Normlinf());
}

} // namespace

TEST(LeanVectorMass, MatchesMfem3D)
{
   Check(3, 2, 1.0);
   Check(3, 4, 2.5);
}

TEST(LeanVectorMass, PointKernelMatchesMfem)
{
   for (const int k : {2, 4, 6, 7})
   {
      SCOPED_TRACE(testing::Message() << "k " << k);
      Mesh serial = CurvedBox(3);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(k, 3);
      ParFiniteElementSpace V(&mesh, &fec, 3);
      RuleBook rules;
      const IntegrationRule& ir = rules.Get(Geometry::CUBE, 2 * k);
      Vector wdet;
      incns::MassWeightDet(V, ir, wdet);
      // MFEM's apply (its own kernels on any device), on local dofs.
      auto* mi = new VectorMassIntegrator;
      mi->SetIntRule(&ir);
      ParBilinearForm form(&V);
      form.AddDomainIntegrator(mi);
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form.Assemble();
      Vector x(V.GetVSize()), y_ref(V.GetVSize()), y(V.GetVSize());
      for (Vector* v : {&x, &y_ref, &y}) { v->UseDevice(true); }
      x.Randomize(11 + mesh.GetMyRank());
      form.Mult(x, y_ref);
      // The point kernel on E-vectors (C = 1: D = W det J).
      const Operator* R = V.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      const DofToQuad& maps = V.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
      ASSERT_EQ(maps.ndof, k + 1);
      ASSERT_EQ(maps.nqpt, k + 1);
      Vector xe(R->Height()), ye(R->Height());
      xe.UseDevice(true);
      ye.UseDevice(true);
      R->Mult(x, xe);
      ye = 0.0;
      ASSERT_TRUE(incns::VectorMassApply3DPoints(maps.ndof, maps.nqpt, mesh.GetNE(),
                  maps.B, wdet, xe, ye));
      R->MultTranspose(ye, y);
      y -= y_ref;
      double diff = y.Normlinf(), scale = y_ref.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_GT(scale, 0.0);
      EXPECT_LE(diff, 1e-13 * scale);
   }
}

TEST(LeanVectorMass, MatchesMfem2D)
{
   Check(2, 3, 1.0);
   Check(2, 3, 0.7);
}
