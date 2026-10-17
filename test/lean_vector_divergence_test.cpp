// LeanVectorDivergenceIntegrator against MFEM's VectorDivergenceIntegrator
// (both partially assembled, MFEM's through the kernels of
// kernel_specializations.cpp): the apply and the transpose apply on a
// non-affine 3D mesh at velocity orders 2 to 7 (both kernel variants:
// all components at once below 7 points per direction, one at a time from 7),
// with and without a constant coefficient, and MFEM's own path in 2D.

#include <gtest/gtest.h>

#include "operators/kernel_specializations.hpp"
#include "operators/lean_vector_divergence.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace mfem;
using incns::LeanVectorDivergenceIntegrator;
using incns::RuleBook;

namespace
{

// Straight-sided but not affine (vertices moved): det J and adj(J) vary
// inside every element. Linear nodes: MFEM's compile-time Jacobian kernels
// cover up to 10 points per direction for them (AddInterpolationKernels).
Mesh DistortedBox(int dim, int n)
{
   Mesh mesh = (dim == 3)
               ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0,
                                       1.0)
               : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                       1.0, 1.0);
   mesh.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      for (int d = 0; d < x.Size(); ++d)
      {
         y(d) += 0.07 * x((d + 1) % x.Size()) * x(d) * (1.0 - x(d));
      }
   });
   return mesh;
}

// max |a - b| / max |b| over all ranks.
double RelDiff(const Vector& a, const Vector& b)
{
   double num = 0.0, den = 0.0;
   const double* pa = a.HostRead();
   const double* pb = b.HostRead();
   for (int i = 0; i < a.Size(); ++i)
   {
      num = std::max(num, std::abs(pa[i] - pb[i]));
      den = std::max(den, std::abs(pb[i]));
   }
   MPI_Allreduce(MPI_IN_PLACE, &num, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &den, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return den > 0.0 ? num / den : num;
}

void Check(int dim, int k, double coeff)
{
   SCOPED_TRACE(testing::Message() << "dim " << dim << ", k " << k
                << ", coefficient " << coeff);
   incns::AddKernelSpecializations();
   Mesh serial = DistortedBox(dim, 2);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fu(k, dim), fp(k - 1, dim);
   ParFiniteElementSpace V(&mesh, &fu, dim), P(&mesh, &fp);
   RuleBook rules;
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = rules.Get(geom, 2 * k + 2);
   ConstantCoefficient q(coeff);

   auto build = [&](BilinearFormIntegrator * integ)
   {
      integ->SetIntRule(&ir);
      auto form = std::make_unique<ParMixedBilinearForm>(&V, &P);
      form->AddDomainIntegrator(integ);
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form->Assemble();
      return form;
   };
   auto ref = build(coeff == 1.0 ? new VectorDivergenceIntegrator
                    : new VectorDivergenceIntegrator(q));
   auto* lean_integ = coeff == 1.0 ? new LeanVectorDivergenceIntegrator
                      : new LeanVectorDivergenceIntegrator(q);
   auto lean = build(lean_integ);
   EXPECT_EQ(lean_integ->Lean(), dim == 3);

   Vector x(V.GetVSize()), y_ref(P.GetVSize()), y(P.GetVSize());
   Vector p(P.GetVSize()), z_ref(V.GetVSize()), z(V.GetVSize());
   for (Vector* v : {&x, &y_ref, &y, &p, &z_ref, &z}) { v->UseDevice(true); }
   x.Randomize(11 + mesh.GetMyRank());
   p.Randomize(23 + mesh.GetMyRank());
   ref->Mult(x, y_ref);
   lean->Mult(x, y);
   EXPECT_LE(RelDiff(y, y_ref), 1e-13) << "apply";
   ref->MultTranspose(p, z_ref);
   lean->MultTranspose(p, z);
   EXPECT_LE(RelDiff(z, z_ref), 1e-13) << "transpose apply";
}

} // namespace

TEST(LeanVectorDivergence, MatchesMfemAllComponentsAtOnce)
{
   for (int k = 2; k <= 4; ++k) { Check(3, k, 1.0); }
}

TEST(LeanVectorDivergence, MatchesMfemOneComponentAtATime)
{
   for (int k = 5; k <= 7; ++k) { Check(3, k, 1.0); }
}

TEST(LeanVectorDivergence, ConstantCoefficient)
{
   Check(3, 3, -0.7);
   Check(3, 6, -0.7);
}

TEST(LeanVectorDivergence, TwoDimensionsUseMfem) { Check(2, 3, 1.0); }
