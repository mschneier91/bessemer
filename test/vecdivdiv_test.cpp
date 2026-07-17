// VDD -- VectorDivDivIntegrator validation (vecdivdiv_spec.md par.7):
//  7.2 full-assembly element matrices equal ElasticityIntegrator(lambda, mu=0)
//      on CURVED meshes, 2D+3D (locks the math before any PA work);
//  7.1 PA operator action equals the fully assembled action, dim x order x
//      coefficient sweep, curved meshes, <= 1e-12 relative;
//  7.3 the collocated-GLL rule (Q1D == D1D, the SEM configuration);
//  7.4 PA diagonal equals the assembled matrix diagonal;
//  7.5 transpose apply equals primal apply (symmetric forwarding).

#include <gtest/gtest.h>

#include "operators/vecdivdiv_integrator.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::Rule1D;
using incns::RuleBook;
using incns::VectorDivDivIntegrator;

namespace
{

// Curved, non-affine tensor mesh: Cartesian box, high-order nodes, then a
// smooth invertible perturbation so the Jacobian genuinely varies per point
// (the adjugate/detJ path is exercised, not just the affine special case).
Mesh MakeCurvedMesh(int dim, int n, int geom_order)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);
   mesh.SetCurvature(geom_order);
   mesh.Transform([](const Vector & x, Vector & p)
   {
      p = x;
      p(0) += 0.05 * std::sin(M_PI * x(0)) * std::cos(M_PI * x(1));
      p(1) += 0.04 * std::cos(M_PI * x(0)) * std::sin(M_PI * x(1));
      if (x.Size() == 3) { p(2) += 0.03 * std::sin(M_PI * x(2)); }
   });
   return mesh;
}

double Coef(const Vector& x)
{
   double v = 1.0 + 0.5 * std::sin(2.0 * x(0)) * std::cos(x(1));
   if (x.Size() == 3) { v += 0.25 * x(2); }
   return v;
}

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

} // namespace

// 7.2 -- element matrices equal the lambda-part of elasticity, forced same
// rule, curved 2D and 3D. Pins conventions (byNODES blocks, weights, adjugate).
TEST(VecDivDiv, ElmatMatchesElasticity)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const int p = (dim == 2) ? 3 : 2;
      Mesh mesh = MakeCurvedMesh(dim, 2, p);
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec); // scalar; elmat is per-element
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const IntegrationRule& ir = rules.Get(geom, 2 * p + dim - 1);

      FunctionCoefficient q(Coef);
      ConstantCoefficient zero(0.0);
      VectorDivDivIntegrator vdd(q);
      ElasticityIntegrator ela(q, zero); // lambda = q, mu = 0
      vdd.SetIntRule(&ir);
      ela.SetIntRule(&ir);

      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         DenseMatrix m_vdd, m_ela;
         vdd.AssembleElementMatrix(*fes.GetFE(e),
                                   *mesh.GetElementTransformation(e), m_vdd);
         ela.AssembleElementMatrix(*fes.GetFE(e),
                                   *mesh.GetElementTransformation(e), m_ela);
         m_ela -= m_vdd;
         EXPECT_LE(m_ela.MaxMaxNorm(), 1e-12 * m_vdd.MaxMaxNorm())
               << "dim=" << dim << " e=" << e;
      }
   }
}

// 7.1 + 7.3 + 7.5 -- PA action vs fully assembled action on curved parallel
// meshes: dim {2,3} x order sweep x coefficient {none, constant, function},
// with both the standard GL rule and the collocated GLL rule (Q1D == D1D).
TEST(VecDivDiv, PaMatchesFullAssembly)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int pmax = (dim == 2) ? 4 : 3;
      for (int p = 1; p <= pmax; ++p)
      {
         Mesh serial = MakeCurvedMesh(dim, (dim == 2) ? 3 : 2, p);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);

         FunctionCoefficient qfun(Coef);
         ConstantCoefficient qconst(0.7);
         for (int cv = 0; cv < 3; ++cv)
         {
            for (int gll = 0; gll < 2; ++gll)
            {
               // GL default rule vs collocated GLL (2p-1 -> p+1 = D1D points).
               const IntegrationRule& ir =
                  gll ? rules.Get(geom, 2 * p - 1, Rule1D::GaussLobatto)
                  : rules.Get(geom, 2 * p + dim - 1);
               auto make = [&]() -> VectorDivDivIntegrator *
               {
                  VectorDivDivIntegrator* i =
                  (cv == 0) ? new VectorDivDivIntegrator()
                  : (cv == 1) ? new VectorDivDivIntegrator(qconst)
                  : new VectorDivDivIntegrator(qfun);
                  i->SetIntRule(&ir);
                  return i;
               };

               ParBilinearForm pa(&fes), fa(&fes);
               pa.AddDomainIntegrator(make());
               fa.AddDomainIntegrator(make());
               pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               pa.Assemble();
               fa.Assemble();
               fa.Finalize();
               Array<int> empty;
               OperatorPtr Apa, Afa;
               pa.FormSystemMatrix(empty, Apa);
               fa.FormSystemMatrix(empty, Afa);

               Vector x(fes.GetTrueVSize()), ypa(x.Size()), yfa(x.Size()),
                      yt(x.Size());
               x.Randomize(7 + p + cv);
               Apa->Mult(x, ypa);
               Afa->Mult(x, yfa);
               yfa -= ypa;
               EXPECT_LE(Norm(yfa), 1e-12 * Norm(ypa))
                     << "dim=" << dim << " p=" << p << " coeff=" << cv
                     << " gll=" << gll;

               // 7.5: symmetric transpose forwarding through the PA operator.
               Apa->MultTranspose(x, yt);
               yt -= ypa;
               EXPECT_LE(Norm(yt), 1e-13 * Norm(ypa));
            }
         }
      }
   }
}

// 7.4 -- PA diagonal equals the assembled matrix diagonal (2D and 3D, curved,
// function coefficient).
TEST(VecDivDiv, DiagonalMatchesAssembled)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int p = (dim == 2) ? 3 : 2;
      Mesh serial = MakeCurvedMesh(dim, 2, p);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
      const IntegrationRule& ir = rules.Get(geom, 2 * p + dim - 1);
      FunctionCoefficient q(Coef);

      ParBilinearForm pa(&fes), fa(&fes);
      auto* ipa = new VectorDivDivIntegrator(q);
      auto* ifa = new VectorDivDivIntegrator(q);
      ipa->SetIntRule(&ir);
      ifa->SetIntRule(&ir);
      pa.AddDomainIntegrator(ipa);
      fa.AddDomainIntegrator(ifa);
      pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      pa.Assemble();
      fa.Assemble();
      fa.Finalize();

      Vector d_pa(fes.GetTrueVSize());
      pa.AssembleDiagonal(d_pa);

      Array<int> empty;
      OperatorPtr Afa;
      fa.FormSystemMatrix(empty, Afa);
      Vector d_fa(fes.GetTrueVSize());
      Afa.As<HypreParMatrix>()->GetDiag(d_fa);

      d_fa -= d_pa;
      EXPECT_LE(Norm(d_fa), 1e-12 * Norm(d_pa)) << "dim=" << dim;
   }
}
