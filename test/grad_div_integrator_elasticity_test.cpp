// GradDivIntegrator vs MFEM's ElasticityIntegrator(lambda = q, mu = 0) -- the
// lambda part of linear elasticity IS grad-div, so the two must agree:
//  - element matrices (legacy full assembly), element by element;
//  - partial assembly: applies and diagonals against elasticity's own PA,
//    with elasticity PA checked against its full assembly in the same run.
// Self-contained on purpose -- MFEM and the integrator header only, no
// bessemer test helpers.

#include <gtest/gtest.h>

#include "operators/grad_div_integrator.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::GradDivIntegrator;

TEST(GradDivIntegratorElasticity, ElementMatricesMatch)
{
   for (int dim : {2, 3})
   {
      for (int p = 1; p <= 3; ++p)
      {
         // Curved (non-affine) mesh, so the Jacobian varies from point to point.
         Mesh mesh = (dim == 2)
                     ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
                     : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
         mesh.SetCurvature(p);
         mesh.Transform([](const Vector & x, Vector & y)
         {
            y = x;
            y(0) += 0.1 * std::sin(M_PI * x(1));
            y(1) += 0.1 * std::sin(M_PI * x(0));
            if (x.Size() == 3) { y(2) += 0.1 * std::sin(M_PI * x(0)); }
         });
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec); // scalar: elmats are per element

         // Spatially varying coefficient; the same rule for both integrators.
         FunctionCoefficient q([](const Vector & x) { return 1.0 + x(0) * x(1); });
         ConstantCoefficient zero(0.0);
         GradDivIntegrator divdiv(q);
         ElasticityIntegrator elasticity(q, zero); // lambda = q, mu = 0
         const IntegrationRule& ir =
            IntRules.Get(fes.GetFE(0)->GetGeomType(), 2 * p + dim - 1);
         divdiv.SetIntRule(&ir);
         elasticity.SetIntRule(&ir);

         for (int e = 0; e < mesh.GetNE(); ++e)
         {
            DenseMatrix A, B;
            divdiv.AssembleElementMatrix(*fes.GetFE(e),
                                         *mesh.GetElementTransformation(e), A);
            elasticity.AssembleElementMatrix(*fes.GetFE(e),
                                             *mesh.GetElementTransformation(e), B);
            B -= A;
            EXPECT_LE(B.MaxMaxNorm(), 1e-12 * A.MaxMaxNorm())
                  << "dim=" << dim << " p=" << p << " element=" << e;
         }
      }
   }
}

// Partial assembly against elasticity's partial assembly -- two independent
// matrix-free implementations of the same operator, applies and diagonals.
// The baseline is checked in the same run: elasticity PA must also match
// elasticity's own full assembly, so if MFEM's elasticity PA were wrong the
// failure says which side it is on. Curved partitioned mesh, varying
// coefficient, dim 2/3, p 1-3, one shared rule.
TEST(GradDivIntegratorElasticity,
     PartialAssemblyMatchesElasticityPartialAssembly)
{
   for (int dim : {2, 3})
   {
      for (int p = 1; p <= 3; ++p)
      {
         Mesh serial = (dim == 2)
                       ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
                       : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
         serial.SetCurvature(p);
         serial.Transform([](const Vector & x, Vector & y)
         {
            y = x;
            y(0) += 0.1 * std::sin(M_PI * x(1));
            y(1) += 0.1 * std::sin(M_PI * x(0));
            if (x.Size() == 3) { y(2) += 0.1 * std::sin(M_PI * x(0)); }
         });
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);

         FunctionCoefficient q([](const Vector & x) { return 1.0 + x(0) * x(1); });
         ConstantCoefficient zero(0.0);
         const IntegrationRule& ir =
            IntRules.Get(fes.GetFE(0)->GetGeomType(), 2 * p + dim - 1);

         // gd_pa: this integrator, PA; el_pa: elasticity (lambda = q, mu = 0),
         // PA; el_fa: elasticity, full assembly (the baseline's own check).
         ParBilinearForm gd_pa(&fes), el_pa(&fes), el_fa(&fes);
         auto* gd = new GradDivIntegrator(q);
         auto* ep = new ElasticityIntegrator(q, zero);
         auto* ef = new ElasticityIntegrator(q, zero);
         gd->SetIntRule(&ir);
         ep->SetIntRule(&ir);
         ef->SetIntRule(&ir);
         gd_pa.AddDomainIntegrator(gd);
         el_pa.AddDomainIntegrator(ep);
         el_fa.AddDomainIntegrator(ef);
         gd_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         el_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         gd_pa.Assemble();
         el_pa.Assemble();
         el_fa.Assemble();
         el_fa.Finalize();
         Array<int> empty;
         OperatorPtr A_gd, A_el, A_fa;
         gd_pa.FormSystemMatrix(empty, A_gd);
         el_pa.FormSystemMatrix(empty, A_el);
         el_fa.FormSystemMatrix(empty, A_fa);

         auto norm = [](const Vector & v)
         {
            return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
         };
         Vector x(fes.GetTrueVSize()), y_gd(x.Size()), y_el(x.Size()),
                y_fa(x.Size());
         x.Randomize(3);
         A_gd->Mult(x, y_gd);
         A_el->Mult(x, y_el);
         A_fa->Mult(x, y_fa);

         // Baseline sanity: elasticity PA == elasticity full assembly.
         Vector d(y_el);
         d -= y_fa;
         EXPECT_LE(norm(d), 1e-12 * norm(y_fa))
               << "MFEM's elasticity PA disagrees with its own full assembly: "
               << "the baseline is wrong, not this integrator (dim=" << dim
               << " p=" << p << ")";
         // The test: this integrator's PA == elasticity PA.
         d = y_gd;
         d -= y_el;
         EXPECT_LE(norm(d), 1e-12 * norm(y_el)) << "apply, dim=" << dim
                                                << " p=" << p;

         // Diagonals, the same two ways.
         Vector g_gd(x.Size()), g_el(x.Size());
         gd_pa.AssembleDiagonal(g_gd);
         el_pa.AssembleDiagonal(g_el);
         g_el -= g_gd;
         EXPECT_LE(norm(g_el), 1e-12 * norm(g_gd)) << "diagonal, dim=" << dim
               << " p=" << p;
      }
   }
}
