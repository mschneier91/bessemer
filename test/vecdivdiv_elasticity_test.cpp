// VectorDivDivIntegrator vs ElasticityIntegrator(lambda = q, mu = 0): the
// grad-div element matrices must be identical, since the lambda part of linear
// elasticity IS grad-div. Self-contained on purpose -- MFEM and the integrator
// header only, no bessemer test helpers.

#include <gtest/gtest.h>

#include "operators/vecdivdiv_integrator.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::VectorDivDivIntegrator;

TEST(VecDivDivElasticity, ElementMatricesMatch)
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
         VectorDivDivIntegrator divdiv(q);
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
