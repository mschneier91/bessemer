// AMR.1 -- the directional gradient indicator G_{K,d} (amr/gradient_indicator).
//  N1 linear field u = A x on an affine (sheared) mesh: G_{K,d} = |A J_{:,d}|
//     exactly, on a conforming mesh and on an anisotropically refined NC mesh
//     (which also pins the halving under refinement: J_child = J_parent with
//     the split column halved);
//  N2 direction blindness: u varying in x only gives G_{K,y} = G_{K,z} = 0;
//  N3 any field on a CURVED NC mesh: the device kernel equals an independent
//     host reference built from MFEM's GridFunction::GetVectorGradient at the
//     same quadrature points.

#include <gtest/gtest.h>

#include "amr/gradient_indicator.hpp"
#include "amr_test_util.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>

using namespace mfem;
using incns::GradientIndicator;
using incns::RuleBook;

namespace
{

// Max-abs global reduction.
double GlobalMax(double v)
{
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return v;
}

const double kA[3][3] = {{1.0, 2.0, -0.5}, {-3.0, 0.5, 0.7}, {0.4, -1.2, 2.0}};
const double kShear[3][3] = {{1.0, 0.3, 0.1}, {0.2, 1.1, 0.0}, {0.1, -0.2, 0.9}};

// A sheared (affine) box, partitioned NC-ready.
std::unique_ptr<ParMesh> ShearedBox(int dim, int n)
{
   Mesh serial = incns::MakeBoxMesh(amr_test::Box(dim, n, false));
   serial.Transform([dim](const Vector & x, Vector & y)
   {
      y.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         y(i) = 0.0;
         for (int j = 0; j < dim; ++j) { y(i) += kShear[i][j] * x(j); }
      }
   });
   return incns::PartitionMesh(serial, true);
}

} // namespace

TEST(AmrIndicator, N1_LinearFieldOnAffineMeshIsExact)
{
   RuleBook rules; // outlives the meshes
   for (int dim : {2, 3})
      for (bool refine : {false, true})
      {
         SCOPED_TRACE("dim=" + std::to_string(dim) + (refine ? " refined" :
                      " conforming"));
         auto mesh = ShearedBox(dim, dim == 2 ? 3 : 2);
         if (refine) { amr_test::RefineBands(*mesh, true); }
         H1_FECollection fec(2, dim);
         ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
         ParGridFunction u(&fes);
         VectorFunctionCoefficient uc(dim, [dim](const Vector & x, Vector & v)
         {
            for (int i = 0; i < dim; ++i)
            {
               v(i) = 0.0;
               for (int j = 0; j < dim; ++j) { v(i) += kA[i][j] * x(j); }
            }
         });
         u.ProjectCoefficient(uc);

         GradientIndicator ind(fes, rules);
         Vector g;
         ind.Compute(u, g);
         const double* G = g.HostRead();

         double err = 0.0, scale = 0.0;
         IntegrationPoint center;
         center.Set3(0.5, 0.5, 0.5);
         for (int e = 0; e < mesh->GetNE(); ++e)
         {
            ElementTransformation& T = *mesh->GetElementTransformation(e);
            T.SetIntPoint(&center);
            const DenseMatrix& J = T.Jacobian();
            for (int d = 0; d < dim; ++d)
            {
               double s = 0.0;
               for (int i = 0; i < dim; ++i)
               {
                  double a = 0.0;
                  for (int j = 0; j < dim; ++j) { a += kA[i][j] * J(j, d); }
                  s += a * a;
               }
               const double ref = std::sqrt(s);
               err = std::max(err, std::fabs(G[d + dim * e] - ref));
               scale = std::max(scale, ref);
            }
         }
         EXPECT_LE(GlobalMax(err), 1e-12 * GlobalMax(scale));
      }
}

TEST(AmrIndicator, N2_DirectionBlindness)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      auto mesh = amr_test::NcBox(amr_test::Box(dim, dim == 2 ? 3 : 2, false));
      amr_test::RefineBands(*mesh, true);
      H1_FECollection fec(3, dim);
      ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
      ParGridFunction u(&fes);
      VectorFunctionCoefficient uc(dim, [](const Vector & x, Vector & v)
      {
         v = 0.0;
         v(0) = x(0) * x(0);
         v(1) = x(0) * x(0) * x(0) - x(0);
      });
      u.ProjectCoefficient(uc);
      GradientIndicator ind(fes, rules);
      Vector g;
      ind.Compute(u, g);
      const double* G = g.HostRead();
      double gx = 0.0, gyz = 0.0;
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         gx = std::max(gx, G[dim * e]);
         for (int d = 1; d < dim; ++d) { gyz = std::max(gyz, std::fabs(G[d + dim * e])); }
      }
      EXPECT_GT(GlobalMax(gx), 0.0);
      EXPECT_LE(GlobalMax(gyz), 1e-14 * GlobalMax(gx));
   }
}

TEST(AmrIndicator, N3_MatchesHostReferenceOnCurvedNcMesh)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      const int p = (dim == 2) ? 3 : 2;
      Mesh serial = incns::MakeBoxMesh(amr_test::Box(dim, dim == 2 ? 4 : 3,
                                       false));
      serial.SetCurvature(p);
      serial.Transform([](const Vector & x, Vector & y)
      {
         y = x;
         y(0) += 0.05 * std::sin(M_PI * x(1));
         y(1) += 0.05 * std::sin(M_PI * x(0));
         if (x.Size() == 3) { y(2) += 0.05 * std::sin(M_PI * x(0)); }
      });
      auto mesh = incns::PartitionMesh(serial, true);
      amr_test::RefineBands(*mesh, true);
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
      ParGridFunction u(&fes);
      VectorFunctionCoefficient uc(dim, [](const Vector & x, Vector & v)
      {
         const double X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
         v(0) = std::sin(2.0 * X) * std::cos(Y) + Z;
         v(1) = std::exp(0.5 * Y) * X;
         if (v.Size() == 3) { v(2) = std::cos(X * Z) + Y * Y; }
      });
      u.ProjectCoefficient(uc);
      {
         Vector t;
         u.GetTrueDofs(t);
         u.SetFromTrueDofs(t);
      }
      GradientIndicator ind(fes, rules);
      Vector g;
      ind.Compute(u, g);
      const double* G = g.HostRead();

      // Host reference: sum_q w_q detJ_q sum_c ((grad_x u) J)_{c,d}^2 / |K|.
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const IntegrationRule& ir = rules.Get(geom, 2 * p + dim - 1);
      u.HostRead();
      DenseMatrix grad, gj;
      double err = 0.0, scale = 0.0;
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         ElementTransformation& T = *mesh->GetElementTransformation(e);
         double vol = 0.0, s[3] = {0.0, 0.0, 0.0};
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint& ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            u.GetVectorGradient(T, grad); // grad(c, k) = du_c/dx_k
            gj.SetSize(dim);
            Mult(grad, T.Jacobian(), gj);  // du_c/dxi_d
            const double wq = ip.weight * T.Weight();
            vol += wq;
            for (int d = 0; d < dim; ++d)
               for (int c = 0; c < dim; ++c) { s[d] += wq * gj(c, d) * gj(c, d); }
         }
         for (int d = 0; d < dim; ++d)
         {
            const double ref = std::sqrt(s[d] / vol);
            err = std::max(err, std::fabs(G[d + dim * e] - ref));
            scale = std::max(scale, ref);
         }
      }
      EXPECT_LE(GlobalMax(err), 1e-12 * GlobalMax(scale));

      // eta = |G_K| per element.
      Vector eta;
      GradientIndicator::Eta(g, dim, eta);
      ASSERT_EQ(eta.Size(), mesh->GetNE());
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         double s2 = 0.0;
         for (int d = 0; d < dim; ++d) { s2 += G[d + dim * e] * G[d + dim * e]; }
         EXPECT_NEAR(eta(e), std::sqrt(s2), 1e-14 * (1.0 + eta(e)));
      }
   }
}
