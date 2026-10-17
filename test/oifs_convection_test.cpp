// incns::OifsConvection (time/oifs_convection): sum_l w_l C(u_l) x with the
// winds combined at each quadrature point and all components in one kernel,
// against what OifsAdvector computed before it -- MFEM's ConvectionIntegrator
// (partial assembly, alpha 1) on the scalar space, one apply per component and
// wind, summed. 3D, velocity orders 2 to 7 at OIFS's 3k rule, a deformed mesh,
// three random winds (and two), random weights. Orders 6 and 7 on
// straight-sided elements: the stored path and the on-the-fly path against the
// same reference; order 6 also on curved elements (their Jacobians formed by
// element chunks).
// Roundoff.

#include <gtest/gtest.h>

#include "time/oifs_convection.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using namespace mfem;

namespace
{

void Deform(const Vector& x, Vector& y)
{
   y = x;
   y(0) += 0.08 * std::sin(2.3 * x(1));
   y(1) += 0.06 * x(0) * x(0);
   y(2) += 0.05 * x(0) * x(1);
}

} // namespace

TEST(OifsConvection, MatchesMfemFormsComponentwise)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (const int k : {2, 3, 4, 5, 6, 7})
   {
      // Straight-sided elements with three and two winds; curved elements
      // (the stored path keeping the Jacobians) with two, at order 6 only:
      // the reference's legacy assembly at orders 6 and 7 is the test's cost.
      for (const auto& [nw, curved] : std::vector<std::pair<int, bool>>
   {{3, false}, {2, false}, {2, true}})
      {
         if (curved && k != 6) { continue; }
         SCOPED_TRACE(testing::Message() << "order " << k << ", " << nw
                      << " winds" << (curved ? ", curved" : ""));
         Mesh serial = Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
         if (curved) { serial.SetCurvature(2); }
         serial.Transform(Deform);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(k, 3);
         ParFiniteElementSpace V(&mesh, &fec, 3), S(&mesh, &fec);
         const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 3 * k);
         ASSERT_TRUE(incns::OifsConvection::Supported(V, ir));
         const bool eligible = incns::OifsConvection::OnTheFlyEligible(V, ir);
         ASSERT_EQ(eligible, k >= 6 && !curved);

         const int n_s = S.GetTrueVSize();
         ASSERT_EQ(V.GetTrueVSize(), 3 * n_s);
         std::vector<std::unique_ptr<ParGridFunction>> u(nw);
         std::vector<std::unique_ptr<VectorGridFunctionCoefficient>> uc(nw);
         std::vector<std::unique_ptr<ParBilinearForm>> forms(nw);
         std::vector<OperatorPtr> ops(nw);
         Array<int> none;
         for (int l = 0; l < nw; ++l)
         {
            Vector ut(V.GetTrueVSize());
            ut.Randomize(11 + 7 * l + 3 * rank);
            ut -= 0.5;
            u[l] = std::make_unique<ParGridFunction>(&V);
            u[l]->SetFromTrueDofs(ut);
            uc[l] = std::make_unique<VectorGridFunctionCoefficient>(u[l].get());
            forms[l] = std::make_unique<ParBilinearForm>(&S);
            auto* ci = new ConvectionIntegrator(*uc[l], 1.0);
            ci->SetIntRule(&ir);
            forms[l]->AddDomainIntegrator(ci);
            // Orders 6 and 7 (Q1D 10, 11): MFEM has no compile-time
            // convection kernel there, and at Q1D 11 its partial assembly does
            // not launch on a GPU (a thread per point); the reference is
            // assembled on the host (legacy assembly, its wind read there).
            if (k >= 6)
            {
               u[l]->HostRead();
               forms[l]->SetAssemblyLevel(AssemblyLevel::LEGACY);
            }
            else { forms[l]->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
            forms[l]->Assemble();
            forms[l]->FormSystemMatrix(none, ops[l]);
         }
         const std::vector<double> w_all = {0.7, -1.3, 0.4};
         const std::vector<double> w(w_all.begin(), w_all.begin() + nw);
         Vector x(V.GetTrueVSize());
         x.Randomize(5 + rank);
         x.UseDevice(true);

         // The reference, component by component and wind by wind (host
         // copies of the components: no aliases).
         std::vector<double> ref(V.GetTrueVSize(), 0.0);
         const double* xh = x.HostRead();
         Vector xc(n_s), t(n_s);
         xc.UseDevice(true);
         t.UseDevice(true);
         for (int c = 0; c < 3; ++c)
         {
            double* d = xc.HostWrite();
            for (int i = 0; i < n_s; ++i) { d[i] = xh[c * n_s + i]; }
            for (int l = 0; l < nw; ++l)
            {
               ops[l]->Mult(xc, t);
               const double* th = t.HostRead();
               for (int i = 0; i < n_s; ++i) { ref[c * n_s + i] += w[l] * th[i]; }
            }
         }

         // The stored path, and the on-the-fly path where eligible, against
         // the same reference.
         for (const bool fly : {false, true})
         {
            if (fly && !eligible) { continue; }
            incns::OifsConvection oc(V, ir, nullptr, fly);
            for (int l = 0; l < nw; ++l) { oc.SetWind(l, *u[l]); }
            Vector y(V.GetTrueVSize());
            y.UseDevice(true);
            oc.Mult(w, x, y);
            const double* yh = y.HostRead();
            double diff = 0.0, size = 0.0;
            for (int i = 0; i < V.GetTrueVSize(); ++i)
            {
               diff = std::max(diff, std::abs(yh[i] - ref[i]));
               size = std::max(size, std::abs(ref[i]));
            }
            MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            if (Mpi::Root())
            {
               mfem::out << "[oifs convection] order " << k << ", " << nw
                         << " winds" << (curved ? ", curved" : "")
                         << (fly ? ", on the fly" : "") << ": max difference "
                         << diff << ", relative " << diff / size << std::endl;
            }
            EXPECT_GT(size, 0.0);
            EXPECT_LE(diff, 1e-12 * size);
         }
      }
   }
}

// Curved elements at orders 6 and 7 (node order = velocity order, as the
// cylinder meshes have them): the Jacobians are formed chunk by chunk, so the
// results must not depend on the chunk size, bitwise; and the self-only apply
// (Convection's) must equal the stored path with the input as its one wind.
TEST(OifsConvection, CurvedChunksAndTheSelfApply)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (const int k : {6, 7})
   {
      SCOPED_TRACE(testing::Message() << "order " << k);
      Mesh serial = Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
      serial.SetCurvature(k);
      serial.Transform(Deform);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(k, 3);
      ParFiniteElementSpace V(&mesh, &fec, 3);
      const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 3 * k);
      ASSERT_TRUE(incns::OifsConvection::Supported(V, ir));
      ASSERT_FALSE(incns::OifsConvection::OnTheFlyEligible(V, ir));
      const int n = V.GetTrueVSize();
      Vector x(n), u2(n);
      x.Randomize(17 + rank);
      x -= 0.5;
      u2.Randomize(29 + rank);
      u2 -= 0.5;
      x.UseDevice(true);
      ParGridFunction xg(&V), ug(&V);
      xg.SetFromTrueDofs(x);
      ug.SetFromTrueDofs(u2);
      auto max_diff = [](const Vector & a, const Vector & b)
      {
         Vector d(a);
         d -= b;
         double m = d.Normlinf();
         MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         return m;
      };
      // The stored path with two winds: one chunk, then chunks of 5 elements.
      const std::vector<double> w = {0.7, -1.3};
      Vector y1(n), y5(n);
      y1.UseDevice(true);
      y5.UseDevice(true);
      for (const int chunk : {0, 5})
      {
         incns::OifsConvection oc(V, ir);
         if (chunk) { oc.SetChunk(chunk); }
         oc.SetWind(0, xg);
         oc.SetWind(1, ug);
         oc.Mult(w, x, chunk ? y5 : y1);
      }
      EXPECT_GT(y1.Normlinf(), 0.0);
      EXPECT_EQ(max_diff(y1, y5), 0.0);
      // The self apply against the stored path with x as its one wind.
      Vector ys(n), yself(n), yself5(n);
      ys.UseDevice(true);
      yself.UseDevice(true);
      yself5.UseDevice(true);
      {
         incns::OifsConvection oc(V, ir);
         oc.SetWind(0, xg);
         oc.Mult({1.0}, x, ys);
      }
      for (const int chunk : {0, 5})
      {
         incns::OifsConvection oc(V, ir, nullptr, false, /*self_only=*/true);
         if (chunk) { oc.SetChunk(chunk); }
         oc.MultSelf(x, chunk ? yself5 : yself);
      }
      EXPECT_EQ(max_diff(ys, yself), 0.0);
      EXPECT_EQ(max_diff(ys, yself5), 0.0);
   }
}
