// incns::OifsConvection (time/oifs_convection): sum_l w_l C(u_l) x with the
// winds combined at each quadrature point and all components in one kernel,
// against what OifsAdvector computed before it -- MFEM's ConvectionIntegrator
// (partial assembly, alpha 1) on the scalar space, one apply per component and
// wind, summed. 3D, velocity orders 2 to 4 at OIFS's 3k rule, a deformed mesh,
// three random winds (and two), random weights. Roundoff.

#include <gtest/gtest.h>

#include "time/oifs_convection.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
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
   for (const int k : {2, 3, 4})
   {
      for (const int nw : {3, 2})
      {
         SCOPED_TRACE(testing::Message() << "order " << k << ", " << nw
                      << " winds");
         Mesh serial = Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
         serial.Transform(Deform);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(k, 3);
         ParFiniteElementSpace V(&mesh, &fec, 3), S(&mesh, &fec);
         const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 3 * k);
         ASSERT_TRUE(incns::OifsConvection::Supported(V, ir));
         incns::OifsConvection oc(V, ir);

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
            oc.SetWind(l, *u[l]);
            uc[l] = std::make_unique<VectorGridFunctionCoefficient>(u[l].get());
            forms[l] = std::make_unique<ParBilinearForm>(&S);
            auto* ci = new ConvectionIntegrator(*uc[l], 1.0);
            ci->SetIntRule(&ir);
            forms[l]->AddDomainIntegrator(ci);
            forms[l]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            forms[l]->Assemble();
            forms[l]->FormSystemMatrix(none, ops[l]);
         }
         const std::vector<double> w_all = {0.7, -1.3, 0.4};
         const std::vector<double> w(w_all.begin(), w_all.begin() + nw);
         Vector x(V.GetTrueVSize()), y(V.GetTrueVSize());
         x.Randomize(5 + rank);
         x.UseDevice(true);
         y.UseDevice(true);
         oc.Mult(w, x, y);

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
         const double* yh = y.HostRead();
         double diff = 0.0, size = 0.0;
         for (int i = 0; i < V.GetTrueVSize(); ++i)
         {
            diff = std::max(diff, std::abs(yh[i] - ref[i]));
            size = std::max(size, std::abs(ref[i]));
         }
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         if (Mpi::Root())
         {
            mfem::out << "[oifs convection] order " << k << ", " << nw
                      << " winds: max difference " << diff << ", relative "
                      << diff / size << std::endl;
         }
         EXPECT_GT(size, 0.0);
         EXPECT_LE(diff, 1e-12 * size);
      }
   }
}
