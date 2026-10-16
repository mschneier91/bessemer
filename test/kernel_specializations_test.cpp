// incns::AddKernelSpecializations (operators/kernel_specializations): MFEM's
// vector mass, vector divergence (and transpose) and nonlinear convection
// partial-assembly applies at the solver's sizes give the same results, to
// roundoff, after the
// compile-time-sized kernels are registered as before (MFEM's generic
// kernels): the same arithmetic, but with fixed loop bounds a compiler may
// vectorize or contract it differently (1.7e-17 absolute on the CPU build,
// 3D, 2026-10-15). In 3D the divergence kernels are bessemer's own (unrolled
// copies of MFEM's). The differences are printed. 2D and 3D, velocity orders
// 2 to 7, a deformed mesh. The generic
// results are taken first, before anything registers: this test has its own
// executable for that reason.

#include <gtest/gtest.h>

#include "operators/kernel_specializations.hpp"
#include "mfem.hpp"

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
   if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
}

// M u, B u, B^T q and (u . grad) u for random u, q (fixed seeds), velocity
// order k; the convection at the solver's dealiased rule (order 3k).
std::vector<Vector> Apply(int dim, int k)
{
   Mesh serial = (dim == 2)
                 ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                 : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
   serial.Transform(Deform);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fu(k, dim), fp(k - 1, dim);
   ParFiniteElementSpace V(&mesh, &fu, dim), P(&mesh, &fp);
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;

   ParBilinearForm m(&V);
   auto* mi = new VectorMassIntegrator;
   mi->SetIntRule(&IntRules.Get(geom, 2 * k));      // k + 1 points
   m.AddDomainIntegrator(mi);
   m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m.Assemble();
   ParMixedBilinearForm b(&V, &P);
   auto* bi = new VectorDivergenceIntegrator;
   bi->SetIntRule(&IntRules.Get(geom, 2 * k + 2));  // k + 2 points
   b.AddDomainIntegrator(bi);
   b.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   b.Assemble();

   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Vector u(V.GetVSize()), q(P.GetVSize());
   u.Randomize(3 + rank);
   q.Randomize(7 + rank);
   u.UseDevice(true);
   q.UseDevice(true);
   Vector mu(V.GetVSize()), bu(P.GetVSize()), btq(V.GetVSize());
   mu.UseDevice(true);
   bu.UseDevice(true);
   btq.UseDevice(true);
   m.Mult(u, mu);
   b.Mult(u, bu);
   b.MultTranspose(q, btq);
   mu.HostRead();
   bu.HostRead();
   btq.HostRead();

   const bool gpu = Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK);
   if (gpu && dim == 3 && 3 * k >= 21) { return {mu, bu, btq, Vector()}; }
   ParNonlinearForm n(&V);
   auto* ni = new VectorConvectionNLFIntegrator;
   ni->SetIntRule(&IntRules.Get(geom, 3 * k));
   n.AddDomainIntegrator(ni);
   n.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   n.Setup();
   Vector ut(V.GetTrueVSize()), cu(V.GetTrueVSize());
   ut.Randomize(11 + rank);
   ut.UseDevice(true);
   cu.UseDevice(true);
   n.Mult(ut, cu);
   cu.HostRead();
   return {mu, bu, btq, cu};
}

} // namespace

TEST(KernelSpecializations, SameResultsAsTheGenericKernels)
{
   const int dims[] = {2, 3}, orders[] = {2, 3, 4, 5, 6, 7};
   // The interpolator's kernels first: the generic and the specialized runs
   // then use the same (correct) mesh Jacobians at order 7's 9 points.
   incns::AddInterpolationKernels();
   std::vector<std::vector<Vector>> generic;
   for (int dim : dims)
   {
      for (int k : orders) { generic.push_back(Apply(dim, k)); }
   }
   incns::AddKernelSpecializations();
   incns::AddKernelSpecializations(); // idempotent
   int c = 0;
   for (int dim : dims)
   {
      for (int k : orders)
      {
         const std::vector<Vector> special = Apply(dim, k);
         const char* names[] = {"mass", "divergence", "divergence transpose",
                                "convection"
                               };
         for (int i = 0; i < 4; ++i)
         {
            // The convection at order 7 in 3D (rule 21, Q1D 11): MFEM's
            // partial assembly (its geometric factors) launches a thread per
            // point, more than a GPU block holds -- compared on the host only.
            if (i == 3 && special[i].Size() == 0) { continue; }
            SCOPED_TRACE(testing::Message() << "dim " << dim << " order " << k
                         << " " << names[i]);
            Vector d(special[i]);
            d -= generic[c][i];
            double diff = d.Normlinf(), size = generic[c][i].Normlinf();
            MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (Mpi::Root())
            {
               mfem::out << "[kernel specializations] dim " << dim << " order " << k
                         << " " << names[i] << ": max difference " << diff
                         << ", relative " << diff / size << std::endl;
            }
            EXPECT_GT(size, 0.0);
            EXPECT_LE(diff, 1e-14 * size);
         }
         ++c;
      }
   }
}

// The mesh Jacobians (GeometricFactors, through MFEM's quadrature
// interpolator) at 7 to 10 Gauss points per direction against the element
// transformations' (host), 2D and 3D, a deformed mesh: MFEM's runtime-size
// interpolation kernels hold 8 points per direction on a GPU and wrote past
// their shared arrays at 9 and 10 until AddKernelSpecializations registered
// those sizes (2026-10-16).
TEST(KernelSpecializations, GeometricFactorsAtManyPoints)
{
   incns::AddKernelSpecializations();
   for (const int dim : {2, 3})
   {
      for (const int n : {7, 8, 9, 10})
      {
         SCOPED_TRACE(testing::Message() << "dim " << dim << ", " << n
                      << " points");
         Mesh serial = (dim == 2)
                       ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                       : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
         serial.Transform(Deform);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
         const IntegrationRule& ir = IntRules.Get(geom, 2 * n - 1);
         ASSERT_EQ(ir.GetNPoints(), (dim == 3) ? n * n* n : n * n);
         const GeometricFactors* gf =
            mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
         const int nq = ir.GetNPoints(), ne = mesh.GetNE();
         const auto J = Reshape(gf->J.HostRead(), nq, dim, dim, ne);
         double diff = 0.0, size = 0.0;
         for (int e = 0; e < ne; ++e)
         {
            ElementTransformation* T = mesh.GetElementTransformation(e);
            for (int q = 0; q < nq; ++q)
            {
               T->SetIntPoint(&ir.IntPoint(q));
               const DenseMatrix& Jh = T->Jacobian();
               for (int i = 0; i < dim; ++i)
               {
                  for (int j = 0; j < dim; ++j)
                  {
                     diff = std::max(diff, std::abs(J(q, i, j, e) - Jh(i, j)));
                     size = std::max(size, std::abs(Jh(i, j)));
                  }
               }
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         if (Mpi::Root())
         {
            mfem::out << "[geometric factors] dim " << dim << ", " << n
                      << " points: max difference " << diff << std::endl;
         }
         EXPECT_GT(size, 0.0);
         EXPECT_LE(diff, 1e-13 * size);
      }
   }
}
