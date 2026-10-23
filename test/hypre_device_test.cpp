// precond/hypre_device: BoomerAMG stays on hypre's fast GPU path.
//  D1 MFEM's default options pass the check on any build.
//  D2 the symmetric smoother is 8 on a host and 18 on a GPU, and passes; on a
//     GPU type 8 (triangular solves) and an unsupported coarsening are
//     reported, on a host nothing ever is;
//  D3 HypreDiagonal of a boundary mass (empty rows off the faces; at np > 1 a
//     rank without the faces) is the summed local diagonals, zero off the
//     faces.

#include <gtest/gtest.h>

#include "precond/hypre_device.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

using incns::HypreAmgDeviceProblems;

TEST(HypreDevice, D1_MfemDefaultsPass)
{
   mfem::HypreBoomerAMG amg;
   EXPECT_EQ(HypreAmgDeviceProblems(amg), "");
}

TEST(HypreDevice, D2_SymmetricSmootherAndWhatIsReported)
{
   const bool gpu = mfem::HypreUsingGPU();
   EXPECT_EQ(incns::SymmetricAmgSmoother(), gpu ? 18 : 8);

   mfem::HypreBoomerAMG amg;
   HYPRE_Solver h = static_cast<HYPRE_Solver>(amg);
   HYPRE_BoomerAMGSetRelaxType(h, incns::SymmetricAmgSmoother());
   EXPECT_EQ(HypreAmgDeviceProblems(amg), "");

   HYPRE_BoomerAMGSetRelaxType(h, 8); // l1 symmetric Gauss-Seidel
   const std::string gs = HypreAmgDeviceProblems(amg);
   if (gpu)
   {
      EXPECT_NE(gs.find("smoother type 8"), std::string::npos) << gs;
      EXPECT_NE(gs.find("triangular"), std::string::npos) << gs;
   }
   else
   {
      EXPECT_EQ(gs, "");
   }

   HYPRE_BoomerAMGSetRelaxType(h, 18);
   HYPRE_BoomerAMGSetCoarsenType(h, 10); // HMIS: the host default
   const std::string hmis = HypreAmgDeviceProblems(amg);
   if (gpu) { EXPECT_NE(hmis.find("coarsening type 10"), std::string::npos) << hmis; }
   else { EXPECT_EQ(hmis, ""); }
}

TEST(HypreDevice, D3_DiagonalOfABoundaryMass)
{
   using namespace mfem;
   // The right end of a long strip: at np > 1 some ranks hold none of it.
   Mesh serial = Mesh::MakeCartesian2D(16, 2, Element::QUADRILATERAL, true, 8.0,
                                       1.0);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&mesh, &fec, 2, Ordering::byNODES);
   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[1] = 1; // x = 8 (MakeCartesian2D: 1 bottom, 2 right, 3 top, 4 left)
   ConstantCoefficient k(3.0);
   ParBilinearForm s(&fes);
   s.AddBoundaryIntegrator(new VectorMassIntegrator(k), marker);
   s.Assemble();
   s.Finalize();
   std::unique_ptr<HypreParMatrix> S(s.ParallelAssemble());

   Vector d;
   incns::HypreDiagonal(*S, d);
   // The reference: the local matrix's diagonal (SparseMatrix::GetDiag finds
   // the entry, zero where none), summed over the ranks sharing a dof.
   Vector dl, ref(fes.GetTrueVSize());
   s.SpMat().GetDiag(dl);
   fes.GetProlongationMatrix()->MultTranspose(dl, ref);
   const real_t* a = d.HostRead();
   const real_t* b = ref.HostRead();
   double m[2] = {0.0, 0.0};
   int zeros = 0;
   for (int i = 0; i < d.Size(); ++i)
   {
      m[0] = std::max(m[0], std::abs(a[i] - b[i]));
      m[1] = std::max(m[1], std::abs(b[i]));
      zeros += b[i] == 0.0;
   }
   MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &zeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   ASSERT_GT(m[1], 0.0);
   EXPECT_GT(zeros, 0) << "no row off the faces: the test would prove nothing";
   EXPECT_LE(m[0], 1e-14 * m[1]);

   // And the matrix is still usable on the device after the host read.
   Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize());
   x.UseDevice(true);
   y.UseDevice(true);
   x = 1.0;
   S->Mult(x, y);
   EXPECT_TRUE(std::isfinite(InnerProduct(MPI_COMM_WORLD, y, y)));
}
