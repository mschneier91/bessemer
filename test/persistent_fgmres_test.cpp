// incns::PersistentFGMRESSolver (solver/persistent_fgmres): the same iterates
// as mfem::FGMRESSolver -- a sequence of solves of a nonsymmetric
// (convection-diffusion-reaction) system with a Jacobi preconditioner,
// different right-hand sides and initial guesses, without and with restarts:
// the same iteration counts and the same solutions (bitwise on the CPU), with
// the work vectors kept between the solves and released on request.

#include <gtest/gtest.h>

#include "solver/persistent_fgmres.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace mfem;

TEST(PersistentFgmres, SameIteratesAsMfem)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Mesh serial = Mesh::MakeCartesian2D(8, 6, Element::QUADRILATERAL, true);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(2, 2);
   ParFiniteElementSpace fes(&mesh, &fec);
   ConstantCoefficient nu(0.05);
   Vector wv(2);
   wv(0) = 1.0;
   wv(1) = 0.4;
   VectorConstantCoefficient w(wv);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(nu));
   a.AddDomainIntegrator(new ConvectionIntegrator(w, 1.0));
   a.AddDomainIntegrator(new MassIntegrator());
   a.Assemble();
   a.Finalize();
   std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
   HypreSmoother jacobi(*A, HypreSmoother::Jacobi);
   const int n = A->Height();

   for (const int kdim : {50, 4}) // without and with restarts
   {
      SCOPED_TRACE(testing::Message() << "kdim " << kdim);
      FGMRESSolver ref(MPI_COMM_WORLD);
      incns::PersistentFGMRESSolver kept(MPI_COMM_WORLD);
      for (FGMRESSolver* s : {&ref, static_cast<FGMRESSolver*>(&kept)})
      {
         s->SetOperator(*A);
         s->SetPreconditioner(jacobi);
         s->SetRelTol(1e-10);
         s->SetAbsTol(0.0);
         s->SetMaxIter(300);
         s->SetKDim(kdim);
         s->SetPrintLevel(0);
         s->iterative_mode = true;
      }
      for (int solve = 0; solve < 4; ++solve)
      {
         Vector b(n), x0(n);
         b.Randomize(10 + solve + 7 * rank);
         x0.Randomize(20 + solve + 7 * rank);
         x0 *= (solve == 2) ? 0.0 : 0.3;
         Vector x_ref(x0), x_kept(x0);
         b.UseDevice(true);
         x_ref.UseDevice(true);
         x_kept.UseDevice(true);
         ref.Mult(b, x_ref);
         kept.Mult(b, x_kept);
         // FGMRES(4) with Jacobi stagnates on this system: equal, not converged.
         if (kdim == 50) { EXPECT_TRUE(ref.GetConverged()); }
         EXPECT_EQ(ref.GetNumIterations(), kept.GetNumIterations());
         EXPECT_EQ(ref.GetConverged(), kept.GetConverged());
         const double* xr = x_ref.HostRead();
         const double* xk = x_kept.HostRead();
         double diff = 0.0, size = 0.0;
         for (int i = 0; i < n; ++i)
         {
            diff = std::max(diff, std::abs(xr[i] - xk[i]));
            size = std::max(size, std::abs(xr[i]));
         }
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         if (Mpi::Root())
         {
            mfem::out << "[persistent fgmres] kdim " << kdim << " solve " << solve
                      << ": " << ref.GetNumIterations() << " iterations, max "
                      << "difference " << diff << " (of " << size << "), kept "
                      << kept.KeptBytes() / 1024.0 << " KiB" << std::endl;
         }
         // Bitwise on the CPU; on a GPU the same kernels on the same data.
         EXPECT_LE(diff, 1e-14 * size);
         EXPECT_GT(kept.KeptBytes(), 0.0);
      }
      kept.ReleaseWork();
      EXPECT_EQ(kept.KeptBytes(), 0.0);
   }
}
