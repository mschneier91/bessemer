// Point-block Jacobi with the grad-div term's nodal coupling
// (PointBlockJacobi::SetGradDiv, GradDivIntegrator::AddNodalOffDiagonalPA).
//  P1 At sampled nodes, the block PBJ inverts is the block of the operator
//     itself: with B_a read off the partially assembled operator column by
//     column and Z_a = PBJ applied to the same unit vectors, B_a Z_a = I to
//     roundoff. 2D and 3D, deformed mesh, variable gamma, byNODES and byVDIM,
//     without and with essential dofs (one component of a boundary node). The
//     sampled blocks do couple the components, or the test would be vacuous.
//  P2 As the preconditioner of CG on sigma M + gamma (div u, div v) with a
//     large gamma, PBJ needs no more iterations than scalar Jacobi.
//  P3 ReleaseBuildData frees the diagonal and the off-diagonals and leaves
//     Mult bit-identical; the accessors are then empty.
//  P4 A new diagonal on a kept preconditioner (StokesSolver's update at a
//     Delta-t change) gives the blocks of a fresh build with that diagonal:
//     Mult bit-identical.
// Self-contained: MFEM, the grad-div integrator and the preconditioner.

#include <gtest/gtest.h>

#include "operators/grad_div_integrator.hpp"
#include "precond/point_block_jacobi.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace mfem;
using incns::GradDivIntegrator;
using incns::PointBlockJacobi;

namespace
{

// A smoothly deformed box, so the Jacobian has off-diagonal entries.
void Deform(const Vector& x, Vector& y)
{
   y = x;
   y(0) += 0.08 * std::sin(2.3 * x(1));
   y(1) += 0.06 * x(0) * x(0);
   if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
}

std::unique_ptr<ParMesh> MakeMesh(int dim)
{
   Mesh serial = (dim == 2)
                 ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                 : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
   serial.Transform(Deform);
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

double GammaFn(const Vector& x)
{
   return 0.7 + 0.5 * x(0) + 0.3 * x(1) * x(1);
}

// sigma M + gamma (div u, div v), partially assembled, with its diagonal and
// a PBJ that knows the grad-div term.
struct Problem
{
   std::unique_ptr<ParMesh> mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ConstantCoefficient sigma;
   std::unique_ptr<Coefficient> gamma;
   ParBilinearForm form;
   GradDivIntegrator* gd = nullptr; // owned by form
   Array<int> ess;
   OperatorPtr A;
   Vector diag;

   Problem(int dim, Ordering::Type ordering, bool essential, double sigma_value,
           double constant_gamma)
      : mesh(MakeMesh(dim)), fec(3, dim), fes(mesh.get(), &fec, dim, ordering),
        sigma(sigma_value), form(&fes)
   {
      if (constant_gamma > 0.0)
      {
         gamma = std::make_unique<ConstantCoefficient>(constant_gamma);
      }
      else
      {
         gamma = std::make_unique<FunctionCoefficient>(GammaFn);
      }
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const IntegrationRule& ir = IntRules.Get(geom, 2 * 3 + dim - 1);
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form.AddDomainIntegrator(new VectorMassIntegrator(sigma));
      gd = new GradDivIntegrator(*gamma, &ir);
      form.AddDomainIntegrator(gd);
      form.Assemble();
      if (essential)
      {
         // Component 0 on boundary attribute 1: nodes with ONE essential
         // component, the case the block rule has to get right.
         Array<int> bdr(mesh->bdr_attributes.Max());
         bdr = 0;
         bdr[0] = 1;
         fes.GetEssentialTrueDofs(bdr, ess, 0);
      }
      form.FormSystemMatrix(ess, A);
      diag.SetSize(fes.GetTrueVSize());
      form.AssembleDiagonal(diag);
   }
};

void CheckBlocks(int dim, Ordering::Type ordering, bool essential)
{
   Problem p(dim, ordering, essential, 3.0, 0.0);
   PointBlockJacobi pbj(p.fes, p.ess);
   pbj.SetDiagonal(p.diag);
   pbj.SetGradDiv(*p.gd);

   const int size = p.fes.GetTrueVSize();
   const int n = size / dim;
   const bool by_vdim = (ordering == Ordering::byVDIM);
   const auto idx = [&](int c, int a) { return by_vdim ? a * dim + c : c * n + a; };
   int rank = 0, np = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &np);

   Vector x(size), y(size), z(size);
   double worst = 0.0, asym = 0.0;
   long long coupled = 0;
   const int samples = 5;
   // One rank probes at a time: a unit vector on one of its own true dofs
   // gives exact columns of the global matrix while the others hold zeros.
   // Every rank makes every (collective) operator apply.
   for (int r = 0; r < np; ++r)
   {
      for (int s = 0; s < samples; ++s)
      {
         const bool mine = (rank == r && n > 0);
         const int a = mine ? (s * (n - 1)) / (samples - 1) : 0;
         double B[3][3] = {{0.0}}, Z[3][3] = {{0.0}};
         for (int j = 0; j < dim; ++j)
         {
            // Host pointers on both sides: the unit vector is written and the
            // results are read where the memory manager is told about it.
            double* xh = x.HostWrite();
            for (int k = 0; k < size; ++k) { xh[k] = 0.0; }
            if (mine) { xh[idx(j, a)] = 1.0; }
            p.A->Mult(x, y);
            pbj.Mult(x, z);
            if (!mine) { continue; }
            const double* yh = y.HostRead();
            const double* zh = z.HostRead();
            for (int i = 0; i < dim; ++i)
            {
               B[i][j] = yh[idx(i, a)];
               Z[i][j] = zh[idx(i, a)];
            }
         }
         if (!mine) { continue; }
         if (s == 1) // one block in the log, for a failure's diagnosis
         {
            for (int i = 0; i < dim; ++i)
            {
               mfem::out << "[pbj grad-div] node " << a << " B row " << i << ":";
               for (int j = 0; j < dim; ++j) { mfem::out << ' ' << B[i][j]; }
               mfem::out << "  Z row:";
               for (int j = 0; j < dim; ++j) { mfem::out << ' ' << Z[i][j]; }
               mfem::out << std::endl;
            }
         }
         for (int i = 0; i < dim; ++i)
         {
            for (int j = 0; j < dim; ++j)
            {
               double acc = 0.0;
               for (int k = 0; k < dim; ++k) { acc += B[i][k] * Z[k][j]; }
               worst = std::max(worst, std::abs(acc - (i == j ? 1.0 : 0.0)));
               asym = std::max(asym, std::abs(B[i][j] - B[j][i]));
               if (i != j && std::abs(B[i][j]) > 1e-6) { ++coupled; }
            }
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &asym, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &coupled, 1, MPI_LONG_LONG, MPI_SUM,
                 MPI_COMM_WORLD);
   EXPECT_LT(worst, 1e-11);
   EXPECT_LT(asym, 1e-12);
   EXPECT_GT(coupled, 0);
}

int CgIterations(const Operator& A, Solver& pc, const Vector& b)
{
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(A);
   cg.SetPreconditioner(pc);
   cg.SetRelTol(1e-8);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(5000);
   cg.SetPrintLevel(-1);
   cg.iterative_mode = false;
   Vector x(b.Size());
   x = 0.0;
   cg.Mult(b, x);
   EXPECT_TRUE(cg.GetConverged());
   return cg.GetNumIterations();
}

} // namespace

TEST(PbjGradDiv, P1_BlocksAreTheOperatorsNodalBlocks)
{
   for (const int dim : {2, 3})
   {
      for (const Ordering::Type ordering : {Ordering::byNODES, Ordering::byVDIM})
      {
         for (const bool essential : {false, true})
         {
            SCOPED_TRACE(testing::Message() << "dim " << dim << (ordering ==
                         Ordering::byVDIM ? " byVDIM" : " byNODES")
                         << (essential ? " essential" : ""));
            CheckBlocks(dim, ordering, essential);
         }
      }
   }
}

TEST(PbjGradDiv, P2_NoMoreCgIterationsThanJacobi)
{
   for (const int dim : {2, 3})
   {
      SCOPED_TRACE(dim);
      // gamma = 25 against sigma = 1: the grad-div term dominates the block.
      Problem p(dim, Ordering::byNODES, false, 1.0, 25.0);
      Vector b(p.fes.GetTrueVSize());
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      b.Randomize(1 + rank);

      OperatorJacobiSmoother jacobi(p.diag, p.ess);
      PointBlockJacobi pbj(p.fes, p.ess);
      pbj.SetDiagonal(p.diag);
      pbj.SetGradDiv(*p.gd);
      const int it_jacobi = CgIterations(*p.A, jacobi, b);
      const int it_pbj = CgIterations(*p.A, pbj, b);
      if (Mpi::Root())
      {
         mfem::out << "[pbj grad-div] dim " << dim << ": CG iterations, Jacobi "
                   << it_jacobi << ", point-block Jacobi " << it_pbj << std::endl;
      }
      EXPECT_LE(it_pbj, it_jacobi);
   }
}

TEST(PbjGradDiv, P3_ReleaseBuildDataKeepsMult)
{
   for (const int dim : {2, 3})
   {
      SCOPED_TRACE(dim);
      Problem p(dim, Ordering::byVDIM, true, 2.0, 0.0);
      PointBlockJacobi pbj(p.fes, p.ess);
      pbj.SetDiagonal(p.diag);
      pbj.SetGradDiv(*p.gd);
      const int n = p.fes.GetTrueVSize();
      Vector x(n), z_before(n), z_after(n);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      x.Randomize(5 + rank);
      pbj.Mult(x, z_before);
      pbj.ReleaseBuildData();
      EXPECT_EQ(pbj.GetDiagonal().Size(), 0);
      EXPECT_EQ(pbj.GetSymmetric().Size(), 0);
      pbj.Mult(x, z_after);
      z_after -= z_before;
      double diff = z_after.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_EQ(diff, 0.0);
   }
}

TEST(PbjGradDiv, P4_NewDiagonalMatchesAFreshBuild)
{
   for (const int dim : {2, 3})
   {
      SCOPED_TRACE(dim);
      // Two mass factors, as before and after a Delta-t change.
      Problem before(dim, Ordering::byNODES, true, 2.0, 0.0);
      Problem after(dim, Ordering::byNODES, true, 5.0, 0.0);
      PointBlockJacobi kept(before.fes, before.ess);
      kept.SetDiagonal(before.diag);
      kept.SetGradDiv(*before.gd);
      kept.SetDiagonal(after.diag);
      PointBlockJacobi fresh(after.fes, after.ess);
      fresh.SetDiagonal(after.diag);
      fresh.SetGradDiv(*after.gd);
      const int n = after.fes.GetTrueVSize();
      Vector x(n), z_kept(n), z_fresh(n);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      x.Randomize(9 + rank);
      kept.Mult(x, z_kept);
      fresh.Mult(x, z_fresh);
      z_kept -= z_fresh;
      double diff = z_kept.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_EQ(diff, 0.0);
   }
}
