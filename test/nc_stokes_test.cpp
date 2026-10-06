// AMR.0 -- the existing solver stack on NONCONFORMING (hanging-node) meshes,
// before any adaptation code exists. Static refinement patterns from
// amr_test_util.hpp put hanging nodes in the interior and on Dirichlet walls.
//  S1 steady Stokes polynomial MMS: the exact solution lies in the discrete
//     space, so every solver path must reproduce it to solver tolerance on an
//     NC mesh (P^T A P, constrained Dirichlet data, null space, LOR-AMG on NC);
//  S2 rotational static pressure (KineticHeadInterpolator) is conforming on
//     NC meshes, and the device path equals the host reference;
//  S3 PBJ / LOR-AMG stay effective near hanging nodes (iteration counts on an
//     NC mesh bounded by the conforming coarse/fine counts);
//  S4 an AMR-enabled mesh with no refinement (EnsureNCMesh + the explicit METIS
//     partition) reproduces the conforming run.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "post/kinetic_head.hpp"
#include "precond/point_block_jacobi.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mfem;
using incns::BoundaryConditions;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::SchurBlockType;
using incns::StokesSolver;
using incns::StokesSolverOptions;
using incns::VelocityPreconditioner;

namespace
{

void ElevatedRules(const RuleBook& rules, int order,
                   const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, order);
   irs[Geometry::CUBE] = &rules.Get(Geometry::CUBE, order);
}

struct SolverPath
{
   const char* name;
   VelocityPreconditioner prec;
   SchurBlockType schur;
};

const SolverPath kPaths[] =
{
   {"mass+jacobi", VelocityPreconditioner::Jacobi, SchurBlockType::Mass},
   {"mass+loramg", VelocityPreconditioner::LORAMG, SchurBlockType::Mass},
   {"cc(loramg)", VelocityPreconditioner::Jacobi, SchurBlockType::CahouetChabard},
};

// Steady Stokes, all-Dirichlet unit box, exact polynomial solution (2D Q3/Q2,
// 3D Q2/Q1 -- the fields of stokes_solver_test), on a refined NC mesh.
void SteadyMms(int dim, bool aniso, const SolverPath& path)
{
   SCOPED_TRACE(std::string("dim=") + std::to_string(dim) + (aniso ? " aniso "
                : " iso ") + path.name);
   const double nu = (dim == 2) ? 0.7 : 1.3;
   VectorFunctionCoefficient u_exact(dim, [dim](const Vector & x, Vector & v)
   {
      if (dim == 2)
      {
         v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
         v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
      }
      else { v(0) = x[1] * x[1]; v(1) = x[2] * x[2]; v(2) = x[0] * x[0]; }
   });
   FunctionCoefficient p_exact([dim](const Vector & x)
   {
      return (dim == 2) ? x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0
             : x[0] + x[1] + x[2] - 1.5;
   });
   VectorFunctionCoefficient forcing(dim, [dim, nu](const Vector & x, Vector & f)
   {
      if (dim == 2)
      {
         const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
         const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
         f(0) = -nu * lap0 + 2.0 * x[0];
         f(1) = -nu * lap1 + 2.0 * x[1];
      }
      else { f = 1.0 - 2.0 * nu; }
   });

   RuleBook rules; // declared before the mesh, so it outlives the mesh
   auto mesh = amr_test::NcBox(amr_test::Box(dim, dim == 2 ? 3 : 2, false));
   const long long ne0 = mesh->GetGlobalNE();
   amr_test::RefineBands(*mesh, aniso);
   ASSERT_TRUE(mesh->Nonconforming());
   ASSERT_GT(mesh->GetGlobalNE(), ne0);

   const int ku = (dim == 2) ? 3 : 2;
   MixedSpaces spaces(*mesh, ku, ku - 1);
   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 2 * dim; ++attr)
   {
      bc.AddVelocityDirichlet(attr, u_exact);
   }
   StokesSolverOptions opts;
   opts.nu = nu;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   opts.velocity_prec = path.prec;
   opts.schur = path.schur;
   StokesSolver solver(spaces, rules, bc, opts);
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 2 * ku + 4, irs);
   const double u_err = u.ComputeL2Error(u_exact, irs);
   const double p_err = p.ComputeL2Error(p_exact, irs);
   const long long ne = mesh->GetGlobalNE(); // collective: never inside Root()
   if (Mpi::Root())
   {
      mfem::out << "[nc mms] dim=" << dim << (aniso ? " aniso " : " iso ")
                << path.name << "  NE=" << ne << "  u_err="
                << u_err << "  p_err=" << p_err << "  iters="
                << solver.Iterations() << std::endl;
   }
   EXPECT_TRUE(solver.Converged());
   EXPECT_LE(u_err, 1e-8);
   EXPECT_LE(p_err, 1e-7);
}

// Smooth test velocity for S2/S3.
void SmoothVelocity(const Vector& x, Vector& u)
{
   const double X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   u(0) = std::sin(M_PI * Y) + 0.5 * Z * Z + 0.3;
   u(1) = std::cos(M_PI * X) * (1.0 + Z);
   if (u.Size() == 3) { u(2) = std::sin(M_PI * X * Y); }
}

void SetConforming(ParGridFunction& gf, VectorCoefficient& c)
{
   gf.ProjectCoefficient(c);
   Vector t;
   gf.GetTrueDofs(t);
   gf.SetFromTrueDofs(t);
}

// GMRES(50) iterations to 1e-10 for A = sigma M + nu K + N preconditioned by
// point-block Jacobi, all-Dirichlet, solver rules. @p rules must outlive
// @p mesh: the mesh caches geometric factors keyed by IntegrationRule*, so a
// rule freed while the mesh lives on leaves a dangling key that a later rule
// allocated at the same address would hit (stale factors -> NaN; seen on the
// debug device).
int PbjIterations(ParMesh& mesh, int p, const RuleBook& rules)
{
   const int dim = mesh.Dimension();
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   const Geometry::Type g = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   Array<int> ess, bdr(mesh.bdr_attributes.Max());
   bdr = 1;
   fes.GetEssentialTrueDofs(bdr, ess);
   ParGridFunction w(&fes);
   VectorFunctionCoefficient wc(dim, SmoothVelocity);
   SetConforming(w, wc);
   ConstantCoefficient sigma(20.0), nu(0.01);
   ParBilinearForm sym(&fes), nform(&fes);
   auto* m = new VectorMassIntegrator(sigma);
   auto* k = new VectorDiffusionIntegrator(nu);
   m->SetIntRule(&rules.Get(g, 2 * p));
   k->SetIntRule(&rules.Get(g, 2 * p + dim - 1));
   sym.AddDomainIntegrator(m);
   sym.AddDomainIntegrator(k);
   auto* rot = new incns::VectorRotationalConvectionIntegrator(w, 1.0);
   rot->SetIntRule(&rules.Get(g, 3 * p));
   nform.AddDomainIntegrator(rot);
   for (ParBilinearForm* f : {&sym, &nform})
   {
      f->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      f->Assemble();
   }
   OperatorPtr S, N;
   sym.FormSystemMatrix(ess, S);
   Array<int> none;
   nform.FormSystemMatrix(none, N);
   ConstrainedOperator Nc(N.Ptr(), ess, false, Operator::DIAG_ZERO);
   SumOperator A(S.Ptr(), 1.0, &Nc, 1.0, false, false);
   incns::PointBlockJacobi pbj(fes, *rot, ess);
   Vector diag(fes.GetTrueVSize());
   sym.AssembleDiagonal(diag);
   pbj.SetDiagonal(diag);
   pbj.UpdateSkew();
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetKDim(50);
   gmres.SetRelTol(1e-10);
   gmres.SetMaxIter(2000);
   gmres.SetPrintLevel(-1);
   gmres.SetOperator(A);
   gmres.SetPreconditioner(pbj);
   Vector b(fes.GetTrueVSize()), x(b.Size());
   b.UseDevice(true);
   x.UseDevice(true);
   b.Randomize(7);
   x = 0.0;
   gmres.Mult(b, x);
   return gmres.GetConverged() ? gmres.GetNumIterations() : -1;
}

// CG iterations to 1e-10 for sigma M + nu K preconditioned by LOR-AMG.
int LorAmgIterations(ParMesh& mesh, int p)
{
   const int dim = mesh.Dimension();
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   Array<int> ess, bdr(mesh.bdr_attributes.Max());
   bdr = 1;
   fes.GetEssentialTrueDofs(bdr, ess);
   ConstantCoefficient sigma(20.0), nu(0.5);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new VectorMassIntegrator(sigma));
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   OperatorPtr A;
   a.FormSystemMatrix(ess, A);
   ParBilinearForm alor(&fes);
   alor.AddDomainIntegrator(new VectorMassIntegrator(sigma));
   alor.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
   LORSolver<HypreBoomerAMG> lor(alor, ess);
   lor.GetSolver().SetPrintLevel(0);
   lor.GetSolver().SetSystemsOptions(dim, true);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*A);
   cg.SetPreconditioner(lor);
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(-1);
   Vector b(fes.GetTrueVSize()), x(b.Size());
   b.UseDevice(true);
   x.UseDevice(true);
   b.Randomize(9);
   x = 0.0;
   cg.Mult(b, x);
   return cg.GetConverged() ? cg.GetNumIterations() : -1;
}

} // namespace

TEST(NcStokes, S1_SteadyMmsExactOnNonconformingMesh)
{
   for (int dim : {2, 3})
      for (bool aniso : {false, true})
         for (const SolverPath& path : kPaths)
         {
            SteadyMms(dim, aniso, path);
         }
}

TEST(NcStokes, S2_StaticPressureIsConforming)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      auto mesh = amr_test::NcBox(amr_test::Box(dim, dim == 2 ? 3 : 2, false));
      amr_test::RefineBands(*mesh, true);
      const int ku = (dim == 2) ? 3 : 2;
      MixedSpaces spaces(*mesh, ku, ku - 1);
      ParGridFunction u(&spaces.Velocity()), ke(&spaces.Pressure()),
                      ke_host(&spaces.Pressure());
      VectorFunctionCoefficient uc(dim, SmoothVelocity);
      SetConforming(u, uc);

      incns::KineticHeadInterpolator kh(spaces.Velocity(), spaces.Pressure());
      kh.Interpolate(u, ke);

      // Conforming: restricting to true dofs and prolongating back changes
      // nothing (hanging values equal the constrained interpolant).
      ParGridFunction back(&spaces.Pressure());
      Vector t;
      ke.GetTrueDofs(t);
      back.SetFromTrueDofs(t);
      back -= ke;
      double d = back.Normlinf(), s = ke.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_LE(d, 1e-14 * s);

      // Host reference (ProjectCoefficient), made conforming the same way.
      incns::KineticHeadInterpolator::InterpolateHost(u, ke_host);
      ke_host.GetTrueDofs(t);
      ke_host.SetFromTrueDofs(t);
      ke_host -= ke;
      double e = ke_host.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_LE(e, 1e-12 * s);
   }
}

TEST(NcStokes, S3_PreconditionersStayEffectiveNearHangingNodes)
{
   RuleBook rules; // outlives every mesh below (see PbjIterations)
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      const int n = (dim == 2) ? 4 : 2, p = (dim == 2) ? 3 : 2;
      auto coarse = amr_test::NcBox(amr_test::Box(dim, n, false));
      auto fine = amr_test::NcBox(amr_test::Box(dim, n, false));
      amr_test::RefineWhere(*fine, [](const Vector&) { return true; },
      static_cast<char>(dim == 3 ? Refinement::XYZ
                        : Refinement::XY));
      auto nc = amr_test::NcBox(amr_test::Box(dim, n, false));
      amr_test::RefineBands(*nc, true);

      const int pbj_c = PbjIterations(*coarse, p, rules),
                pbj_f = PbjIterations(*fine, p, rules),
                pbj_nc = PbjIterations(*nc, p, rules);
      const int lor_c = LorAmgIterations(*coarse, p),
                lor_f = LorAmgIterations(*fine, p), lor_nc = LorAmgIterations(*nc, p);
      if (Mpi::Root())
      {
         mfem::out << "[nc pc] dim=" << dim << "  PBJ coarse/fine/nc = " << pbj_c
                   << "/" << pbj_f << "/" << pbj_nc << "  LOR-AMG = " << lor_c
                   << "/" << lor_f << "/" << lor_nc << std::endl;
      }
      ASSERT_GT(pbj_nc, 0);
      ASSERT_GT(lor_nc, 0);
      EXPECT_LE(pbj_nc, 1.5 * std::max(pbj_c, pbj_f) + 2);
      EXPECT_LE(lor_nc, 1.5 * std::max(lor_c, lor_f) + 2);
   }
}

TEST(NcStokes, S4_NonconformingReadyMeshMatchesConforming)
{
   // Fully periodic 2D TGV (Stokes), 5 fixed steps: the conforming mesh and
   // the AMR-ready (EnsureNCMesh + explicit METIS partition) mesh solve the
   // same discrete system -- same errors to solver tolerance, iteration
   // counts within +-1.
   const double nu = 0.1;
   for (SchurBlockType schur :
        {
           SchurBlockType::Mass,
           SchurBlockType::CahouetChabard
        })
   {
      SCOPED_TRACE(schur == SchurBlockType::Mass ? "mass" : "cc");
      std::vector<double> err[2];
      std::vector<int> its[2];
      for (int run = 0; run < 2; ++run)
      {
         RuleBook rules; // outlives the mesh (geometric-factor cache keys)
         Mesh serial = incns::MakeBoxMesh(amr_test::Box(2, 4, true, 2 * M_PI));
         auto mesh = incns::PartitionMesh(serial, run == 1);
         EXPECT_EQ(mesh->Nonconforming(), run == 1);
         MixedSpaces spaces(*mesh, 3, 2);
         BoundaryConditions bc(spaces.Velocity());
         Vector zero(2);
         zero = 0.0;
         VectorConstantCoefficient f(zero);
         incns::TimeIntegratorOptions opts;
         opts.nu = nu;
         opts.dt = 0.05;
         opts.rtol = 1e-12;
         opts.max_iter = 2000;
         opts.schur = schur;
         incns::StokesTimeIntegrator stepper(spaces, rules, bc, f, opts);
         VectorFunctionCoefficient u_exact = incns::tgv2d::VelocityCoefficient(nu);
         stepper.SetInitialVelocity(u_exact);
         const IntegrationRule* irs[Geometry::NumGeom];
         ElevatedRules(rules, 10, irs);
         for (int s = 0; s < 5; ++s)
         {
            stepper.Step();
            u_exact.SetTime(stepper.Time());
            err[run].push_back(stepper.Velocity().ComputeL2Error(u_exact, irs));
            its[run].push_back(stepper.LastIterations());
         }
      }
      for (int s = 0; s < 5; ++s)
      {
         EXPECT_NEAR(err[1][s], err[0][s], 1e-8 * err[0][s]) << "step " << s;
         EXPECT_LE(std::abs(its[1][s] - its[0][s]), 1) << "step " << s;
      }
   }
}
