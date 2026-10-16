// The Schur complement (Uzawa) outer method (solver/schur_solver.hpp,
// StokesSolverOptions::outer = OuterMethod::Schur) against the coupled FGMRES
// path. Both solve the same eliminated system; the answer must agree to the
// solver tolerance, whatever the Schur preconditioner, inner tolerance mode
// or initial guess. Nothing here pins a case-level default: the low-level tests
// set what they compare, and U5/U6 run the Parameters defaults of the day with
// only the outer method changed.
//  U1 S = B A^-1 B^T is symmetric (tight inner solves) and its image carries
//     no constant pressure on an enclosed box.
//  U2 One unsteady Stokes solve, Schur vs coupled, on an enclosed box (null
//     space), a periodic box (null space) and a channel with an outflow (none),
//     with and without grad-div, under the Cahouet-Chabard and the legacy
//     Laplacian Schur PC: flexible CG gives the coupled u and p to 1e-6,
//     and the coupled residual it reports meets the tolerance.
//  U3 The inner tolerance modes (tight, loose, relaxed) give the same answer.
//  U4 The pressure projection: after a solve, the same system with the
//     right-hand side scaled is guessed exactly (its increment is in the
//     basis), and the answer is unchanged.
//  U5 A Navier-Stokes Taylor-Green march at the case defaults: Schur gives
//     the coupled velocity to 1e-6.
//  U6 A restart of a Schur march (pressure projection in the checkpoint)
//     matches the uninterrupted one.
//  U7 VelocityCG (the Schur method's in-place velocity CG) takes MFEM's
//     CGSolver's iterations and gives its solution, and leaves the residual
//     in the right-hand side.
//  U8 A Delta-t refresh of a Schur solver (the velocity CG re-pointed, the
//     point-block Jacobi updated in place, no block preconditioner) gives
//     the answer of a solver built at the new step.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "post/checkpoint.hpp"
#include "quadrature/rule_book.hpp"
#include "repro_tolerance.hpp"
#include "solver/case.hpp"
#include "solver/schur_complement.hpp"
#include "solver/stokes_solver.hpp"
#include "solver/velocity_cg.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <string>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::InnerTolerance;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::OuterMethod;
using incns::RuleBook;
using incns::StokesSolver;
using incns::StokesSolverOptions;

namespace
{

enum class Domain { Enclosed, Periodic, Channel };

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

double RelDiff(const Vector& a, const Vector& b)
{
   Vector d(a);
   d -= b;
   return Norm(d) / std::max(Norm(a), 1e-300);
}

// The attribute of the boundary elements on the face x_ci = val (box meshes).
int FaceAttribute(ParMesh& mesh, int ci, double val)
{
   int attr = 0;
   Vector c;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      if (std::abs(c(ci) - val) < 1e-9) { attr = mesh.GetBdrAttribute(be); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, mesh.GetComm());
   return attr;
}

// A small 2D problem: mesh, spaces, boundary conditions, forcing. The
// RuleBook is declared before the mesh (developing.md, RuleBook lifetime).
struct Problem
{
   RuleBook rules;
   std::unique_ptr<ParMesh> mesh;
   std::unique_ptr<MixedSpaces> spaces;
   std::unique_ptr<BoundaryConditions> bc;
   std::unique_ptr<VectorFunctionCoefficient> inflow;
   Vector f; ///< Momentum right-hand side (true dofs).

   explicit Problem(Domain d, int n = 6)
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {n, n, 0};
      if (d == Domain::Periodic)
      {
         s.lengths = {2.0 * M_PI, 2.0 * M_PI};
      }
      else
      {
         s.lengths = {d == Domain::Channel ? 2.0 : 1.0, 1.0};
         s.periodic = {false, false, false};
      }
      Mesh serial = MakeBoxMesh(s);
      mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
      spaces = std::make_unique<MixedSpaces>(*mesh, 3, 2);
      bc = std::make_unique<BoundaryConditions>(spaces->Velocity());
      if (d == Domain::Enclosed)
      {
         for (int a : mesh->bdr_attributes) { bc->AddNoSlip(a); }
      }
      else if (d == Domain::Channel)
      {
         const int in = FaceAttribute(*mesh, 0, 0.0);
         const int out = FaceAttribute(*mesh, 0, 2.0);
         inflow = std::make_unique<VectorFunctionCoefficient>(
                     2, [](const Vector & x, Vector & u)
         {
            u(0) = 4.0 * x(1) * (1.0 - x(1));
            u(1) = 0.0;
         });
         for (int a : mesh->bdr_attributes)
         {
            if (a == in) { bc->AddVelocityDirichlet(a, *inflow); }
            else if (a == out) { bc->AddOutflow(a); }
            else { bc->AddNoSlip(a); }
         }
      }

      // A solenoidal part plus grad(sin x sin y), so that the pressure is
      // of order one on every domain (without the gradient the periodic
      // pressure is near zero and its relative difference is noise).
      VectorFunctionCoefficient forcing(2, [](const Vector & x, Vector & g)
      {
         g(0) = std::sin(x(0)) * std::cos(x(1)) + 0.3 * x(1) +
                std::cos(x(0)) * std::sin(x(1));
         g(1) = -std::cos(x(0)) * std::sin(x(1)) + 0.2 * x(0) * x(0) +
                std::sin(x(0)) * std::cos(x(1));
      });
      ParLinearForm form(&spaces->Velocity());
      auto* fi = new VectorDomainLFIntegrator(forcing);
      fi->SetIntRule(&rules.Get(Geometry::SQUARE, 2 * 3 + 2));
      form.AddDomainIntegrator(fi);
      form.Assemble();
      f.SetSize(spaces->Velocity().GetTrueVSize());
      form.ParallelAssemble(f);
   }
};

// The Cahouet-Chabard solver options both outer methods share.
StokesSolverOptions BaseOptions(double grad_div)
{
   StokesSolverOptions o;
   o.nu = 0.01;
   o.mass_coeff = 50.0; // a BDF-like step: beta0 / dt
   o.schur = incns::SchurBlockType::CahouetChabard;
   o.cc.a_pc = incns::APC::JacobiPCG;
   o.grad_div = grad_div;
   o.rtol = 1e-10;
   o.max_iter = 500;
   o.kdim = 200;
   return o;
}

struct Answer
{
   Vector u, p;
   int iterations = 0;
   bool converged = false;
   double final_residual = 0.0;
   double tolerance = 0.0;
};

// One solve from zero.
Answer SolveOnce(Problem& pr, const StokesSolverOptions& o)
{
   StokesSolver solver(*pr.spaces, pr.rules, *pr.bc, o);
   ParGridFunction u(&pr.spaces->Velocity()), p(&pr.spaces->Pressure());
   u = 0.0;
   p = 0.0;
   solver.SolveTrue(pr.f, u, p);
   Answer a;
   a.u.SetSize(pr.spaces->Velocity().GetTrueVSize());
   a.p.SetSize(pr.spaces->Pressure().GetTrueVSize());
   u.GetTrueDofs(a.u);
   p.GetTrueDofs(a.p);
   a.iterations = solver.Iterations();
   a.converged = solver.Converged();
   a.final_residual = solver.Stats().final_residual;
   a.tolerance = solver.Stats().tolerance;
   return a;
}

void ExpectSameAnswer(const Answer& ref, const Answer& got,
                      const std::string& what)
{
   EXPECT_TRUE(ref.converged) << what << " (coupled)";
   EXPECT_TRUE(got.converged) << what;
   EXPECT_LE(got.final_residual, got.tolerance * (1.0 + 1e-12)) << what;
   const double du = RelDiff(ref.u, got.u);
   const double dp = RelDiff(ref.p, got.p);
   if (Mpi::Root())
   {
      mfem::out << "[schur] " << what << ": " << got.iterations
                << " iterations (coupled " << ref.iterations << "), du " << du
                << ", dp " << dp << std::endl;
   }
   EXPECT_LE(du, 1e-6) << what;
   EXPECT_LE(dp, 1e-6) << what;
}

} // namespace

TEST(SchurSolver, U1_SchurComplementIsSymmetric)
{
   Problem pr(Domain::Enclosed);
   StokesSolverOptions o = BaseOptions(0.0);
   StokesSolver solver(*pr.spaces, pr.rules, *pr.bc, o);
   incns::StokesOperator& op = solver.Blocks();

   OperatorJacobiSmoother jacobi(op.MomentumDiagonal(), op.EssentialTrueDofs());
   incns::VelocityCG cg(MPI_COMM_WORLD, op.FullMomentum().Height());
   cg.SetOperator(op.FullMomentum());
   cg.SetPreconditioner(jacobi);
   cg.SetMaxIter(2000);
   incns::ConstantPressureProjector proj(pr.spaces->Pressure());
   incns::SchurComplementOperator S(op.Divergence(), cg, &proj);
   S.SetInnerTolerance(1e-14);

   const int n_p = pr.spaces->Pressure().GetTrueVSize();
   Vector x(n_p), y(n_p), sx(n_p), sy(n_p);
   x.Randomize(1 + Mpi::WorldRank());
   y.Randomize(101 + Mpi::WorldRank());
   proj.Project(x);
   proj.Project(y);
   S.Mult(x, sx);
   S.Mult(y, sy);
   const double xsy = InnerProduct(MPI_COMM_WORLD, x, sy);
   const double ysx = InnerProduct(MPI_COMM_WORLD, y, sx);
   EXPECT_NEAR(xsy, ysx, 1e-10 * std::abs(xsy));
   EXPECT_GT(InnerProduct(MPI_COMM_WORLD, x, sx), 0.0); // positive off 1
   EXPECT_NEAR(proj.ConstantComponent(sx), 0.0, 1e-14 * Norm(sx));
   EXPECT_EQ(S.Applications(), 2);
}

TEST(SchurSolver, U2_MatchesTheCoupledSolve)
{
   const struct { Domain d; const char* name; } domains[] =
   {
      {Domain::Enclosed, "enclosed"}, {Domain::Periodic, "periodic"},
      {Domain::Channel, "channel"}
   };
   for (const auto& dom : domains)
   {
      for (double gd : {0.0, 1.0})
      {
         Problem pr(dom.d);
         const Answer ref = SolveOnce(pr, BaseOptions(gd));
         const std::string what = std::string(dom.name) + " gd " +
                                  std::to_string(gd);
         StokesSolverOptions o = BaseOptions(gd);
         o.outer = OuterMethod::Schur;
         ExpectSameAnswer(ref, SolveOnce(pr, o), what);
         // The legacy Laplacian model (one V-cycle, no inner CG: a fixed
         // linear Schur PC). Compare with the coupled solve under the same
         // preconditioner.
         StokesSolverOptions legacy = BaseOptions(gd);
         legacy.cc.schur_model = incns::SchurModel::LaplacianLegacy;
         const Answer ref_legacy = SolveOnce(pr, legacy);
         legacy.outer = OuterMethod::Schur;
         ExpectSameAnswer(ref_legacy, SolveOnce(pr, legacy), what + " legacy");
      }
   }
}

TEST(SchurSolver, U3_InnerToleranceModesAgree)
{
   Problem pr(Domain::Channel);
   const Answer ref = SolveOnce(pr, BaseOptions(1.0));
   for (InnerTolerance t :
        {InnerTolerance::Tight, InnerTolerance::Loose, InnerTolerance::Relaxed})
   {
      StokesSolverOptions o = BaseOptions(1.0);
      o.outer = OuterMethod::Schur;
      o.uzawa.inner = t;
      // A fixed loose inner tolerance leaves a true-residual gap of about
      // eps_loose times the cycle's starting residual, so at rtol 1e-10 it
      // needs more cycles than the default 3 (CI: 3 left 8.0e-11 against a
      // 6.3e-11 target). Every mode gets the same cap.
      o.uzawa.max_cycles = 10;
      const char* name = t == InnerTolerance::Tight ? "tight" :
                         t == InnerTolerance::Loose ? "loose" : "relaxed";
      ExpectSameAnswer(ref, SolveOnce(pr, o), std::string("inner ") + name);
   }
}

TEST(SchurSolver, U4_PressureProjectionGuessesAScaledSystem)
{
   // Enclosed box, homogeneous Dirichlet data: the solution scales with f.
   StokesSolverOptions o = BaseOptions(1.0);
   o.outer = OuterMethod::Schur;
   o.initial_guess = incns::InitialGuess::Project;
   o.projection_k = 4;
   Problem enc(Domain::Enclosed);
   StokesSolver s2(*enc.spaces, enc.rules, *enc.bc, o);
   ParGridFunction u2(&enc.spaces->Velocity()), p2(&enc.spaces->Pressure());
   u2 = 0.0;
   p2 = 0.0;
   s2.SolveTrue(enc.f, u2, p2);
   ASSERT_TRUE(s2.Converged());
   EXPECT_FALSE(s2.Stats().uzawa.projected); // nothing to project from yet
   Vector f15(enc.f);
   f15 *= 1.5;
   s2.SolveTrue(f15, u2, p2);
   ASSERT_TRUE(s2.Converged());
   const incns::SchurSolveStats& z = s2.Stats().uzawa;
   EXPECT_TRUE(z.projected);
   if (Mpi::Root())
   {
      mfem::out << "[schur] projection: residual " << z.start_residual
                << " -> " << z.guess_residual << ", " << z.iterations
                << " iterations" << std::endl;
   }
   // From (u_1, p_1) the step still needs 0.5 p_1, and the first solve's
   // increment p_1 is in the basis: the guess is exact to its tolerance.
   EXPECT_LE(z.guess_residual, 1e-6 * z.start_residual);

   // And the answer is the scaled one.
   Vector ut(enc.spaces->Velocity().GetTrueVSize());
   u2.GetTrueDofs(ut);
   const Answer ref = SolveOnce(enc, BaseOptions(1.0));
   Vector u_ref(ref.u);
   u_ref *= 1.5;
   EXPECT_LE(RelDiff(u_ref, ut), 1e-6);
}

namespace
{

incns::Parameters TgvParams(OuterMethod outer)
{
   // The case defaults of the day; only the outer method (and a tight
   // tolerance, so the two methods can be compared closely) is set here.
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.nu = 0.01;
   p.mesh.dim = 2;
   p.mesh.num_elems = {8, 8, 8};
   p.mesh.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   p.dt = 0.05;
   p.t_final = 0.5;
   p.initial_velocity = "taylor_green_2d";
   p.krylov_rtol = 1e-10;
   p.outer = outer;
   p.Normalize();
   return p;
}

Vector March(const incns::Parameters& params, double stop_at = -1.0,
             const std::string& checkpoint = "")
{
   Mesh serial = MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::Case flow(mesh, params);
   auto u0 = incns::MakeInitialVelocity(params);
   flow.SetInitialVelocity(*u0);
   while (!flow.Done() && (stop_at < 0.0 || flow.Time() < stop_at))
   {
      flow.Step();
   }
   if (!checkpoint.empty())
   {
      incns::Checkpoint::Write(checkpoint, flow.Integrator(), params);
   }
   Vector u(flow.Spaces().Velocity().GetTrueVSize());
   flow.Velocity().GetTrueDofs(u);
   return u;
}

} // namespace

TEST(SchurSolver, U5_CaseMarchMatchesTheCoupledOne)
{
   const Vector coupled = March(TgvParams(OuterMethod::Coupled));
   const Vector schur = March(TgvParams(OuterMethod::Schur));
   const double d = RelDiff(coupled, schur);
   if (Mpi::Root())
   {
      mfem::out << "[schur] Taylor-Green march: reldiff " << d << std::endl;
   }
   EXPECT_LE(d, 1e-6);
}

TEST(SchurSolver, U6_RestartMatchesUninterrupted)
{
   incns::Parameters p = TgvParams(OuterMethod::Schur);
   p.initial_guess = incns::InitialGuess::Project; // the basis is checkpointed
   const Vector ref = March(p);
   const std::string dir = "chk_schur_np" + std::to_string(Mpi::WorldSize());
   March(p, 0.25, dir);
   incns::Parameters second = p;
   second.restart_from = dir;
   const Vector restarted = March(second);
   EXPECT_LE(RelDiff(ref, restarted), incns_test::ReproTol(1e-13));
}

TEST(SchurSolver, U7_VelocityCgIsMfemCg)
{
   Problem pr(Domain::Channel);
   StokesSolverOptions o = BaseOptions(1.0);
   StokesSolver solver(*pr.spaces, pr.rules, *pr.bc, o);
   incns::StokesOperator& op = solver.Blocks();
   const Operator& A = op.FullMomentum();
   const int n = A.Height();
   OperatorJacobiSmoother jacobi(op.MomentumDiagonal(), op.EssentialTrueDofs());

   // A right-hand side zero on the essential rows, as the Schur method's.
   Vector b(n);
   b.Randomize(7 + Mpi::WorldRank());
   b.SetSubVector(op.EssentialTrueDofs(), 0.0);

   CGSolver ref(MPI_COMM_WORLD);
   ref.SetOperator(A); // before the PC: MFEM forwards SetOperator to it
   ref.SetPreconditioner(jacobi);
   ref.SetRelTol(1e-9);
   ref.SetAbsTol(0.0);
   ref.SetMaxIter(500);
   ref.SetPrintLevel(-1);
   ref.iterative_mode = false;
   Vector x_ref(n);
   x_ref = 0.0;
   ref.Mult(b, x_ref);
   ASSERT_TRUE(ref.GetConverged());

   incns::VelocityCG cg(MPI_COMM_WORLD, n);
   cg.SetOperator(A);
   cg.SetPreconditioner(jacobi);
   cg.SetRelTol(1e-9);
   cg.SetMaxIter(500);
   Vector r(b), x(n);
   x = 1.0; // ignored: the solve starts from zero
   cg.Solve(r, x);
   EXPECT_TRUE(cg.GetConverged());
   EXPECT_EQ(cg.GetNumIterations(), ref.GetNumIterations());
   EXPECT_LE(RelDiff(x_ref, x), 1e-12);

   // r is b - A x (the recurrence's: equal to roundoff relative to b).
   Vector ax(n);
   A.Mult(x, ax);
   Vector res(b);
   res -= ax;
   res -= r;
   EXPECT_LE(Norm(res), 1e-10 * Norm(b));

   // The iteration cap.
   cg.SetMaxIter(3);
   r = b;
   cg.Solve(r, x);
   EXPECT_FALSE(cg.GetConverged());
   EXPECT_EQ(cg.GetNumIterations(), 3);
}

TEST(SchurSolver, U8_RefreshMatchesAFreshSolver)
{
   for (const bool pbj : {true, false})
   {
      const std::string pc = pbj ? "pbj" : "jacobi";
      StokesSolverOptions o = BaseOptions(1.0);
      o.outer = OuterMethod::Schur;
      o.cc.a_pcg_pc = pbj ? incns::APcgPC::PointBlockJacobi
                      : incns::APcgPC::Jacobi;
      Problem pr(Domain::Channel);
      StokesSolver solver(*pr.spaces, pr.rules, *pr.bc, o);
      ParGridFunction u(&pr.spaces->Velocity()), p(&pr.spaces->Pressure());
      u = 0.0;
      p = 0.0;
      solver.SolveTrue(pr.f, u, p);
      ASSERT_TRUE(solver.Converged()) << pc;
      solver.Refresh(80.0);
      u = 0.0;
      p = 0.0;
      solver.SolveTrue(pr.f, u, p);
      Answer got;
      got.u.SetSize(pr.spaces->Velocity().GetTrueVSize());
      got.p.SetSize(pr.spaces->Pressure().GetTrueVSize());
      u.GetTrueDofs(got.u);
      p.GetTrueDofs(got.p);
      got.iterations = solver.Iterations();
      got.converged = solver.Converged();
      got.final_residual = solver.Stats().final_residual;
      got.tolerance = solver.Stats().tolerance;

      StokesSolverOptions fresh = o;
      fresh.outer = OuterMethod::Coupled;
      fresh.mass_coeff = 80.0;
      ExpectSameAnswer(SolveOnce(pr, fresh), got, "refresh, a_pcg_pc " + pc);
   }
}
