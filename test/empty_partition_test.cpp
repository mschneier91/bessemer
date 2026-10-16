// Ranks that own no elements: a mesh with fewer elements than ranks (the
// 3-element Gmsh mesh of benchmark_setup_test M1 at np 4), or any partition
// that leaves a rank empty, must run like any other. Here every element is on
// rank 0, so every other rank is empty at np 2 and np 4 (np 1 is the serial
// run, the case the others reproduce):
//  E1 a Stokes cavity (a moving lid) on that partition matches the default
//     partition's run to the solver tolerance (StokesSolver::SolveTrue once
//     never built its block vectors on an empty rank, whose block sizes are
//     all 0: a null dereference at the first solve, 2026-10-16);
//  E2 Navier-Stokes Taylor-Green, IMEX and OIFS under CFL control, the same,
//     with the diagnostics (kinetic energy, dissipation, divergence, CFL);
//  E3 a checkpoint written mid-run restarts on that partition and finishes
//     as the uninterrupted run does (bitwise on a host backend), with the
//     solvers' guess state: the step reference's A x_prev and the
//     extrapolation's x_prev (an empty rank once wrote no files for them,
//     and the restart aborted on it).

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/periodic_box.hpp"
#include "post/checkpoint.hpp"
#include "solver/case.hpp"
#include "repro_tolerance.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <string>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

// The case's box mesh: every element on rank 0 (all_on_root), or the default
// partition.
std::unique_ptr<ParMesh> BoxMesh(const Parameters& p, bool all_on_root)
{
   if (!all_on_root) { return incns::MakeCaseMesh(p); }
   Mesh serial = incns::MakeBoxMesh(p.mesh);
   Array<int> partition(serial.GetNE());
   partition = 0;
   auto mesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial,
                                         partition.GetData());
   EXPECT_EQ(mesh->GetNE() > 0, Mpi::Root()); // the other ranks are empty
   return mesh;
}

struct Outcome
{
   int steps = 0;
   double t = 0.0;
   double u_norm = 0.0, p_norm = 0.0; ///< l2 norms of the true dofs.
   double ke = 0.0, dissipation = 0.0, divergence = 0.0, cfl = 0.0;
   Vector u; ///< Velocity true dofs (comparable on one partition only).
};

Outcome Collect(Case& flow, const Parameters& p)
{
   Outcome o;
   o.steps = flow.Integrator().StepCount();
   o.t = flow.Time();
   o.u.SetSize(flow.Spaces().Velocity().GetTrueVSize());
   flow.Velocity().GetTrueDofs(o.u);
   Vector pt(flow.Spaces().Pressure().GetTrueVSize());
   flow.Pressure().GetTrueDofs(pt);
   o.u_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, o.u, o.u));
   o.p_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, pt, pt));
   o.ke = flow.KineticEnergy();
   o.dissipation = flow.DissipationRate();
   o.divergence = flow.DivergenceNorm();
   if (p.equation == incns::Equation::NavierStokes)
   {
      o.cfl = flow.ConvectiveCflNumber();
   }
   return o;
}

// March @p p to t_final on the given partition.
Outcome March(const Parameters& p, bool all_on_root)
{
   auto mesh = BoxMesh(p, all_on_root);
   Case flow(*mesh, p);
   std::unique_ptr<VectorCoefficient> u0;
   if (p.initial_velocity != "zero")
   {
      u0 = incns::MakeInitialVelocity(p);
      flow.SetInitialVelocity(*u0);
   }
   flow.Run();
   EXPECT_TRUE(flow.Done());
   return Collect(flow, p);
}

// The empty-rank run agrees with the default partition's to the solver
// tolerance (the partition changes the preconditioner, not the problem).
void ExpectSameFlow(const Outcome& a, const Outcome& b, double rel)
{
   EXPECT_EQ(a.steps, b.steps);
   EXPECT_NEAR(a.t, b.t, 1e-12);
   EXPECT_NEAR(a.u_norm, b.u_norm, rel * b.u_norm);
   EXPECT_NEAR(a.p_norm, b.p_norm, rel * b.p_norm);
   EXPECT_NEAR(a.ke, b.ke, rel * b.ke);
   EXPECT_NEAR(a.dissipation, b.dissipation, rel * std::abs(b.dissipation));
   EXPECT_NEAR(a.divergence, b.divergence, rel * b.divergence + 1e-12);
   EXPECT_NEAR(a.cfl, b.cfl, rel * b.cfl);
}

Parameters Tgv(const std::string& convection, const std::string& guess)
{
   return Parameters::LoadYAMLString(
             "equation: navier_stokes\n"
             "initial_velocity: taylor_green_2d\n"
             "physics:\n  nu: 0.05\n"
             "mesh:\n  dim: 2\n  elements: [4, 4]\n"
             "time:\n  dt: 0.05\n  t_final: 0.3\n  step_control: cfl\n"
             "  convection: " + convection + "\n"
             "solver:\n  rtol: 1.0e-10\n  initial_guess: " + guess + "\n");
}

} // namespace

TEST(EmptyPartition, E1_StokesCavity)
{
   const Parameters p = Parameters::LoadYAMLString(
                           "equation: stokes\n"
                           "physics:\n  nu: 1.0\n"
                           "mesh:\n  dim: 2\n  elements: [4, 4]\n  lengths: [1.0, 1.0]\n"
                           "  periodic: [false, false]\n"
                           "time:\n  dt: 0.05\n  t_final: 0.2\n  step_control: fixed\n"
                           "solver:\n  rtol: 1.0e-10\n"
                           "boundary_conditions:\n"
                           "  - {select: [ymax], type: velocity, value: [1.0, 0.0]}\n"
                           "  - {select: [xmin, xmax, ymin], type: no_slip}\n");
   const Outcome empty = March(p, true);
   const Outcome ref = March(p, false);
   EXPECT_GT(ref.ke, 0.0);
   ExpectSameFlow(empty, ref, 1e-8);
}

TEST(EmptyPartition, E2_NavierStokesTaylorGreen)
{
   for (const std::string convection : {"imex", "oifs"})
   {
      SCOPED_TRACE(convection);
      const Parameters p = Tgv(convection, "project");
      const Outcome empty = March(p, true);
      const Outcome ref = March(p, false);
      EXPECT_GT(ref.cfl, 0.0);
      ExpectSameFlow(empty, ref, 1e-8);
   }
}

TEST(EmptyPartition, E3_CheckpointRestart)
{
   // project: the step reference's A x_prev (the projection's image);
   // extrapolate: the extrapolation's x_prev (no A x_prev kept then).
   for (const std::string guess : {"project", "extrapolate"})
   {
      SCOPED_TRACE(guess);
      const Parameters p = Tgv("imex", guess);
      const Outcome ref = March(p, true);

      // Unique per rank count: ctest runs the np 1/2/4 instances concurrently.
      const std::string dir = "chk_empty_" + guess + "_np" +
                              std::to_string(Mpi::WorldSize());
      {
         auto mesh = BoxMesh(p, true);
         Case flow(*mesh, p);
         auto u0 = incns::MakeInitialVelocity(p);
         flow.SetInitialVelocity(*u0);
         while (flow.Time() < 0.15) { flow.Step(); }
         incns::Checkpoint::Write(dir, flow.Integrator(), p);
      }
      Parameters second = p;
      second.restart_from = dir;
      const Outcome restarted = March(second, true);

      EXPECT_EQ(restarted.steps, ref.steps);
      EXPECT_NEAR(restarted.t, ref.t, 1e-12);
      Vector d(restarted.u);
      d -= ref.u;
      const double rd = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d)) /
                        ref.u_norm;
      EXPECT_LE(rd, incns_test::ReproTol(1e-13));
   }
}
