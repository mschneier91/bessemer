// AMR.5 -- checkpoint/restart of an ADAPTED run. The checkpoint stores the
// refinement history; a restart replays it on the initial mesh (same np), which
// reproduces the adapted mesh, its partition and its dof numbering exactly, and
// then continues -- including further events.
//  C1 an interrupted run (checkpoint written right after an event, with
//     rebalancing at np > 1) restarted from its checkpoint matches the
//     uninterrupted run to ReproTol(1e-13): same step count, same final mesh
//     size, same velocity. 2D periodic NSE (convective) and 3D walled Stokes.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "repro_tolerance.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <functional>
#include <memory>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

struct Final
{
   Vector u;
   int steps = 0;
   long long ne = 0;
};

using Setup = std::function<void(Case&, incns::BoundaryConditions&)>;

// March to t_final; optionally stop after @p stop_cycle and write a checkpoint.
Final March(const Parameters& p, const Setup& setup, int stop_cycle = -1,
            const std::string& dir = "")
{
   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   setup(flow, bc);
   int cycle = flow.Integrator().StepCount();
   while (!flow.Done())
   {
      flow.Step();
      ++cycle;
      if (cycle == stop_cycle)
      {
         flow.WriteCheckpoint(dir);
         break;
      }
   }
   Final f;
   f.steps = flow.Integrator().StepCount();
   f.u.SetSize(flow.Spaces().Velocity().GetTrueVSize());
   flow.Velocity().GetTrueDofs(f.u);
   f.ne = mesh->GetGlobalNE();
   return f;
}

void CheckRestart(const Parameters& p, const Setup& setup, int stop_cycle,
                  const std::string& tag)
{
   const Final ref = March(p, setup);
   // Unique per rank count: ctest runs the np 1/2/4 instances concurrently.
   const std::string dir = "chk_amr_" + tag + "_np" +
                           std::to_string(Mpi::WorldSize());
   March(p, setup, stop_cycle, dir);
   Parameters q = p;
   q.restart_from = dir;
   const Final rst = March(q, setup);
   EXPECT_EQ(rst.steps, ref.steps);
   EXPECT_EQ(rst.ne, ref.ne);
   ASSERT_EQ(rst.u.Size(), ref.u.Size());
   Vector d(rst.u);
   d -= ref.u;
   const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d) /
                                InnerProduct(MPI_COMM_WORLD, ref.u, ref.u));
   if (Mpi::Root())
   {
      mfem::out << "[amr checkpoint:" << tag << "] steps " << rst.steps
                << ", elements " << rst.ne << ", reldiff " << rel << std::endl;
   }
   EXPECT_LE(rel, incns_test::ReproTol(1e-13));
}

} // namespace

TEST(AmrCheckpoint, C1_RestartOfAdaptedRunMatchesUninterrupted2D)
{
   if (amr_test::DebugDeviceSkipsPeriodicNcSolves())
   {
      GTEST_SKIP() << "periodic NC solve: MFEM debug-device alias false "
                   "positive (see amr_test_util.hpp / CLAUDE.md)";
   }
   Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.nu = 0.05;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.05;
   p.t_final = 0.6;
   // Fixed steps keep the event/checkpoint cycles on known steps (CFL-mode
   // restart: checkpoint_test).
   p.step_control = incns::StepControl::Fixed;
   p.krylov_rtol = 1e-12;
   p.amr.enabled = true;
   p.amr.interval = 3;
   p.amr.anisotropic = true;
   p.amr.theta = 0.7;
   p.amr.max_elements = 120;
   p.Normalize();
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   CheckRestart(p, [&u0](Case & c, incns::BoundaryConditions&)
   {
      c.SetInitialVelocity(u0);
   }, 6, "2d");
}

TEST(AmrCheckpoint, C1_RestartOfAdaptedRunMatchesUninterrupted3D)
{
   Parameters p;
   p.equation = incns::Equation::Stokes;
   p.nu = 1.3;
   p.order_u = 2;
   p.order_p = 1;
   p.mesh = amr_test::Box(3, 2, false);
   p.dt = 0.02;
   p.t_final = 0.12;
   p.krylov_rtol = 1e-12;
   p.amr.enabled = true;
   p.amr.interval = 2;
   p.amr.anisotropic = true;
   p.amr.theta = 0.6;
   p.amr.max_elements = 60;
   p.Normalize();
   VectorFunctionCoefficient u_ex(3, [](const Vector & x, double t, Vector & v)
   {
      const double g = 1.0 + t + 0.5 * t * t;
      v(0) = g * x[1] * x[1];
      v(1) = g * x[2] * x[2];
      v(2) = g * x[0] * x[0];
   });
   const double nu = p.nu;
   VectorFunctionCoefficient f(3, [nu](const Vector & x, double t, Vector & v)
   {
      const double g = 1.0 + t + 0.5 * t * t, gp = 1.0 + t;
      v(0) = gp * x[1] * x[1] + g * (1.0 - 2.0 * nu);
      v(1) = gp * x[2] * x[2] + g * (1.0 - 2.0 * nu);
      v(2) = gp * x[0] * x[0] + g * (1.0 - 2.0 * nu);
   });
   CheckRestart(p, [&u_ex, &f](Case & c, incns::BoundaryConditions & bc)
   {
      for (int a = 1; a <= 6; ++a) { bc.AddVelocityDirichlet(a, u_ex); }
      c.SetBoundaryConditions(bc);
      c.SetInitialVelocity(u_ex);
      c.SetForcing(f);
   }, 2, "3d"); // checkpoint after the cycle-2 event; the restart adapts again
   // at cycle 4
}
