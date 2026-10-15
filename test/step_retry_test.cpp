// A failed implicit solve rejects the step instead of ending the run
// (time.on_solver_failure: cut_step; TimeIntegratorOptions::retry_on_failure).
// Failures are injected with the integrator's verdict hook (SetSolveVeto), so
// the path is exercised deterministically. 2D Taylor-Green, 8x8 Q3/Q2.
//  R1 Fixed steps: two cuts within one step quarter dt; dt is held for that
//     step and five more, then grows by 1.2 per step back to the original and
//     stays there; two rejections are counted; the error at the end is no
//     worse than 1.5 x an undisturbed run's.
//  R2 CFL-controlled steps: one cut halves the attempted dt; it does not grow
//     for five more steps; later it does; one rejection is counted.
//  R3 KrylovDimForMemory: the restart length that fits, by hand.

#include <gtest/gtest.h>

#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <functional>
#include <vector>

using namespace mfem;

namespace
{

struct March
{
   std::vector<double> dts; ///< dt of every accepted step
   double err = 0.0;        ///< velocity L2 error against the exact solution
   int rejected = 0;
};

March Tgv(incns::StepControl control, std::function<bool(int, int)> veto)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.nu = 0.01;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh.dim = 2;
   p.mesh.num_elems = {8, 8, 0};
   p.mesh.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   p.step_control = control;
   p.cfl_target = 0.3;
   p.dt = 0.02;
   p.t_final = 0.6;
   p.on_solver_failure = incns::SolverFailure::CutStep;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   if (veto) { flow.Integrator().SetSolveVeto(veto); }
   March r;
   double t_prev = flow.Time();
   while (!flow.Done())
   {
      flow.Step();
      r.dts.push_back(flow.Time() - t_prev);
      t_prev = flow.Time();
   }
   VectorFunctionCoefficient ue = incns::tgv2d::VelocityCoefficient(p.nu);
   ue.SetTime(flow.Time());
   const IntegrationRule* irs[Geometry::NumGeom] = {};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 14);
   r.err = flow.Velocity().ComputeL2Error(ue, irs);
   r.rejected = flow.Integrator().RejectedSteps();
   return r;
}

} // namespace

TEST(StepRetry, R1_FixedStepsCutHoldAndGrowBack)
{
   const March ref = Tgv(incns::StepControl::Fixed, nullptr);
   EXPECT_EQ(ref.rejected, 0);
   for (double dt : ref.dts) { EXPECT_NEAR(dt, 0.02, 1e-12); }

   // The fifth step (four accepted so far) fails twice.
   const March r = Tgv(incns::StepControl::Fixed, [](int accepted, int attempt)
   {
      return accepted == 4 && attempt < 2;
   });
   EXPECT_EQ(r.rejected, 2);
   ASSERT_GE(r.dts.size(), 20u);
   double expect = 0.005;
   for (std::size_t i = 0; i < r.dts.size(); ++i)
   {
      double want = 0.02;
      if (i >= 4 && i <= 9) { want = 0.005; } // the retried step and five more
      else if (i >= 10)
      {
         expect = std::min(0.02, expect * 1.2);
         want = expect;
      }
      EXPECT_NEAR(r.dts[i], want, 1e-12) << "step " << i;
   }
   if (Mpi::Root())
   {
      mfem::out << "[step retry] fixed: error " << r.err << " against "
                << ref.err << " undisturbed; " << r.dts.size() << " steps"
                << std::endl;
   }
   EXPECT_LT(r.err, 1.5 * ref.err);
}

TEST(StepRetry, R2_CflStepsCutAndHold)
{
   const March ref = Tgv(incns::StepControl::Cfl, nullptr);
   const March r = Tgv(incns::StepControl::Cfl, [](int accepted, int attempt)
   {
      return accepted == 5 && attempt == 0;
   });
   EXPECT_EQ(ref.rejected, 0);
   EXPECT_EQ(r.rejected, 1);
   ASSERT_GE(r.dts.size(), 16u);
   ASSERT_GE(ref.dts.size(), 6u);
   // The same march up to the rejected step, then half of what it tried.
   // (To 1e-6, not to roundoff: the CFL step follows the solution, and on a
   // GPU two runs in one process differ at the solve tolerance -- hypre's
   // coarsening draws from one random stream.)
   for (int i = 0; i < 5; ++i)
   {
      EXPECT_NEAR(r.dts[i], ref.dts[i], 1e-6 * ref.dts[i]);
   }
   const double cut = 0.5 * ref.dts[5];
   EXPECT_NEAR(r.dts[5], cut, 1e-6 * cut);
   for (int i = 6; i <= 10; ++i)
   {
      EXPECT_LE(r.dts[i], cut * (1.0 + 1e-5)) << "step " << i << " grew";
   }
   // The flow decays, so the CFL step wants to grow: after the hold it does.
   double later = 0.0;
   for (std::size_t i = 11; i < r.dts.size() - 1; ++i)
   {
      later = std::max(later, r.dts[i]);
   }
   EXPECT_GT(later, cut * 1.1);
   EXPECT_LT(r.err, 1.5 * ref.err);
}

TEST(StepRetry, R3_KrylovDimForMemory)
{
   using incns::StokesTimeIntegrator;
   const double GiB = 1073741824.0;
   // 80 GiB device, 40 GiB free, 1e7 system entries (80 MB a vector), 9e6
   // velocity dofs. Margin 0.03 * 80 GiB + 6 * 72 MB; (free - margin) / 80 MB
   // = 499.3 vectors, of which 4 are not Krylov pairs: (499.3 - 4) / 2 = 247.
   const double free = 40 * GiB, total = 80 * GiB;
   const double margin = 0.03 * total + 6.0 * 8.0 * 9e6;
   const int expect = static_cast<int>(std::floor(((free - margin) / 8e7 - 4.0) /
                                       2.0));
   EXPECT_EQ(expect, 247);
   EXPECT_EQ(StokesTimeIntegrator::KrylovDimForMemory(free, total, 10000000,
             9000000, 1000),
             expect);
   EXPECT_EQ(StokesTimeIntegrator::KrylovDimForMemory(free, total, 10000000,
             9000000, 200), 200);
   // Nearly full: never below 1.
   EXPECT_EQ(StokesTimeIntegrator::KrylovDimForMemory(1.0 * GiB, total, 10000000,
             9000000, 200), 1);
}
