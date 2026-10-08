// Convective CFL ceiling (time/cfl, amr_spec.md D6).
//  C1 Nek5000's CFL definition: the 1D inverse GLL spacings match Nek's
//     getdr by hand (Q3); for a constant velocity on an affine sheared mesh
//     the rate equals max_K sum_d |(J_K^{-1} u)_d| / dxi_end exactly (the max
//     sits at a corner node, end spacing in every direction), conforming and
//     refined NC (refined cells halve J's split column, doubling that entry);
//  C2 adaptive Navier-Stokes TGV with a binding time.cfl_max: the controller
//     keeps c * dt within the limit, and its steps are smaller than without
//     -- for BOTH the IMEX convective and the semi-implicit rotational form
//     (the latter transports vorticity explicitly; DFG 2D-3 study 2026-10-06);
//  C3 CFL-controlled steps (time.cfl_target, one solve per step) on a
//     decaying NSE TGV, both forms, BDF2/EXT3: every step after the startup
//     pair has c(u^n) dt <= cfl_target, dt never grows by more than the growth
//     factor, it does grow as the flow decays, and the run lands on t_final.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/case.hpp"
#include "time/cfl.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>

using namespace mfem;

namespace
{

const double kShear[3][3] = {{1.0, 0.3, 0.1}, {0.2, 1.1, 0.0}, {0.1, -0.2, 0.9}};
const double kU[3] = {2.0, -0.7, 0.4};

} // namespace

TEST(Cfl, C1_ConstantVelocityOnAffineMeshIsExact)
{
   incns::RuleBook rules;
   for (int dim : {2, 3})
      for (bool refine : {false, true})
      {
         SCOPED_TRACE("dim=" + std::to_string(dim) + (refine ? " refined" : ""));
         Mesh serial = incns::MakeBoxMesh(amr_test::Box(dim, dim == 2 ? 4 : 2, false));
         serial.Transform([dim](const Vector & x, Vector & y)
         {
            y.SetSize(dim);
            for (int i = 0; i < dim; ++i)
            {
               y(i) = 0.0;
               for (int j = 0; j < dim; ++j) { y(i) += kShear[i][j] * x(j); }
            }
         });
         auto mesh = incns::PartitionMesh(serial, true);
         if (refine) { amr_test::RefineBands(*mesh, true); }
         const int k = 3;
         H1_FECollection fec(k, dim);
         ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
         ParGridFunction u(&fes);
         Vector uv(dim);
         for (int i = 0; i < dim; ++i) { uv(i) = kU[i]; }
         VectorConstantCoefficient uc(uv);
         u.ProjectCoefficient(uc);

         incns::ConvectiveCfl cfl(fes, rules);
         const double rate = cfl.Rate(u);

         double ref = 0.0;
         IntegrationPoint c;
         c.Set3(0.5, 0.5, 0.5);
         for (int e = 0; e < mesh->GetNE(); ++e)
         {
            ElementTransformation& T = *mesh->GetElementTransformation(e);
            T.SetIntPoint(&c);
            DenseMatrix Jinv(T.InverseJacobian());
            Vector xi(dim);
            Jinv.Mult(uv, xi);
            double s = 0.0;
            for (int d = 0; d < dim; ++d) { s += std::fabs(xi(d)); }
            ref = std::max(ref, s);
         }
         MPI_Allreduce(MPI_IN_PLACE, &ref, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         ref *= incns::ConvectiveCfl::InverseNodeSpacing(k)(0);
         EXPECT_NEAR(rate, ref, 1e-12 * ref);
      }
   // Q3 GLL points on [0,1]: 0, (1 - 1/sqrt 5)/2, (1 + 1/sqrt 5)/2, 1. Nek's
   // getdr: one-sided at the ends, half central difference inside.
   const Vector inv = incns::ConvectiveCfl::InverseNodeSpacing(3);
   const double z1 = 0.5 * (1.0 - 1.0 / std::sqrt(5.0)), z2 = 1.0 - z1;
   EXPECT_NEAR(inv(0), 1.0 / z1, 1e-12);
   EXPECT_NEAR(inv(1), 1.0 / (0.5 * z2), 1e-12);
   EXPECT_NEAR(inv(2), 1.0 / (0.5 * (1.0 - z1)), 1e-12);
   EXPECT_NEAR(inv(3), 1.0 / z1, 1e-12);
   EXPECT_NEAR(inv(0), 3.6180339887, 1e-9); // = Nek's 2 * 1/0.5528 on [-1,1]
}

namespace
{

void AdaptiveStepRespectsTheCeiling(incns::ConvectiveForm form)
{
   auto run = [form](double cfl_max, double & final_cfl, double & max_dt)
   {
      incns::Parameters p;
      p.equation = incns::Equation::NavierStokes;
      p.convective_form = form;
      p.nu = 0.01;
      p.order_u = 3;
      p.order_p = 2;
      p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
      p.dt = 0.02;
      p.t_final = 0.6;
      p.adaptive = true;
      p.controller.atol = 1e-3; // loose: accuracy alone would take big steps
      p.controller.rtol = 1e-3;
      p.cfl_max = cfl_max;
      p.Normalize();
      auto mesh = incns::MakeCaseMesh(p);
      incns::Case flow(*mesh, p);
      VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
      flow.SetInitialVelocity(u0);
      flow.Run();
      final_cfl = flow.ConvectiveCflNumber();
      max_dt = 0.0;
      for (const incns::StepAttempt& a : flow.Integrator().Controller()->History())
      {
         if (a.accepted) { max_dt = std::max(max_dt, a.dt); }
      }
   };
   double cfl_free = 0.0, dt_free = 0.0, cfl_cap = 0.0, dt_cap = 0.0;
   run(0.0, cfl_free, dt_free);
   const double limit = 0.5 * cfl_free; // binds: the free run went past it
   run(limit, cfl_cap, dt_cap);
   if (Mpi::Root())
   {
      mfem::out << "[cfl] free: cfl=" << cfl_free << " max dt=" << dt_free
                << "   capped (" << limit << "): cfl=" << cfl_cap << " max dt="
                << dt_cap << std::endl;
   }
   EXPECT_LE(cfl_cap, 1.05 * limit);
   EXPECT_LT(dt_cap, dt_free);
}

} // namespace

TEST(Cfl, C2_AdaptiveStepRespectsTheCeiling)
{
   AdaptiveStepRespectsTheCeiling(incns::ConvectiveForm::Convective);
}

TEST(Cfl, C2_RotationalFormRespectsTheCeiling)
{
   AdaptiveStepRespectsTheCeiling(incns::ConvectiveForm::Rotational);
}

namespace
{

void CflControlledSteps(incns::ConvectiveForm form)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = form;
   p.nu = 0.05;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.02;
   p.t_final = 1.0;
   p.cfl_target = 0.3; // Nek-scale CFL (time/cfl.hpp)
   p.ext_order = 3;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   const double growth = incns::TimeIntegratorOptions().cfl_growth;
   double dt_prev = 0.0, dt_first = 0.0, dt_last = 0.0, worst = 0.0;
   int steps = 0;
   while (!flow.Done())
   {
      // c(u^n): the rate the controller sees before the step.
      const double rate_before = flow.ConvectiveCflNumber() /
                                 flow.Integrator().CurrentDt();
      flow.Step();
      ++steps;
      const double dt = flow.Integrator().CurrentDt();
      if (steps > 2)
      {
         worst = std::max(worst, rate_before * dt / p.cfl_target);
         EXPECT_LE(dt, growth * dt_prev * (1.0 + 1e-12)) << "step " << steps;
         if (dt_first == 0.0) { dt_first = dt; }
         dt_last = dt;
      }
      dt_prev = dt;
   }
   if (Mpi::Root())
   {
      mfem::out << "[cfl-steps] form=" << static_cast<int>(form) << " steps="
                << steps << " dt " << dt_first << " -> " << dt_last
                << " max(c dt / target)=" << worst << std::endl;
   }
   EXPECT_LE(worst, 1.0 + 1e-12);
   EXPECT_GT(worst, 0.5); // the target actually binds
   EXPECT_NEAR(flow.Time(), p.t_final, 1e-12);
   // The TGV decays like exp(-2 nu t): the controlled dt grows.
   EXPECT_GT(dt_last, dt_first);
}

} // namespace

TEST(Cfl, C3_CflControlledSteps)
{
   CflControlledSteps(incns::ConvectiveForm::Convective);
   CflControlledSteps(incns::ConvectiveForm::Rotational);
}

