// OIFS (time/oifs.hpp): the BDF history advected by RK4 substeps.
//  O1 the advector alone: a smooth periodic field carried by a constant wind
//     matches the exact translation (spatial accuracy of the lumped-mass,
//     dealiased advection), and its time error against a tiny-substep
//     reference falls like the 4th power of the substep (RK4);
//  O2 the point of OIFS: the periodic NSE Taylor-Green vortex (exact, nu =
//     0.01, coarse 8x8 Q3 over 2 pi) under CFL-controlled steps at CFL 2 and
//     4 -- beyond the IMEX limit (~0.7) -- runs stably in 3.5-4x fewer steps
//     than IMEX at CFL 0.5, AS ACCURATELY, with the OIFS production scheme
//     (BDF3 + the collocated mass, both automatic under OIFS; human
//     2026-10-09). Measured 2026-10-09: IMEX 0.097 (38 steps); OIFS 0.074
//     (11) / 0.072 (9); with the consistent BDF mass 0.085 / 0.079; BDF2 +
//     consistent mass was 0.164 / 0.204 (its trajectory error at steps 4-8x
//     larger).

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/oifs.hpp"
#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <deque>
#include <vector>

using namespace mfem;

namespace
{

incns::BoxSpec PeriodicUnitBox(int n)
{
   incns::BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {true, true, true};
   return s;
}

} // namespace

TEST(Oifs, O1_AdvectionTranslatesWithRk4TimeOrder)
{
   incns::RuleBook rules; // before the mesh (rule-keyed caches)
   Mesh serial = incns::MakeBoxMesh(PeriodicUnitBox(12));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::MixedSpaces spaces(mesh, 3, 2);
   incns::BoundaryConditions bc(spaces.Velocity()); // fully periodic
   ParFiniteElementSpace& V = spaces.Velocity();
   const double wx = 1.0, wy = 0.5, T = 0.5;

   auto field = [](const Vector & x, double t, Vector & f, double ax, double ay)
   {
      const double X = x(0) - ax * t, Y = x(1) - ay * t;
      f(0) = std::sin(2.0 * M_PI * X) * std::cos(2.0 * M_PI * Y);
      f(1) = 0.5 * std::cos(2.0 * M_PI * X) * std::sin(4.0 * M_PI * Y);
   };
   auto to_true = [&](VectorCoefficient & coef)
   {
      ParGridFunction g(&V);
      g.ProjectCoefficient(coef);
      Vector t(V.GetTrueVSize());
      g.GetTrueDofs(t);
      return t;
   };
   VectorFunctionCoefficient wind_c(2, [&](const Vector&, Vector & w)
   {
      w(0) = wx;
      w(1) = wy;
   });
   VectorFunctionCoefficient f0_c(2, [&](const Vector & x, Vector & f)
   {
      field(x, 0.0, f, wx, wy);
   });
   VectorFunctionCoefficient fT_c(2, [&](const Vector & x, Vector & f)
   {
      field(x, T, f, wx, wy);
   });
   const std::deque<Vector> wind = {to_true(wind_c)};
   const std::deque<Vector> fields = {to_true(f0_c)};

   auto advect = [&](double sub_cfl, int& substeps)
   {
      incns::OifsAdvector adv(spaces, rules, bc, sub_cfl);
      Vector phi;
      adv.AdvectWith(wind, {0.0}, {0.0, 1.0}, fields, {0.0}, T, phi);
      substeps = adv.LastSubsteps();
      return phi;
   };
   int n_ref = 0;
   const Vector ref = advect(0.02, n_ref);
   const double ref_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, ref, ref));

   // Spatial accuracy: the reference against the exact translation.
   ParGridFunction g(&V);
   g.SetFromTrueDofs(ref);
   const IntegrationRule* irs[Geometry::NumGeom] = {};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 12);
   const double l2 = g.ComputeL2Error(fT_c, irs);
   if (Mpi::Root())
   {
      mfem::out << "[oifs O1] reference (" << n_ref << " substeps) L2 error vs "
                "exact translation: " << l2 << "\n";
   }
   EXPECT_LT(l2, 2e-3);

   // Time order: errors against the reference at three substep sizes.
   std::vector<double> err, ds;
   for (double cfl : {0.8, 0.4, 0.2})
   {
      int n = 0;
      Vector phi = advect(cfl, n);
      phi -= ref;
      err.push_back(std::sqrt(InnerProduct(MPI_COMM_WORLD, phi, phi)) / ref_norm);
      ds.push_back(T / n);
   }
   for (std::size_t i = 1; i < err.size(); ++i)
   {
      const double order = std::log(err[i - 1] / err[i]) / std::log(
                              ds[i - 1] / ds[i]);
      if (Mpi::Root())
      {
         mfem::out << "[oifs O1] ds " << ds[i] << " time error " << err[i]
                   << " observed order " << order << "\n";
      }
      EXPECT_GT(order, 3.5);
   }
}

namespace
{

struct TgvRun
{
   double err = 0.0; ///< velocity L2 error vs the exact TGV at t_final
   int steps = 0;
};

TgvRun Tgv(incns::ConvectionTreatment conv, double cfl)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convection_treatment = conv;
   p.nu = 0.01;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = PeriodicUnitBox(8);
   p.mesh.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   p.step_control = incns::StepControl::Cfl;
   p.cfl_target = cfl;
   p.dt = 0.25;      // the first step is capped by the target anyway
   p.t_final = 4.0;  // long enough that the x1.2 growth limit is not the story
   p.krylov_rtol = 1e-12;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   TgvRun r;
   while (!flow.Done()) { flow.Step(); ++r.steps; }
   VectorFunctionCoefficient ue = incns::tgv2d::VelocityCoefficient(p.nu);
   ue.SetTime(flow.Time());
   const IntegrationRule* irs[Geometry::NumGeom] = {};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 14);
   r.err = flow.Velocity().ComputeL2Error(ue, irs);
   return r;
}

} // namespace

TEST(Oifs, O2_TgvBeyondTheImexCflLimit)
{
   using incns::ConvectionTreatment;
   const TgvRun imex = Tgv(ConvectionTreatment::Imex, 0.5);
   const TgvRun oifs2 = Tgv(ConvectionTreatment::Oifs, 2.0);
   const TgvRun oifs4 = Tgv(ConvectionTreatment::Oifs, 4.0);
   if (Mpi::Root())
   {
      mfem::out << "[oifs O2] TGV nu 0.01, t = 4: IMEX CFL 0.5 err " << imex.err
                << " (" << imex.steps << " steps); OIFS CFL 2 err " << oifs2.err
                << " (" << oifs2.steps << "); OIFS CFL 4 err " << oifs4.err
                << " (" << oifs4.steps << ")\n";
   }
   EXPECT_LT(oifs2.err, 1.2 * imex.err);
   EXPECT_LT(oifs4.err, 1.2 * imex.err);
   EXPECT_LE(3 * oifs2.steps, imex.steps);
   EXPECT_LE(4 * oifs4.steps, imex.steps);
}


// O3 the row-sum-lumped substep mass on a NONCONFORMING (AMR) mesh: the
// periodic translation of O1 on the 12x12 mesh with a band of elements
// refined once (hanging nodes on both band edges) is no less accurate than
// on the conforming 12x12 mesh (measured 2026-10-09: 4.08e-4 vs 4.23e-4;
// uniform 24x24: 2.4e-5). Guards the lumping on hanging-node meshes, where
// the square-cylinder runs live.
TEST(Oifs, O3_NonconformingAdvection)
{
   if (amr_test::DebugDeviceSkipsPeriodicNcSolves())
   {
      GTEST_SKIP() << "periodic NC mesh: MFEM debug-device alias false "
                   "positive (CLAUDE.md section 6)";
   }
   const double wx = 1.0, wy = 0.5, T = 0.5;
   auto field = [](const Vector & x, double t, Vector & f, double ax, double ay)
   {
      const double X = x(0) - ax * t, Y = x(1) - ay * t;
      f(0) = std::sin(2.0 * M_PI * X) * std::cos(2.0 * M_PI * Y);
      f(1) = 0.5 * std::cos(2.0 * M_PI * X) * std::sin(4.0 * M_PI * Y);
   };
   double err[2] = {0.0, 0.0};
   for (int mode : {0, 1}) // 0 conforming, 1 a band refined (hanging nodes)
   {
      incns::RuleBook rules;
      Mesh serial = incns::MakeBoxMesh(PeriodicUnitBox(12));
      if (mode == 1)
      {
         serial.EnsureNCMesh();
         Array<int> els;
         for (int e = 0; e < serial.GetNE(); ++e)
         {
            Vector c(2);
            serial.GetElementCenter(e, c);
            if (c(0) > 0.3 && c(0) < 0.7) { els.Append(e); }
         }
         serial.GeneralRefinement(els, 0, 1);
      }
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, 3, 2);
      incns::BoundaryConditions bc(spaces.Velocity());
      ParFiniteElementSpace& V = spaces.Velocity();
      auto to_true = [&](VectorCoefficient & coef)
      {
         ParGridFunction g(&V);
         g.ProjectCoefficient(coef);
         Vector t(V.GetTrueVSize());
         g.GetTrueDofs(t);
         return t;
      };
      VectorFunctionCoefficient wind_c(2, [&](const Vector&, Vector & w) { w(0) = wx; w(1) = wy; });
      VectorFunctionCoefficient f0_c(2, [&](const Vector & x, Vector & f) { field(x, 0.0, f, wx, wy); });
      VectorFunctionCoefficient fT_c(2, [&](const Vector & x, Vector & f) { field(x, T, f, wx, wy); });
      const std::deque<Vector> wind = {to_true(wind_c)};
      const std::deque<Vector> fields = {to_true(f0_c)};
      incns::OifsAdvector adv(spaces, rules, bc, 0.05);
      Vector phi;
      adv.AdvectWith(wind, {0.0}, {0.0, 1.0}, fields, {0.0}, T, phi);
      ParGridFunction g(&V);
      g.SetFromTrueDofs(phi);
      const IntegrationRule* irs[Geometry::NumGeom] = {};
      irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 12);
      const double l2 = g.ComputeL2Error(fT_c, irs);
      err[mode] = l2;
   }
   if (Mpi::Root())
   {
      mfem::out << "[oifs O3] translation L2 error: conforming " << err[0]
                << ", band refined (nonconforming) " << err[1] << "\n";
   }
   EXPECT_LT(err[1], 1.1 * err[0]);
}
