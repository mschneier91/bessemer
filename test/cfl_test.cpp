// Convective CFL ceiling (time/cfl, docs/design/amr_spec.md D6).
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
//  C4 The Jacobians the rate reads, computed at every call instead of cached
//     in the mesh, equal MFEM's cached ones bit for bit on a curved mesh:
//     dropping the cache changes no rate and no step size.
//  C5 A NaN in the velocity gives an infinite rate, for Rate and MaxRate (a
//     max reduction would skip it and report a finite rate: a diverged run
//     went on until an unrelated check failed), and the finite rate is the
//     same as before; RequireFiniteRate accepts it.
//  C6 MaxRateAt (the per-level cache: a step's history levels keyed by their
//     times) gives MaxRate's value bitwise, computed or from the cache. (A
//     space change clears the cache; the integrator is rebuilt then anyway.)

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
#include <limits>
#include <memory>

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

TEST(Cfl, C4_PerCallJacobiansEqualTheCachedOnes)
{
   for (const int dim : {2, 3})
   {
      SCOPED_TRACE(dim);
      Mesh serial = (dim == 2)
                    ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true)
                    : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
      serial.SetCurvature(3);
      serial.Transform([](const Vector & x, Vector & y)
      {
         y = x;
         y(0) += 0.08 * std::sin(2.3 * x(1));
         y(1) += 0.06 * x(0) * x(0);
         if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
      });
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::RuleBook rules;
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const IntegrationRule& ir =
         rules.Get(geom, 2 * 3 - 1, incns::Rule1D::GaussLobatto);
      const MemoryType mt = Device::GetDeviceMemoryType();
      const GeometricFactors fresh(*mesh.GetNodes(), ir,
                                   GeometricFactors::JACOBIANS, mt);
      const GeometricFactors* cached =
         mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS, mt);
      ASSERT_EQ(fresh.J.Size(), cached->J.Size());
      const double* a = fresh.J.HostRead();
      const double* b = cached->J.HostRead();
      double diff = 0.0, size = 0.0;
      for (int i = 0; i < fresh.J.Size(); ++i)
      {
         diff = std::max(diff, std::abs(a[i] - b[i]));
         size = std::max(size, std::abs(b[i]));
      }
      MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_GT(size, 0.0);
      EXPECT_EQ(diff, 0.0);
   }
}

// Straight-sided elements (vertices moved, linear nodes): the on-the-fly
// rate (the velocity through the gather map, the Jacobians from the vertices)
// against the element copy and MFEM's Jacobians at the nodes, to roundoff;
// curved elements keep the latter.
TEST(Cfl, C7_OnTheFlyRateOnStraightSidedElements)
{
   for (const int dim : {2, 3})
   {
      for (const int k : {3, 6})
      {
         SCOPED_TRACE(testing::Message() << "dim " << dim << ", order " << k);
         for (const bool curved : {false, true})
         {
            Mesh serial = (dim == 2)
                          ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true)
                          : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
            if (curved) { serial.SetCurvature(2); }
            serial.Transform([](const Vector & x, Vector & y)
            {
               y = x;
               y(0) += 0.08 * std::sin(2.3 * x(1));
               y(1) += 0.06 * x(0) * x(0);
               if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
            });
            ParMesh mesh(MPI_COMM_WORLD, serial);
            incns::RuleBook rules;
            H1_FECollection fec(k, dim);
            ParFiniteElementSpace V(&mesh, &fec, dim);
            ParGridFunction u(&V);
            Vector ut(V.GetTrueVSize());
            ut.Randomize(3 + Mpi::WorldRank());
            ut -= 0.5;
            u.SetFromTrueDofs(ut);
            incns::ConvectiveCfl cfl(V, rules);
            EXPECT_EQ(cfl.OnTheFly(), !curved);
            if (curved) { continue; }
            const double fly = cfl.Rate(u);
            cfl.SetOnTheFly(false);
            ASSERT_FALSE(cfl.OnTheFly());
            const double ref = cfl.Rate(u);
            EXPECT_GT(ref, 0.0);
            EXPECT_LE(std::abs(fly - ref), 1e-13 * ref);
         }
      }
   }
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
      p.step_control = incns::StepControl::Error;
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
   p.step_control = incns::StepControl::Cfl;
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

TEST(Cfl, C5_NanVelocityGivesAnInfiniteRate)
{
   incns::RuleBook rules;
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Mesh serial = incns::MakeBoxMesh(amr_test::Box(dim, dim == 2 ? 4 : 2, false));
      auto mesh = incns::PartitionMesh(serial, true);
      H1_FECollection fec(3, dim);
      ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
      ParGridFunction u(&fes), v(&fes);
      Vector uv(dim);
      for (int i = 0; i < dim; ++i) { uv(i) = kU[i]; }
      VectorConstantCoefficient uc(uv);
      u.ProjectCoefficient(uc);
      v = u;

      incns::ConvectiveCfl cfl(fes, rules);
      const double finite = cfl.Rate(u);
      EXPECT_TRUE(std::isfinite(finite));
      EXPECT_GT(finite, 0.0);
      incns::RequireFiniteRate(finite, 0.0); // returns

      // One NaN dof on every rank (every rank owns elements here).
      v.HostReadWrite()[0] = std::numeric_limits<double>::quiet_NaN();
      EXPECT_TRUE(std::isinf(cfl.Rate(v)));
      EXPECT_TRUE(std::isinf(cfl.MaxRate({&u, &v})));
      EXPECT_EQ(cfl.Rate(u), finite); // unchanged for a finite velocity
   }
}

TEST(Cfl, C6_LevelCacheMatchesMaxRateBitwise)
{
   incns::RuleBook rules;
   Mesh serial = incns::MakeBoxMesh(amr_test::Box(3, 2, false));
   auto mesh = incns::PartitionMesh(serial, true);
   H1_FECollection fec(3, 3);
   ParFiniteElementSpace fes(mesh.get(), &fec, 3, Ordering::byNODES);
   std::vector<std::unique_ptr<ParGridFunction>> u;
   for (int l = 0; l < 4; ++l)
   {
      u.push_back(std::make_unique<ParGridFunction>(&fes));
      VectorFunctionCoefficient f(3, [l](const Vector & x, Vector & y)
      {
         y.SetSize(3);
         y(0) = std::sin(x(1) + 0.3 * l);
         y(1) = 0.5 * std::cos(x(2) - l);
         y(2) = 0.2 * (1 + l) * x(0);
      });
      u[l]->ProjectCoefficient(f);
   }
   incns::ConvectiveCfl cfl(fes, rules);
   // A step's three levels, then the next step's (one new, two cached).
   const double a = cfl.MaxRateAt({u[0].get(), u[1].get(), u[2].get()}, {0.3, 0.2, 0.1});
   EXPECT_EQ(a, cfl.MaxRate({u[0].get(), u[1].get(), u[2].get()}));
   const double b = cfl.MaxRateAt({u[3].get(), u[0].get(), u[1].get()}, {0.4, 0.3, 0.2});
   EXPECT_EQ(b, cfl.MaxRate({u[3].get(), u[0].get(), u[1].get()}));
   EXPECT_EQ(cfl.MaxRateAt({u[3].get()}, {0.4}), cfl.Rate(*u[3]));
   EXPECT_EQ(cfl.MaxRateAt({u[2].get()}, {0.1}), cfl.Rate(*u[2]));
   // Cached: the same value again, without an evaluation.
   EXPECT_EQ(cfl.MaxRateAt({u[0].get(), u[1].get()}, {0.3, 0.2}),
             cfl.MaxRate({u[0].get(), u[1].get()}));
}
