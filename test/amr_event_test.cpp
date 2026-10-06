// AMR.4 -- adaptation events through the Case surface.
//  E1 the unsteady polynomial MMS (exact in space and time, time-dependent
//     Dirichlet) stays exact at EVERY step through anisotropic events -- the
//     exact solution lies in both spaces and the transfer is exact -- 2D/3D,
//     fixed-step and adaptive;
//  E2 a no-op event (nothing marked, the export/rebuild/import path forced)
//     reproduces the uninterrupted run to ReproTol(1e-13): Stokes adaptive and
//     NSE rotational (whose warm start is the Bernoulli head);
//  E3 (inside E2) the adaptive controller's step record and PI memory survive;
//  E4 the history projection makes transferred velocity discretely
//     divergence-free on the refined mesh, changing it only slightly;
//  E5 initial passes refine on the IC and the march starts from the IC
//     projected on the final mesh.

#include <gtest/gtest.h>

#include "amr/history_projection.hpp"
#include "amr/mesh_adapter.hpp"
#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "operators/stokes_operator.hpp"
#include "repro_tolerance.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }

void ElevatedRules(int order, const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, order);
   irs[Geometry::CUBE] = &IntRules.Get(Geometry::CUBE, order);
}

// The unsteady polynomial MMS of unsteady_mms_test (2D Q3/Q2, 3D Q2/Q1).
struct Mms
{
   int dim;
   double nu;
   std::unique_ptr<VectorFunctionCoefficient> u;
   std::unique_ptr<FunctionCoefficient> p;
   std::unique_ptr<VectorFunctionCoefficient> f;
   explicit Mms(int d) : dim(d), nu(d == 2 ? 0.7 : 1.3)
   {
      const double n = nu;
      if (dim == 2)
      {
         u = std::make_unique<VectorFunctionCoefficient>(2, [](const Vector & x,
             double t, Vector & v)
         {
            v(0) = G(t) * 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
            v(1) = -G(t) * 3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
         });
         p = std::make_unique<FunctionCoefficient>([](const Vector & x, double t)
         {
            return G(t) * (x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0);
         });
         f = std::make_unique<VectorFunctionCoefficient>(2, [n](const Vector & x,
             double t, Vector & v)
         {
            const double us0 = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
            const double us1 = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
            const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
            const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
            v(0) = Gp(t) * us0 + G(t) * (-n * lap0 + 2.0 * x[0]);
            v(1) = Gp(t) * us1 + G(t) * (-n * lap1 + 2.0 * x[1]);
         });
      }
      else
      {
         u = std::make_unique<VectorFunctionCoefficient>(3, [](const Vector & x,
             double t, Vector & v)
         {
            v(0) = G(t) * x[1] * x[1];
            v(1) = G(t) * x[2] * x[2];
            v(2) = G(t) * x[0] * x[0];
         });
         p = std::make_unique<FunctionCoefficient>([](const Vector & x, double t)
         {
            return G(t) * (x[0] + x[1] + x[2] - 1.5);
         });
         f = std::make_unique<VectorFunctionCoefficient>(3, [n](const Vector & x,
             double t, Vector & v)
         {
            v(0) = Gp(t) * x[1] * x[1] + G(t) * (1.0 - 2.0 * n);
            v(1) = Gp(t) * x[2] * x[2] + G(t) * (1.0 - 2.0 * n);
            v(2) = Gp(t) * x[0] * x[0] + G(t) * (1.0 - 2.0 * n);
         });
      }
   }
};

Parameters MmsParams(int dim, bool adaptive)
{
   Parameters p;
   p.equation = incns::Equation::Stokes;
   p.nu = (dim == 2) ? 0.7 : 1.3;
   p.order_u = (dim == 2) ? 3 : 2;
   p.order_p = p.order_u - 1;
   p.mesh = amr_test::Box(dim, dim == 2 ? 3 : 2, false);
   p.dt = 0.02;
   p.t_final = 0.12;
   p.adaptive = adaptive;
   p.controller.atol = 1e-8;
   p.controller.rtol = 1e-6;
   p.krylov_rtol = 1e-12;
   p.max_iter = 5000;
   p.kdim = 400;
   p.amr.enabled = true;
   p.amr.interval = 2;
   p.amr.anisotropic = true;
   p.amr.theta = 0.6;
   p.Normalize();
   return p;
}

} // namespace

TEST(AmrEvent, E1_UnsteadyMmsStaysExactThroughEvents)
{
   for (int dim : {2, 3})
      for (bool adaptive : {false, true})
      {
         SCOPED_TRACE("dim=" + std::to_string(dim) + (adaptive ? " adaptive" :
                      " fixed"));
         Mms mms(dim);
         const Parameters params = MmsParams(dim, adaptive);
         auto mesh = incns::MakeCaseMesh(params);
         Case flow(*mesh, params);
         incns::BoundaryConditions bc(flow.Spaces().Velocity());
         for (int a = 1; a <= 2 * dim; ++a) { bc.AddVelocityDirichlet(a, *mms.u); }
         flow.SetBoundaryConditions(bc);
         flow.SetInitialVelocity(*mms.u);
         flow.SetForcing(*mms.f);

         const IntegrationRule* irs[Geometry::NumGeom];
         ElevatedRules(2 * params.order_u + 4, irs);
         int step = 0;
         while (!flow.Done())
         {
            flow.Step();
            ++step;
            mms.u->SetTime(flow.Time());
            mms.p->SetTime(flow.Time());
            EXPECT_LE(flow.Velocity().ComputeL2Error(*mms.u, irs), 1e-8)
                  << "step " << step << " t=" << flow.Time();
            if (step >= 2)
            {
               EXPECT_LE(flow.Pressure().ComputeL2Error(*mms.p, irs), 1e-7)
                     << "step " << step;
            }
         }
         // Events happened and refined.
         int refining = 0;
         for (const incns::AdaptStats& s : flow.AdaptHistory())
         {
            if (s.ne_after > s.ne_before) { ++refining; }
         }
         EXPECT_GE(refining, 1);
      }
}

namespace
{

struct RunResult
{
   Vector u;
   int steps = 0;
   std::vector<double> dts;
};

// Fully periodic 2D TGV through the Case; optionally a forced no-op event.
RunResult RunTgv(bool rotational, bool adaptive, int noop_event_at)
{
   Parameters p;
   p.equation = rotational ? incns::Equation::NavierStokes
                : incns::Equation::Stokes;
   p.convective_form = rotational ? incns::ConvectiveForm::Rotational
                       : incns::ConvectiveForm::Convective;
   p.nu = 0.05;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.05;
   p.t_final = 0.5;
   p.adaptive = adaptive;
   p.controller.atol = 1e-6;
   p.controller.rtol = 1e-4;
   p.krylov_rtol = 1e-12;
   p.amr.enabled = true;
   p.amr.interval = 0;               // only the forced event
   p.amr.threshold_mode = incns::AmrThreshold::Absolute;
   p.amr.tolerance = 1e30;           // marks nothing
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   int steps = 0;
   while (!flow.Done())
   {
      flow.Step();
      ++steps;
      if (steps == noop_event_at)
      {
         const incns::AdaptStats st = flow.Adapt(/*force_rebuild=*/true);
         EXPECT_TRUE(st.rebuilt);
         EXPECT_EQ(st.ne_after, st.ne_before);
      }
   }
   RunResult r;
   r.steps = steps;
   r.u.SetSize(flow.Spaces().Velocity().GetTrueVSize());
   flow.Velocity().GetTrueDofs(r.u);
   if (const incns::AdaptiveController* c = flow.Integrator().Controller())
   {
      for (const incns::StepAttempt& a : c->History()) { r.dts.push_back(a.dt); }
   }
   return r;
}

} // namespace

TEST(AmrEvent, E2_NoOpEventReproducesUninterruptedRun)
{
   struct Cfg { bool rotational, adaptive; } cfgs[] = {{false, true},
      {true, false}
   };
   for (const Cfg& c : cfgs)
   {
      SCOPED_TRACE(std::string(c.rotational ? "nse rotational" : "stokes") +
                   (c.adaptive ? " adaptive" : " fixed"));
      const RunResult ref = RunTgv(c.rotational, c.adaptive, -1);
      const RunResult ev = RunTgv(c.rotational, c.adaptive, 3);
      EXPECT_EQ(ev.steps, ref.steps);
      Vector d(ev.u);
      d -= ref.u;
      const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d) /
                                   InnerProduct(MPI_COMM_WORLD, ref.u, ref.u));
      EXPECT_LE(rel, incns_test::ReproTol(1e-13));
      // E3: the controller record (every attempted dt) survived the event.
      ASSERT_EQ(ev.dts.size(), ref.dts.size());
      for (std::size_t i = 0; i < ref.dts.size(); ++i)
      {
         EXPECT_NEAR(ev.dts[i], ref.dts[i], 1e-12 * ref.dts[i]) << "attempt " << i;
      }
   }
}

TEST(AmrEvent, E4_HistoryProjectionRemovesDiscreteDivergence)
{
   // TGV velocity on a coarse periodic mesh, refined in a band: the
   // transferred field is not discretely divergence-free on the new mesh.
   incns::RuleBook rules;
   const double nu = 0.05;
   auto mesh = amr_test::NcBox(amr_test::Box(2, 4, true, 2.0 * M_PI));
   incns::MixedSpaces spaces(*mesh, 3, 2);
   incns::BoundaryConditions bc(spaces.Velocity());
   ParGridFunction u(&spaces.Velocity());
   VectorFunctionCoefficient uc = incns::tgv2d::VelocityCoefficient(nu);
   u.ProjectCoefficient(uc);
   {
      Vector t;
      u.GetTrueDofs(t);
      u.SetFromTrueDofs(t);
   }
   Array<Refinement> refs;
   Vector c(2);
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      mesh->GetElementCenter(e, c);
      if (c(0) < 2.0) { refs.Append(Refinement(e, Refinement::XY)); }
   }
   incns::MeshAdapter::Refine(*mesh, spaces, &bc, refs, 1, false, {&u});

   incns::IntegratorState state;
   state.u_hist.resize(1);
   u.GetTrueDofs(state.u_hist[0]);
   state.times = {0.0};
   const Vector before = state.u_hist[0];

   incns::StokesOperator op(spaces, rules);
   auto div_norm = [&](const Vector & v)
   {
      Vector bv(spaces.Pressure().GetTrueVSize());
      op.Divergence().Mult(v, bv);
      return std::sqrt(InnerProduct(MPI_COMM_WORLD, bv, bv));
   };
   const double div0 = div_norm(before);
   incns::ProjectHistoryDivergenceFree(spaces, rules, bc, state);
   const double div1 = div_norm(state.u_hist[0]);
   Vector d(state.u_hist[0]);
   d -= before;
   const double change = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d) /
                                   InnerProduct(MPI_COMM_WORLD, before, before));
   if (Mpi::Root())
   {
      mfem::out << "[e4] ||B u|| transferred = " << div0 << ", projected = "
                << div1 << ", relative change = " << change << std::endl;
   }
   EXPECT_GT(div0, 1e-8);
   EXPECT_LE(div1, 1e-9 * div0);
   EXPECT_LE(change, 1e-2);
}

TEST(AmrEvent, E5_InitialPassesStartFromTheProjectedIc)
{
   Parameters p;
   p.nu = 0.05;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.05;
   p.t_final = 0.1;
   p.amr.enabled = true;
   p.amr.interval = 0;
   p.amr.initial_passes = 2;
   p.amr.theta = 0.8;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   // Velocity() triggers setup -- the initial passes, then the integrator --
   // without a solve (stepping on a refined periodic mesh is E1/E2's job).
   ParGridFunction& u = flow.Velocity();

   const auto& log = flow.AdaptHistory();
   ASSERT_EQ(log.size(), 2u);
   EXPECT_GT(log[0].ne_after, log[0].ne_before);
   EXPECT_GT(log[1].ne_after, log[1].ne_before);
   EXPECT_EQ(mesh->GetGlobalNE(), log[1].ne_after);
   // The velocity space lives on the final mesh, and the march starts from the
   // IC projected (conformingly) there.
   EXPECT_EQ(flow.Spaces().Velocity().GetParMesh(), mesh.get());
   EXPECT_TRUE(mesh->Nonconforming());
   ParGridFunction ref(&flow.Spaces().Velocity());
   u0.SetTime(0.0);
   ref.ProjectCoefficient(u0);
   Vector t;
   ref.GetTrueDofs(t);
   ref.SetFromTrueDofs(t);
   ref -= u;
   double d = ref.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LE(d, 1e-14);
}
