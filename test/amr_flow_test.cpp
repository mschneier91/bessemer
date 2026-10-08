// AMR.6 -- AMR in actual flows.
//  T1 Navier-Stokes TGV (convective and rotational) with adaptation events is
//     at least as accurate as the same run without AMR: kinetic energy vs the
//     analytic decay, and the final velocity error -- i.e. events (transfer,
//     history projection, rebuild) inject no error of their own, and the
//     refinement pays off (nu = 0.2, where uniform refinement converges);
//  A1 anisotropy pays: a boundary-layer field u = (tanh((y - 1/2)/delta), 0)
//     refined by initial passes splits only across the layer (no X splits),
//     and reaches the isotropic run's indicator level with far fewer elements.

#include <gtest/gtest.h>

#include "amr/gradient_indicator.hpp"
#include "amr_test_util.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "post/diagnostics.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

struct TgvRun
{
   double worst_ke = 0.0; ///< max over steps of |KE - KE_exact| / KE_exact
   double u_err = 0.0;    ///< final velocity L2 error
   long long ne = 0;      ///< final element count
};

TgvRun RunTgv(incns::ConvectiveForm form, bool amr)
{
   Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = form;
   // nu = 0.2: refinement must reduce the error here. (At nu = 0.05 the base
   // convective-form run is pre-asymptotic on these meshes -- uniform 4x4 ->
   // 8x8 Q3 RAISES its t = 0.4 error, 0.012 -> 0.024, before 16x16 drops it to
   // 0.0027 -- so "AMR is no worse" would not be a statement about AMR.)
   p.nu = 0.2;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.02;
   p.t_final = 0.4;
   // Fixed steps: the AMR and uniform runs then share the temporal error, so
   // the comparison is about space (CFL control would shrink the AMR run's dt).
   p.step_control = incns::StepControl::Fixed;
   p.krylov_rtol = 1e-12;
   p.amr.enabled = amr;
   p.amr.interval = 4;
   p.amr.theta = 0.6;
   p.amr.max_elements = 100;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   TgvRun r;
   while (!flow.Done())
   {
      flow.Step();
      const double ke = flow.KineticEnergy();
      const double ke_ex = incns::tgv2d::KineticEnergy(flow.Time(), p.nu);
      r.worst_ke = std::max(r.worst_ke, std::abs(ke - ke_ex) / ke_ex);
   }
   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 10);
   u0.SetTime(flow.Time());
   r.u_err = flow.Velocity().ComputeL2Error(u0, irs);
   r.ne = mesh->GetGlobalNE();
   return r;
}

} // namespace

TEST(AmrFlow, T1_TgvWithEventsIsNoLessAccurate)
{
   if (amr_test::DebugDeviceSkipsPeriodicNcSolves())
   {
      GTEST_SKIP() << "periodic NC solve: MFEM debug-device alias false "
                   "positive (see amr_test_util.hpp / CLAUDE.md)";
   }
   for (incns::ConvectiveForm form :
        {
           incns::ConvectiveForm::Convective,
           incns::ConvectiveForm::Rotational
        })
   {
      const char* name = (form == incns::ConvectiveForm::Convective)
                         ? "convective" : "rotational";
      SCOPED_TRACE(name);
      const TgvRun base = RunTgv(form, false), adapted = RunTgv(form, true);
      if (Mpi::Root())
      {
         mfem::out << "[amr tgv " << name << "] no AMR: ke drift "
                   << base.worst_ke << ", u_err " << base.u_err << ", NE "
                   << base.ne << "   AMR: ke drift " << adapted.worst_ke
                   << ", u_err " << adapted.u_err << ", NE " << adapted.ne
                   << std::endl;
      }
      EXPECT_GT(adapted.ne, base.ne); // events refined
      EXPECT_LE(adapted.worst_ke, 1.05 * base.worst_ke);
      EXPECT_LE(adapted.u_err, 1.05 * base.u_err);
   }
}

TEST(AmrFlow, A1_AnisotropicRefinementResolvesALayerCheaply)
{
   struct Result { long long ne; double max_eta; long long x_splits; };
   auto run = [](bool aniso)
   {
      Parameters p;
      p.nu = 0.01;
      p.order_u = 3;
      p.order_p = 2;
      p.mesh = amr_test::Box(2, 4, false);
      p.amr.enabled = true;
      p.amr.interval = 0;
      p.amr.initial_passes = 4;
      p.amr.anisotropic = aniso;
      p.amr.theta = 0.5;
      p.Normalize();
      auto mesh = incns::MakeCaseMesh(p);
      Case flow(*mesh, p);
      VectorFunctionCoefficient u0(2, [](const Vector & x, Vector & v)
      {
         v(0) = std::tanh((x(1) - 0.5) / 0.05);
         v(1) = 0.0;
      });
      flow.SetInitialVelocity(u0);
      ParGridFunction& u = flow.Velocity(); // setup: the initial passes
      incns::RuleBook rules;
      incns::GradientIndicator ind(flow.Spaces().Velocity(), rules);
      Vector g, eta;
      ind.Compute(u, g);
      incns::GradientIndicator::Eta(g, 2, eta);
      double mx = eta.Size() ? eta.Max() : 0.0;
      MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      Result r{mesh->GetGlobalNE(), mx, 0};
      for (const incns::AdaptStats& s : flow.AdaptHistory())
      {
         r.x_splits += s.first_pass.per_dir[0];
      }
      return r;
   };
   const Result aniso = run(true), iso = run(false);
   if (Mpi::Root())
   {
      mfem::out << "[amr aniso] anisotropic: NE " << aniso.ne << ", max eta "
                << aniso.max_eta << ", X splits " << aniso.x_splits
                << "   isotropic: NE " << iso.ne << ", max eta " << iso.max_eta
                << std::endl;
   }
   EXPECT_EQ(aniso.x_splits, 0);                 // only split across the layer
   EXPECT_LE(aniso.max_eta, 1.05 * iso.max_eta); // same resolution of it...
   EXPECT_LT(aniso.ne, iso.ne / 2);              // ...at under half the cells
}
