// Sprint 2.2 -- the 2D TGV oracle for the NSE path (convection ON).
//
// WHY THIS EXISTS ALONGSIDE nse_mms_test
// --------------------------------------
// The NSE MMS puts the convective term into a MANUFACTURED FORCING: the solver
// computes N(u) and the forcing cancels it. That is a strong test of the
// discretization, but its exact solution is one the Stokes path also reproduces,
// and the convective contribution reaches the residual through f. Here the
// forcing is IDENTICALLY ZERO. The 2D TGV velocity solves UNFORCED Navier-Stokes
// exactly, because (u.grad)u is a pure gradient absorbed into the pressure:
//
//     (u.grad)u + grad(p) = 0,   p = (cos 2x + cos 2y)/4 * e^{-4 nu t}
//
// So with convection ON and f = 0, the scheme must reproduce the SAME velocity
// as the Stokes TGV while producing a NONZERO pressure that matches p above.
// Nothing about that can be faked by a convection operator that is silently
// zero, mis-scaled, or cancelling against a forcing -- there is no forcing.
// (Verified numerically before writing this test: |(u.grad)u + grad p| < 1e-10
// and the unforced NSE residual < 1e-10 at 200 random points.)
//
// This is also the harness Sprint 2.2b wants. The skew-symmetric and rotational
// forms differ from the convective form in their DISCRETE ENERGY behaviour, not
// in their consistency, so a form comparison needs a case with a known analytic
// energy history to measure drift against. That is EnergyDecayMatchesAnalytic
// below; on the periodic box with div u = 0 the analytic history is
//
//     KE(t)  = pi^2 e^{-4 nu t},   eps(t) = 4 nu pi^2 e^{-4 nu t},
//     d(KE)/dt = -eps(t)   exactly.
//
// Both closed forms were checked against 200x200 Gauss-Legendre quadrature to
// ~1e-16 relative before use (see [[feedback-verify-mms-forcing-numerically]]).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/periodic_box.hpp"
#include "post/diagnostics.hpp"
#include "post/pressure_mean.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
constexpr double kNu = 0.05;   // mild viscosity: convection stays significant
constexpr double kTFinal = 0.5;

// Fully periodic [0,2pi]^2 box -- the domain the analytic TGV is posed on, and
// the reason there are no velocity Dirichlet BCs anywhere in this file.
BoxSpec PeriodicBox(int n)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, n};
   s.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   s.periodic = {true, true, true};
   return s;
}

struct TgvResult
{
   double u_err = 0.0;      // ||u - u_exact||_L2 at t_final
   double p_err = 0.0;      // ||p - p_exact||_L2 at t_final (zero-mean matched)
   double p_norm = 0.0;     // ||p||_L2 at t_final
   std::vector<double> times;
   std::vector<double> energy;
   std::vector<double> dissipation;
   std::vector<double> divergence;
};

// March the UNFORCED TGV with convection on/off. Zero forcing throughout: the
// convective term is balanced by the pressure, not by f.
TgvResult MarchTgv(int n, int ku, double dt, bool convection,
                   bool record_history = false)
{
   Mesh serial = MakeBoxMesh(PeriodicBox(n));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity()); // periodic: no essential dofs

   Vector zero_vec(2);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_forcing(zero_vec);

   TimeIntegratorOptions opts;
   opts.nu = kNu;
   opts.dt = dt;
   opts.t_final = kTFinal;
   opts.convection = convection;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesTimeIntegrator stepper(spaces, rules, bc, zero_forcing, opts);

   VectorFunctionCoefficient u0(2, [](const Vector & x, double t, Vector & u)
   { incns::tgv2d::Velocity(x, t, kNu, u); });
   stepper.SetInitialVelocity(u0);

   TgvResult r;
   auto record = [&]()
   {
      r.times.push_back(stepper.Time());
      r.energy.push_back(incns::KineticEnergy(stepper.Velocity(), rules));
      r.dissipation.push_back(
         incns::DissipationRate(stepper.Velocity(), kNu, rules));
      r.divergence.push_back(
         incns::DivergenceNorm(stepper.Velocity(), rules));
   };

   if (record_history)
   {
      record();
      const int max_steps = static_cast<int>(std::ceil(kTFinal / dt)) + 2;
      int steps = 0;
      while (stepper.Time() < kTFinal - 1e-12 && steps < max_steps)
      {
         stepper.Step();
         ++steps;
         record();
      }
   }
   else
   {
      stepper.Run();
   }
   EXPECT_NEAR(stepper.Time(), kTFinal, 1e-10);

   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, 2 * ku + 4);

   u0.SetTime(stepper.Time());
   r.u_err = stepper.Velocity().ComputeL2Error(u0, irs);

   // The discrete pressure is fixed only up to the constant nullspace mode on a
   // fully periodic box, and the analytic p already has zero mean (checked
   // numerically), so normalize before comparing: the test measures pressure
   // SHAPE, not gauge. SubtractMean is mass-weighted and MPI-collective -- a
   // nodal average would be wrong at high order.
   FunctionCoefficient p_exact(
      [](const Vector & x, double t) { return incns::tgv2d::Pressure(x, t, kNu); });
   p_exact.SetTime(stepper.Time());

   ParGridFunction p_shifted(stepper.Pressure());
   incns::SubtractMean(p_shifted, rules);

   r.p_err = p_shifted.ComputeL2Error(p_exact, irs);
   ConstantCoefficient zero_p(0.0);
   r.p_norm = p_shifted.ComputeL2Error(zero_p, irs);
   return r;
}

double Rate(double e_coarse, double e_fine, double refine)
{
   return std::log(e_coarse / e_fine) / std::log(refine);
}

// Print the numbers behind every assertion, on every run -- same discipline as
// nse_mms_test: a rate/drift test is easy to misread as green when it is
// actually vacuous.
void Report(const char* label, const std::vector<double>& v)
{
   if (!Mpi::Root()) { return; }
   std::cout << "[ TGV NSE  ] " << label << ":";
   for (double x : v) { std::cout << " " << x; }
   std::cout << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// The core oracle: UNFORCED NSE reproduces the TGV velocity, and produces the
// analytic NONZERO pressure. A convection operator that is silently zero fails
// the pressure half; one that is mis-scaled fails both.
// ---------------------------------------------------------------------------
TEST(TgvNse, UnforcedReproducesAnalyticVelocityAndPressure)
{
   const TgvResult r = MarchTgv(/*n=*/8, /*ku=*/3, /*dt=*/0.005, true);
   Report("unforced (u_err, p_err, p_norm)", {r.u_err, r.p_err, r.p_norm});

   // Velocity: spatial + temporal error at this resolution. Q3 on 8^2 elements
   // resolves a single-mode TGV very well, so this is a tight bound.
   EXPECT_LT(r.u_err, 1e-4) << "unforced NSE did not reproduce the TGV velocity";

   // Pressure must be genuinely O(1) -- this is the half that a dead convection
   // operator cannot fake. The Stokes TGV pressure sits at the spatial floor
   // (~2e-4 here, see ConvectionOffGivesZeroPressure), so this threshold is ~500x
   // above what a solver ignoring convection can produce.
   //
   //   ||p_exact||_L2 = ||(cos2x + cos2y)/4||_L2 * e^{-4 nu t}
   //                  = (pi/2) * e^{-4 nu t} = 1.4213 at t = 0.5, nu = 0.05
   //   (checked numerically, and the analytic p has zero mean as assumed above).
   EXPECT_GT(r.p_norm, 0.5)
         << "pressure is far below the analytic 1.421: convection is not "
         << "contributing (the Stokes TGV has p -> 0; the NSE TGV must not)";

   // And it must match the analytic pressure in shape.
   EXPECT_LT(r.p_err / r.p_norm, 5e-3)
         << "pressure does not match (cos2x + cos2y)/4 * e^{-4 nu t}; "
         << "p_err=" << r.p_err << " ||p||=" << r.p_norm;
}

// ---------------------------------------------------------------------------
// Convection OFF on this SAME case must land somewhere clearly different: the
// Stokes TGV has p == 0. This is the control that proves the test above is
// actually sensitive to the convection switch.
// ---------------------------------------------------------------------------
TEST(TgvNse, ConvectionOffGivesZeroPressure)
{
   const TgvResult on  = MarchTgv(8, 3, 0.005, true);
   const TgvResult off = MarchTgv(8, 3, 0.005, false);
   Report("p_norm (on, off)", {on.p_norm, off.p_norm});

   // Both reproduce the same VELOCITY (that is the TGV's special property), so
   // velocity alone cannot distinguish the two paths ...
   EXPECT_LT(off.u_err, 1e-4);
   // ... but the pressure separates them by orders of magnitude.
   //
   // NOTE the Stokes-TGV pressure is NOT ~0 discretely. The continuum value is
   // zero, but the projected velocity is not exactly divergence-free in the FE
   // sense, so a small pressure appears at the SPATIAL discretization scale and
   // vanishes under h-refinement, not dt-refinement. tgv_stokes_temporal_test
   // measures it at ~2e-4 for n=8 Q3/Q2 (dt-independent, constant to 1% across
   // a dt sweep) and asserts a 1e-3 ceiling. Use that same ceiling here -- a
   // tighter bound would fail on a CORRECT solver.
   EXPECT_LT(off.p_norm, 1e-3)
         << "Stokes TGV pressure should sit at the spatial-discretization "
         << "floor (~2e-4 at this resolution), got " << off.p_norm;
   // The NSE pressure is O(1), so the separation is ~3 orders of magnitude even
   // against that floor -- the margin the control actually rests on.
   EXPECT_GT(on.p_norm / std::max(off.p_norm, 1e-300), 50.0)
         << "convection on/off produced indistinguishable pressure -- the "
         << "convection switch is not reaching the solve; on=" << on.p_norm
         << " off=" << off.p_norm;
}

// ---------------------------------------------------------------------------
// Temporal order on the unforced problem. Unlike the MMS, the exact solution
// here is NOT polynomial in t (it decays exponentially), so BOTH the implicit
// BDF side and the explicit EXT side commit truncation error; the composite
// scheme is 2nd order.
// ---------------------------------------------------------------------------
TEST(TgvNse, TemporalOrder)
{
   const int n = 6, ku = 3;
   const double e1 = MarchTgv(n, ku, 0.02,  true).u_err;
   const double e2 = MarchTgv(n, ku, 0.01,  true).u_err;
   const double e3 = MarchTgv(n, ku, 0.005, true).u_err;
   const double r1 = Rate(e1, e2, 2.0), r2 = Rate(e2, e3, 2.0);
   Report("temporal (e1, e2, e3, r1, r2)", {e1, e2, e3, r1, r2});

   // Vacuous-pass guard: if the coarse error were already at the spatial-error
   // floor, the rates below would be measuring the floor, not the scheme.
   ASSERT_GT(e1, 1e-9)
         << "coarse-dt error is at the spatial floor (" << e1 << "); the rate "
         << "assertions would be vacuous -- coarsen the mesh or raise dt";
   EXPECT_GT(r1, 1.7) << "e(0.02)=" << e1 << " e(0.01)=" << e2;
   EXPECT_GT(r2, 1.7) << "e(0.01)=" << e2 << " e(0.005)=" << e3;
}

// ---------------------------------------------------------------------------
// THE 2.2b HARNESS: discrete energy decay against the analytic history.
//
// This is the measurement the convective-form comparison will read. Today it
// pins the convective form's behaviour; when the skew-symmetric and rotational
// forms land, re-running this with each form is the experiment -- the forms
// differ in discrete energy conservation, and this is where that shows up.
// ---------------------------------------------------------------------------
TEST(TgvNse, EnergyDecayMatchesAnalytic)
{
   const TgvResult r = MarchTgv(/*n=*/8, /*ku=*/3, /*dt=*/0.005, true,
                                /*record_history=*/true);
   ASSERT_GE(r.times.size(), 2u);

   double worst_ke = 0.0, worst_eps = 0.0, worst_div = 0.0;
   for (size_t i = 0; i < r.times.size(); ++i)
   {
      const double t = r.times[i];
      const double ke_ana = incns::tgv2d::KineticEnergy(t, kNu);
      const double eps_ana = incns::tgv2d::DissipationRate(t, kNu);
      worst_ke = std::max(worst_ke, std::abs(r.energy[i] - ke_ana) / ke_ana);
      worst_eps = std::max(worst_eps,
                           std::abs(r.dissipation[i] - eps_ana) / eps_ana);
      worst_div = std::max(worst_div, r.divergence[i]);
   }
   Report("energy (KE0, KE_end, worst_ke, worst_eps, worst_div)",
   {r.energy.front(), r.energy.back(), worst_ke, worst_eps, worst_div});

   // Sanity: the flow must actually have decayed, otherwise "tracks the
   // analytic history" is a statement about a constant.
   EXPECT_LT(r.energy.back(), 0.95 * r.energy.front())
         << "kinetic energy did not decay -- nothing is being measured";

   // Track the analytic decay. The bound is loose enough to absorb the O(dt^2)
   // splitting error but far tighter than the ~10% drift a mis-scaled or
   // energy-inconsistent convective term would produce.
   EXPECT_LT(worst_ke, 5e-3)
         << "kinetic energy drifted from pi^2 e^{-4 nu t} by " << worst_ke;
   EXPECT_LT(worst_eps, 5e-3)
         << "dissipation drifted from 4 nu pi^2 e^{-4 nu t} by " << worst_eps;

   // Discrete divergence-free-ness: convection must not pollute the constraint.
   EXPECT_LT(worst_div, 1e-8) << "||div u|| grew to " << worst_div;
}
