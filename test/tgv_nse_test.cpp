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
// HOW IT ASSERTS: every accuracy claim in this file is a CONVERGENCE RATE under
// refinement, never a raw error magnitude -- see the threshold block below for
// why, and for the orders each study expects.
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
// nu = 1 (was 0.05). At nu=0.05 with a small dt the implicit block
// (1/dt)M + nu*K is mass-dominated and the Schur complement conditions poorly,
// so FGMRES iteration counts -- the real per-step cost -- are far higher. nu=1
// strengthens the viscous block and cuts them hard, which is what makes a
// multi-mesh rate study affordable here at all.
//
// The TGV decays as e^(-2 nu t) (velocity) and e^(-4 nu t) (pressure), so the
// SHORTER march below keeps the flow from decaying into noise: at t_final=0.1
// the pressure is still 67% of its initial magnitude, so the convection oracle
// (a NONZERO O(1) pressure that a dead convective term cannot fake) still bites.
constexpr double kNu = 1.0;
constexpr double kTFinal = 0.1;

// ---------------------------------------------------------------------------
// WHAT THIS FILE ASSERTS: RATES, NOT ERROR MAGNITUDES.
//
// A raw error magnitude is a statement about the RESOLUTION, not about the
// scheme -- pick a finer mesh and it moves, so no threshold on it can confirm
// correctness. The observed convergence RATE under refinement is the thing that
// pins the discretization: it must match the theoretical order, and a
// mis-scaled, mis-signed, or aliased convective term breaks the rate even when
// the raw error looks small.
//
// The first version of this test asserted magnitudes (u_err < 1e-4,
// worst_div < 1e-8, ...) and all four subtests failed on a solver that is in
// fact correct (job 42595862, 2026-07-25) -- the numbers were estimates, and
// they were measuring the mesh. Every accuracy assertion here is now a rate.
//
// Theoretical orders for Taylor-Hood Q(ku)/Q(ku-1), here Q3/Q2 (ku = 3):
//
//   SPACE -- Taylor-Hood velocity is one order BETTER than pressure:
//     velocity L2   O(h^(ku+1))       = h^4   -- full interpolation order of Q3
//     pressure L2   O(h^ku)           = h^3   -- i.e. h^(kp+1) with kp = ku-1;
//                                               the pressure space is Q2, one
//                                               degree below the velocity. NOTE
//                                               this is h^(kp+1), NOT h^ku by
//                                               coincidence -- if ku changes,
//                                               re-derive from kp = ku-1.
//     div u    L2   O(h^ku)           = h^3   -- div maps Q3 velocity into the
//                                               Q2 pressure test space, so the
//                                               weak constraint is controlled at
//                                               the pressure order, one below
//                                               the velocity. Asserted loosely
//                                               (>2) since this is an analogy
//                                               argument, not a cited estimate.
//   TIME  (BDF2 implicit viscous/pressure + EXT2 explicit convection)
//     velocity      O(dt^2)
//     pressure      O(dt^2) expected here, asserted >1.5. In a SPLIT scheme the
//                   pressure typically drops one order; this solver is fully
//                   coupled (the block solve does velocity and pressure
//                   simultaneously) so it should retain 2, but the pressure is
//                   where a splitting-error regression would surface first --
//                   which is exactly why it is asserted separately from u.
//
// Rates are asserted with a slack of ~0.4-0.5 below theory: coarse-mesh
// pre-asymptotic behaviour and GPU non-determinism (~1.5e-11 relative, far
// below any scale here) both perturb the measured value slightly.
//
// PRE-ASYMPTOTIC BEHAVIOUR IS REAL HERE AND IS NOT A DEFECT. With convection ON
// the velocity needs n ~ 48 to reach its h^4 limit (measured 3.30 -> 3.93 over
// n = 8..48); with convection OFF it is already at 3.98 by n=8. The bounds
// below are therefore set from what is measurable at THIS test's resolution,
// not from the theoretical limit -- see the oracle test for the full ladder.
//
// AND: any spatial study must stay clear of the O(dt^2) TEMPORAL floor. Once
// the spatial error approaches dt^2 the ladder measures dt and the rate
// collapses; a ku=5 sweep hit exactly that (dt^2 = 1.56e-06 vs u_err 1.53e-06,
// rate 1.65). The oracle test asserts this explicitly.
// ---------------------------------------------------------------------------

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
   double u_norm =
      0.0;     // ||u_exact||_L2 at t_final -- the scale u_err lives on
   double p_err = 0.0;      // ||p - p_exact||_L2 at t_final (zero-mean matched)
   double p_norm = 0.0;     // ||p||_L2 at t_final
   Vector u_true;           // velocity true dofs at t_final (same-mesh comparisons)
   Vector p_true;           // pressure true dofs, zero-mean (same-mesh, temporal)
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

   // The scale u_err is measured against. ||u_exact|| = sqrt(2 KE(t)) on the
   // full box; computed here by quadrature rather than the closed form so the
   // relative error is not hostage to a second analytic expression.
   {
      ParGridFunction u_ex(&spaces.Velocity());
      u_ex.ProjectCoefficient(u0);
      VectorConstantCoefficient zero_u(zero_vec);
      r.u_norm = u_ex.ComputeL2Error(zero_u, irs);
   }

   // True dofs for same-mesh comparisons (the temporal study), where the
   // spatial error must cancel instead of dominating.
   r.u_true.SetSize(spaces.Velocity().GetTrueVSize());
   stepper.Velocity().GetTrueDofs(r.u_true);

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

   // Pressure true dofs from the ZERO-MEAN field: on the fully periodic box the
   // constant nullspace mode is arbitrary and two runs at different dt can land
   // on different constants, so differencing the raw pressure would measure the
   // gauge, not the temporal error.
   r.p_true.SetSize(spaces.Pressure().GetTrueVSize());
   p_shifted.GetTrueDofs(r.p_true);
   return r;
}

double Rate(double e_coarse, double e_fine, double refine)
{
   return std::log(e_coarse / e_fine) / std::log(refine);
}

// Discrete l2 norm of a true-dof difference. On a FIXED mesh this is a fixed
// norm, so temporal rates computed from it are unaffected by the choice.
double DiffNorm(const Vector& a, const Vector& b, MPI_Comm comm)
{
   Vector d(a);
   d -= b;
   return std::sqrt(InnerProduct(comm, d, d));
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
   const int ku = 3;
   // dt small enough that the TEMPORAL error stays below the spatial error at
   // the FINEST mesh -- otherwise the spatial rate saturates on the dt floor
   // and reports a falsely low order. Checked by the saturation guard below.
   //
   // The ladder starts at n=6, NOT n=4. n=4 is pre-asymptotic on this problem:
   // its velocity error sits ~47x BELOW an h^4 curve fitted through the finer
   // meshes, so including it produced a spurious NEGATIVE rate (-0.98) and sent
   // an earlier version of this investigation chasing a defect that was not
   // there. A periodic direction also needs >= 3 elements (periodic_box.cpp),
   // so n=4 is only one element clear of the mesh's own floor.
   //
   // Rates are read from the 8 -> 12 step (1.5x refinement); Rate() takes the
   // ratio as an argument precisely so a non-uniform ladder stays honest.
   const double dt = 0.0025;
   const int ns[3] = {6, 8, 12};
   const double kRefine[2] = {8.0 / 6.0, 1.5};

   double ue[3], pe[3];
   double u_norm = 0.0, p_norm_fine = 0.0;
   for (int i = 0; i < 3; ++i)
   {
      const TgvResult r = MarchTgv(ns[i], ku, dt, true);
      ue[i] = r.u_err;
      pe[i] = r.p_err;
      u_norm = r.u_norm;
      p_norm_fine = r.p_norm;
      Report("unforced n", {double(ns[i]), r.u_err, r.p_err, r.p_norm});
   }

   const double ru1 = Rate(ue[0], ue[1], kRefine[0]);
   const double ru2 = Rate(ue[1], ue[2], kRefine[1]);
   const double rp1 = Rate(pe[0], pe[1], kRefine[0]);
   const double rp2 = Rate(pe[1], pe[2], kRefine[1]);
   Report("spatial rates (u: r1 r2 | p: r1 r2)", {ru1, ru2, rp1, rp2});

   // Saturation guard: if the finest-mesh error had hit the temporal-error
   // floor, the second rate would be measuring dt, not h. At 1.5x refinement
   // an O(h^4) velocity error should drop by ~5x; require at least 2x.
   ASSERT_LT(ue[2], 0.5 * ue[1])
         << "velocity error stopped converging (e(n=8)=" << ue[1]
         << " e(n=12)=" << ue[2] << ") -- likely saturated on the dt floor; "
         << "reduce dt before reading these rates";

   // Explicit dt-floor guard. The scheme is O(dt^2), so once the SPATIAL error
   // approaches dt^2 the ladder measures dt and the rate collapses -- this is
   // not hypothetical, it is exactly what a ku=5 sweep did (dt^2 = 1.56e-06,
   // u_err = 1.53e-06, rate fell to 1.65). Today e(n=12) = 3.0e-04 against a
   // floor of 6.3e-06, ~48x clear. Fires if someone coarsens dt or refines the
   // mesh without re-checking the pair.
   {
      const double dt_floor = dt * dt;
      ASSERT_GT(ue[2], 10.0 * dt_floor)
            << "finest-mesh velocity error " << ue[2] << " is within 10x of the "
            << "O(dt^2) temporal floor " << dt_floor << " -- these spatial rates "
            << "are measuring dt, not h. Reduce dt (or coarsen the mesh).";
   }

   // Taylor-Hood Q3/Q2 velocity converges at O(h^4), ONE ORDER BETTER than the
   // O(h^3) pressure. On THIS problem the convection-on velocity approaches
   // that limit slowly -- it is still pre-asymptotic at the resolutions this
   // test can afford -- so the assertion is set below 4 deliberately.
   //
   // RESOLVED 2026-07-27 (scripts/probe_spatial.sh, nu=1, t_final=0.1, ku=3,
   // dt=0.0003125, n = 8/12/16/24/32/48). Both rates converge to theory:
   //     convection ON  velocity  3.296 3.557 3.748 3.867 3.926   -> 4
   //     convection ON  pressure  2.642 2.783 2.877 2.935 2.966   -> 3
   //     convection OFF velocity  3.981 3.989 3.994 3.995 3.957   (flat h^4)
   // and the velocity error pulls steadily AHEAD of the pressure error, as
   // Taylor-Hood requires: u_err/p_err = 0.066, 0.051, 0.040, 0.028, 0.022,
   // 0.0147 -- the velocity ends ~68x more accurate than the pressure. That is
   // the h^4-vs-h^3 separation showing up directly.
   //
   // So the convection-ON path is simply PRE-ASYMPTOTIC at the resolutions this
   // test can afford: it carries a larger subdominant term than the Stokes path
   // and needs n ~ 48 to wash it out. There is no defect and no coupling
   // pathology -- do not re-open this.
   //
   // A misreading worth recording so it is not re-derived: a degree sweep
   // appears to show rates ABOVE theory (ku=4: 6.04, 5.66 vs h^5) and then a
   // collapse (ku=5: 4.32, 1.65 vs h^6). That is the O(dt^2) TEMPORAL floor,
   // not a spatial effect -- dt^2 = 1.56e-06 at dt=0.00125 and the ku=5 n=12
   // velocity error is 1.53e-06, sitting exactly on it. ANY spatial study whose
   // error approaches dt^2 is measuring dt, so dt must fall as the mesh refines.
   //
   // Asserted >2.9 against the 3.296 measured at n=8->12 (the pair this reads),
   // ~0.4 of margin. A dead or mis-scaled convective term does not land 0.4
   // low -- it breaks the rate outright.
   EXPECT_GT(ru2, 2.9) << "velocity spatial rate " << ru2 << ", expected ~3.3 "
                       << "at this resolution, en route to 4 (e: " << ue[0]
                       << " " << ue[1] << " " << ue[2] << ")";
   // Pressure at O(h^3), approached from below (2.43 -> 2.63 -> 2.80 measured).
   // This is the half a dead convection operator cannot fake: with convection
   // off the pressure is ~0 and has no rate at all.
   EXPECT_GT(rp2, 2.3) << "pressure spatial rate " << rp2 << ", expected ~3 "
                       << "(e: " << pe[0] << " " << pe[1] << " " << pe[2] << ")";

   // Convergence is toward the RIGHT function: the pressure must be genuinely
   // O(1), not converging to zero. ||p_exact|| = (pi/2) e^{-4 nu t} = 1.0529 at
   // nu=1, t=0.1 (measured 1.05292-1.05294 across the whole ladder). A solver
   // ignoring convection produces ~3e-4 here, so this separates the two by
   // ~3.5 orders of magnitude. (Magnitude, deliberately: this is a QUALITATIVE
   // check that convection contributes at all, not an accuracy claim.)
   EXPECT_GT(p_norm_fine, 0.5)
         << "pressure is far below the analytic 1.053: convection is not "
         << "contributing (the Stokes TGV has p -> 0; the NSE TGV must not)";
   EXPECT_GT(u_norm, 1.0) << "sanity: ||u_exact|| should be ~3.64 at nu=1, "
                          << "t_final=0.1, got " << u_norm;
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
   // velocity alone cannot distinguish the two paths. Asserted as a RATE -- the
   // Stokes path must converge at the same O(h^4) as the NSE path, which is a
   // statement about the scheme; the raw error would only describe the mesh.
   // Uncoupled from the pressure, so this one really is the full O(h^4) of the
   // Q3 velocity space -- and it is FLAT, not climbing: measured 3.965, 3.982,
   // 3.990 at n=6/8/12/16 (scripts/probe_spatial.sh). That contrast with the
   // convection-ON rate (~3, pressure-limited) is the point of this test.
   {
      const double e_c = MarchTgv(6, 3, 0.005, false).u_err;   // n=8 is `off`
      const double r = Rate(e_c, off.u_err, 8.0 / 6.0);
      Report("convection-off velocity (e_n6, e_n8, rate)", {e_c, off.u_err, r});
      EXPECT_GT(r, 3.5) << "convection-off (Stokes) velocity spatial rate " << r
                        << ", expected ~4; e(n=6)=" << e_c
                        << " e(n=8)=" << off.u_err;
   }
   // ... but the pressure separates them by orders of magnitude.
   //
   // NOTE the Stokes-TGV pressure is NOT ~0 discretely. The continuum value is
   // zero, but the projected velocity is not exactly divergence-free in the FE
   // sense, so a small pressure appears at the SPATIAL discretization scale and
   // vanishes under h-refinement, not dt-refinement. tgv_stokes_temporal_test
   // measures it at ~2e-4 for n=8 Q3/Q2 (dt-independent, constant to 1% across
   // a dt sweep) and asserts a 1e-3 ceiling. Use that same ceiling here -- a
   // tighter bound would fail on a CORRECT solver. At nu=1 the measured value
   // is 3.33e-04 at n=8, so the ceiling keeps ~3x margin.
   EXPECT_LT(off.p_norm, 1e-3)
         << "Stokes TGV pressure should sit at the spatial-discretization "
         << "floor (~3.3e-4 at this resolution), got " << off.p_norm;
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
// Errors are measured against a SAME-MESH fine-dt reference, NOT against the
// analytic solution. This is the technique tgv_stokes_temporal_test already
// uses, and it is load-bearing here: at any resolution this test can afford,
// the SPATIAL error dwarfs the temporal one, so differencing against u_exact
// measures the mesh and reports rate ~= 0 no matter how correct the scheme is.
// Differencing two runs on the SAME mesh cancels the spatial error exactly and
// leaves the temporal error alone.
//
// (This is precisely how the original version of this test failed: it reported
// e = 0.04757, 0.04754, 0.04753 over a 4x dt refinement -- flat, i.e. pure
// spatial floor -- for rates of 0.0009. The scheme was never the problem.)
TEST(TgvNse, TemporalOrder)
{
   const int n = 6, ku = 3;
   const TgvResult ref = MarchTgv(n, ku, 0.0003125, true);

   // Step sizes scaled down with t_final (0.5 -> 0.1). The old 0.02/0.01/0.005
   // would be only 5/10/20 steps on the short march, making the trapezoidal
   // STARTUP step ~20% of the coarse march and polluting the order it measures.
   // These give 25/50/100 steps, matching what the long march used to provide,
   // and the reference is 8x finer than the finest of them.
   const double dts[3] = {0.004, 0.002, 0.001};
   double e[3], ep[3];
   for (int i = 0; i < 3; ++i)
   {
      const TgvResult run = MarchTgv(n, ku, dts[i], true);
      e[i]  = DiffNorm(run.u_true, ref.u_true, MPI_COMM_WORLD);
      ep[i] = DiffNorm(run.p_true, ref.p_true, MPI_COMM_WORLD);
   }
   const double r1 = Rate(e[0], e[1], 2.0), r2 = Rate(e[1], e[2], 2.0);
   const double rp1 = Rate(ep[0], ep[1], 2.0), rp2 = Rate(ep[1], ep[2], 2.0);
   Report("temporal u (e1, e2, e3, r1, r2)", {e[0], e[1], e[2], r1, r2});
   Report("temporal p (e1, e2, e3, r1, r2)", {ep[0], ep[1], ep[2], rp1, rp2});

   // Vacuous-pass guard, now against the REFERENCE-differencing floor: the
   // finest dt must stay clearly above the solver tolerance (rtol=1e-12) and
   // clearly below the coarse error, or the rates measure noise.
   ASSERT_GT(e[2], 1e-10)
         << "finest-dt difference is at the solver-tolerance floor (" << e[2]
         << "); the rate assertions would be vacuous -- raise dt or tighten rtol";
   ASSERT_LT(e[2], 0.5 * e[0])
         << "errors are not decreasing under dt refinement (e1=" << e[0]
         << " e3=" << e[2] << ") -- the reference is too close to the coarse runs";

   // VELOCITY in time: BDF2 implicit + EXT2 explicit convection -> O(dt^2).
   EXPECT_GT(r1, 1.7) << "e(dt1)=" << e[0] << " e(dt2)=" << e[1];
   EXPECT_GT(r2, 1.7) << "e(dt2)=" << e[1] << " e(dt3)=" << e[2];

   // PRESSURE in time. Asserted SEPARATELY because a velocity-only temporal
   // study cannot see a pressure-in-time defect: in a splitting/IMEX scheme the
   // pressure characteristically converges one order BELOW the velocity, so a
   // pressure that has silently dropped to O(dt) still leaves the velocity rate
   // at a clean 2. Expect ~2 for this fully-coupled block solve (the pressure
   // is solved simultaneously, not projected in a separate step), but assert
   // only >1.5 -- the pressure is the more delicate of the two and coarse-dt
   // pre-asymptotic behaviour shows up here first.
   ASSERT_GT(ep[2], 1e-10)
         << "finest-dt pressure difference is at the solver floor (" << ep[2]
         << "); the pressure rate assertions would be vacuous";
   EXPECT_GT(rp1, 1.5) << "pressure temporal rate " << rp1
                       << "; e(dt1)=" << ep[0] << " e(dt2)=" << ep[1];
   EXPECT_GT(rp2, 1.5) << "pressure temporal rate " << rp2
                       << "; e(dt2)=" << ep[1] << " e(dt3)=" << ep[2];
}

// ---------------------------------------------------------------------------
// THE 2.2b HARNESS: discrete energy decay against the analytic history.
//
// This is the measurement the convective-form comparison will read. Today it
// pins the convective form's behaviour; when the skew-symmetric and rotational
// forms land, re-running this with each form is the experiment -- the forms
// differ in discrete energy conservation, and this is where that shows up.
// ---------------------------------------------------------------------------
// Drift from the analytic energy history is itself a discretization error, so
// it too is asserted as a RATE under h-refinement, not as a magnitude. This is
// what makes the test a usable 2.2b instrument: when the skew-symmetric and
// rotational forms land, a form that is energy-inconsistent shows up as a
// BROKEN RATE here, which no single-resolution drift bound could distinguish
// from "the mesh is a bit coarse".
TEST(TgvNse, EnergyDecayMatchesAnalytic)
{
   const int ku = 3;
   // dt is 16x SMALLER than the oracle study's, and that is load-bearing. The
   // energy drifts are far smaller quantities than the velocity L2 error, so
   // they reach the O(dt^2) temporal floor at a much coarser mesh. Measured at
   // the oracle's dt=0.0025 (dt^2 = 6.25e-06):
   //     n     ke_drift    eps_drift   eps/dt^2
   //     6     4.89e-05    2.12e-05     3.4x
   //     8     1.27e-05    4.82e-06     0.77x   <- already BELOW the floor
   //    12     4.25e-06    3.23e-06     0.52x   <- pinned on it
   // giving rates ke 2.71 and eps 0.99 -- both measuring dt, not h. On the
   // 6->8 step, where the errors are still above the floor, they are 4.67 and
   // 4.30, i.e. at or above theory. The discretization was never the problem.
   //
   // dt = 0.00015625 puts the floor at 2.4e-08, ~130x below the smallest drift
   // on the ladder. The dt-floor guard below fails loudly if that stops holding.
   const double dt = 0.00015625;
   const int ns[3] = {6, 8, 12};   // n=4 is pre-asymptotic here too -- see above
   const double kRefine = 1.5;     // the 8 -> 12 step the rates are read from

   double ke_drift[3], eps_drift[3], div_norm[3];
   double ke_first = 0.0, ke_last = 0.0;

   for (int i = 0; i < 3; ++i)
   {
      const TgvResult r = MarchTgv(ns[i], ku, dt, true, /*record_history=*/true);
      ASSERT_GE(r.times.size(), 2u);

      double worst_ke = 0.0, worst_eps = 0.0, worst_div = 0.0;
      for (size_t j = 0; j < r.times.size(); ++j)
      {
         const double t = r.times[j];
         const double ke_ana = incns::tgv2d::KineticEnergy(t, kNu);
         const double eps_ana = incns::tgv2d::DissipationRate(t, kNu);
         worst_ke = std::max(worst_ke, std::abs(r.energy[j] - ke_ana) / ke_ana);
         worst_eps = std::max(worst_eps,
                              std::abs(r.dissipation[j] - eps_ana) / eps_ana);
         worst_div = std::max(worst_div, r.divergence[j]);
      }
      ke_drift[i] = worst_ke;
      eps_drift[i] = worst_eps;
      div_norm[i] = worst_div;
      if (i == 1) { ke_first = r.energy.front(); ke_last = r.energy.back(); }
      Report("energy n", {double(ns[i]), worst_ke, worst_eps, worst_div});
   }

   const double r_ke  = Rate(ke_drift[1], ke_drift[2], kRefine);
   const double r_eps = Rate(eps_drift[1], eps_drift[2], kRefine);
   const double r_div = Rate(div_norm[1], div_norm[2], kRefine);
   Report("energy rates (ke, eps, div)", {r_ke, r_eps, r_div});

   // Sanity: the flow must actually have decayed, otherwise "tracks the
   // analytic history" is a statement about a constant.
   EXPECT_LT(ke_last, 0.95 * ke_first)
         << "kinetic energy did not decay -- nothing is being measured";

   // KE is quadratic in u, so its error inherits the velocity rate: O(h^(ku+1))
   // in u gives O(h^(ku+1)) in KE to leading order. Dissipation involves grad u
   // and loses one order -> O(h^ku). Slack is generous because these are
   // worst-over-history maxima, which are noisier than an endpoint error.
   //
   // Same dt-floor guard as the oracle study: these drifts are SMALL absolute
   // quantities, so they hit the O(dt^2) floor at a much coarser mesh than the
   // velocity L2 error does -- which is exactly what happened at the oracle's
   // dt (see the table above). Fires before the rate assertions can report a
   // floor as a discretization failure.
   {
      const double dt_floor = dt * dt;
      ASSERT_GT(eps_drift[2], 10.0 * dt_floor)
            << "finest-mesh dissipation drift " << eps_drift[2] << " is within "
            << "10x of the O(dt^2) temporal floor " << dt_floor << " -- these "
            << "rates are measuring dt, not h. Reduce dt.";
      ASSERT_GT(ke_drift[2], 10.0 * dt_floor)
            << "finest-mesh energy drift " << ke_drift[2] << " is within 10x of "
            << "the O(dt^2) temporal floor " << dt_floor << " -- these rates are "
            << "measuring dt, not h. Reduce dt.";
   }
   // MEASURED at the dt above (np1, 2026-07-27): ke 5.52, eps 4.12, div 2.36,
   // with the finest mesh still 41x / 29x clear of the dt^2 floor. ke and eps
   // sit ABOVE their theoretical 4 and 3 -- the same pre-asymptotic overshoot
   // the velocity ladder shows, where a subdominant term is still decaying.
   //
   // Bounds are deliberately NOT tightened to those values: an overshoot is by
   // definition transient, so a bound fitted to it would fail the moment the
   // rate settles onto theory. >3.0 and >2.0 sit just below the THEORETICAL
   // orders, which is the durable statement, and still leave 2.5x / 2.1x margin
   // against what is measured today.
   EXPECT_GT(r_ke, 3.0)
         << "kinetic-energy drift is not converging (rate " << r_ke
         << ", expected ~4): drift " << ke_drift[1] << " -> " << ke_drift[2];
   EXPECT_GT(r_eps, 2.0)
         << "dissipation drift is not converging (rate " << r_eps
         << ", expected ~3): drift " << eps_drift[1] << " -> " << eps_drift[2];

   // Discrete divergence-free-ness. The Taylor-Hood velocity is only WEAKLY
   // divergence-free -- div u is tested against the pressure space, not driven
   // to zero pointwise -- so ||div u|| is an O(h^ku) discretization quantity,
   // NOT machine zero. (The original 1e-8 bound was unreachable by any correct
   // solver; the passing diagnostics_test.cpp:157 uses < 1e-2 for this reason.)
   // Asserting the RATE is what actually confirms the constraint: a convective
   // term polluting it would stop div u from converging at all.
   EXPECT_GT(r_div, 2.0)
         << "||div u|| is not converging under refinement (rate " << r_div
         << ", expected ~3): " << div_norm[1] << " -> " << div_norm[2]
         << " -- convection may be polluting the divergence constraint";
}
