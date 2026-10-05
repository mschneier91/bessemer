// Sprint 2.2: Navier-Stokes MMS -- 2D AND 3D.
//
// The manufactured velocity/pressure fields are the SAME ones the Stokes
// unsteady MMS uses (test/unsteady_mms_test.cpp), with the convective term
// added to the forcing. So the exact solution is unchanged and any difference
// from the Stokes result is attributable to the convection wiring alone.
//
// WHY THIS IS AN ORDER TEST, NOT AN EXACTNESS TEST
// -----------------------------------------------
// The Stokes MMS reproduces its solution to solver tolerance at any dt, because
// the implicit BDF side integrates the quadratic-in-time factor
// G(t) = 1 + t + t^2/2 exactly. That does NOT carry over to NSE: the convective
// term is quadratic IN G, hence degree 4 in t, and it is treated EXPLICITLY by
// AB/EXT extrapolation. EXT2 is exact only for degree <= 1, so it commits a
// genuine O(dt^2) splitting error. Asserting machine-precision exactness here
// would be asserting something false -- the honest check is that the error
// converges at the design rate.
//
//   (u.grad)u, verified numerically before use (not hand-algebra):
//     2D  u = G(t) * ( 3x^3y^2, -3x^2y^3 )   ->  9*G^2 * ( x^5y^4,  x^4y^5 )
//     3D  u = G(t) * ( y^2, z^2, x^2 )       ->  2*G^2 * ( y z^2, z x^2, x y^2 )
//   Both fields are divergence-free (also checked numerically).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "post/pressure_mean.hpp"
#include "spaces/mixed_spaces.hpp"
#include "quadrature/rule_book.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::ConvectiveForm;
using incns::RotationVelocityPC;
using incns::SchurBlockType;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }

BoxSpec UnitBox(int dim, int n)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// How a run treats the nonlinear term and preconditions the velocity block.
struct MmsConfig
{
   ConvectiveForm form = ConvectiveForm::Convective;
   RotationVelocityPC rotation_pc = RotationVelocityPC::Symmetric;
   SchurBlockType schur = SchurBlockType::Mass;
   /// Mass path's velocity PC (the CC path's a_pc default is LOR-AMG).
   incns::VelocityPreconditioner velocity_prec =
      incns::VelocityPreconditioner::Jacobi;
   bool rotation_in_lor = false; ///< Rotation term in the LOR-AMG operator.
};

struct MmsResult
{
   double u_err = 0.0; // final velocity L2 error
   double p_err = 0.0; // final static-pressure L2 error (zero-mean matched);
   // only when an exact pressure is given
   Vector u_true;      // final velocity true dofs (config comparisons)
};

// March the NSE MMS to t_final at step dt. All boundaries are Dirichlet, so
// the pressure is mean-normalized; @p p_exact (optional) is compared after
// the same normalization.
MmsResult NseMmsRun(int dim, int n, int ku, double nu, double dt,
                    double t_final, VectorFunctionCoefficient& u_exact,
                    VectorFunctionCoefficient& forcing,
                    const MmsConfig& cfg = MmsConfig(),
                    FunctionCoefficient* p_exact = nullptr)
{
   Mesh serial = MakeBoxMesh(UnitBox(dim, n));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;

   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 2 * dim; ++attr)
   {
      bc.AddVelocityDirichlet(attr, u_exact); // time-dependent Dirichlet data
   }

   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = dt;
   opts.t_final = t_final;
   opts.convection = true;   // <-- the thing under test
   opts.convective_form = cfg.form;
   opts.rotation_pc = cfg.rotation_pc;
   opts.schur = cfg.schur;
   opts.velocity_prec = cfg.velocity_prec;
   opts.rotation_in_lor = cfg.rotation_in_lor;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesTimeIntegrator stepper(spaces, rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);

   // Step to t_final. Guard the loop count so a stalled march fails loudly
   // rather than spinning.
   const int max_steps = static_cast<int>(std::ceil(t_final / dt)) + 2;
   int steps = 0;
   while (stepper.Time() < t_final - 1e-12 && steps < max_steps)
   {
      stepper.Step();
      ++steps;
   }

   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   irs[geom] = &rules.Get(geom, 2 * ku + 4);
   u_exact.SetTime(stepper.Time());
   MmsResult r;
   r.u_err = stepper.Velocity().ComputeL2Error(u_exact, irs);
   r.u_true.SetSize(spaces.Velocity().GetTrueVSize());
   stepper.Velocity().GetTrueDofs(r.u_true);
   if (p_exact)
   {
      p_exact->SetTime(stepper.Time());
      ParGridFunction pe(&spaces.Pressure());
      pe.ProjectCoefficient(*p_exact); // exact: p is in the pressure space
      incns::SubtractMean(pe, rules);
      ParGridFunction ph(stepper.Pressure());
      incns::SubtractMean(ph, rules);
      ph -= pe;
      ConstantCoefficient zero(0.0);
      r.p_err = ph.ComputeL2Error(zero, irs);
   }
   return r;
}

// The velocity error alone (the convective-form studies).
double NseMmsError(int dim, int n, int ku, double nu, double dt, double t_final,
                   VectorFunctionCoefficient& u_exact,
                   VectorFunctionCoefficient& forcing,
                   const MmsConfig& cfg = MmsConfig())
{
   return NseMmsRun(dim, n, ku, nu, dt, t_final, u_exact, forcing, cfg).u_err;
}

// --- 2D fields -------------------------------------------------------------
void Fields2D(double nu, std::unique_ptr<VectorFunctionCoefficient>& u_exact,
              std::unique_ptr<VectorFunctionCoefficient>& forcing)
{
   u_exact = std::make_unique<VectorFunctionCoefficient>(
                2, [](const Vector & x, double t, Vector & v)
   {
      v(0) =  G(t) * 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      v(1) = -G(t) * 3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
   });
   // f = u_t + (u.grad)u - nu lap(u) + grad(p), same u/p as the Stokes MMS
   // plus the convective term.
   forcing = std::make_unique<VectorFunctionCoefficient>(
                2, [nu](const Vector & x, double t, Vector & f)
   {
      const double g = G(t), gp = Gp(t);
      const double us0 =  3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      const double us1 = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
      const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
      const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
      const double c0 = 9.0 * g * g * std::pow(x[0], 5) * std::pow(x[1], 4);
      const double c1 = 9.0 * g * g * std::pow(x[0], 4) * std::pow(x[1], 5);
      f(0) = gp * us0 + g * (-nu * lap0 + 2.0 * x[0]) + c0;
      f(1) = gp * us1 + g * (-nu * lap1 + 2.0 * x[1]) + c1;
   });
}

// --- 3D fields -------------------------------------------------------------
void Fields3D(double nu, std::unique_ptr<VectorFunctionCoefficient>& u_exact,
              std::unique_ptr<VectorFunctionCoefficient>& forcing)
{
   u_exact = std::make_unique<VectorFunctionCoefficient>(
                3, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * x[1];
      v(1) = G(t) * x[2] * x[2];
      v(2) = G(t) * x[0] * x[0];
   });
   forcing = std::make_unique<VectorFunctionCoefficient>(
                3, [nu](const Vector & x, double t, Vector & f)
   {
      const double g = G(t), gp = Gp(t);
      // Stokes part (lap(u_s) = 2 in each component, grad(p_s) = (1,1,1)).
      f(0) = gp * x[1] * x[1] + g * (1.0 - 2.0 * nu);
      f(1) = gp * x[2] * x[2] + g * (1.0 - 2.0 * nu);
      f(2) = gp * x[0] * x[0] + g * (1.0 - 2.0 * nu);
      // + (u.grad)u = 2 G^2 (y z^2, z x^2, x y^2)
      f(0) += 2.0 * g * g * x[1] * x[2] * x[2];
      f(1) += 2.0 * g * g * x[2] * x[0] * x[0];
      f(2) += 2.0 * g * g * x[0] * x[1] * x[1];
   });
}

// Observed convergence rate between two step sizes.
double Rate(double e_coarse, double e_fine, double refine)
{
   return std::log(e_coarse / e_fine) / std::log(refine);
}

// Print the errors and rates on EVERY run, not just on failure. A rate test is
// easy to read as green when it is actually vacuous, so the numbers behind the
// assertion should be visible in the log without editing the test.
void ReportRates(const char* label, double e1, double e2, double e3)
{
   if (mfem::Mpi::Root())
   {
      std::cout << "[ NSE MMS  ] " << label
                << ": e(0.02)=" << e1
                << "  e(0.01)=" << e2
                << "  e(0.005)=" << e3
                << "  rate(1->2)=" << Rate(e1, e2, 2.0)
                << "  rate(2->3)=" << Rate(e2, e3, 2.0) << std::endl;
   }
}

// A splitting error this small would mean the convective term is not actually
// being exercised (or is being cancelled), in which case the RATE is measuring
// solver noise and proves nothing. The 2D/3D fields here produce an O(dt^2)
// error comfortably above this.
constexpr double kNoiseFloor = 1e-11;

} // namespace

// ---------------------------------------------------------------------------
// 2D: temporal order of the IMEX splitting.
// ---------------------------------------------------------------------------
TEST(NseMms, TemporalOrder2D)
{
   const double nu = 0.7, t_final = 0.2;
   std::unique_ptr<VectorFunctionCoefficient> u_exact, forcing;
   Fields2D(nu, u_exact, forcing);

   // Spatial error is negligible here (the fields are polynomials the space
   // represents exactly), so the measured error is pure temporal splitting.
   const double e1 = NseMmsError(2, 3, 3, nu, 0.02,  t_final, *u_exact, *forcing);
   const double e2 = NseMmsError(2, 3, 3, nu, 0.01,  t_final, *u_exact, *forcing);
   const double e3 = NseMmsError(2, 3, 3, nu, 0.005, t_final, *u_exact, *forcing);

   ReportRates("2D", e1, e2, e3);
   const double r1 = Rate(e1, e2, 2.0);
   const double r2 = Rate(e2, e3, 2.0);
   // Guard against a VACUOUS pass: if the coarsest error were already at solver
   // noise, the rates below would be measuring nothing.
   ASSERT_GT(e1, kNoiseFloor)
         << "coarse-dt error is at noise level (" << e1 << ") -- the rate "
         << "assertions would be vacuous; is convection actually on?";
   // BDF2 + EXT2 is formally 2nd order. Allow the usual slack for a short
   // march; the point is that it is clearly 2, not 1 (which is what a wrong
   // extrapolation order or a missed startup term would give).
   EXPECT_GT(r1, 1.7) << "e(0.02)=" << e1 << " e(0.01)=" << e2;
   EXPECT_GT(r2, 1.7) << "e(0.01)=" << e2 << " e(0.005)=" << e3;
   EXPECT_LT(e3, e1) << "refinement did not reduce the error";
}

// ---------------------------------------------------------------------------
// 3D: the case the user requires for sign-off.
// ---------------------------------------------------------------------------
TEST(NseMms, TemporalOrder3D)
{
   const double nu = 1.3, t_final = 0.2;
   std::unique_ptr<VectorFunctionCoefficient> u_exact, forcing;
   Fields3D(nu, u_exact, forcing);

   const double e1 = NseMmsError(3, 2, 2, nu, 0.02,  t_final, *u_exact, *forcing);
   const double e2 = NseMmsError(3, 2, 2, nu, 0.01,  t_final, *u_exact, *forcing);
   const double e3 = NseMmsError(3, 2, 2, nu, 0.005, t_final, *u_exact, *forcing);

   ReportRates("3D", e1, e2, e3);
   const double r1 = Rate(e1, e2, 2.0);
   const double r2 = Rate(e2, e3, 2.0);
   ASSERT_GT(e1, kNoiseFloor)
         << "coarse-dt error is at noise level (" << e1 << ") -- the rate "
         << "assertions would be vacuous; is convection actually on?";
   EXPECT_GT(r1, 1.7) << "e(0.02)=" << e1 << " e(0.01)=" << e2;
   EXPECT_GT(r2, 1.7) << "e(0.01)=" << e2 << " e(0.005)=" << e3;
   EXPECT_LT(e3, e1) << "refinement did not reduce the error";
}

// ---------------------------------------------------------------------------
// Convection OFF must reproduce the Stokes result bit-for-bit-ish: this pins
// that the NSE path is genuinely additive and cannot silently perturb Stokes.
// ---------------------------------------------------------------------------
TEST(NseMms, ConvectionOffMatchesStokes)
{
   const double nu = 1.3, t_final = 0.1, dt = 0.02;
   // Stokes fields (no convective term in the forcing).
   VectorFunctionCoefficient u_exact(
      3, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * x[1];
      v(1) = G(t) * x[2] * x[2];
      v(2) = G(t) * x[0] * x[0];
   });
   VectorFunctionCoefficient forcing(
      3, [nu](const Vector & x, double t, Vector & f)
   {
      f(0) = Gp(t) * x[1] * x[1] + G(t) * (1.0 - 2.0 * nu);
      f(1) = Gp(t) * x[2] * x[2] + G(t) * (1.0 - 2.0 * nu);
      f(2) = Gp(t) * x[0] * x[0] + G(t) * (1.0 - 2.0 * nu);
   });

   Mesh serial = MakeBoxMesh(UnitBox(3, 2));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 6; ++attr) { bc.AddVelocityDirichlet(attr, u_exact); }

   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = dt;
   opts.t_final = t_final;
   opts.convection = false;  // Stokes path, explicitly
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesTimeIntegrator stepper(spaces, rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);
   for (int s = 0; s < 5; ++s) { stepper.Step(); }

   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::CUBE] = &rules.Get(Geometry::CUBE, 2 * 2 + 4);
   u_exact.SetTime(stepper.Time());
   // With convection off this is the Stokes MMS, which reproduces exactly.
   EXPECT_LE(stepper.Velocity().ComputeL2Error(u_exact, irs), 1e-8);
}

// ===========================================================================
// ROTATIONAL FORM (rotational_convection_pa_spec.md, step 6): the lagged-
// vorticity term (curl w*) x u in the IMPLICIT velocity block, w* the EXT
// extrapolation of the history; the solve yields the Bernoulli head
// P = p + 1/2|u|^2 and Pressure() reports static p.
//
// The SAME exact solution and forcing apply unchanged: the forcing is the
// physical f = u_t + (u.grad)u - nu lap u + grad p, and (u.grad)u =
// (curl u) x u + grad(1/2|u|^2), so (u, P) solves the rotational equation
// exactly. Unlike the periodic TGV -- whose convective term is a pure gradient
// the pressure absorbs -- these fields have a NON-gradient (curl u) x u, so the
// velocity genuinely depends on the rotation term (and on its Dirichlet
// elimination: the boundary data is inhomogeneous and time-dependent).
// ===========================================================================
namespace
{
MmsConfig Rotational(RotationVelocityPC pc = RotationVelocityPC::Symmetric,
                     SchurBlockType schur = SchurBlockType::Mass)
{
   MmsConfig c;
   c.form = ConvectiveForm::Rotational;
   c.rotation_pc = pc;
   c.schur = schur;
   return c;
}
} // namespace

// Temporal order of the semi-implicit scheme: BDF2 with EXT2-lagged vorticity
// (EXT1 + trapezoidal split on the starter) is second order.
//
// Measured AGAINST A FINE-dt REFERENCE ON THE SAME MESH, not against the exact
// solution: unlike the convective form, the rotational solve's pressure is
// P = p + 1/2|u|^2, degree 10 (2D) / 4 (3D) here -- outside the pressure space
// -- so the MMS is no longer spatially exact. Its O(h^k) floor (6.4e-4 2D,
// 1.6e-3 3D, flat in dt) would swamp the O(dt^2) error; differencing against
// a same-mesh reference cancels it, exactly as the TGV temporal study does.
void RotationalTemporalOrder(int dim, int n, int ku, double nu)
{
   const double t_final = 0.2;
   std::unique_ptr<VectorFunctionCoefficient> u_exact, forcing;
   if (dim == 2) { Fields2D(nu, u_exact, forcing); }
   else { Fields3D(nu, u_exact, forcing); }
   const MmsConfig rot = Rotational();
   const Vector ref = NseMmsRun(dim, n, ku, nu, 0.00125, t_final, *u_exact,
                                *forcing, rot).u_true;
   const double dts[3] = {0.02, 0.01, 0.005};
   double e[3];
   for (int i = 0; i < 3; ++i)
   {
      Vector d = NseMmsRun(dim, n, ku, nu, dts[i], t_final, *u_exact, *forcing,
                           rot).u_true;
      d -= ref;
      e[i] = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d));
   }
   ReportRates(dim == 2 ? "2D rotational (vs dt/16 ref)"
               : "3D rotational (vs dt/16 ref)", e[0], e[1], e[2]);
   ASSERT_GT(e[2], 1e-10)
         << "finest-dt difference is at the solver floor (" << e[2]
         << ") -- the rate assertions would be vacuous";
   EXPECT_GT(Rate(e[0], e[1], 2.0), 1.7) << "e=" << e[0] << " -> " << e[1];
   EXPECT_GT(Rate(e[1], e[2], 2.0), 1.7) << "e=" << e[1] << " -> " << e[2];
}

// nu = 0.05, deliberately LOW: an O(dt) defect confined to the first
// (trapezoidal) step -- e.g. a dropped explicit half of N -- decays like
// exp(-2 nu pi^2 t); at the convective study's nu = 0.7 it is damped ~16x by
// t_final and hides under the O(dt^2) error. At 0.05 it survives (~0.8x).
TEST(NseMms, RotationalTemporalOrder2D) { RotationalTemporalOrder(2, 3, 3, 0.05); }

TEST(NseMms, RotationalTemporalOrder3D) { RotationalTemporalOrder(3, 2, 2, 0.05); }

// Every velocity-block PC (symmetric / point-block Jacobi once / PBJ-GMRES)
// on both Schur paths solves the SAME discrete system: the outer FGMRES
// applies the true operator, so the marches must agree to solver tolerance.
// A PC that broke the essential rows or applied a stale skew would show up as
// non-convergence (the stepper aborts) or a different answer.
TEST(NseMms, RotationalPreconditionersAgree)
{
   const double nu = 0.7, t_final = 0.06, dt = 0.02;
   std::unique_ptr<VectorFunctionCoefficient> u_exact, forcing;
   Fields2D(nu, u_exact, forcing);
   std::vector<MmsResult> runs;
   for (SchurBlockType schur : {SchurBlockType::Mass, SchurBlockType::CahouetChabard})
      for (RotationVelocityPC pc :
           {
              RotationVelocityPC::Symmetric,
              RotationVelocityPC::PbjOnly,
              RotationVelocityPC::PbjKrylov
           })
      {
         runs.push_back(NseMmsRun(2, 3, 3, nu, dt, t_final, *u_exact, *forcing,
                                  Rotational(pc, schur)));
      }
   // LOR-AMG with the rotation term in the LOR operator, re-set-up every
   // step, on both Schur paths (Mass: velocity_prec LORAMG; CC: a_pc LORAMG,
   // its default). Also exercises the w -> LOR-space copy in a real march.
   for (SchurBlockType schur : {SchurBlockType::Mass, SchurBlockType::CahouetChabard})
   {
      MmsConfig cfg = Rotational(RotationVelocityPC::Symmetric, schur);
      cfg.velocity_prec = incns::VelocityPreconditioner::LORAMG;
      cfg.rotation_in_lor = true;
      runs.push_back(NseMmsRun(2, 3, 3, nu, dt, t_final, *u_exact, *forcing,
                               cfg));
   }
   const Vector& ref = runs.front().u_true;
   const double ref_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, ref, ref));
   for (std::size_t i = 1; i < runs.size(); ++i)
   {
      Vector d(runs[i].u_true);
      d -= ref;
      const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d)) / ref_norm;
      EXPECT_LE(rel, 1e-9) << "configuration " << i << " (schur x pc, row-major;"
                           << " 6, 7 = LOR-AMG with N on Mass, CC)";
   }
}

// Static pressure: Pressure() = P - I(1/2|u|^2) must converge to the
// manufactured p under h-refinement. Without the correction it would sit at
// the O(1) offset 1/2|u|^2 (here up to ~20) and not converge at all.
TEST(NseMms, RotationalStaticPressure)
{
   const double nu = 0.7, t_final = 0.05, dt = 0.0025;
   std::unique_ptr<VectorFunctionCoefficient> u_exact, forcing;
   Fields2D(nu, u_exact, forcing);
   FunctionCoefficient p_exact([](const Vector & x, double t)
   { return G(t) * (x[0] * x[0] + x[1] * x[1]); }); // grad p = G (2x, 2y)
   const MmsResult c = NseMmsRun(2, 3, 3, nu, dt, t_final, *u_exact, *forcing,
                                 Rotational(), &p_exact);
   const MmsResult f = NseMmsRun(2, 6, 3, nu, dt, t_final, *u_exact, *forcing,
                                 Rotational(), &p_exact);
   const double rate = Rate(c.p_err, f.p_err, 2.0);
   const double rate_u = Rate(c.u_err, f.u_err, 2.0);
   if (mfem::Mpi::Root())
   {
      std::cout << "[ NSE MMS  ] rotational static p: e(n=3)=" << c.p_err
                << "  e(n=6)=" << f.p_err << "  rate=" << rate
                << " | u: e(n=3)=" << c.u_err << "  e(n=6)=" << f.u_err
                << "  rate=" << rate_u << std::endl;
   }
   // The velocity against the EXACT solution (the temporal study differences
   // against a reference, so a consistent defect -- e.g. a Dirichlet
   // elimination missing N u_D -- cancels there; it cannot cancel here). Q3
   // velocity, but limited by the pressure (P is outside Q2): asserted > 2.
   EXPECT_GT(rate_u, 2.0);
   // Q2 pressure: O(h^3) for the static pressure (the 1/2|u|^2 interpolant
   // and P_h are both third order). Asserted > 2.
   EXPECT_GT(rate, 2.0);
   EXPECT_LT(f.p_err, 1e-2);
}
