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
#include "spaces/mixed_spaces.hpp"
#include "quadrature/rule_book.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
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

// March the NSE MMS to t_final at step dt; return the final velocity L2 error.
double NseMmsError(int dim, int n, int ku, double nu, double dt, double t_final,
                   VectorFunctionCoefficient& u_exact,
                   VectorFunctionCoefficient& forcing)
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
   return stepper.Velocity().ComputeL2Error(u_exact, irs);
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

   const double r1 = Rate(e1, e2, 2.0);
   const double r2 = Rate(e2, e3, 2.0);
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

   const double r1 = Rate(e1, e2, 2.0);
   const double r2 = Rate(e2, e3, 2.0);
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
