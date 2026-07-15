// Sprint 1.8 gate #1 -- unsteady polynomial MMS (runs BEFORE any TGV rate is
// measured): multiply the 1.5 steady polynomial solutions by a quadratic
// g(t) = 1 + t + t^2/2. The trapezoidal starter and BDF2 are both exact on
// quadratics in time, and the spatial parts lie in the discrete spaces, so the
// ENTIRE unsteady solve -- history, the starter->BDF2 ramp, forcing evaluated
// at t^{n+1}, and TIME-DEPENDENT Dirichlet data re-eliminated every step --
// must reproduce the exact solution to solver tolerance at EVERY step. This is
// the only Sprint-1 test that exercises time-dependent Dirichlet elimination
// (the periodic TGV never touches that path). 2D and 3D.
//
// Known property (documented in time_integrator.hpp): the trapezoidal step's
// pressure is the time-average (p^0+p^1)/2, not p(t^1), so pressure is checked
// from step 2 on; velocity is checked at every step.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <functional>

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

void RunUnsteadyMms(int dim, int n, int ku, double nu,
                    VectorFunctionCoefficient& u_exact,
                    FunctionCoefficient& p_exact,
                    VectorFunctionCoefficient& forcing,
                    incns::VelocityPreconditioner prec =
                       incns::VelocityPreconditioner::Jacobi)
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
   opts.dt = 0.02;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   opts.velocity_prec = prec;
   StokesTimeIntegrator stepper(spaces, rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);

   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   irs[geom] = &rules.Get(geom, 2 * ku + 4);

   for (int step = 1; step <= 5; ++step)
   {
      stepper.Step();
      u_exact.SetTime(stepper.Time());
      p_exact.SetTime(stepper.Time());
      const double u_err = stepper.Velocity().ComputeL2Error(u_exact, irs);
      EXPECT_LE(u_err, 1e-8) << "dim=" << dim << " step " << step
                             << " (t=" << stepper.Time() << ")";
      if (step >= 2)
      {
         const double p_err = stepper.Pressure().ComputeL2Error(p_exact, irs);
         EXPECT_LE(p_err, 1e-7) << "dim=" << dim << " step " << step;
      }
   }
}

// Build the 2D/3D coefficients and run, parametrized by the velocity-block
// preconditioner so Jacobi (default) and BoomerAMG share the exact same MMS.
void Unsteady2D(incns::VelocityPreconditioner prec)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      v(1) = -G(t) * 3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
   });
   FunctionCoefficient p_exact([](const Vector & x, double t)
   {
      return G(t) * (x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0);
   });
   // f = g' u_s - nu g lap(u_s) + g grad(p_s)
   VectorFunctionCoefficient forcing(2, [nu](const Vector & x, double t,
                                     Vector & f)
   {
      const double us0 = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      const double us1 = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
      const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
      const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
      f(0) = Gp(t) * us0 + G(t) * (-nu * lap0 + 2.0 * x[0]);
      f(1) = Gp(t) * us1 + G(t) * (-nu * lap1 + 2.0 * x[1]);
   });

   RunUnsteadyMms(2, 3, 3, nu, u_exact, p_exact, forcing, prec);
}

void Unsteady3D(incns::VelocityPreconditioner prec)
{
   const double nu = 1.3;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * x[1];
      v(1) = G(t) * x[2] * x[2];
      v(2) = G(t) * x[0] * x[0];
   });
   FunctionCoefficient p_exact([](const Vector & x, double t)
   {
      return G(t) * (x[0] + x[1] + x[2] - 1.5);
   });
   VectorFunctionCoefficient forcing(3, [nu](const Vector & x, double t,
                                     Vector & f)
   {
      f(0) = Gp(t) * x[1] * x[1] + G(t) * (1.0 - 2.0 * nu);
      f(1) = Gp(t) * x[2] * x[2] + G(t) * (1.0 - 2.0 * nu);
      f(2) = Gp(t) * x[0] * x[0] + G(t) * (1.0 - 2.0 * nu);
   });

   RunUnsteadyMms(3, 2, 2, nu, u_exact, p_exact, forcing, prec);
}
} // namespace

TEST(UnsteadyMms, QuadraticInTime2D)
{
   Unsteady2D(incns::VelocityPreconditioner::Jacobi);
}

TEST(UnsteadyMms, QuadraticInTime3D)
{
   Unsteady3D(incns::VelocityPreconditioner::Jacobi);
}

// H5: the same unsteady MMS must reproduce exactly with BoomerAMG on the
// velocity block (a different assembly route -- the assembled momentum matrix).
TEST(UnsteadyMms, QuadraticInTime2D_BoomerAMG)
{
   Unsteady2D(incns::VelocityPreconditioner::BoomerAMG);
}

TEST(UnsteadyMms, QuadraticInTime3D_BoomerAMG)
{
   Unsteady3D(incns::VelocityPreconditioner::BoomerAMG);
}
