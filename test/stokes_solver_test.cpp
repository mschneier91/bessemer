// Sprint 1.5 green criteria for src/solver/stokes_solver:
//
//  * Steady polynomial exactness MMS with NONTRIVIAL pressure, 2D and 3D: a
//    divergence-free polynomial velocity (degree <= k_u) and a zero-mean
//    polynomial pressure (degree <= k_p), f = -nu*lap(u) + grad(p), exact
//    velocity imposed as Dirichlet data on the whole box. The exact solution
//    lies in the discrete space, so the solve must reproduce it to (tightened)
//    Krylov tolerance -- any h-dependent error is a bug. All-Dirichlet means the
//    pressure null space exists, so the orthogonalization path runs against a
//    nontrivial pressure; comparison is after the mean-zero shift (the exact
//    pressures below are already zero-mean).
//
//  * Spatial order (2D trig MMS): L2 rates >= k_u+1 - 0.2 (velocity) and
//    k_p+1 - 0.2 (pressure) for the Q3/Q2 pair, with elevated error quadrature
//    and Krylov tolerance well below the finest discretization error.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "post/pressure_mean.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesSolver;
using incns::StokesSolverOptions;
using incns::VelocityPreconditioner;

namespace
{

BoxSpec UnitBox(int dim, int n)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// Elevated error-measurement quadrature (convergence-study hygiene: never the
// default rule).
void ElevatedRules(const RuleBook& rules, int order,
                   const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, order);
   irs[Geometry::CUBE] = &rules.Get(Geometry::CUBE, order);
}

struct MmsResult
{
   double u_err, p_err;
   int iterations;
   bool converged;
};

// Solve steady Stokes with exact-velocity Dirichlet data on the whole box.
// With stretch=true the box gets wall-normal tanh clustering in every
// direction; the exact polynomial solution still lies in the discrete space, so
// the solve must reproduce it -- this is what verifies the operators stay exact
// under the per-element metric factors of a non-uniform mesh.
MmsResult SolveMms(int dim, int n, int ku, double nu,
                   VectorFunctionCoefficient& u_exact,
                   FunctionCoefficient& p_exact,
                   VectorFunctionCoefficient& forcing,
                   double rtol, bool stretch = false,
                   incns::VelocityPreconditioner prec =
                      incns::VelocityPreconditioner::Jacobi)
{
   BoxSpec box = UnitBox(dim, n);
   if (stretch)
   {
      for (int d = 0; d < dim; ++d)
      {
         box.stretch[d] = incns::Stretch::TwoSidedTanh;
         box.stretch_beta[d] = 2.0;
      }
   }
   Mesh serial = MakeBoxMesh(box);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;

   BoundaryConditions bc(spaces.Velocity());
   const int n_attr = 2 * dim; // Cartesian box: 4 sides in 2D, 6 faces in 3D
   for (int attr = 1; attr <= n_attr; ++attr)
   {
      bc.AddVelocityDirichlet(attr, u_exact);
   }
   EXPECT_TRUE(bc.PressureNullspaceExists()); // all-Dirichlet by construction

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.rtol = rtol;
   opts.max_iter = 5000;
   opts.kdim = 400;
   opts.velocity_prec = prec;
   StokesSolver solver(spaces, rules, bc, opts);

   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 2 * ku + 4, irs);
   MmsResult r;
   r.u_err = u.ComputeL2Error(u_exact, irs);
   r.p_err = p.ComputeL2Error(p_exact, irs);
   r.iterations = solver.Iterations();
   r.converged = solver.Converged();
   if (Mpi::Root())
   {
      mfem::out << "[mms] dim=" << dim << " n=" << n << " Q" << ku << "/Q"
                << ku - 1 << "  u_err=" << r.u_err << "  p_err=" << r.p_err
                << "  iters=" << r.iterations << std::endl;
   }
   return r;
}

} // namespace

// 2D, Q3/Q2: u = curl(x^3 y^3) = (3x^3y^2, -3x^2y^3) (div-free, in Q3),
// p = x^2 + y^2 - 2/3 (zero mean, in Q2), f = -nu*lap(u) + grad(p).
TEST(StokesSolver, PolynomialExactness2D)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
   {
      v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0;
   });
   VectorFunctionCoefficient forcing(2, [nu](const Vector & x, Vector & f)
   {
      const double lap_u0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
      const double lap_u1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
      f(0) = -nu * lap_u0 + 2.0 * x[0];
      f(1) = -nu * lap_u1 + 2.0 * x[1];
   });

   const MmsResult r = SolveMms(2, 3, 3, nu, u_exact, p_exact, forcing, 1e-12);
   EXPECT_TRUE(r.converged) << "FGMRES did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

// 3D, Q2/Q1: u = (y^2, z^2, x^2) (div-free, in Q2), p = x + y + z - 3/2
// (zero mean, in Q1), f = -nu*(2,2,2) + (1,1,1).
TEST(StokesSolver, PolynomialExactness3D)
{
   const double nu = 1.3;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, Vector & v)
   {
      v(0) = x[1] * x[1];
      v(1) = x[2] * x[2];
      v(2) = x[0] * x[0];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] + x[1] + x[2] - 1.5;
   });
   VectorFunctionCoefficient forcing(3, [nu](const Vector&, Vector & f)
   {
      f = 1.0 - 2.0 * nu;
   });

   const MmsResult r = SolveMms(3, 2, 2, nu, u_exact, p_exact, forcing, 1e-12);
   EXPECT_TRUE(r.converged) << "FGMRES did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

// Same 2D polynomial MMS on a wall-normal-STRETCHED box (H2). The elements are
// non-uniform in size; a polynomial-exact reproduction proves the mass /
// viscous / divergence operators stay exact under the per-element metric.
TEST(StokesSolver, PolynomialExactness2DStretched)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
   {
      v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0;
   });
   VectorFunctionCoefficient forcing(2, [nu](const Vector & x, Vector & f)
   {
      const double lap_u0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
      const double lap_u1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
      f(0) = -nu * lap_u0 + 2.0 * x[0];
      f(1) = -nu * lap_u1 + 2.0 * x[1];
   });

   const MmsResult r =
      SolveMms(2, 3, 3, nu, u_exact, p_exact, forcing, 1e-12, /*stretch=*/true);
   EXPECT_TRUE(r.converged) << "FGMRES did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

// Same 3D polynomial MMS on a stretched hex box (H2, Sprint 1's only 3D cover).
TEST(StokesSolver, PolynomialExactness3DStretched)
{
   const double nu = 1.3;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, Vector & v)
   {
      v(0) = x[1] * x[1];
      v(1) = x[2] * x[2];
      v(2) = x[0] * x[0];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] + x[1] + x[2] - 1.5;
   });
   VectorFunctionCoefficient forcing(3, [nu](const Vector&, Vector & f)
   {
      f = 1.0 - 2.0 * nu;
   });

   const MmsResult r =
      SolveMms(3, 2, 2, nu, u_exact, p_exact, forcing, 1e-12, /*stretch=*/true);
   EXPECT_TRUE(r.converged) << "FGMRES did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

// H5: the steady polynomial MMS must reproduce exactly with BoomerAMG on the
// velocity block (needs the assembled momentum matrix, a different route).
TEST(StokesSolver, PolynomialExactness2DBoomerAMG)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
   {
      v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
      v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] * x[0] + x[1] * x[1] - 2.0 / 3.0;
   });
   VectorFunctionCoefficient forcing(2, [nu](const Vector & x, Vector & f)
   {
      const double lap_u0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
      const double lap_u1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
      f(0) = -nu * lap_u0 + 2.0 * x[0];
      f(1) = -nu * lap_u1 + 2.0 * x[1];
   });

   const MmsResult r = SolveMms(2, 3, 3, nu, u_exact, p_exact, forcing, 1e-12,
                                /*stretch=*/false, VelocityPreconditioner::BoomerAMG);
   EXPECT_TRUE(r.converged) << "FGMRES(AMG) did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

TEST(StokesSolver, PolynomialExactness3DBoomerAMG)
{
   const double nu = 1.3;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, Vector & v)
   {
      v(0) = x[1] * x[1];
      v(1) = x[2] * x[2];
      v(2) = x[0] * x[0];
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return x[0] + x[1] + x[2] - 1.5;
   });
   VectorFunctionCoefficient forcing(3, [nu](const Vector&, Vector & f)
   {
      f = 1.0 - 2.0 * nu;
   });

   const MmsResult r = SolveMms(3, 2, 2, nu, u_exact, p_exact, forcing, 1e-12,
                                /*stretch=*/false, VelocityPreconditioner::BoomerAMG);
   EXPECT_TRUE(r.converged) << "FGMRES(AMG) did not converge (" << r.iterations
                            << " iterations)";
   EXPECT_LE(r.u_err, 1e-8);
   EXPECT_LE(r.p_err, 1e-7);
}

// Spatial convergence, 2D trig MMS on Q3/Q2 (rates: velocity 4, pressure 3):
// psi = sin(pi x) sin(pi y), u = curl(psi), p = cos(pi x) cos(pi y) (zero mean).
// u is an eigenfunction of the Laplacian: lap(u) = -2 pi^2 u.
TEST(StokesSolver, SpatialOrder2D)
{
   const double nu = 1.0;
   const double pi = M_PI;
   VectorFunctionCoefficient u_exact(2, [pi](const Vector & x, Vector & v)
   {
      v(0) = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      v(1) = -pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
   });
   FunctionCoefficient p_exact([pi](const Vector & x)
   {
      return std::cos(pi * x[0]) * std::cos(pi * x[1]);
   });
   VectorFunctionCoefficient forcing(2, [pi, nu](const Vector & x, Vector & f)
   {
      const double u0 = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      const double u1 = -pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
      const double px = -pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      const double py = -pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
      f(0) = 2.0 * pi * pi * nu * u0 + px;
      f(1) = 2.0 * pi * pi * nu * u1 + py;
   });

   double u_err[3], p_err[3];
   const int meshes[3] = {4, 8, 16};
   for (int i = 0; i < 3; ++i)
   {
      const MmsResult r = SolveMms(2, meshes[i], 3, nu, u_exact, p_exact,
                                   forcing, 1e-12);
      ASSERT_TRUE(r.converged) << "mesh n=" << meshes[i] << " ("
                               << r.iterations << " iterations)";
      u_err[i] = r.u_err;
      p_err[i] = r.p_err;
   }

   const double u_rate = std::log2(u_err[1] / u_err[2]);
   const double p_rate = std::log2(p_err[1] / p_err[2]);
   EXPECT_GE(u_rate, 4.0 - 0.2) << "velocity L2 rate too low";
   EXPECT_GE(p_rate, 3.0 - 0.2) << "pressure L2 rate too low";
   // Errors must actually decrease from the coarsest mesh, too.
   EXPECT_LT(u_err[1], u_err[0]);
   EXPECT_LT(p_err[1], p_err[0]);
}
