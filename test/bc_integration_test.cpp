// Stokes BC integration matrix -- exact (polynomial) solutions across the BC
// combinations the unit tests never SOLVE, only detect:
//
//  1. Steady Poiseuille 2D: Dirichlet inflow/walls + OUTFLOW (do-nothing).
//     The first solve ever through the Neumann branch. p = 2 nu (L - x)
//     satisfies the do-nothing condition nu (grad u) n - p n = 0 exactly at
//     the outlet (du/dx = 0 and p(L) = 0), so no boundary term is missing.
//     The outflow FIXES the pressure level: p is asserted in ABSOLUTE value
//     (no mean shift), pinning the no-nullspace branch end to end.
//  2. Unsteady Poiseuille 2D: the same times a quadratic g(t) -- per-step
//     Dirichlet re-elimination WITH an outflow present; the viscous and
//     pressure terms cancel so f = g' u_s. Exact at every step.
//  3. Steady 3D duct MMS: inflow/no-slip/outflow in 3D (the only non-Dirichlet
//     3D coverage). u = (y(1-y) z(1-z), 0, 0), p = L - x, nonzero forcing.
//  4. Unsteady periodic channel: x-PERIODIC + Dirichlet walls -- the DNS
//     target topology, solved (previously only the null-space detection was
//     tested). No outflow => the constant pressure null space exists; the
//     exact pressure is 0, so the orthogonalization + mean-zero path is
//     checked in the mixed periodic/Dirichlet setting.
//
// All solutions lie in the discrete spaces (velocity <= Q2, pressure <= Q1),
// so every case must reproduce them to (tightened) Krylov tolerance -- any
// h-dependent error is a bug. Boundary attributes are located geometrically
// (face-center coordinates), never hardcoded.
//
// Known limitation (deliberate): outflow with NONZERO traction needs boundary
// linear forms the library does not have -- a feature gap, not a test gap.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
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
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
constexpr double kLen = 2.0; // channel length (x); height (and depth) are 1

double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }

// Locate the boundary attribute whose faces sit at coordinate[ci] == value
// (globally reduced; a rank may own no such faces). Never hardcode attrs.
int FindBoundaryAttrAt(ParMesh& mesh, int ci, double value)
{
   int attr = 0;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      Vector c;
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      if (std::abs(c[ci] - value) < 1e-10) { attr = mesh.GetBdrAttribute(be); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MFEM_VERIFY(attr > 0, "no boundary attribute found at the requested plane");
   return attr;
}

void ElevatedRules(const RuleBook& rules, int dim, int order,
                   const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   irs[geom] = &rules.Get(geom, order);
}
} // namespace

// --- 1. Steady Poiseuille with an outflow ----------------------------------
TEST(BcIntegration, SteadyPoiseuilleOutflow2D)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
   {
      v(0) = x[1] * (1.0 - x[1]);
      v(1) = 0.0;
   });
   FunctionCoefficient p_exact([nu](const Vector & x)
   {
      return 2.0 * nu * (kLen - x[0]); // p(L) = 0: do-nothing holds exactly
   });
   Vector zf(2);
   zf = 0.0;
   VectorConstantCoefficient forcing(zf); // Poiseuille: f = 0

   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 2, 0};
   s.lengths = {kLen, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   RuleBook rules;

   const int outflow_attr = FindBoundaryAttrAt(mesh, 0, kLen);
   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 4; ++attr)
   {
      if (attr == outflow_attr) { bc.AddOutflow(attr); }
      else { bc.AddVelocityDirichlet(attr, u_exact); }
   }
   ASSERT_FALSE(bc.PressureNullspaceExists()); // outflow fixes the level

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesSolver solver(spaces, rules, bc, opts);
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);
   ASSERT_TRUE(solver.Converged());

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 2, 8, irs);
   EXPECT_LE(u.ComputeL2Error(u_exact, irs), 1e-8);
   // ABSOLUTE pressure: the outflow determines the level; no mean shift runs.
   EXPECT_LE(p.ComputeL2Error(p_exact, irs), 1e-7);
}

// --- 2. Unsteady Poiseuille with an outflow ---------------------------------
TEST(BcIntegration, UnsteadyPoiseuilleOutflow2D)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * (1.0 - x[1]);
      v(1) = 0.0;
   });
   FunctionCoefficient p_exact([nu](const Vector & x, double t)
   {
      return 2.0 * nu * G(t) * (kLen - x[0]);
   });
   // u_t - nu lap(u) + grad(p) = g' u_s + (2 nu g - 2 nu g) x_hat = g' u_s.
   VectorFunctionCoefficient forcing(2, [](const Vector & x, double t, Vector & f)
   {
      f(0) = Gp(t) * x[1] * (1.0 - x[1]);
      f(1) = 0.0;
   });

   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 2, 0};
   s.lengths = {kLen, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   RuleBook rules;

   const int outflow_attr = FindBoundaryAttrAt(mesh, 0, kLen);
   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 4; ++attr)
   {
      if (attr == outflow_attr) { bc.AddOutflow(attr); }
      else { bc.AddVelocityDirichlet(attr, u_exact); } // time-dependent data
   }
   ASSERT_FALSE(bc.PressureNullspaceExists());

   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = 0.02;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesTimeIntegrator stepper(spaces, rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 2, 8, irs);
   for (int step = 1; step <= 5; ++step)
   {
      stepper.Step();
      u_exact.SetTime(stepper.Time());
      p_exact.SetTime(stepper.Time());
      EXPECT_LE(stepper.Velocity().ComputeL2Error(u_exact, irs), 1e-8)
            << "step " << step;
      if (step >= 2) // trapezoidal starter's pressure is the time-average
      {
         EXPECT_LE(stepper.Pressure().ComputeL2Error(p_exact, irs), 1e-7)
               << "step " << step;
      }
   }
}

// --- 3. Steady 3D duct MMS with an outflow ----------------------------------
TEST(BcIntegration, SteadyDuctOutflow3D)
{
   const double nu = 1.3;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, Vector & v)
   {
      v = 0.0;
      v(0) = x[1] * (1.0 - x[1]) * x[2] * (1.0 - x[2]);
   });
   FunctionCoefficient p_exact([](const Vector & x)
   {
      return kLen - x[0]; // p(L) = 0
   });
   // f = -nu lap(u) + grad(p):
   // lap(u1) = -2 z(1-z) - 2 y(1-y);  grad p = (-1, 0, 0).
   VectorFunctionCoefficient forcing(3, [nu](const Vector & x, Vector & f)
   {
      f = 0.0;
      f(0) = 2.0 * nu * (x[2] * (1.0 - x[2]) + x[1] * (1.0 - x[1])) - 1.0;
   });

   BoxSpec s;
   s.dim = 3;
   s.num_elems = {2, 2, 2};
   s.lengths = {kLen, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   RuleBook rules;

   const int outflow_attr = FindBoundaryAttrAt(mesh, 0, kLen);
   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 6; ++attr)
   {
      if (attr == outflow_attr) { bc.AddOutflow(attr); }
      else { bc.AddVelocityDirichlet(attr, u_exact); }
   }
   ASSERT_FALSE(bc.PressureNullspaceExists());

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesSolver solver(spaces, rules, bc, opts);
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);
   ASSERT_TRUE(solver.Converged());

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 3, 8, irs);
   EXPECT_LE(u.ComputeL2Error(u_exact, irs), 1e-8);
   EXPECT_LE(p.ComputeL2Error(p_exact, irs), 1e-7); // absolute level again
}

// --- 4. Unsteady periodic channel (x-periodic + Dirichlet walls) ------------
TEST(BcIntegration, UnsteadyPeriodicChannel2D)
{
   const double nu = 0.7;
   VectorFunctionCoefficient u_exact(2, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * (1.0 - x[1]);
      v(1) = 0.0;
   });
   FunctionCoefficient p_exact([](const Vector&, double) { return 0.0; });
   // u_t - nu lap(u) + grad(p) = g' u_s + 2 nu g x_hat  (p = 0).
   VectorFunctionCoefficient forcing(2, [nu](const Vector & x, double t,
                                     Vector & f)
   {
      f(0) = Gp(t) * x[1] * (1.0 - x[1]) + 2.0 * nu * G(t);
      f(1) = 0.0;
   });

   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 2, 0};
   s.lengths = {kLen, 1.0, 1.0};
   s.periodic = {true, false, false}; // streamwise periodic, walls open
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   RuleBook rules;

   // Only the walls carry REAL boundary faces (x faces are periodic).
   const int bottom = FindBoundaryAttrAt(mesh, 1, 0.0);
   const int top = FindBoundaryAttrAt(mesh, 1, 1.0);
   BoundaryConditions bc(spaces.Velocity());
   bc.AddVelocityDirichlet(bottom, u_exact);
   bc.AddVelocityDirichlet(top, u_exact);
   ASSERT_TRUE(bc.PressureNullspaceExists()); // enclosed: no outflow anywhere

   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = 0.02;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   StokesTimeIntegrator stepper(spaces, rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);

   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(rules, 2, 8, irs);
   for (int step = 1; step <= 5; ++step)
   {
      stepper.Step();
      u_exact.SetTime(stepper.Time());
      EXPECT_LE(stepper.Velocity().ComputeL2Error(u_exact, irs), 1e-8)
            << "step " << step;
      if (step >= 2)
      {
         // Exact p = 0 (mean-zero already): the orthogonalized, mean-shifted
         // discrete pressure must vanish to solver tolerance.
         EXPECT_LE(stepper.Pressure().ComputeL2Error(p_exact, irs), 1e-7)
               << "step " << step;
      }
   }
}
