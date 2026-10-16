// H4 -- physical diagnostics (kinetic energy, viscous dissipation, divergence
// norm). Two kinds of check:
//
//  * Algebraic, exact to machine precision on closed-form fields (localizes a
//    bug to the routine, no solver involved).
//  * The flagship: the unforced 2D TGV solves unsteady Stokes with velocity
//    decaying as e^{-2 nu t}, so its kinetic energy must follow the exact
//    e^{-4 nu t} law -- a sharper temporal-accuracy check than any iteration
//    baseline. Its divergence must stay ~0 (the same measure the divergence
//    fast-tier check uses).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/periodic_box.hpp"
#include "post/diagnostics.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::DivergenceNorm;
using incns::DissipationRate;
using incns::KineticEnergy;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
// Unit box, non-periodic, n^2 (or n^3) Q_k/Q_{k-1} (default Q3/Q2), for the
// algebraic checks.
MixedSpaces* MakeUnit(ParMesh& mesh, int dim, int n, int order = 3)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   mesh = ParMesh(MPI_COMM_WORLD, serial);
   return new MixedSpaces(mesh, order, order - 1);
}
} // namespace

// u = (3, 4) constant on the unit square: KE = 0.5*(9+16)*|Omega| = 12.5.
TEST(Diagnostics, KineticEnergyConstantField)
{
   ParMesh mesh;
   MixedSpaces* spaces = MakeUnit(mesh, 2, 2);
   RuleBook rules;
   ParGridFunction u(&spaces->Velocity());
   Vector c(2);
   c(0) = 3.0;
   c(1) = 4.0;
   VectorConstantCoefficient cc(c);
   u.ProjectCoefficient(cc);
   EXPECT_NEAR(KineticEnergy(u, rules), 12.5, 1e-12);
   delete spaces;
}

// u = (x, y): div u = 2, ||div u|| = sqrt(int 4) = 2 on the unit square.
// (Matches the closed form the grad-div/divergence checks use.)
TEST(Diagnostics, DivergenceKnownField)
{
   ParMesh mesh;
   MixedSpaces* spaces = MakeUnit(mesh, 2, 3);
   RuleBook rules;
   ParGridFunction u(&spaces->Velocity());
   VectorFunctionCoefficient lin(2, [](const Vector & x, Vector & v)
   {
      v(0) = x[0];
      v(1) = x[1];
   });
   u.ProjectCoefficient(lin);
   EXPECT_NEAR(DivergenceNorm(u, rules), 2.0, 1e-12);
   delete spaces;
}

// u = (y, 0): grad u has a single unit entry, int|grad u|^2 = |Omega| = 1, so
// dissipation = nu * 1. Check with nu = 0.5.
TEST(Diagnostics, DissipationLinearShear)
{
   ParMesh mesh;
   MixedSpaces* spaces = MakeUnit(mesh, 2, 3);
   RuleBook rules;
   ParGridFunction u(&spaces->Velocity());
   VectorFunctionCoefficient shear(2, [](const Vector & x, Vector & v)
   {
      v(0) = x[1];
      v(1) = 0.0;
   });
   u.ProjectCoefficient(shear);
   EXPECT_NEAR(DissipationRate(u, 0.5, rules), 0.5, 1e-12);
   delete spaces;
}

// FlowDiagnostics (the assembled-operator form the solver logs with) returns
// the same three numbers as the free functions, in 2D and 3D, velocity orders
// 2 to 6 (each 3D order has its own compile-time-sized kernel), on a field
// that is neither divergence-free nor polynomial; and it reports a refined
// space.
TEST(Diagnostics, AssembledFormsMatchTheFreeFunctions)
{
   for (const int dim : {2, 3})
   {
      for (const int order : {2, 3, 4, 5, 6})
      {
         SCOPED_TRACE(testing::Message() << "dim " << dim << " order " << order);
         ParMesh mesh;
         MixedSpaces* spaces = MakeUnit(mesh, dim, dim == 2 ? 4 : 2, order);
         RuleBook rules;
         ParGridFunction u(&spaces->Velocity());
         VectorFunctionCoefficient f(dim, [dim](const Vector & x, Vector & v)
         {
            v(0) = std::sin(1.3 * x[0]) * std::cos(0.7 * x[1]) + x[1] * x[1];
            v(1) = std::cos(0.9 * x[0] + 0.4 * x[1]) + 2.0 * x[0];
            if (dim == 3)
            {
               v(0) += 0.5 * x[2];
               v(2) = std::sin(x[0] + x[1] + 1.1 * x[2]);
            }
         });
         u.ProjectCoefficient(f);
         const incns::FlowDiagnostics d(spaces->Velocity(), rules);
         EXPECT_FALSE(d.Stale());
         const double ke = KineticEnergy(u, rules);
         const double eps = DissipationRate(u, 0.37, rules);
         const double div = DivergenceNorm(u, rules);
         ASSERT_GT(div, 0.1);
         EXPECT_NEAR(d.KineticEnergy(u), ke, 1e-12 * ke);
         EXPECT_NEAR(d.DissipationRate(u, 0.37), eps, 1e-12 * eps);
         EXPECT_NEAR(d.DivergenceNorm(u), div, 1e-12 * div);
         delete spaces;
      }
   }
}

// Flagship: TGV-Stokes kinetic energy follows the exact e^{-4 nu t} law, and
// the divergence stays ~0. Fine mesh + BDF3 (test-only 3rd order) so the
// discretization error is well below the tolerance.
TEST(Diagnostics, TgvStokesKineticEnergyDecay)
{
   const double nu = 1.0;
   const double t_final = 0.4;

   BoxSpec s;
   s.dim = 2;
   s.num_elems = {8, 8, 0};      // [0,2pi]^2, default lengths, fully periodic
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity()); // empty on periodic mesh = periodic

   Vector zerov(2);
   zerov = 0.0;
   VectorConstantCoefficient zero_forcing(zerov);

   VectorFunctionCoefficient u0(2, [nu](const Vector & x, double t, Vector & u)
   { incns::tgv2d::Velocity(x, t, nu, u); });

   // Initial KE from the projected discrete field (so the projection error
   // largely cancels in the ratio, isolating the temporal-decay accuracy).
   u0.SetTime(0.0);
   ParGridFunction u_init(&spaces.Velocity());
   u_init.ProjectCoefficient(u0);
   const double ke0 = KineticEnergy(u_init, rules);
   EXPECT_GT(ke0, 0.0);

   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = t_final / 40.0;
   opts.t_final = t_final;
   opts.order = 3;               // test-only BDF3
   opts.rtol = 1e-12;
   StokesTimeIntegrator stepper(spaces, rules, bc, zero_forcing, opts);
   stepper.SetInitialVelocity(u0);
   stepper.Run();
   ASSERT_NEAR(stepper.Time(), t_final, 1e-12);

   const double ke_end = KineticEnergy(stepper.Velocity(), rules);
   const double expected = ke0 * std::exp(-4.0 * nu * t_final);
   // Relative error against the exact decay law.
   EXPECT_LT(std::abs(ke_end - expected) / expected, 1e-3);

   // Incompressibility sanity: Taylor-Hood is only WEAKLY divergence-free, so
   // ||div u|| sits at the discretization floor (~1e-3 here), not machine zero
   // -- grad-div tightens it. The routine's exactness is pinned separately by
   // DivergenceKnownField above.
   EXPECT_LT(DivergenceNorm(stepper.Velocity(), rules), 1e-2);
}
