// Sprint 1.8 gate #2 -- TGV-Stokes temporal order (runs only because the
// unsteady MMS gate exists and passes first). The 2D TGV velocity solves
// unforced unsteady STOKES exactly (lap(u) = -2u; its convective term is a
// pure gradient absorbed into the NSE pressure), so with convection off the
// velocity matches and the pressure -> 0.
//
// Temporal refinement at a fixed mesh, errors measured against a SAME-MESH
// fine-dt reference solution so the (larger) spatial error cancels exactly and
// the rate is purely temporal. Assert order ~= 2 for BDF2 and ~= 3 for BDF3
// (test-only mode: the 1.9 LTE estimator is trustworthy only if this path is
// genuinely 3rd order).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
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
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
constexpr double kNu = 1.0;
constexpr double kTFinal = 0.4;

struct TgvRun
{
   Vector u_true;   // velocity true dofs at t_final
   double p_norm;   // ||p||_L2 at t_final
};

// March the unforced TGV on the fully periodic [0,2pi]^2 box (n^2 Q3/Q2).
TgvRun MarchTgv(MixedSpaces& spaces, const RuleBook& rules,
                BoundaryConditions& bc, VectorCoefficient& zero_forcing,
                double dt, int order)
{
   TimeIntegratorOptions opts;
   opts.nu = kNu;
   opts.dt = dt;
   opts.t_final = kTFinal;
   opts.order = order;
   opts.rtol = 1e-11;
   StokesTimeIntegrator stepper(spaces, rules, bc, zero_forcing, opts);

   VectorFunctionCoefficient u0(2, [](const Vector & x, double t, Vector & u)
   { incns::tgv2d::Velocity(x, t, kNu, u); });
   stepper.SetInitialVelocity(u0);
   stepper.Run();
   EXPECT_NEAR(stepper.Time(), kTFinal, 1e-12);

   TgvRun r;
   r.u_true.SetSize(spaces.Velocity().GetTrueVSize());
   stepper.Velocity().GetTrueDofs(r.u_true);
   ConstantCoefficient zero_p(0.0);
   r.p_norm = stepper.Pressure().ComputeL2Error(zero_p);
   return r;
}

// Discrete L2 (mass-weighted would be tighter, but the plain l2 true-dof norm
// of the difference is a fixed norm on a fixed mesh -- rates are unaffected).
double DiffNorm(const Vector& a, const Vector& b, MPI_Comm comm)
{
   Vector d(a);
   d -= b;
   return std::sqrt(InnerProduct(comm, d, d));
}

void TemporalOrderStudy(int order, double expected_rate)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {8, 8, 0}; // fully periodic (default), 2pi box
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity());
   Vector zero_vec(2);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_forcing(zero_vec);

   const TgvRun ref = MarchTgv(spaces, rules, bc, zero_forcing, 0.003125, order);

   const double dts[3] = {0.1, 0.05, 0.025};
   double errs[3];
   for (int i = 0; i < 3; ++i)
   {
      const TgvRun run = MarchTgv(spaces, rules, bc, zero_forcing, dts[i], order);
      errs[i] = DiffNorm(run.u_true, ref.u_true, MPI_COMM_WORLD);
      // Stokes-TGV pressure is identically zero in the continuum. Discretely
      // the projected velocity is not exactly divergence-free in the FE sense,
      // so a small pressure appears at the SPATIAL discretization scale
      // (~2e-4 at n=8 Q3/Q2, dt-independent -- measured constant to 1% across
      // this dt sweep). It vanishes under h-refinement, not dt-refinement;
      // assert the ceiling, well below the O(1) NSE pressure it must not be.
      EXPECT_LE(run.p_norm, 1e-3) << "dt=" << dts[i];
      if (Mpi::Root())
      {
         mfem::out << "[tgv-bdf" << order << "] dt=" << dts[i]
                   << "  err=" << errs[i] << "  ||p||=" << run.p_norm
                   << std::endl;
      }
   }

   const double rate01 = std::log2(errs[0] / errs[1]);
   const double rate12 = std::log2(errs[1] / errs[2]);
   if (Mpi::Root())
   {
      mfem::out << "[tgv-bdf" << order << "] rates: " << rate01 << ", "
                << rate12 << std::endl;
   }
   EXPECT_GE(rate12, expected_rate - 0.2);
   EXPECT_LT(errs[2], errs[1]);
   EXPECT_LT(errs[1], errs[0]);
}
} // namespace

TEST(TgvStokesTemporal, Bdf2SecondOrder) { TemporalOrderStudy(2, 2.0); }

TEST(TgvStokesTemporal, Bdf3ThirdOrderTestMode) { TemporalOrderStudy(3, 3.0); }
