// Sprint 1.12 -- non-dimensionalization:
//  * Normalize() arithmetic, field by field (dimensional mode), idempotence,
//    and the dimensionless mode (records Re = 1/nu, rescales nothing).
//  * physics.Re deck convenience (nu = 1/Re in dimensionless mode).
//  * GOLD equivalence test: a dimensional TGV case (box 2*pi*L_ref, velocity
//    scale U_ref, dimensional nu = U*L/Re, IC supplied as a DIMENSIONAL
//    function through WrapDimensionalVelocity) marches to the same solution as
//    the standard dimensionless TGV at the same Re -- to roundoff, at every
//    rank count. This one test pins every transform (lengths, dt, nu, IC
//    scaling, coordinate mapping) simultaneously.

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/nondimensionalization.hpp"
#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/stokes_case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <fstream>

using namespace mfem;
using incns::MakeBoxMesh;
using incns::Parameters;
using incns::ScalingMode;
using incns::StokesCase;
using incns::WrapDimensionalVelocity;

TEST(Nondim, NormalizeArithmeticDimensional)
{
   const double L = 0.05, U = 3.0, nu_dim = 1.5e-3; // Re = 100
   Parameters p;
   p.nondim.mode = ScalingMode::Dimensional;
   p.nondim.L_ref = L;
   p.nondim.U_ref = U;
   p.mesh.dim = 2;
   p.mesh.lengths = {2.0 * M_PI * L, 2.0 * M_PI * L, 2.0 * M_PI * L};
   p.nu = nu_dim;
   p.dt = 0.001;      // seconds
   p.t_final = 0.02;  // seconds
   p.controller.atol = 0.6; // m/s

   p.Normalize();

   EXPECT_TRUE(p.nondim.normalized);
   EXPECT_NEAR(p.nondim.Re, U * L / nu_dim, 1e-12);
   EXPECT_NEAR(p.mesh.lengths[0], 2.0 * M_PI, 1e-13);
   EXPECT_NEAR(p.dt, 0.001 * U / L, 1e-13);
   EXPECT_NEAR(p.t_final, 0.02 * U / L, 1e-13);
   EXPECT_NEAR(p.nu, 1.0 / p.nondim.Re, 1e-15);
   EXPECT_NEAR(p.controller.atol, 0.6 / U, 1e-15);

   // Idempotent: a second call changes nothing.
   const double dt_after = p.dt, nu_after = p.nu;
   p.Normalize();
   EXPECT_DOUBLE_EQ(p.dt, dt_after);
   EXPECT_DOUBLE_EQ(p.nu, nu_after);
}

TEST(Nondim, DimensionlessModeRecordsReOnly)
{
   Parameters p;
   p.nu = 0.01;
   p.dt = 0.5;
   p.Normalize();
   EXPECT_NEAR(p.nondim.Re, 100.0, 1e-12);
   EXPECT_DOUBLE_EQ(p.dt, 0.5);      // untouched
   EXPECT_DOUBLE_EQ(p.nu, 0.01);     // untouched
   EXPECT_DOUBLE_EQ(p.nondim.TRef(), 1.0);
}

// physics.Re in a deck sets nu = 1/Re (dimensionless-mode convenience).
TEST(Nondim, DeckReConvenience)
{
   const std::string path = "nondim_re_deck.yaml";
   if (Mpi::Root())
   {
      std::ofstream f(path);
      f << "physics:\n  Re: 250.0\n";
   }
   MPI_Barrier(MPI_COMM_WORLD);
   const Parameters p = Parameters::LoadYAML(path);
   EXPECT_NEAR(p.nu, 1.0 / 250.0, 1e-15);
   EXPECT_NEAR(p.nondim.Re, 250.0, 1e-10);
}

// The gold test: dimensional TGV == dimensionless TGV after scaling.
TEST(Nondim, DimensionalTgvMatchesDimensionlessTwin)
{
   const double Re = 100.0;
   const double L = 0.05, U = 3.0;

   // --- dimensionless reference ------------------------------------------------
   Parameters ref;
   ref.nu = 1.0 / Re;
   ref.mesh.dim = 2;
   ref.mesh.num_elems = {8, 8, 8};
   ref.dt = 0.02;
   ref.t_final = 0.2;
   ref.initial_velocity = "taylor_green_2d";
   ref.Normalize();

   Mesh ref_serial = MakeBoxMesh(ref.mesh);
   ParMesh ref_mesh(MPI_COMM_WORLD, ref_serial);
   StokesCase ref_case(ref_mesh, ref);
   auto ref_ic = incns::MakeInitialVelocity(ref);
   ref_case.SetInitialVelocity(*ref_ic);
   ref_case.Run();
   Vector u_ref(ref_case.Spaces().Velocity().GetTrueVSize());
   ref_case.Velocity().GetTrueDofs(u_ref);

   // --- dimensional twin -------------------------------------------------------
   Parameters dim;
   dim.nondim.mode = ScalingMode::Dimensional;
   dim.nondim.L_ref = L;
   dim.nondim.U_ref = U;
   dim.nu = U * L / Re;                    // dimensional viscosity
   dim.mesh.dim = 2;
   dim.mesh.num_elems = {8, 8, 8};
   dim.mesh.lengths = {2.0 * M_PI * L, 2.0 * M_PI * L, 2.0 * M_PI * L};
   dim.dt = 0.02 * L / U;                  // seconds
   dim.t_final = 0.2 * L / U;              // seconds
   dim.Normalize();

   Mesh dim_serial = MakeBoxMesh(dim.mesh); // built from NORMALIZED lengths
   ParMesh dim_mesh(MPI_COMM_WORLD, dim_serial);
   StokesCase dim_case(dim_mesh, dim);

   // IC as a DIMENSIONAL function of dimensional x: U * tgv(x / L) at t = 0.
   auto u0 = WrapDimensionalVelocity(
                [L, U](const Vector & x_dim, double, Vector & u)
   {
      Vector x_star(x_dim);
      x_star /= L;
      incns::tgv2d::Velocity(x_star, 0.0, /*nu (unused at t=0)*/ 1.0, u);
      u *= U;
   },
   dim.nondim, 2);
   dim_case.SetInitialVelocity(*u0);
   dim_case.Run();

   EXPECT_NEAR(dim_case.Time(), ref_case.Time(), 1e-12);
   EXPECT_NEAR(dim_case.TimeDimensional(), ref_case.Time() * L / U, 1e-12);

   Vector u_dim(dim_case.Spaces().Velocity().GetTrueVSize());
   dim_case.Velocity().GetTrueDofs(u_dim);

   Vector diff(u_ref);
   diff -= u_dim;
   const double d = std::sqrt(InnerProduct(MPI_COMM_WORLD, diff, diff));
   const double n = std::sqrt(InnerProduct(MPI_COMM_WORLD, u_ref, u_ref));
   // Not bitwise: the scale/unscale round-trips (2*pi*L)/L etc. cost a few
   // ulps that propagate through the marches; 1e-11 relative is roundoff-level
   // agreement for ~15 solves.
   EXPECT_LE(d, 1e-11 * n);
}
