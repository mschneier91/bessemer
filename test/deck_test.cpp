// Sprint 1.11 green criterion -- the YAML TGV deck reproduces the in-code
// driver's results within tolerance THROUGH THE SAME LIBRARY SURFACE
// (StokesCase), and the written ParaView output exists with the high-order
// settings. Also pins the deck loader field by field against the committed
// cases/tgv2d_stokes.yaml.

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/stokes_case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <string>

using namespace mfem;
using incns::MakeBoxMesh;
using incns::MakeInitialVelocity;
using incns::Parameters;
using incns::StokesCase;

namespace
{
// March a TGV case and return the final velocity true dofs.
Vector RunCase(const Parameters& params)
{
   Mesh serial = MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   StokesCase flow_case(mesh, params);
   auto u0 = MakeInitialVelocity(params);
   flow_case.SetInitialVelocity(*u0);
   flow_case.Run();
   EXPECT_NEAR(flow_case.Time(), params.t_final, 1e-10);

   Vector u_true(flow_case.Spaces().Velocity().GetTrueVSize());
   flow_case.Velocity().GetTrueDofs(u_true);
   return u_true;
}
} // namespace

// Field-by-field: the committed deck loads to exactly the values it states,
// and unstated fields keep the library defaults.
TEST(Deck, LoadYamlFields)
{
   const Parameters p = Parameters::LoadYAML(INCNS_TGV_DECK);
   EXPECT_DOUBLE_EQ(p.nu, 1.0);
   EXPECT_EQ(p.mesh.dim, 2);
   EXPECT_EQ(p.mesh.num_elems[0], 8);
   EXPECT_EQ(p.mesh.num_elems[1], 8);
   EXPECT_TRUE(p.mesh.periodic[0]);
   EXPECT_TRUE(p.mesh.periodic[1]);
   EXPECT_NEAR(p.mesh.lengths[0], 2.0 * M_PI, 1e-12);
   EXPECT_DOUBLE_EQ(p.dt, 0.02);
   EXPECT_DOUBLE_EQ(p.t_final, 0.2);
   EXPECT_EQ(p.initial_velocity, "taylor_green_2d");
   EXPECT_TRUE(p.output.enabled);
   EXPECT_EQ(p.output.interval, 5);
   // Unstated fields: library defaults (Q3/Q2, BDF2, fixed step, no grad-div).
   EXPECT_EQ(p.order_u, 3);
   EXPECT_EQ(p.order_p, 2);
   EXPECT_EQ(p.time_order, 2);
   EXPECT_FALSE(p.adaptive);
   EXPECT_DOUBLE_EQ(p.grad_div, 0.0);
}

// The deck-driven case and the equivalent in-code case produce the same
// solution through the same surface (identical code path, so the difference
// is pure roundoff).
TEST(Deck, YamlReproducesInCodeDriver)
{
   // In-code: state the same case programmatically (output off for speed).
   Parameters in_code;
   in_code.nu = 1.0;
   in_code.mesh.dim = 2;
   in_code.mesh.num_elems = {8, 8, 8};
   in_code.dt = 0.02;
   in_code.t_final = 0.2;
   in_code.initial_velocity = "taylor_green_2d";
   in_code.output.enabled = false;
   in_code.Normalize(); // in-code drivers normalize explicitly (LoadYAML does it)

   Parameters deck = Parameters::LoadYAML(INCNS_TGV_DECK);
   deck.output.enabled = false; // same physics; skip I/O here

   const Vector u_code = RunCase(in_code);
   const Vector u_deck = RunCase(deck);

   Vector diff(u_code);
   diff -= u_deck;
   const double d = std::sqrt(InnerProduct(MPI_COMM_WORLD, diff, diff));
   const double ref = std::sqrt(InnerProduct(MPI_COMM_WORLD, u_code, u_code));
   EXPECT_LE(d, 1e-12 * ref);
}

// Output pipeline: running with output enabled writes a ParaView collection
// (the .pvd index and per-cycle data) into the requested prefix path. The
// high-order settings (SetHighOrderOutput + LOD >= k_u) are hard-wired in
// post/output -- this exercises the writer end to end.
TEST(Deck, ParaViewOutputIsWritten)
{
   Parameters params = Parameters::LoadYAML(INCNS_TGV_DECK);
   params.t_final = 0.04; // two steps: initial + final snapshots
   params.output.enabled = true;
   params.output.path = "deck_test_out";
   params.output.name = "tgv_ho";
   params.output.interval = 1;

   RunCase(params);

   MPI_Barrier(MPI_COMM_WORLD); // all ranks finish writing before rank 0 looks
   if (Mpi::Root())
   {
      // ParaViewDataCollection lays out <path>/<name>/<name>.pvd.
      const std::string pvd = "deck_test_out/tgv_ho/tgv_ho.pvd";
      std::ifstream f(pvd);
      EXPECT_TRUE(f.good()) << pvd << " was not written";
      std::string content((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
      EXPECT_NE(content.find("Cycle000002"), std::string::npos)
            << "expected the final-cycle dataset in the collection index";
   }
}
