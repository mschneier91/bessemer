// Sprint 1.11 green criterion -- the YAML TGV deck reproduces the in-code
// driver's results within tolerance THROUGH THE SAME LIBRARY SURFACE
// (Case), and the written ParaView output exists with the high-order
// settings. Also pins the deck loader field by field against the committed
// cases/tgv2d_stokes.yaml.

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/case.hpp"
#include "repro_tolerance.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace mfem;
using incns::MakeBoxMesh;
using incns::MakeInitialVelocity;
using incns::Parameters;
using incns::Case;

namespace
{
// March a TGV case and return the final velocity true dofs.
Vector RunCase(const Parameters& params)
{
   Mesh serial = MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   Case flow_case(mesh, params);
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
// is pure roundoff). NOTE: on a real GPU backend two identical runs are not
// bitwise reproducible, so this doubles as a determinism probe -- ReproTol().
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
   EXPECT_LE(d, incns_test::ReproTol(1e-12) * ref);
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

// The rotational-form keys: `physics.convective_form` and `solver.rotation_pc`
// parse, unstated ones keep the defaults (Convective / Symmetric), and the
// Case actually hands them to the integrator -- the rotational TGV march
// differs from the convective one at discretization level (same physics,
// different scheme) while agreeing with it to O(h, dt).
TEST(Deck, RotationalFormKeys)
{
   const Parameters defaults = Parameters::LoadYAML(INCNS_TGV_DECK);
   EXPECT_EQ(defaults.convective_form, incns::ConvectiveForm::Convective);
   EXPECT_EQ(defaults.rotation_pc, incns::RotationVelocityPC::Symmetric);

   // One file per rank: every rank parses, none races another's write.
   const std::string path =
      "deck_rotational_rank" + std::to_string(Mpi::WorldRank()) + ".yaml";
   {
      std::ofstream f(path);
      f << "equation: navier_stokes\n"
        << "physics:\n  nu: 1.0\n  convective_form: rotational\n"
        << "solver:\n  rotation_pc: pbj_krylov\n  rotation_lor: true\n";
   }
   const Parameters p = Parameters::LoadYAML(path);
   std::remove(path.c_str());
   EXPECT_EQ(p.equation, incns::Equation::NavierStokes);
   EXPECT_EQ(p.convective_form, incns::ConvectiveForm::Rotational);
   EXPECT_EQ(p.rotation_pc, incns::RotationVelocityPC::PbjKrylov);
   EXPECT_TRUE(p.rotation_in_lor);
   EXPECT_FALSE(defaults.rotation_in_lor);

   // End to end through Case on the TGV deck, NSE, a few steps.
   Parameters conv = defaults;
   conv.equation = incns::Equation::NavierStokes;
   conv.t_final = 3 * conv.dt;
   conv.output.enabled = false;
   Parameters rot = conv;
   rot.convective_form = incns::ConvectiveForm::Rotational;
   rot.rotation_pc = incns::RotationVelocityPC::PbjOnly;
   const Vector uc = RunCase(conv), ur = RunCase(rot);
   Vector d(ur);
   d -= uc;
   const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, d, d) /
                                InnerProduct(MPI_COMM_WORLD, uc, uc));
   EXPECT_GT(rel, 1e-12) << "rotational run identical to convective: the "
                         << "Case did not pass the convective form through";
   EXPECT_LT(rel, 1e-2) << "rotational and convective TGV disagree beyond "
                        << "discretization error";
}

// The `amr:` section and `time.cfl_max` parse; unstated keys keep their
// defaults; min_size is normalized by L_ref in dimensional mode.
TEST(Deck, AmrKeys)
{
   const Parameters defaults = Parameters::LoadYAML(INCNS_TGV_DECK);
   EXPECT_FALSE(defaults.amr.enabled);
   EXPECT_EQ(defaults.cfl_max, 0.0);

   const std::string path =
      "deck_amr_rank" + std::to_string(Mpi::WorldRank()) + ".yaml";
   {
      std::ofstream f(path);
      f << "nondimensionalization:\n  mode: dimensional\n  L_ref: 2.0\n"
        << "  U_ref: 1.0\n"
        << "physics:\n  nu: 0.01\n"
        << "time:\n  dt: 0.01\n  t_final: 1.0\n  cfl_max: 0.8\n"
        << "amr:\n  enabled: true\n  interval: 20\n  initial_passes: 2\n"
        << "  passes_per_event: 3\n  anisotropic: true\n  aniso_ratio: 0.4\n"
        << "  threshold_mode: absolute\n  tolerance: 0.02\n  min_size: 0.1\n"
        << "  max_elements: 5000\n  nc_limit: 2\n  rebalance: false\n"
        << "  project_history: false\n  write_indicator: true\n";
   }
   const Parameters p = Parameters::LoadYAML(path);
   std::remove(path.c_str());
   EXPECT_TRUE(p.amr.enabled);
   EXPECT_EQ(p.amr.interval, 20);
   EXPECT_EQ(p.amr.initial_passes, 2);
   EXPECT_EQ(p.amr.passes_per_event, 3);
   EXPECT_TRUE(p.amr.anisotropic);
   EXPECT_DOUBLE_EQ(p.amr.aniso_ratio, 0.4);
   EXPECT_EQ(p.amr.threshold_mode, incns::AmrThreshold::Absolute);
   EXPECT_DOUBLE_EQ(p.amr.tolerance, 0.02);
   EXPECT_DOUBLE_EQ(p.amr.min_size, 0.05); // 0.1 / L_ref
   EXPECT_EQ(p.amr.max_elements, 5000);
   EXPECT_EQ(p.amr.nc_limit, 2);
   EXPECT_FALSE(p.amr.rebalance);
   EXPECT_FALSE(p.amr.project_history);
   EXPECT_TRUE(p.amr.write_indicator);
   EXPECT_DOUBLE_EQ(p.cfl_max, 0.8);
}

TEST(Deck, ForcesKeys)
{
   const Parameters defaults = Parameters::LoadYAML(INCNS_TGV_DECK);
   EXPECT_FALSE(defaults.forces.enabled);

   const std::string path =
      "deck_forces_rank" + std::to_string(Mpi::WorldRank()) + ".yaml";
   {
      std::ofstream f(path);
      f << "nondimensionalization:\n  mode: dimensional\n  L_ref: 2.0\n"
        << "  U_ref: 4.0\n"
        << "physics:\n  nu: 0.01\n"
        << "mesh:\n  dim: 3\n  elements: [2, 2, 2]\n  lengths: [1, 1, 1]\n"
        << "time:\n  dt: 0.01\n  t_final: 1.0\n"
        << "forces:\n  enabled: true\n  attributes: [4, 7]\n"
        << "  reference_velocity: 2.0\n  reference_area: 0.5\n  interval: 0\n";
   }
   const Parameters p = Parameters::LoadYAML(path);
   std::remove(path.c_str());
   EXPECT_TRUE(p.forces.enabled);
   EXPECT_EQ(p.forces.attributes, (std::vector<int> {4, 7}));
   EXPECT_DOUBLE_EQ(p.forces.reference_velocity, 0.5); // 2.0 / U_ref
   EXPECT_DOUBLE_EQ(p.forces.reference_area, 0.125);   // 0.5 / L_ref^2 (3D)
   EXPECT_EQ(p.forces.interval, 0);
}
