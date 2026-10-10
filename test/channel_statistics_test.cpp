// Channel statistics (post/channel_statistics) and their place in a run:
//  S1 plane averages of a field the Q3 space represents exactly, on a
//     stretched 3D box: <U>, <V>, <W> and the Reynolds stresses at every
//     station match the analytic plane averages to round-off, and the
//     stations are the node heights (layers x k + 1);
//  S2 the wall shear stress (both walls, sign toward the interior), u_tau,
//     Re_tau and the bulk velocity of the same field, exactly;
//  S3 time averages: the trapezoid rule over unevenly spaced samples (exact
//     for a field linear in t), samples before start_time ignored;
//  S4 Save()/Load(): averaging, saving, loading into a fresh object and
//     continuing gives exactly the uninterrupted averages;
//  S5 in a run (2D forced channel from the channel initial condition):
//     interrupted by a checkpoint and restarted, the averages equal the
//     uninterrupted run's (the state travels with the checkpoint), and the
//     summary and profile CSV carry them.

#include <gtest/gtest.h>

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "post/channel_statistics.hpp"
#include "repro_tolerance.hpp"
#include "solver/case.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mfem;

namespace
{
// A stretched, walled 3D box: x in [0, 2], y in [0, 2] (walls), z in [0, 1];
// 2 x 3 x 2 elements, tanh clustering in y.
std::unique_ptr<ParMesh> Box3D()
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "mesh:\n  dim: 3\n  elements: [2, 3, 2]\n"
                                  "  lengths: [2.0, 2.0, 1.0]\n"
                                  "  periodic: [false, false, false]\n"
                                  "  stretch: [none, tanh, none]\n");
   return incns::MakeCaseMesh(p);
}

// Cubic profiles in y plus zero-mean linear parts in x and z: exactly in Q3.
double P(double y) { return 1.0 + y - 0.25 * y * y * y; }
double Q(double y) { return 0.2 * y * y - 0.1 * y; }
double R(double y) { return 0.5 - 0.3 * y; }
const double kAlpha = 0.4, kBeta = 0.7, kLx = 2.0, kLz = 1.0;

void Fill(ParGridFunction& u, double scale)
{
   VectorFunctionCoefficient c(3, [scale](const Vector & x, Vector & v)
   {
      v(0) = scale * (P(x(1)) + kAlpha * (x(0) - 1.0));
      v(1) = scale * (Q(x(1)) + kBeta * (x(2) - 0.5));
      v(2) = scale * R(x(1));
   });
   u.ProjectCoefficient(c);
}
} // namespace

TEST(ChannelStatistics, S1_PlaneAveragesAreExact)
{
   auto mesh = Box3D();
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParGridFunction u(&spaces.Velocity());
   Fill(u, 1.0);
   incns::ChannelStatistics cs(spaces.Velocity(), 0.01, 0.0);
   ASSERT_EQ(cs.Stations().size(),
             3u * 3u + 1u); // 3 layers, Q3, shared lines once
   cs.Sample(u, 0.0);
   cs.Sample(u, 1.0);
   EXPECT_DOUBLE_EQ(cs.AveragingTime(), 1.0);
   const incns::ChannelProfiles p = cs.Averages();
   const double uu = kAlpha * kAlpha * kLx * kLx / 12.0;
   const double vv = kBeta * kBeta * kLz * kLz / 12.0;
   for (std::size_t s = 0; s < p.y.size(); ++s)
   {
      const double y = p.y[s];
      EXPECT_NEAR(p.U[s], P(y), 1e-12) << "y = " << y;
      EXPECT_NEAR(p.V[s], Q(y), 1e-12) << "y = " << y;
      EXPECT_NEAR(p.W[s], R(y), 1e-12) << "y = " << y;
      EXPECT_NEAR(p.uu[s], uu, 1e-12) << "y = " << y;
      EXPECT_NEAR(p.vv[s], vv, 1e-12) << "y = " << y;
      EXPECT_NEAR(p.ww[s], 0.0, 1e-12) << "y = " << y;
      EXPECT_NEAR(p.uv[s], 0.0, 1e-12) << "y = " << y;
   }
   EXPECT_NEAR(p.y.front(), 0.0, 1e-14);
   EXPECT_NEAR(p.y.back(), 2.0, 1e-14);
}

TEST(ChannelStatistics, S2_WallShearUTauReTauBulk)
{
   auto mesh = Box3D();
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParGridFunction u(&spaces.Velocity());
   Fill(u, 1.0);
   const double nu = 0.01;
   incns::ChannelStatistics cs(spaces.Velocity(), nu, 0.0);
   cs.Sample(u, 0.0);
   // dP/dy = 1 at y = 0 and 1 - 0.75 * 4 = -2 at y = 2: tau_w = nu (1 + 2) / 2.
   const double tau = 1.5 * nu;
   EXPECT_NEAR(cs.TauWall(), tau, 1e-14);
   EXPECT_NEAR(cs.TauWallNow(), tau, 1e-14);
   EXPECT_NEAR(cs.UTau(), std::sqrt(tau), 1e-12);
   EXPECT_NEAR(cs.ReTau(), std::sqrt(tau) * 1.0 / nu, 1e-9);
   EXPECT_DOUBLE_EQ(cs.Delta(), 1.0);
   // (1/2) int_0^2 P dy = (2 + 2 - 1) / 2.
   EXPECT_NEAR(cs.BulkVelocity(), 1.5, 1e-13);
}

TEST(ChannelStatistics, S3_TrapezoidTimeAverages)
{
   auto mesh = Box3D();
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParGridFunction u(&spaces.Velocity());
   incns::ChannelStatistics cs(spaces.Velocity(), 0.01, 0.1);
   // a(t) = 1 + t at uneven times; t = 0 is before start_time.
   for (double t : {0.0, 0.1, 0.35, 0.4, 1.0})
   {
      Fill(u, 1.0 + t);
      cs.Sample(u, t);
   }
   EXPECT_EQ(cs.Samples(), 4);
   EXPECT_NEAR(cs.AveragingTime(), 0.9, 1e-15);
   // The mean of a over [0.1, 1] is 1.55 (the trapezoid rule is exact).
   const incns::ChannelProfiles p = cs.Averages();
   for (std::size_t s = 0; s < p.y.size(); ++s)
   {
      EXPECT_NEAR(p.U[s], 1.55 * P(p.y[s]), 1e-12);
      EXPECT_NEAR(p.W[s], 1.55 * R(p.y[s]), 1e-12);
   }
}

TEST(ChannelStatistics, S4_SaveLoadContinuesExactly)
{
   auto mesh = Box3D();
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParGridFunction u(&spaces.Velocity());
   const double nu = 0.01;
   const std::vector<double> times = {0.0, 0.2, 0.5, 0.6, 1.3};
   auto scale = [](double t) { return 1.0 + t * (1.0 - 0.3 * t); };

   incns::ChannelStatistics whole(spaces.Velocity(), nu, 0.0);
   for (double t : times)
   {
      Fill(u, scale(t));
      whole.Sample(u, t);
   }

   const std::string path =
      "chstats_np" + std::to_string(Mpi::WorldSize()) + ".txt";
   {
      incns::ChannelStatistics first(spaces.Velocity(), nu, 0.0);
      for (int i = 0; i < 3; ++i)
      {
         Fill(u, scale(times[i]));
         first.Sample(u, times[i]);
      }
      first.Save(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   incns::ChannelStatistics second(spaces.Velocity(), nu, 0.0);
   second.Load(path);
   for (std::size_t i = 3; i < times.size(); ++i)
   {
      Fill(u, scale(times[i]));
      second.Sample(u, times[i]);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::remove(path.c_str()); }

   EXPECT_EQ(second.Samples(), whole.Samples());
   EXPECT_EQ(second.AveragingTime(), whole.AveragingTime());
   const incns::ChannelProfiles a = whole.Averages(), b = second.Averages();
   for (std::size_t s = 0; s < a.y.size(); ++s)
   {
      EXPECT_EQ(a.U[s], b.U[s]);
      EXPECT_EQ(a.uu[s], b.uu[s]);
      EXPECT_EQ(a.uv[s], b.uv[s]);
   }
   EXPECT_EQ(whole.TauWall(), second.TauWall());
}

namespace
{
// A 2D forced channel (u_tau = 1, Re_tau = 1 / nu = 50) from the channel
// initial condition, fixed steps.
incns::Parameters ChannelDeck(const std::string& out, double t_final,
                              const std::string& extra)
{
   const std::string deck =
      "equation: navier_stokes\n"
      "physics:\n  nu: 0.02\n"
      "mesh:\n  dim: 2\n  elements: [4, 4]\n  lengths: [6.283185307179586, 2.0]\n"
      "  periodic: [true, false]\n  stretch: [none, tanh]\n"
      "time:\n  dt: 0.002\n  t_final: " + std::to_string(t_final) + "\n"
      "  step_control: fixed\n"
      "initial_velocity: channel\n"
      "initial:\n  perturbation: 0.05\n  seed: 3\n"
      "forcing:\n  body_force: [1.0, 0.0]\n"
      "boundary_conditions:\n  - {select: [ymin, ymax], type: no_slip}\n"
      "channel_statistics:\n  enabled: true\n"
      "output:\n  path: " + out + "\n  name: chan\n" + extra;
   return incns::Parameters::LoadYAMLString(deck);
}

struct ChannelRun
{
   incns::ChannelProfiles prof;
   double tau = 0.0;
   long samples = 0;
};

ChannelRun RunChannel(const incns::Parameters& p, const std::string& summary)
{
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   auto u0 = incns::MakeInitialVelocity(p);
   flow.SetInitialVelocity(*u0);
   flow.Run();
   if (!summary.empty()) { flow.WriteSummary(summary, "ok"); }
   ChannelRun r;
   r.prof = flow.ChannelStats()->Averages();
   r.tau = flow.ChannelStats()->TauWall();
   r.samples = flow.ChannelStats()->Samples();
   return r;
}
} // namespace

TEST(ChannelStatistics, S5_RestartContinuesTheAverages)
{
   const std::string base =
      "chstats_run_np" + std::to_string(Mpi::WorldSize());
   const std::string chk = base + "/chk";
   const double dt = 0.002;

   const ChannelRun ref = RunChannel(ChannelDeck(base + "/ref", 6 * dt, ""),
                                     base + "/ref_summary.json");
   // 3 steps with a checkpoint at step 3, then a restart to step 6.
   RunChannel(ChannelDeck(base + "/a", 3 * dt,
                          "checkpoint:\n  enabled: true\n  interval: 3\n  path: " +
                          chk + "\n"), "");
   const ChannelRun restarted =
      RunChannel(ChannelDeck(base + "/b", 6 * dt, "restart: " + chk + "\n"), "");

   EXPECT_EQ(ref.samples, 7); // the initial state + 6 steps
   EXPECT_EQ(restarted.samples, ref.samples);
   const double tol = incns_test::ReproTol(1e-13);
   double scale = 0.0;
   for (double v : ref.prof.U) { scale = std::max(scale, std::abs(v)); }
   for (std::size_t s = 0; s < ref.prof.y.size(); ++s)
   {
      EXPECT_NEAR(restarted.prof.U[s], ref.prof.U[s], tol * scale) << s;
      EXPECT_NEAR(restarted.prof.uu[s], ref.prof.uu[s], tol * scale * scale) << s;
      EXPECT_NEAR(restarted.prof.uv[s], ref.prof.uv[s], tol * scale * scale) << s;
   }
   EXPECT_NEAR(restarted.tau, ref.tau, tol * std::abs(ref.tau));
   // u_tau = 1 by the forcing: the start (Reichardt's profile) is close.
   EXPECT_GT(ref.tau, 0.5);
   EXPECT_LT(ref.tau, 2.0);

   MPI_Barrier(MPI_COMM_WORLD);
   const YAML::Node sum = YAML::LoadFile(base + "/ref_summary.json");
   EXPECT_EQ(sum["channel"]["samples"].as<int>(), 7);
   EXPECT_NEAR(sum["channel"]["re_tau_target"].as<double>(), 50.0, 1e-9);
   EXPECT_GT(sum["channel"]["u_bulk"].as<double>(), 0.0);
   std::ifstream csv(base + "/ref/chan_profiles.csv");
   ASSERT_TRUE(csv.good());
   int rows = 0;
   std::string line;
   while (std::getline(csv, line)) { if (!line.empty() && line[0] != '#') { ++rows; } }
   EXPECT_EQ(rows, static_cast<int>(ref.prof.y.size()) + 1); // + the header
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(base); }
}
