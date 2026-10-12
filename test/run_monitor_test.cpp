// Run monitoring and summaries (post/force_statistics, util/json, the time-
// based AMR schedule and Case::WriteSummary):
//  R1 force statistics on a synthetic shedding signal sampled at non-uniform
//     times: the period, mean drag, mean and rms lift come out exactly
//     (up to the trapezoid rule's O(h^2)), the peak by its parabola, and a
//     non-periodic signal reports no periods;
//  R2 the JSON writer: nesting, insertion order, escaping, non-finite -> null;
//  R3 time-based AMR (amr.every_time) on a periodic Taylor-Green box: events
//     at multiples of every_time inside [start_time, end_time] only;
//  R4 Case::WriteSummary writes valid JSON (read back with yaml-cpp) with the
//     status, time-step statistics and the mesh size;
//  R5 a restart continues the per-step history file instead of replacing it
//     (chained runs need every segment's rows);
//  R6 a restarted run with forces writes its summary before its first step
//     (run_case's "running" summary) and then continues: no force is asked of
//     the restored integrator before it has stepped.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "post/force_statistics.hpp"
#include "solver/case.hpp"
#include "util/json.hpp"
#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mfem;

TEST(RunMonitor, R1_ForceStatisticsOfASheddingSignal)
{
   // cd = 1.5 + 0.1 sin(2 w t), cl = 0.4 sqrt(2) sin(w t + 0.3): mean drag
   // 1.5, mean lift 0, rms lift 0.4, period 2 pi / w. Samples at a slowly
   // varying step (like CFL-controlled runs).
   const double w = 2.0 * M_PI / 6.6, T = 2.0 * M_PI / w;
   incns::ForceStatistics fs;
   double t = 0.0;
   int k = 0;
   while (t < 30.0 * T)
   {
      fs.Add(t, 1.5 + 0.1 * std::sin(2.0 * w * t),
             0.4 * std::sqrt(2.0) * std::sin(w * t + 0.3));
      t += 0.004 * T * (1.0 + 0.3 * std::sin(0.01 * k++));
   }
   const incns::ForceStats s = fs.Compute(10);
   ASSERT_TRUE(s.periodic);
   EXPECT_EQ(s.periods, 10);
   EXPECT_NEAR(s.period, T, 1e-6 * T);
   EXPECT_NEAR(s.cd_mean, 1.5, 1e-5);
   EXPECT_NEAR(s.cl_mean, 0.0, 1e-5);
   EXPECT_NEAR(s.cl_rms, 0.4, 1e-5);
   EXPECT_LT(s.period_spread, 1e-5);
   EXPECT_NEAR(s.cd_max, 1.6, 1e-6);
   // The refined peak sits at a maximum of sin(2 w t): 2 w t = pi/2 mod 2 pi.
   const double phase = std::fmod(2.0 * w * s.t_cd_max, 2.0 * M_PI);
   EXPECT_NEAR(phase, 0.5 * M_PI, 1e-3);

   incns::ForceStatistics flat;
   for (int i = 0; i < 100; ++i) { flat.Add(0.1 * i, 2.0 + 0.01 * i, 0.0); }
   const incns::ForceStats f = flat.Compute(10);
   EXPECT_FALSE(f.periodic);
   EXPECT_DOUBLE_EQ(f.cd_final, 2.99);
   EXPECT_DOUBLE_EQ(f.cd_max, 2.99);
}

TEST(RunMonitor, R2_JsonWriter)
{
   incns::Json j;
   j["status"].Set("ok");
   j["time"]["steps"].Set(42);
   j["time"]["dt"].Set(0.5);
   j["nan"].Set(std::nan(""));
   j["quote"].Set("a\"b\\c\nd");
   j["flag"].Set(true);
   const std::string expected =
      "{\n"
      "  \"status\": \"ok\",\n"
      "  \"time\": {\n"
      "    \"steps\": 42,\n"
      "    \"dt\": 0.5\n"
      "  },\n"
      "  \"nan\": null,\n"
      "  \"quote\": \"a\\\"b\\\\c\\nd\",\n"
      "  \"flag\": true\n"
      "}\n";
   EXPECT_EQ(j.Dump(), expected);
   const YAML::Node back = YAML::Load(j.Dump()); // JSON is YAML
   EXPECT_EQ(back["time"]["steps"].as<int>(), 42);
   EXPECT_EQ(back["quote"].as<std::string>(), "a\"b\\c\nd");
}

namespace
{
incns::Parameters TgvBox()
{
   incns::Parameters p = incns::Parameters::LoadYAMLString(
                            "equation: navier_stokes\n"
                            "physics:\n  nu: 0.05\n"
                            "mesh:\n  dim: 2\n  elements: [4, 4]\n"
                            "  lengths: [6.283185307179586, 6.283185307179586]\n"
                            "time:\n  dt: 0.02\n  t_final: 0.4\n  step_control: fixed\n"
                            "initial_velocity: taylor_green_2d\n");
   return p;
}
} // namespace

TEST(RunMonitor, R3_TimeBasedAmrEvents)
{
   if (amr_test::DebugDeviceSkipsPeriodicNcSolves())
   {
      GTEST_SKIP() << "periodic NC solve: MFEM debug-device alias false "
                   "positive (see amr_test_util.hpp / CLAUDE.md)";
   }
   incns::Parameters p = TgvBox();
   p.amr.enabled = true;
   p.amr.every_time = 0.1;
   p.amr.start_time = 0.15;
   p.amr.end_time = 0.35;
   p.amr.threshold_mode = incns::AmrThreshold::Relative;
   p.amr.theta = 0.9;
   p.amr.max_elements = 64;
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   auto u0 = incns::MakeInitialVelocity(p);
   flow.SetInitialVelocity(*u0);
   flow.Run();
   // Multiples of 0.1 in [0.15, 0.35]: t = 0.2 and 0.3 (one event each).
   const auto& log = flow.AdaptHistory();
   ASSERT_EQ(log.size(), 2u);
   EXPECT_NEAR(log[0].time, 0.2, 1e-9);
   EXPECT_NEAR(log[1].time, 0.3, 1e-9);
}

TEST(RunMonitor, R4_SummaryIsValidJson)
{
   incns::Parameters p = TgvBox();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   auto u0 = incns::MakeInitialVelocity(p);
   flow.SetInitialVelocity(*u0);
   flow.Run();
   const std::string path =
      "run_monitor_summary_np" + std::to_string(Mpi::WorldSize()) + ".json";
   flow.WriteSummary(path, "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   const YAML::Node s = YAML::LoadFile(path);
   EXPECT_EQ(s["status"].as<std::string>(), "ok");
   EXPECT_EQ(s["case"]["equation"].as<std::string>(), "navier_stokes");
   EXPECT_EQ(s["time"]["steps"].as<int>(), 20);
   EXPECT_NEAR(s["time"]["t"].as<double>(), 0.4, 1e-12);
   EXPECT_NEAR(s["time"]["dt_max"].as<double>(), 0.02, 1e-12);
   EXPECT_EQ(s["mesh"]["elements"].as<int>(), 16);
   EXPECT_GT(s["diagnostics"]["kinetic_energy"].as<double>(), 0.0);
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::remove(path.c_str()); }
}

TEST(RunMonitor, R5_RestartAppendsTheHistory)
{
   const std::string dir = "run_monitor_r5_np" + std::to_string(Mpi::WorldSize());
   auto deck = [&](double t_final, const std::string & extra)
   {
      incns::Parameters p = incns::Parameters::LoadYAMLString(
                               "equation: navier_stokes\n"
                               "physics:\n  nu: 0.05\n"
                               "mesh:\n  dim: 2\n  elements: [4, 4]\n"
                               "  lengths: [6.283185307179586, 6.283185307179586]\n"
                               "time:\n  dt: 0.02\n  t_final: " + std::to_string(t_final) +
                               "\n  step_control: fixed\n"
                               "initial_velocity: taylor_green_2d\n"
                               "output:\n  path: " + dir + "\n  name: r5\n  history: true\n"
                               "checkpoint:\n  enabled: true\n  interval: 2\n  path: " + dir +
                               "/chk\n" + extra);
      auto mesh = incns::MakeCaseMesh(p);
      incns::Case flow(*mesh, p);
      auto u0 = incns::MakeInitialVelocity(p);
      flow.SetInitialVelocity(*u0);
      flow.Run();
   };
   deck(0.06,
        "");                           // 3 steps; the checkpoint is at step 2
   deck(0.08, "restart: " + dir + "/chk\n"); // steps 3 and 4 from it
   MPI_Barrier(MPI_COMM_WORLD);
   std::ifstream f(dir + "/r5_history.csv");
   ASSERT_TRUE(f.good());
   std::string line;
   int headers = 0;
   std::vector<double> t;
   while (std::getline(f, line))
   {
      if (line.rfind("t,", 0) == 0) { ++headers; continue; }
      t.push_back(std::stod(line.substr(0, line.find(','))));
   }
   EXPECT_EQ(headers, 1);
   // Steps 1-3 of the first run, then 3-4 of the restarted one (step 3 twice).
   ASSERT_EQ(t.size(), 5u);
   EXPECT_NEAR(t.front(), 0.02, 1e-12);
   EXPECT_NEAR(t[2], 0.06, 1e-12);
   EXPECT_NEAR(t[3], 0.06, 1e-12);
   EXPECT_NEAR(t.back(), 0.08, 1e-12);
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(dir); }
}

TEST(RunMonitor, R6_RestartWithForcesWritesItsSummary)
{
   const std::string dir = "run_monitor_r6_np" + std::to_string(Mpi::WorldSize());
   auto make = [&](double t_final, const std::string & extra)
   {
      return incns::Parameters::LoadYAMLString(
                "equation: navier_stokes\n"
                "physics:\n  Re: 200\n"
                "discretization:\n  order_u: 2\n  order_p: 1\n"
                "mesh:\n  dim: 2\n  geometry: square_cylinder\n"
                "initial_velocity: uniform\n"
                "time:\n  dt: 0.01\n  t_final: " + std::to_string(t_final) +
                "\n  step_control: fixed\n"
                "boundary_conditions:\n"
                "  - {select: [inflow, sides], type: velocity, value: [1, 0]}\n"
                "  - {select: [outflow], type: outflow}\n"
                "  - {select: [body], type: no_slip}\n"
                "forces:\n  enabled: true\n  boundaries: [body]\n  statistics: true\n"
                "output:\n  path: " + dir + "\n  name: r6\n  history: true\n"
                "checkpoint:\n  enabled: true\n  interval: 2\n  path: " + dir + "/chk\n" +
                extra);
   };
   auto run = [&](const incns::Parameters & p, bool summary_first)
   {
      auto mesh = incns::MakeCaseMesh(p);
      incns::Case flow(*mesh, p);
      auto u0 = incns::MakeInitialVelocity(p);
      flow.SetInitialVelocity(*u0);
      if (summary_first) { flow.WriteSummary(dir + "/r6_summary.json", "running"); }
      flow.Run();
      flow.WriteSummary(dir + "/r6_summary.json", "ok");
   };
   run(make(0.03, ""), true);                           // checkpoint at step 2
   run(make(0.05, "restart: " + dir + "/chk\n"),
       true); // the "running" summary first
   MPI_Barrier(MPI_COMM_WORLD);
   const YAML::Node s = YAML::LoadFile(dir + "/r6_summary.json");
   EXPECT_EQ(s["status"].as<std::string>(), "ok");
   EXPECT_NEAR(s["time"]["t"].as<double>(), 0.05, 1e-12);
   EXPECT_TRUE(std::isfinite(s["forces"]["cd"].as<double>()));
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(dir); }
}
