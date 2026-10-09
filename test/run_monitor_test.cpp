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
//     status, time-step statistics and the mesh size.

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
