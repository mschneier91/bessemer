// The profile a run leaves behind (deck output.profile; apps/run_case writes it
// next to the summary).
//  W1 Case::WriteProfile after a short run: the file has the profiler's region
//     tree -- case::run once, time_integrator::step once per step and inside
//     case::run, every region's path ending in its own name, depths that nest,
//     min <= max, exclusive <= inclusive, and children that fit in their parent.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "util/profiler.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

struct Row
{
   int depth = 0;
   std::string region, path;
   long long calls = 0;
   double incl_max = 0.0, incl_min = 0.0, excl_max = 0.0, excl_min = 0.0;
};

std::vector<std::string> Split(const std::string& line)
{
   std::vector<std::string> out;
   std::istringstream is(line);
   for (std::string f; std::getline(is, f, ',');) { out.push_back(f); }
   return out;
}

} // namespace

TEST(ProfileOutput, W1_RunWritesTheRegionTree)
{
   incns::Profiler::Instance().Reset();
   Parameters p;
   p.equation = incns::Equation::Stokes;
   p.nu = 0.1;
   p.order_u = 2;
   p.order_p = 1;
   p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
   p.dt = 0.05;
   p.t_final = 0.15;
   p.step_control = incns::StepControl::Fixed;
   p.Normalize();
   EXPECT_EQ(p.output.profile, incns::ProfileOutput::Wall); // the default

   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   flow.Run();
   const int steps = flow.Integrator().StepCount();
   ASSERT_GT(steps, 0);

   // Unique per rank count: ctest runs the np 1/2/4 instances concurrently.
   const std::string path = "profile_output_np" +
                            std::to_string(Mpi::WorldSize()) + ".csv";
   flow.WriteProfile(path); // collective; rank 0 writes
   incns::Profiler::Instance().Reset();
   if (!Mpi::Root()) { return; }

   std::ifstream f(path);
   ASSERT_TRUE(f.good()) << path;
   std::string line;
   ASSERT_TRUE(std::getline(f, line));
   EXPECT_EQ(line.rfind("# bessemer profile", 0), 0u);
   ASSERT_TRUE(std::getline(f, line));
   EXPECT_EQ(line.rfind("# mode: wall", 0), 0u);
   ASSERT_TRUE(std::getline(f, line));
   EXPECT_EQ(line.rfind("depth,region,path,calls,", 0), 0u);

   std::vector<Row> rows;
   while (std::getline(f, line))
   {
      const std::vector<std::string> c = Split(line);
      ASSERT_EQ(c.size(), 9u) << line;
      Row r;
      r.depth = std::stoi(c[0]);
      r.region = c[1];
      r.path = c[2];
      r.calls = std::stoll(c[3]);
      r.incl_max = std::stod(c[4]);
      r.incl_min = std::stod(c[5]);
      r.excl_max = std::stod(c[6]);
      r.excl_min = std::stod(c[7]);
      rows.push_back(r);
   }
   f.close();
   std::remove(path.c_str());
   ASSERT_FALSE(rows.empty());

   int run_rows = 0, step_rows = 0;
   for (std::size_t i = 0; i < rows.size(); ++i)
   {
      const Row& r = rows[i];
      SCOPED_TRACE(r.path);
      // The path ends in the region's own name, and depths nest.
      ASSERT_GE(r.path.size(), r.region.size());
      EXPECT_EQ(r.path.substr(r.path.size() - r.region.size()), r.region);
      EXPECT_LE(r.depth, (i == 0) ? 0 : rows[i - 1].depth + 1);
      EXPECT_GT(r.calls, 0);
      EXPECT_GE(r.incl_min, 0.0);
      EXPECT_LE(r.incl_min, r.incl_max * (1.0 + 1e-12) + 1e-12);
      EXPECT_LE(r.excl_max, r.incl_max * (1.0 + 1e-12) + 1e-12);
      // The direct children fit in their parent (each rank's children do, so
      // the sum of their minima is bounded by the parent's maximum).
      double children = 0.0;
      for (std::size_t j = i + 1; j < rows.size() && rows[j].depth > r.depth; ++j)
      {
         if (rows[j].depth == r.depth + 1) { children += rows[j].incl_min; }
      }
      EXPECT_LE(children, r.incl_max * (1.0 + 1e-9) + 1e-9);

      if (r.region == "case::run")
      {
         ++run_rows;
         EXPECT_EQ(r.calls, 1);
      }
      if (r.region == "time_integrator::step")
      {
         ++step_rows;
         EXPECT_EQ(r.calls, steps);
         EXPECT_NE(r.path.find("case::run/"), std::string::npos);
      }
   }
   EXPECT_EQ(run_rows, 1);
   EXPECT_EQ(step_rows, 1);
}
