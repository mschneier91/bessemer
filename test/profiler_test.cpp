// Unit test for src/util/profiler with profiling ENABLED (this TU is compiled
// with INCNS_ENABLE_PROFILING). Uses an injected fake clock so nesting,
// inclusive/exclusive accounting, and the MPI max/min reduction are exact rather
// than wall-clock-flaky. A companion binary (profiler_off_test) covers the
// disabled toggle.

#include <gtest/gtest.h>

#include "util/profiler.hpp"
#include "mfem.hpp"

#include <sstream>
#include <string>
#include <vector>

using incns::Profiler;

TEST(Profiler, NestingAndInclusiveExclusive)
{
   Profiler& p = Profiler::Instance();
   p.Reset();
   double now = 0.0;
   p.SetClock([&] { return now; });

   {
      INCNS_PROFILE("outer");       // enter outer at t=0
      now = 1.0;
      {
         INCNS_PROFILE("inner");    // enter inner at t=1
         now = 3.0;
      }                             // inner inclusive = 2
      now = 4.0;
   }                                // outer inclusive = 4

   const auto rep = p.Report();
   ASSERT_EQ(rep.size(), 2u);
   EXPECT_EQ(rep[0].name, "outer");
   EXPECT_EQ(rep[0].depth, 0);
   EXPECT_EQ(rep[1].name, "inner");
   EXPECT_EQ(rep[1].depth, 1);
   EXPECT_NEAR(rep[0].incl_max, 4.0, 1e-12);
   EXPECT_NEAR(rep[1].incl_max, 2.0, 1e-12);
   EXPECT_NEAR(rep[0].excl_max, 2.0, 1e-12); // 4 - 2 (inner)
   EXPECT_NEAR(rep[1].excl_max, 2.0, 1e-12);
   EXPECT_EQ(rep[0].calls, 1);
   p.Reset();
}

TEST(Profiler, RepeatedEntryAccumulates)
{
   Profiler& p = Profiler::Instance();
   p.Reset();
   double now = 0.0;
   p.SetClock([&] { return now; });

   for (int i = 0; i < 3; ++i)
   {
      const double base = now;
      INCNS_PROFILE("loopbody");
      now = base + 2.0;
   }

   const auto rep = p.Report();
   ASSERT_EQ(rep.size(), 1u);
   EXPECT_EQ(rep[0].name, "loopbody");
   EXPECT_EQ(rep[0].calls, 3);
   EXPECT_NEAR(rep[0].incl_max, 6.0, 1e-12);
   p.Reset();
}

// Each rank makes "work" take (rank+1) seconds; the reduction must recover
// max = nranks and min = 1 on every rank.
TEST(Profiler, MpiMaxMinReduction)
{
   Profiler& p = Profiler::Instance();
   p.Reset();
   const int rank = mfem::Mpi::WorldRank();
   const int nranks = mfem::Mpi::WorldSize();
   double now = 0.0;
   p.SetClock([&] { return now; });

   {
      INCNS_PROFILE("work");
      now = static_cast<double>(rank + 1);
   }

   const auto rep = p.Report();
   ASSERT_EQ(rep.size(), 1u);
   EXPECT_EQ(rep[0].name, "work");
   EXPECT_NEAR(rep[0].incl_max, static_cast<double>(nranks), 1e-12);
   EXPECT_NEAR(rep[0].incl_min, 1.0, 1e-12);
   p.Reset();
}

TEST(Profiler, SyncHookRunsAtEveryBoundary)
{
   Profiler& p = Profiler::Instance();
   p.Reset();
   EXPECT_FALSE(p.Synchronized());
   double now = 0.0;
   p.SetClock([&] { return now; });
   int syncs = 0;
   p.SetSync([&] { ++syncs; });
   EXPECT_TRUE(p.Synchronized());
   {
      INCNS_PROFILE("outer");
      EXPECT_EQ(syncs, 1); // entering outer
      {
         INCNS_PROFILE("inner");
         EXPECT_EQ(syncs, 2);
      }
      EXPECT_EQ(syncs, 3); // leaving inner
   }
   EXPECT_EQ(syncs, 4);
   p.Reset(); // removes the hook
   EXPECT_FALSE(p.Synchronized());
   {
      INCNS_PROFILE("after_reset");
   }
   EXPECT_EQ(syncs, 4);
   p.Reset();
}

TEST(Profiler, CsvHasOneRowPerRegionWithItsPath)
{
   Profiler& p = Profiler::Instance();
   p.Reset();
   double now = 0.0;
   p.SetClock([&] { return now; });
   {
      INCNS_PROFILE("outer");
      now = 1.0;
      for (int i = 0; i < 2; ++i)
      {
         INCNS_PROFILE("inner");
         now += 1.0;
      }
      now = 4.0;
   }
   {
      INCNS_PROFILE("second");
      now = 5.0;
   }
   // Both writes are collective, so every rank makes both before rank 0 goes
   // on alone to read them. Only rank 0 writes.
   std::ostringstream os, os2;
   p.WriteCsv(os);
   p.SetSync([] {}); // with a hook installed the file says so
   p.WriteCsv(os2);
   p.Reset();
   if (!mfem::Mpi::Root())
   {
      EXPECT_TRUE(os.str().empty());
      EXPECT_TRUE(os2.str().empty());
      return;
   }
   std::vector<std::string> lines;
   std::istringstream is(os.str());
   for (std::string l; std::getline(is, l);) { lines.push_back(l); }
   ASSERT_EQ(lines.size(), 6u); // two comments, the header, three regions
   EXPECT_EQ(lines[0].rfind("# bessemer profile", 0), 0u);
   EXPECT_EQ(lines[1].rfind("# mode: wall", 0), 0u);
   EXPECT_EQ(lines[2], "depth,region,path,calls,incl_max,incl_min,excl_max,"
             "excl_min,pct_of_parent");
   EXPECT_EQ(lines[3], "0,outer,outer,1,4,4,2,2,80");
   EXPECT_EQ(lines[4], "1,inner,outer/inner,2,2,2,2,2,50");
   EXPECT_EQ(lines[5], "0,second,second,1,1,1,1,1,20");
   EXPECT_NE(os2.str().find("# mode: sync"), std::string::npos);
}
