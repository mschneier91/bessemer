// Unit test for src/util/profiler with profiling ENABLED (this TU is compiled
// with INCNS_ENABLE_PROFILING). Uses an injected fake clock so nesting,
// inclusive/exclusive accounting, and the MPI max/min reduction are exact rather
// than wall-clock-flaky. A companion binary (profiler_off_test) covers the
// disabled toggle.

#include <gtest/gtest.h>

#include "util/profiler.hpp"
#include "mfem.hpp"

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
