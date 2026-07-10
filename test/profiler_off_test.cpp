// Companion to profiler_test: this TU is compiled WITHOUT INCNS_ENABLE_PROFILING.
// It proves the INCNS_PROFILE macro (a) still compiles and (b) is a complete
// no-op -- no region is recorded -- when profiling is toggled off.

#include <gtest/gtest.h>

#include "util/profiler.hpp"

using incns::Profiler;

TEST(ProfilerOff, MacroCompilesAndRecordsNothing)
{
   Profiler& p = Profiler::Instance();
   p.Reset();

   {
      INCNS_PROFILE("should_not_be_recorded");
      volatile int work = 41;
      work += 1;
      (void)work;
   }

   const auto rep = p.Report();
   EXPECT_TRUE(rep.empty());
   p.Reset();
}
