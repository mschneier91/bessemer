// Shared test entry point: initialise MPI + MFEM once, then run googletest.
// Linked into every module test so each ctest binary is MPI-aware. Non-root
// ranks stay silent (pass/fail still propagates via the process exit code, which
// mpirun aggregates), so parallel runs don't interleave duplicate output.

#include <gtest/gtest.h>

#include "mfem.hpp"

int main(int argc, char** argv)
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();

   ::testing::InitGoogleTest(&argc, argv);

   if (!mfem::Mpi::Root())
   {
      auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_result_printer());
   }

   return RUN_ALL_TESTS();
}
