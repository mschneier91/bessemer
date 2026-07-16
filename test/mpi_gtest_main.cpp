// Shared test entry point: initialise MPI + MFEM once, then run googletest.
// Linked into every module test so each ctest binary is MPI-aware. Non-root
// ranks stay silent (pass/fail still propagates via the process exit code, which
// mpirun aggregates), so parallel runs don't interleave duplicate output.

#include <gtest/gtest.h>

#include "util/device.hpp"
#include "mfem.hpp"

int main(int argc, char** argv)
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();

   // Configure the MFEM device once, before any test allocates a vector. The
   // default is "cpu"; INCNS_DEVICE overrides it (see ConfigureDevice), so
   // `INCNS_DEVICE=debug` runs the whole suite on the mprotect-guarded debug
   // device that faults on silent host access of device memory.
   incns::ConfigureDevice("cpu");

   ::testing::InitGoogleTest(&argc, argv);

   if (!mfem::Mpi::Root())
   {
      auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_result_printer());
   }

   return RUN_ALL_TESTS();
}
