// precond/hypre_device: BoomerAMG stays on hypre's fast GPU path.
//  D1 MFEM's default options pass the check on any build.
//  D2 the symmetric smoother is 8 on a host and 18 on a GPU, and passes; on a
//     GPU type 8 (triangular solves) and an unsupported coarsening are
//     reported, on a host nothing ever is.

#include <gtest/gtest.h>

#include "precond/hypre_device.hpp"
#include "mfem.hpp"

#include <string>

using incns::HypreAmgDeviceProblems;

TEST(HypreDevice, D1_MfemDefaultsPass)
{
   mfem::HypreBoomerAMG amg;
   EXPECT_EQ(HypreAmgDeviceProblems(amg), "");
}

TEST(HypreDevice, D2_SymmetricSmootherAndWhatIsReported)
{
   const bool gpu = mfem::HypreUsingGPU();
   EXPECT_EQ(incns::SymmetricAmgSmoother(), gpu ? 18 : 8);

   mfem::HypreBoomerAMG amg;
   HYPRE_Solver h = static_cast<HYPRE_Solver>(amg);
   HYPRE_BoomerAMGSetRelaxType(h, incns::SymmetricAmgSmoother());
   EXPECT_EQ(HypreAmgDeviceProblems(amg), "");

   HYPRE_BoomerAMGSetRelaxType(h, 8); // l1 symmetric Gauss-Seidel
   const std::string gs = HypreAmgDeviceProblems(amg);
   if (gpu)
   {
      EXPECT_NE(gs.find("smoother type 8"), std::string::npos) << gs;
      EXPECT_NE(gs.find("triangular"), std::string::npos) << gs;
   }
   else
   {
      EXPECT_EQ(gs, "");
   }

   HYPRE_BoomerAMGSetRelaxType(h, 18);
   HYPRE_BoomerAMGSetCoarsenType(h, 10); // HMIS: the host default
   const std::string hmis = HypreAmgDeviceProblems(amg);
   if (gpu) { EXPECT_NE(hmis.find("coarsening type 10"), std::string::npos) << hmis; }
   else { EXPECT_EQ(hmis, ""); }
}
