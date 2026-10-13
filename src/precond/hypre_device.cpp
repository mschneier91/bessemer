#include "precond/hypre_device.hpp"

#include "_hypre_parcsr_ls.h"

#include <sstream>

namespace incns
{

int SymmetricAmgSmoother()
{
   return mfem::HypreUsingGPU() ? 18 : 8;
}

std::string HypreAmgDeviceProblems(mfem::HypreBoomerAMG& amg)
{
   if (!mfem::HypreUsingGPU()) { return ""; }
   HYPRE_Solver h = static_cast<HYPRE_Solver>(amg);
   hypre_ParAMGData* data = static_cast<hypre_ParAMGData*>(static_cast<void*>(h));
   std::ostringstream out;

   HYPRE_Int coarsen = -1, interp = -1;
   hypre_BoomerAMGGetCoarsenType(data, &coarsen);
   hypre_BoomerAMGGetInterpType(data, &interp);
   if (coarsen != 8)
   {
      out << "coarsening type " << coarsen << " is not GPU-supported (PMIS, 8, "
          "is). ";
   }
   if (interp != 3 && interp != 6 && interp != 14 && interp != 15 && interp != 18)
   {
      out << "interpolation type " << interp << " is not GPU-supported (3, 6, "
          "14, 15 and 18 are). ";
   }

   bool jacobi_only = true;
   for (int k = 1; k <= 2; ++k) // the smoother on the way down, then up
   {
      HYPRE_Int relax = -1;
      hypre_BoomerAMGGetCycleRelaxType(data, &relax, k);
      const char* where = (k == 1) ? "down" : "up";
      if (relax != 7 && relax != 18) { jacobi_only = false; }
      if (relax == 7 || relax == 18 || relax == 11 || relax == 12 || relax == 16)
      {
         continue;
      }
      if (relax == 3 || relax == 4 || relax == 6 || relax == 8 || relax == 13 ||
          relax == 14 || relax == 88 || relax == 89)
      {
         out << "smoother type " << relax << " (" << where << ") runs as sparse "
             "triangular solves on a GPU, about a hundred times slower than "
             "l1-Jacobi (18). ";
      }
      else
      {
         out << "smoother type " << relax << " (" << where << ") is not "
             "GPU-supported (7, 18, 11, 12 and 16 are). ";
      }
   }
   if (hypre_ParAMGDataRelaxOrder(data) != 0 && !jacobi_only)
   {
      out << "C/F relaxation order is GPU-supported only with smoothers 7 and "
          "18. ";
   }
   if (hypre_ParAMGDataAggNumLevels(data) > 0)
   {
      const HYPRE_Int agg_interp = hypre_ParAMGDataAggInterpType(data);
      if (agg_interp != 5 && agg_interp != 7)
      {
         out << "aggressive coarsening with second-stage interpolation type "
             << agg_interp << " is not GPU-supported (5 and 7 are). ";
      }
   }
#if !defined(HYPRE_USING_CUSOLVER) && !defined(HYPRE_USING_MAGMA)
   {
      HYPRE_Int coarse = -1;
      hypre_BoomerAMGGetCycleRelaxType(data, &coarse, 3);
      if (coarse == 9 || coarse == 19 || coarse == 98 || coarse == 99 ||
          coarse == 198 || coarse == 199)
      {
         out << "the coarsest-level direct solve (type " << coarse << ") falls "
             "back to the host: hypre was built without cuSOLVER and MAGMA. ";
      }
   }
#endif
   return out.str();
}

void VerifyHypreAmgOnDevice(mfem::HypreBoomerAMG& amg, const char* what)
{
   const std::string problems = HypreAmgDeviceProblems(amg);
   MFEM_VERIFY(problems.empty(), what << " would not run on hypre's fast GPU "
               "path: " << problems << "(precond/hypre_device.hpp)");
}

} // namespace incns
