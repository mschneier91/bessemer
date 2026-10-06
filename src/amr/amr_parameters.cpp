#include "amr/amr_parameters.hpp"

#include "mfem.hpp"

namespace incns
{

void AmrParameters::Validate() const
{
   MFEM_VERIFY(interval >= 0, "amr: interval must be >= 0");
   MFEM_VERIFY(initial_passes >= 0, "amr: initial_passes must be >= 0");
   MFEM_VERIFY(passes_per_event >= 1, "amr: passes_per_event must be >= 1");
   MFEM_VERIFY(aniso_ratio > 0.0 && aniso_ratio <= 1.0,
               "amr: aniso_ratio must lie in (0, 1]");
   MFEM_VERIFY(theta > 0.0 && theta <= 1.0, "amr: theta must lie in (0, 1]");
   MFEM_VERIFY(tolerance > 0.0, "amr: tolerance must be positive");
   MFEM_VERIFY(min_size >= 0.0, "amr: min_size must be >= 0");
   MFEM_VERIFY(max_elements >= 0, "amr: max_elements must be >= 0");
   MFEM_VERIFY(nc_limit >= 1, "amr: nc_limit must be >= 1");
}

} // namespace incns
