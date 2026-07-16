#include "util/device.hpp"

#include "mfem.hpp"

#include <cstdlib>

namespace incns
{

void ConfigureDevice(const std::string& backend)
{
   // Environment override: INCNS_DEVICE, when set, wins over the requested
   // backend. This lets the whole test suite (or any driver) be re-run on a
   // different backend -- e.g. INCNS_DEVICE=debug forces MFEM's mprotect-guarded
   // "debug" device, which faults on any un-annotated host access of device
   // memory (a silent CPU fallback on a real GPU). See scripts/debug_device.sh.
   const char* env = std::getenv("INCNS_DEVICE");
   const std::string want = (env && env[0] != '\0') ? std::string(env) : backend;

   // One mfem::Device per process, for the program's lifetime. Held in a
   // function-local static so any driver (C++ or Python) can request it without
   // threading a Device object through the call tree.
   static mfem::Device device;
   static bool configured = false;
   static std::string current;

   if (!configured)
   {
      device.Configure(want);
      configured = true;
      current = want;
   }
   else
   {
      // Idempotent for the resolved backend. With INCNS_DEVICE set, every call
      // resolves to the same override, so a per-Case ConfigureDevice("cpu")
      // after the harness already selected "debug" is a no-op, not a conflict.
      MFEM_VERIFY(current == want,
                  "ConfigureDevice: the device is already configured as '"
                  << current << "' and cannot be reconfigured to '" << want
                  << "' (one mfem::Device per process)");
   }
}

} // namespace incns
