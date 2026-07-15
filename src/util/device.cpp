#include "util/device.hpp"

#include "mfem.hpp"

namespace incns
{

void ConfigureDevice(const std::string& backend)
{
   // One mfem::Device per process, for the program's lifetime. Held in a
   // function-local static so any driver (C++ or Python) can request it without
   // threading a Device object through the call tree.
   static mfem::Device device;
   static bool configured = false;
   static std::string current;

   if (!configured)
   {
      device.Configure(backend);
      configured = true;
      current = backend;
   }
   else
   {
      MFEM_VERIFY(current == backend,
                  "ConfigureDevice: the device is already configured as '"
                  << current << "' and cannot be reconfigured to '" << backend
                  << "' (one mfem::Device per process)");
   }
}

} // namespace incns
