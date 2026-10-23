// scripts/debug_device_shim.cpp -- preloaded by scripts/debug_device.sh only.
//
// MFEM's debug device mprotects whichever copy (host or device) is not in use
// and records it per allocation (h_rw, d_rw). MemoryManager::GetAliasHostPtr
// (4.10; master as of 2026-10-22) protects the base's device range but records
// only the host side (h_rw = false), where GetAliasDevicePtr records both. The
// base's next device access then skips its Unprotect and faults: a false
// positive wherever a device vector is read on the host through an alias. In
// the CPU build hypre runs on the host, so every HypreParMatrix::Mult does it
// (hit by run_monitor_test R5, checkpoint_test's rolling checkpoint and
// mesh_census_test C2, 2026-10-22). Real GPUs have no page protection and the
// data is right, so this changes the debug device's bookkeeping only.
//
// The fix MFEM needs is `mem->d_rw = mem->h_rw = false;` in GetAliasHostPtr.
// Without its internals, this interposes GetAliasHostPtr and reaches that end
// state through the originals: the host read (with its copy); a device read of
// the alias without a copy (records both sides, protects the host range); the
// host read again without a copy (reopens the host range, protects the device
// range). Drop it when MFEM records both sides.
#include <dlfcn.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace mfem
{
// The member's signature only (access does not enter the mangled name).
class MemoryManager
{
public:
   void* GetAliasHostPtr(const void* ptr, std::size_t bytes, bool copy);
};
} // namespace mfem

namespace
{
using AliasPtrFn = void* (*)(void*, const void*, std::size_t, bool);

AliasPtrFn Original(const char* name)
{
   void* f = dlsym(RTLD_NEXT, name);
   if (!f)
   {
      std::fprintf(stderr, "debug_device_shim: %s not found in MFEM\n", name);
      std::abort();
   }
   return reinterpret_cast<AliasPtrFn>(f);
}
} // namespace

void* mfem::MemoryManager::GetAliasHostPtr(const void* ptr, std::size_t bytes,
      bool copy)
{
   static const AliasPtrFn host =
      Original("_ZN4mfem13MemoryManager15GetAliasHostPtrEPKvmb");
   static const AliasPtrFn device =
      Original("_ZN4mfem13MemoryManager17GetAliasDevicePtrEPKvmb");
   host(this, ptr, bytes, copy);
   device(this, ptr, bytes, false);
   return host(this, ptr, bytes, false);
}
