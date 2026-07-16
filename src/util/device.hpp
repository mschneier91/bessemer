/**
 * @file device.hpp
 * @brief Process-wide MFEM device (backend) configuration.
 */
#ifndef INCNS_UTIL_DEVICE_HPP
#define INCNS_UTIL_DEVICE_HPP

#include <string>

namespace incns
{

/**
 * @brief Configure the MFEM device backend once per process.
 *
 * Constructs the single mfem::Device with @p backend (e.g. @c "cpu",
 * @c "cuda", @c "hip", @c "cuda:cpu") the first time it is called and is a
 * no-op on later calls with the SAME backend; a different backend is an error
 * (MFEM allows only one Device per process). Call it during case setup, before
 * any mesh / finite element space / true-dof vector is allocated, so all
 * partial-assembly operators and their vectors live in the right memory space.
 *
 * The environment variable @c INCNS_DEVICE, when set, overrides @p backend for
 * the whole process -- used to re-run the suite on MFEM's mprotect-guarded
 * @c "debug" device (catches silent host fallbacks) without editing callers.
 *
 * @param backend The mfem::Device configuration string (default when
 *                @c INCNS_DEVICE is unset).
 */
void ConfigureDevice(const std::string& backend);

} // namespace incns

#endif // INCNS_UTIL_DEVICE_HPP
