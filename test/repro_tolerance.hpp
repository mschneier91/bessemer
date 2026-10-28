// Tolerance for tests that assert BITWISE REPRODUCIBILITY -- "the same
// arithmetic path run twice gives the same answer" -- as opposed to tests that
// assert a physical or discretization error bound.
//
// On a real GPU backend that premise is false. hypre's CUDA BoomerAMG is not
// run-to-run deterministic (PMIS coarsening makes randomized choices and the
// SpMV accumulates with atomics), so the coarse hierarchy and therefore the
// Krylov iterates differ between two runs of the same case. Measured on an
// H100 (2026-07-21): the SAME checkpoint test binary run twice on the same node
// gave 1.1316426869909901e-11 / 1.1316426014074982e-11 (fixed), 1.5653972e-11 /
// 1.5095617e-11 (adaptive), 1.2767666e-11 / 1.2767626e-11 (rolling). deck_test,
// which compares two identical runs and never touches checkpoint I/O, sits at
// the same 1.47e-11 relative. So ~1.5e-11 is this hardware's reproducibility
// floor, not a defect in the code under test.
//
// The host bound is kept as-is: on CPU the runs ARE bitwise identical and that
// is a genuinely valuable guard, so it must not be relaxed. DEBUG_DEVICE is
// deliberately NOT in the mask -- it executes forall bodies on the host and is
// bitwise reproducible, so the debug sweep keeps the strict bound too.

#pragma once

#include "mfem.hpp"

namespace incns_test
{

/**
 * @brief Relative tolerance for a bitwise-reproducibility assertion.
 * @param host_tol The bound to demand when the arithmetic really is
 *                 reproducible (CPU and debug-device backends).
 * @param gpu_tol  The bound on a real GPU: by default ~60x the measured
 *                 1.5e-11 floor of one run against another, still orders
 *                 below any discretization-error scale these cases resolve.
 *                 Runs whose iterations stop at a solver tolerance (coupled
 *                 FSI steps) differ at that tolerance instead.
 * @return @p host_tol on host backends; @p gpu_tol on a real GPU.
 */
/// @brief Whether the backend is a real GPU, where two runs of a case are
/// not bitwise reproducible (see above).
inline bool OnRealGpu()
{
   return mfem::Device::Allows(mfem::Backend::CUDA_MASK |
                               mfem::Backend::HIP_MASK);
}

inline double ReproTol(double host_tol, double gpu_tol = 1e-9)
{
   return OnRealGpu() ? gpu_tol : host_tol;
}

} // namespace incns_test
