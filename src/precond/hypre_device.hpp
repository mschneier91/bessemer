/**
 * @file hypre_device.hpp
 * @brief Keeps BoomerAMG on hypre's fast GPU path: the smoother to use, and a
 *        check of the options that were set.
 */
#pragma once

#include "mfem.hpp"

#include <string>

namespace incns
{

/**
 * @brief The smoother for a SYMMETRIC V-cycle (one that may precondition CG).
 *
 * Host: l1 hybrid symmetric Gauss-Seidel (hypre type 8). GPU: l1-Jacobi (type
 * 18). Type 8 does run on a GPU, but as two sparse triangular solves per sweep
 * (cuSPARSE): measured on an H100 at 86 ms per V-cycle for 32,768 unknowns and
 * 670 ms for 262,144, against 0.74 ms with l1-Jacobi (2026-10-12), with the
 * same outer iteration count. l1-Jacobi is symmetric, and it is the smoother
 * hypre documents for GPUs and MFEM's own GPU default.
 * @return The hypre relaxation type.
 */
int SymmetricAmgSmoother();

/**
 * @brief What would keep this AMG's V-cycle off hypre's fast GPU path.
 *
 * Checks the options against the list in hypre's manual ("GPU-supported
 * Options", BoomerAMG chapter): coarsening PMIS (8); interpolation 3, 6, 14,
 * 15 or 18; smoothers 7, 18, 11, 12 or 16 on the way down and up (3, 4, 6 and
 * 8 are reported too: they run, through triangular solves, about a hundred
 * times slower); relaxation order 0, or C/F with 7 or 18; aggressive
 * coarsening only with second-stage interpolation 5 or 7; and a coarsest-level
 * direct solve only when hypre was built with cuSOLVER or MAGMA.
 * @param amg The solver, after its options are set (before or after setup).
 * @return One sentence per problem; empty when there is none, and always
 *         empty when hypre does not run on a GPU.
 */
std::string HypreAmgDeviceProblems(mfem::HypreBoomerAMG& amg);

/**
 * @brief Abort, naming the problems, unless HypreAmgDeviceProblems is empty.
 * @param amg  The solver, after its options are set.
 * @param what Who owns it, for the message (e.g. "the pressure LOR-AMG").
 */
void VerifyHypreAmgOnDevice(mfem::HypreBoomerAMG& amg, const char* what);

/**
 * @brief The diagonal of a square HypreParMatrix whose rows may have no
 *        diagonal entry, zero there: boundary and interface masses, whose
 *        rows off their faces are empty (and a rank without such faces has
 *        no entry at all).
 *
 * MFEM's HypreParMatrix::GetDiag, and hypre's Jacobi (HypreSmoother), read
 * each row's first entry as its diagonal: an empty row gives the next row's
 * entry, or reads past the arrays -- garbage on a host, an illegal access or
 * a NaN on a GPU (linear_elasticity_test E6 and robin_test R1 at np 2, fsi_test
 * F3 now and then; 2026-10-23). Host work, once per assembly.
 * @param A The matrix (its data are read on the host, then handed back).
 * @param d The diagonal (true dofs).
 */
void HypreDiagonal(const mfem::HypreParMatrix& A, mfem::Vector& d);

} // namespace incns
