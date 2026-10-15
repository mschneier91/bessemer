/**
 * @file kernel_specializations.hpp
 * @brief Compile-time-sized MFEM partial-assembly kernels for the sizes the
 *        solver uses, registered with MFEM's kernel dispatch.
 */
#pragma once

namespace incns
{

/**
 * @brief Register MFEM kernel specializations for velocity orders 2 to 4
 *        (Q_k / Q_{k-1}): the vector mass at k+1 dofs and k+1 points per
 *        direction (the 2k Gauss rule and the collocated GLL one), and the
 *        vector divergence and its transpose at k+1 trial and k test dofs and
 *        k+2 points (the 2k+2 rule), in 2D and 3D.
 *
 * MFEM's default lists do not have these sizes, so it ran its generic
 * kernels (sizes at run time) for them: 53 % of the GPU kernel time of a
 * Taylor-Green 32^3 Q3/Q2 step (Nsight Systems, 2026-10-15). The specialized
 * kernels are the same code with compile-time sizes; the results agree to
 * roundoff (a compiler may order the arithmetic differently with fixed
 * bounds; kernel_specializations_test). Process-wide and idempotent;
 * StokesOperator calls it before it assembles anything.
 */
void AddKernelSpecializations();

} // namespace incns
