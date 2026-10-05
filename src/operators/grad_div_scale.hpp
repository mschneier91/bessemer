/**
 * @file grad_div_scale.hpp
 * @brief How the grad-div coefficient gamma is scaled (mesh size vs viscosity).
 */
#pragma once

namespace incns
{

/**
 * @brief Scaling of the grad-div coefficient gamma = grad_div * scale.
 *
 * The runtime scale @c grad_div (aka c_gd) multiplies a chosen quantity:
 *  - @c OrderH  (default): gamma(x) = c_gd * h_K per element -- order-h,
 *    spatially varying, negligible in the pressure Schur block;
 *  - @c OrderNu: gamma = c_gd * nu -- order-viscosity, constant in space. Note
 *    this is NOT negligible in the Schur complement (its mass term scales like
 *    nu + gamma); gamma still does not enter the Schur block automatically, so
 *    if that matters set cc.nu_pc = nu * (1 + c_gd) explicitly.
 */
enum class GradDivScale
{
   OrderH, ///< gamma = c_gd * h_K per element (default).
   OrderNu ///< gamma = c_gd * nu (constant in space).
};

} // namespace incns
