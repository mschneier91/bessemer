/**
 * @file initial_conditions.hpp
 * @brief Named analytic initial-velocity registry for YAML decks.
 */
#pragma once

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Build the named initial-velocity coefficient a deck selected.
 *
 * Supported names: @c "zero" (all components zero), @c "uniform" (section
 * @c initial), @c "taylor_green_2d" (the analytic TGV of exact/tgv2d at t =
 * 0, using Parameters::nu), @c "taylor_green_3d" (u = (sin x cos y cos z,
 * -cos x sin y cos z, 0), the Re 1600 benchmark's start on a 2 pi periodic
 * box) and @c "channel" (MakeChannelInitialVelocity). The
 * registry is how decks pick analytic ICs without recompiling a driver; new
 * cases add entries here, never edits to the solver core.
 *
 * @param params Case parameters (name, dimension, viscosity).
 * @return The owning coefficient; aborts on an unknown name.
 */
std::unique_ptr<mfem::VectorCoefficient>
MakeInitialVelocity(const Parameters& params);

/**
 * @brief Reichardt's law of the wall, @f$ u^+(y^+) = \ln(1 + \kappa y^+)/\kappa
 *        + 7.8\,(1 - e^{-y^+/11} - (y^+/11)\,e^{-y^+/3}) @f$, @f$\kappa@f$ =
 *        0.41: linear at the wall, logarithmic away from it.
 * @param y_plus Wall distance in wall units.
 * @return The mean velocity in wall units.
 */
double ReichardtUPlus(double y_plus);

/**
 * @brief The turbulent-channel start (deck @c initial_velocity: channel):
 *        walls at y = 0 and y = 2 delta (box mesh, x streamwise), u_tau =
 *        sqrt(f_x delta) from the forcing (the mean pressure gradient), the
 *        mean profile u_tau ReichardtUPlus(y+) for Re_tau = u_tau delta / nu,
 *        plus @c initial.perturbation times the centreline velocity in a
 *        divergence-free perturbation u' = curl(psi), psi = (1 - eta^2)^2 x
 *        (streamwise modes 1-3, spanwise 1-3 in 3D, phases from
 *        @c initial.seed), eta = (y - delta) / delta. u' and its wall-normal
 *        derivative vanish at the walls; the field is exactly
 *        divergence-free.
 * @param params Case parameters (box lengths, nu, forcing, initial).
 * @return The owning coefficient.
 */
std::unique_ptr<mfem::VectorCoefficient>
MakeChannelInitialVelocity(const Parameters& params);

} // namespace incns
