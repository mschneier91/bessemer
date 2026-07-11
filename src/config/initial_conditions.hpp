/**
 * @file initial_conditions.hpp
 * @brief Named analytic initial-velocity registry for YAML decks.
 */
#ifndef INCNS_CONFIG_INITIAL_CONDITIONS_HPP
#define INCNS_CONFIG_INITIAL_CONDITIONS_HPP

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Build the named initial-velocity coefficient a deck selected.
 *
 * Supported names: @c "zero" (all components zero) and @c "taylor_green_2d"
 * (the analytic TGV of exact/tgv2d at t = 0, using Parameters::nu). The
 * registry is how decks pick analytic ICs without recompiling a driver; new
 * cases add entries here, never edits to the solver core.
 *
 * @param params Case parameters (name, dimension, viscosity).
 * @return The owning coefficient; aborts on an unknown name.
 */
std::unique_ptr<mfem::VectorCoefficient>
MakeInitialVelocity(const Parameters& params);

} // namespace incns

#endif // INCNS_CONFIG_INITIAL_CONDITIONS_HPP
