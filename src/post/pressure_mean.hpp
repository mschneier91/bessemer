/**
 * @file pressure_mean.hpp
 * @brief Mass-weighted mean-zero normalization of the pressure field.
 */
#ifndef INCNS_POST_PRESSURE_MEAN_HPP
#define INCNS_POST_PRESSURE_MEAN_HPP

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief Mass-weighted (L2) mean of a field over the domain, @c (∫p)/|Ω|.
 *
 * Computed as a quadrature integral against the finite element basis (a
 * mass-matrix weighting), @b not an average of nodal values -- the nodal average
 * is wrong at high order. Collective over the field's MPI communicator.
 *
 * @param p     Field to average (typically the pressure).
 * @param rules Quadrature source; a rule exact for the field's order is used.
 * @return The mass-weighted mean value.
 */
double MassWeightedMean(const mfem::ParGridFunction& p, const RuleBook& rules);

/**
 * @brief Shift a field in place to zero mass-weighted mean: @c p <- p - mean(p).
 *
 * Subtracting a constant reduces the mean by exactly that constant, so the
 * result has mean zero to quadrature precision. Collective.
 *
 * @param p     Field to normalize in place.
 * @param rules Quadrature source.
 */
void SubtractMean(mfem::ParGridFunction& p, const RuleBook& rules);

} // namespace incns

#endif // INCNS_POST_PRESSURE_MEAN_HPP
