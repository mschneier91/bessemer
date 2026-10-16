/**
 * @file pressure_mean.hpp
 * @brief Mass-weighted mean-zero normalization of the pressure field.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <memory>

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

/**
 * @brief MassWeightedMean / SubtractMean with what depends only on the space
 *        kept between calls.
 *
 * The free functions assemble the functional (∫ phi_i on the true dofs) and
 * the domain volume at every call: a linear form and three vectors, device
 * allocations and synchronizing frees on a GPU, once per solve when the
 * solver normalizes the pressure. Here they are computed on the first call,
 * the same way, and again only when the space changes (its sequence number):
 * the same values, bitwise (pressure_mean_test).
 */
class PressureMean
{
public:
   /// @param rules Quadrature source (borrowed).
   explicit PressureMean(const RuleBook& rules) : rules_(rules) {}

   /// As MassWeightedMean(p, rules). Collective.
   double Mean(const mfem::ParGridFunction& p) const;

   /// As SubtractMean(p, rules). Collective.
   void Subtract(mfem::ParGridFunction& p) const;

private:
   /// The functional and the volume for @p fes (when new or changed).
   void Setup(mfem::ParFiniteElementSpace& fes) const;

   const RuleBook& rules_;                              ///< Borrowed.
   mutable const mfem::ParFiniteElementSpace* fes_ = nullptr; ///< Set up for.
   mutable long sequence_ = -1;                         ///< Its sequence.
   mutable std::unique_ptr<mfem::Vector> ones_true_;    ///< ∫ phi_i, true dofs.
   mutable double volume_ = 0.0;                        ///< |Omega|.
   mutable mfem::Vector p_true_;                        ///< Work: p's true dofs.
};

} // namespace incns
