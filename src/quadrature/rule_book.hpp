/**
 * @file rule_book.hpp
 * @brief Central owner of quadrature rules (per-integrator order and 1D type).
 */
#ifndef INCNS_QUADRATURE_RULE_BOOK_HPP
#define INCNS_QUADRATURE_RULE_BOOK_HPP

#include "mfem.hpp"

namespace incns
{

/// 1D quadrature families the RuleBook can hand out.
enum class Rule1D
{
   GaussLegendre, ///< Default everywhere; abscissae strictly interior.
   GaussLobatto   ///< Collocated-mass option; abscissae include the endpoints.
};

/**
 * @brief Central owner of quadrature rules for the whole solve.
 *
 * MFEM integrators hold *non-owning* @c const @c IntegrationRule* pointers, so
 * the rules must outlive every operator that references them. Construct one
 * RuleBook that lives for the whole solve and hand out references from it: never
 * build a local mfem::IntegrationRules inside an integrator, and never pass the
 * address of a temporary rule.
 *
 * It owns one mfem::IntegrationRules container per 1D family (Gauss-Legendre by
 * default, Gauss-Lobatto for the collocated-mass option) and selects both the
 * polynomial exactness order and the 1D type per request. Elevated rules (for
 * error measurement, and later for dealiasing) are just @ref Get at a higher
 * order -- there is no separate API. The class is non-copyable and non-movable
 * so pointers into it cannot be invalidated.
 */
class RuleBook
{
public:
   RuleBook();

   RuleBook(const RuleBook&) = delete;
   RuleBook& operator=(const RuleBook&) = delete;

   /**
    * @brief Quadrature rule exact for polynomials up to a given order.
    * @param geom   Reference geometry (e.g. mfem::Geometry::SQUARE).
    * @param order  Polynomial degree the rule integrates exactly (>= 0).
    * @param family 1D quadrature family (defaults to Gauss-Legendre).
    * @return A rule whose reference is stable for this RuleBook's lifetime;
    *         repeat calls with the same arguments return the same object.
    */
   const mfem::IntegrationRule& Get(
      mfem::Geometry::Type geom, int order,
      Rule1D family = Rule1D::GaussLegendre) const;

   /**
    * @brief Collocated Gauss-Lobatto mass rule for degree-k elements.
    *
    * Yields @c k+1 GLL points per direction, collocated with the degree-k GLL
    * nodal H1 basis so the mass matrix is diagonal (SEM lumping). Deliberately
    * under-integrated -- exact to degree @c 2k-1 versus the @c 2k mass integrand
    * -- which is standard spectral-element practice and does not degrade
    * convergence order.
    *
    * @param geom          Reference geometry (mfem::Geometry::SQUARE / CUBE).
    * @param element_order Element polynomial order @c k (>= 1).
    * @return The collocated GLL rule (equivalently @ref Get at order @c 2k-1 in
    *         the Gauss-Lobatto family).
    */
   const mfem::IntegrationRule& CollocatedMass(
      mfem::Geometry::Type geom, int element_order) const;

private:
   /// Gauss-Legendre rules. @c mutable: IntegrationRules::Get lazily caches, so
   /// it is non-const, while querying the RuleBook is logically const.
   mutable mfem::IntegrationRules gl_;
   /// Gauss-Lobatto rules (collocated-mass option); @c mutable for the same reason.
   mutable mfem::IntegrationRules gll_;
};

} // namespace incns

#endif // INCNS_QUADRATURE_RULE_BOOK_HPP
