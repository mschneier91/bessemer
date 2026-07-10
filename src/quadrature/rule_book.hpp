#ifndef INCNS_QUADRATURE_RULE_BOOK_HPP
#define INCNS_QUADRATURE_RULE_BOOK_HPP

#include "mfem.hpp"

namespace incns
{

/// 1D quadrature families the RuleBook can hand out.
enum class Rule1D
{
   GaussLegendre, // default everywhere
   GaussLobatto   // collocated-mass option (endpoints included)
};

/// Central owner of quadrature rules. MFEM integrators hold *non-owning*
/// `const IntegrationRule*`, so the rules must outlive every operator that
/// references them: construct one RuleBook that lives for the whole solve and
/// hand out references from it -- never build a local IntegrationRules in an
/// integrator, and never pass the address of a temporary rule.
///
/// It owns one IntegrationRules container per 1D family (Gauss-Legendre by
/// default, Gauss-Lobatto for the collocated-mass option) and selects both the
/// polynomial exactness order and the 1D type per request. Elevated rules (for
/// error measurement, and later for dealiasing) are just `Get` at a higher
/// order -- no separate API.
class RuleBook
{
public:
   RuleBook();

   // Non-copyable / non-movable: consumers hold pointers into the containers.
   RuleBook(const RuleBook&) = delete;
   RuleBook& operator=(const RuleBook&) = delete;

   /// Rule exact for polynomials up to `order`, on `geom`, in the given 1D
   /// family. The returned reference is stable for this RuleBook's lifetime
   /// (repeat calls with the same arguments return the same object).
   const mfem::IntegrationRule& Get(
      mfem::Geometry::Type geom, int order,
      Rule1D family = Rule1D::GaussLegendre) const;

   /// Collocated Gauss-Lobatto mass rule for degree-k elements: k+1 GLL points
   /// per direction, collocated with the GLL nodal H1 basis so the mass matrix
   /// is diagonal (SEM lumping). Deliberately under-integrated -- exact to
   /// degree 2k-1 vs the 2k mass integrand -- which is standard spectral-element
   /// practice and does not degrade convergence order.
   const mfem::IntegrationRule& CollocatedMass(
      mfem::Geometry::Type geom, int element_order) const;

private:
   // `mutable`: IntegrationRules::Get lazily materialises and caches rules, so
   // it is not const, but querying the RuleBook is logically const.
   mutable mfem::IntegrationRules gl_;
   mutable mfem::IntegrationRules gll_;
};

} // namespace incns

#endif // INCNS_QUADRATURE_RULE_BOOK_HPP
