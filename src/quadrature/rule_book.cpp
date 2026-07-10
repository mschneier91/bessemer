#include "quadrature/rule_book.hpp"

namespace incns
{

using namespace mfem;

RuleBook::RuleBook()
   : gl_(0, Quadrature1D::GaussLegendre),
     gll_(0, Quadrature1D::GaussLobatto)
{
}

const IntegrationRule& RuleBook::Get(Geometry::Type geom, int order,
                                     Rule1D family) const
{
   MFEM_VERIFY(order >= 0, "RuleBook: quadrature order must be >= 0");
   IntegrationRules& rules = (family == Rule1D::GaussLobatto) ? gll_ : gl_;
   return rules.Get(geom, order);
}

const IntegrationRule& RuleBook::CollocatedMass(Geometry::Type geom,
      int element_order) const
{
   MFEM_VERIFY(element_order >= 1,
               "RuleBook: element order must be >= 1 for collocated mass");
   // Requesting exactness order 2k-1 from the Gauss-Lobatto family yields k+1
   // points per direction, collocated with the degree-k GLL nodal basis.
   return Get(geom, 2 * element_order - 1, Rule1D::GaussLobatto);
}

} // namespace incns
