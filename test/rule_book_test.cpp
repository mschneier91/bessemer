// Unit test for src/quadrature/rule_book: correct point count and 1D family per
// (order, type) request, GLL-includes-endpoints / GL-does-not family fingerprint,
// the collocated-mass point count, and the stable-address contract that the
// non-owning integrator pointers rely on. Pure quadrature -- no solver.

#include <gtest/gtest.h>

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

using namespace mfem;
using incns::Rule1D;
using incns::RuleBook;

namespace
{
// Do the abscissae reach both reference-element endpoints (0 and 1)?
bool IncludesEndpoints(const IntegrationRule& r)
{
   double mn = 1.0, mx = 0.0;
   for (int i = 0; i < r.GetNPoints(); ++i)
   {
      const double x = r.IntPoint(i).x;
      mn = std::min(mn, x);
      mx = std::max(mx, x);
   }
   return mn < 1e-12 && mx > 1.0 - 1e-12;
}
} // namespace

// Gauss-Legendre: n = floor(p/2)+1 points on a segment, strictly interior.
TEST(RuleBook, GaussLegendreSegmentCountsAndInterior)
{
   RuleBook rb;
   for (int p = 1; p <= 6; ++p)
   {
      const IntegrationRule& r = rb.Get(Geometry::SEGMENT, p, Rule1D::GaussLegendre);
      EXPECT_EQ(r.GetNPoints(), p / 2 + 1) << "order " << p;
      EXPECT_FALSE(IncludesEndpoints(r)) << "GL order " << p << " hit an endpoint";
   }
}

// Gauss-Lobatto: n = floor(p/2)+2 points on a segment, endpoints included.
TEST(RuleBook, GaussLobattoSegmentCountsAndEndpoints)
{
   RuleBook rb;
   for (int p = 1; p <= 6; ++p)
   {
      const IntegrationRule& r = rb.Get(Geometry::SEGMENT, p, Rule1D::GaussLobatto);
      EXPECT_EQ(r.GetNPoints(), p / 2 + 2) << "order " << p;
      EXPECT_TRUE(IncludesEndpoints(r)) << "GLL order " << p << " missed an endpoint";
   }
}

// GL default matches an explicit GL request (same object).
TEST(RuleBook, DefaultFamilyIsGaussLegendre)
{
   RuleBook rb;
   EXPECT_EQ(&rb.Get(Geometry::SQUARE, 4),
             &rb.Get(Geometry::SQUARE, 4, Rule1D::GaussLegendre));
}

// Collocated mass: k+1 points per direction (2D and 3D), and it is exactly the
// GLL rule of exactness order 2k-1.
TEST(RuleBook, CollocatedMassPointCounts)
{
   RuleBook rb;
   for (int k = 1; k <= 4; ++k)
   {
      const IntegrationRule& sq = rb.CollocatedMass(Geometry::SQUARE, k);
      const IntegrationRule& cu = rb.CollocatedMass(Geometry::CUBE, k);
      EXPECT_EQ(sq.GetNPoints(), (k + 1) * (k + 1)) << "k=" << k;
      EXPECT_EQ(cu.GetNPoints(), (k + 1) * (k + 1) * (k + 1)) << "k=" << k;
      EXPECT_TRUE(IncludesEndpoints(sq));
      EXPECT_EQ(&sq, &rb.Get(Geometry::SQUARE, 2 * k - 1, Rule1D::GaussLobatto));
   }
}

// GL and GLL of the same order are distinct rules.
TEST(RuleBook, FamiliesAreDistinct)
{
   RuleBook rb;
   const IntegrationRule& gl = rb.Get(Geometry::SQUARE, 3, Rule1D::GaussLegendre);
   const IntegrationRule& gll = rb.Get(Geometry::SQUARE, 3, Rule1D::GaussLobatto);
   EXPECT_NE(&gl, &gll);
   EXPECT_NE(gl.GetNPoints(), gll.GetNPoints());
}

// Lifetime / non-owning-pointer contract: repeat requests return the SAME
// object, so a stored `const IntegrationRule*` stays valid.
TEST(RuleBook, StableAddresses)
{
   RuleBook rb;
   EXPECT_EQ(&rb.Get(Geometry::SQUARE, 6), &rb.Get(Geometry::SQUARE, 6));
   EXPECT_EQ(&rb.Get(Geometry::CUBE, 5, Rule1D::GaussLobatto),
             &rb.Get(Geometry::CUBE, 5, Rule1D::GaussLobatto));
}
