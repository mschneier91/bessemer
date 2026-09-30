// Sprint 2.1 FLAGSHIP: dealiasing / nonlinear-term exactness.
//
// The claim under test (CLAUDE.md "Dealiasing"): the convective term must be
// integrated by OVER-INTEGRATION -- a rule exact for the quadratic product --
// and NOT by the default/collocated rule, which aliases.
//
// The test is an algebraic exactness check, deliberately not a convergence
// study, so a failure localizes to the quadrature rather than to the solver:
//
//   For a polynomial velocity u of degree k on an AFFINE mesh, the analytic
//   convective term (u.grad)u is itself a polynomial (degree 2k-1), and it lies
//   in the degree-k space only when we choose u so that it does. So instead of
//   comparing against an interpolant (which would confound interpolation error
//   with quadrature error), we compare the WEAK form against an exactly
//   computed reference:
//
//       N(u)_i = \int (u.grad)u . phi_i
//
//   computed once with a very-high-order rule (the trusted reference) and once
//   with the operator's rule. Over-integration must match the reference to
//   machine precision; the collocation rule must NOT (that is the aliasing).
//
// Asserting only the first half would pass trivially if someone raised every
// rule in the RuleBook; asserting the second half is what pins the choice.

#include <gtest/gtest.h>

#include "operators/convection.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::Convection;
using incns::MixedSpaces;
using incns::Rule1D;
using incns::RuleBook;

namespace
{

// AFFINE Cartesian mesh. Affine matters: the Jacobian is then constant per
// element, so the integrand's polynomial degree is exactly 3k-1 and "exact to
// degree 3k" is a statement we can hold the quadrature to. On a curved mesh the
// integrand is rational and no finite rule is exact -- that case is covered by
// the operator's other tests, not by this exactness claim.
Mesh MakeAffineMesh(int dim, int n)
{
   return (dim == 2)
          ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false, 1.0, 1.0)
          : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
}

// A polynomial velocity field of degree exactly k in each component, chosen so
// that (u.grad)u is nonzero and genuinely mixes the components (a field like
// u = (y^k, 0) would make the nonlinearity degenerate).
void PolyVelocity(int k, const Vector& x, Vector& u)
{
   const double xx = x[0], yy = x[1];
   const double zz = (x.Size() == 3) ? x[2] : 0.0;
   u(0) = std::pow(yy, k) + 0.5 * std::pow(xx, k);
   u(1) = std::pow(xx, k) - 0.25 * std::pow(yy, k);
   if (x.Size() == 3)
   {
      u(0) += 0.25 * std::pow(zz, k);
      u(2) = std::pow(xx, k) + 0.5 * std::pow(zz, k);
   }
}

// Reference weak convective term, assembled with an explicitly supplied rule.
// Uses the SAME MFEM integrator as the operator so the only variable is the
// quadrature -- this isolates aliasing from any difference in the math.
void WeakConvection(ParFiniteElementSpace& vfes, const Vector& u_true,
                    const IntegrationRule& ir, Vector& out)
{
   ParNonlinearForm form(&vfes);
   auto* nlfi = new VectorConvectionNLFIntegrator();
   nlfi->SetIntRule(&ir);
   form.AddDomainIntegrator(nlfi);
   // Full (legacy) assembly for the reference: no PA, so the reference path
   // shares as little machinery as possible with the operator under test.
   out.SetSize(vfes.GetTrueVSize());
   form.Mult(u_true, out);
}

double RelDiff(const Vector& a, const Vector& b)
{
   Vector d(a.Size());
   subtract(a, b, d);
   const double nb = b.Norml2();
   return (nb > 0.0) ? d.Norml2() / nb : d.Norml2();
}

} // namespace

// ---------------------------------------------------------------------------
// THE FLAGSHIP ASSERTION, both halves.
// ---------------------------------------------------------------------------
class ConvectionExactness
   : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(ConvectionExactness, OverIntegratedIsExactAndCollocatedIsNot)
{
   const int dim = std::get<0>(GetParam());
   const int k   = std::get<1>(GetParam());

   Mesh serial = MakeAffineMesh(dim, 2);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   RuleBook rules;
   MixedSpaces spaces(mesh, k, k - 1);

   // Interpolate the polynomial velocity; degree k lies in the space exactly,
   // so u_true carries no interpolation error into the comparison.
   ParGridFunction u_gf(&spaces.Velocity());
   VectorFunctionCoefficient u_coeff(
   dim, [k](const Vector & x, Vector & v) { PolyVelocity(k, x, v); });
   u_gf.ProjectCoefficient(u_coeff);
   Vector u_true(spaces.Velocity().GetTrueVSize());
   u_gf.GetTrueDofs(u_true);

   const Geometry::Type geom =
      (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // --- the trusted reference: far above the integrand's degree -------------
   // 3k-1 is the true degree; integrating to 3k+6 is unambiguously exact, so
   // any disagreement below is quadrature error in the rule under test.
   Vector ref;
   WeakConvection(spaces.Velocity(), u_true,
                  rules.Get(geom, 3 * k + 6), ref);
   ASSERT_GT(ref.Norml2(), 0.0) << "reference term vanished -- bad test field";

   // --- half 1: the operator's dealiased rule REPRODUCES it ----------------
   Convection conv(spaces, rules);
   Vector got(spaces.Velocity().GetTrueVSize());
   conv.Mult(u_true, got);
   const double rel_dealiased = RelDiff(got, ref);
   EXPECT_LT(rel_dealiased, 1e-12)
         << "dim=" << dim << " k=" << k
         << ": over-integrated convection is NOT exact (rel=" << rel_dealiased
         << ") -- the dealiasing rule is too low";

   // --- half 2: the COLLOCATED rule does NOT -------------------------------
   // This is the half that pins the design decision. The GLL collocation rule
   // at 2k-1 is what MFEM's navier miniapp uses and what this project rejects.
   Vector aliased;
   WeakConvection(spaces.Velocity(), u_true,
                  rules.Get(geom, 2 * k - 1, Rule1D::GaussLobatto), aliased);
   const double rel_collocated = RelDiff(aliased, ref);
   EXPECT_GT(rel_collocated, 1e-10)
         << "dim=" << dim << " k=" << k
         << ": the collocated rule reproduced the term exactly (rel="
         << rel_collocated << ") -- then this test proves nothing about "
         << "dealiasing; the field or the rule is degenerate";
}

INSTANTIATE_TEST_SUITE_P(
   DimOrder, ConvectionExactness,
   ::testing::Combine(::testing::Values(2, 3), ::testing::Values(2, 3, 4)));

// The rule the operator actually selected is part of the contract: a silent
// downgrade to the default rule is exactly the regression this guards.
TEST(Convection, UsesTheDealiasedRuleNotTheDefault)
{
   Mesh serial = MakeAffineMesh(2, 2);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   RuleBook rules;
   const int k = 3;
   MixedSpaces spaces(mesh, k, k - 1);
   Convection conv(spaces, rules);

   const IntegrationRule& want = rules.Get(Geometry::SQUARE,
                                           Convection::DealiasedOrder(k));
   EXPECT_EQ(&conv.Rule(), &want)
         << "convection is not using the RuleBook's dealiased rule";
   EXPECT_EQ(Convection::DealiasedOrder(k), 3 * k);
   // Strictly more points than the linear blocks' ~2k rule -- the concrete
   // statement of "over-integrated".
   EXPECT_GT(conv.Rule().GetNPoints(),
             rules.Get(Geometry::SQUARE, 2 * k).GetNPoints());
}
