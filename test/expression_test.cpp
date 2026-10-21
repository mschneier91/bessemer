// Deck expressions (config/expression): values exact to roundoff against the
// same arithmetic written in C++, at np 1/2/4 (the parser is rank-local).
//  E1 precedence and associativity (+ - * / ^ **, unary minus, parentheses);
//  E2 the functions, constants, variables and named parameters;
//  E3 errors name the problem: unknown names and functions, argument counts,
//     unbalanced parentheses, trailing input, shadowing parameters;
//  E4 the vector form evaluates per component and as a VectorCoefficient.

#include <gtest/gtest.h>

#include "config/expression.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

using incns::Expression;
using incns::VectorExpression;

namespace
{

double At(const std::string& s, double x = 0.3, double y = -0.7,
          double z = 1.1, double t = 0.25)
{
   const double v[4] = {x, y, z, t};
   return Expression(s).Eval(v);
}

void ExpectError(const std::string& s, const std::string& fragment)
{
   try
   {
      Expression e(s, {{"A", 2.0}});
      ADD_FAILURE() << "'" << s << "' parsed";
   }
   catch (const std::invalid_argument& err)
   {
      EXPECT_NE(std::string(err.what()).find(fragment), std::string::npos)
            << err.what();
   }
}

} // namespace

TEST(Expression, E1_Precedence)
{
   EXPECT_DOUBLE_EQ(At("1 + 2 * 3"), 7.0);
   EXPECT_DOUBLE_EQ(At("(1 + 2) * 3"), 9.0);
   EXPECT_DOUBLE_EQ(At("8 / 4 / 2"), 1.0);
   EXPECT_DOUBLE_EQ(At("10 - 4 - 3"), 3.0);
   EXPECT_DOUBLE_EQ(At("2 ^ 3 ^ 2"), 512.0);
   EXPECT_DOUBLE_EQ(At("2 ** 3 ** 2"), 512.0);
   EXPECT_DOUBLE_EQ(At("-2 ^ 2"), -4.0);
   EXPECT_DOUBLE_EQ(At("2 ^ -1"), 0.5);
   EXPECT_DOUBLE_EQ(At("--3"), 3.0);
   EXPECT_DOUBLE_EQ(At("+3 * -2"), -6.0);
   EXPECT_DOUBLE_EQ(At("1.5e-1 * 2E1"), 3.0);
   EXPECT_DOUBLE_EQ(At(".5 + 2 * x * y"), 0.5 + 2 * 0.3 * -0.7);
}

TEST(Expression, E2_FunctionsVariablesParameters)
{
   const double x = 0.3, y = -0.7, z = 1.1, t = 0.25;
   EXPECT_DOUBLE_EQ(At("sin(pi * x)^2 * cos(z)"),
                    std::pow(std::sin(M_PI * x), 2) * std::cos(z));
   EXPECT_DOUBLE_EQ(At("exp(-t) + log(z) + sqrt(z) + abs(y) + tanh(x)"),
                    std::exp(-t) + std::log(z) + std::sqrt(z) + std::abs(y) +
                    std::tanh(x));
   EXPECT_DOUBLE_EQ(At("tan(x) + asin(x) + acos(x) + atan(y) + sinh(y) + "
                       "cosh(y)"),
                    std::tan(x) + std::asin(x) + std::acos(x) + std::atan(y) +
                    std::sinh(y) + std::cosh(y));
   EXPECT_DOUBLE_EQ(At("min(x, y) + max(x, y) + pow(z, 3) + atan2(y, x)"),
                    std::min(x, y) + std::max(x, y) + std::pow(z, 3) +
                    std::atan2(y, x));
   EXPECT_DOUBLE_EQ(At("e"), std::exp(1.0));
   const Expression p("A * t * 4 * x * (L - x) / L^2", {{"A", 0.2}, {"L", 2.0}});
   const double v[4] = {x, y, z, t};
   EXPECT_DOUBLE_EQ(p.Eval(v), 0.2 * t * 4 * x * (2.0 - x) / 4.0);
   EXPECT_TRUE(p.DependsOnTime());
   EXPECT_FALSE(Expression("x + y").DependsOnTime());
   // z is 0 for a 2D point.
   mfem::Vector x2(2);
   x2(0) = 1.0;
   x2(1) = 2.0;
   EXPECT_DOUBLE_EQ(Expression("x + 10 * y + 100 * z + t").Eval(x2, 0.5), 21.5);
   // A deep expression (beyond the fixed evaluation stack).
   std::string deep = "1";
   for (int i = 0; i < 100; ++i) { deep = "(1 + " + deep + ")"; }
   EXPECT_DOUBLE_EQ(At(deep), 101.0);
   std::string wide = "x";
   for (int i = 0; i < 100; ++i) { wide = "x + (" + wide + ")"; }
   EXPECT_NEAR(At(wide), 101 * x, 1e-13);
}

TEST(Expression, E3_Errors)
{
   ExpectError("x + q", "unknown name 'q'");
   ExpectError("foo(x)", "unknown function 'foo'");
   ExpectError("sin(x, y)", "sin takes 1 argument(s), not 2");
   ExpectError("max(x)", "max takes 2 argument(s), not 1");
   ExpectError("(x + 1", "missing ')'");
   ExpectError("x + 1)", "unexpected ')'");
   ExpectError("x +", "unexpected end");
   ExpectError("2 x", "unexpected 'x'");
   ExpectError("", "unexpected end");
   EXPECT_THROW(Expression("x", {{"t", 1.0}}), std::invalid_argument);
   EXPECT_THROW(Expression("x", {{"sin", 1.0}}), std::invalid_argument);
   EXPECT_THROW(Expression("x", {{"pi", 1.0}}), std::invalid_argument);
}

TEST(Expression, E4_VectorExpression)
{
   auto v = std::make_shared<VectorExpression>(
               std::vector<std::string> {"0", "A * x * t", "-z"},
   std::map<std::string, double> {{"A", 3.0}});
   mfem::Vector x(3), out;
   x(0) = 0.5;
   x(1) = 0.0;
   x(2) = 2.0;
   v->Eval(x, 2.0, out);
   ASSERT_EQ(out.Size(), 3);
   EXPECT_DOUBLE_EQ(out(0), 0.0);
   EXPECT_DOUBLE_EQ(out(1), 3.0);
   EXPECT_DOUBLE_EQ(out(2), -2.0);

   // As a coefficient: evaluated at the physical point, at its time.
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1,
                     mfem::Element::HEXAHEDRON,
                     2.0, 2.0, 2.0);
   std::unique_ptr<mfem::VectorCoefficient> c = VectorExpression::Coefficient(v);
   c->SetTime(0.5);
   mfem::ElementTransformation* T = mesh.GetElementTransformation(0);
   mfem::IntegrationPoint ip;
   ip.Set3(0.25, 0.5, 0.75); // physical (0.5, 1, 1.5)
   T->SetIntPoint(&ip);
   c->Eval(out, *T, ip);
   EXPECT_DOUBLE_EQ(out(1), 3.0 * 0.5 * 0.5);
   EXPECT_DOUBLE_EQ(out(2), -1.5);
}
