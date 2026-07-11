// Sprint 1.7 green criterion -- multistep coefficients:
//  * uniform-step BDF1/2/3 and AB(EXT)1/2/3 weights equal their known values;
//  * under non-uniform step ratios, the BDF operator differentiates 1, t, t^2,
//    t^3 exactly to the scheme's order and EXT reproduces polynomial values
//    exactly -- this guards the variable-step bookkeeping (the #1 adaptive-BDF
//    bug: silently reusing uniform coefficients).
// Pure algebra, identical on every rank.

#include <gtest/gtest.h>

#include "time/multistep_coeffs.hpp"

#include <cmath>
#include <vector>

using incns::BdfWeights;
using incns::ExtrapolationWeights;

namespace
{
// Apply weights to samples of f at the nodes.
double Apply(const std::vector<double>& w, const std::vector<double>& t,
             double (*f)(double))
{
   double s = 0.0;
   for (std::size_t j = 0; j < w.size(); ++j) { s += w[j] * f(t[j]); }
   return s;
}

double One(double) { return 1.0; }
double T1(double t) { return t; }
double T2(double t) { return t * t; }
double T3(double t) { return t * t * t; }
} // namespace

TEST(MultistepCoeffs, UniformBdfKnownValues)
{
   const double h = 0.1, t0 = 0.7;

   // BDF1: c*h = {1, -1}
   auto c1 = BdfWeights({t0, t0 - h});
   EXPECT_NEAR(c1[0] * h, 1.0, 1e-13);
   EXPECT_NEAR(c1[1] * h, -1.0, 1e-13);

   // BDF2: c*h = {3/2, -2, 1/2}
   auto c2 = BdfWeights({t0, t0 - h, t0 - 2 * h});
   EXPECT_NEAR(c2[0] * h, 1.5, 1e-13);
   EXPECT_NEAR(c2[1] * h, -2.0, 1e-13);
   EXPECT_NEAR(c2[2] * h, 0.5, 1e-13);

   // BDF3: c*h = {11/6, -3, 3/2, -1/3}
   auto c3 = BdfWeights({t0, t0 - h, t0 - 2 * h, t0 - 3 * h});
   EXPECT_NEAR(c3[0] * h, 11.0 / 6.0, 1e-13);
   EXPECT_NEAR(c3[1] * h, -3.0, 1e-13);
   EXPECT_NEAR(c3[2] * h, 1.5, 1e-13);
   EXPECT_NEAR(c3[3] * h, -1.0 / 3.0, 1e-13);
}

TEST(MultistepCoeffs, UniformExtrapolationKnownValues)
{
   const double h = 0.05, tn = 1.2, target = tn + h;

   // EXT1/AB1: {1}
   auto g1 = ExtrapolationWeights(target, {tn});
   EXPECT_NEAR(g1[0], 1.0, 1e-13);

   // EXT2/AB2: {2, -1}
   auto g2 = ExtrapolationWeights(target, {tn, tn - h});
   EXPECT_NEAR(g2[0], 2.0, 1e-13);
   EXPECT_NEAR(g2[1], -1.0, 1e-13);

   // EXT3/AB3: {3, -3, 1}
   auto g3 = ExtrapolationWeights(target, {tn, tn - h, tn - 2 * h});
   EXPECT_NEAR(g3[0], 3.0, 1e-13);
   EXPECT_NEAR(g3[1], -3.0, 1e-13);
   EXPECT_NEAR(g3[2], 1.0, 1e-13);
}

// Non-uniform nodes (ratios ~0.86, ~1.25): BDF-k differentiates polynomials of
// degree <= k exactly at the newest node.
TEST(MultistepCoeffs, VariableStepBdfDifferentiatesPolynomialsExactly)
{
   const std::vector<double> t4 = {0.30, 0.17, 0.05, -0.10}; // steps .13/.12/.15
   const double t0 = t4[0];

   // BDF3 (4 nodes): exact through t^3.
   auto c3 = BdfWeights(t4);
   EXPECT_NEAR(Apply(c3, t4, One), 0.0, 1e-12);
   EXPECT_NEAR(Apply(c3, t4, T1), 1.0, 1e-12);
   EXPECT_NEAR(Apply(c3, t4, T2), 2.0 * t0, 1e-12);
   EXPECT_NEAR(Apply(c3, t4, T3), 3.0 * t0 * t0, 1e-12);

   // BDF2 (3 nodes): exact through t^2; t^3 must NOT be exact (fingerprint
   // that the order is what it claims, not accidentally higher).
   const std::vector<double> t3(t4.begin(), t4.end() - 1);
   auto c2 = BdfWeights(t3);
   EXPECT_NEAR(Apply(c2, t3, One), 0.0, 1e-12);
   EXPECT_NEAR(Apply(c2, t3, T1), 1.0, 1e-12);
   EXPECT_NEAR(Apply(c2, t3, T2), 2.0 * t0, 1e-12);
   EXPECT_GT(std::abs(Apply(c2, t3, T3) - 3.0 * t0 * t0), 1e-4);
}

// Non-uniform extrapolation reproduces polynomial VALUES at the target exactly
// (degree <= k-1 for k history nodes).
TEST(MultistepCoeffs, VariableStepExtrapolationIsExact)
{
   const double target = 0.30;
   const std::vector<double> hist = {0.17, 0.05, -0.10};

   auto g3 = ExtrapolationWeights(target, hist); // quadratic: exact to t^2
   EXPECT_NEAR(Apply(g3, hist, One), 1.0, 1e-12);
   EXPECT_NEAR(Apply(g3, hist, T1), target, 1e-12);
   EXPECT_NEAR(Apply(g3, hist, T2), target * target, 1e-12);
   EXPECT_GT(std::abs(Apply(g3, hist, T3) - target * target * target), 1e-4);

   const std::vector<double> hist2(hist.begin(), hist.end() - 1);
   auto g2 = ExtrapolationWeights(target, hist2); // linear: exact to t
   EXPECT_NEAR(Apply(g2, hist2, One), 1.0, 1e-12);
   EXPECT_NEAR(Apply(g2, hist2, T1), target, 1e-12);
}
