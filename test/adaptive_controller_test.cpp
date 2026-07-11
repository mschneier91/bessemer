// Sprint 1.9 unit test -- adaptive controller (pure logic, synthetic LTE
// sequences; no solver, no stepping): expected step-size updates from the
// documented control law, rejections, the abs/rel disable-by-tiny-tol
// behavior, the dt-ceiling hook, and the step-history record.

#include <gtest/gtest.h>

#include "time/adaptive_controller.hpp"

#include <cmath>

using incns::AdaptiveController;
using incns::AdaptiveControllerOptions;

namespace
{
AdaptiveControllerOptions AbsOnly(double atol)
{
   AdaptiveControllerOptions o;
   o.atol = atol;
   o.rtol = 1e-16; // relative term effectively disabled
   return o;
}
} // namespace

// First controlled step uses the elementary law dt*safety*es^{-1/3}; large
// proposals are clamped by the growth cap.
TEST(AdaptiveController, ElementaryLawAndGrowthCap)
{
   AdaptiveController c(AbsOnly(1e-6));

   // es = 1e-3: raw factor 0.9 * (1e-3)^{-1/3} = 9 -> clamped to 2.
   EXPECT_TRUE(c.Evaluate(0.0, 0.1, 1e-9, 1.0));
   EXPECT_NEAR(c.NextDt(), 0.2, 1e-14);

   // es = 0.5: factor 0.9 * 0.5^{-1/3}... but this is now an accepted step
   // with an accepted predecessor -> PI law (checked in PiLaw test).
}

// Rejection: es > 1 shrinks via the elementary law and is recorded.
TEST(AdaptiveController, RejectionShrinks)
{
   AdaptiveController c(AbsOnly(1e-6));

   // es = lte/threshold (threshold = atol + rtol*u); elementary retry law.
   const double thr = 1e-6 + 1e-16 * 1.0;
   EXPECT_FALSE(c.Evaluate(0.0, 0.1, 8e-6, 1.0));
   const double expected = 0.9 * 0.1 * std::pow(8e-6 / thr, -1.0 / 3.0);
   EXPECT_NEAR(c.NextDt(), expected, 1e-14);
   EXPECT_EQ(c.TotalRejections(), 1);
   EXPECT_EQ(c.ConsecutiveRejections(), 1);

   // Successful retry resets the consecutive counter.
   EXPECT_TRUE(c.Evaluate(0.0, 0.045, 1e-7, 1.0));
   EXPECT_EQ(c.TotalRejections(), 1);
   EXPECT_EQ(c.ConsecutiveRejections(), 0);
}

// Accepted step with an accepted predecessor uses the PI (Soderlind PI.4.2)
// law: dt_new = safety * dt * es^{-0.6/p} * es_prev^{0.2/p}, p = 3.
TEST(AdaptiveController, PiLaw)
{
   AdaptiveController c(AbsOnly(1e-6));

   // es = lte/(atol + rtol*u) exactly as documented.
   const double thr = 1e-6 + 1e-16 * 1.0;
   const double es = 5e-7 / thr;

   // Step 1 (elementary): factor f1 = 0.9 * es^{-1/3}.
   EXPECT_TRUE(c.Evaluate(0.0, 0.1, 5e-7, 1.0));
   const double f1 = 0.9 * std::pow(es, -1.0 / 3.0);
   const double dt1 = 0.1 * f1;
   EXPECT_NEAR(c.NextDt(), dt1, 1e-14);

   // Step 2 (PI): same es, es_prev = es.
   EXPECT_TRUE(c.Evaluate(0.1, dt1, 5e-7, 1.0));
   const double f2 =
      0.9 * std::pow(es, -0.6 / 3.0) * std::pow(es, 0.2 / 3.0);
   EXPECT_NEAR(c.NextDt(), dt1 * f2, 1e-14);
}

// Mixed tolerance: the acceptance threshold is atol + rtol*||u||; a tiny
// value on either side disables that term.
TEST(AdaptiveController, MixedToleranceAndDisabling)
{
   // Pure absolute (rtol ~ 0): a huge ||u|| does not loosen the threshold.
   {
      AdaptiveController c(AbsOnly(1e-3));
      EXPECT_FALSE(c.Evaluate(0.0, 0.1, 2e-3, /*u_norm=*/1e6));
      EXPECT_TRUE(c.Evaluate(0.0, 0.05, 0.9e-3, 1e6));
   }
   // Pure relative (atol ~ 0): the threshold scales with ||u||.
   {
      AdaptiveControllerOptions o;
      o.atol = 1e-16;
      o.rtol = 1e-6;
      AdaptiveController c(o);
      EXPECT_TRUE(c.Evaluate(0.0, 0.1, 0.5, /*u_norm=*/1e6));  // thr ~ 1
      EXPECT_FALSE(c.Evaluate(0.0, 0.1, 0.5, /*u_norm=*/1e3)); // thr ~ 1e-3
   }
   // Both active: threshold = atol + rtol*||u||.
   {
      AdaptiveControllerOptions o;
      o.atol = 1e-3;
      o.rtol = 1e-3;
      AdaptiveController c(o);
      EXPECT_TRUE(c.Evaluate(0.0, 0.1, 1.5e-3, /*u_norm=*/1.0)); // thr = 2e-3
      EXPECT_FALSE(c.Evaluate(0.0, 0.1, 2.5e-3, 1.0));
   }
}

// The pluggable dt ceiling caps every proposal (stability hook -- inert by
// default, wired to the convective CFL in Sprint 2).
TEST(AdaptiveController, DtCeilingHook)
{
   AdaptiveController c(AbsOnly(1e-6));
   c.SetDtCeiling([](double) { return 0.05; });

   EXPECT_TRUE(c.Evaluate(0.0, 0.1, 1e-12, 1.0)); // wants strong growth
   EXPECT_NEAR(c.NextDt(), 0.05, 1e-14);          // capped
}

// Every attempt is recorded, in order, with the data the adaptive-mode check
// consumes (never reconstructed from logs).
TEST(AdaptiveController, HistoryRecord)
{
   AdaptiveController c(AbsOnly(1e-6));
   c.Evaluate(0.0, 0.1, 8e-6, 2.0);   // reject
   c.Evaluate(0.0, 0.045, 1e-7, 2.0); // accept (retry)
   c.Evaluate(0.045, c.NextDt(), 1e-7, 2.0); // accept

   const auto& h = c.History();
   ASSERT_EQ(h.size(), 3u);
   EXPECT_FALSE(h[0].accepted);
   EXPECT_TRUE(h[1].accepted);
   EXPECT_TRUE(h[2].accepted);
   EXPECT_DOUBLE_EQ(h[0].t, 0.0);
   EXPECT_DOUBLE_EQ(h[0].dt, 0.1);
   EXPECT_DOUBLE_EQ(h[0].lte_norm, 8e-6);
   EXPECT_DOUBLE_EQ(h[0].u_norm, 2.0);
   EXPECT_GT(h[2].t, h[1].t); // time advanced after the accepted retry
}
