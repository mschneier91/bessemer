#include "time/adaptive_controller.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <algorithm>
#include <cmath>
#include <limits>

namespace incns
{

AdaptiveController::AdaptiveController(const AdaptiveControllerOptions& opts)
   : opts_(opts),
     ceiling_([](double) { return std::numeric_limits<double>::infinity(); })
{
   MFEM_VERIFY(opts_.atol >= 0.0 && opts_.rtol >= 0.0 &&
               (opts_.atol > 0.0 || opts_.rtol > 0.0),
               "adaptive_controller: need a positive atol or rtol");
   MFEM_VERIFY(opts_.safety > 0.0 && opts_.safety < 1.0,
               "adaptive_controller: safety must be in (0,1)");
   MFEM_VERIFY(opts_.shrink_cap > 0.0 && opts_.shrink_cap < 1.0 &&
               opts_.growth_cap > 1.0,
               "adaptive_controller: caps must satisfy 0 < shrink < 1 < growth");
   MFEM_VERIFY(opts_.lte_order > 0.0, "adaptive_controller: bad lte_order");
}

bool AdaptiveController::Evaluate(double t, double dt, double lte_norm,
                                  double u_norm)
{
   MFEM_VERIFY(dt > 0.0 && lte_norm >= 0.0 && u_norm >= 0.0,
               "adaptive_controller: bad Evaluate arguments");

   const double threshold = opts_.atol + opts_.rtol * u_norm;
   // Floor the scaled error so an exactly-resolved step still yields a finite
   // growth proposal (then the growth cap governs).
   const double es = std::max(lte_norm / threshold, 1e-14);
   const bool accept = (lte_norm <= threshold);

   history_.push_back({t, dt, lte_norm, u_norm, threshold, accept});

   const double p = opts_.lte_order;
   double factor;
   if (accept && prev_es_ > 0.0)
   {
      // PI (Soderlind PI.4.2): smooth step sequences, less ringing than the
      // elementary controller.
      factor = opts_.safety * std::pow(es, -0.6 / p) *
               std::pow(prev_es_, 0.2 / p);
   }
   else
   {
      // First controlled step, or retry after a rejection: elementary control.
      factor = opts_.safety * std::pow(es, -1.0 / p);
   }
   factor = std::clamp(factor, opts_.shrink_cap, opts_.growth_cap);

   double dt_new = factor * dt;
   dt_new = std::min(dt_new, ceiling_(t)); // stability hook (inert in Sprint 1)
   MFEM_VERIFY(dt_new >= opts_.dt_min,
               "adaptive_controller: step size collapsed below dt_min");
   next_dt_ = dt_new;

   if (accept)
   {
      prev_es_ = es;
      consecutive_rejections_ = 0;
   }
   else
   {
      ++total_rejections_;
      ++consecutive_rejections_;
      MFEM_VERIFY(consecutive_rejections_ <= opts_.max_consecutive_rejections,
                  "adaptive_controller: too many consecutive rejections");
   }
   return accept;
}

} // namespace incns
