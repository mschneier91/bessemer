/**
 * @file adaptive_controller.hpp
 * @brief PI (Gustafsson) step-size controller with mixed abs/rel tolerance,
 *        rejection handling, a pluggable dt ceiling, and step-history recording.
 */
#ifndef INCNS_TIME_ADAPTIVE_CONTROLLER_HPP
#define INCNS_TIME_ADAPTIVE_CONTROLLER_HPP

#include <functional>
#include <vector>

namespace incns
{

/// Options for the adaptive step-size controller.
struct AdaptiveControllerOptions
{
   /// Absolute tolerance. Setting it to ~1e-16 makes its term negligible,
   /// giving pure relative control (and vice versa for rtol).
   double atol = 1e-8;
   /// Relative tolerance, applied to the GLOBAL solution norm:
   /// accept when @c ||LTE|| <= atol + rtol*||u||. Deliberately the global
   /// form, never a per-DOF WRMS weight (which blows up where a velocity
   /// component crosses zero -- the 2D TGV vanishes on whole lines).
   double rtol = 1e-6;
   double safety = 0.9;       ///< Safety factor on every proposed step.
   double growth_cap = 2.0;   ///< Max dt growth ratio per step.
   double shrink_cap = 0.2;   ///< Min dt shrink ratio per step.
   double dt_min = 1e-14;     ///< Hard failure below this step size.
   int max_consecutive_rejections = 10; ///< Hard failure guard.
   /// Order of the local truncation error of the ADVANCING scheme (BDF2 has
   /// LTE O(dt^3), so 3): the controller exponents are 1/lte_order-scaled.
   double lte_order = 3.0;
};

/// One attempted step, as recorded by the controller.
struct StepAttempt
{
   double t = 0.0;         ///< Time at the start of the attempt.
   double dt = 0.0;        ///< Attempted step size.
   double lte_norm = 0.0;  ///< ||LTE|| estimate supplied for the attempt.
   double u_norm = 0.0;    ///< Solution norm used for the relative term.
   double threshold = 0.0; ///< atol + rtol*||u|| at the attempt.
   bool accepted = false;  ///< Outcome.
};

/**
 * @brief Accuracy-driven step-size controller (the stability side is the
 *        pluggable dt ceiling).
 *
 * Control law, with the scaled error @c es = ||LTE|| / (atol + rtol*||u||)
 * and @c p = lte_order:
 *  - accept when @c es <= 1;
 *  - accepted step with an accepted predecessor (PI, Soderlind PI.4.2):
 *    @c dt_new = safety * dt * es^(-0.6/p) * es_prev^(+0.2/p);
 *  - first controlled step, or retry after a rejection (elementary I control):
 *    @c dt_new = safety * dt * es^(-1/p).
 * The ratio @c dt_new/dt is clamped to [shrink_cap, growth_cap], and the
 * result to the pluggable ceiling (inert -- infinity -- in Sprint 1; Sprint 2
 * wires the convective-CFL bound; the LTE controller sees accuracy only and
 * would otherwise push dt past the explicit stability limit and thrash).
 *
 * Every attempt is recorded as a StepAttempt and exposed programmatically --
 * the adaptive-mode check consumes this record, never parsed logs. Purely
 * serial arithmetic: the caller supplies globally-reduced norms, so identical
 * decisions on every rank.
 */
class AdaptiveController
{
public:
   /// Ceiling callback signature: max admissible dt at time @p t.
   using DtCeilingFn = std::function<double(double t)>;

   /// @param opts Tolerances, caps, and control constants.
   explicit AdaptiveController(const AdaptiveControllerOptions& opts);

   /**
    * @brief Install the dt ceiling hook (stability, not accuracy).
    * @param ceiling Callback returning the cap; replaces the inert default.
    */
   void SetDtCeiling(DtCeilingFn ceiling) { ceiling_ = std::move(ceiling); }

   /**
    * @brief Judge an attempted step and update the proposed next step size.
    * @param t        Time at the start of the attempt.
    * @param dt       Attempted step size.
    * @param lte_norm Global norm of the LTE estimate (velocity only).
    * @param u_norm   Global norm of the velocity (for the relative term).
    * @return True if the step is accepted; NextDt() is the following step
    *         size. False if rejected; NextDt() is the retry step size.
    */
   bool Evaluate(double t, double dt, double lte_norm, double u_norm);

   /// @return The proposed next (or retry) step size.
   double NextDt() const { return next_dt_; }

   /// @return Every attempted step, in order (accepted and rejected).
   const std::vector<StepAttempt>& History() const { return history_; }

   /// @return Total rejected attempts so far.
   int TotalRejections() const { return total_rejections_; }

   /// @return Rejections since the last accepted step.
   int ConsecutiveRejections() const { return consecutive_rejections_; }

private:
   AdaptiveControllerOptions opts_;   ///< Options.
   DtCeilingFn ceiling_;              ///< dt cap hook (inert by default).
   std::vector<StepAttempt> history_; ///< All attempts.
   double next_dt_ = 0.0;             ///< Proposed next step size.
   double prev_es_ = -1.0;            ///< Scaled error of the last accepted step.
   int total_rejections_ = 0;         ///< Rejection count.
   int consecutive_rejections_ = 0;   ///< Rejections since last accept.
};

} // namespace incns

#endif // INCNS_TIME_ADAPTIVE_CONTROLLER_HPP
