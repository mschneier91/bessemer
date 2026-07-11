/**
 * @file multistep_coeffs.hpp
 * @brief Variable-step BDF and AB/EXT multistep coefficients.
 */
#ifndef INCNS_TIME_MULTISTEP_COEFFS_HPP
#define INCNS_TIME_MULTISTEP_COEFFS_HPP

#include <vector>

namespace incns
{

/**
 * @brief BDF (backward differentiation) weights on arbitrary node times.
 *
 * Given node times @c {t_{n+1}, t_n, ..., t_{n+1-k}} (strictly decreasing,
 * newest first), returns weights @c c such that
 * @code
 *   u'(t_{n+1}) ~= sum_j c[j] * u(times[j])
 * @endcode
 * exactly for polynomials of degree <= k (Lagrange differentiation at the
 * newest node). The implicit BDF factor of the momentum block is
 * @c beta0/dt = c[0].
 *
 * Under non-uniform steps the weights depend on the step-size ratios; they are
 * recomputed from the actual times each step -- NEVER reuse the uniform-step
 * values when dt varies (the classic adaptive-BDF bug; it silently drops order
 * and can destabilize).
 *
 * Uniform-step specializations (step h): BDF1 c*h = {1, -1};
 * BDF2 c*h = {3/2, -2, 1/2}; BDF3 c*h = {11/6, -3, 3/2, -1/3}.
 *
 * @param times Node times, newest first, strictly decreasing; size k+1 with
 *              1 <= k <= 3 (the orders this project uses).
 * @return Weights c, one per node, in the order of @p times.
 */
std::vector<double> BdfWeights(const std::vector<double>& times);

/**
 * @brief Extrapolation (AB/EXT) weights to a target time from history nodes.
 *
 * Given history times @c {t_n, t_{n-1}, ..., t_{n+1-k}} (strictly decreasing)
 * and a target @c t_target > times[0], returns weights @c g such that
 * @code
 *   u(t_target) ~= sum_j g[j] * u(times[j])
 * @endcode
 * exactly for polynomials of degree <= k-1 (Lagrange evaluation at the target).
 * This is the explicit half of the IMEX pairing: in Sprint 2 it extrapolates
 * the convection term to t^{n+1} (dormant for Stokes -- pure BDF).
 *
 * Uniform-step specializations: EXT1/AB1 {1}; EXT2/AB2 {2, -1};
 * EXT3/AB3 {3, -3, 1}.
 *
 * @param t_target Time to extrapolate to (beyond the newest history node).
 * @param times    History node times, newest first, strictly decreasing;
 *                 size k with 1 <= k <= 3.
 * @return Weights g, one per history node, in the order of @p times.
 */
std::vector<double> ExtrapolationWeights(double t_target,
      const std::vector<double>& times);

} // namespace incns

#endif // INCNS_TIME_MULTISTEP_COEFFS_HPP
