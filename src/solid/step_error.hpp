/**
 * @file step_error.hpp
 * @brief Local error estimates of the solid's time steps, for its
 *        error-controlled steps (docs/elasticity.md section 2): the
 *        generalized-alpha estimate of Zienkiewicz and Xie, and BDF's
 *        predictor-corrector difference (Milne's device). Neither needs a
 *        solve of its own.
 */
#pragma once

#include <cmath>
#include <vector>

namespace incns
{

/**
 * @brief Zienkiewicz and Xie (1991): a Newmark-type step's local error in the
 *        displacement is e = (beta - 1/6) dt^2 (a^{n+1} - a^n), third order.
 * @param beta The scheme's beta (generalized-alpha's).
 * @param dt   The step.
 * @return |beta - 1/6| dt^2.
 */
inline double NewmarkErrorFactor(double beta, double dt)
{
   return std::abs(beta - 1.0 / 6.0) * dt * dt;
}

/**
 * @param x The nodes t_j of a polynomial (distinct).
 * @param t Where it is evaluated.
 * @return The Lagrange weights w_j: p(t) = sum_j w_j y_j.
 */
inline std::vector<double> ExtrapolationWeights(const std::vector<double>& x,
      double t)
{
   const std::size_t m = x.size();
   std::vector<double> w(m, 1.0);
   for (std::size_t j = 0; j < m; ++j)
   {
      for (std::size_t i = 0; i < m; ++i)
      {
         if (i != j) { w[j] *= (t - x[i]) / (x[j] - x[i]); }
      }
   }
   return w;
}

/**
 * @brief Milne's device for a BDF step of order q: the corrector u_c (the
 *        step's solution) and the predictor u_p (the degree-q polynomial
 *        through the q + 1 previous levels, at the new time) err by E_c and
 *        E_p times dt^{q+1} u^{(q+1)}, so the step's local error is
 *        E_c / (E_c - E_p) (u_c - u_p). With constant steps 2/11 for BDF2.
 * @param c     The BDF derivative weights at times[0..q] (BdfWeights).
 * @param times t^{n+1}, t^n, ..., t^{n-q}: q + 2 times.
 * @return E_c / (E_c - E_p).
 */
inline double BdfMilneFactor(const std::vector<double>& c,
                             const std::vector<double>& times)
{
   const int q = static_cast<int>(c.size()) - 1;
   double fact = 1.0;
   for (int k = 2; k <= q + 1; ++k) { fact *= k; }
   // The corrector: sum_j c_j y(t_j) - y'(t_0) = R y^{(q+1)} for y = (t -
   // t_0)^{q+1} / (q+1)!, and c_0 (u_c - u) = -R y^{(q+1)}.
   double r = 0.0;
   for (int j = 1; j <= q; ++j)
   {
      r += c[j] * std::pow(times[j] - times[0], q + 1) / fact;
   }
   const double ec = -r / c[0];
   // The predictor: u_p - u = -prod_j (t_0 - t_j) / (q+1)! u^{(q+1)}.
   double prod = 1.0;
   for (int j = 1; j <= q + 1; ++j) { prod *= times[0] - times[j]; }
   const double ep = -prod / fact;
   return ec / (ec - ep);
}

} // namespace incns
