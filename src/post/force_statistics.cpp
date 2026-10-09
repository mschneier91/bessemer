#include "post/force_statistics.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace incns
{

void ForceStatistics::Add(double t, double cd, double cl)
{
   MFEM_VERIFY(t_.empty()
               || t > t_.back(), "force_statistics: times must increase");
   t_.push_back(t);
   cd_.push_back(cd);
   cl_.push_back(cl);
}

double ForceStatistics::Integral(const std::vector<double>& f, double ta,
                                 double tb) const
{
   double sum = 0.0;
   for (std::size_t k = 1; k < t_.size(); ++k)
   {
      const double t0 = std::max(t_[k - 1], ta), t1 = std::min(t_[k], tb);
      if (t1 <= t0) { continue; }
      const double h = t_[k] - t_[k - 1];
      auto at = [&](double t)
      {
         const double w = (t - t_[k - 1]) / h;
         return (1.0 - w) * f[k - 1] + w * f[k];
      };
      sum += 0.5 * (t1 - t0) * (at(t0) + at(t1));
   }
   return sum;
}

namespace
{
// Maximum of samples f(t): the parabola through the discrete maximum and its
// neighbours (any spacing; CFL-controlled steps vary). Returns {t*, f*}.
std::pair<double, double> PeakOf(const std::vector<double>& t,
                                 const std::vector<double>& f)
{
   const std::size_t n = f.size();
   const std::size_t k = static_cast<std::size_t>(
                            std::max_element(f.begin(), f.end()) - f.begin());
   if (k == 0 || k + 1 >= n) { return {t[k], f[k]}; }
   const double t0 = t[k - 1], t1 = t[k], t2 = t[k + 1];
   const double d01 = (f[k] - f[k - 1]) / (t1 - t0);
   const double d12 = (f[k + 1] - f[k]) / (t2 - t1);
   const double a = (d12 - d01) / (t2 - t0); // second divided difference
   if (a >= 0.0) { return {t[k], f[k]}; }
   const double ts = 0.5 * (t0 + t1) - d01 / (2.0 * a);
   return {ts, f[k - 1] + d01* (ts - t0) + a* (ts - t0)* (ts - t1)};
}
} // namespace

ForceStats ForceStatistics::Compute(int n_periods) const
{
   ForceStats s;
   s.samples = Size();
   if (t_.empty()) { return s; }
   s.cd_final = cd_.back();
   s.cl_final = cl_.back();
   std::tie(s.t_cd_max, s.cd_max) = PeakOf(t_, cd_);
   std::tie(s.t_cl_max, s.cl_max) = PeakOf(t_, cl_);
   if (t_.size() < 3 || n_periods < 1) { return s; }

   // Up-crossings of C_L through its mean over the second half of the run.
   const double t_end = t_.back(), t_half = 0.5 * (t_.front() + t_end);
   const double level = Integral(cl_, t_half, t_end) / (t_end - t_half);
   std::vector<double> up;
   for (std::size_t k = 1; k < t_.size(); ++k)
   {
      const double a = cl_[k - 1] - level, b = cl_[k] - level;
      if (a < 0.0 && b >= 0.0)
      {
         up.push_back(t_[k - 1] + (t_[k] - t_[k - 1]) * (-a) / (b - a));
      }
   }
   const int n = std::min<int>(n_periods, static_cast<int>(up.size()) - 1);
   if (n < 2) { return s; }
   s.periodic = true;
   s.periods = n;
   s.t_start = up[up.size() - 1 - n];
   s.t_end = up.back();
   const double T = s.t_end - s.t_start;
   s.cd_mean = Integral(cd_, s.t_start, s.t_end) / T;
   s.cl_mean = Integral(cl_, s.t_start, s.t_end) / T;
   std::vector<double> cl2(cl_.size());
   for (std::size_t k = 0; k < cl_.size(); ++k) { cl2[k] = cl_[k] * cl_[k]; }
   s.cl_rms = std::sqrt(Integral(cl2, s.t_start, s.t_end) / T);
   s.period = T / n;
   double pmin = 1e300, pmax = 0.0;
   for (std::size_t k = up.size() - n; k < up.size(); ++k)
   {
      pmin = std::min(pmin, up[k] - up[k - 1]);
      pmax = std::max(pmax, up[k] - up[k - 1]);
   }
   s.period_spread = (pmax - pmin) / s.period;
   return s;
}

} // namespace incns
