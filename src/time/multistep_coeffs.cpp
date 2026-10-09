#include "time/multistep_coeffs.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <cmath>

namespace incns
{

namespace
{
void VerifyDecreasing(const std::vector<double>& times, std::size_t min_size,
                      std::size_t max_size, const char* who)
{
   MFEM_VERIFY(times.size() >= min_size && times.size() <= max_size,
               "multistep_coeffs: unsupported node count for " << who);
   for (std::size_t i = 1; i < times.size(); ++i)
   {
      MFEM_VERIFY(times[i] < times[i - 1],
                  "multistep_coeffs: times must be strictly decreasing");
   }
}
} // namespace

std::vector<double> BdfWeights(const std::vector<double>& times)
{
   // k = order = times.size() - 1 in [1, 3].
   VerifyDecreasing(times, 2, 4, "BdfWeights");
   const std::size_t m = times.size();
   const double t0 = times[0];

   // c_j = L_j'(t0) for the Lagrange basis L_j on the nodes.
   //   j = 0:  L_0'(t0) = sum_{i != 0} 1/(t0 - t_i)
   //   j != 0: L_j'(t0) = [prod_{i != 0, j} (t0 - t_i)/(t_j - t_i)] / (t_j - t0)
   std::vector<double> c(m, 0.0);
   for (std::size_t i = 1; i < m; ++i) { c[0] += 1.0 / (t0 - times[i]); }
   for (std::size_t j = 1; j < m; ++j)
   {
      double prod = 1.0;
      for (std::size_t i = 1; i < m; ++i)
      {
         if (i == j) { continue; }
         prod *= (t0 - times[i]) / (times[j] - times[i]);
      }
      c[j] = prod / (times[j] - t0);
   }
   return c;
}

std::vector<double> ExtrapolationWeights(double t_target,
      const std::vector<double>& times)
{
   VerifyDecreasing(times, 1, 3, "ExtrapolationWeights");
   MFEM_VERIFY(t_target > times[0],
               "multistep_coeffs: extrapolation target must be ahead of the "
               "newest history node");
   const std::size_t m = times.size();

   // g_j = L_j(t_target) on the history nodes.
   std::vector<double> g(m, 1.0);
   for (std::size_t j = 0; j < m; ++j)
   {
      for (std::size_t i = 0; i < m; ++i)
      {
         if (i == j) { continue; }
         g[j] *= (t_target - times[i]) / (times[j] - times[i]);
      }
   }
   return g;
}

std::vector<double> LagrangeWeights(double t, const std::vector<double>& nodes)
{
   const std::size_t m = nodes.size();
   MFEM_VERIFY(m >= 1, "multistep_coeffs: LagrangeWeights needs a node");
   std::vector<double> w(m, 1.0);
   for (std::size_t l = 0; l < m; ++l)
   {
      for (std::size_t q = 0; q < m; ++q)
      {
         if (q == l) { continue; }
         const double d = nodes[l] - nodes[q];
         MFEM_VERIFY(std::abs(d) > 0.0, "multistep_coeffs: LagrangeWeights "
                     "needs distinct nodes");
         w[l] *= (t - nodes[q]) / d;
      }
   }
   return w;
}

} // namespace incns
