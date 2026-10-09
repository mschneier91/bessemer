/**
 * @file force_statistics.hpp
 * @brief Statistics of a force-coefficient time series: whole-run extrema,
 *        final values, and averages over an integer number of shedding
 *        periods (mean drag, mean and rms lift, the period), as the vortex
 *        shedding benchmarks report them.
 */
#pragma once

#include <vector>

namespace incns
{

/// Statistics of a C_D(t), C_L(t) series (ForceStatistics::Compute).
struct ForceStats
{
   int samples = 0;        ///< Samples recorded.
   double cd_final = 0.0;  ///< C_D at the last sample.
   double cl_final = 0.0;  ///< C_L at the last sample.
   /// Largest C_D over the run (parabola through the discrete maximum and
   /// its neighbours, as the DFG 2D-3 benchmark's peak times need).
   double cd_max = 0.0;
   double t_cd_max = 0.0;  ///< Its time.
   double cl_max = 0.0;    ///< Largest C_L over the run (same refinement).
   double t_cl_max = 0.0;  ///< Its time.
   /// Whether enough lift periods were found to average over (>= 2).
   bool periodic = false;
   int periods = 0;           ///< Periods averaged over.
   double t_start = 0.0;      ///< Averaging window start (a C_L up-crossing).
   double t_end = 0.0;        ///< Averaging window end (a C_L up-crossing).
   double cd_mean = 0.0;      ///< Time-mean C_D over the window.
   double cl_mean = 0.0;      ///< Time-mean C_L over the window.
   double cl_rms = 0.0;       ///< rms C_L over the window.
   double period = 0.0;       ///< Mean shedding period over the window.
   double period_spread = 0.0; ///< (longest - shortest period) / mean period.
};

/**
 * @brief Accumulates (t, C_D, C_L) samples and computes ForceStats.
 *
 * Periods are found from up-crossings of C_L through its mean over the second
 * half of the run; the averages cover the last N whole periods, integrated in
 * time by the trapezoid rule on the (possibly non-uniform) samples. The same
 * algorithm the square-cylinder benchmark was validated with.
 */
class ForceStatistics
{
public:
   /**
    * @brief Record one sample (times must increase).
    * @param t  Time.
    * @param cd Drag coefficient.
    * @param cl Lift coefficient.
    */
   void Add(double t, double cd, double cl);

   /**
    * @brief The statistics of the samples so far.
    * @param n_periods Periods to average over (the last ones); fewer when the
    *                  run has fewer.
    * @return The statistics (periodic = false when < 2 periods exist).
    */
   ForceStats Compute(int n_periods) const;

   /// @return Number of samples recorded.
   int Size() const { return static_cast<int>(t_.size()); }

private:
   /**
    * @brief Time integral of a sampled quantity (trapezoid rule, linear
    *        interpolation at the window ends).
    * @param f  Samples at the recorded times.
    * @param ta Window start.
    * @param tb Window end.
    * @return The integral over [ta, tb].
    */
   double Integral(const std::vector<double>& f, double ta, double tb) const;

   std::vector<double> t_;  ///< Sample times.
   std::vector<double> cd_; ///< C_D samples.
   std::vector<double> cl_; ///< C_L samples.
};

} // namespace incns
