/**
 * @file run_monitor.hpp
 * @brief Watches a run step by step: progress lines, the per-step history
 *        CSV, time-step statistics, force-coefficient statistics, and
 *        divergence. Case owns one; all ranks call it (only rank 0 writes).
 */
#pragma once

#include "config/parameters.hpp"
#include "post/force_statistics.hpp"
#include "mfem.hpp"

#include <fstream>
#include <limits>
#include <string>

namespace incns
{

/// What one accepted step produced (filled by Case::Step).
struct StepRecord
{
   double t = 0.0;          ///< Time after the step.
   double dt = 0.0;         ///< The step's size.
   int iterations = 0;      ///< Outer solver iterations.
   int substeps = 0;        ///< OIFS substeps (0 with IMEX).
   bool has_forces = false; ///< Whether cd / cl were evaluated.
   double cd = 0.0;         ///< Drag coefficient.
   double cl = 0.0;         ///< Lift coefficient.
   long long elements = 0;  ///< Global element count.
   double step_wall = 0.0;  ///< Wall time of the step.
   double velocity_norm = 0.0; ///< ||u|| (true dofs), for divergence checks.
};

/**
 * @brief Collects StepRecords and reports on them.
 */
class RunMonitor
{
public:
   /**
    * @param p    Parameters (output.progress / history, forces statistics,
    *             t_final).
    * @param comm Communicator (rank 0 writes).
    */
   RunMonitor(const Parameters& p, MPI_Comm comm);

   /// Close the history file.
   ~RunMonitor();

   /**
    * @brief Record a step: statistics, history, divergence check.
    * @param r The step.
    */
   void Record(const StepRecord& r);

   /**
    * @param t Time after the latest step.
    * @return Whether a progress line is due (identical on every rank).
    */
   bool ProgressDue(double t) const;

   /**
    * @brief Print a progress line for the latest step (rank 0) and flush the
    *        history.
    * @param cfl      Measured CFL number.
    * @param where    Where the CFL rate peaks (element centre).
    * @param elements Current global element count (after any AMR event of
    *                 the step).
    */
   void PrintProgress(double cfl, const mfem::Vector& where, long long elements);

   /// @return Whether the run diverged (non-finite or huge forces / velocity).
   bool Diverged() const { return diverged_; }

   /// @return The reason the run diverged (empty if it did not).
   const std::string& DivergenceReason() const { return why_; }

   /// @return Accepted steps recorded.
   long long Steps() const { return steps_; }
   /// @return Smallest step size recorded.
   double DtMin() const { return dt_min_; }
   /// @return Largest step size recorded.
   double DtMax() const { return dt_max_; }
   /// @return Sum of the steps' wall times.
   double StepWall() const { return step_wall_; }
   /// @return Wall time since the first recorded step.
   double Wall() const;
   /// @return The force-coefficient series (statistics on).
   const ForceStatistics& Forces() const { return forces_; }
   /// @return The latest record.
   const StepRecord& Last() const { return last_; }

   /// Write any history lines not yet written (rank 0).
   void FlushHistory();

private:
   bool root_;               ///< Rank 0 writes.
   double progress_;         ///< Progress interval (0 = off).
   double t_final_;          ///< For the ETA.
   double next_progress_ = 0.0; ///< Next progress time.
   bool started_ = false;    ///< A step has been recorded.
   double wall0_ = 0.0;      ///< Wall clock at the first record.
   double t0_ = 0.0;         ///< Time before the first recorded step.
   long long steps_ = 0;     ///< Steps recorded.
   double dt_min_ = std::numeric_limits<double>::infinity(); ///< Smallest dt.
   double dt_max_ = 0.0;     ///< Largest dt.
   double win_dt_min_ =
      std::numeric_limits<double>::infinity(); ///< Since the last line.
   double win_dt_max_ = 0.0; ///< Largest dt since the last progress line.
   double step_wall_ = 0.0;  ///< Sum of step wall times.
   bool diverged_ = false;   ///< Divergence detected.
   std::string why_;         ///< Why it diverged.
   StepRecord last_;         ///< Latest record.
   ForceStatistics forces_;  ///< C_D, C_L series (when records carry forces).
   std::ofstream history_;   ///< Per-step CSV (rank 0, when enabled).
   std::string pending_;     ///< History lines not yet written.
};

} // namespace incns
