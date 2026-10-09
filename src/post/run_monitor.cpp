#include "post/run_monitor.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>

namespace incns
{

using namespace mfem;

RunMonitor::RunMonitor(const Parameters& p, MPI_Comm comm)
   : progress_(p.output.progress), t_final_(p.t_final)
{
   int rank = 0;
   MPI_Comm_rank(comm, &rank);
   root_ = (rank == 0);
   if (root_ && p.output.history)
   {
      std::filesystem::create_directories(p.output.path);
      history_.open(p.output.path + "/" + p.output.name + "_history.csv");
      history_ << "t,dt,c_d,c_l,iterations,substeps,elements,step_wall\n";
   }
}

RunMonitor::~RunMonitor() { FlushHistory(); }

double RunMonitor::Wall() const
{
   return started_ ? MPI_Wtime() - wall0_ : 0.0;
}

void RunMonitor::Record(const StepRecord& r)
{
   if (!started_)
   {
      started_ = true;
      wall0_ = MPI_Wtime() - r.step_wall;
      t0_ = r.t - r.dt;
      next_progress_ = (progress_ > 0.0)
                       ? (std::floor(t0_ / progress_ + 1e-9) + 1.0) * progress_ : 0.0;
   }
   last_ = r;
   ++steps_;
   dt_min_ = std::min(dt_min_, r.dt);
   dt_max_ = std::max(dt_max_, r.dt);
   win_dt_min_ = std::min(win_dt_min_, r.dt);
   win_dt_max_ = std::max(win_dt_max_, r.dt);
   step_wall_ += r.step_wall;
   if (r.has_forces)
   {
      forces_.Add(r.t, r.cd, r.cl);
      if (!std::isfinite(r.cd) || !std::isfinite(r.cl) || std::abs(r.cd) > 1e3 ||
          std::abs(r.cl) > 1e3)
      {
         diverged_ = true;
         std::ostringstream w;
         w << "force coefficients C_D = " << r.cd << ", C_L = " << r.cl
           << " at t = " << r.t;
         why_ = w.str();
      }
   }
   if (!std::isfinite(r.velocity_norm))
   {
      diverged_ = true;
      why_ = "non-finite velocity at t = " + std::to_string(r.t);
   }
   if (root_ && history_.is_open())
   {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%.12g,%.6e,%.10g,%.10g,%d,%d,%lld,%.4g\n",
                    r.t, r.dt, r.has_forces ? r.cd : NAN, r.has_forces ? r.cl : NAN,
                    r.iterations, r.substeps, r.elements, r.step_wall);
      pending_ += buf;
   }
}

bool RunMonitor::ProgressDue(double t) const
{
   return progress_ > 0.0 && started_ && t >= next_progress_ - 1e-12;
}

void RunMonitor::PrintProgress(double cfl, const Vector& where,
                               long long elements)
{
   next_progress_ += progress_;
   if (last_.t >= next_progress_) // a step longer than the interval
   {
      next_progress_ = (std::floor(last_.t / progress_ + 1e-9) + 1.0) * progress_;
   }
   if (root_)
   {
      const double wall = Wall();
      const double done = last_.t - t0_, left = t_final_ - last_.t;
      const double eta = (done > 0.0) ? wall * left / done : 0.0;
      std::string at;
      if (where.Size() >= 2)
      {
         char b[64];
         std::snprintf(b, sizeof(b), " @(%.2f,%.2f)", where(0), where(1));
         at = b;
      }
      std::string f;
      if (last_.has_forces)
      {
         char b[64];
         std::snprintf(b, sizeof(b), "  C_D %.4f  C_L %+.4f", last_.cd, last_.cl);
         f = b;
      }
      std::printf("t %8.3f  dt %.3e [%.3e, %.3e]  CFL %.2f%s%s  its %d  "
                  "elements %lld  steps %lld  wall %.0f s  ETA %.1f min\n",
                  last_.t, last_.dt, win_dt_min_, win_dt_max_, cfl, at.c_str(),
                  f.c_str(), last_.iterations, elements, steps_, wall,
                  eta / 60.0);
      std::fflush(stdout);
   }
   win_dt_min_ = std::numeric_limits<double>::infinity();
   win_dt_max_ = 0.0;
   FlushHistory();
}

void RunMonitor::FlushHistory()
{
   if (root_ && history_.is_open() && !pending_.empty())
   {
      history_ << pending_;
      history_.flush();
      pending_.clear();
   }
}

} // namespace incns
