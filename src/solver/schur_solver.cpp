#include "solver/schur_solver.hpp"

#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace incns
{

using namespace mfem;

namespace
{
// Velocity solves outside the Schur operator are corrections to an l2
// residual target: at most this many per solve.
constexpr int kMaxCorrections = 4;
// The finest relative tolerance a correction asks of the velocity Krylov
// solver.
constexpr double kMinCorrectionTol = 1e-13;

} // namespace

SchurSolver::SchurSolver(MPI_Comm comm, const Operator& A, const Operator& B,
                         Solver& a_pc, Solver& schur_pc,
                         const ConstantPressureProjector* proj,
                         const SchurSolveOptions& opts)
   : comm_(comm), A_(&A), B_(B), a_pc_(&a_pc), schur_pc_(&schur_pc),
     proj_(proj), opts_(opts), vel_(comm, A.Height()),
     r_(B.Height()), z_(B.Height()), d_(B.Height()), q_(B.Height())
{
   MFEM_VERIFY(opts_.max_cycles >= 1 && opts_.inner_max_iter >= 1,
               "schur_solver: max_cycles and inner_max_iter must be >= 1");
   MFEM_VERIFY(opts_.eps_tight > 0.0 && opts_.eps_tight <= opts_.eps_max &&
               opts_.eps_loose > 0.0 && opts_.relax_c > 0.0,
               "schur_solver: need 0 < eps_tight <= eps_max, eps_loose > 0, "
               "relax_c > 0");
   for (Vector* v : {&r_, &z_, &d_, &q_}) { v->UseDevice(true); }
   vel_.SetOperator(*A_);
   vel_.SetPreconditioner(*a_pc_);
   vel_.SetAbsTol(0.0);
   vel_.SetRelTol(opts_.eps_tight);
   vel_.SetMaxIter(opts_.inner_max_iter);
   sop_ = std::make_unique<SchurComplementOperator>(B_, vel_, proj_);
}

void SchurSolver::EnablePressureProjection(int k)
{
   projection_ = std::make_unique<SolutionProjection>(k, comm_);
   for (Vector* v : {&p_start_, &r_start_})
   {
      v->SetSize(B_.Height(), Device::GetDeviceMemoryType());
      v->UseDevice(true);
   }
}

void SchurSolver::Rewire(const Operator& A, Solver& a_pc, Solver& schur_pc)
{
   A_ = &A;
   a_pc_ = &a_pc;
   schur_pc_ = &schur_pc;
   // Re-pointed, not rebuilt: no allocation, and the preconditioner's own
   // SetOperator is not called (it was built for this operator).
   vel_.SetOperator(A);
   vel_.SetPreconditioner(a_pc);
}

double SchurSolver::MomentumResidual(const Vector& f, const Vector& p,
                                     const Vector& u) const
{
   Vector& r = sop_->RhsWork();
   Vector& t = sop_->VelocityWork();
   A_->Mult(u, r);
   B_.MultTranspose(p, t);
   r += t;
   subtract(f, r, r); // f - A u - B^T p (zero on the essential rows)
   return std::sqrt(Dot(r, r));
}

double SchurSolver::PressureResidual(const Vector& g, const Vector& u,
                                     Vector& r) const
{
   B_.Mult(u, r);
   r -= g; // B u - g: the residual of S p = B A^-1 f - g
   if (proj_) { proj_->Project(r); }
   return std::sqrt(Dot(r, r));
}

bool SchurSolver::SolveVelocity(const Vector& f, const Vector& p, Vector& u,
                                double target, double& res)
{
   INCNS_PROFILE("schur_solver::velocity_solve");
   const double t0 = MPI_Wtime();
   bool met = false;
   for (int pass = 0; ; ++pass)
   {
      res = MomentumResidual(f, p, u); // into RhsWork
      if (res <= target) { met = true; break; }
      if (pass == kMaxCorrections) { break; }
      // A correction du = A^-1 r to the residual target (the Krylov solver's
      // tolerance is relative to its own norm of r: half the ratio, floored).
      Vector& r = sop_->RhsWork();
      Vector& du = sop_->VelocityWork();
      vel_.SetRelTol(std::max(kMinCorrectionTol, 0.5 * target / res));
      vel_.Solve(r, du); // r is recomputed by the next pass
      ++stats_.velocity_solves;
      stats_.velocity_iterations += vel_.GetNumIterations();
      u += du;
   }
   stats_.velocity_time += MPI_Wtime() - t0;
   return met;
}

double SchurSolver::InnerRelTol(double rnorm, double tau_p) const
{
   switch (opts_.inner)
   {
      case InnerTolerance::Tight: return opts_.eps_tight;
      case InnerTolerance::Loose: return opts_.eps_loose;
      case InnerTolerance::Relaxed:
      default:
      {
         // eps_k = min(eps_max, c tau / ||r_{k-1}||), floored at eps_tight:
         // tight while the residual is large, loose once its corrections
         // carry small weights.
         const double e = (rnorm > 0.0) ? opts_.relax_c * tau_p / rnorm
                          : opts_.eps_max;
         return std::min(opts_.eps_max, std::max(opts_.eps_tight, e));
      }
   }
}

void SchurSolver::CgCycle(Vector& u, Vector& p, double rnorm, double tau_p,
                          const Vector& f, int cycle)
{
   INCNS_PROFILE("schur_solver::cg");
   const bool root = Mpi::Root();
   // z = P_S^-1 r. On singular domains CC projects its inner solve, not its
   // mass term: project z too, so the directions carry no constant.
   schur_pc_->Mult(r_, z_);
   if (proj_) { proj_->Project(z_); }
   d_ = z_;
   double rz = Dot(r_, z_);
   while (stats_.iterations < max_iter_)
   {
      const double eps = InnerRelTol(rnorm, tau_p);
      sop_->SetInnerTolerance(eps);
      sop_->Mult(d_, q_); // q = S d; v = A^-1 B^T d kept
      const double dq = Dot(d_, q_);
      if (!(dq > 0.0))
      {
         stats_.breakdown = true;
         if (root && print_level_ >= 0)
         {
            mfem::out << "[incns] schur_solver: FCG breakdown, (d, S d) = "
                      << dq << std::endl;
         }
         break;
      }
      const double gamma = rz / dq;
      p.Add(gamma, d_);
      u.Add(-gamma, sop_->LastVelocity()); // u stays A^-1 (f - B^T p)
      r_.Add(-gamma, q_);
      if (proj_) { proj_->Project(r_); }
      rnorm = std::sqrt(Dot(r_, r_));
      ++stats_.iterations;

      if (opts_.history)
      {
         SchurIterationRecord rec;
         rec.cycle = cycle;
         rec.iteration = stats_.iterations;
         rec.residual = rnorm;
         rec.inner_tolerance = eps;
         rec.inner_iterations = sop_->LastIterations();
         // The coupled residual of the carried iterate: the pressure part is
         // r exactly, the momentum part the inner solves' accumulated error.
         const double m = MomentumResidual(f, p, u);
         rec.coupled_residual = std::sqrt(m * m + rnorm * rnorm);
         history_.push_back(rec);
      }
      if (root && print_level_ >= 1)
      {
         mfem::out << "   Schur FCG cycle "
                   << cycle << " iteration " << stats_.iterations
                   << "  ||r|| = " << rnorm << "  inner tol " << eps << " ("
                   << sop_->LastIterations() << " its)" << std::endl;
      }
      if (rnorm <= tau_p) { break; }

      schur_pc_->Mult(r_, z_);
      if (proj_) { proj_->Project(z_); }
      // Flexible CG (Notay 2000): Polak-Ribiere beta = (z_new, r_new - r) /
      // (r, z), with r_new - r = -gamma q: no extra vector. It reduces to
      // plain CG's Fletcher-Reeves beta when P_S^-1 is fixed and linear, and
      // tolerates CC's fixed-count inner CG and the inexact S.
      const double beta = -gamma * Dot(z_, q_) / rz;
      d_ *= beta;
      d_ += z_;
      rz = Dot(r_, z_);
   }
}

void SchurSolver::Mult(const Vector& b, Vector& x)
{
   INCNS_PROFILE("schur_solver");
   const int n_u = A_->Height();
   const int n_p = B_.Height();
   MFEM_VERIFY(b.Size() == n_u + n_p && x.Size() == n_u + n_p,
               "schur_solver: the vectors do not match the blocks");
   Vector f, g, u, p;
   f.MakeRef(const_cast<Vector&>(b), 0, n_u);
   g.MakeRef(const_cast<Vector&>(b), n_u, n_p);
   u.MakeRef(x, 0, n_u);
   p.MakeRef(x, n_u, n_p);
   // Aliases on device memory follow the base vector's validity.
   for (Vector* v : {&f, &g, &u, &p}) { v->UseDevice(true); }
   f.SyncMemory(b);
   g.SyncMemory(b);
   u.SyncMemory(x);
   p.SyncMemory(x);

   stats_ = SchurSolveStats();
   history_.clear();
   sop_->ResetCounters();
   converged_ = false;

   // tau as FGMRES's: max(atol, rtol ||b - K x_0||).
   double tau = atol_;
   if (rtol_ > 0.0)
   {
      const double m0 = MomentumResidual(f, p, u);
      const double p0 = PressureResidual(g, u, r_);
      tau = std::max(tau, rtol_ * std::sqrt(m0 * m0 + p0 * p0));
   }
   stats_.tolerance = tau;
   double tau_p = tau / std::sqrt(2.0);
   const double tau_u = tau / std::sqrt(2.0);

   // System 1: u = A^-1 (f - B^T p), from the warm start.
   double res_u = 0.0;
   bool vel_ok = SolveVelocity(f, p, u, tau_u, res_u);

   // The pressure projection: p += X W^T r, r -= W W^T r (its image). u is
   // then offset by A^-1 B^T times that correction until System 3.
   double rnorm = PressureResidual(g, u, r_);
   bool fresh = true; // r_ holds the residual of the current (u, p)
   // The pressure's own target (human 2026-10-17): the constraint residual
   // this step leaves after the velocity solve with the old pressure, reduced
   // by p_reduction_, never looser than the coupled target. With the coupled
   // target alone, a reference dominated by the momentum residual (explicit
   // convection moving the right-hand side) let every solve end right here,
   // the pressure never updated after the first step, and a wall-bounded IMEX
   // run diverged (the curved cylinder, 2026-10-17).
   if (p_reduction_ > 0.0)
   {
      tau_p = std::min(tau_p, std::max(atol_ / std::sqrt(2.0),
                                       p_reduction_ * rnorm));
   }
   stats_.pressure_tolerance = tau_p;
   if (projection_)
   {
      p_start_ = p;
      r_start_ = r_;
      stats_.start_residual = rnorm;
      if (projection_->Guess(r_start_, d_, &q_))
      {
         p += d_;
         r_ -= q_;
         if (proj_) { proj_->Project(r_); }
         rnorm = std::sqrt(Dot(r_, r_));
         stats_.projected = true;
         vel_ok = false; // u is not A^-1 (f - B^T p) any more
      }
      stats_.guess_residual = rnorm;
   }

   for (int cycle = 0; ; ++cycle)
   {
      // The TRUE pressure residual of (u, p), u solved from p.
      if (!fresh) { rnorm = PressureResidual(g, u, r_); }
      fresh = false;
      if (rnorm <= tau_p && vel_ok) { converged_ = true; break; }
      if (cycle == opts_.max_cycles || stats_.iterations >= max_iter_ ||
          stats_.breakdown) { break; }
      ++stats_.cycles;
      // A projected guess may already meet the pressure target: then only
      // the velocity is left to solve.
      if (rnorm > tau_p)
      {
         CgCycle(u, p, rnorm, tau_p, f, cycle);
      }
      // System 3: u = A^-1 (f - B^T p), from the carried u.
      vel_ok = SolveVelocity(f, p, u, tau_u, res_u);
   }

   // The increment and its image S (p - p_n) = r_n - r, both residuals true.
   if (projection_ && converged_)
   {
      d_ = p;
      d_ -= p_start_;
      q_ = r_start_;
      q_ -= r_;
      projection_->AddWithImage(d_, q_);
   }

   stats_.momentum_residual = res_u;
   stats_.pressure_residual = rnorm;
   stats_.final_residual = std::sqrt(res_u * res_u + rnorm * rnorm);
   stats_.schur_applications = sop_->Applications();
   stats_.schur_inner_iterations = sop_->InnerIterations();
   stats_.schur_cap_hits = sop_->CapHits();
   stats_.schur_inner_time = sop_->InnerTime();
   if (Mpi::Root() && print_level_ >= 0)
   {
      mfem::out << "   Schur solve: " << stats_.iterations << " iterations, "
                << stats_.cycles << " cycles, coupled residual "
                << stats_.final_residual << " (tolerance " << tau << ")"
                << (converged_ ? "" : " NOT CONVERGED") << std::endl;
   }
   // Write the aliases back into the base vector's memory.
   u.SyncAliasMemory(x);
   p.SyncAliasMemory(x);
}

} // namespace incns
