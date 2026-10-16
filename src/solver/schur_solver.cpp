#include "solver/schur_solver.hpp"

#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
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

// Hands an iterative solver's residual norms, as it reports them after each
// iteration (and at the start), to a callback.
class ResidualHook : public IterativeSolverMonitor
{
public:
   explicit ResidualHook(std::function<void(int, double)> f)
      : f_(std::move(f)) {}

   void MonitorResidual(int it, real_t norm, const Vector&, bool final) override
   {
      if (!final) { f_(it, norm); }
   }

private:
   std::function<void(int, double)> f_;
};
} // namespace

SchurSolver::SchurSolver(MPI_Comm comm, const Operator& A, const Operator& B,
                         Solver& a_pc, bool symmetric_a, Solver& schur_pc,
                         const ConstantPressureProjector* proj,
                         const SchurSolveOptions& opts)
   : comm_(comm), A_(&A), B_(B), a_pc_(&a_pc), symmetric_a_(symmetric_a),
     schur_pc_(&schur_pc), proj_(proj), opts_(opts),
     r_(B.Height()), z_(B.Height()), d_(B.Height()), q_(B.Height())
{
   MFEM_VERIFY(opts_.method == SchurKrylov::FGMRES || symmetric_a_,
               "schur_solver: with the rotation term S is nonsymmetric: "
               "solver.uzawa.method fgmres (cg and flexible_cg need a "
               "symmetric momentum block)");
   MFEM_VERIFY(opts_.max_cycles >= 1 && opts_.inner_max_iter >= 1 &&
               opts_.kdim >= 1 && opts_.inner_kdim >= 1,
               "schur_solver: max_cycles, inner_max_iter, kdim and inner_kdim "
               "must be >= 1");
   MFEM_VERIFY(opts_.eps_tight > 0.0 && opts_.eps_tight <= opts_.eps_max &&
               opts_.eps_loose > 0.0 && opts_.relax_c > 0.0,
               "schur_solver: need 0 < eps_tight <= eps_max, eps_loose > 0, "
               "relax_c > 0");
   for (Vector* v : {&r_, &z_, &d_, &q_}) { v->UseDevice(true); }
   BuildVelocitySolver();
   sop_ = std::make_unique<SchurComplementOperator>(B_, *vel_, proj_);
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

void SchurSolver::BuildVelocitySolver()
{
   std::unique_ptr<IterativeSolver> s;
   if (symmetric_a_) { s = std::make_unique<CGSolver>(comm_); }
   else
   {
      auto g = std::make_unique<GMRESSolver>(comm_);
      g->SetKDim(opts_.inner_kdim);
      s = std::move(g);
   }
   // SetOperator before SetPreconditioner: MFEM forwards SetOperator to a PC
   // that is already set, and the velocity PCs keep their own operator.
   s->SetOperator(*A_);
   s->SetPreconditioner(*a_pc_);
   s->SetAbsTol(0.0);
   s->SetRelTol(opts_.eps_tight);
   s->SetMaxIter(opts_.inner_max_iter);
   s->SetPrintLevel(-1);
   s->iterative_mode = false;
   // The Schur operator borrows the solver: re-point it before the old one
   // is destroyed.
   if (sop_) { sop_->SetVelocitySolver(*s); }
   vel_ = std::move(s);
}

void SchurSolver::Rewire(const Operator& A, Solver& a_pc, Solver& schur_pc)
{
   A_ = &A;
   a_pc_ = &a_pc;
   schur_pc_ = &schur_pc;
   BuildVelocitySolver();
   if (fgmres_) { fgmres_->SetPreconditioner(*schur_pc_); }
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
      vel_->SetRelTol(std::max(kMinCorrectionTol, 0.5 * target / res));
      du = 0.0;
      vel_->Mult(r, du);
      ++stats_.velocity_solves;
      stats_.velocity_iterations += vel_->GetNumIterations();
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
   const bool flexible = (opts_.method == SchurKrylov::FlexibleCG);
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
            mfem::out << "[incns] schur_solver: CG breakdown, (d, S d) = "
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
         mfem::out << "   Schur " << (flexible ? "FCG" : "CG") << " cycle "
                   << cycle << " iteration " << stats_.iterations
                   << "  ||r|| = " << rnorm << "  inner tol " << eps << " ("
                   << sop_->LastIterations() << " its)" << std::endl;
      }
      if (rnorm <= tau_p) { break; }

      schur_pc_->Mult(r_, z_);
      if (proj_) { proj_->Project(z_); }
      const double rz_new = Dot(r_, z_);
      // Plain CG: Fletcher-Reeves (r_new, z_new) / (r, z). Flexible CG
      // (Notay): Polak-Ribiere (z_new, r_new - r) / (r, z), with
      // r_new - r = -gamma q: no extra vector.
      const double beta = flexible ? -gamma * Dot(z_, q_) / rz : rz_new / rz;
      d_ *= beta;
      d_ += z_;
      rz = rz_new;
   }
}

void SchurSolver::FgmresCycle(Vector& p, double tau_p, int cycle)
{
   INCNS_PROFILE("schur_solver::fgmres");
   if (!fgmres_)
   {
      fgmres_ = std::make_unique<PersistentFGMRESSolver>(comm_);
      // After each iteration: record it, and set the next application's
      // inner tolerance from the residual FGMRES reports.
      monitor_ = std::make_unique<ResidualHook>([this](int it, double norm)
      {
         if (it > 0 && opts_.history)
         {
            SchurIterationRecord rec;
            rec.cycle = fg_cycle_;
            rec.iteration = fg_base_ + it;
            rec.residual = norm;
            rec.inner_tolerance = fg_inner_tol_;
            rec.inner_iterations = sop_->LastIterations();
            history_.push_back(rec);
         }
         fg_inner_tol_ = InnerRelTol(norm, fg_tau_p_);
         sop_->SetInnerTolerance(fg_inner_tol_);
      });
      fgmres_->SetOperator(*sop_);
      fgmres_->SetPreconditioner(*schur_pc_);
      fgmres_->SetMonitor(*monitor_);
      fgmres_->SetPrintLevel(-1);
      fgmres_->iterative_mode = false;
   }
   fgmres_->SetKDim(opts_.kdim);
   fgmres_->SetRelTol(0.0);
   fgmres_->SetAbsTol(tau_p);
   fgmres_->SetMaxIter(std::max(1, max_iter_ - stats_.iterations));
   if (print_level_ >= 1) { fgmres_->SetPrintLevel(1); }
   fg_cycle_ = cycle;
   fg_base_ = stats_.iterations;
   fg_tau_p_ = tau_p;
   fg_inner_tol_ = InnerRelTol(std::sqrt(Dot(r_, r_)), tau_p);
   sop_->SetInnerTolerance(fg_inner_tol_);
   // S dp = r: dp in d_ (zero start), then p += dp.
   d_ = 0.0;
   fgmres_->Mult(r_, d_);
   p += d_;
   stats_.iterations += fgmres_->GetNumIterations();
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
   const double tau_p = tau / std::sqrt(2.0);
   const double tau_u = tau / std::sqrt(2.0);

   // System 1: u = A^-1 (f - B^T p), from the warm start.
   double res_u = 0.0;
   bool vel_ok = SolveVelocity(f, p, u, tau_u, res_u);

   // The pressure projection: p += X W^T r, r -= W W^T r (its image). u is
   // then offset by A^-1 B^T times that correction until System 3.
   double rnorm = PressureResidual(g, u, r_);
   bool fresh = true; // r_ holds the residual of the current (u, p)
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
         if (opts_.method == SchurKrylov::FGMRES)
         {
            FgmresCycle(p, tau_p, cycle);
         }
         else
         {
            CgCycle(u, p, rnorm, tau_p, f, cycle);
         }
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
