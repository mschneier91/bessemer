#include "time/time_integrator.hpp"

#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

StokesTimeIntegrator::StokesTimeIntegrator(MixedSpaces& spaces,
      const RuleBook& rules,
      BoundaryConditions& bc,
      VectorCoefficient& forcing,
      const TimeIntegratorOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), forcing_(forcing), opts_(opts),
     u_(&spaces.Velocity()), p_(&spaces.Pressure()),
     u2_scratch_(&spaces.Velocity()), p2_scratch_(&spaces.Pressure()),
     u3_scratch_(&spaces.Velocity()), p3_scratch_(&spaces.Pressure()),
     dt_(opts.dt)
{
   INCNS_PROFILE("time_integrator::setup");

   MFEM_VERIFY(opts_.dt > 0.0, "time_integrator: dt must be positive");
   MFEM_VERIFY(opts_.order == 2 || opts_.order == 3,
               "time_integrator: order must be 2 (production) or 3 (test-only)");

   // Trapezoidal starter: [(1/dt) M + (nu/2) K] u^1 = ... -- realized by
   // halving the viscosity passed to the solver (the Schur scale follows).
   {
      StokesSolverOptions so;
      so.nu = 0.5 * opts_.nu;
      so.collocated_mass = opts_.collocated_mass;
      so.grad_div = opts_.grad_div;
      so.grad_div_scale = opts_.grad_div_scale;
      so.velocity_prec = opts_.velocity_prec;
      so.amg_reuse = opts_.amg_reuse;
      so.schur = opts_.schur;
      so.cc = opts_.cc;
      so.mass_coeff = 1.0 / dt_;
      so.rtol = opts_.rtol;
      so.atol = opts_.atol;
      so.max_iter = opts_.max_iter;
      so.kdim = opts_.kdim;
      so.print_level = opts_.print_level;
      trap_ = std::make_unique<StokesSolver>(spaces_, rules_, bc_, so);
      trap_c0_ = 1.0 / dt_;
   }

   if (opts_.adaptive)
   {
      controller_ = std::make_unique<AdaptiveController>(opts_.controller);
   }

   u_ = 0.0;
   p_ = 0.0;
}

void StokesTimeIntegrator::SetInitialVelocity(VectorCoefficient& u0)
{
   MFEM_VERIFY(step_count_ == 0, "time_integrator: already stepping");
   u0.SetTime(0.0);
   u_.ProjectCoefficient(u0);

   Vector u_true(spaces_.Velocity().GetTrueVSize());
   u_.GetTrueDofs(u_true);
   hist_.clear();
   hist_times_.clear();
   hist_.push_front(std::move(u_true));
   hist_times_.push_front(0.0);
}

void StokesTimeIntegrator::SetHistory(const std::vector<Vector>& states,
                                      const std::vector<double>& times,
                                      int completed_steps, double next_dt,
                                      const Vector* pressure)
{
   MFEM_VERIFY(!states.empty() && states.size() == times.size(),
               "time_integrator: bad history sizes");
   MFEM_VERIFY(completed_steps >= static_cast<int>(states.size()) - 1,
               "time_integrator: inconsistent completed_steps");
   MFEM_VERIFY(next_dt > 0.0, "time_integrator: bad next_dt");
   const int n_u = spaces_.Velocity().GetTrueVSize();

   hist_.clear();
   hist_times_.clear();
   for (std::size_t j = 0; j < states.size(); ++j)
   {
      MFEM_VERIFY(states[j].Size() == n_u,
                  "time_integrator: history size does not match the velocity "
                  "space (rank/partition mismatch?)");
      if (j > 0)
      {
         MFEM_VERIFY(times[j] < times[j - 1],
                     "time_integrator: history times must be decreasing");
      }
      hist_.push_back(states[j]);
      hist_times_.push_back(times[j]);
   }

   t_ = times[0];
   step_count_ = completed_steps;
   dt_ = next_dt;
   u_.SetFromTrueDofs(hist_[0]);
   if (pressure) { p_.SetFromTrueDofs(*pressure); }
   else { p_ = 0.0; }
}

void StokesTimeIntegrator::SetDtCeiling(AdaptiveController::DtCeilingFn ceiling)
{
   MFEM_VERIFY(controller_, "time_integrator: dt ceiling requires adaptive mode");
   controller_->SetDtCeiling(std::move(ceiling));
}

bool StokesTimeIntegrator::Done() const
{
   if (opts_.adaptive)
   {
      return t_ >= opts_.t_final - 1e-12 * std::max(1.0, std::abs(opts_.t_final));
   }
   return t_ >= opts_.t_final - 0.5 * opts_.dt;
}

StokesSolver& StokesTimeIntegrator::EnsureBdfSolver(SolverCache& cache,
      double c0)
{
   if (!cache.solver)
   {
      INCNS_PROFILE("time_integrator::build_solver");
      StokesSolverOptions so;
      so.nu = opts_.nu;
      so.collocated_mass = opts_.collocated_mass;
      so.grad_div = opts_.grad_div;
      so.grad_div_scale = opts_.grad_div_scale;
      so.velocity_prec = opts_.velocity_prec;
      so.amg_reuse = opts_.amg_reuse;
      so.schur = opts_.schur;
      so.cc = opts_.cc;
      so.mass_coeff = c0; // the leading BDF weight beta0/dt -- exact match
      so.rtol = opts_.rtol;
      so.atol = opts_.atol;
      so.max_iter = opts_.max_iter;
      so.kdim = opts_.kdim;
      so.print_level = opts_.print_level;
      cache.solver = std::make_unique<StokesSolver>(spaces_, rules_, bc_, so);
      cache.c0 = c0;
   }
   else if (std::abs(c0 - cache.c0) > 1e-12 * std::abs(c0))
   {
      // Delta-t changed: refresh only the momentum block, not the whole solver.
      cache.solver->Refresh(c0);
      cache.c0 = c0;
   }
   return *cache.solver;
}

void StokesTimeIntegrator::AssembleForcing(double t, Vector& F)
{
   forcing_.SetTime(t);
   ParLinearForm f_form(&spaces_.Velocity());
   auto* fi = new VectorDomainLFIntegrator(forcing_);
   const int dim = spaces_.Dim();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   fi->SetIntRule(&rules_.Get(geom, 2 * spaces_.OrderU() + 2));
   f_form.AddDomainIntegrator(fi);
   f_form.UseFastAssembly(true); // device-ready assembly path (project policy)
   f_form.Assemble();
   std::unique_ptr<HypreParVector> f_true(f_form.ParallelAssemble());
   F = *f_true;
}

void StokesTimeIntegrator::AssembleBdfRhs(const std::vector<double>& c,
      double t_new, Vector& b)
{
   // BDF-k: M sum_j c_j u^{n+1-j} + nu K u^{n+1} + B^T p = F^{n+1}
   // => A u^{n+1} - B^T p = F^{n+1} - M sum_{j>=1} c_j u^{n+1-j}.
   const int n_u = spaces_.Velocity().GetTrueVSize();
   AssembleForcing(t_new, b);
   Vector combo(n_u), tmp(n_u);
   combo = 0.0;
   for (std::size_t j = 1; j < c.size(); ++j)
   {
      combo.Add(c[j], hist_[j - 1]);
   }
   // Any solver's Mass() is the same plain M (mass_coeff scales only the
   // momentum block); the trapezoidal solver always exists.
   trap_->Blocks().Mass().Mult(combo, tmp);
   b -= tmp;
}

void StokesTimeIntegrator::Commit(double t_new, const Vector& u_true,
                                  ParGridFunction& u_gf, ParGridFunction& p_gf)
{
   if (&u_gf != &u_) { u_ = u_gf; }
   if (&p_gf != &p_) { p_ = p_gf; }

   hist_.push_front(u_true);
   hist_times_.push_front(t_new);
   const std::size_t needed =
      (opts_.adaptive || opts_.order == 3) ? 3 : 2;
   while (hist_.size() > needed)
   {
      hist_.pop_back();
      hist_times_.pop_back();
   }

   t_ = t_new;
   ++step_count_;
}

void StokesTimeIntegrator::StepStartup()
{
   const double t_new = t_ + dt_;
   const int n_u = spaces_.Velocity().GetTrueVSize();
   bc_.SetTime(t_new);

   Vector b(n_u);
   StokesSolver* solver = nullptr;

   if (step_count_ == 0)
   {
      // Trapezoidal starter:
      //   (1/dt) M (u^1 - u^0) + (nu/2) K (u^1 + u^0) + B^T p = (F^0 + F^1)/2
      // => A u^1 - B^T p = (F^0+F^1)/2 + (1/dt) M u^0 - (nu/2) K u^0.
      Vector f0(n_u), f1(n_u), tmp(n_u);
      AssembleForcing(t_, f0);
      AssembleForcing(t_new, f1);
      b = f0;
      b += f1;
      b *= 0.5;
      trap_->Blocks().Mass().Mult(hist_[0], tmp);
      b.Add(1.0 / dt_, tmp);
      trap_->Blocks().ViscousUnconstrained().Mult(hist_[0], tmp); // (nu/2) K u^0
      b -= tmp;
      // The starter's momentum block is (1/dt) M + (nu/2) K; refresh it if the
      // first step is (re)tried at a different dt (adaptive rejection).
      if (std::abs(trap_c0_ - 1.0 / dt_) > 1e-12 / dt_)
      {
         trap_->Refresh(1.0 / dt_);
         trap_c0_ = 1.0 / dt_;
      }
      solver = trap_.get();
   }
   else
   {
      // Second step: BDF2 from {u^1, u^0} (weights from the actual times).
      std::vector<double> times = {t_new, hist_times_[0], hist_times_[1]};
      const std::vector<double> c = BdfWeights(times);
      solver = &EnsureBdfSolver(bdf2_, c[0]);
      AssembleBdfRhs(c, t_new, b);
   }

   solver->SolveTrue(b, u_, p_);
   last_iterations_ = solver->Iterations();
   MFEM_VERIFY(solver->Converged(),
               "time_integrator: implicit solve did not converge at t = "
               << t_new);

   Vector u_true(n_u);
   u_.GetTrueDofs(u_true);
   Commit(t_new, u_true, u_, p_);
}

void StokesTimeIntegrator::StepFixed()
{
   const double t_new = t_ + dt_;
   const int n_u = spaces_.Velocity().GetTrueVSize();
   bc_.SetTime(t_new);

   const bool use_bdf3 = (opts_.order == 3 && step_count_ >= 2);
   const int k = use_bdf3 ? 3 : 2;
   std::vector<double> times;
   times.push_back(t_new);
   for (int j = 0; j < k; ++j) { times.push_back(hist_times_[j]); }
   const std::vector<double> c = BdfWeights(times);
   StokesSolver& solver =
      EnsureBdfSolver(use_bdf3 ? bdf3_ : bdf2_, c[0]);

   Vector b(n_u);
   AssembleBdfRhs(c, t_new, b);
   solver.SolveTrue(b, u_, p_);
   last_iterations_ = solver.Iterations();
   MFEM_VERIFY(solver.Converged(),
               "time_integrator: implicit solve did not converge at t = "
               << t_new);

   Vector u_true(n_u);
   u_.GetTrueDofs(u_true);
   Commit(t_new, u_true, u_, p_);
}

void StokesTimeIntegrator::StepAdaptive()
{
   const int n_u = spaces_.Velocity().GetTrueVSize();
   const MPI_Comm comm = spaces_.Velocity().GetComm();

   while (true)
   {
      // Never overshoot t_final.
      const double dt = std::min(dt_, opts_.t_final - t_);
      const double t_new = t_ + dt;
      bc_.SetTime(t_new);

      // BDF2 candidate -- the only one that may advance the solution.
      std::vector<double> t2 = {t_new, hist_times_[0], hist_times_[1]};
      const std::vector<double> c2 = BdfWeights(t2);
      Vector b2(n_u);
      AssembleBdfRhs(c2, t_new, b2);
      u2_scratch_ = u_;
      p2_scratch_ = p_;
      StokesSolver& s2 = EnsureBdfSolver(bdf2_, c2[0]);
      s2.SolveTrue(b2, u2_scratch_, p2_scratch_);
      last_iterations_ = s2.Iterations();
      MFEM_VERIFY(s2.Converged(),
                  "time_integrator: BDF2 solve did not converge at t = "
                  << t_new);

      // BDF3 candidate -- auxiliary, formed ONLY for the LTE estimate (the
      // difference approximates the LTE of the order-2 step); never advances.
      std::vector<double> t3 = {t_new, hist_times_[0], hist_times_[1],
                                hist_times_[2]
                               };
      const std::vector<double> c3 = BdfWeights(t3);
      Vector b3(n_u);
      AssembleBdfRhs(c3, t_new, b3);
      u3_scratch_ = u_;
      p3_scratch_ = p_;
      StokesSolver& s3 = EnsureBdfSolver(bdf3_, c3[0]);
      s3.SolveTrue(b3, u3_scratch_, p3_scratch_);
      MFEM_VERIFY(s3.Converged(),
                  "time_integrator: BDF3 solve did not converge at t = "
                  << t_new);

      // Control on VELOCITY only (pressure is algebraic), global norms.
      Vector u2_true(n_u), u3_true(n_u);
      u2_scratch_.GetTrueDofs(u2_true);
      u3_scratch_.GetTrueDofs(u3_true);
      Vector diff(u2_true);
      diff -= u3_true;
      const double lte = std::sqrt(InnerProduct(comm, diff, diff));
      const double u_norm = std::sqrt(InnerProduct(comm, u2_true, u2_true));

      const bool accepted = controller_->Evaluate(t_, dt, lte, u_norm);
      dt_ = controller_->NextDt();
      if (accepted)
      {
         Commit(t_new, u2_true, u2_scratch_, p2_scratch_);
         return;
      }
      // Rejected: retry from the same state at the controller's smaller dt.
   }
}

void StokesTimeIntegrator::Step()
{
   INCNS_PROFILE("time_integrator::step");
   MFEM_VERIFY(!hist_.empty(),
               "time_integrator: call SetInitialVelocity first");

   // The first two steps have too little history for the BDF3 estimator, so
   // they run un-adapted at the initial dt (trapezoidal starter, then BDF2).
   if (step_count_ < 2)
   {
      StepStartup();
      return;
   }
   if (opts_.adaptive)
   {
      StepAdaptive();
      return;
   }
   StepFixed();
}

void StokesTimeIntegrator::Run()
{
   INCNS_PROFILE("time_integrator::run");
   while (!Done()) { Step(); }
}

} // namespace incns
