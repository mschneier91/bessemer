#include "time/time_integrator.hpp"

#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

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
     u_(&spaces.Velocity()), p_(&spaces.Pressure())
{
   INCNS_PROFILE("time_integrator::setup");

   MFEM_VERIFY(opts_.dt > 0.0, "time_integrator: dt must be positive");
   MFEM_VERIFY(opts_.order == 2 || opts_.order == 3,
               "time_integrator: order must be 2 (production) or 3 (test-only)");

   auto make_solver = [&](double nu_eff, double mass_coeff)
   {
      StokesSolverOptions so;
      so.nu = nu_eff;
      so.collocated_mass = opts_.collocated_mass;
      so.mass_coeff = mass_coeff;
      so.rtol = opts_.rtol;
      so.atol = opts_.atol;
      so.max_iter = opts_.max_iter;
      so.kdim = opts_.kdim;
      so.print_level = opts_.print_level;
      return std::make_unique<StokesSolver>(spaces_, rules_, bc_, so);
   };

   const double dt = opts_.dt;
   // Trapezoidal starter: [(1/dt) M + (nu/2) K] u^1 = ... -- realized by
   // halving the viscosity passed to the solver (the Schur scale follows).
   trap_ = make_solver(0.5 * opts_.nu, 1.0 / dt);
   // Uniform-step BDF factors beta0/dt; the per-step weights are recomputed
   // from the stored times in Step() and checked against these.
   bdf2_ = make_solver(opts_.nu, 1.5 / dt);
   if (opts_.order == 3) { bdf3_ = make_solver(opts_.nu, (11.0 / 6.0) / dt); }

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

void StokesTimeIntegrator::Step()
{
   INCNS_PROFILE("time_integrator::step");

   MFEM_VERIFY(!hist_.empty(),
               "time_integrator: call SetInitialVelocity first");

   const double dt = opts_.dt;
   const double t_new = t_ + dt;
   const int n_u = spaces_.Velocity().GetTrueVSize();

   // Time-dependent data advanced EVERY step: Dirichlet re-elimination happens
   // inside SolveTrue with the BC coefficients at t_new.
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
      b.Add(1.0 / dt, tmp);
      trap_->Blocks().ViscousUnconstrained().Mult(hist_[0], tmp); // (nu/2) K u^0
      b -= tmp;
      solver = trap_.get();
   }
   else
   {
      // BDF-k: M sum_j c_j u^{n+1-j} + nu K u^{n+1} + B^T p = F^{n+1}
      // => A u^{n+1} - B^T p = F^{n+1} - M sum_{j>=1} c_j u^{n+1-j},
      // with A's mass factor beta0/dt = c[0]. Weights come from the ACTUAL
      // stored times (variable-step-ready; uniform here).
      const bool use_bdf3 = (opts_.order == 3 && step_count_ >= 2);
      const int k = use_bdf3 ? 3 : 2;
      solver = use_bdf3 ? bdf3_.get() : bdf2_.get();

      std::vector<double> times;
      times.push_back(t_new);
      for (int j = 0; j < k; ++j) { times.push_back(hist_times_[j]); }
      const std::vector<double> c = BdfWeights(times);

      // The solver's momentum block was assembled with the uniform-step
      // beta0/dt; the recomputed weight must match (guards step bookkeeping).
      MFEM_VERIFY(std::abs(c[0] - solver->Blocks().Options().mass_coeff) <=
                  1e-10 * std::abs(c[0]),
                  "time_integrator: BDF leading weight does not match the "
                  "assembled mass factor (step bookkeeping bug)");

      AssembleForcing(t_new, b);
      Vector combo(n_u), tmp(n_u);
      combo = 0.0;
      for (int j = 1; j <= k; ++j) { combo.Add(c[j], hist_[j - 1]); }
      solver->Blocks().Mass().Mult(combo, tmp);
      b -= tmp;
   }

   // u_/p_ carry the previous step's fields: warm start + Dirichlet target.
   solver->SolveTrue(b, u_, p_);
   last_iterations_ = solver->Iterations();
   MFEM_VERIFY(solver->Converged(),
               "time_integrator: implicit solve did not converge at t = "
               << t_new);

   Vector u_true(n_u);
   u_.GetTrueDofs(u_true);
   hist_.push_front(std::move(u_true));
   hist_times_.push_front(t_new);
   const std::size_t needed = (opts_.order == 3) ? 3 : 2;
   while (hist_.size() > needed)
   {
      hist_.pop_back();
      hist_times_.pop_back();
   }

   t_ = t_new;
   ++step_count_;
}

void StokesTimeIntegrator::Run()
{
   INCNS_PROFILE("time_integrator::run");
   while (!Done()) { Step(); }
}

} // namespace incns
