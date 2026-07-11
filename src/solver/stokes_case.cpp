#include "solver/stokes_case.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesCase::StokesCase(ParMesh& mesh, const Parameters& params)
   : mesh_(mesh), params_(params),
     spaces_(mesh, params.order_u, params.order_p),
     own_bc_(spaces_.Velocity()), bc_(&own_bc_),
     zero_vec_(params.mesh.dim), zero_forcing_((zero_vec_ = 0.0, zero_vec_)),
     forcing_(&zero_forcing_), initial_(nullptr)
{
}

void StokesCase::SetBoundaryConditions(BoundaryConditions& bc)
{
   MFEM_VERIFY(!integrator_, "stokes_case: BCs must be set before stepping");
   bc_ = &bc;
}

void StokesCase::SetInitialVelocity(VectorCoefficient& u0)
{
   MFEM_VERIFY(!integrator_,
               "stokes_case: the IC must be set before stepping");
   initial_ = &u0;
}

void StokesCase::SetForcing(VectorCoefficient& f)
{
   MFEM_VERIFY(!integrator_,
               "stokes_case: the forcing must be set before stepping");
   forcing_ = &f;
}

void StokesCase::EnsureSetup()
{
   if (integrator_) { return; }
   INCNS_PROFILE("stokes_case::setup");

   TimeIntegratorOptions opts;
   opts.nu = params_.nu;
   opts.dt = params_.dt;
   opts.t_final = params_.t_final;
   opts.order = params_.time_order;
   opts.adaptive = params_.adaptive;
   opts.controller = params_.controller;
   opts.collocated_mass = params_.collocated_mass;
   opts.grad_div = params_.grad_div;
   opts.rtol = params_.krylov_rtol;
   opts.atol = params_.krylov_atol;
   opts.max_iter = params_.max_iter;
   opts.kdim = params_.kdim;
   opts.print_level = params_.print_level;
   integrator_ = std::make_unique<StokesTimeIntegrator>(spaces_, rules_, *bc_,
                 *forcing_, opts);

   if (initial_) { integrator_->SetInitialVelocity(*initial_); }
   else
   {
      integrator_->SetInitialVelocity(zero_forcing_); // zero start
   }

   if (params_.output.enabled)
   {
      output_ = std::make_unique<OutputWriter>(mesh_, integrator_->Velocity(),
                integrator_->Pressure(),
                params_.output, params_.order_u);
      output_->MaybeSave(0, 0.0); // initial state
   }
}

void StokesCase::Step()
{
   EnsureSetup();
   integrator_->Step();
   ++cycle_;
   if (output_) { output_->MaybeSave(cycle_, integrator_->Time()); }
}

void StokesCase::Run()
{
   INCNS_PROFILE("stokes_case::run");
   EnsureSetup();
   while (!integrator_->Done()) { Step(); }
}

double StokesCase::Time() const
{
   return integrator_ ? integrator_->Time() : 0.0;
}

bool StokesCase::Done() const
{
   return integrator_ && integrator_->Done();
}

ParGridFunction& StokesCase::Velocity()
{
   EnsureSetup();
   return integrator_->Velocity();
}

ParGridFunction& StokesCase::Pressure()
{
   EnsureSetup();
   return integrator_->Pressure();
}

StokesTimeIntegrator& StokesCase::Integrator()
{
   EnsureSetup();
   return *integrator_;
}

} // namespace incns
