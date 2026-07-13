#include "solver/case.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

Case::Case(ParMesh& mesh, const Parameters& params)
   : mesh_(mesh), params_(params),
     spaces_(mesh, params.order_u, params.order_p),
     own_bc_(spaces_.Velocity()), bc_(&own_bc_),
     zero_vec_(params.mesh.dim), zero_forcing_((zero_vec_ = 0.0, zero_vec_)),
     forcing_(&zero_forcing_), initial_(nullptr)
{
   // The mesh must have been built from NORMALIZED parameters -- the case
   // cannot detect a dimensional mesh after the fact, only this flag can.
   MFEM_VERIFY(params_.nondim.normalized,
               "case: call Parameters::Normalize() before building the "
               "mesh and the case (LoadYAML does it automatically)");
}

void Case::SetBoundaryConditions(BoundaryConditions& bc)
{
   MFEM_VERIFY(!integrator_, "case: BCs must be set before stepping");
   bc_ = &bc;
}

void Case::SetInitialVelocity(VectorCoefficient& u0)
{
   MFEM_VERIFY(!integrator_,
               "case: the IC must be set before stepping");
   initial_ = &u0;
}

void Case::SetForcing(VectorCoefficient& f)
{
   MFEM_VERIFY(!integrator_,
               "case: the forcing must be set before stepping");
   forcing_ = &f;
}

void Case::EnsureSetup()
{
   if (integrator_) { return; }
   INCNS_PROFILE("case::setup");

   // The Case dispatches on the equation set. Navier-Stokes (convection,
   // dealiasing, the NSE stepper) is Sprint 2 -- the selector exists now but is
   // rejected until then, rather than silently solving Stokes.
   MFEM_VERIFY(params_.equation == Equation::Stokes,
               "Case: NavierStokes is not available until Sprint 2; set "
               "the equation to Stokes");

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

   if (Mpi::Root() && params_.print_level >= 0)
   {
      mfem::out << "[incns] Re = " << params_.nondim.Re
                << (params_.nondim.mode == ScalingMode::Dimensional
                    ? "  (dimensional inputs: L_ref = " : "")
                << (params_.nondim.mode == ScalingMode::Dimensional
                    ? std::to_string(params_.nondim.L_ref) + ", U_ref = " +
                    std::to_string(params_.nondim.U_ref) + ", T_ref = " +
                    std::to_string(params_.nondim.TRef()) + ")"
                    : "")
                << std::endl;
   }

   if (initial_) { integrator_->SetInitialVelocity(*initial_); }
   else
   {
      integrator_->SetInitialVelocity(zero_forcing_); // zero start
   }

   // Restart replaces whatever initial state was just set: the restored
   // history skips the startup ramp and continues the interrupted march.
   if (!params_.restart_from.empty())
   {
      Checkpoint::Read(params_.restart_from, *integrator_);
      cycle_ = integrator_->StepCount(); // output/checkpoint cycles continue
   }

   if (params_.output.enabled)
   {
      output_ = std::make_unique<OutputWriter>(mesh_, integrator_->Velocity(),
                integrator_->Pressure(),
                params_.output, params_.order_u,
                params_.nondim);
      output_->MaybeSave(0, 0.0); // initial state
   }
}

void Case::Step()
{
   EnsureSetup();
   integrator_->Step();
   ++cycle_;
   if (output_) { output_->MaybeSave(cycle_, integrator_->Time()); }
   if (params_.checkpoint.enabled &&
       cycle_ % params_.checkpoint.interval == 0)
   {
      Checkpoint::Write(params_.checkpoint.path, *integrator_, params_);
   }

   // Bad-reference-scale guard (dimensional mode): with a well-chosen U_ref
   // the nondimensional velocity stays O(1); a large drift means the declared
   // scales do not match the flow and the tolerances/residual balance suffer.
   if (params_.nondim.mode == ScalingMode::Dimensional && !warned_scaling_)
   {
      Vector u_true(spaces_.Velocity().GetTrueVSize());
      integrator_->Velocity().GetTrueDofs(u_true);
      double linf = u_true.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &linf, 1, MPI_DOUBLE, MPI_MAX,
                    spaces_.Velocity().GetComm());
      if (linf > 50.0)
      {
         warned_scaling_ = true;
         if (Mpi::Root())
         {
            mfem::out << "[incns] WARNING: ||u*||_inf = " << linf
                      << " is far from O(1); U_ref likely does not match the "
                      "flow -- tolerances and the residual balance degrade."
                      << std::endl;
         }
      }
   }
}

void Case::Run()
{
   INCNS_PROFILE("case::run");
   EnsureSetup();
   while (!integrator_->Done()) { Step(); }
}

double Case::Time() const
{
   return integrator_ ? integrator_->Time() : 0.0;
}

bool Case::Done() const
{
   return integrator_ && integrator_->Done();
}

ParGridFunction& Case::Velocity()
{
   EnsureSetup();
   return integrator_->Velocity();
}

ParGridFunction& Case::Pressure()
{
   EnsureSetup();
   return integrator_->Pressure();
}

StokesTimeIntegrator& Case::Integrator()
{
   EnsureSetup();
   return *integrator_;
}

} // namespace incns
