#include "solver/case.hpp"

#include "amr/gradient_indicator.hpp"
#include "amr/history_projection.hpp"
#include "amr/mesh_adapter.hpp"
#include "util/profiler.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

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

Case::~Case()
{
   diag_log_.reset();
   output_.reset();
   integrator_.reset(); // every operator holding a factor pointer goes first
   mesh_.DeleteGeometricFactors();
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

void Case::BuildIntegrator()
{
   // The Case dispatches on the equation set. Both are available as of Sprint
   // 2.2: NavierStokes just adds the dealiased, AB/EXT-extrapolated convection
   // term to the right-hand side -- the implicit block solve is identical.
   TimeIntegratorOptions opts;
   opts.convection = (params_.equation == Equation::NavierStokes);
   opts.convective_form = params_.convective_form;
   opts.rotation_pc = params_.rotation_pc;
   opts.rotation_in_lor = params_.rotation_in_lor;
   {
      const bool rotational =
         (params_.equation == Equation::NavierStokes &&
          params_.convective_form == ConvectiveForm::Rotational);
      MFEM_VERIFY(rotational || params_.rotation_schur.mode ==
                  RotationalSchurPreconditioner::Mode::CahouetChabard,
                  "case: solver.rotation_schur tensor/auto needs the rotational "
                  "form (physics.convective_form: rotational)");
      opts.rotation_schur = params_.rotation_schur;
      opts.rotation_diagnostics = rotational && params_.rotation_log_interval > 0;
      opts.rotation_picard = rotational ? params_.rotation_picard : 0;
   }
   opts.nu = params_.nu;
   opts.dt = params_.dt;
   opts.t_final = params_.t_final;
   opts.order = params_.time_order;
   opts.adaptive = params_.adaptive;
   opts.controller = params_.controller;
   opts.collocated_mass = params_.collocated_mass;
   opts.grad_div = params_.grad_div;
   opts.grad_div_scale = params_.grad_div_scale;
   opts.velocity_prec = params_.velocity_prec;
   opts.amg_reuse = params_.amg_reuse;
   opts.schur = params_.schur;
   opts.cc = params_.cc;
   opts.rtol = params_.krylov_rtol;
   opts.atol = params_.krylov_atol;
   opts.max_iter = params_.max_iter;
   opts.kdim = params_.kdim;
   opts.print_level = params_.print_level;
   integrator_ = std::make_unique<StokesTimeIntegrator>(spaces_, rules_, *bc_,
                 *forcing_, opts);
   body_force_.reset();
   if (params_.forces.enabled)
   {
      body_force_ = std::make_unique<BodyForce>(spaces_.Velocity(),
                    params_.forces.attributes);
   }
}

void Case::BuildOutput(bool restart)
{
   output_.reset();
   if (!params_.output.enabled) { return; }
   output_ = std::make_unique<OutputWriter>(mesh_, integrator_->Velocity(),
             integrator_->Pressure(), params_.output,
             params_.order_u, params_.nondim, restart);
   if (amr_eta_)
   {
      output_->RegisterExtra("amr_eta", amr_eta_.get());
      output_->RegisterExtra("amr_level", amr_level_.get());
   }
}

void Case::EnsureSetup()
{
   if (integrator_) { return; }
   INCNS_PROFILE("case::setup");

   // Restart of an adapted run: replay its refinement history on the initial
   // mesh -- same deterministic operations, same np, so the same mesh,
   // partition and dof numbering as when the checkpoint was written.
   if (!params_.restart_from.empty())
   {
      std::vector<RefinementRecord> recs =
         Checkpoint::ReadRefinements(params_.restart_from);
      MFEM_VERIFY(recs.empty() || params_.amr.enabled, "case: the checkpoint "
                  "holds an adapted mesh; restart it with amr.enabled");
      for (const RefinementRecord& r : recs)
      {
         MeshAdapter::Refine(mesh_, spaces_, bc_, r.refs, r.nc_limit,
                             r.rebalance, {});
      }
      refine_log_ = std::move(recs);
   }
   // amr.initial_passes refine on the analytic initial condition BEFORE any
   // operator exists (nothing to transfer: the IC is re-projected).
   InitialRefinement();
   BuildIntegrator();

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

   SetupCfl();
   if (params_.amr.enabled && params_.amr.write_indicator)
   {
      UpdateAmrCellData();
   }
   if (params_.output.enabled)
   {
      BuildOutput(/*restart=*/false);
      output_->MaybeSave(0, 0.0); // initial state

      if (params_.output.diagnostics)
      {
         const std::string csv =
            params_.output.path + "/" + params_.output.name + "_diagnostics.csv";
         diag_log_ = std::make_unique<DiagnosticsLog>(csv, mesh_.GetComm());
         MaybeLogDiagnostics(0, 0.0); // initial state
      }
   }
}

void Case::MaybeLogDiagnostics(int cycle, double time)
{
   if (!diag_log_ || cycle % params_.output.interval != 0) { return; }
   diag_log_->Write(time, KineticEnergy(), DissipationRate(), DivergenceNorm());
}

void Case::Step()
{
   EnsureSetup();
   const double t_prev = integrator_->Time();
   const double wall0 = MPI_Wtime();
   integrator_->Step();
   const double step_wall = MPI_Wtime() - wall0;
   ++cycle_;
   // Forces and the rotation log use the step's own solver: log them before
   // an event rebuilds the integrator.
   MaybeLogForces();
   MaybeLogRotation(integrator_->Time() - t_prev, step_wall);
   // An adaptation event comes BEFORE output and checkpoint: both then see the
   // refined mesh with the (exactly transferred) state. The rebuilt integrator
   // has no step of its own yet, so the force of the step just taken is
   // cached first (BodyForceVector returns it until the next step).
   if (AdaptDue())
   {
      if (body_force_) { force_cache_ = BodyForceVector(); }
      Adapt();
   }
   if (output_) { output_->MaybeSave(cycle_, integrator_->Time()); }
   MaybeLogDiagnostics(cycle_, integrator_->Time());
   if (params_.checkpoint.enabled &&
       cycle_ % params_.checkpoint.interval == 0)
   {
      WriteCheckpoint(params_.checkpoint.path);
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

double Case::KineticEnergy()
{
   EnsureSetup();
   return incns::KineticEnergy(integrator_->Velocity(), rules_);
}

double Case::DissipationRate()
{
   EnsureSetup();
   return incns::DissipationRate(integrator_->Velocity(), params_.nu, rules_);
}

double Case::DivergenceNorm()
{
   EnsureSetup();
   return incns::DivergenceNorm(integrator_->Velocity(), rules_);
}

bool Case::AdaptDue() const
{
   const AmrParameters& a = params_.amr;
   return a.enabled && a.interval > 0 && cycle_ > 0 && cycle_ % a.interval == 0
          && integrator_->StepCount() >= 2 && !integrator_->Done();
}

void Case::InitialRefinement()
{
   const AmrParameters& a = params_.amr;
   if (!a.enabled || a.initial_passes == 0 || !params_.restart_from.empty() ||
       !initial_)
   {
      return;
   }
   INCNS_PROFILE("amr::initial");
   const RefinementMarker marker(a);
   for (int pass = 0; pass < a.initial_passes; ++pass)
   {
      AdaptStats st;
      const double t0 = MPI_Wtime();
      st.ne_before = mesh_.GetGlobalNE();
      Array<Refinement> refs;
      {
         // The analytic IC on the current mesh, made conforming.
         ParGridFunction u0(&spaces_.Velocity());
         initial_->SetTime(0.0);
         u0.ProjectCoefficient(*initial_);
         Vector t;
         u0.GetTrueDofs(t);
         u0.SetFromTrueDofs(t);
         GradientIndicator ind(spaces_.Velocity(), rules_);
         Vector g;
         ind.Compute(u0, g);
         marker.Mark(mesh_, g, refs, &st.first_pass);
      }
      if (st.first_pass.marked == 0) { break; }
      MeshAdapter::Refine(mesh_, spaces_, bc_, refs, a.nc_limit, a.rebalance,
                          {});
      refine_log_.push_back({refs, a.nc_limit, a.rebalance});
      st.passes = 1;
      st.ne_after = mesh_.GetGlobalNE();
      st.seconds = MPI_Wtime() - t0;
      adapt_log_.push_back(st);
      if (Mpi::Root() && params_.print_level >= 0)
      {
         mfem::out << "[amr] initial pass " << pass + 1 << ": marked "
                   << st.first_pass.marked << ", elements " << st.ne_before
                   << " -> " << st.ne_after << std::endl;
      }
   }
}

AdaptStats Case::Adapt(bool force_rebuild)
{
   EnsureSetup();
   const AmrParameters& a = params_.amr;
   MFEM_VERIFY(a.enabled, "case: Adapt() needs amr.enabled (the mesh must be "
               "built nonconforming-ready by MakeCaseMesh)");
   MFEM_VERIFY(integrator_->StepCount() >= 2, "case: no adaptation inside the "
               "startup ramp (the restored history would skip it)");
   INCNS_PROFILE("amr::event");
   const double t0 = MPI_Wtime();
   AdaptStats st;
   st.time = integrator_->Time();
   st.cycle = cycle_;
   st.ne_before = mesh_.GetGlobalNE();
   const RefinementMarker marker(a);

   // First pass marks on the live velocity: if nothing is marked, nothing
   // is torn down.
   Array<Refinement> refs;
   {
      GradientIndicator ind(spaces_.Velocity(), rules_);
      Vector g;
      ind.Compute(integrator_->Velocity(), g);
      marker.Mark(mesh_, g, refs, &st.first_pass);
   }
   if (st.first_pass.marked == 0 && !force_rebuild)
   {
      st.ne_after = st.ne_before;
      st.seconds = MPI_Wtime() - t0;
      adapt_log_.push_back(st);
      return st;
   }

   // In-memory checkpoint; then everything mesh-dependent goes before the
   // mesh changes (operators hold pointers into the mesh's geometric data).
   IntegratorState state = integrator_->ExportState();
   output_.reset();
   integrator_.reset();

   // The state as fields on the current spaces, carried across refinement.
   std::vector<std::unique_ptr<ParGridFunction>> hist_gf;
   std::vector<ParGridFunction*> fields;
   for (const Vector& u : state.u_hist)
   {
      hist_gf.push_back(std::make_unique<ParGridFunction>(&spaces_.Velocity()));
      hist_gf.back()->SetFromTrueDofs(u);
      fields.push_back(hist_gf.back().get());
   }
   ParGridFunction p_gf(&spaces_.Pressure());
   p_gf.SetFromTrueDofs(state.pressure);
   fields.push_back(&p_gf);

   bool refined = false;
   for (int pass = 0; pass < a.passes_per_event; ++pass)
   {
      if (pass > 0)
      {
         // Later passes mark on the transferred newest velocity.
         GradientIndicator ind(spaces_.Velocity(), rules_);
         Vector g;
         ind.Compute(*hist_gf.front(), g);
         MarkStats ms;
         marker.Mark(mesh_, g, refs, &ms);
         if (ms.marked == 0) { break; }
      }
      if (refs.Size() == 0 && pass == 0 && st.first_pass.marked == 0) { break; }
      MeshAdapter::Refine(mesh_, spaces_, bc_, refs, a.nc_limit, a.rebalance,
                          fields);
      refine_log_.push_back({refs, a.nc_limit, a.rebalance});
      refined = true;
      ++st.passes;
   }

   // Back to true dofs on the (possibly) refined spaces.
   for (std::size_t j = 0; j < state.u_hist.size(); ++j)
   {
      state.u_hist[j].SetSize(spaces_.Velocity().GetTrueVSize());
      hist_gf[j]->GetTrueDofs(state.u_hist[j]);
   }
   state.pressure.SetSize(spaces_.Pressure().GetTrueVSize());
   p_gf.GetTrueDofs(state.pressure);
   hist_gf.clear();

   if (refined && a.project_history)
   {
      HistoryProjectionOptions po;
      po.rtol = params_.krylov_rtol;
      po.max_iter = params_.max_iter;
      po.kdim = params_.kdim;
      po.print_level = params_.print_level;
      st.projection_iterations =
         ProjectHistoryDivergenceFree(spaces_, rules_, *bc_, state, po);
   }

   BuildIntegrator();
   integrator_->ImportState(state);
   SetupCfl();
   st.rebuilt = true;
   if (a.write_indicator) { UpdateAmrCellData(); }
   BuildOutput(/*restart=*/true);

   st.ne_after = mesh_.GetGlobalNE();
   st.seconds = MPI_Wtime() - t0;
   adapt_log_.push_back(st);
   if (Mpi::Root() && params_.print_level >= 0)
   {
      mfem::out << "[amr] event at t = " << st.time << " (cycle " << st.cycle
                << "): marked " << st.first_pass.marked << " (dirs "
                << st.first_pass.per_dir[0] << "/" << st.first_pass.per_dir[1]
                << "/" << st.first_pass.per_dir[2] << "), elements "
                << st.ne_before << " -> " << st.ne_after << ", "
                << st.seconds << " s" << std::endl;
   }
   return st;
}

void Case::SetForceBody(const std::vector<int>& attributes)
{
   MFEM_VERIFY(!attributes.empty(), "case: SetForceBody() needs an attribute");
   params_.forces.enabled = true;
   params_.forces.attributes = attributes;
   if (integrator_) // already set up: rebuild on the current space
   {
      body_force_ = std::make_unique<BodyForce>(spaces_.Velocity(), attributes);
   }
}

Vector Case::BodyForceVector()
{
   EnsureSetup();
   MFEM_VERIFY(body_force_, "case: BodyForceVector() needs forces.enabled");
   // Right after an AMR event the rebuilt integrator has not stepped yet: the
   // force of the step that produced the current state was cached before the
   // event (Step()).
   if (!integrator_->LastSolver() && force_cache_.Size() > 0)
   {
      return force_cache_;
   }
   Vector r;
   integrator_->MomentumResidual(r);
   return body_force_->Force(r);
}

Vector Case::ForceCoefficients()
{
   return BodyForce::Coefficients(BodyForceVector(),
                                  params_.forces.reference_velocity,
                                  params_.forces.reference_area);
}

void Case::MaybeLogForces()
{
   if (!body_force_ || params_.forces.interval == 0 ||
       cycle_ % params_.forces.interval != 0) { return; }
   const Vector F = BodyForceVector();
   const Vector C = BodyForce::Coefficients(F, params_.forces.reference_velocity,
                    params_.forces.reference_area);
   if (!Mpi::Root()) { return; }
   const std::string path =
      params_.output.path + "/" + params_.output.name + "_forces.csv";
   const bool fresh = !std::filesystem::exists(path)
                      || cycle_ == params_.forces.interval;
   if (fresh) { std::filesystem::create_directories(params_.output.path); }
   std::ofstream f(path, fresh ? std::ios::trunc : std::ios::app);
   if (fresh)
   {
      f << "t";
      const char* ax = "xyz";
      for (int d = 0; d < F.Size(); ++d) { f << ",F" << ax[d]; }
      for (int d = 0; d < F.Size(); ++d) { f << ",C" << ax[d]; }
      f << "\n";
   }
   f << std::setprecision(16) << integrator_->Time();
   for (int d = 0; d < F.Size(); ++d) { f << "," << F(d); }
   for (int d = 0; d < F.Size(); ++d) { f << "," << C(d); }
   f << "\n";
}

void Case::MaybeLogRotation(double dt, double step_wall)
{
   if (params_.rotation_log_interval == 0 ||
       cycle_ % params_.rotation_log_interval != 0) { return; }
   const StokesSolver* solver = integrator_->LastSolver();
   if (!solver || !solver->HasRotation()) { return; }
   const SolveStats& s = solver->Stats();
   if (!Mpi::Root()) { return; }
   const std::string path =
      params_.output.path + "/" + params_.output.name + "_rotation.csv";
   const bool fresh = !rotation_log_started_;
   if (fresh) { std::filesystem::create_directories(params_.output.path); }
   std::ofstream f(path, fresh ? std::ios::trunc : std::ios::app);
   if (fresh)
   {
      f << "step,t,dt,sigma,mu_max,vol_fraction,vhat_max,schur_tensor,"
        "schur_switches,outer_iterations,vel_applications,vel_inner_iterations,"
        "vel_cap_hits,schur_applications,schur_time,vel_time,solve_time,"
        "step_time\n";
      rotation_log_started_ = true;
   }
   f << std::setprecision(10) << cycle_ << "," << integrator_->Time() << ","
     << dt << "," << s.sigma << "," << s.rotation.max_mu << ","
     << s.rotation.vol_fraction << "," << s.vhat_max << ","
     << (s.tensor_active ? 1 : 0) << "," << s.schur_switches << ","
     << s.outer_iterations << "," << s.vel_applications << ","
     << s.vel_inner_iterations << "," << s.vel_cap_hits << ","
     << s.schur_applications << "," << s.schur_time << "," << s.vel_time
     << "," << s.solve_time << "," << step_wall << "\n";
}

void Case::WriteCheckpoint(const std::string& dir)
{
   EnsureSetup();
   Checkpoint::Write(dir, *integrator_, params_, &refine_log_);
}

void Case::SetupCfl()
{
   cfl_.reset();
   // Only the IMEX convective form treats convection explicitly; the
   // rotational form is semi-implicit and has no convective CFL limit.
   if (params_.cfl_max <= 0.0 || params_.equation != Equation::NavierStokes ||
       params_.convective_form != ConvectiveForm::Convective)
   {
      return;
   }
   cfl_ = std::make_unique<ConvectiveCfl>(spaces_.Velocity(), rules_);
   if (params_.adaptive)
   {
      // Stability ceiling for the accuracy-driven controller: the largest dt
      // whose CFL number stays within cfl_max for the current velocity.
      integrator_->SetDtCeiling([this](double)
      {
         const double c = cfl_->Rate(integrator_->Velocity());
         return c > 0.0 ? params_.cfl_max / c
                : std::numeric_limits<double>::infinity();
      });
   }
   else
   {
      const double cfl = ConvectiveCflNumber();
      MFEM_VERIFY(cfl <= params_.cfl_max, "case: the convective CFL number "
                  << cfl << " at dt = " << integrator_->CurrentDt()
                  << " exceeds time.cfl_max = " << params_.cfl_max
                  << " (after an AMR event, smaller elements lower the stable "
                  "step) -- reduce dt or use adaptive stepping");
   }
}

double Case::ConvectiveCflNumber()
{
   EnsureSetup();
   ConvectiveCfl local(spaces_.Velocity(), rules_);
   const ConvectiveCfl& c = cfl_ ? *cfl_ : local;
   return c.Rate(integrator_->Velocity()) * integrator_->CurrentDt();
}

void Case::UpdateAmrCellData()
{
   if (!amr_fec_) { amr_fec_ = std::make_unique<L2_FECollection>(0, mesh_.Dimension()); }
   amr_level_.reset();
   amr_eta_.reset();
   amr_fes_ = std::make_unique<ParFiniteElementSpace>(&mesh_, amr_fec_.get());
   amr_eta_ = std::make_unique<ParGridFunction>(amr_fes_.get());
   amr_level_ = std::make_unique<ParGridFunction>(amr_fes_.get());
   GradientIndicator ind(spaces_.Velocity(), rules_);
   Vector g, eta;
   ind.Compute(integrator_->Velocity(), g);
   GradientIndicator::Eta(g, mesh_.Dimension(), eta);
   // One dof per element (L2, order 0): element e's value is dof e.
   double* E = amr_eta_->HostWrite();
   double* Lv = amr_level_->HostWrite();
   for (int e = 0; e < mesh_.GetNE(); ++e)
   {
      E[e] = eta(e);
      Lv[e] = mesh_.ncmesh ? mesh_.ncmesh->GetElementDepth(e) : 0.0;
   }
}

} // namespace incns
