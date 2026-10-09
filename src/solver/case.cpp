#include "solver/case.hpp"

#include "amr/gradient_indicator.hpp"
#include "amr/history_projection.hpp"
#include "amr/mesh_adapter.hpp"
#include "bc/boundary_names.hpp"
#include "util/json.hpp"
#include "util/profiler.hpp"

#include <array>
#include <cmath>
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
   bc_explicit_ = true;
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
   opts.outflow = params_.outflow;
   opts.convection_treatment = params_.convection_treatment;
   opts.oifs_cfl = params_.oifs_cfl;
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
      opts.ext_order = params_.ext_order;
      // The integrator's switch for CFL-controlled steps is cfl_target > 0.
      MFEM_VERIFY(params_.cfl_target > 0.0 && params_.dt_max >= 0.0,
                  "case: time.cfl_target must be > 0 and time.dt_max >= 0 "
                  "(choose the mode with Parameters::step_control)");
      opts.cfl_target = params_.CflSteps() ? params_.cfl_target : 0.0;
      opts.dt_max = params_.dt_max;
   }
   opts.nu = params_.nu;
   opts.dt = params_.dt;
   opts.t_final = params_.t_final;
   opts.order = params_.BdfOrder();
   opts.adaptive = (params_.step_control == StepControl::Error);
   opts.controller = params_.controller;
   opts.collocated_mass = params_.CollocatedMass();
   opts.grad_div = params_.grad_div;
   opts.grad_div_scale = params_.grad_div_scale;
   opts.velocity_prec = params_.velocity_prec;
   opts.amg_reuse = params_.amg_reuse;
   opts.schur = params_.schur;
   opts.cc = params_.cc;
   // Rotation in the LOR operator only exists for the LOR-AMG velocity PC:
   // solver.rotation_lor implies it (the case-level default is JacobiPCG).
   if (params_.rotation_in_lor) { opts.cc.a_pc = APC::LORAMG; }
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

   // The deck's boundary groups, unless a driver supplied the conditions:
   // every real boundary must then be in exactly one group (an uncovered
   // wall would silently become do-nothing).
   if (!bc_explicit_ && !params_.boundary_conditions.empty())
   {
      deck_bc_ = std::make_unique<DeckBoundaryConditions>(params_, mesh_);
      deck_bc_->Apply(own_bc_, nullptr, /*require_coverage=*/true);
   }
   // The force body by boundary name.
   for (const std::string& name : params_.forces.boundaries)
   {
      params_.forces.attributes.push_back(
                                  NamedBoundaryAttribute(params_, mesh_, name));
   }
   params_.forces.boundaries.clear();

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
   ne_ = mesh_.GetGlobalNE();
   monitor_ = std::make_unique<RunMonitor>(params_, mesh_.GetComm());
   // A restart past checkpoint.at_time must not overwrite that checkpoint.
   chk_at_written_ = params_.checkpoint.at_time > 0.0 &&
                     integrator_->Time() >= params_.checkpoint.at_time;
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
   RecordStep(t_prev, step_wall);
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
   MaybeAdaptInTime(t_prev);
   if (output_) { output_->MaybeSave(cycle_, integrator_->Time()); }
   MaybeLogDiagnostics(cycle_, integrator_->Time());
   if (params_.checkpoint.enabled &&
       cycle_ % params_.checkpoint.interval == 0)
   {
      WriteCheckpoint(params_.checkpoint.path);
   }
   if (params_.checkpoint.at_time > 0.0 && !chk_at_written_ &&
       integrator_->Time() >= params_.checkpoint.at_time)
   {
      chk_at_written_ = true;
      WriteCheckpoint(params_.checkpoint.path);
      if (Mpi::Root())
      {
         mfem::out << "[incns] checkpoint written at t = " << integrator_->Time()
                   << " to " << params_.checkpoint.path << std::endl;
      }
   }
   if (monitor_->ProgressDue(integrator_->Time()))
   {
      Vector where;
      const double cfl = (params_.equation == Equation::NavierStokes)
                         ? ConvectiveCflNumber(&where) : 0.0;
      monitor_->PrintProgress(cfl, where, ne_);
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
   while (!integrator_->Done() && !monitor_->Diverged()) { Step(); }
   monitor_->FlushHistory();
   if (monitor_->Diverged() && Mpi::Root())
   {
      mfem::out << "[incns] DIVERGED: " << monitor_->DivergenceReason() << std::endl;
   }
}

bool Case::Diverged() const { return monitor_ && monitor_->Diverged(); }

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
   return a.enabled && a.every_time == 0.0 && a.interval > 0 && cycle_ > 0
          && cycle_ % a.interval == 0
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
   ne_ = st.ne_after;
   if (st.ne_after != st.ne_before) { ++amr_events_; }
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
   // Both Navier-Stokes forms are CFL-limited: the IMEX convective form treats
   // velocity transport explicitly, and the semi-implicit rotational form --
   // energy-stable for any dt -- treats VORTICITY transport explicitly (curl
   // of w* x u^{n+1} is (u^{n+1}.grad) w*). Measured on DFG 2D-3 (2026-10-06,
   // docs/imex_vs_semi_implicit.md): both lose stability at the same step;
   // the rotational one silently, with bounded but wrong forces.
   const bool cfl_steps = params_.CflSteps();
   if ((params_.cfl_max <= 0.0 && !cfl_steps) ||
       params_.equation != Equation::NavierStokes)
   {
      return;
   }
   cfl_ = std::make_unique<ConvectiveCfl>(spaces_.Velocity(), rules_);
   if (cfl_steps)
   {
      // CFL-controlled steps: the integrator asks for the rate before every
      // step. Before the first step (not after an AMR event -- the imported
      // state carries its dt) the initial dt is capped to the target too.
      integrator_->SetCflRate([this]()
      {
         return cfl_->Rate(integrator_->Velocity());
      });
      if (integrator_->StepCount() == 0)
      {
         const double c = cfl_->Rate(integrator_->Velocity());
         if (c > 0.0 && params_.dt * c > params_.cfl_target)
         {
            integrator_->SetStepSize(params_.cfl_target / c);
         }
      }
      return; // dt is controlled: the fixed-step abort below does not apply
   }
   if (params_.step_control == StepControl::Error)
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
                  "step) -- reduce dt or use time.step_control cfl or error");
   }
}

double Case::ConvectiveCflNumber(Vector* where)
{
   EnsureSetup();
   ConvectiveCfl local(spaces_.Velocity(), rules_);
   const ConvectiveCfl& c = cfl_ ? *cfl_ : local;
   const double rate = where ? c.RateAndLocation(integrator_->Velocity(), *where)
                       : c.Rate(integrator_->Velocity());
   return rate * integrator_->CurrentDt();
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


void Case::RecordStep(double t_prev, double step_wall)
{
   StepRecord r;
   r.t = integrator_->Time();
   r.dt = r.t - t_prev;
   r.iterations = integrator_->LastIterations();
   r.substeps = integrator_->LastOifsSubsteps();
   r.elements = ne_;
   r.step_wall = step_wall;
   if (body_force_ && params_.forces.statistics)
   {
      const Vector C = ForceCoefficients();
      r.has_forces = true;
      r.cd = C(0);
      r.cl = (C.Size() > 1) ? C(1) : 0.0;
   }
   Vector u(spaces_.Velocity().GetTrueVSize());
   integrator_->Velocity().GetTrueDofs(u);
   r.velocity_norm = std::sqrt(InnerProduct(mesh_.GetComm(), u, u));
   monitor_->Record(r);
}

void Case::MaybeAdaptInTime(double t_prev)
{
   const AmrParameters& a = params_.amr;
   if (!a.enabled || a.every_time <= 0.0 || integrator_->Done() ||
       integrator_->StepCount() < 2) { return; }
   const double t = integrator_->Time();
   // An event when the step crossed a multiple of every_time (one per step,
   // however many multiples a long step crossed).
   if (std::floor(t / a.every_time + 1e-9) <=
       std::floor(t_prev / a.every_time + 1e-9)) { return; }
   if (t < a.start_time - 1e-12 || (a.end_time > 0.0 && t > a.end_time)) { return; }
   const int passes = amr_time_started_ ? a.passes_per_event :
                      a.first_event_passes;
   amr_time_started_ = true;
   if (body_force_) { force_cache_ = BodyForceVector(); }
   for (int k = 0; k < passes; ++k)
   {
      const AdaptStats st = Adapt();
      if (st.ne_after == st.ne_before) { break; }
   }
}

namespace
{
// A scalar field's value at a point (collective: every rank passes the same
// point; the owning rank's value is reduced).
double PointValue(ParMesh& mesh, ParGridFunction& f,
                  const std::array<double, 3>& x)
{
   f.HostRead(); // evaluated on the host below
   const int dim = mesh.Dimension();
   DenseMatrix pts(dim, 1);
   for (int d = 0; d < dim; ++d) { pts(d, 0) = x[d]; }
   Array<int> elems;
   Array<IntegrationPoint> ips;
   mesh.FindPoints(pts, elems, ips, false);
   double buf[2] = {0.0, 0.0};
   if (elems[0] >= 0)
   {
      buf[0] = f.GetValue(elems[0], ips[0]);
      buf[1] = 1.0;
   }
   MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return buf[1] > 0.0 ? buf[0] / buf[1] : std::nan("");
}

const char* GeometryName(MeshGeometry g)
{
   switch (g)
   {
      case MeshGeometry::Box: return "box";
      case MeshGeometry::SquareCylinder: return "square_cylinder";
      case MeshGeometry::CylinderChannel: return "cylinder_channel";
   }
   return "?";
}
} // namespace

void Case::WriteSummary(const std::string& path, const std::string& status)
{
   EnsureSetup();
   Json j;
   j["status"].Set(status);
   if (monitor_->Diverged()) { j["divergence"].Set(monitor_->DivergenceReason()); }

   Json& c = j["case"];
   const bool nse = (params_.equation == Equation::NavierStokes);
   c["equation"].Set(nse ? "navier_stokes" : "stokes");
   c["geometry"].Set(GeometryName(params_.geometry));
   c["dim"].Set(params_.mesh.dim);
   c["nu"].Set(params_.nu); // the Reynolds number depends on the case's scales
   c["order_u"].Set(params_.order_u);
   c["order_p"].Set(params_.order_p);
   if (nse)
   {
      c["convection"].Set(params_.convection_treatment ==
                          ConvectionTreatment::Oifs ? "oifs" : "imex");
   }
   c["bdf_order"].Set(params_.BdfOrder());
   c["step_control"].Set(params_.step_control == StepControl::Cfl ? "cfl" :
                         params_.step_control == StepControl::Fixed ? "fixed" : "error");
   if (params_.CflSteps()) { c["cfl_target"].Set(params_.cfl_target); }
   int np = 1;
   MPI_Comm_size(mesh_.GetComm(), &np);
   c["mpi_ranks"].Set(np);
   c["mfem"].Set(GetGitStr());

   Json& t = j["time"];
   t["t"].Set(Time());
   t["t_final"].Set(params_.t_final);
   t["steps"].Set(monitor_->Steps());
   t["dt_min"].Set(monitor_->DtMin());
   t["dt_max"].Set(monitor_->DtMax());
   t["wall_seconds"].Set(monitor_->Wall());
   t["solve_seconds"].Set(monitor_->StepWall());

   Json& m = j["mesh"];
   m["elements"].Set(ne_);
   m["velocity_dofs"].Set(static_cast<long long>
                          (spaces_.Velocity().GlobalTrueVSize()));
   m["pressure_dofs"].Set(static_cast<long long>
                          (spaces_.Pressure().GlobalTrueVSize()));
   m["amr_events"].Set(amr_events_);

   Json& d = j["diagnostics"];
   d["kinetic_energy"].Set(KineticEnergy());
   d["divergence_norm"].Set(DivergenceNorm());

   // Computed values the reference section may compare against.
   std::vector<std::pair<std::string, double>> got;
   if (body_force_ && integrator_->StepCount() > 0)
   {
      Json& f = j["forces"];
      const Vector C = ForceCoefficients();
      f["cd"].Set(C(0));
      f["cl"].Set(C.Size() > 1 ? C(1) : 0.0);
      got.push_back({"cd", C(0)});
      got.push_back({"cl", C.Size() > 1 ? C(1) : 0.0});
      if (params_.forces.statistics)
      {
         const ForceStats fs = monitor_->Forces().Compute(
                                  params_.forces.average_periods);
         f["cd_max"].Set(fs.cd_max);
         f["t_cd_max"].Set(fs.t_cd_max);
         f["cl_max"].Set(fs.cl_max);
         f["t_cl_max"].Set(fs.t_cl_max);
         got.push_back({"cd_max", fs.cd_max});
         got.push_back({"t_cd_max", fs.t_cd_max});
         got.push_back({"cl_max", fs.cl_max});
         got.push_back({"t_cl_max", fs.t_cl_max});
         Json& per = f["periodic"];
         per["found"].Set(fs.periodic);
         if (fs.periodic)
         {
            // Strouhal number St = D / (U T): in 2D the reference "area" is a length.
            const double st = params_.forces.reference_area /
                              (params_.forces.reference_velocity * fs.period);
            per["periods"].Set(fs.periods);
            per["t_start"].Set(fs.t_start);
            per["t_end"].Set(fs.t_end);
            per["cd_mean"].Set(fs.cd_mean);
            per["cl_mean"].Set(fs.cl_mean);
            per["cl_rms"].Set(fs.cl_rms);
            per["period"].Set(fs.period);
            per["period_spread"].Set(fs.period_spread);
            if (params_.mesh.dim == 2) { per["strouhal"].Set(st); }
            got.push_back({"cd_mean", fs.cd_mean});
            got.push_back({"cl_mean", fs.cl_mean});
            got.push_back({"cl_rms", fs.cl_rms});
            if (params_.mesh.dim == 2) { got.push_back({"strouhal", st}); }
         }
      }
   }
   if (params_.probes.pressure_difference.size() == 2)
   {
      const double dp = PointValue(mesh_, Pressure(),
                                   params_.probes.pressure_difference[0]) -
                        PointValue(mesh_, Pressure(), params_.probes.pressure_difference[1]);
      j["probes"]["pressure_difference"].Set(dp);
      got.push_back({"pressure_difference", dp});
   }

   const ReferenceValues& rv = params_.reference;
   const std::vector<std::pair<std::string, double>> refs =
   {
      {"cd", rv.cd}, {"cl", rv.cl}, {"cd_mean", rv.cd_mean}, {"cl_mean", rv.cl_mean},
      {"cl_rms", rv.cl_rms}, {"strouhal", rv.strouhal}, {"cd_max", rv.cd_max},
      {"t_cd_max", rv.t_cd_max}, {"cl_max", rv.cl_max}, {"t_cl_max", rv.t_cl_max},
      {"pressure_difference", rv.pressure_difference}
   };
   for (const auto& [key, ref] : refs)
   {
      if (!std::isfinite(ref)) { continue; }
      Json& e = j["reference"][key];
      e["reference"].Set(ref);
      for (const auto& [k2, val] : got)
      {
         if (k2 != key) { continue; }
         e["value"].Set(val);
         const bool is_time = (key.rfind("t_", 0) == 0);
         if (is_time) { e["abs_error"].Set(std::abs(val - ref)); }
         else { e["rel_error"].Set(std::abs(val - ref) / std::abs(ref)); }
      }
   }

   if (Mpi::Root())
   {
      const std::filesystem::path out(path);
      if (out.has_parent_path()) { std::filesystem::create_directories(out.parent_path()); }
      std::ofstream f(path);
      MFEM_VERIFY(f.good(), "case: cannot write the summary " << path);
      f << j.Dump();
   }
}

} // namespace incns
