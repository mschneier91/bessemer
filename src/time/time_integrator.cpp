#include "time/cfl.hpp"
#include "time/time_integrator.hpp"

#include "post/pressure_mean.hpp"
#include "time/multistep_coeffs.hpp"
#include "util/device.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
     rotational_(opts.convection &&
                 opts.convective_form == ConvectiveForm::Rotational),
     u_(&spaces.Velocity()), p_(&spaces.Pressure()),
     u2_scratch_(&spaces.Velocity()), p2_scratch_(&spaces.Pressure()),
     u3_scratch_(&spaces.Velocity()), p3_scratch_(&spaces.Pressure()),
     w_star_(&spaces.Velocity()), p_static_(&spaces.Pressure()),
     ke_(&spaces.Pressure()),
     dt_(opts.dt)
{
   INCNS_PROFILE("time_integrator::setup");
   scratch_ = std::make_unique<ElementScratch>(spaces.Velocity());

   MFEM_VERIFY(opts_.dt > 0.0, "time_integrator: dt must be positive");
   MFEM_VERIFY(opts_.order == 2 || opts_.order == 3,
               "time_integrator: order must be 2 or 3");

   MFEM_VERIFY(opts_.convection ||
               opts_.convective_form == ConvectiveForm::Convective,
               "time_integrator: a convective form other than Convective "
               "needs convection (the Navier-Stokes equation)");
   MFEM_VERIFY(opts_.ext_order == 2 || opts_.ext_order == 3,
               "time_integrator: ext_order must be 2 or 3");
   MFEM_VERIFY(opts_.cfl_target >= 0.0 && opts_.dt_max >= 0.0 &&
               opts_.cfl_growth >= 1.0 && opts_.cfl_hysteresis >= 0.0,
               "time_integrator: bad CFL step-control settings");
   MFEM_VERIFY(opts_.convection_treatment != ConvectionTreatment::Auto,
               "time_integrator: resolve ConvectionTreatment::Auto first "
               "(Parameters::Convection())");
   MFEM_VERIFY(opts_.convection_treatment == ConvectionTreatment::Imex ||
               (opts_.convection && !rotational_), "time_integrator: OIFS "
               "needs the convective form of the Navier-Stokes equation");
   MFEM_VERIFY(!opts_.retry_on_failure ||
               (opts_.retry_factor > 0.0 && opts_.retry_factor < 1.0 &&
                opts_.retry_hold >= 0 && opts_.retry_max >= 1 &&
                opts_.retry_dt_min >= 0.0),
               "time_integrator: bad step-retry settings");
   MFEM_VERIFY(!(opts_.adaptive && opts_.cfl_target > 0.0),
               "time_integrator: choose error-controlled (adaptive) OR "
               "CFL-controlled steps, not both");

   // NSE (2.2): the dealiased convection operator, built only when asked -- the
   // Stokes path then allocates nothing and is bit-for-bit unchanged. IMEX, so
   // this never touches the implicit block solve. The ROTATIONAL form has no
   // explicit part: its lagged-vorticity term lives in every solver's
   // momentum block, reading w_star_ (zero until the first step sets it).
   if (opts_.convection && !rotational_ &&
       opts_.convection_treatment == ConvectionTreatment::Oifs)
   {
      // OIFS: the BDF history is advected (time/oifs.hpp) -- no explicit
      // convection on the right-hand side.
      // No error control under OIFS (human 2026-10-09): its step is chosen
      // as a multiple of the CFL step anyway, so an estimator's second solve
      // per step buys nothing.
      MFEM_VERIFY(!opts_.adaptive, "time_integrator: OIFS supports fixed and "
                  "CFL-controlled steps, not error control (by design: its "
                  "step is a CFL multiple)");
      // oifs_ itself is built below, once the BDF step's mass exists.
      // The convection form serves the Dirichlet nodes the wind enters
      // through (OifsBoundaryValues, the starter): a mesh with none anywhere
      // (periodic boxes) skips it -- its applies were discarded there, and at
      // velocity order 7 its 3k rule (11 points per direction) is more than
      // MFEM's partial assembly launches on a GPU.
      int walls = bc_.EssentialTrueDofs().Size() > 0 ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &walls, 1, MPI_INT, MPI_MAX,
                    spaces_.Velocity().GetComm());
      oifs_walls_ = (walls != 0);
      if (oifs_walls_) { oifs_conv_ = std::make_unique<Convection>(spaces_, rules_); }
      // Directional do-nothing: explicit and extrapolated in the BDF step,
      // exactly as IMEX treats it (SubtractOifsDirectionalDoNothing). Inside
      // the substeps it is consistent but stiff at backflow nodes (rate
      // ~6|u.n|/h): Braack & Mucha's square at CFL 2 gave j1 2.3% off the
      // paper, converging only as CFL -> 0.25 (measured 2026-10-09).
      if (opts_.outflow == OutflowCondition::Directional && bc_.HasOutflow())
      {
         oifs_ddn_ = std::make_unique<Convection>(spaces_, rules_);
         oifs_ddn_->EnableDirectionalDoNothing(bc_.OutflowAttributes());
      }
   }
   else if (opts_.convection && !rotational_)
   {
      INCNS_PROFILE("time_integrator::convection_setup");
      convection_ = std::make_unique<Convection>(spaces_, rules_);
      // Directional do-nothing: the boundary part of the convective flux on
      // the outflow, explicit and extrapolated together with N.
      if (opts_.outflow == OutflowCondition::Directional && bc_.HasOutflow())
      {
         convection_->EnableDirectionalDoNothing(bc_.OutflowAttributes());
      }
   }
   // The lagged velocity exists for the rotational form only.
   if (rotational_) { w_star_ = 0.0; }
   else { static_cast<Vector&>(w_star_).Destroy(); } // frees the data only
   p_static_ = 0.0;
   if (rotational_)
   {
      kinetic_head_ = std::make_unique<KineticHeadInterpolator>(
                         spaces_.Velocity(), spaces_.Pressure());
   }

   // The trapezoidal starter solver is built on first use (Trap()): a march
   // that resumes with its history (a restart, an adapted mesh) never
   // needs it.

   if (opts_.convection && !rotational_ &&
       opts_.convection_treatment == ConvectionTreatment::Oifs)
   {
      // The substeps invert the BDF step's own mass (every solver's Mass() is
      // the same plain M; the right-hand side uses trap_'s). OifsAdvector
      // aborts if its inverse does not reproduce it.
      oifs_ = std::make_unique<OifsAdvector>(spaces_, rules_, bc_,
                                             opts_.oifs_cfl, Trap().Blocks().Mass(),
                                             Trap().Blocks().MassDiagonal(),
                                             scratch_.get());
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
   // Nonconforming (AMR) mesh: ProjectCoefficient writes the nodal value at
   // hanging nodes, not the constrained interpolant -- make u_ conforming
   // (P R u). Conforming meshes skip it to stay bitwise unchanged.
   if (!spaces_.Velocity().Conforming()) { u_.SetFromTrueDofs(u_true); }
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
   last_solver_ = nullptr; // no step taken from this state yet
   u_.SetFromTrueDofs(hist_[0]);
   if (pressure) { p_.SetFromTrueDofs(*pressure); }
   else { p_ = 0.0; }
   // p_ is the solver's variable (the Bernoulli head in the rotational form);
   // the reported static pressure follows from it and u_.
   if (rotational_) { UpdateStaticPressure(); }
}

IntegratorState StokesTimeIntegrator::ExportState() const
{
   IntegratorState s;
   s.u_hist.assign(hist_.begin(), hist_.end());
   s.times.assign(hist_times_.begin(), hist_times_.end());
   s.completed_steps = step_count_;
   s.next_dt = dt_;
   s.pressure.SetSize(spaces_.Pressure().GetTrueVSize());
   s.pressure.UseDevice(true);
   p_.GetTrueDofs(s.pressure);
   s.has_controller = (controller_ != nullptr);
   if (controller_) { s.controller = controller_->ExportState(); }
   return s;
}

void StokesTimeIntegrator::ImportState(const IntegratorState& s)
{
   SetHistory(s.u_hist, s.times, s.completed_steps, s.next_dt,
              s.pressure.Size() ? &s.pressure : nullptr);
   MFEM_VERIFY(s.has_controller == (controller_ != nullptr),
               "time_integrator: the state's adaptive mode does not match");
   if (controller_) { controller_->ImportState(s.controller); }
}

void StokesTimeIntegrator::MomentumResidual(Vector& r) const
{
   MFEM_VERIFY(last_solver_, "time_integrator: MomentumResidual() needs a "
               "step since construction / the last state import");
   const int n_u = spaces_.Velocity().GetTrueVSize();
   Vector u_true(n_u), p_true(spaces_.Pressure().GetTrueVSize()), tmp(n_u);
   u_true.UseDevice(true);
   p_true.UseDevice(true);
   tmp.UseDevice(true);
   u_.GetTrueDofs(u_true);
   p_.GetTrueDofs(p_true); // physical p on both Schur paths (P if rotational)
   MFEM_VERIFY(opts_.keep_rhs, "time_integrator: MomentumResidual() needs "
               "keep_rhs (the raw right-hand side is not kept)");
   StokesOperator& op = last_solver_->Blocks();
   r.SetSize(n_u);
   r.UseDevice(true);
   op.MomentumUnconstrained().Mult(u_true, r);           // A u
   if (op.Rotation())
   {
      op.RotationUnconstrained().Mult(u_true, tmp);      // + N u
      r += tmp;
   }
   op.DivergenceUnconstrained().MultTranspose(p_true, tmp);
   r -= tmp;                                             // - B^T p
   r -= last_b_;                                         // - b
}

void StokesTimeIntegrator::SetDtCeiling(AdaptiveController::DtCeilingFn ceiling)
{
   MFEM_VERIFY(controller_, "time_integrator: dt ceiling requires adaptive mode");
   controller_->SetDtCeiling(std::move(ceiling));
}

void StokesTimeIntegrator::SetStepSize(double dt)
{
   MFEM_VERIFY(dt > 0.0, "time_integrator: step size must be positive");
   MFEM_VERIFY(step_count_ == 0 || opts_.cfl_target > 0.0, "time_integrator: "
               "a fixed-step march never changes dt after it has started");
   dt_ = dt;
}

std::size_t StokesTimeIntegrator::ExtrapolationOrder() const
{
   const std::size_t want = static_cast<std::size_t>(
                               std::max(opts_.ext_order, opts_.order == 3 ? 3 : 2));
   return std::min<std::size_t>(hist_.size(), want);
}

void StokesTimeIntegrator::ChooseCflStep()
{
   INCNS_PROFILE("time_integrator::cfl_step");
   MFEM_VERIFY(cfl_rate_, "time_integrator: CFL-controlled steps need "
               "SetCflRate");
   // With OIFS, through the advector's per-level cache (the same rate, same
   // definition and space): the substeps then reuse it instead of evaluating
   // this velocity again.
   const double c = oifs_ ? oifs_->RateAt(u_, t_) : cfl_rate_();
   RequireFiniteRate(c, t_);
   double dt = dt_;
   const double dt_cfl = (c > 0.0) ? opts_.cfl_target / c
                         : std::numeric_limits<double>::infinity();
   if (dt_cfl < dt)
   {
      dt = dt_cfl; // shrink at once: stability
   }
   else if (hold_left_ == 0 && dt_cfl > dt * (1.0 + opts_.cfl_hysteresis))
   {
      dt = std::min(dt_cfl, dt * opts_.cfl_growth);
   }
   if (opts_.dt_max > 0.0) { dt = std::min(dt, opts_.dt_max); }
   // Land on t_final: never overshoot, and avoid a sliver of a last step by
   // splitting the remainder evenly when it is under two steps.
   const double rest = opts_.t_final - t_;
   if (rest <= dt) { dt = rest; }
   else if (rest < 2.0 * dt) { dt = 0.5 * rest; }
   dt_ = dt;
}

bool StokesTimeIntegrator::Done() const
{
   if (opts_.adaptive || opts_.cfl_target > 0.0)
   {
      return t_ >= opts_.t_final - 1e-12 * std::max(1.0, std::abs(opts_.t_final));
   }
   return t_ >= opts_.t_final - 0.5 * opts_.dt;
}

StokesSolver& StokesTimeIntegrator::Trap()
{
   if (trap_) { return *trap_; }
   INCNS_PROFILE("time_integrator::build_starter");
   // Trapezoidal starter: [(1/dt) M + (nu/2) K] u^1 = ... -- realized by
   // halving the viscosity passed to the solver (the Schur scale follows).
   StokesSolverOptions so;
   so.nu = 0.5 * opts_.nu;
   so.collocated_mass = opts_.collocated_mass;
   so.scratch = scratch_.get();
   so.grad_div = opts_.grad_div;
   so.grad_div_scale = opts_.grad_div_scale;
   so.grad_div_nu = opts_.grad_div_nu;
   so.grad_div_nu_ref = opts_.nu; // gamma from the fluid's nu
   so.compact_diffusion = opts_.compact_diffusion;
   so.fuse_grad_div = opts_.fuse_grad_div;
   so.velocity_prec = opts_.velocity_prec;
   so.amg_reuse = opts_.amg_reuse;
   so.schur = opts_.schur;
   so.cc = opts_.cc;
   so.mass_coeff = 1.0 / dt_;
   if (rotational_)
   {
      // Trapezoidal split, like the viscous term: alpha = 1/2 here, the
      // other half of N goes to the right-hand side explicitly.
      so.lagged_velocity = &w_star_;
      so.rotation_alpha = 0.5;
      so.rotation_pc = opts_.rotation_pc;
      so.rotation_in_lor = opts_.rotation_in_lor;
      so.rotation_schur = opts_.rotation_schur;
      so.rotation_diagnostics = opts_.rotation_diagnostics;
   }
   so.rtol = opts_.rtol;
   so.atol = opts_.atol;
   so.tolerance_reference = opts_.tolerance_reference;
   so.initial_guess = opts_.initial_guess;
   so.projection_k = opts_.projection_k;
   so.projection_rebuild_threshold = opts_.projection_rebuild_threshold;
   so.max_iter = opts_.max_iter;
   so.kdim = opts_.kdim;
   so.print_level = opts_.print_level;
   so.outer = opts_.outer;
   so.uzawa = opts_.uzawa;
   trap_ = std::make_unique<StokesSolver>(spaces_, rules_, bc_, so);
   trap_c0_ = 1.0 / dt_;
   ApplyPendingGuess("trap", *trap_);
   return *trap_;
}

void StokesTimeIntegrator::ExportSolverStates(
   std::map<std::string, SolverGuessState>& states) const
{
   states.clear();
   if (trap_) { trap_->ExportGuessState(states["trap"]); }
   if (bdf2_.solver) { bdf2_.solver->ExportGuessState(states["bdf2"]); }
   if (bdf3_.solver) { bdf3_.solver->ExportGuessState(states["bdf3"]); }
}

void StokesTimeIntegrator::ImportSolverStates(
   std::map<std::string, SolverGuessState>& states)
{
   pending_guess_.swap(states);
   states.clear();
   if (trap_) { ApplyPendingGuess("trap", *trap_); }
   if (bdf2_.solver) { ApplyPendingGuess("bdf2", *bdf2_.solver); }
   if (bdf3_.solver) { ApplyPendingGuess("bdf3", *bdf3_.solver); }
}

void StokesTimeIntegrator::ApplyPendingGuess(const std::string& name,
      StokesSolver& solver)
{
   auto it = pending_guess_.find(name);
   if (it == pending_guess_.end()) { return; }
   solver.ImportGuessState(it->second);
   pending_guess_.erase(it);
}

Operator& StokesTimeIntegrator::PlainMass()
{
   if (trap_) { return trap_->Blocks().Mass(); }
   if (bdf2_.solver) { return bdf2_.solver->Blocks().Mass(); }
   if (bdf3_.solver) { return bdf3_.solver->Blocks().Mass(); }
   return Trap().Blocks().Mass();
}

bool StokesTimeIntegrator::AdoptSolver(SolverCache& cache, double c0)
{
   if (opts_.adaptive || rotational_) { return false; }
   SolverCache* other = (&cache == &bdf2_) ? &bdf3_ : &bdf2_;
   std::unique_ptr<StokesSolver>* from =
      trap_ ? &trap_ : (other->solver ? &other->solver : nullptr);
   if (!from || !(*from)->CanChangeViscosity()) { return false; }
   INCNS_PROFILE("time_integrator::adopt_solver");
   cache.solver = std::move(*from); // the object stays put: last_solver_ and
   // the OIFS advector's borrowed mass remain valid
   other->c0 = -1.0;
   // The starter's (nu/2) K, for its right-hand side only.
   cache.solver->Blocks().ReleaseViscousUnconstrained();
   cache.solver->Refresh(c0, opts_.nu);
   cache.c0 = c0;
   return true;
}

void StokesTimeIntegrator::ReleaseStarter()
{
   if (!trap_ || oifs_ || opts_.adaptive) { return; }
   if (last_solver_ == trap_.get()) { last_solver_ = nullptr; }
   trap_.reset();
}

StokesSolver& StokesTimeIntegrator::EnsureBdfSolver(SolverCache& cache,
      double c0)
{
   const char* name = (&cache == &bdf2_) ? "bdf2" : "bdf3";
   if (!cache.solver && AdoptSolver(cache, c0))
   {
      ApplyPendingGuess(name, *cache.solver);
      return *cache.solver;
   }
   if (!cache.solver)
   {
      // No solver to adopt: the starter goes first, so that it and the new
      // solver never coexist.
      ReleaseStarter();
      INCNS_PROFILE("time_integrator::build_solver");
      StokesSolverOptions so;
      so.nu = opts_.nu;
      so.collocated_mass = opts_.collocated_mass;
      so.scratch = scratch_.get();
      so.grad_div = opts_.grad_div;
      so.grad_div_scale = opts_.grad_div_scale;
      so.grad_div_nu = opts_.grad_div_nu;
      so.grad_div_nu_ref = opts_.nu; // gamma from the fluid's nu
      so.compact_diffusion = opts_.compact_diffusion;
      so.fuse_grad_div = opts_.fuse_grad_div;
      so.velocity_prec = opts_.velocity_prec;
      so.amg_reuse = opts_.amg_reuse;
      so.schur = opts_.schur;
      so.cc = opts_.cc;
      so.mass_coeff = c0; // the leading BDF weight beta0/dt -- exact match
      if (rotational_)
      {
         so.lagged_velocity = &w_star_;
         so.rotation_alpha = 1.0;
         so.rotation_pc = opts_.rotation_pc;
         so.rotation_in_lor = opts_.rotation_in_lor;
         so.rotation_schur = opts_.rotation_schur;
         so.rotation_diagnostics = opts_.rotation_diagnostics;
      }
      so.rtol = opts_.rtol;
      so.atol = opts_.atol;
      so.tolerance_reference = opts_.tolerance_reference;
      so.initial_guess = opts_.initial_guess;
      so.projection_k = opts_.projection_k;
      so.projection_rebuild_threshold = opts_.projection_rebuild_threshold;
      so.max_iter = opts_.max_iter;
      so.kdim = opts_.kdim;
      so.print_level = opts_.print_level;
      so.outer = opts_.outer;
      so.uzawa = opts_.uzawa;
      cache.solver = std::make_unique<StokesSolver>(spaces_, rules_, bc_, so);
      cache.c0 = c0;
      ApplyPendingGuess(name, *cache.solver);
   }
   else if (std::abs(c0 - cache.c0) > 1e-12 * std::abs(c0))
   {
      // Delta-t changed: refresh only the momentum block, not the whole solver.
      cache.solver->Refresh(c0);
      cache.c0 = c0;
   }
   return *cache.solver;
}

namespace
{
// A device-aware work vector of size n (no reallocation when it has it).
void Scratch(Vector& v, int n)
{
   v.UseDevice(true);
   v.SetSize(n);
}
} // namespace

void StokesTimeIntegrator::AssembleForcing(double t, Vector& F)
{
   INCNS_PROFILE("time_integrator::forcing");
   // A constant coefficient (a body force, or none): assembled once, reused
   // while the constant stays the same.
   const auto* cf = dynamic_cast<const VectorConstantCoefficient*>(&forcing_);
   if (cf && cf->GetVec().Normlinf() == 0.0)
   {
      // No body force (the default): F = 0, and nothing is kept for it.
      Scratch(F, spaces_.Velocity().GetTrueVSize());
      F = 0.0;
      return;
   }
   if (cf && forcing_value_.Size() == cf->GetVec().Size())
   {
      bool same = true;
      for (int i = 0; i < forcing_value_.Size(); ++i)
      {
         same = same && forcing_value_(i) == cf->GetVec()(i);
      }
      if (same)
      {
         Scratch(F, forcing_const_.Size());
         F = forcing_const_;
         return;
      }
   }
   forcing_.SetTime(t);
   ParLinearForm f_form(&spaces_.Velocity());
   auto* fi = new VectorDomainLFIntegrator(forcing_);
   const int dim = spaces_.Dim();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   fi->SetIntRule(&rules_.Get(geom, 2 * spaces_.OrderU() + 2));
   f_form.AddDomainIntegrator(fi);
   f_form.UseFastAssembly(true); // device-ready assembly path (project policy)
   f_form.Assemble();
   // Straight into F (MFEM-owned), not via ParallelAssemble()'s hypre-backed
   // HypreParVector -- see MassWeightedMean (post/pressure_mean.cpp) for the
   // debug-device page-protection fault that pattern causes.
   F.SetSize(spaces_.Velocity().GetTrueVSize());
   f_form.ParallelAssemble(F);
   if (cf)
   {
      Scratch(forcing_const_, F.Size());
      forcing_const_ = F;
      forcing_value_ = cf->GetVec(); // host copy of the constant
   }
}

void StokesTimeIntegrator::AssembleBdfRhs(const std::vector<double>& c,
      double t_new, Vector& b, int ext)
{
   INCNS_PROFILE("time_integrator::bdf_rhs");
   // BDF-k: M sum_j c_j u^{n+1-j} + nu K u^{n+1} + B^T p = F^{n+1}
   // => A u^{n+1} - B^T p = F^{n+1} - M sum_{j>=1} c_j u^{n+1-j}.
   const int n_u = spaces_.Velocity().GetTrueVSize();
   AssembleForcing(t_new, b);
   Vector& combo = rhs_combo_;
   Vector& tmp = rhs_tmp_;
   Scratch(combo, n_u);
   Scratch(tmp, n_u);
   if (oifs_)
   {
      // OIFS: the history levels advected to t_new replace both the plain
      // history and the explicit convection.
      INCNS_PROFILE("oifs::advect");
      oifs_->Advect(c, hist_, hist_times_, t_new,
                    static_cast<int>(ExtrapolationOrder()), combo);
      OifsBoundaryValues(c, t_new, combo);
      PlainMass().Mult(combo, tmp);
      b -= tmp;
      SubtractOifsDirectionalDoNothing(t_new, b);
      return;
   }
   combo = 0.0;
   for (std::size_t j = 1; j < c.size(); ++j)
   {
      combo.Add(c[j], hist_[j - 1]);
   }
   // Any solver's Mass() is the same plain M (mass_coeff scales only the
   // momentum block).
   PlainMass().Mult(combo, tmp);
   b -= tmp;

   SubtractConvection(t_new, b, ext);
}

void StokesTimeIntegrator::OifsBoundaryValues(const std::vector<double>& c,
      double t_new, Vector& phi)
{
   INCNS_PROFILE("oifs::boundary_values");
   if (!oifs_walls_) { return; } // no Dirichlet nodes on any rank
   // Collective (N applies): every rank runs it, imposed dofs or not.
   const int n_u = spaces_.Velocity().GetTrueVSize();
   const std::size_t k = c.size() - 1;
   Vector& bd = oifs_bd_;
   Vector& q = oifs_q_;
   Vector& nj = oifs_nj_;
   Scratch(bd, n_u);
   Scratch(q, n_u);
   Scratch(nj, n_u);
   bd = 0.0;
   double ctau = 0.0;
   for (std::size_t j = 1; j <= k; ++j)
   {
      bd.Add(c[j], hist_[j - 1]);
      ctau += c[j] * (t_new - hist_times_[j - 1]);
   }
   const std::size_t m = ExtrapolationOrder();
   std::vector<double> times(hist_times_.begin(), hist_times_.begin() + m);
   const std::vector<double> g = ExtrapolationWeights(t_new, times);
   q = 0.0;
   for (std::size_t l = 0; l < m; ++l)
   {
      oifs_conv_->Mult(hist_[l], nj);
      q.Add(g[l], nj);
   }
   oifs_->MassInverse(q, q); // (u.grad)u at the nodes
   bd.Add(-ctau, q);
   const Array<int>& imp = oifs_->ImposedDofs();
   Vector& vals = oifs_vals_;
   vals.UseDevice(true); // the gather stays on the device
   bd.GetSubVector(imp, vals);
   phi.SetSubVector(imp, vals);
}

void StokesTimeIntegrator::SubtractOifsDirectionalDoNothing(double t_new,
      Vector& b)
{
   if (!oifs_ddn_) { return; }
   // EXT of the history the step has (EXT1 on the starter).
   const std::size_t m = ExtrapolationOrder();
   std::vector<double> times(hist_times_.begin(), hist_times_.begin() + m);
   const std::vector<double> g = ExtrapolationWeights(t_new, times);
   Vector y;
   for (std::size_t l = 0; l < m; ++l)
   {
      oifs_ddn_->MultDirectionalDoNothing(hist_[l], y);
      b.Add(-g[l], y);
   }
}

void StokesTimeIntegrator::SubtractConvection(double t_new, Vector& b, int ext)
{
   if (!convection_) { return; }
   INCNS_PROFILE("time_integrator::convection");

   // Match the extrapolation order to the history actually available (EXT1
   // on the first step, EXT2 on the second, then ext_order): a fixed order
   // would read history entries that do not exist yet during the startup ramp.
   const std::size_t k = (ext > 0) ? std::min<std::size_t>(hist_.size(),
                         static_cast<std::size_t>(ext))
                         : ExtrapolationOrder();
   MFEM_VERIFY(k >= 1, "time_integrator: convection needs velocity history");

   std::vector<double> times(hist_times_.begin(), hist_times_.begin() + k);
   const std::vector<double> g = ExtrapolationWeights(t_new, times);

   const int n_u = spaces_.Velocity().GetTrueVSize();
   Vector& nu_j = conv_nj_;
   Scratch(nu_j, n_u);
   // Device-aware Vector ops throughout -- NEVER a hand-written mfem::forall
   // (CLAUDE.md: it silently becomes a host loop over device pointers). MFEM's
   // navier miniapp combines its AB history with a forall; this is the
   // device-safe equivalent of that step.
   for (std::size_t j = 0; j < k; ++j)
   {
      convection_->Mult(hist_[j], nu_j);
      // Convection is on the LHS of the momentum equation, so it leaves the
      // right-hand side with a minus sign.
      b.Add(-g[j], nu_j);
   }
}

void StokesTimeIntegrator::UpdateLaggedVelocity(double t_new, int ext)
{
   INCNS_PROFILE("time_integrator::lagged_velocity");
   // Same order matching as SubtractConvection (ExtrapolationOrder). The
   // term does no work for ANY w* (skew), so the extrapolation affects
   // accuracy only -- and stability of the (explicit) vorticity transport.
   const std::size_t k = (ext > 0) ? std::min<std::size_t>(hist_.size(),
                         static_cast<std::size_t>(ext))
                         : ExtrapolationOrder();
   MFEM_VERIFY(k >= 1, "time_integrator: rotation needs velocity history");
   std::vector<double> times(hist_times_.begin(), hist_times_.begin() + k);
   const std::vector<double> g = ExtrapolationWeights(t_new, times);
   Vector w(spaces_.Velocity().GetTrueVSize());
   w = 0.0;
   for (std::size_t j = 0; j < k; ++j) { w.Add(g[j], hist_[j]); }
   w_star_.SetFromTrueDofs(w); // distributed: the rotation setup reads it
}

void StokesTimeIntegrator::UpdateStaticPressure()
{
   // The rotational solve yields the Bernoulli head P = p + 1/2|u|^2; report
   // static p = P - I(1/2|u_h|^2) (nodal interpolant on the pressure space --
   // continuous since u_h is), computed on the device. Every step, not
   // lazily: OutputWriter holds Pressure() by reference and reads it at each
   // save. The internal p_ stays P: it is the solver's warm start.
   INCNS_PROFILE("time_integrator::static_pressure");
   kinetic_head_->Interpolate(u_, ke_);
   p_static_ = p_;
   p_static_ -= ke_;
   if (bc_.PressureNullspaceExists()) { SubtractMean(p_static_, rules_); }
}

void StokesTimeIntegrator::Commit(double t_new, const Vector& u_true,
                                  ParGridFunction& u_gf, ParGridFunction& p_gf)
{
   INCNS_PROFILE("time_integrator::commit");
   if (&u_gf != &u_) { u_ = u_gf; }
   if (&p_gf != &p_) { p_ = p_gf; }

   // BDF2 needs 2 levels; the BDF3 estimator / test mode and EXT3 need 3
   // (EXT only matters with convection: Stokes keeps 2 under ext_order 3).
   const std::size_t needed =
      (opts_.adaptive || opts_.order == 3 ||
       (opts_.convection && opts_.ext_order == 3)) ? 3 : 2;
   // The level that drops out lends its memory to the new one (no
   // allocation and free per step). mfem::Vector has no move operations:
   // the memory changes hands by Swap(), and the deque holds an empty Vector
   // (no allocation) until it does.
   Vector slot;
   if (hist_.size() >= needed)
   {
      slot.Swap(hist_.back());
      hist_.pop_back();
      hist_times_.pop_back();
   }
   Scratch(slot, u_true.Size());
   slot = u_true;
   hist_.emplace_front();
   hist_.front().Swap(slot);
   hist_times_.push_front(t_new);
   while (hist_.size() > needed)
   {
      hist_.pop_back();
      hist_times_.pop_back();
   }

   t_ = t_new;
   ++step_count_;
   if (hold_left_ > 0) { --hold_left_; }
   if (rotational_) { UpdateStaticPressure(); }
}

int StokesTimeIntegrator::KrylovDimForMemory(double free_bytes,
      double total_bytes,
      long long system_size,
      long long velocity_size,
      int kdim, int extra_vectors)
{
   const double word = sizeof(double);
   const double margin = 0.03 * total_bytes + 6.0 * word * velocity_size;
   const double vector = word * static_cast<double>(system_size);
   // k + 1 basis vectors and k preconditioned ones, FGMRES's residual, and
   // the solve's right-hand side and solution: 2k + 4 vectors.
   const double fit =
      std::floor(((free_bytes - margin) / std::max(vector, 1.0) - 4.0 -
                  extra_vectors) / 2.0);
   return static_cast<int>(std::max(1.0, std::min<double>(kdim, fit)));
}

void StokesTimeIntegrator::CapKrylovDim(StokesSolver& solver)
{
   if (!opts_.krylov_memory_cap) { return; }
   // The Schur outer method keeps no coupled Krylov basis to cap.
   if (solver.UsesSchurOuter()) { return; }
   double free = 0.0, total = 0.0;
   int on_device = DeviceMemoryInfo(free, total) ? 1 : 0;
   MPI_Comm comm = spaces_.Velocity().GetComm();
   MPI_Allreduce(MPI_IN_PLACE, &on_device, 1, MPI_INT, MPI_MIN, comm);
   if (!on_device) { return; } // host memory: no cap
   // A solver's first application allocates its preconditioners' solve
   // workspace (9.3 GB at 64^3 Q3, the pressure AMG's most of it): make it
   // before reading the free memory, not during the solve.
   if (!solver.HasSolved())
   {
      solver.WarmUp();
      DeviceMemoryInfo(free, total);
   }
   // FGMRES's work vectors from the last solve are kept for the next: free
   // for this count (it assumes every Krylov vector is still to allocate).
   free += solver.KeptKrylovBytes();
   int k = KrylovDimForMemory(free, total, solver.SystemSize(),
                              spaces_.Velocity().GetTrueVSize(), opts_.kdim,
                              solver.GuessVectors());
   MPI_Allreduce(MPI_IN_PLACE, &k, 1, MPI_INT, MPI_MIN, comm);
   if (k != solver.KrylovDim()) { solver.SetKrylovDim(k); }
   // Report a cap when it appears and when it moves by more than 10 %
   // (free memory varies a little from step to step).
   const bool report =
      (k < opts_.kdim && logged_kdim_ < 0) ||
      (logged_kdim_ > 0
       && std::abs(k - logged_kdim_) > std::max(2, logged_kdim_ / 10));
   if (report)
   {
      if (Mpi::Root())
      {
         mfem::out << "[incns] FGMRES restart length " << k << " (solver.kdim "
                   << opts_.kdim << "): what fits in "
                   << free / 1073741824.0 << " GiB free of "
                   << total / 1073741824.0 << " GiB"
                   << (k < 5 ? " -- the GPU is nearly full" : "") << std::endl;
      }
      logged_kdim_ = k;
   }
}

bool StokesTimeIntegrator::AcceptSolve(StokesSolver& solver, double t_new)
{
   bool ok = solver.Converged();
   vetoed_ = false;
   if (ok && solve_veto_ && solve_veto_(step_count_, attempt_))
   {
      ok = false;
      vetoed_ = true;
   }
   if (ok) { return true; }
   MFEM_VERIFY(opts_.retry_on_failure && !opts_.adaptive,
               "time_integrator: implicit solve did not converge at t = "
               << t_new << " (" << solver.Iterations() << " iterations; "
               "time.on_solver_failure: cut_step retries with a smaller step)");
   // Back to the last accepted state: nothing was committed, and the next
   // attempt must not start from a half-converged iterate.
   u_.SetFromTrueDofs(hist_[0]);
   static_cast<Vector&>(p_) = p_saved_;
   return false;
}

bool StokesTimeIntegrator::StepStartup()
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
      // OIFS: u^0 advected to t^1 takes u^0's place in the explicit terms.
      Vector u0_adv;
      if (oifs_)
      {
         oifs_->Advect({0.0, 1.0}, hist_, hist_times_, t_new, 1, u0_adv);
         // Imposed boundary nodes: the one-step characteristic expansion
         // u^0 - dt (u^0.grad)u^0 (see OifsBoundaryValues).
         if (oifs_walls_)
         {
            Vector q(n_u);
            oifs_conv_->Mult(hist_[0], q);
            oifs_->MassInverse(q, q);
            Vector bd(hist_[0]), vals;
            bd.Add(-dt_, q);
            bd.GetSubVector(oifs_->ImposedDofs(), vals);
            u0_adv.SetSubVector(oifs_->ImposedDofs(), vals);
         }
      }
      const Vector& u0 = oifs_ ? u0_adv : hist_[0];
      Trap().Blocks().Mass().Mult(u0, tmp);
      b.Add(1.0 / dt_, tmp);
      trap_->Blocks().ViscousUnconstrained().Mult(u0, tmp); // (nu/2) K u^0
      b -= tmp;
      // Needed for this one product: free it before the solve allocates its
      // workspace (its PA data is as large as the momentum block's diffusion
      // data, and it set the run's peak GPU memory, 2026-10-13).
      trap_->Blocks().ReleaseViscousUnconstrained();
      // NSE: the starter builds its RHS inline (it does NOT go through
      // AssembleBdfRhs), so the convection term has to be subtracted here too
      // -- otherwise step 1 would silently solve Stokes and cost the march its
      // temporal order. Only one history entry exists, so this is EXT1.
      SubtractConvection(t_new, b);
      SubtractOifsDirectionalDoNothing(t_new, b); // OIFS: EXT1 here
      // The starter's momentum block is (1/dt) M + (nu/2) K; refresh it if the
      // first step is (re)tried at a different dt (adaptive rejection).
      if (std::abs(trap_c0_ - 1.0 / dt_) > 1e-12 / dt_)
      {
         trap_->Refresh(1.0 / dt_);
         trap_c0_ = 1.0 / dt_;
      }
      solver = trap_.get();
      if (rotational_)
      {
         // w* = u^0 (EXT1). The solver's block carries (1/2) N u^1 (alpha =
         // 1/2); the explicit other half, (1/2) N u^0, mirrors the viscous
         // split above. After UpdateRotation, so it uses this step's w*.
         UpdateLaggedVelocity(t_new);
         trap_->UpdateRotation();
         trap_->Blocks().RotationUnconstrained().Mult(hist_[0], tmp);
         b -= tmp;
      }
   }
   else
   {
      // Second step: BDF2 from {u^1, u^0} (weights from the actual times).
      std::vector<double> times = {t_new, hist_times_[0], hist_times_[1]};
      const std::vector<double> c = BdfWeights(times);
      solver = &EnsureBdfSolver(bdf2_, c[0]);
      AssembleBdfRhs(c, t_new, b);
      if (rotational_)
      {
         UpdateLaggedVelocity(t_new);
         solver->UpdateRotation();
      }
   }

   if (opts_.keep_rhs) { last_b_ = b; } // raw RHS, for MomentumResidual
   if (opts_.retry_on_failure) { p_saved_ = p_; }
   CapKrylovDim(*solver);
   solver->SolveTrue(b, u_, p_);
   last_iterations_ = solver->Iterations();
   last_solver_ = solver;
   if (!AcceptSolve(*solver, t_new)) { return false; }


   Vector& u_true = u_true_;
   Scratch(u_true, n_u);
   u_.GetTrueDofs(u_true);
   Commit(t_new, u_true, u_, p_);
   return true;
}

bool StokesTimeIntegrator::StepFixed()
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

   Vector& b = rhs_b_;
   Scratch(b, n_u);
   AssembleBdfRhs(c, t_new, b);
   if (rotational_)
   {
      UpdateLaggedVelocity(t_new);
      solver.UpdateRotation();
   }
   if (opts_.keep_rhs) { last_b_ = b; } // raw RHS, for MomentumResidual
   if (opts_.retry_on_failure) { p_saved_ = p_; }
   CapKrylovDim(solver);
   solver.SolveTrue(b, u_, p_);
   last_iterations_ = solver.Iterations();
   last_solver_ = &solver;
   if (!AcceptSolve(solver, t_new)) { return false; }

   Vector& u_true = u_true_;
   Scratch(u_true, n_u);
   u_.GetTrueDofs(u_true);
   Commit(t_new, u_true, u_, p_);
   return true;
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

      // An EMBEDDED pair: the advancing BDF2 candidate uses the configured
      // extrapolation (ext_order), the auxiliary order-3 candidate BDF3/EXT3
      // -- a genuine third-order solution of the SAME step. With EXT2 in both
      // (before 2026-10-07) the convective splitting error was identical in
      // the two candidates and cancelled in their difference: the estimator
      // was blind to it, the dominant error of CFL-limited NSE steps.
      // Rotational form: each candidate gets its own w* (its solver caches the
      // vorticity at UpdateRotation, so re-setting w* afterwards is safe).

      // BDF2 candidate -- the only one that may advance the solution.
      if (rotational_) { UpdateLaggedVelocity(t_new); }
      std::vector<double> t2 = {t_new, hist_times_[0], hist_times_[1]};
      const std::vector<double> c2 = BdfWeights(t2);
      Vector b2(n_u);
      AssembleBdfRhs(c2, t_new, b2);
      u2_scratch_ = u_;
      p2_scratch_ = p_;
      StokesSolver& s2 = EnsureBdfSolver(bdf2_, c2[0]);
      s2.UpdateRotation(); // no-op without the rotational form
      CapKrylovDim(s2);
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
      AssembleBdfRhs(c3, t_new, b3, 3);
      if (rotational_) { UpdateLaggedVelocity(t_new, 3); }
      u3_scratch_ = u_;
      p3_scratch_ = p_;
      StokesSolver& s3 = EnsureBdfSolver(bdf3_, c3[0]);
      s3.UpdateRotation();
      CapKrylovDim(s3);
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
         last_solver_ = &s2; // the BDF2 step is the one that advances
         if (opts_.keep_rhs) { last_b_ = b2; }
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
   const bool startup = (step_count_ < 2);
   if (!startup && opts_.adaptive)
   {
      StepAdaptive();
      return;
   }
   if (!startup && opts_.cfl_target > 0.0) { ChooseCflStep(); }
   // Fixed steps: after a cut and its hold, grow back to the step before it.
   if (!startup && dt_before_cut_ > 0.0 && hold_left_ == 0)
   {
      dt_ = std::min(dt_before_cut_, dt_ * opts_.cfl_growth);
      if (dt_ >= dt_before_cut_) { dt_before_cut_ = 0.0; }
   }
   for (attempt_ = 0;; ++attempt_)
   {
      // Neither returns false unless retry_on_failure (AcceptSolve aborts).
      if (startup ? StepStartup() : StepFixed()) { break; }
      ++rejected_steps_;
      const double cut = dt_ * opts_.retry_factor;
      MFEM_VERIFY(attempt_ < opts_.retry_max && cut > 0.0 &&
                  cut >= opts_.retry_dt_min, "time_integrator: the implicit "
                  "solve at t = " << t_ << " still does not converge after "
                  << attempt_ << " step cuts (dt = " << dt_ << "; "
                  "time.retry_max, time.retry_dt_min)");
      if (Mpi::Root())
      {
         mfem::out << "[incns] step " << step_count_ + 1 << " from t = " << t_
                   << (vetoed_ ? ": the implicit solve was rejected (test hook) "
                       "after " : ": the implicit solve did not converge in ")
                   << last_iterations_ << " iterations at dt = " << dt_
                   << "; retrying at dt = " << cut << std::endl;
      }
      if (opts_.cfl_target <= 0.0 && dt_before_cut_ == 0.0)
      {
         dt_before_cut_ = dt_;
      }
      dt_ = cut;
      // This step, once accepted, and retry_hold more: no growth.
      hold_left_ = opts_.retry_hold + 1;
   }
}

void StokesTimeIntegrator::Run()
{
   INCNS_PROFILE("time_integrator::run");
   while (!Done()) { Step(); }
}

} // namespace incns
