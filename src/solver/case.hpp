/**
 * @file case.hpp
 * @brief The public case surface: mesh + Parameters in, unsteady Stokes march
 *        with optional ParaView output out.
 */
#pragma once

#include "ale/mesh_convection.hpp"
#include "ale/mesh_motion.hpp"
#include "amr/refinement_marker.hpp"
#include "bc/boundary_conditions.hpp"
#include "bc/deck_boundary.hpp"
#include "config/parameters.hpp"
#include "post/run_monitor.hpp"
#include "post/checkpoint.hpp"
#include "post/diagnostics.hpp"
#include "post/body_force.hpp"
#include "post/channel_statistics.hpp"
#include "post/output.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/cfl.hpp"
#include "time/time_integrator.hpp"
#include "util/json.hpp"
#include "mfem.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace incns
{

/// What one AMR event (or initial pass) did; identical on every rank.
struct AdaptStats
{
   double time = 0.0;          ///< Simulation time of the event.
   int cycle = 0;              ///< Accepted-step count at the event.
   int passes = 0;             ///< Refinement passes that marked something.
   MarkStats first_pass;       ///< Marking statistics of the first pass.
   long long ne_before = 0;    ///< Global elements before.
   long long ne_after = 0;     ///< Global elements after.
   int projection_iterations = 0; ///< History-projection outer iterations.
   long long merged = 0;       ///< Families merged (amr.derefine).
   double seconds = 0.0;       ///< Wall time of the whole event.
   bool rebuilt = false;       ///< The integrator was rebuilt.
};

/**
 * @brief One unsteady Stokes case, driven by a Parameters block.
 *
 * This is the surface every driver touches -- a YAML deck (via
 * Parameters::LoadYAML + apps/run_case) and an in-code driver
 * (apps/taylor_green) go through this SAME class; that is the point of the
 * library split, and the solver core is never edited to run a new case.
 *
 * Usage: construct over a partitioned mesh; optionally populate boundary
 * conditions (an empty BC set on a periodic mesh means fully periodic) and set
 * the initial velocity/forcing; then Run() marches to t_final and writes
 * ParaView output per the deck's output settings, or drive manually with
 * Step()/Time()/Done(). BCs and the initial condition must be final before the
 * first step (the operators eliminate the essential dofs at setup).
 *
 * In Sprint 2 NavierStokesSolver composes the same machinery and adds the
 * convection term; this surface is what it will slot into.
 */
class Case
{
public:
   /**
    * @brief Build spaces and defaults over a mesh.
    * @param mesh   Partitioned quad/hex mesh (borrowed, must outlive this).
    * @param params Case parameters (copied).
    */
   Case(mfem::ParMesh& mesh, const Parameters& params);

   /**
    * @brief Tear down, then drop the mesh's cached geometric factors.
    *
    * MFEM caches geometric factors on the mesh keyed by IntegrationRule
    * pointer, and requires each rule to outlive its cache entry. The rules
    * belong to this Case's RuleBook but the mesh is borrowed and may outlive
    * the Case; a later rule allocated at a freed address would otherwise hit
    * a stale entry (wrong factors, NaNs).
    */
   ~Case();

   /// @return The mixed spaces (e.g. to build a BoundaryConditions over).
   MixedSpaces& Spaces() { return spaces_; }

   /**
    * @brief Use an externally built BC set (borrowed) instead of the default
    *        empty one. Must be called before the first Step()/Run().
    * @param bc Boundary conditions over this case's velocity space.
    */
   void SetBoundaryConditions(BoundaryConditions& bc);

   /**
    * @brief Move the mesh with @p motion at every step: the ALE fluid
    *        (docs/ale_fluid.md). Before the first Step()/Run(). Under
    *        convection Auto the case then runs IMEX (OIFS on a moving mesh
    *        is not built); OIFS, the rotational form, error control and AMR
    *        are refused. A moving wall's no-slip condition is u = w: give its
    *        attribute MeshVelocityCoefficient() as Dirichlet data.
    *        A deck whose boundary groups have `motion` builds its own
    *        (the linear elastic model of section mesh_motion) in the
    *        constructor; this is for a driver's own motion and model.
    * @param motion The motion of this case's mesh (borrowed; built on it
    *        before this case caches any geometry, at the case's start time).
    */
   void SetMeshMotion(MeshMotion& motion);

   /// @return Whether the mesh moves (a deck's moving boundaries, or
   ///         SetMeshMotion()).
   bool MovingMesh() const { return motion_ != nullptr; }

   /// @return The mesh motion. @pre MovingMesh().
   MeshMotion& Motion();

   /// @return The mesh velocity w as a coefficient (u = w on a moving
   ///         wall). @pre MovingMesh().
   mfem::VectorCoefficient& MeshVelocityCoefficient();

   /// @return The mesh velocity w on the velocity space (zero before the
   ///         first step). @pre MovingMesh().
   const mfem::ParGridFunction& MeshVelocity() const;

   /// @return The BDF weight of the newest node positions in MeshVelocity()
   ///         (dw / dx^{n+1}; 0 before the first step). @pre MovingMesh().
   double MeshVelocityLead() const;

   /**
    * @brief Set the initial velocity (t = 0). Pressure has no independent
    *        initial condition. Must be called before the first Step()/Run().
    * @param u0 Initial velocity coefficient (borrowed).
    */
   void SetInitialVelocity(mfem::VectorCoefficient& u0);

   /**
    * @brief Set the momentum forcing f(x, t) (borrowed; zero by default).
    * @param f Forcing coefficient; SetTime is driven by the stepper.
    */
   void SetForcing(mfem::VectorCoefficient& f);

   /// March to t_final, writing output per the parameters' output settings;
   /// stops early if the run diverges (Diverged()).
   void Run();

   /// @return Whether the run diverged (non-finite or huge forces / velocity).
   bool Diverged() const;

   /**
    * @brief Write the run summary as JSON: status, case settings, time-step
    *        statistics, mesh size, diagnostics, force statistics, probes, and
    *        the errors against the deck's reference values. Collective; rank
    *        0 writes.
    * @param path   Output file (its directory is created).
    * @param status "running" before the run, then "ok" or "diverged".
    */
   void WriteSummary(const std::string& path, const std::string& status);

   /**
    * @brief Write the profiler's region tree as CSV (Profiler::WriteCsv):
    *        where the wall time of this process's runs went. Collective; rank
    *        0 writes.
    * @param path Output file (its directory is created).
    */
   void WriteProfile(const std::string& path) const;

   /// Adds sections and reference values to WriteSummary(): called on every
   /// rank (it may be collective) with the summary and the (key, value)
   /// pairs compared with the deck's reference values.
   using SummaryHook = std::function<void(Json&,
                                          std::vector<std::pair<std::string, double>>&)>;

   /// @param hook The summary hook (SummaryHook; FsiCase's coupling section).
   void SetSummaryHook(SummaryHook hook) { summary_hook_ = std::move(hook); }

   /// A driver's own part of a checkpoint (FsiCase: the solid and the
   /// coupling), written into its directory after the flow's state.
   using CheckpointHook = std::function<void(const std::string& dir)>;
   /// @param hook Called by WriteCheckpoint() with the directory, on every
   ///             rank (collective).
   void SetCheckpointHook(CheckpointHook hook)
   {
      checkpoint_hook_ = std::move(hook);
   }

   /// After a march driven from outside Run() (Step()s, trial steps):
   /// flush the history and write what Run() writes at its end.
   void FinishRun();

   /// Advance one accepted step (writes output on the configured interval).
   void Step();

   /**
    * @brief Make the next step as a trial (a coupled FSI step's
    *        subiteration): the integrator steps, and nothing is recorded or
    *        written until CommitTrialStep(). UndoTrialStep() takes it back.
    */
   /// @return False when its solve failed under SolverFailure::Report: the
   ///         flow is back at the step's start, with nothing to undo.
   bool TrialStep();

   /// Take back the trial step: the mesh and the march return to before it.
   void UndoTrialStep();

   /// Accept the trial step: record it, log, write output and checkpoints as
   /// Step() does.
   void CommitTrialStep();

   /**
    * @brief The displacement of the boundary groups of type `interface`, on
    *        the mesh motion's space (the reference mesh): the coupling sets
    *        it before each trial step. Null without such groups.
    * @return The field, or null.
    */
   mfem::ParGridFunction* InterfaceDisplacement()
   {
      return interface_disp_.get();
   }

   /**
    * @brief The Robin coefficient of the deck's interface groups under
    *        fsi.coupling robin_neumann, in place of the constant
    *        Parameters::FsiFluidRobin(). Before the first step.
    * @param alpha     The coefficient, kinematic (borrowed).
    * @param alpha_max An upper bound (the LOR preconditioner's constant).
    */
   void SetRobinCoefficient(mfem::Coefficient& alpha, double alpha_max)
   {
      robin_alpha_ = &alpha;
      robin_alpha_max_ = alpha_max;
   }

   /// @return Current time (nondimensional, t*).
   double Time() const;

   /// @return Current time in dimensional units, t* x T_ref (equal to Time()
   ///         in Dimensionless mode where T_ref = 1).
   double TimeDimensional() const { return Time() * params_.nondim.TRef(); }

   /// @return True once t_final is reached.
   bool Done() const;

   /// @return The velocity field.
   mfem::ParGridFunction& Velocity();

   /// @return The channel statistics (deck channel_statistics), or null.
   const ChannelStatistics* ChannelStats() const { return channel_stats_.get(); }

   /**
    * @brief Write the channel profiles CSV (ChannelStatistics::WriteCsv) to
    *        @p path, its directory created. No-op without channel statistics.
    *        Collective; rank 0 writes.
    * @param path Output file.
    */
   void WriteChannelProfiles(const std::string& path) const;

   /// @return <output.path>/<output.name>_profiles.csv, where Run() and every
   ///         checkpoint write the channel profiles.
   std::string ChannelProfilesPath() const;

   /// @return The pressure field.
   mfem::ParGridFunction& Pressure();

   /// @return The underlying integrator (iterations, adaptive controller).
   StokesTimeIntegrator& Integrator();

   /// @return Kinetic energy 0.5*int|u|^2 of the current velocity (global).
   double KineticEnergy();

   /// @return Viscous dissipation nu*int|grad u|^2 of the current velocity.
   double DissipationRate();

   /// @return Divergence norm ||div u||_L2 of the current velocity (global).
   double DivergenceNorm();

   /**
    * @brief Run one adaptation event now (Step() runs them every
    *        amr.interval accepted steps): indicator, marking, refinement,
    *        exact state transfer, optional divergence projection of the
    *        history, and a rebuild of the integrator and output. Collective.
    * @param force_rebuild Rebuild and re-import even when nothing is marked
    *        (test hook: exercises the export/rebuild/import path alone).
    * @return What the event did (also appended to AdaptHistory()).
    * @pre amr.enabled, and the startup ramp is over (2 accepted steps).
    */
   AdaptStats Adapt(bool force_rebuild = false);

   /// @return Every AMR event and initial pass so far, in order.
   const std::vector<AdaptStats>& AdaptHistory() const { return adapt_log_; }

   /**
    * @brief The directional convective CFL number of the current convective
    *        velocity (u; on a moving mesh u - w, as the CFL step control's)
    *        at the current step size, c * dt (see time/cfl.hpp). Collective.
    * @param where Optional output: the centre of the element where the rate
    *              peaks (diagnostics), on every rank.
    * @return The CFL number (0 for a zero velocity).
    */
   double ConvectiveCflNumber(mfem::Vector* where = nullptr);

   /**
    * @brief Write a rolling checkpoint of the current state, including the
    *        AMR refinement history (Step() calls this on the configured
    *        interval). Collective.
    * @param dir Checkpoint directory.
    */
   void WriteCheckpoint(const std::string& dir);

   /**
    * @brief Select the body for lift/drag after construction (enables
    *        forces) -- for drivers that resolve the attributes from the mesh,
    *        e.g. a box face by name. Overrides forces.attributes.
    * @param attributes Boundary attributes forming the body (non-empty).
    */
   void SetForceBody(const std::vector<int>& attributes);

   /**
    * @brief Force of the fluid on the body (forces.attributes) at the last
    *        step, by John's volume-integral formulation (post/body_force).
    *        Collective.
    * @return F, size dim (drag = x, lift = y).
    * @pre forces.enabled, and a step taken since setup / the last AMR event.
    */
   mfem::Vector BodyForceVector();

   /**
    * @brief Force coefficients C = 2 F / (U^2 A) with the forces.* reference
    *        scales. Collective.
    * @return C, size dim.
    */
   mfem::Vector ForceCoefficients();

private:
   /// Build the integrator/output on first use (BCs must be final by then).
   void EnsureSetup();

   /**
    * @brief Time-based AMR (amr.every_time): run the event if the step just
    *        taken crossed one. Collective.
    * @param t_prev Time before the step.
    */
   void MaybeAdaptInTime(double t_prev);

   /**
    * @brief Record the step that ended at the current time (monitor,
    *        progress, history). Collective.
    * @param t_prev    Time before the step.
    * @param step_wall Its wall time.
    */
   void RecordStep(double t_prev, double step_wall);

   /// (Re)build the time integrator from the parameters on the current mesh.
   void BuildIntegrator();

   /**
    * @brief (Re)build the ParaView writer (no-op when output is disabled).
    * @param restart Continue the existing collection (after an AMR event).
    */
   void BuildOutput(bool restart);

   /// @return The diagnostics for the current mesh (assembled on first use
   ///         and again after an adaptation event).
   FlowDiagnostics& Diagnostics();

   /// amr.initial_passes: refine on the analytic initial condition.
   void InitialRefinement();

   /// @return Is an adaptation event due after the step just taken?
   bool AdaptDue() const;

   /// amr.write_indicator: recompute the eta / level cell data.
   void UpdateAmrCellData();

   /// Navier-Stokes (both forms) with CFL steps or time.cfl_max: (re)build
   /// the CFL estimator on the current mesh. CFL mode installs the rate the
   /// integrator steers by (and caps the first step); error mode installs the
   /// cfl_max dt ceiling; fixed-step mode verifies the current dt.
   void SetupCfl();

   /// @return The convective velocity of the current state: u, on a moving
   ///         mesh u - w (in relative_u_).
   const mfem::ParGridFunction& ConvectiveVelocity();

   /**
    * @brief Append a diagnostics row on the output interval (no-op if
    *        disabled).
    * @param cycle Accepted-step count (the output cycle).
    * @param time  Current simulation time.
    */
   void MaybeLogDiagnostics(int cycle, double time);

   /**
    * @brief Suggest memory.max_dofs in the log when it applies (OIFS at
    *        velocity orders 6 and 7 on straight-sided elements, not on) and
    *        a GPU is 85 % full. Collective.
    */
   void MaybeSuggestMaxDofs();

   mfem::ParMesh& mesh_;      ///< Mesh (borrowed).
   Parameters params_;        ///< Case parameters (copied).
   RuleBook rules_;           ///< Quadrature owner (outlives all operators).
   MixedSpaces spaces_;       ///< Velocity/pressure spaces.
   BoundaryConditions own_bc_; ///< Default (empty) BC set.
   BoundaryConditions* bc_;    ///< Active BC set (own_bc_ or user-supplied).
   mfem::Vector zero_vec_;     ///< Zero vector for the default forcing.
   /// RecordStep's copy of the velocity's true dofs (for its norm): kept, a
   /// new one cost a device allocation and a synchronizing free per step.
   mfem::Vector record_u_;
   mfem::VectorConstantCoefficient zero_forcing_; ///< Default forcing.
   mfem::VectorCoefficient* forcing_;  ///< Active forcing.
   mfem::VectorCoefficient* initial_;  ///< Initial velocity (null = zero).

   std::unique_ptr<StokesTimeIntegrator> integrator_; ///< Built lazily.
   /// ALE: build own_motion_ from the deck's moving boundary groups (the
   /// constructor, before anything caches geometry).
   void BuildDeckMotion();
   /// The deck's moving groups are all no_slip walls (u = w): a closed
   /// domain must then keep its volume (RequireConstantVolume).
   bool walls_follow_mesh_ = true;

   /// ALE: the mesh displacements of the deck's moving boundary groups
   /// (outlive own_motion_, whose model reads them).
   std::vector<std::unique_ptr<mfem::VectorFunctionCoefficient>>
   deck_motion_coeffs_;
   /// ALE: the mesh motion a deck's moving boundaries build (null: none, or
   /// a driver's own through SetMeshMotion).
   std::unique_ptr<MeshMotion> own_motion_;
   /// ALE: the mesh motion (own_motion_ or a driver's; null: fixed mesh).
   MeshMotion* motion_ = nullptr;
   /// ALE: the mesh velocity on the velocity space.
   std::unique_ptr<MeshVelocityField> mesh_w_;
   /// ALE: u - w for the CFL rate.
   std::unique_ptr<mfem::ParGridFunction> relative_u_;
   std::unique_ptr<OutputWriter> output_;             ///< Built when enabled.
   std::unique_ptr<DiagnosticsLog> diag_log_;         ///< Built when enabled.
   /// The diagnostics' assembled forms; rebuilt when the mesh changes.
   std::unique_ptr<FlowDiagnostics> flow_diag_;
   int cycle_ = 0; ///< Accepted-step counter for output.
   /// The march before a trial step (TrialStep()).
   IntegratorState trial_before_;
   double trial_t_prev_ = 0.0; ///< Time before the trial step.
   double trial_wall_ = 0.0;   ///< Wall time of the trial step.
   bool in_trial_ = false;     ///< Whether a trial step awaits its verdict.
   /// The interface groups' displacement (InterfaceDisplacement()).
   std::unique_ptr<mfem::ParGridFunction> interface_disp_;
   /// Its coefficient, the interface groups' mesh-motion data.
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> interface_coeff_;
   /// Record, log and write after a step (Step(), CommitTrialStep()).
   /// @param t_prev Time before the step. @param step_wall Its wall time.
   void AfterStep(double t_prev, double step_wall);
   bool warned_scaling_ = false; ///< One-shot bad-U_ref drift warning issued.
   SummaryHook summary_hook_; ///< See SetSummaryHook().
   CheckpointHook checkpoint_hook_; ///< See SetCheckpointHook().
   mfem::Coefficient* robin_alpha_ = nullptr; ///< See SetRobinCoefficient().
   double robin_alpha_max_ = 0.0; ///< See SetRobinCoefficient().
   bool memory_hint_done_ = false; ///< memory.max_dofs considered (once).
   std::vector<AdaptStats> adapt_log_; ///< AMR events and initial passes.
   /// Convective CFL estimator on the current mesh (time.cfl_max, NSE only).
   std::unique_ptr<ConvectiveCfl> cfl_;
   /// Every refinement batch applied to the mesh, in order (initial passes,
   /// events, a restart's replay) -- what a checkpoint stores to rebuild it.
   std::vector<RefinementRecord> refine_log_;
   /// Lift/drag evaluator on the current velocity space (forces.enabled).
   std::unique_ptr<BodyForce> body_force_;
   /// The force of the step taken just before an AMR event (the rebuilt
   /// integrator cannot form that step's residual); empty otherwise.
   mfem::Vector force_cache_;
   /// Write a forces CSV row on the forces interval (rank 0).
   void MaybeLogForces();

   /**
    * @brief Append the step's rotation diagnostics to
    *        `<output.path>/<output.name>_rotation.csv` every
    *        solver.rotation_log_interval steps (rotational form only; rank 0
    *        writes, the statistics are already global).
    * @param dt        Size of the step just taken.
    * @param step_wall Wall time of the step [s].
    */
   void MaybeLogRotation(double dt, double step_wall);
   bool rotation_log_started_ = false; ///< Header written (file truncated).

   /**
    * @brief Append the step's solver statistics to
    *        `<output.path>/<output.name>_solver.csv` every
    *        solver.log_interval steps, either outer method, and with
    *        solver.uzawa.history the Schur method's iterations to
    *        `<output.name>_uzawa_history.csv` (rank 0 writes; the device
    *        memory is the largest over the ranks).
    * @param dt        Size of the step just taken.
    * @param step_wall Wall time of the step [s].
    */
   void MaybeLogSolver(double dt, double step_wall);
   bool solver_log_started_ = false; ///< Header written (files truncated).
   /// amr.write_indicator: piecewise-constant space and fields for the
   /// indicator and the refinement depth (rebuilt after each event).
   std::unique_ptr<mfem::L2_FECollection> amr_fec_;
   std::unique_ptr<mfem::ParFiniteElementSpace> amr_fes_; ///< P0 space.
   std::unique_ptr<mfem::ParGridFunction> amr_eta_;   ///< eta_K at the event.
   std::unique_ptr<mfem::ParGridFunction> amr_level_; ///< Refinement depth.
   bool bc_explicit_ = false; ///< SetBoundaryConditions was called.
   /// The deck's boundary groups (when the deck declares them and no driver
   /// set boundary conditions).
   std::unique_ptr<DeckBoundaryConditions> deck_bc_;
   std::unique_ptr<RunMonitor> monitor_; ///< Progress, history, statistics.
   long long ne_ = 0;               ///< Global element count (kept current).
   bool amr_time_started_ = false;  ///< A time-based AMR event has run.
   bool chk_at_written_ = false;    ///< checkpoint.at_time written.
   int amr_events_ = 0;             ///< AMR events that changed the mesh.
   /// The deck's constant body force (forcing.body_force), when given.
   std::unique_ptr<mfem::VectorConstantCoefficient> deck_forcing_;
   /// Plane and time averages (deck channel_statistics), when enabled.
   std::unique_ptr<ChannelStatistics> channel_stats_;
};

} // namespace incns
