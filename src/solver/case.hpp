/**
 * @file case.hpp
 * @brief The public case surface: mesh + Parameters in, unsteady Stokes march
 *        with optional ParaView output out.
 */
#pragma once

#include "amr/refinement_marker.hpp"
#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "post/checkpoint.hpp"
#include "post/diagnostics.hpp"
#include "post/body_force.hpp"
#include "post/output.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/cfl.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <memory>
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
   int projection_iterations = 0; ///< History-projection FGMRES iterations.
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

   /// March to t_final, writing output per the parameters' output settings.
   void Run();

   /// Advance one accepted step (writes output on the configured interval).
   void Step();

   /// @return Current time (nondimensional, t*).
   double Time() const;

   /// @return Current time in dimensional units, t* x T_ref (equal to Time()
   ///         in Dimensionless mode where T_ref = 1).
   double TimeDimensional() const { return Time() * params_.nondim.TRef(); }

   /// @return True once t_final is reached.
   bool Done() const;

   /// @return The velocity field.
   mfem::ParGridFunction& Velocity();

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
    * @brief The directional convective CFL number of the current velocity at
    *        the current step size, c * dt (see time/cfl.hpp). Collective.
    * @return The CFL number (0 for a zero velocity).
    */
   double ConvectiveCflNumber();

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

   /// (Re)build the time integrator from the parameters on the current mesh.
   void BuildIntegrator();

   /**
    * @brief (Re)build the ParaView writer (no-op when output is disabled).
    * @param restart Continue the existing collection (after an AMR event).
    */
   void BuildOutput(bool restart);

   /// amr.initial_passes: refine on the analytic initial condition.
   void InitialRefinement();

   /// @return Is an adaptation event due after the step just taken?
   bool AdaptDue() const;

   /// amr.write_indicator: recompute the eta / level cell data.
   void UpdateAmrCellData();

   /// time.cfl_max with the convective NSE form: (re)build the CFL estimator
   /// on the current mesh; adaptive mode installs the dt ceiling, fixed-step
   /// mode verifies the current dt.
   void SetupCfl();

   /**
    * @brief Append a diagnostics row on the output interval (no-op if
    *        disabled).
    * @param cycle Accepted-step count (the output cycle).
    * @param time  Current simulation time.
    */
   void MaybeLogDiagnostics(int cycle, double time);

   mfem::ParMesh& mesh_;      ///< Mesh (borrowed).
   Parameters params_;        ///< Case parameters (copied).
   RuleBook rules_;           ///< Quadrature owner (outlives all operators).
   MixedSpaces spaces_;       ///< Velocity/pressure spaces.
   BoundaryConditions own_bc_; ///< Default (empty) BC set.
   BoundaryConditions* bc_;    ///< Active BC set (own_bc_ or user-supplied).
   mfem::Vector zero_vec_;     ///< Zero vector for the default forcing.
   mfem::VectorConstantCoefficient zero_forcing_; ///< Default forcing.
   mfem::VectorCoefficient* forcing_;  ///< Active forcing.
   mfem::VectorCoefficient* initial_;  ///< Initial velocity (null = zero).

   std::unique_ptr<StokesTimeIntegrator> integrator_; ///< Built lazily.
   std::unique_ptr<OutputWriter> output_;             ///< Built when enabled.
   std::unique_ptr<DiagnosticsLog> diag_log_;         ///< Built when enabled.
   int cycle_ = 0; ///< Accepted-step counter for output.
   bool warned_scaling_ = false; ///< One-shot bad-U_ref drift warning issued.
   std::vector<AdaptStats> adapt_log_; ///< AMR events and initial passes.
   /// Convective CFL estimator on the current mesh (time.cfl_max, NSE only).
   std::unique_ptr<ConvectiveCfl> cfl_;
   /// Every refinement batch applied to the mesh, in order (initial passes,
   /// events, a restart's replay) -- what a checkpoint stores to rebuild it.
   std::vector<RefinementRecord> refine_log_;
   /// Lift/drag evaluator on the current velocity space (forces.enabled).
   std::unique_ptr<BodyForce> body_force_;
   /// Write a forces CSV row on the forces interval (rank 0).
   void MaybeLogForces();
   /// amr.write_indicator: piecewise-constant space and fields for the
   /// indicator and the refinement depth (rebuilt after each event).
   std::unique_ptr<mfem::L2_FECollection> amr_fec_;
   std::unique_ptr<mfem::ParFiniteElementSpace> amr_fes_; ///< P0 space.
   std::unique_ptr<mfem::ParGridFunction> amr_eta_;   ///< eta_K at the event.
   std::unique_ptr<mfem::ParGridFunction> amr_level_; ///< Refinement depth.
};

} // namespace incns
