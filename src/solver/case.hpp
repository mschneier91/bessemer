/**
 * @file case.hpp
 * @brief The public case surface: mesh + Parameters in, unsteady Stokes march
 *        with optional ParaView output out.
 */
#ifndef INCNS_SOLVER_CASE_HPP
#define INCNS_SOLVER_CASE_HPP

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "post/checkpoint.hpp"
#include "post/diagnostics.hpp"
#include "post/output.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

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

private:
   /// Build the integrator/output on first use (BCs must be final by then).
   void EnsureSetup();

   /// Append a diagnostics row on the output interval (no-op if disabled).
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
};

} // namespace incns

#endif // INCNS_SOLVER_CASE_HPP
