/**
 * @file diagnostics.hpp
 * @brief Global physical diagnostics of a velocity field (energy, dissipation,
 *        divergence) and an optional time-series CSV writer.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "post/flow_sums.hpp"
#include "mfem.hpp"

#include <fstream>
#include <string>

namespace incns
{

class ElementScratch;

/**
 * @brief Kinetic energy, @f$ \tfrac12 \int_\Omega |u|^2 @f$.
 *
 * A single MPI-reduced scalar, integrated with an elevated rule from the
 * RuleBook (the default rule under-integrates the quadratic |u|^2 for smooth
 * non-polynomial fields).
 *
 * @param u     Velocity field (parallel).
 * @param rules Owns the integration rules (must outlive this call).
 * @return The global kinetic energy.
 */
double KineticEnergy(const mfem::ParGridFunction& u, const RuleBook& rules);

/**
 * @brief Viscous dissipation rate, @f$ \nu \int_\Omega |\nabla u|^2 @f$.
 *
 * Computed from the vector-diffusion form @f$ (\nabla u, \nabla u) @f$ via
 * matrix-free partial assembly, reduced across ranks.
 *
 * @param u     Velocity field (parallel).
 * @param nu    Kinematic viscosity (nondimensional 1/Re in dimensionless mode).
 * @param rules Owns the integration rules (must outlive this call).
 * @return The global viscous dissipation rate.
 */
double DissipationRate(const mfem::ParGridFunction& u, double nu,
                       const RuleBook& rules);

/**
 * @brief Divergence norm, @f$ \|\nabla\cdot u\|_{L2}
 *        = \sqrt{\int_\Omega (\nabla\cdot u)^2} @f$.
 *
 * The same measure the divergence fast-tier check uses; a mass-conservation
 * diagnostic (should be ~0, and drop with grad-div on).
 *
 * @param u     Velocity field (parallel).
 * @param rules Owns the integration rules (must outlive this call).
 * @return The global divergence norm.
 */
double DivergenceNorm(const mfem::ParGridFunction& u, const RuleBook& rules);

/**
 * @brief The three diagnostics above for repeated use on one space: the same
 *        integrals with the same elevated rule, evaluated on the device.
 *
 * KineticEnergy() and DivergenceNorm() walk the elements on the host and
 * DissipationRate() assembles its form on every call: fine for a test, but
 * 0.7 s per call for 4,096 Q3 elements on a GPU and 5.4 s for 32,768
 * (2026-10-12), 9 % of a Taylor-Green run that logs every step. Here the
 * velocity and its gradient are interpolated to the quadrature points and
 * the three integrands summed in one device pass (post/flow_sums), with
 * nothing stored: three partially assembled forms for the same values held
 * about 10 GB for 64^3 Q3 elements (2026-10-13).
 *
 * The rule belongs to the space as it was at construction: after the mesh is
 * refined, build a new object (Stale() tells).
 */
class FlowDiagnostics
{
public:
   /// The three values, from one pass over the elements.
   struct Values
   {
      double kinetic_energy = 0.0;   ///< As incns::KineticEnergy.
      double dissipation_rate = 0.0; ///< As incns::DissipationRate.
      double divergence_norm = 0.0;  ///< As incns::DivergenceNorm.
   };

   /**
    * @brief Bind to a velocity space.
    * @param vfes  Velocity space (borrowed; vdim == dim, quads/hexes).
    * @param rules Owns the integration rules (must outlive this object).
    */
   FlowDiagnostics(mfem::ParFiniteElementSpace& vfes, const RuleBook& rules);

   /**
    * @brief Borrow the velocity's element vector from @p scratch for the
    *        next evaluations instead of allocating one each time (a whole
    *        velocity E-vector: at the end of a run, with the device nearly
    *        full, that allocation was the one to fail). The diagnostics run
    *        between steps, when no operator holds the scratch; lend again
    *        whenever the scratch may have been rebuilt.
    * @param scratch The buffers (borrowed; must outlive the evaluations).
    */
   void UseScratch(ElementScratch& scratch);

   /// @return Whether the space changed (was refined) since construction.
   bool Stale() const { return vfes_.GetSequence() != sequence_; }

   /**
    * @brief All three diagnostics in one pass.
    * @param u  Velocity on the space given at construction.
    * @param nu Kinematic viscosity (for the dissipation rate).
    * @return The values.
    */
   Values All(const mfem::ParGridFunction& u, double nu) const;

   /**
    * @brief Kinetic energy, as incns::KineticEnergy.
    * @param u Velocity on the space given at construction.
    * @return The global kinetic energy.
    */
   double KineticEnergy(const mfem::ParGridFunction& u) const;

   /**
    * @brief Viscous dissipation rate, as incns::DissipationRate.
    * @param u  Velocity on the space given at construction.
    * @param nu Kinematic viscosity.
    * @return The global viscous dissipation rate.
    */
   double DissipationRate(const mfem::ParGridFunction& u, double nu) const;

   /**
    * @brief Divergence norm, as incns::DivergenceNorm.
    * @param u Velocity on the space given at construction.
    * @return The global divergence norm.
    */
   double DivergenceNorm(const mfem::ParGridFunction& u) const;

private:
   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   long sequence_;                     ///< Its sequence number at construction.
   const mfem::IntegrationRule* ir_;   ///< The elevated rule (RuleBook's).
   mutable FlowSumsWork work_;         ///< Kept per-element work vectors.
};

/**
 * @brief Appends (t, KE, dissipation, ||div u||) rows to a CSV time series.
 *
 * Rank 0 owns the file; other ranks are no-ops. Opt-in via the output settings
 * (see OutputParameters::diagnostics); written on the same interval as the
 * ParaView snapshots.
 */
class DiagnosticsLog
{
public:
   /**
    * @brief Open the CSV and write the header (rank 0 only).
    * @param path   Full CSV path.
    * @param comm   Communicator (used to pick the writer rank).
    * @param append Continue an existing file (a restart) instead of starting
    *               a new one; rows the earlier segment wrote after the
    *               checkpoint's time are then superseded (readers drop them
    *               where time goes back).
    */
   DiagnosticsLog(const std::string& path, MPI_Comm comm, bool append = false);

   /**
    * @brief Append one row (rank 0 only); values are already MPI-reduced.
    * @param t          Physical time.
    * @param energy     Kinetic energy.
    * @param dissipation Viscous dissipation rate.
    * @param divergence  Divergence norm.
    */
   void Write(double t, double energy, double dissipation, double divergence);

private:
   int rank_ = 0;         ///< This rank (only rank 0 writes).
   std::ofstream os_;     ///< Open on rank 0 only.
};

} // namespace incns
