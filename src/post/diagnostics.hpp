/**
 * @file diagnostics.hpp
 * @brief Global physical diagnostics of a velocity field (energy, dissipation,
 *        divergence) and an optional time-series CSV writer.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <fstream>
#include <string>

namespace incns
{

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
 *        integrals with the same elevated rule, as quadratic forms of three
 *        partially assembled operators.
 *
 * KineticEnergy() and DivergenceNorm() walk the elements on the host and
 * DissipationRate() assembles its form on every call: fine for a test, but
 * 0.7 s per call for 4,096 Q3 elements on a GPU and 5.4 s for 32,768
 * (2026-10-12), 9 % of a Taylor-Green run that logs every step. Here the mass,
 * diffusion and grad-div forms are assembled once, and each value is one
 * operator apply and one dot product on the device:
 * @f$ \tfrac12 U^T M U @f$, @f$ \nu\,U^T K U @f$, @f$ \sqrt{U^T D U} @f$.
 *
 * The forms belong to the space as it was at construction: after the mesh is
 * refined, build a new object (Stale() tells).
 */
class FlowDiagnostics
{
public:
   /**
    * @brief Assemble the three forms.
    * @param vfes  Velocity space (borrowed; vdim == dim, quads/hexes).
    * @param rules Owns the integration rules (must outlive this object).
    */
   FlowDiagnostics(mfem::ParFiniteElementSpace& vfes, const RuleBook& rules);

   /// @return Whether the space changed (was refined) since construction.
   bool Stale() const { return vfes_.GetSequence() != sequence_; }

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
   /**
    * @brief @f$ U^T A U @f$ on the true dofs, reduced over the ranks.
    * @param A One of the three operators.
    * @param u Velocity field.
    * @return The quadratic form.
    */
   double Quadratic(const mfem::Operator& A, const mfem::ParGridFunction& u) const;

   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   long sequence_;                     ///< Its sequence number at construction.
   mfem::ParBilinearForm mass_;        ///< (u, v), elevated rule.
   mfem::ParBilinearForm diffusion_;   ///< (grad u, grad v), elevated rule.
   mfem::ParBilinearForm divdiv_;      ///< (div u, div v), elevated rule.
   mfem::OperatorPtr M_;               ///< mass_ on the true dofs.
   mfem::OperatorPtr K_;               ///< diffusion_ on the true dofs.
   mfem::OperatorPtr D_;               ///< divdiv_ on the true dofs.
   mutable mfem::Vector u_true_;       ///< True-dof scratch (device).
   mutable mfem::Vector work_;         ///< True-dof scratch (device).
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
