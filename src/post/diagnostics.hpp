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
    *               a new one; rows the interrupted run wrote after its
    *               checkpoint then repeat (readers keep the last per time).
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
