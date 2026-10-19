/**
 * @file elasticity_case.hpp
 * @brief A deck-driven linear elasticity case (equation: linear_elasticity):
 *        the solver built from Parameters::solid, the deck's boundary groups
 *        applied, a static solve or a dynamic march, the run summary, the
 *        history CSV and ParaView output.
 */
#pragma once

#include "config/parameters.hpp"
#include "solid/linear_elasticity.hpp"
#include "mfem.hpp"

#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @param p Parameters (solid section, discretization.order_u, mesh.dim).
 * @return The solver options a deck gives: order, Lame parameters of the
 *         plane model in effect, CG settings, density and time scheme.
 */
LinearElasticity::Options ElasticityOptions(const Parameters& p);

/**
 * @brief The solid counterpart of Case: a static linear elasticity solve, or
 *        a dynamic march from rest, on a case mesh, with the deck's boundary
 *        groups (fixed, displacement, traction, free; every real boundary in
 *        exactly one group).
 */
class ElasticityCase
{
public:
   /**
    * @brief Build the solver and apply the deck's boundary groups. Collective.
    * @param mesh   The case mesh (borrowed; must outlive the case).
    * @param params Normalized parameters with equation linear_elasticity
    *               (borrowed).
    */
   ElasticityCase(mfem::ParMesh& mesh, const Parameters& params);

   /// @brief Solve; writes ParaView output when output.enabled. Collective.
   /// @return Whether the CG converged.
   bool Solve();

   /**
    * @brief The deck's analysis: Solve() when static; when dynamic, a march
    *        from rest at t = 0 to time.t_final in steps of time.dt (the last
    *        one shortened), with `<path>/<name>_history.csv` when
    *        output.history (t, energies, the probe's displacement) and
    *        ParaView output every output.interval steps when output.enabled.
    *        Collective.
    * @return Whether every CG converged.
    */
   bool Run();

   /// @return The solver (displacement, energies, iterations).
   LinearElasticity& Solver() { return *solver_; }

   /**
    * @brief Write the run summary (JSON) on rank 0. Collective.
    * @param path   File path.
    * @param status "running", "ok" or "diverged".
    */
   void WriteSummary(const std::string& path, const std::string& status);

private:
   /// @return The probe's displacement (zero if absent). Collective.
   mfem::Vector ProbeDisplacement();

   mfem::ParMesh& mesh_;      ///< The mesh (borrowed).
   const Parameters& params_; ///< The deck's parameters (borrowed).
   std::unique_ptr<LinearElasticity> solver_; ///< The solver.
   /// Deck-given displacements, tractions and body force (outlive the solves).
   std::vector<std::unique_ptr<mfem::VectorCoefficient>> coeffs_;
   double solve_seconds_ = 0.0; ///< Wall time of Solve() or the march.
   bool solved_ = false;        ///< Whether Solve() or a march ran.
   bool converged_ = true;      ///< Whether every solve converged.
};

} // namespace incns
