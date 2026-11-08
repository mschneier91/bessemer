/**
 * @file deck_boundary.hpp
 * @brief The deck's boundary_conditions groups applied to a
 *        BoundaryConditions set: selections resolved to attributes, deck-given
 *        velocities (constant, parabolic or pipe, optionally time-modulated) built
 *        as coefficients, and a coverage check (every real boundary in exactly
 *        one group).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mfem.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace incns
{

/// Number of travelling modes in the pipe profile's perturbation.
constexpr int kPipePerturbationModes = 6;

/**
 * @brief The deck's pipe inflow (profile: pipe) at a point, without the time
 *        modulation: Poiseuille flow plus the optional perturbation.
 *
 * With a the unit axis, r the distance from the axis and rho = r / R:
 * u = [2 u_mean (1 - rho^2) + eps u_mean / sqrt(K) sum_k S_m(rho)
 * cos(m_k theta + phi_k) sin(2 pi f_k t + psi_k)] a, K = 6, m_k = 1 +
 * (k - 1) mod 3, f_k = k f / K, S_m(rho) = rho^m (1 - rho^2) scaled to peak 1,
 * theta measured in a fixed basis of the cross-section, the phases phi_k and
 * psi_k from the seed (the same on every rank and platform). Each mode has
 * m >= 1, so the perturbation carries no net flux through the cross-section,
 * and it vanishes at the wall. Zero outside the pipe (rho >= 1).
 *
 * @param spec A BcSpec with profile Pipe.
 * @param x    The point (3D).
 * @param t    Time (only the perturbation depends on it).
 * @param u    The velocity (resized to 3).
 */
void PipeVelocity(const BcSpec& spec, const mfem::Vector& x, double t,
                  mfem::Vector& u);

/**
 * @brief Builds and owns the boundary data of a deck's boundary_conditions.
 *
 * Shared by Case (decks run by apps/run_case: full coverage required) and the
 * Python bindings (which may add further conditions themselves).
 */
class DeckBoundaryConditions
{
public:
   /// Field of a velocity_dirichlet group, bound by its group name (null if
   /// none is bound).
   using FieldLookup =
      std::function<mfem::VectorCoefficient*(const std::string& group)>;

   /**
    * @brief Resolve every group's selection to attributes. Collective.
    * @param p    Parameters (the groups, geometry, dimension).
    * @param mesh The case mesh.
    */
   DeckBoundaryConditions(const Parameters& p, mfem::ParMesh& mesh);

   /**
    * @return Coverage problems: real boundary attributes no group selects or
    *         that two groups select (empty when every real boundary is in
    *         exactly one group).
    */
   std::vector<std::string> CoverageProblems() const;

   /**
    * @brief Add the groups to @p bc.
    * @param bc               Boundary conditions to fill.
    * @param field            Fields of velocity_dirichlet groups (may be empty:
    *                         such a group is then an error).
    * @param require_coverage Abort with CoverageProblems() unless empty.
    * @param mesh_velocity    The mesh velocity w (ALE): a no_slip group with
    *                         `motion` takes u = w from it (needed when one
    *                         has; borrowed).
    * @param robin_alpha      The Robin coefficient of the interface groups
    *                         under fsi.coupling robin_neumann (borrowed);
    *                         null: the constant Parameters::FsiFluidRobin().
    * @param robin_alpha_max  An upper bound of robin_alpha.
    */
   void Apply(BoundaryConditions& bc, const FieldLookup& field,
              bool require_coverage,
              mfem::VectorCoefficient* mesh_velocity = nullptr,
              mfem::Coefficient* robin_alpha = nullptr,
              double robin_alpha_max = 0.0);

private:
   const Parameters& params_;              ///< The deck's parameters.
   std::vector<std::vector<int>> attrs_;   ///< Attributes per group.
   std::vector<int> real_;                 ///< Every real boundary attribute.
   std::map<int, std::string> attr_name_;  ///< Boundary name per attribute.
   /// Deck-given velocity coefficients (outlive the BoundaryConditions use).
   std::vector<std::unique_ptr<mfem::VectorCoefficient>> coeffs_;
};

} // namespace incns
