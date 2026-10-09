/**
 * @file deck_boundary.hpp
 * @brief The deck's boundary_conditions groups applied to a
 *        BoundaryConditions set: selections resolved to attributes, deck-given
 *        velocities (constant or parabolic, optionally time-modulated) built
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
    */
   void Apply(BoundaryConditions& bc, const FieldLookup& field,
              bool require_coverage);

private:
   const Parameters& params_;              ///< The deck's parameters.
   std::vector<std::vector<int>> attrs_;   ///< Attributes per group.
   std::vector<int> real_;                 ///< Every real boundary attribute.
   std::map<int, std::string> attr_name_;  ///< Boundary name per attribute.
   /// Deck-given velocity coefficients (outlive the BoundaryConditions use).
   std::vector<std::unique_ptr<mfem::VectorCoefficient>> coeffs_;
};

} // namespace incns
