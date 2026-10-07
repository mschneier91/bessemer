/**
 * @file mesh_adapter.hpp
 * @brief Applies refinements to the mesh and carries fields across (exact
 *        transfer), with optional rebalancing (amr_spec.md, Section 5.1).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief One refinement batch exactly as MeshAdapter::Refine applied it, with
 *        rank-LOCAL element indices.
 *
 * Replaying a run's records in order on the same initial mesh at the same rank
 * count reproduces the refined mesh, its partition AND its entity/dof
 * numbering exactly (the same sequence of deterministic MFEM operations) --
 * which is how a checkpoint restores an adapted mesh without relying on a
 * printed mesh being renumbered the same way on reload.
 */
struct RefinementRecord
{
   mfem::Array<mfem::Refinement> refs; ///< Local refinements of the batch.
   int nc_limit = 1;                   ///< NC limit it was applied with.
   bool rebalance = false;             ///< Whether it was rebalanced after.
};

/**
 * @brief Refine (and optionally rebalance) a mesh in place, updating the
 *        mixed spaces, the boundary conditions and a set of fields.
 *
 * Refinement nests the spaces, so MFEM's update interpolation reproduces each
 * field exactly on the refined mesh; rebalancing only moves elements between
 * ranks. The ParMesh object keeps its identity, so references to it (spaces,
 * output collections) stay valid. Every other mesh-dependent object --
 * operators, solvers, preconditioners -- must be rebuilt by the caller.
 *
 * MFEM's ordering rule, enforced here: after EACH mesh change (refinement,
 * then rebalancing) the spaces are updated and then every field, and the
 * spaces' transfer operators are released only after the last field.
 */
class MeshAdapter
{
public:
   /**
    * @brief Refine, optionally rebalance, and transfer. Collective.
    * @param mesh      Nonconforming-ready partitioned mesh (refined in place).
    * @param spaces    Spaces on @p mesh (updated in place).
    * @param bc        Boundary conditions over spaces.Velocity() (their
    *                  essential dofs are rebuilt); may be null.
    * @param refs      Local refinements (from RefinementMarker::Mark).
    * @param nc_limit  Max refinement-level difference across faces (>= 1).
    * @param rebalance Re-partition afterwards (only acts at np > 1).
    * @param fields    Fields on spaces.Velocity() or spaces.Pressure() to carry
    *                  across (not owned); each is updated in place.
    */
   static void Refine(mfem::ParMesh& mesh, MixedSpaces& spaces,
                      BoundaryConditions* bc,
                      const mfem::Array<mfem::Refinement>& refs, int nc_limit,
                      bool rebalance,
                      const std::vector<mfem::ParGridFunction*>& fields);
};

} // namespace incns
