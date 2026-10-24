/**
 * @file mesh_adapter.hpp
 * @brief Applies refinements (exact transfer) and coarsenings (L2 projection)
 *        to the mesh and carries fields across, with optional rebalancing
 *        (docs/design/amr_spec.md, Sections 5.1 and 13).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief One refinement or coarsening batch exactly as MeshAdapter applied it,
 *        with rank-LOCAL element indices.
 *
 * Replaying a run's records in order on the same initial mesh at the same rank
 * count reproduces the adapted mesh, its partition AND its entity/dof
 * numbering exactly (the same sequence of deterministic MFEM operations) --
 * which is how a checkpoint restores an adapted mesh without relying on a
 * printed mesh being renumbered the same way on reload.
 */
struct RefinementRecord
{
   mfem::Array<mfem::Refinement> refs; ///< Local refinements of the batch.
   int nc_limit = 1;                   ///< NC limit it was applied with.
   bool rebalance = false;             ///< Whether it was rebalanced after.
   /// A coarsening batch instead (MeshAdapter::Derefine; refs empty).
   bool derefine = false;
   /// Coarsening: the rank each local element moved to first.
   mfem::Array<int> gather;
   /// Coarsening: 1 for each local element of a family it merged (indexed,
   /// as gather, before that move).
   mfem::Array<int> merge;

   /// A refinement batch.
   static RefinementRecord Refinements(const mfem::Array<mfem::Refinement>& r,
                                       int nc_limit, bool rebalance)
   {
      RefinementRecord rec;
      rec.refs = r;
      rec.nc_limit = nc_limit;
      rec.rebalance = rebalance;
      return rec;
   }
};

/**
 * @brief Refine or coarsen (and optionally rebalance) a mesh in place,
 *        updating the mixed spaces, the boundary conditions and a set of
 *        fields.
 *
 * Refinement nests the spaces, so MFEM's update interpolation reproduces each
 * field exactly on the refined mesh; rebalancing only moves elements between
 * ranks. Coarsening loses information: see Derefine. The ParMesh object keeps
 * its identity, so references to it (spaces, output collections) stay valid.
 * Every other mesh-dependent object -- operators, solvers, preconditioners --
 * must be rebuilt by the caller.
 *
 * MFEM's ordering rule, enforced here: after EACH mesh change (refinement or
 * coarsening, then rebalancing) the spaces are updated and then every field,
 * and the spaces' transfer operators are released only after the last field.
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

   /**
    * @brief The families of sibling elements to merge (collective): those
    *        whose children's largest @p values entry is below @p threshold,
    *        and whose merge keeps @p nc_limit (MFEM's check). A family that
    *        straddles ranks is assigned to the lowest of them, the move
    *        MFEM's ParNCMesh::Derefine would make.
    * @param mesh      Nonconforming partitioned mesh.
    * @param values    One value per local element
    *                  (RefinementMarker::DerefinementValues).
    * @param threshold Merge a family when all its values are below it.
    * @param nc_limit  Max refinement-level difference across faces (>= 1).
    * @param gather    Output: the rank each local element must move to.
    * @param merge     Output: 1 for each local element of a family to merge.
    * @return Global number of families to merge.
    */
   static long long SelectDerefinements(mfem::ParMesh& mesh,
                                        const mfem::Vector& values,
                                        double threshold, int nc_limit,
                                        mfem::Array<int>& gather,
                                        mfem::Array<int>& merge);

   /**
    * @brief Merge the families SelectDerefinements chose, optionally
    *        rebalance, and carry fields across. Collective.
    *
    * MFEM's own derefinement update interpolates a field at the coarse
    * element's nodes: exact for a field the coarse space holds, but otherwise
    * neither conservative nor an L2 contraction. Each field in @p projected
    * (velocity space) is instead replaced by its L2 projection onto the
    * coarse space: the coarse function whose moments against every coarse
    * basis function are the fine field's. Each coarse element's moments are
    * the sum over its children c of P_c^T (M_c u_c), P_c the child's local
    * interpolation from its parent, then one mass solve gives the field. The
    * sum needs every child on its parent's rank, so the families are first
    * gathered there (Rebalance with @p gather). The element masses use MFEM's
    * default rule (2 k + the Jacobian's order), exact on affine elements.
    *
    * @param mesh      Nonconforming partitioned mesh (coarsened in place).
    * @param spaces    Spaces on @p mesh (updated in place).
    * @param bc        Boundary conditions over spaces.Velocity(); may be null.
    * @param gather    From SelectDerefinements (or a RefinementRecord).
    * @param merge     From SelectDerefinements (or a RefinementRecord).
    * @param nc_limit  The same NC limit as the selection.
    * @param rebalance Re-partition afterwards (only acts at np > 1).
    * @param fields    Fields on spaces.Velocity() or spaces.Pressure(),
    *                  carried by MFEM's interpolation (not owned).
    * @param projected Fields on spaces.Velocity(), replaced by their L2
    *                  projections (not owned).
    * @return Whether the mesh coarsened (false if MFEM merged nothing).
    */
   static bool Derefine(mfem::ParMesh& mesh, MixedSpaces& spaces,
                        BoundaryConditions* bc,
                        const mfem::Array<int>& gather,
                        const mfem::Array<int>& merge, int nc_limit,
                        bool rebalance,
                        const std::vector<mfem::ParGridFunction*>& fields,
                        const std::vector<mfem::ParGridFunction*>& projected = {});

   /**
    * @brief Apply one recorded batch again without fields (a restart's
    *        replay): Refine or Derefine as @p rec says. Collective.
    */
   static void Replay(mfem::ParMesh& mesh, MixedSpaces& spaces,
                      BoundaryConditions* bc, const RefinementRecord& rec);
};

} // namespace incns
