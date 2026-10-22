/**
 * @file interface_transfer.hpp
 * @brief Interpolates a field of one mesh onto the boundary nodes of
 *        another: the interface map of a partitioned fluid-structure
 *        coupling (docs/fsi.md).
 */
#pragma once

#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief Point interpolation from a source mesh to the nodes of a
 *        destination space on some of its boundaries.
 *
 * At construction the destination's nodes on the given boundary attributes
 * are located in the source mesh (both in their reference configurations:
 * neither mesh may move between construction and Transfer()). Every rank
 * gathers every node and searches its own elements (Mesh::FindPoints, a local
 * search); the lowest rank that finds a node evaluates the source there.
 * A node no rank finds aborts. Transfer() then needs one reduction of the
 * interface's values, no search.
 *
 * On matching meshes (the destination's nodes are source nodes, as on the
 * Turek-Hron fluid and solid meshes) and a source of at most the
 * destination's order, the result is the source's nodal values: the map is
 * exact. On non-matching interfaces it is interpolation, not a conservative
 * projection; the gslib-based map (MFEM's FindPointsGSLIB) is the planned
 * alternative for large interfaces.
 */
class InterfaceTransfer
{
public:
   /**
    * @param source      Mesh of the fields to transfer (borrowed; must outlive
    *                    this object and not move).
    * @param destination Space of the transferred field (borrowed): a vector
    *                    H1 space whose vdim is the source field's.
    * @param attributes  Boundary attributes of the destination's mesh whose
    *                    nodes receive values.
    */
   InterfaceTransfer(mfem::ParMesh& source,
                     mfem::ParFiniteElementSpace& destination,
                     const std::vector<int>& attributes);

   /**
    * @brief Set @p out (on the destination space) to @p in (on a space of
    *        the source mesh) at the interface nodes. Other entries are left
    *        alone. Collective.
    * @param in  Field on a space of the source mesh.
    * @param out Field on the destination space.
    */
   void Transfer(const mfem::ParGridFunction& in,
                 mfem::ParGridFunction& out) const;

   /// @return The interface nodes in the whole problem (all ranks).
   int GlobalPoints() const { return n_global_; }

   /// @return The local vdofs that receive values (vdim per node, by
   ///         component: entry k of component c at [c * LocalPoints() + k]).
   const mfem::Array<int>& Vdofs() const { return vdofs_; }

   /// @return The interface nodes this rank fills.
   int LocalPoints() const { return n_local_; }

private:
   mfem::ParMesh& source_;            ///< Source mesh (borrowed).
   mfem::ParFiniteElementSpace& dst_; ///< Destination space (borrowed).
   int vdim_ = 0;                     ///< Components per node.
   int n_local_ = 0;                  ///< This rank's interface nodes.
   int n_global_ = 0;                 ///< All ranks' interface nodes.
   mfem::Array<int> vdofs_;           ///< Destination vdofs, by component.
   std::vector<int> offsets_;         ///< Each rank's first point (+ total).
   std::vector<int> counts_;          ///< Each rank's point count.
   /// Points this rank evaluates (it owns them): global index, element, ip.
   std::vector<int> owned_;
   std::vector<int> owned_elem_;                   ///< Their source elements.
   std::vector<mfem::IntegrationPoint> owned_ip_;  ///< Their reference points.
};

} // namespace incns
