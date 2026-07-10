/**
 * @file periodic_box.hpp
 * @brief Cartesian quad/hex box mesh factory with optional periodicity.
 */
#ifndef INCNS_MESH_PERIODIC_BOX_HPP
#define INCNS_MESH_PERIODIC_BOX_HPP

#include "mfem.hpp"

#include <array>

namespace incns
{

/// Default box edge length: 2*pi, the natural period for the Taylor-Green vortex.
inline constexpr double kBoxDefaultLength = 6.283185307179586;

/**
 * @brief Specification for a Cartesian box of tensor-product cells (quads in 2D,
 *        hexes in 3D) with optional per-direction periodicity.
 *
 * Every array is indexed by spatial direction; index @c [2] is ignored when
 * @ref dim is 2.
 */
struct BoxSpec
{
   /// Spatial dimension (2 or 3).
   int dim = 2;
   /// Number of elements per direction.
   std::array<int, 3> num_elems = {4, 4, 4};
   /// Physical edge length per direction.
   std::array<double, 3> lengths =
   {kBoxDefaultLength, kBoxDefaultLength, kBoxDefaultLength};
   /// Whether each direction is periodic.
   std::array<bool, 3> periodic = {true, true, true};
};

/**
 * @brief Build a serial Cartesian mesh of quads (2D) or hexes (3D) with the
 *        requested per-direction periodicity.
 *
 * Periodicity is applied at the mesh level via mfem::Mesh::MakePeriodic together
 * with mfem::Mesh::CreatePeriodicVertexMapping. The caller partitions the result
 * into a mfem::ParMesh.
 *
 * @param spec Box dimension, element counts, edge lengths, and periodicity.
 * @return A serial periodic mesh of tensor-product cells.
 *
 * @pre Each periodic direction has at least 3 elements. With fewer, an element
 *      becomes its own wrap-around neighbour on both sides and MFEM rejects the
 *      topology (a face/edge shared by more than two elements). Enforced with
 *      MFEM_VERIFY.
 */
mfem::Mesh MakeBoxMesh(const BoxSpec& spec);

/**
 * @brief Assert that every element is a tensor-product cell.
 *
 * Requires mfem::Geometry::SQUARE in 2D and mfem::Geometry::CUBE in 3D, aborting
 * via MFEM_VERIFY on any simplex or other geometry. Intended to run at mesh-load
 * time so non-conforming meshes are rejected up front, not mishandled downstream.
 *
 * @param mesh Mesh to validate.
 */
void AssertTensorProductGeometry(const mfem::Mesh& mesh);

} // namespace incns

#endif // INCNS_MESH_PERIODIC_BOX_HPP
