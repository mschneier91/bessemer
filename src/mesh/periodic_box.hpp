#ifndef INCNS_MESH_PERIODIC_BOX_HPP
#define INCNS_MESH_PERIODIC_BOX_HPP

#include "mfem.hpp"

#include <array>

namespace incns
{

// Default box edge length: 2*pi, the natural period for the Taylor-Green vortex.
inline constexpr double kBoxDefaultLength = 6.283185307179586;

/// Specification for a Cartesian box of tensor-product cells (quads in 2D,
/// hexes in 3D) with optional per-direction periodicity. Index [2] is ignored
/// when dim == 2.
struct BoxSpec
{
   int dim = 2;                                    // 2 or 3
   std::array<int, 3> num_elems = {4, 4, 4};       // elements per direction
   std::array<double, 3> lengths = {kBoxDefaultLength, kBoxDefaultLength, kBoxDefaultLength};
   std::array<bool, 3> periodic = {true, true, true};
};

/// Build a serial Cartesian mesh of quads (2D) or hexes (3D) with the requested
/// per-direction periodicity (mesh-level, via Mesh::MakePeriodic +
/// CreatePeriodicVertexMapping). The caller partitions it into a ParMesh.
///
/// Precondition: each PERIODIC direction needs >= 3 elements. With fewer, an
/// element becomes its own wrap-around neighbour on both sides and MFEM rejects
/// the topology (a face/edge shared by >2 elements). Enforced with MFEM_VERIFY.
mfem::Mesh MakeBoxMesh(const BoxSpec& spec);

/// Assert every element is a tensor-product cell (Geometry::SQUARE in 2D,
/// Geometry::CUBE in 3D); aborts on any simplex/other geometry. Meant to be
/// called at mesh-load time so non-conforming meshes are rejected up front,
/// not silently mishandled downstream.
void AssertTensorProductGeometry(const mfem::Mesh& mesh);

} // namespace incns

#endif // INCNS_MESH_PERIODIC_BOX_HPP
