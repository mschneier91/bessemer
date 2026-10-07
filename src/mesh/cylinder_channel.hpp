/**
 * @file cylinder_channel.hpp
 * @brief Quad mesh of a channel with a circular cylinder -- the DFG "flow
 *        around a cylinder" benchmark geometry (Schaefer & Turek 1996; the
 *        reference lift/drag values of John 2004 are for it).
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/// Boundary attributes of MakeCylinderChannelMesh.
enum CylinderChannelAttribute : int
{
   kCylinderInflow = 1,  ///< x = 0
   kCylinderOutflow = 2, ///< x = length
   kCylinderWalls = 3,   ///< y = 0 and y = height
   kCylinderBody = 4     ///< the cylinder surface
};

/**
 * @brief Geometry and resolution of the cylinder channel. Defaults are the DFG
 *        benchmark: channel [0, 2.2] x [0, 0.41], cylinder of radius 0.05
 *        centred at (0.2, 0.2).
 *
 * Layout: a 3 x 3 block grid with lines at x = 0, cx - a, cx + a, length and
 * y = 0, cy - a, cy + a, height (a = box_half). The centre block is an O-grid
 * ring between the cylinder and the square of half-width a: n_side cells per
 * square side (so 4 n_side around), n_ring cells radially. The other eight
 * blocks are Cartesian and conform to the ring along the square.
 */
struct CylinderChannelSpec
{
   double length = 2.2;  ///< Channel length (x).
   double height = 0.41; ///< Channel height (y).
   double cx = 0.2;      ///< Cylinder centre x.
   double cy = 0.2;      ///< Cylinder centre y.
   double radius = 0.05; ///< Cylinder radius.
   double box_half = 0.1; ///< Half-width a of the O-grid's outer square.
   int n_side = 4;       ///< Cells along each side of the square.
   int n_ring = 3;       ///< Radial cell layers in the ring.
   /// Radial growth ratio of the ring layers (> 1 clusters at the cylinder).
   double ring_grading = 1.5;
   int n_up = 2;         ///< Cells in x upstream of the square.
   int n_down = 16;      ///< Cells in x downstream of the square.
   /// Growth ratio of the downstream cell widths (> 1 coarsens toward the
   /// outflow).
   double down_grading = 1.1;
   int n_below = 2;      ///< Cells in y below the square.
   int n_above = 2;      ///< Cells in y above the square.
   int order = 3;        ///< Geometry (curvature) order.
};

/**
 * @brief Build the serial cylinder-channel quad mesh with exactly curved
 *        geometry: the ring elements' high-order nodes are placed by the same
 *        transfinite map that defines the ring (circle-to-square blending
 *        along rays from the centre), so the cylinder is represented to the
 *        geometry order, and shared nodes agree between elements.
 *
 * Boundary attributes: CylinderChannelAttribute. Element attribute 1.
 * Partition the result with PartitionMesh (nonconforming for AMR).
 *
 * @param spec Geometry and resolution.
 * @return The mesh (2D, quads, curved with Nodes of order spec.order).
 */
mfem::Mesh MakeCylinderChannelMesh(const CylinderChannelSpec& spec);

} // namespace incns
