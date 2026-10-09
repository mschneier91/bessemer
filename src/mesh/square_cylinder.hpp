/**
 * @file square_cylinder.hpp
 * @brief Quad mesh of a square cylinder in a large rectangular domain (Joly,
 *        Etienne & Pelletier, J. Fluids Struct. 28 (2012) 232-243, Fig. 3:
 *        inlet 60 D upstream of the square's centre, outlet 120 D downstream,
 *        sides 60 D away).
 */
#pragma once

#include "mfem.hpp"

#include <vector>

namespace incns
{

/// Boundary attributes of MakeSquareCylinderMesh.
enum SquareCylinderAttribute : int
{
   kSquareInflow = 1,  ///< x = -upstream
   kSquareOutflow = 2, ///< x = +downstream
   kSquareSides = 3,   ///< y = -half_height and y = +half_height
   kSquareBody = 4     ///< the square's four faces
};

/**
 * @brief Geometry and resolution of the square-cylinder domain; lengths in
 *        units of the side D (the square is centred at the origin).
 *
 * A tensor-product grid with the square's cells left out:
 *  - along each face, @c n_face cells graded toward the corners (where the
 *    flow separates) by @c corner_ratio; the corner cell width w0 follows;
 *  - upstream and to the sides, widths grow from w0 by @c far_ratio;
 *  - downstream, widths grow from w0 by @c wake_ratio to @c wake_h, stay at
 *    @c wake_h up to x = @c wake_end (the near wake, where the vortex street
 *    forms), then grow by @c far_ratio to the outlet.
 * Each graded segment is rescaled to end exactly on its boundary.
 */
struct SquareCylinderSpec
{
   double side = 1.0;          ///< Square side D.
   double upstream = 60.0;     ///< Inlet distance from the centre.
   double downstream = 120.0;  ///< Outlet distance from the centre.
   double half_height = 60.0;  ///< Side boundaries at y = +-half_height.
   int n_face = 6;             ///< Cells along each face (even).
   double corner_ratio = 1.5;  ///< Growth from the corners to mid-face.
   double far_ratio = 1.3;     ///< Growth upstream, sideways, far downstream.
   double wake_ratio = 1.15;   ///< Growth from the square into the near wake.
   double wake_h = 0.5;        ///< Cell width in the near wake.
   double wake_end = 10.0;     ///< x where the near wake's uniform cells end.
};

/**
 * @brief Build the serial square-cylinder quad mesh (straight-sided, Q1
 *        geometry). Boundary attributes: SquareCylinderAttribute; element
 *        attribute 1. Partition the result with PartitionMesh.
 * @param spec Geometry and resolution.
 * @return The mesh.
 */
mfem::Mesh MakeSquareCylinderMesh(const SquareCylinderSpec& spec);

/**
 * @brief The 1D node positions of a spec along x or y (exposed for tests and
 *        for reporting the resolution).
 * @param spec Geometry and resolution.
 * @param x    True for the x direction (with the wake), false for y.
 * @return Increasing node coordinates, first/last on the outer boundary.
 */
std::vector<double> SquareCylinderNodes(const SquareCylinderSpec& spec, bool x);

} // namespace incns
