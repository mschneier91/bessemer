/**
 * @file mesh_file.hpp
 * @brief Meshes from files (deck `mesh.geometry: file`): Gmsh (.msh, ASCII
 *        2.2 or 4.x) or MFEM (.mesh), quadrilaterals or hexahedra, with the
 *        boundary names a Gmsh file carries in its physical groups.
 */
#pragma once

#include "mfem.hpp"

#include <string>
#include <utility>
#include <vector>

namespace incns
{

/**
 * @brief The named physical groups of one dimension in a Gmsh file, from its
 *        `$PhysicalNames` section (the same in format 2.2 and 4.x). MFEM reads
 *        a Gmsh element's physical tag as its attribute, so for @p dim = mesh
 *        dimension - 1 these are the boundary names and attributes.
 * @param path Gmsh file; a file without `$PhysicalNames` gives none.
 * @param dim  Dimension of the groups wanted.
 * @return (name, physical tag) pairs, in file order.
 */
std::vector<std::pair<std::string, int>>
GmshPhysicalNames(const std::string& path, int dim);

/**
 * @brief Load a mesh file as a serial mesh and check it can be used: the
 *        dimension @p dim, every element a quadrilateral (2D) or hexahedron
 *        (3D), positive attributes on every boundary element.
 * @param path Gmsh or MFEM mesh file.
 * @param dim  Expected spatial dimension (the deck's mesh.dim).
 * @return The mesh (curved, if the file is high order).
 */
mfem::Mesh LoadMeshFile(const std::string& path, int dim);

} // namespace incns
