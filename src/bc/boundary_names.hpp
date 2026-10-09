/**
 * @file boundary_names.hpp
 * @brief Boundary names a deck may select by: box faces (xmin ... zmax,
 *        resolved geometrically) or the cylinder geometries' named
 *        boundaries (inflow, outflow, sides / walls, body / cylinder).
 */
#pragma once

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <string>
#include <vector>

namespace incns
{

/**
 * @brief The boundary attribute of a named boundary of the case geometry.
 *        Collective (box faces are found geometrically across ranks).
 * @param p    Parameters (geometry, box lengths and periodicity).
 * @param mesh The case mesh.
 * @param name A boundary name (BoundaryNames lists the valid ones).
 * @return The attribute (aborts on an unknown name or a periodic face).
 */
int NamedBoundaryAttribute(const Parameters& p, mfem::ParMesh& mesh,
                           const std::string& name);

/**
 * @brief Every real (non-periodic) boundary attribute of the case mesh.
 *        Collective.
 * @param p    Parameters.
 * @param mesh The case mesh.
 * @return The attributes, ascending.
 */
std::vector<int> AllBoundaryAttributes(const Parameters& p,
                                       mfem::ParMesh& mesh);

/**
 * @param p Parameters.
 * @return The boundary names valid for p's geometry (real faces only).
 */
std::vector<std::string> BoundaryNames(const Parameters& p);

} // namespace incns
