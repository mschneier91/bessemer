/**
 * @file refinement_marker.hpp
 * @brief Turns the directional indicator into refinements: which elements,
 *        which directions, within the size and element-count caps, and free
 *        of 3D parallel anisotropic conflicts (docs/design/amr_spec.md, Section 3).
 */
#pragma once

#include "amr/amr_parameters.hpp"
#include "mfem.hpp"

namespace incns
{

/// Global statistics of one marking pass (identical on every rank).
struct MarkStats
{
   double max_eta = 0.0;        ///< Global max of eta_K.
   double threshold = 0.0;      ///< eta threshold actually applied.
   long long marked = 0;        ///< Elements marked.
   long long per_dir[3] = {0, 0, 0}; ///< Marked elements splitting each direction.
   long long ne_before = 0;     ///< Global element count before refinement.
   long long projected_ne = 0;  ///< Element count after the marked splits.
   int conflict_upgrades = 0;   ///< Entries upgraded to XYZ (3D parallel).
};

/**
 * @brief Selects refinements from the directional indicator G_{K,d}.
 *
 * 1. eta_K = |G_K|; mark K when eta_K >= threshold (theta * max eta in
 *    relative mode, the tolerance in absolute mode; nothing if max eta = 0).
 * 2. Directions: all (isotropic), or {d : G_{K,d} >= aniso_ratio * max_d G}.
 * 3. Minimum size: direction d of K is not split if the child's extent
 *    |J_{:,d}|/2 (at the element center) would fall below min_size; an
 *    element left with no direction is unmarked.
 * 4. Element cap: if NE + sum(2^(number of split directions) - 1) would
 *    exceed max_elements, the threshold is raised by bisection until it fits
 *    (largest eta first). Indicators equal to within 1e-8 relative are taken
 *    together, so the choice does not depend on roundoff: when the threshold
 *    would split such a group, the whole group is marked if that exceeds the
 *    cap by at most 10%, and none of it otherwise.
 * 5. 3D, np > 1: anisotropic entries in conflict with a face neighbour are
 *    upgraded to XYZ (ResolveAnisotropicConflicts) -- MFEM's parallel
 *    anisotropic hex refinement requires it.
 */
class RefinementMarker
{
public:
   /// @param opts AMR settings (copied).
   explicit RefinementMarker(const AmrParameters& opts) : opts_(opts) {}

   /**
    * @brief Mark refinements on @p mesh (collective).
    * @param mesh  Partitioned mesh (must be nonconforming-ready).
    * @param g     Directional indicator, layout g[d + dim * e], NE local
    *              elements (from GradientIndicator::Compute).
    * @param refs  Output: local refinements (index, direction mask).
    * @param stats Optional global statistics.
    */
   void Mark(mfem::ParMesh& mesh, const mfem::Vector& g,
             mfem::Array<mfem::Refinement>& refs,
             MarkStats* stats = nullptr) const;

   /**
    * @brief Upgrade anisotropic refinements that conflict across a face to
    *        XYZ until mfem::ParMesh::AnisotropicConflict reports none
    *        (at most 5 rounds; then every non-XYZ 3D entry becomes XYZ).
    *        No-op in 2D and at np = 1 (serial NCMesh forces the needed
    *        refinements itself). Collective.
    * @param mesh Partitioned mesh.
    * @param refs Refinements to fix in place.
    * @return Number of local entries upgraded.
    */
   static int ResolveAnisotropicConflicts(mfem::ParMesh& mesh,
                                          mfem::Array<mfem::Refinement>& refs);

   /**
    * @brief Element extents |J_{:,d}| at each local element's center.
    * @param mesh Mesh.
    * @param h    Output, layout h[d + dim * e].
    */
   static void ElementExtents(mfem::Mesh& mesh, mfem::Vector& h);

   /**
    * @brief Coarsening values for MeshAdapter::SelectDerefinements
    *        (collective): 2 eta_K. A merged element's G_{P,d}^2 is the
    *        volume-weighted mean of its children's (s_d G_{c,d})^2, s_d = 2
    *        in the directions the parent was split in and 1 in the others,
    *        so its eta is at most twice its children's largest.
    * @param mesh   Partitioned mesh.
    * @param g      Directional indicator, layout g[d + dim * e].
    * @param values Output: 2 eta_K per local element.
    * @return The threshold a family's values must all be below:
    *         derefine_fraction times the refinement threshold (the tolerance,
    *         or theta times the global max eta), so that a merged element is
    *         not marked again (hysteresis).
    */
   double DerefinementValues(mfem::ParMesh& mesh, const mfem::Vector& g,
                             mfem::Vector& values) const;

private:
   AmrParameters opts_; ///< Settings.
};

} // namespace incns
