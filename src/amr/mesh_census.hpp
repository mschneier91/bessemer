/**
 * @file mesh_census.hpp
 * @brief Where an adaptively refined mesh has its elements: counts per
 *        streamwise band, after every refinement pass of a checkpointed run.
 *
 * Used to check an `amr.end_time` freeze of a bluff-body run: refinement is
 * refinement-only, so the mesh keeps growing behind the
 * vortices; freezing it is safe for the forces if the bands near the body have
 * stopped growing and the growth is far downstream. apps/mesh_census runs it.
 */
#pragma once

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <string>
#include <vector>

namespace incns
{

/// Elements per band of the streamwise coordinate x - x0.
struct MeshCensus
{
   long long total = 0;                 ///< Global element count.
   std::vector<long long> count;        ///< Elements per band.
   std::vector<long long> refined;      ///< Of those, refined at least once.
   /// Largest |y - y0| of a refined element per band (0 = none refined).
   std::vector<double> max_abs_y_refined;
   /// Smallest element size per band, sqrt(area) in 2D, cbrt(volume) in 3D
   /// (0 = empty band).
   std::vector<double> min_size;
   std::vector<int> max_depth;          ///< Deepest refinement per band.
};

/**
 * @brief Count the elements of @p mesh per band of x - x0 (collective).
 * @param mesh  The mesh (nonconforming for refinement depths; conforming
 *              meshes report depth 0).
 * @param edges Band edges of x - x0, increasing: bands (-inf, e0), [e0, e1),
 *              ..., [e_last, inf), by each element's centre.
 * @param x0    Origin of x (the body's centre).
 * @param y0    Origin of y for max_abs_y_refined.
 * @return The census (the same on every rank).
 */
MeshCensus TakeMeshCensus(mfem::ParMesh& mesh, const std::vector<double>& edges,
                          double x0 = 0.0, double y0 = 0.0);

/**
 * @brief Rebuild a run's mesh from its checkpoint, one refinement pass at a
 *        time, and take a census after each (collective). The base mesh comes
 *        from @p params (the run's deck), the passes from the checkpoint's
 *        refinement record, replayed exactly as a restart does.
 * @param params Normalized parameters of the run (its deck).
 * @param dir    Checkpoint directory; it must have been written at this
 *               number of ranks (checked).
 * @param edges  Band edges, as TakeMeshCensus.
 * @param x0     Origin of x.
 * @param y0     Origin of y.
 * @return The census of the base mesh, then one per recorded pass.
 */
std::vector<MeshCensus> ReplayMeshCensus(const Parameters& params,
      const std::string& dir,
      const std::vector<double>& edges,
      double x0 = 0.0, double y0 = 0.0);

} // namespace incns
