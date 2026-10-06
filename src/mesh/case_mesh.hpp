/**
 * @file case_mesh.hpp
 * @brief The single place a case's partitioned mesh is built: the box factory,
 *        made nonconforming-ready when AMR is on.
 */
#pragma once

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Partition a serial mesh, optionally as a nonconforming (AMR-ready)
 *        mesh.
 *
 * Nonconforming: the serial mesh is converted with Mesh::EnsureNCMesh()
 * BEFORE partitioning -- MFEM cannot convert a conforming ParMesh afterwards
 * (ParMesh::NonconformingRefinement aborts). The METIS partition that
 * ParMesh(comm, serial) would compute is generated first and passed
 * explicitly: an NC ParMesh built without one uses NCMesh's space-filling-
 * curve partition instead, which would shift every per-np iteration count
 * merely by enabling AMR. No refinement happens here, so the discretization
 * is unchanged.
 *
 * @param serial         Serial quad/hex mesh (modified: becomes NC when
 *                       @p nonconforming).
 * @param nonconforming  Build an NC (refinable) ParMesh.
 * @param comm           Communicator.
 * @return The partitioned mesh.
 */
std::unique_ptr<mfem::ParMesh> PartitionMesh(mfem::Mesh& serial,
      bool nonconforming,
      MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build the partitioned mesh for a case: the box from
 *        Parameters::mesh, nonconforming when Parameters::amr is enabled.
 * @param params Normalized case parameters.
 * @param comm   Communicator.
 * @return The partitioned mesh (the caller owns it and must keep it alive
 *         for the Case's lifetime).
 */
std::unique_ptr<mfem::ParMesh> MakeCaseMesh(const Parameters& params,
      MPI_Comm comm = MPI_COMM_WORLD);

} // namespace incns
