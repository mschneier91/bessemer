#include "mesh/case_mesh.hpp"

#include "mesh/periodic_box.hpp"

namespace incns
{

using namespace mfem;

std::unique_ptr<ParMesh> PartitionMesh(Mesh& serial, bool nonconforming,
                                       MPI_Comm comm)
{
   if (!nonconforming) { return std::make_unique<ParMesh>(comm, serial); }

   int np = 1;
   MPI_Comm_size(comm, &np);
   // The partition ParMesh(comm, serial) computes for a conforming mesh
   // (part_method 1 = METIS k-way), taken BEFORE the NC conversion.
   std::unique_ptr<int[]> partition(serial.GeneratePartitioning(np, 1));
   serial.EnsureNCMesh();
   return std::make_unique<ParMesh>(comm, serial, partition.get());
}

std::unique_ptr<ParMesh> MakeCaseMesh(const Parameters& params, MPI_Comm comm)
{
   MFEM_VERIFY(params.nondim.normalized,
               "case_mesh: call Parameters::Normalize() before building the "
               "mesh (LoadYAML does it automatically)");
   Mesh serial = MakeBoxMesh(params.mesh);
   return PartitionMesh(serial, params.amr.enabled, comm);
}

} // namespace incns
