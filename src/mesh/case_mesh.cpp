#include "mesh/case_mesh.hpp"

#include "mesh/cylinder_channel.hpp"
#include "mesh/mesh_file.hpp"
#include "mesh/periodic_box.hpp"
#include "mesh/square_cylinder.hpp"

#include <algorithm>
#include <cmath>

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
   switch (params.geometry)
   {
      case MeshGeometry::SquareCylinder:
      {
         Mesh serial = MakeSquareCylinderMesh(params.square_cylinder);
         return PartitionMesh(serial, params.amr.enabled, comm);
      }
      case MeshGeometry::CylinderChannel:
      {
         // Level L: counts x 2^L, gradings g -> g^(2^-L), so levels nest (a
         // geometric run of n cells with ratio g splits exactly into 2n cells
         // with ratio sqrt(g)). Curved nodes of the velocity order (>= 2).
         CylinderChannelSpec spec = params.cylinder_channel;
         const int L = params.cylinder_channel_level;
         MFEM_VERIFY(L >= 0 && L <= 4, "case_mesh: cylinder_channel level 0..4");
         const int m = 1 << L;
         const double root = 1.0 / m;
         spec.n_side *= m;
         spec.n_ring *= m;
         spec.n_down *= m;
         spec.n_up *= m;
         spec.n_below *= m;
         spec.n_above *= m;
         spec.ring_grading = std::pow(spec.ring_grading, root);
         spec.down_grading = std::pow(spec.down_grading, root);
         spec.order = std::max(params.order_u, 2);
         Mesh serial = MakeCylinderChannelMesh(spec);
         if (params.mesh.dim == 2)
         {
            return PartitionMesh(serial, params.amr.enabled, comm);
         }
         // 3D (the DFG 3D-xZ channels): extruded in z, curved nodes included
         // (MFEM's Extrude2D extrudes them). Its end faces z = 0 and z = depth
         // are channel walls: they join the walls attribute, found by
         // position so the result does not depend on Extrude2D's numbering.
         const double depth = params.cylinder_channel_depth;
         std::unique_ptr<Mesh> ext(Extrude2D(&serial,
                                             params.cylinder_channel_nz * m, depth));
         Vector c;
         for (int b = 0; b < ext->GetNBE(); ++b)
         {
            ElementTransformation* T = ext->GetBdrElementTransformation(b);
            T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
            if (std::abs(c(2)) < 1e-9 * depth || std::abs(c(2) - depth) < 1e-9 * depth)
            {
               ext->SetBdrAttribute(b, kCylinderWalls);
            }
         }
         ext->SetAttributes();
         return PartitionMesh(*ext, params.amr.enabled, comm);
      }
      case MeshGeometry::File:
      {
         // Every rank reads the file (fine for the meshes a desktop runs; a
         // very large mesh would want a pre-partitioned format).
         Mesh serial = LoadMeshFile(params.mesh_file, params.mesh.dim);
         return PartitionMesh(serial, params.amr.enabled, comm);
      }
      case MeshGeometry::Box:
         break;
   }
   Mesh serial = MakeBoxMesh(params.mesh);
   return PartitionMesh(serial, params.amr.enabled, comm);
}

} // namespace incns
