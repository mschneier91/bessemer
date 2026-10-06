#include "amr/mesh_adapter.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

void MeshAdapter::Refine(ParMesh& mesh, MixedSpaces& spaces,
                         BoundaryConditions* bc,
                         const Array<Refinement>& refs, int nc_limit,
                         bool rebalance,
                         const std::vector<ParGridFunction*>& fields)
{
   INCNS_PROFILE("amr::refine");
   MFEM_VERIFY(mesh.Nonconforming(), "mesh_adapter: the mesh is conforming -- "
               "build it with MakeCaseMesh/PartitionMesh(nonconforming) so it "
               "can be refined");
   MFEM_VERIFY(nc_limit >= 1, "mesh_adapter: nc_limit must be >= 1");
   for (const ParGridFunction* gf : fields)
   {
      MFEM_VERIFY(gf->ParFESpace() == &spaces.Velocity() ||
                  gf->ParFESpace() == &spaces.Pressure(),
                  "mesh_adapter: every field must live on the mixed spaces");
   }

   auto update_all = [&]()
   {
      spaces.Update();
      for (ParGridFunction* gf : fields) { gf->Update(); }
   };

   mesh.GeneralRefinement(refs, /*nonconforming=*/1, nc_limit);
   update_all();
   if (rebalance && mesh.GetNRanks() > 1)
   {
      INCNS_PROFILE("amr::rebalance");
      mesh.Rebalance();
      update_all();
   }
   spaces.UpdatesFinished();
   if (bc) { bc->Update(); }
}

} // namespace incns
