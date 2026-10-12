#include "amr/mesh_census.hpp"

#include "amr/mesh_adapter.hpp"
#include "mesh/case_mesh.hpp"
#include "post/checkpoint.hpp"
#include "spaces/mixed_spaces.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace incns
{

using namespace mfem;

MeshCensus TakeMeshCensus(ParMesh& mesh, const std::vector<double>& edges,
                          double x0, double y0)
{
   MFEM_VERIFY(std::is_sorted(edges.begin(), edges.end()),
               "mesh_census: band edges must increase");
   const std::size_t nb = edges.size() + 1;
   const int dim = mesh.Dimension();
   std::vector<long long> count(nb, 0), refined(nb, 0);
   std::vector<double> max_y(nb, 0.0);
   std::vector<double> min_size(nb, std::numeric_limits<double>::max());
   std::vector<int> depth(nb, 0);
   Vector c(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      const double x = c(0) - x0;
      const std::size_t b =
         static_cast<std::size_t>(std::upper_bound(edges.begin(), edges.end(), x) -
                                  edges.begin());
      ++count[b];
      const double size = std::pow(mesh.GetElementVolume(e), 1.0 / dim);
      min_size[b] = std::min(min_size[b], size);
      const int d = mesh.ncmesh ? mesh.ncmesh->GetElementDepth(e) : 0;
      depth[b] = std::max(depth[b], d);
      if (d > 0)
      {
         ++refined[b];
         max_y[b] = std::max(max_y[b], std::abs(c(1) - y0));
      }
   }
   const MPI_Comm comm = mesh.GetComm();
   const int n = static_cast<int>(nb);
   MPI_Allreduce(MPI_IN_PLACE, count.data(), n, MPI_LONG_LONG, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, refined.data(), n, MPI_LONG_LONG, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, max_y.data(), n, MPI_DOUBLE, MPI_MAX, comm);
   MPI_Allreduce(MPI_IN_PLACE, min_size.data(), n, MPI_DOUBLE, MPI_MIN, comm);
   MPI_Allreduce(MPI_IN_PLACE, depth.data(), n, MPI_INT, MPI_MAX, comm);
   MeshCensus out;
   for (std::size_t b = 0; b < nb; ++b)
   {
      out.total += count[b];
      if (count[b] == 0) { min_size[b] = 0.0; }
   }
   out.count = std::move(count);
   out.refined = std::move(refined);
   out.max_abs_y_refined = std::move(max_y);
   out.min_size = std::move(min_size);
   out.max_depth = std::move(depth);
   return out;
}

std::vector<MeshCensus> ReplayMeshCensus(const Parameters& params,
      const std::string& dir,
      const std::vector<double>& edges,
      double x0, double y0)
{
   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   const YAML::Node meta = YAML::LoadFile(dir + "/meta.yaml");
   MFEM_VERIFY(meta["np"] && meta["np"].as<int>() == np, "mesh_census: the "
               "checkpoint " << dir << " was written at "
               << (meta["np"] ? meta["np"].as<int>() : -1) << " ranks; replay it at "
               "the same number (this run has " << np << ")");
   MFEM_VERIFY(params.amr.enabled, "mesh_census: the deck has amr.enabled false; "
               "its mesh would be conforming, not the run's");
   const std::vector<RefinementRecord> recs = Checkpoint::ReadRefinements(dir);

   // The same base mesh, partition and refinement sequence a restart uses
   // (Case::EnsureSetup): the spaces only follow the mesh here.
   std::unique_ptr<ParMesh> mesh = MakeCaseMesh(params);
   MixedSpaces spaces(*mesh, params.order_u, params.order_p);
   std::vector<MeshCensus> out;
   out.push_back(TakeMeshCensus(*mesh, edges, x0, y0));
   for (const RefinementRecord& r : recs)
   {
      MeshAdapter::Refine(*mesh, spaces, nullptr, r.refs, r.nc_limit, r.rebalance,
                          {});
      out.push_back(TakeMeshCensus(*mesh, edges, x0, y0));
   }
   return out;
}

} // namespace incns
