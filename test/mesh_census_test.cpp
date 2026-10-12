// Mesh census (amr/mesh_census, apps/mesh_census): where an AMR run put its
// elements.
//  C1 TakeMeshCensus on a box: the counts per streamwise band, the refined
//     counts, the lateral reach and smallest size of refined elements, and the
//     refinement depth, exactly, before and after refining a known strip;
//  C2 ReplayMeshCensus rebuilds a checkpointed AMR run's mesh pass by pass:
//     one census per recorded refinement pass, the last one equal to the
//     run's own final mesh (same global count, same counts per band).

#include <gtest/gtest.h>

#include "amr/mesh_census.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <filesystem>
#include <string>

using namespace mfem;

TEST(MeshCensus, C1_BandsOfABox)
{
   // An 8 x 4 box of unit squares, x in [0, 8]; nonconforming (amr.enabled).
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "mesh:\n  dim: 2\n  elements: [8, 4]\n  lengths: [8.0, 4.0]\n"
                                  "  periodic: [false, false]\n"
                                  "amr:\n  enabled: true\n");
   auto mesh = incns::MakeCaseMesh(p);
   const std::vector<double> edges = {2.0, 4.0};
   incns::MeshCensus c = incns::TakeMeshCensus(*mesh, edges);
   EXPECT_EQ(c.total, 32);
   EXPECT_EQ(c.count, (std::vector<long long> {8, 8, 16}));
   EXPECT_EQ(c.refined, (std::vector<long long> {0, 0, 0}));
   EXPECT_DOUBLE_EQ(c.min_size[1], 1.0);
   EXPECT_EQ(c.max_depth[0], 0);

   // Refine the strip x < 2 once: 8 squares -> 32 of side 1/2.
   Array<int> els;
   Vector ctr(2);
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      mesh->GetElementCenter(e, ctr);
      if (ctr(0) < 2.0) { els.Append(e); }
   }
   mesh->GeneralRefinement(els, 1, 1);
   c = incns::TakeMeshCensus(*mesh, edges, 0.0, 2.0);
   EXPECT_EQ(c.total, 56);
   EXPECT_EQ(c.count, (std::vector<long long> {32, 8, 16}));
   EXPECT_EQ(c.refined, (std::vector<long long> {32, 0, 0}));
   EXPECT_DOUBLE_EQ(c.min_size[0], 0.5);
   EXPECT_EQ(c.max_depth[0], 1);
   EXPECT_DOUBLE_EQ(c.max_abs_y_refined[0],
                    1.75); // centres at y = 0.25 ... 3.75, y0 = 2
   EXPECT_DOUBLE_EQ(c.max_abs_y_refined[2], 0.0);
}

TEST(MeshCensus, C2_ReplayRebuildsTheRunsMesh)
{
   const std::string dir = "mesh_census_c2_np" + std::to_string(Mpi::WorldSize());
   // A lid-driven cavity: the lid's corners refine within a few steps.
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "equation: navier_stokes\n"
                                  "physics:\n  Re: 100\n"
                                  "discretization:\n  order_u: 2\n  order_p: 1\n"
                                  "mesh:\n  dim: 2\n  elements: [6, 6]\n  lengths: [1.0, 1.0]\n"
                                  "  periodic: [false, false]\n"
                                  "time:\n  dt: 0.01\n  t_final: 0.06\n  step_control: fixed\n"
                                  "boundary_conditions:\n"
                                  "  - {select: [ymax], type: velocity, value: [1, 0]}\n"
                                  "  - {select: [xmin, xmax, ymin], type: no_slip}\n"
                                  "amr:\n  enabled: true\n  every_time: 0.02\n  tolerance: 0.05\n"
                                  "checkpoint:\n  enabled: true\n  interval: 6\n  path: " + dir + "\n");
   long long run_ne = 0;
   int passes = 0;
   std::vector<double> edges = {0.25, 0.5, 0.75};
   incns::MeshCensus run_census;
   {
      auto mesh = incns::MakeCaseMesh(p);
      incns::Case flow(*mesh, p);
      auto u0 = incns::MakeInitialVelocity(p);
      flow.SetInitialVelocity(*u0);
      flow.Run();
      run_ne = mesh->GetGlobalNE();
      for (const incns::AdaptStats& a : flow.AdaptHistory()) { passes += a.passes; }
      run_census = incns::TakeMeshCensus(*mesh, edges);
   }
   ASSERT_GT(passes, 0) << "the run did not refine: the test would prove nothing";
   MPI_Barrier(MPI_COMM_WORLD);
   const std::vector<incns::MeshCensus> census =
      incns::ReplayMeshCensus(p, dir, edges);
   EXPECT_EQ(census.size(), static_cast<std::size_t>(passes) + 1);
   EXPECT_LT(census.front().total, run_ne);
   EXPECT_EQ(census.back().total, run_ne);
   EXPECT_EQ(census.back().count, run_census.count);
   EXPECT_EQ(census.back().refined, run_census.refined);
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(dir); }
}
