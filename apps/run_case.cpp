// Generic YAML-driven driver: `run_case <deck.yaml>` -- no recompilation per
// case. The deck supplies parameters, mesh, initial condition (by name), and
// output settings; the case goes through the same Case surface as any
// in-code driver. `run_case --deck-reference` prints every deck key with its
// default (docs/deck_reference.md is that output).

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <cstring>
#include <iostream>
#include <memory>

using namespace mfem;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if (argc < 2 || !std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h"))
   {
      if (Mpi::Root())
      {
         std::cerr << "usage: run_case <deck.yaml>\n"
                   "       run_case --deck-reference   (every deck key, Markdown)"
                   << std::endl;
      }
      return argc < 2 ? 1 : 0;
   }
   if (!std::strcmp(argv[1], "--deck-reference"))
   {
      if (Mpi::Root()) { std::cout << incns::Parameters::DeckReferenceMarkdown(); }
      return 0;
   }

   const incns::Parameters params = incns::Parameters::LoadYAML(argv[1]);
   incns::ConfigureDevice(params.device);

   // The one mesh factory: nonconforming-ready when params.amr is enabled.
   std::unique_ptr<ParMesh> pmesh = incns::MakeCaseMesh(params);
   ParMesh& mesh = *pmesh;

   incns::Case flow_case(mesh, params);
   auto u0 = incns::MakeInitialVelocity(params);
   flow_case.SetInitialVelocity(*u0);
   flow_case.Run();

   // Global diagnostics on rank 0 only.
   Vector u_true(flow_case.Spaces().Velocity().GetTrueVSize());
   flow_case.Velocity().GetTrueDofs(u_true);
   const double u_norm =
      std::sqrt(InnerProduct(MPI_COMM_WORLD, u_true, u_true));
   if (Mpi::Root())
   {
      std::cout << "run_case: t=" << flow_case.Time() << "  ||u||=" << u_norm
                << "\nrun_case: done" << std::endl;
   }
   return 0;
}
