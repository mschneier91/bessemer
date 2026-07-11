// Generic YAML-driven driver: `run_case <deck.yaml>` -- no recompilation per
// case. The deck supplies parameters, mesh, initial condition (by name), and
// output settings; the case goes through the same StokesCase surface as any
// in-code driver.

#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/stokes_case.hpp"
#include "mfem.hpp"

#include <iostream>

using namespace mfem;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if (argc < 2)
   {
      if (Mpi::Root())
      {
         std::cerr << "usage: run_case <deck.yaml>" << std::endl;
      }
      return 1;
   }

   const incns::Parameters params = incns::Parameters::LoadYAML(argv[1]);

   Mesh serial = incns::MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);

   incns::StokesCase flow_case(mesh, params);
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
