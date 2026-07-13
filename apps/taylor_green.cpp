// Example IN-CODE driver: the 2D Taylor-Green vortex on a fully periodic box,
// solved as unsteady Stokes (the TGV velocity solves unforced Stokes exactly;
// pressure -> 0). Demonstrates the library surface a .cpp case uses --
// Parameters built programmatically, then the same Case the YAML driver
// (run_case) goes through. Prints the final-time velocity error against the
// analytic decay.

#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <iostream>

using namespace mfem;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   incns::Parameters params;
   params.nu = 1.0;
   params.mesh.dim = 2;
   params.mesh.num_elems = {8, 8, 8};
   params.dt = 0.02;
   params.t_final = 0.2;
   params.initial_velocity = "taylor_green_2d";
   params.output.enabled = false;
   params.Normalize(); // dimensionless inputs: records Re, no rescaling

   Mesh serial = incns::MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);

   incns::Case tgv(mesh, params);
   const double nu = params.nu;
   VectorFunctionCoefficient u0(2, [nu](const Vector & x, double t, Vector & u)
   { incns::tgv2d::Velocity(x, t, nu, u); });
   tgv.SetInitialVelocity(u0);
   tgv.Run();

   u0.SetTime(tgv.Time());
   const double err = tgv.Velocity().ComputeL2Error(u0);
   if (Mpi::Root())
   {
      std::cout << "taylor_green: t=" << tgv.Time() << "  ||u - u_exact||_L2="
                << err << "\ntaylor_green: done" << std::endl;
   }
   return 0;
}
