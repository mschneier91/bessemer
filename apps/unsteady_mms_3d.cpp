// In-code TIMING driver: a 3D quadratic-in-time manufactured solution (MMS)
// solved as unsteady Stokes on a Dirichlet-walled unit cube, run through the
// SAME production Case pipeline as run_case / apps/taylor_green. The point is a
// runnable, self-validating 3D case that can be TIMED CPU-vs-GPU.
//
// Why it stays exact at any resolution: the spatial fields are low-degree
// polynomials that lie in the Q_{ku}/Q_{ku-1} Taylor-Hood spaces, and the
// quadratic-in-time factor g(t) = 1 + t + t^2/2 is integrated exactly by the
// trapezoidal starter and BDF2. So the whole unsteady solve -- history, the
// starter->BDF2 ramp, forcing at t^{n+1}, and time-dependent Dirichlet data
// re-eliminated every step -- reproduces the exact solution to solver
// tolerance regardless of -n / -dt. That makes ||u - u_exact|| a pure
// correctness signal (~solver tol) that does NOT move as you scale -n up to
// load a GPU; the wall time is what you compare.
//
// Coefficients are lifted verbatim from Unsteady3D in
// test/unsteady_mms_test.cpp: nu = 1.3, u = g(t) (y^2, z^2, x^2),
// p = g(t) (x + y + z - 3/2), forcing f = u_t - nu*lap(u) + grad(p).
//
// Examples (time the march, CPU vs one H100):
//   mpirun -n 1 build/cpu/apps/unsteady_mms_3d  -d cpu  -n 24 -tf 0.2
//   mpirun -n 1 build/cuda/apps/unsteady_mms_3d -d cuda -n 24 -tf 0.2 -prec amg
// Scale -n up until the device wins; small (8^3-class) problems are too little
// work to show a GPU off.

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <iostream>
#include <string>

using namespace mfem;

namespace
{
// The quadratic-in-time factor and its derivative (see unsteady_mms_test.cpp):
// exact for the trapezoidal starter and BDF2, so the MMS reproduces at any dt.
double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }
} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // CLI, so the case can be swept for timing without recompiling per size.
   int n = 16;               // elements per direction (n^3 hexes)
   int order_u = 3;          // velocity order (Q3/Q2 Taylor-Hood by default)
   double dt = 0.02;
   double t_final = 0.2;
   const char* device = "cpu"; // MFEM backend: cpu | cuda | ...
   const char* prec = "jacobi"; // velocity-block preconditioner: jacobi | amg

   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--num-elems", "Elements per direction (n^3 hexes).");
   args.AddOption(&order_u, "-ou", "--order-u", "Velocity polynomial order k_u.");
   args.AddOption(&dt, "-dt", "--dt", "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final", "End time.");
   args.AddOption(&device, "-d", "--device", "MFEM backend: cpu, cuda, ...");
   args.AddOption(&prec, "-prec", "--preconditioner",
                  "Velocity-block preconditioner: jacobi or amg.");
   args.ParseCheck(); // prints options on the root; exits on a bad parse

   incns::Parameters params;
   params.nu = 1.3;                          // matches Unsteady3D
   params.device = device;
   params.mesh.dim = 3;
   params.mesh.num_elems = {n, n, n};
   params.mesh.lengths = {1.0, 1.0, 1.0};    // unit cube [0,1]^3
   params.mesh.periodic = {false, false, false}; // Dirichlet walls all round
   params.order_u = order_u;
   params.order_p = order_u - 1;
   params.dt = dt;
   params.t_final = t_final;
   params.velocity_prec = (std::string(prec) == "amg")
                          ? incns::VelocityPreconditioner::BoomerAMG
                          : incns::VelocityPreconditioner::Jacobi;
   params.output.enabled = false;
   params.Normalize();  // dimensionless inputs: records Re, no rescaling
   incns::ConfigureDevice(params.device);

   const double nu = params.nu;

   // Manufactured fields (lifted verbatim from Unsteady3D).
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * x[1];
      v(1) = G(t) * x[2] * x[2];
      v(2) = G(t) * x[0] * x[0];
   });
   FunctionCoefficient p_exact([](const Vector & x, double t)
   {
      return G(t) * (x[0] + x[1] + x[2] - 1.5);
   });
   // f = u_t - nu lap(u) + grad(p).
   VectorFunctionCoefficient forcing(3, [nu](const Vector & x, double t,
                                     Vector & f)
   {
      f(0) = Gp(t) * x[1] * x[1] + G(t) * (1.0 - 2.0 * nu);
      f(1) = Gp(t) * x[2] * x[2] + G(t) * (1.0 - 2.0 * nu);
      f(2) = Gp(t) * x[0] * x[0] + G(t) * (1.0 - 2.0 * nu);
   });

   Mesh serial = incns::MakeBoxMesh(params.mesh);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::Case mms(mesh, params);

   // Time-dependent Dirichlet walls on all 2*dim box faces (re-eliminated every
   // step). The BC set and coefficients are borrowed -- keep them alive here.
   incns::BoundaryConditions bc(mms.Spaces().Velocity());
   for (int attr = 1; attr <= 2 * params.mesh.dim; ++attr)
   {
      bc.AddVelocityDirichlet(attr, u_exact);
   }
   mms.SetBoundaryConditions(bc);
   mms.SetInitialVelocity(u_exact);
   mms.SetForcing(forcing);

   const HYPRE_BigInt u_dofs = mms.Spaces().GlobalVelocityTDofs();
   const HYPRE_BigInt p_dofs = mms.Spaces().GlobalPressureTDofs();

   // Time the march -- this is the number STEP 4 wants.
   StopWatch sw;
   sw.Start();
   mms.Run();
   sw.Stop();
   const double run_s = sw.RealTime();

   // Correctness: the MMS is reproduced exactly, so this is ~solver tolerance
   // and is independent of resolution (it confirms the solve is still correct).
   u_exact.SetTime(mms.Time());
   p_exact.SetTime(mms.Time());
   const double u_err = mms.Velocity().ComputeL2Error(u_exact);
   const double p_err = mms.Pressure().ComputeL2Error(p_exact);

   if (Mpi::Root())
   {
      std::cout << "unsteady_mms_3d: device=" << params.device
                << " np=" << Mpi::WorldSize() << " n=" << n
                << " order_u=" << order_u << " prec=" << prec
                << " u_dofs=" << u_dofs << " p_dofs=" << p_dofs
                << " dt=" << dt << " t_final=" << t_final
                << "\nunsteady_mms_3d: run_wall_s=" << run_s
                << "  ||u - u_exact||_L2=" << u_err
                << "  ||p - p_exact||_L2=" << p_err
                << "\nunsteady_mms_3d: done" << std::endl;
   }
   return 0;
}
