// Velocity-block preconditioner of the Cahouet-Chabard path across the
// viscous ratio v-hat = nu / (sigma (h/p)^2) (precond/viscous_ratio): one
// generalized-Stokes solve A = sigma M + nu K (+ grad-div gamma = c h_K) on an
// enclosed no-slip box, Q3/Q2, outer FGMRES to 1e-8, for
//   loramg            one LOR-BoomerAMG V-cycle (the CahouetChabardConfig
//                     default),
//   jacobi_chebyshev  order-4 Chebyshev in the Jacobi-preconditioned block,
//   jacobi_pcg        CG with Jacobi to rtol 1e-2 (at most 50 iterations).
// sigma sweeps from 0 (steady: viscous-dominated, v-hat = inf) to 1e4 (deep
// mass-dominated, the CFL-limited steps of explicit convection). Prints outer
// iterations, inner (PCG) iterations per outer iteration, and solve wall
// time. Not a ctest -- run by hand:
//   mpirun -np 4 build/cpu/bench/bench_velocity_pc [-d 2] [-n 16] [-nu 1e-3]

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "precond/viscous_ratio.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace mfem;

int main(int argc, char** argv)
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   int dim = 2, n = 16;
   double nu = 1e-3;
   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim", "Dimension (2 or 3).");
   args.AddOption(&n, "-n", "--n", "Elements per direction.");
   args.AddOption(&nu, "-nu", "--nu", "Viscosity.");
   args.ParseCheck();

   incns::RuleBook rules; // before the mesh (rule-keyed caches)
   incns::BoxSpec box;
   box.dim = dim;
   box.num_elems = {n, n, n};
   box.lengths = {1.0, 1.0, 1.0};
   box.periodic = {false, false, false};
   Mesh serial = incns::MakeBoxMesh(box);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::MixedSpaces spaces(mesh, 3, 2);
   incns::BoundaryConditions bc(spaces.Velocity());
   for (int a : mesh.bdr_attributes) { bc.AddNoSlip(a); }
   incns::ViscousRatioDiagnostic vr(spaces.Velocity(), rules);

   VectorFunctionCoefficient forcing(dim, [](const Vector & x, Vector & f)
   {
      f = 0.0;
      f(0) = std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
      f(1) = -std::cos(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]);
   });

   if (Mpi::Root())
   {
      std::printf("dim=%d n=%d Q3/Q2 nu=%g np=%d\n", dim, n, nu,
                  Mpi::WorldSize());
      std::printf("%-6s %-8s %-9s | %-26s | %-26s | %-26s\n", "gd", "sigma",
                  "vhat", "loramg: its  inner  time", "jacobi_cheb: its  time",
                  "jacobi_pcg: its inner time");
   }
   for (double gd : {0.0, 1.0})
   {
      for (double sigma : {0.0, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4})
      {
         double res[3][3];
         int k = 0;
         for (incns::APC apc : {incns::APC::LORAMG, incns::APC::JacobiChebyshev,
                                incns::APC::JacobiPCG})
         {
            incns::StokesSolverOptions so;
            so.nu = nu;
            so.mass_coeff = sigma;
            so.grad_div = gd;
            so.schur = incns::SchurBlockType::CahouetChabard;
            so.cc.a_pc = apc;
            so.rtol = 1e-8;
            so.max_iter = 2000;
            so.kdim = 400;
            incns::StokesSolver solver(spaces, rules, bc, so);
            ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
            solver.Solve(forcing, u, p);
            const incns::SolveStats& st = solver.Stats();
            res[k][0] = solver.Converged() ? solver.Iterations() : -1;
            res[k][1] = st.outer_iterations > 0 ?
                        double(st.vel_inner_iterations) / st.outer_iterations : 0.0;
            res[k][2] = st.solve_time;
            ++k;
         }
         const double vhat = sigma > 0.0 ? vr.VHatMax(sigma, nu)
                             : std::numeric_limits<double>::infinity();
         if (Mpi::Root())
         {
            std::printf("%-6g %-8g %-9.3g | %5.0f %6.1f %8.3f s    | %5.0f %8.3f s"
                        "        | %5.0f %5.1f %8.3f s\n", gd, sigma, vhat,
                        res[0][0], res[0][1], res[0][2], res[1][0], res[1][2],
                        res[2][0], res[2][1], res[2][2]);
            std::fflush(stdout);
         }
      }
   }
   return 0;
}
