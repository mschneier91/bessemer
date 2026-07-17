// CC.8 slow tier -- the FULL T3 (nu, sigma, h, p) robustness grid
// (SPEC par.10 T3; the fast tier carries a representative subset in
// cc_robustness_test). ~240 2D solves + a 3D subset; registered with the
// `slow` ctest label at np = 4 ONLY -- run on demand via
//   scripts/test.sh cpu -L slow
// never in the agentic loop (docs/precond_cc.md).
//
// Asserts: every solve converges, and the outer FGMRES count stays under the
// stored cap across the whole grid (bounded-and-flat is the CC property).
// Prints the full iteration table for the record.

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::SchurBlockType;
using incns::StokesSolver;
using incns::StokesSolverOptions;

namespace
{

int SolveCount(int dim, int n, int ku, double nu, double sigma,
               incns::PcQuadrature pcq = incns::PcQuadrature::Inherit,
               double* wall = nullptr)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity());
   for (int a = 1; a <= 2 * dim; ++a) { bc.AddNoSlip(a); }

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.mass_coeff = sigma;
   opts.rtol = 1e-8;
   opts.max_iter = 1500;
   opts.kdim = 400;
   opts.schur = SchurBlockType::CahouetChabard;
   opts.cc.pc_quadrature = pcq;
   StokesSolver solver(spaces, rules, bc, opts);

   VectorFunctionCoefficient forcing(dim, [dim](const Vector & x, Vector & f)
   {
      f(0) = std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
      f(1) = -std::cos(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]);
      if (dim == 3) { f(2) = 0.0; }
   });
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   const double t0 = MPI_Wtime();
   solver.Solve(forcing, u, p);
   if (wall) { *wall = MPI_Wtime() - t0; }
   EXPECT_TRUE(solver.Converged())
         << "dim=" << dim << " n=" << n << " ku=" << ku << " nu=" << nu
         << " sigma=" << sigma;
   return solver.Iterations();
}

} // namespace

TEST(CcSweepSlow, FullGrid2D)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const int cap =
      root["solver_iterations"]["cc_robustness"]["slow_sweep_cc_cap"].as<int>();

   int worst = 0;
   for (double nu : {1.0, 1e-2, 1e-4, 1e-6})
      for (double sigma : {0.0, 1.0, 1e2, 1e4, 1e6})
         for (int n : {4, 8, 16})
            for (int ku : {2, 3, 4, 5})
            {
               const int it = SolveCount(2, n, ku, nu, sigma);
               if (Mpi::Root())
               {
                  mfem::out << "[sweep2d] nu=" << nu << " sigma=" << sigma
                            << " n=" << n << " Q" << ku << "/Q" << ku - 1
                            << "  iters=" << it << std::endl;
               }
               EXPECT_LE(it, cap);
               worst = std::max(worst, it);
            }
   if (Mpi::Root())
   {
      mfem::out << "[sweep2d] worst outer count over the grid: " << worst
                << " (cap " << cap << ")" << std::endl;
   }
}

TEST(CcSweepSlow, Subset3D)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const int cap =
      root["solver_iterations"]["cc_robustness"]["slow_sweep_cc_cap"].as<int>();

   int worst = 0;
   for (double nu : {1.0, 1e-4})
      for (double sigma : {1.0, 1e4})
         for (int n : {2, 4})
            for (int ku : {2, 3})
            {
               const int it = SolveCount(3, n, ku, nu, sigma);
               if (Mpi::Root())
               {
                  mfem::out << "[sweep3d] nu=" << nu << " sigma=" << sigma
                            << " n=" << n << " Q" << ku << "/Q" << ku - 1
                            << "  iters=" << it << std::endl;
               }
               EXPECT_LE(it, cap);
               worst = std::max(worst, it);
            }
   if (Mpi::Root())
   {
      mfem::out << "[sweep3d] worst outer count over the grid: " << worst
                << " (cap " << cap << ")" << std::endl;
   }
}

// pc_quadrature comparison: Inherit (Chebyshev on the consistent mass) vs
// GllCollocated (PC-own diagonal masses) -- outer iterations AND wall time,
// at Q3 and Q5 (the production order ceiling). Measurement + convergence
// assertions; the default choice is a human decision informed by this table.
TEST(CcSweepSlow, PcQuadratureCompare)
{
   for (int ku : {3, 5})
      for (double sigma : {10.0, 1e3, 1e5})
         for (int n : {8, 16})
         {
            double t_inh = 0.0, t_col = 0.0;
            const int it_inh = SolveCount(2, n, ku, 1e-3, sigma,
                                          incns::PcQuadrature::Inherit, &t_inh);
            const int it_col = SolveCount(2, n, ku, 1e-3, sigma,
                                          incns::PcQuadrature::GllCollocated,
                                          &t_col);
            if (Mpi::Root())
            {
               mfem::out << "[pcq] Q" << ku << " n=" << n << " sigma=" << sigma
                         << "  inherit: " << it_inh << " it, "
                         << 1e3 * t_inh << " ms   gll_colloc: " << it_col
                         << " it, " << 1e3 * t_col << " ms" << std::endl;
            }
         }
}
