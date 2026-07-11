// Sprint 1.6 green criterion -- Schur block quality (pressure mass):
// on a small fully periodic 2D Stokes problem at FIXED dt (momentum block
// mass_coeff*M + nu*K), FGMRES with the S_hat^{-1} = nu*M_p^{-1} Schur block
// converges in a bounded, MESH-ROBUST iteration count. dt-robustness is
// deliberately NOT asserted -- iteration growth at small dt is the known,
// accepted Sprint-1 limitation (Cahouet-Chabard fixes it in Sprint 2).
//
// Baselines live in test/baselines.yaml (single source of truth -- never magic
// numbers here). Jacobi-preconditioned counts must agree across np to within
// the stored band (a wider spread is a parallel bug, not noise).

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <string>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesSolver;
using incns::StokesSolverOptions;

namespace
{

// One implicit solve on the fully periodic [0,2pi]^2 box at fixed dt.
int SolveAndCountIterations(int n, double nu, double mass_coeff, double rtol)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, 0}; // fully periodic (default), TGV box lengths (2pi)
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity()); // empty: fully periodic

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.mass_coeff = mass_coeff;
   opts.rtol = rtol;
   opts.max_iter = 2000;
   opts.kdim = 300;
   StokesSolver solver(spaces, rules, bc, opts);

   // TGV-shaped periodic forcing (divergence-free, zero mean).
   VectorFunctionCoefficient forcing(2, [](const Vector & x, Vector & f)
   {
      f(0) = std::sin(x[0]) * std::cos(x[1]);
      f(1) = -std::cos(x[0]) * std::sin(x[1]);
   });

   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);
   EXPECT_TRUE(solver.Converged()) << "n=" << n;
   return solver.Iterations();
}

} // namespace

TEST(SchurQuality, MeshRobustIterationsAtFixedDt)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const YAML::Node node = root["solver_iterations"]["schur_quality_2d_periodic"];
   ASSERT_TRUE(node) << "baseline block missing from baselines.yaml";

   const double nu = node["nu"].as<double>();
   const double mass_coeff = node["mass_coeff"].as<double>();
   const double rtol = node["rtol"].as<double>();
   const int band = node["band"].as<int>();
   const int flat_band = node["flat_band"].as<int>();
   const YAML::Node base = node["jacobi"];

   int iters_min = 1 << 30, iters_max = 0;
   for (const auto& entry : base)
   {
      const std::string key = entry.first.as<std::string>(); // "n4", "n8", ...
      const int n = std::stoi(key.substr(1));
      const int expected = entry.second.as<int>();

      const int iters = SolveAndCountIterations(n, nu, mass_coeff, rtol);
      if (Mpi::Root())
      {
         mfem::out << "[schur] n=" << n << " iters=" << iters
                   << " baseline=" << expected << std::endl;
      }
      EXPECT_LE(std::abs(iters - expected), band)
            << "n=" << n << ": iteration count drifted from baseline";
      iters_min = std::min(iters_min, iters);
      iters_max = std::max(iters_max, iters);
   }

   // Mesh robustness: the count may not grow materially with refinement.
   EXPECT_LE(iters_max - iters_min, flat_band)
         << "iteration count is not mesh-robust";
}
