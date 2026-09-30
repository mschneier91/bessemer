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
#include <vector>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesSolver;
using incns::StokesSolverOptions;
using incns::VelocityPreconditioner;

namespace
{

// One implicit solve on the fully periodic [0,2pi]^2 box at fixed dt.
int SolveAndCountIterations(int n, double nu, double mass_coeff, double rtol,
                            double grad_div,
                            VelocityPreconditioner prec =
                               VelocityPreconditioner::Jacobi)
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
   opts.grad_div = grad_div;
   opts.rtol = rtol;
   opts.max_iter = 2000;
   opts.kdim = 300;
   opts.velocity_prec = prec;
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

// Steady, VISCOUS-DOMINATED (mass_coeff = 0) Stokes on the unit box with
// homogeneous velocity Dirichlet on every side and a smooth forcing -- the
// regime where the velocity block is Laplacian-like and the preconditioner
// choice actually matters. Returns the FGMRES iteration count.
int SolveViscousDirichletCount(int n, VelocityPreconditioner prec)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, 0};
   s.lengths = {1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;

   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 4; ++attr) { bc.AddNoSlip(attr); }

   StokesSolverOptions opts;
   opts.nu = 1.0;
   opts.mass_coeff = 0.0; // steady: pure viscous velocity block
   opts.rtol = 1e-8;
   opts.max_iter = 2000;
   opts.kdim = 300;
   opts.velocity_prec = prec;
   StokesSolver solver(spaces, rules, bc, opts);

   VectorFunctionCoefficient forcing(2, [](const Vector & x, Vector & f)
   {
      f(0) = std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
      f(1) = std::cos(M_PI * x[0]) * std::cos(M_PI * x[1]);
   });

   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);
   EXPECT_TRUE(solver.Converged()) << "n=" << n;
   return solver.Iterations();
}

// Build a solver ONCE at the reference mass factor c0_ref (freezing the AMG
// hierarchy there when amg_reuse is on), then Refresh it across a sweep of mass
// factors c0, returning the FGMRES count at each. Exercises the refresh
// mechanism and the frozen c0_ref*M + nu*K hierarchy across Delta-t. `periodic`
// selects a fully periodic box (no walls -- nu*K is singular, so the SPD frozen
// operator matters) vs a no-slip box.
std::vector<int> RefreshSweepIterations(int n, VelocityPreconditioner prec,
                                        bool amg_reuse, bool periodic,
                                        double c0_ref,
                                        const std::vector<double>& c0s)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, 0};
   if (!periodic) { s.lengths = {1.0, 1.0}; s.periodic = {false, false, false}; }
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity());
   if (!periodic) { for (int a = 1; a <= 4; ++a) { bc.AddNoSlip(a); } }

   StokesSolverOptions opts;
   opts.nu = 1.0;
   opts.mass_coeff = c0_ref; // AMG hierarchy is frozen HERE when amg_reuse
   opts.rtol = 1e-8;
   opts.max_iter = 2000;
   opts.kdim = 300;
   opts.velocity_prec = prec;
   opts.amg_reuse = amg_reuse;
   StokesSolver solver(spaces, rules, bc, opts);

   VectorFunctionCoefficient forcing(2, [](const Vector & x, Vector & f)
   {
      f(0) = std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
      f(1) = -std::cos(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]);
   });

   std::vector<int> iters;
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   for (double c0 : c0s)
   {
      solver.Refresh(c0);
      solver.Solve(forcing, u, p);
      EXPECT_TRUE(solver.Converged()) << "c0=" << c0;
      iters.push_back(solver.Iterations());
   }
   return iters;
}

} // namespace

// H6 Tier 3: one FROZEN c0_ref*M + nu*K LOR-AMG hierarchy, reused across a
// Delta-t band via Refresh (never rebuilt). Frozen at c0_ref = 100; swept over a
// factor-10 band each way. The preconditioned condition number is bounded by
// ~max(c0/c0_ref, c0_ref/c0), so FGMRES stays bounded while dt hovers near its
// reference -- the empirical proof that freezing is sound AND that the refresh
// mechanism is correct.
TEST(SchurQuality, FrozenAmgRefreshSweepStaysBounded)
{
   const double c0_ref = 100.0;
   const std::vector<double> c0s = {10.0, 30.0, 100.0, 300.0, 1000.0};
   const std::vector<int> amg =
      RefreshSweepIterations(8, VelocityPreconditioner::LORAMG,
                             /*amg_reuse=*/true, /*periodic=*/false, c0_ref, c0s);
   if (Mpi::Root())
   {
      for (std::size_t i = 0; i < c0s.size(); ++i)
      {
         mfem::out << "[frozen_amg] c0=" << c0s[i] << " iters=" << amg[i]
                   << std::endl;
      }
   }
   for (int it : amg) { EXPECT_LE(it, 90) << "frozen AMG not bounded over band"; }
}

// H6 Tier 3, periodic guard: a fully periodic box has NO walls, so nu*K is
// singular and only the SPD frozen operator c0_ref*M + nu*K yields a usable
// hierarchy. The refresh + frozen path must still converge across a Delta-t band.
TEST(SchurQuality, FrozenAmgPeriodicConverges)
{
   const std::vector<double> c0s = {100.0, 50.0, 200.0};
   const std::vector<int> amg =
      RefreshSweepIterations(8, VelocityPreconditioner::LORAMG,
                             /*amg_reuse=*/true, /*periodic=*/true,
                             /*c0_ref=*/100.0, c0s);
   for (int it : amg) { EXPECT_LE(it, 90) << "frozen AMG failed on periodic"; }
}

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

   // Separate baselines for gamma = 0 (jacobi) and gamma > 0 (jacobi_gdpos):
   // grad-div changes the velocity block, and a regression in one must not be
   // masked by the other. gamma NEVER enters the Schur block (user decision) --
   // both runs use the identical nu * M_p^{-1} preconditioner.
   struct Config { const char* key; double c_gd; };
   const Config configs[2] =
   {
      {"jacobi", 0.0},
      {"jacobi_gdpos", node["grad_div_scale"].as<double>()}
   };
   for (const Config& cfg : configs)
   {
      const YAML::Node base = node[cfg.key];
      ASSERT_TRUE(base) << cfg.key << " baseline block missing";

      int iters_min = 1 << 30, iters_max = 0;
      for (const auto& entry : base)
      {
         const std::string key = entry.first.as<std::string>(); // "n4", ...
         const int n = std::stoi(key.substr(1));
         const int expected = entry.second.as<int>();

         const int iters =
            SolveAndCountIterations(n, nu, mass_coeff, rtol, cfg.c_gd);
         if (Mpi::Root())
         {
            mfem::out << "[schur:" << cfg.key << "] n=" << n << " iters="
                      << iters << " baseline=" << expected << std::endl;
         }
         EXPECT_LE(std::abs(iters - expected), band)
               << cfg.key << " n=" << n
               << ": iteration count drifted from baseline";
         iters_min = std::min(iters_min, iters);
         iters_max = std::max(iters_max, iters);
      }

      // Mesh robustness: the count may not grow materially with refinement.
      EXPECT_LE(iters_max - iters_min, flat_band)
            << cfg.key << ": iteration count is not mesh-robust";
   }
}

// H5: LOR-BoomerAMG velocity block in the VISCOUS-DOMINATED regime, where the
// preconditioner choice actually matters. Asserts three things:
//   (1) per-np LOR-AMG counts match the seeded baseline (regression guard);
//   (2) MESH ROBUSTNESS -- the count barely grows n=4..16 (the whole point of
//       the spectrally-equivalent LOR rediscretization);
//   (3) LOR-AMG beats Jacobi decisively at the finest mesh (Jacobi grows
//       ~quadratically here: 63 -> 160 -> 594). Counts are rank-sensitive, so
//       the baseline is stored PER-NP.
TEST(SchurQuality, LorAmgViscousMeshRobust)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const YAML::Node node =
      root["solver_iterations"]["lor_amg_viscous_dirichlet"];
   ASSERT_TRUE(node) << "lor_amg_viscous_dirichlet baseline missing";

   const int band = node["band"].as<int>();
   const int mesh_robust_band = node["mesh_robust_band"].as<int>();
   const int jacobi_ratio = node["jacobi_finest_ratio"].as<int>();

   // On a real CUDA/HIP backend, hypre's BoomerAMG smoother drops from
   // l1-Gauss-Seidel to l1-Jacobi (mfem linalg/hypre.hpp:1161 -- GS is
   // sequential and cannot run on device), which raises the LOR-AMG count by a
   // small constant. Override the host count with the device-seeded `cuda` value
   // FOR THIS np when present; fall back to the host baseline for any np not yet
   // device-calibrated (e.g. np4 before it is seeded), so the test still runs and
   // prints its counts instead of aborting. Mirrors incns_test::ReproTol;
   // DEBUG_DEVICE runs forall bodies on the host and is bitwise-reproducible, so
   // it is deliberately excluded from the mask and keeps the strict host counts.
   const bool real_gpu = mfem::Device::Allows(mfem::Backend::CUDA_MASK |
                         mfem::Backend::HIP_MASK);
   const std::string npkey = "np" + std::to_string(Mpi::WorldSize());
   YAML::Node base = node[npkey];
   if (real_gpu && node["cuda"] && node["cuda"][npkey])
   {
      base = node["cuda"][npkey];
   }
   ASSERT_TRUE(base) << "lor_amg baseline missing for " << npkey;

   int amg_min = 1 << 30, amg_max = 0, amg_finest = 0, n_finest = 0;
   for (const auto& entry : base)
   {
      const std::string key = entry.first.as<std::string>(); // "n4", ...
      const int n = std::stoi(key.substr(1));
      const int expected = entry.second.as<int>();
      const int iters =
         SolveViscousDirichletCount(n, VelocityPreconditioner::LORAMG);
      if (Mpi::Root())
      {
         mfem::out << "[lor_amg " << npkey << "] n=" << n << " iters=" << iters
                   << " baseline=" << expected << std::endl;
      }
      EXPECT_LE(std::abs(iters - expected), band)
            << "lor_amg " << npkey << " n=" << n << ": drifted from baseline";
      amg_min = std::min(amg_min, iters);
      amg_max = std::max(amg_max, iters);
      if (n > n_finest) { n_finest = n; amg_finest = iters; }
   }

   // (2) Mesh robustness -- spectrally-equivalent LOR keeps the count flat.
   EXPECT_LE(amg_max - amg_min, mesh_robust_band)
         << "lor_amg is not mesh-robust (max-min too large)";

   // (3) Decisively better than Jacobi at the finest mesh.
   const int jac_finest =
      SolveViscousDirichletCount(n_finest, VelocityPreconditioner::Jacobi);
   if (Mpi::Root())
   {
      mfem::out << "[lor_amg " << npkey << "] finest n=" << n_finest
                << ": jacobi=" << jac_finest << " lor_amg=" << amg_finest
                << std::endl;
   }
   EXPECT_GT(jac_finest, jacobi_ratio * amg_finest)
         << "lor_amg should decisively beat Jacobi in the viscous regime";
}
