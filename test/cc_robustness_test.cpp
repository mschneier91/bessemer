// CC.8 -- T3 (fast-tier subset) + T5 (legacy regression), the tests that
// encode WHY Cahouet-Chabard exists and why the consistent BM^-1B^T is the
// default:
//
//  * T3 sigma-sweep (Delta-t robustness): outer FGMRES counts with the CC
//    Schur block stay bounded/flat from the viscous limit (sigma = 0) to the
//    deep mass-dominated limit (sigma = 1e5), where the Sprint-1 mass block
//    provably degrades. This is the property the whole preconditioner buys.
//  * T3 nu-sweep at tau ~ h: CC counts bounded over nu in [1e-6, 1].
//  * T5: on a finer mesh at tight tolerance, LaplacianLegacy needs measurably
//    more iterations than ConsistentBMB -- the guard that permanently encodes
//    the design decision (Creff-Guermond).
//
// Caps/bands live in test/baselines.yaml (single source of truth). The full
// (nu, sigma, h, p) grid is the `slow`-tier sweep (cc_sweep_slow_test).

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
#include <string>
#include <vector>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::SchurBlockType;
using incns::SchurModel;
using incns::StokesSolver;
using incns::StokesSolverOptions;

namespace
{

// One implicit generalized-Stokes solve (enclosed no-slip unit box, Q3/Q2,
// singular pressure) at the given (nu, sigma); returns FGMRES iterations.
int SolveCount(int n, double nu, double sigma, double rtol,
               SchurBlockType schur,
               SchurModel model = SchurModel::ConsistentBMB, int ku = 3,
               bool with_outflow = false)
{
   // The MASS path gets the LOR-AMG velocity block so the comparison isolates
   // the SCHUR block (CC's velocity block is LOR-AMG by default; a Jacobi
   // velocity block would dominate the mass-path counts in viscous regimes).
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, 0};
   s.lengths = {1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh serial = MakeBoxMesh(s);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;
   BoundaryConditions bc(spaces.Velocity());
   const int last = with_outflow ? 3 : 4;
   for (int a = 1; a <= last; ++a) { bc.AddNoSlip(a); }
   if (with_outflow) { bc.AddOutflow(4); }

   StokesSolverOptions opts;
   opts.nu = nu;
   opts.mass_coeff = sigma;
   opts.rtol = rtol;
   opts.max_iter = 1500;
   opts.kdim = 400;
   opts.schur = schur;
   opts.cc.schur_model = model;
   opts.velocity_prec = incns::VelocityPreconditioner::LORAMG;
   StokesSolver solver(spaces, rules, bc, opts);

   VectorFunctionCoefficient forcing(2, [](const Vector & x, Vector & f)
   {
      f(0) = std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
      f(1) = -std::cos(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]);
   });
   ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
   solver.Solve(forcing, u, p);
   EXPECT_TRUE(solver.Converged())
         << "nu=" << nu << " sigma=" << sigma << " schur="
         << static_cast<int>(schur);
   return solver.Iterations();
}

} // namespace

// T3 sigma-sweep: CC bounded/flat from viscous to deep mass-dominated; the
// mass block degrades at large sigma (the KNOWN Sprint-1 limitation CC fixes).
TEST(CcRobustness, SigmaSweepDeltaTRobust)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const YAML::Node node = root["solver_iterations"]["cc_robustness"];
   ASSERT_TRUE(node) << "cc_robustness baseline block missing";
   const int cc_cap = node["sigma_sweep_cc_cap"].as<int>();
   const double mass_ratio = node["sigma_mass_degradation_ratio"].as<double>();

   const std::vector<double> sigmas = {0.0, 10.0, 1e3, 1e5};
   int cc_max = 0, mass_at_stiff = 0, cc_at_stiff = 0;
   for (double sg : sigmas)
   {
      const int cc = SolveCount(8, 1.0, sg, 1e-8, SchurBlockType::CahouetChabard);
      const int ms = SolveCount(8, 1.0, sg, 1e-8, SchurBlockType::Mass);
      if (Mpi::Root())
      {
         mfem::out << "[t3-sigma] sigma=" << sg << "  cc=" << cc
                   << "  mass=" << ms << std::endl;
      }
      cc_max = std::max(cc_max, cc);
      if (sg == sigmas.back()) { cc_at_stiff = cc; mass_at_stiff = ms; }
   }
   EXPECT_LE(cc_max, cc_cap) << "CC not sigma-robust";
   // The whole point: at tiny Delta-t the mass block degrades, CC does not.
   EXPECT_GE(mass_at_stiff, mass_ratio * cc_at_stiff)
         << "expected the mass block to degrade at large sigma";
}

// T3 nu-sweep at tau ~ h (sigma = 12 ~ 1.5/h at h = 1/8): CC bounded over
// six decades of viscosity.
TEST(CcRobustness, NuSweepBounded)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const YAML::Node node = root["solver_iterations"]["cc_robustness"];
   ASSERT_TRUE(node);
   const int cap = node["nu_sweep_cc_cap"].as<int>();

   for (double nu : {1.0, 1e-2, 1e-4, 1e-6})
   {
      const int cc =
         SolveCount(8, nu, 12.0, 1e-8, SchurBlockType::CahouetChabard);
      if (Mpi::Root())
      {
         mfem::out << "[t3-nu] nu=" << nu << "  cc=" << cc << std::endl;
      }
      EXPECT_LE(cc, cap) << "CC not nu-robust at nu=" << nu;
   }
}

// T5: LaplacianLegacy vs ConsistentBMB under refinement at tight tolerance.
// MEASURED honestly: on the friendly ENCLOSED unit square the two are
// indistinguishable (legacy even marginally better) -- the L_p surrogate's
// structural error is its BOUNDARY treatment, so the discriminating domain is
// the one WITH an outflow boundary, where the gap exists and GROWS with
// refinement (measured +3/+5/+6 outer iterations at n = 8/16/32 while the
// consistent counts DROP). C&G's dramatic degradation lives at much finer
// meshes -- slow-tier territory; this fast guard pins the direction.
TEST(CcRobustness, LegacyModeDegrades)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE);
   const YAML::Node node = root["solver_iterations"]["cc_robustness"];
   ASSERT_TRUE(node);
   const int cons_cap = node["t5_consistent_cap"].as<int>();
   const double ratio = node["t5_legacy_min_ratio"].as<double>();

   const double nu = 1e-4, sigma = 1e5, rtol = 1e-10;
   int cons_max = 0, cons_fin = 0, leg_fin = 0;
   for (int n : {8, 16, 32})
   {
      const int cons = SolveCount(n, nu, sigma, rtol,
                                  SchurBlockType::CahouetChabard,
                                  SchurModel::ConsistentBMB, 3,
                                  /*with_outflow=*/true);
      const int leg = SolveCount(n, nu, sigma, rtol,
                                 SchurBlockType::CahouetChabard,
                                 SchurModel::LaplacianLegacy, 3,
                                 /*with_outflow=*/true);
      if (Mpi::Root())
      {
         mfem::out << "[t5] n=" << n << "  consistent=" << cons
                   << "  legacy=" << leg << std::endl;
      }
      cons_max = std::max(cons_max, cons);
      if (n == 32) { cons_fin = cons; leg_fin = leg; }
   }
   // Consistent stays FLAT under refinement; legacy exceeds it at the finest
   // mesh (mesh-growth is the C&G degradation signature).
   EXPECT_LE(cons_max, cons_cap);
   EXPECT_GE(leg_fin, ratio * cons_fin)
         << "legacy should degrade vs consistent on the finest mesh";
}
