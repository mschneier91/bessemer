// Restart of a deck's dynamic solid march (ElasticityCase, `checkpoint` and
// `restart`), at np 1/2/4: H8's cantilever to t = 0.12 with a checkpoint
// every 4 steps, restarted from step 4 to t = 0.2, against the uninterrupted
// run to t = 0.2:
//  R1 linear elasticity, generalized-alpha, BDF2 and BDF3;
//  R2 hyperelasticity, St. Venant-Kirchhoff (generalized-alpha, BDF2) and
//     neo-Hookean (generalized-alpha);
// the displacement and velocity bitwise on a host, the step count and time,
// the history CSV continued (its rows after the checkpoint are the
// uninterrupted run's), and the summary's iteration totals of the whole run.
// Runs on whatever device the test main configures (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "repro_tolerance.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

int Np()
{
   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   return np;
}

/// H8's cantilever, dynamic: @p equation with the solid keys @p scheme, to
/// @p t_final, output in @p out, and the top-level keys @p extra.
std::string Deck(const std::string& equation, const std::string& scheme,
                 double t_final, const std::string& out,
                 const std::string& extra)
{
   std::ostringstream d;
   d << "equation: " << equation << "\n"
     << "mesh:\n  dim: 2\n  elements: [6, 2]\n  lengths: [3.0, 0.5]\n"
     << "  periodic: [false, false]\n"
     << "solid:\n  youngs_modulus: 100.0\n  poisson_ratio: 0.3\n"
     << "  order: 3\n  rtol: 1.0e-13\n  analysis: dynamic\n  density: 2.0\n"
     << scheme << "  probe: [3.0, 0.25]\n"
     << "time:\n  dt: 0.02\n  t_final: " << t_final << "\n"
     << "boundary_conditions:\n"
     << "  - {select: [xmin], type: fixed}\n"
     << "  - {select: [xmax], type: traction, value: [0.0, -0.5]}\n"
     << "  - {select: [ymin, ymax], type: free}\n"
     << "output:\n  path: " << out << "\n  name: beam\n  history: true\n"
     << extra;
   return d.str();
}

/// What a run leaves.
struct Result
{
   Vector u, v;             ///< Displacement and velocity (true dofs).
   int steps = 0;           ///< Steps.
   double time = 0.0;       ///< Time.
   std::vector<std::string> rows; ///< The history CSV's lines (rank 0).
   std::string summary;     ///< The summary JSON (rank 0).
};

/// Run @p deck through ElasticityCase.
Result Run(const std::string& deck, const std::string& out)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   EXPECT_TRUE(c.Run());
   Result r;
   auto take = [&r](auto & s)
   {
      s.Displacement().GetTrueDofs(r.u);
      s.Velocity().GetTrueDofs(r.v);
      r.u.HostRead();
      r.v.HostRead();
      r.steps = s.StepCount();
      r.time = s.Time();
   };
   if (p.equation == incns::Equation::Hyperelasticity) { take(c.HyperelasticSolver()); }
   else { take(c.Solver()); }
   c.WriteSummary(out + "/beam_summary.json", "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::ifstream h(out + "/beam_history.csv");
      std::string line;
      while (std::getline(h, line)) { r.rows.push_back(line); }
      std::ifstream s(out + "/beam_summary.json");
      std::stringstream ss;
      ss << s.rdbuf();
      r.summary = ss.str();
   }
   return r;
}

double MaxDiff(const Vector& a, const Vector& b)
{
   a.HostRead(); // on a device the results are current there only
   b.HostRead();
   double m = a.Size() == b.Size() ? 0.0 : INFINITY;
   for (int i = 0; i < std::min(a.Size(), b.Size()); ++i)
   {
      m = std::max(m, std::abs(a(i) - b(i)));
   }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

/// @return The number after `"key": ` in @p json (-1 if absent).
double JsonNumber(const std::string& json, const std::string& key)
{
   const std::size_t i = json.find("\"" + key + "\": ");
   if (i == std::string::npos) { return -1.0; }
   return std::stod(json.substr(i + key.size() + 4));
}

/// The restart against the uninterrupted run, for one deck.
void RestartMatchesUninterrupted(const std::string& name,
                                 const std::string& equation,
                                 const std::string& scheme,
                                 const std::vector<std::string>& totals)
{
   const std::string base = "solid_restart_" + name + "_np" +
                            std::to_string(Np());
   const std::string chk = base + "_chk";
   // The uninterrupted run: 10 steps.
   const Result a = Run(Deck(equation, scheme, 0.2, base + "_a", ""), base + "_a");
   // To t = 0.12, a checkpoint every 4 steps: step 4's is kept.
   Run(Deck(equation, scheme, 0.12, base + "_b",
            "checkpoint:\n  enabled: true\n  interval: 4\n  path: " + chk + "\n"),
       base + "_b");
   // From step 4 to t = 0.2, into the same output.
   const Result c = Run(Deck(equation, scheme, 0.2, base + "_b",
                             "restart: " + chk + "\n"), base + "_b");
   EXPECT_EQ(a.steps, 10) << name;
   EXPECT_EQ(c.steps, a.steps) << name;
   EXPECT_EQ(c.time, a.time) << name;
   // Bitwise on a host; hypre's GPU AMG is not run-to-run reproducible.
   const double scale = std::max(1.0, a.u.Normlinf());
   EXPECT_LE(MaxDiff(a.u, c.u), incns_test::ReproTol(0.0, 1e-9) * scale) << name;
   EXPECT_LE(MaxDiff(a.v, c.v), incns_test::ReproTol(0.0, 1e-9) *
             std::max(1.0, a.v.Normlinf())) << name;
   if (Mpi::Root())
   {
      // a: the header, the initial state and 10 steps; b and c: the header,
      // the initial state, b's 6 steps, then c's steps 5 to 10.
      ASSERT_EQ(a.rows.size(), 12u) << name;
      ASSERT_EQ(c.rows.size(), 14u) << name;
      // Bitwise on a host: the rows print the same digits.
      const bool exact = !mfem::Device::Allows(mfem::Backend::CUDA_MASK |
                         mfem::Backend::HIP_MASK);
      for (int k = 5; k <= 10; ++k)
      {
         const std::string& ra = a.rows[static_cast<std::size_t>(k + 1)];
         const std::string& rc = c.rows[static_cast<std::size_t>(k + 3)];
         EXPECT_EQ(ra.substr(0, ra.find(',')), std::to_string(k)) << name;
         EXPECT_EQ(rc.substr(0, rc.find(',')), std::to_string(k)) << name;
         if (exact) { EXPECT_EQ(rc, ra) << name << " step " << k; }
      }
      for (const std::string& key : totals)
      {
         const double ta = JsonNumber(a.summary, key);
         EXPECT_GT(ta, 0.0) << name << " " << key;
         if (exact) { EXPECT_EQ(JsonNumber(c.summary, key), ta) << name << " " << key; }
      }
      EXPECT_EQ(JsonNumber(c.summary, "steps"), 10.0) << name;
      for (const std::string& d : {base + "_a", base + "_b", chk})
      {
         std::filesystem::remove_all(d);
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace

// R1
TEST(SolidRestart, R1_LinearElasticity)
{
   const std::vector<std::string> totals = {"cg_iterations_total"};
   RestartMatchesUninterrupted("le_alpha", "linear_elasticity",
                               "  rho_inf: 0.8\n", totals);
   RestartMatchesUninterrupted("le_bdf2", "linear_elasticity",
                               "  time_scheme: bdf\n  bdf_order: 2\n", totals);
   RestartMatchesUninterrupted("le_bdf3", "linear_elasticity",
                               "  time_scheme: bdf\n  bdf_order: 3\n", totals);
}

// R2
TEST(SolidRestart, R2_Hyperelasticity)
{
   const std::vector<std::string> totals = {"cg_iterations_total",
                                            "newton_iterations_total"
                                           };
   RestartMatchesUninterrupted("svk_alpha", "hyperelasticity",
                               "  rho_inf: 0.8\n", totals);
   RestartMatchesUninterrupted("svk_bdf2", "hyperelasticity",
                               "  time_scheme: bdf\n", totals);
   RestartMatchesUninterrupted("nh_alpha", "hyperelasticity",
                               "  model: neo_hookean\n  rho_inf: 0.8\n", totals);
}
