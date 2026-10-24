// Adaptive steps of a deck's dynamic solid march (ElasticityCase,
// time.step_control and time.on_solver_failure), at np 1/2/4, on H8's
// cantilever with its tip moved by A (1 - cos w t)^2 (smooth from rest):
//  A1 the local error estimates (StepError()) have their order: halving dt
//     divides the estimate at the same time by 2^p (p = 3 for
//     generalized-alpha and BDF2, 4 for BDF3), linear and hyperelastic;
//  A2 error control: tightening time.rtol 100-fold divides the error against
//     a fine fixed-step run several times (roughly 100^(2/3)), at more steps;
//  A3 cut_step: a hyperelastic march whose Newton fails at time.dt (a sudden
//     load, newton_max_iter 3) is taken again at smaller steps and reaches
//     t_final; under abort it stops at the failed step;
//  A4 an error-controlled run restarted from a checkpoint (checkpoint.at_time)
//     is the uninterrupted one, bitwise on a host.
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
#include <iostream>
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

/// H8's cantilever: @p equation and the solid keys @p solid, to @p t_final
/// at @p dt, the tip moved smoothly (or loaded suddenly with @p sudden), and
/// the time keys @p time and top-level keys @p extra.
std::string Deck(const std::string& equation, const std::string& solid,
                 double dt, double t_final, const std::string& time,
                 const std::string& extra, bool sudden = false)
{
   std::ostringstream d;
   d << "equation: " << equation << "\n"
     << "mesh:\n  dim: 2\n  elements: [6, 2]\n  lengths: [3.0, 0.5]\n"
     << "  periodic: [false, false]\n"
     << "solid:\n  youngs_modulus: 100.0\n  poisson_ratio: 0.3\n"
     << "  order: 3\n  rtol: 1.0e-13\n  analysis: dynamic\n  density: 2.0\n"
     << solid << "  probe: [3.0, 0.25]\n"
     << "time:\n  dt: " << dt << "\n  t_final: " << t_final << "\n" << time
     << "boundary_conditions:\n"
     << "  - {select: [xmin], type: fixed}\n";
   if (sudden)
   {
      d << "  - {select: [xmax], type: traction, value: [0.0, -0.5]}\n";
   }
   else
   {
      d << "  - select: [xmax]\n    type: displacement\n"
        << "    expression: [\"0\", \"A * (1 - cos(w * t))^2\"]\n"
        << "    parameters: {A: 0.05, w: 2.0}\n";
   }
   d << "  - {select: [ymin, ymax], type: free}\n" << extra;
   return d.str();
}

/// What a run leaves.
struct Result
{
   bool ok = false;     ///< Run()'s verdict.
   Vector u;            ///< Displacement (true dofs, host).
   int steps = 0;       ///< Steps.
   double time = 0.0;   ///< Time.
   double error = -1.0; ///< The last step's StepError().
   std::string summary; ///< The summary JSON (rank 0).
};

Result RunDeck(const std::string& deck,
               const std::string& summary_path = "")
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   Result r;
   r.ok = c.Run();
   auto take = [&r](auto & s)
   {
      s.Displacement().GetTrueDofs(r.u);
      r.u.HostRead();
      r.steps = s.StepCount();
      r.time = s.Time();
      r.error = s.StepError();
   };
   if (p.equation == incns::Equation::Hyperelasticity) { take(c.HyperelasticSolver()); }
   else { take(c.Solver()); }
   if (!summary_path.empty())
   {
      c.WriteSummary(summary_path, r.ok ? "ok" : "diverged");
      MPI_Barrier(MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         std::ifstream f(summary_path);
         std::stringstream ss;
         ss << f.rdbuf();
         r.summary = ss.str();
         std::filesystem::remove(summary_path);
      }
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

double MaxAbs(const Vector& a)
{
   a.HostRead();
   double m = a.Normlinf();
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

struct Scheme
{
   const char* name;
   const char* equation;
   const char* keys; ///< Solid keys.
   double order;     ///< StepError()'s order in dt.
};

const Scheme kSchemes[] =
{
   {"linear, generalized-alpha", "linear_elasticity", "  rho_inf: 0.8\n", 3.0},
   {"linear, BDF2", "linear_elasticity", "  time_scheme: bdf\n", 3.0},
   {"linear, BDF3", "linear_elasticity", "  time_scheme: bdf\n  bdf_order: 3\n", 4.0},
   {"St. Venant-Kirchhoff, generalized-alpha", "hyperelasticity", "  rho_inf: 0.8\n", 3.0},
   {"St. Venant-Kirchhoff, BDF2", "hyperelasticity", "  time_scheme: bdf\n", 3.0},
};

} // namespace

/// The RMS of the solver's StepError() over its steps ending in (t0, t1],
/// marching it directly at a constant @p dt from rest; @p last its values at
/// the window's last three steps.
template <class Solver>
double WindowError(Solver& s, double dt, double t0, double t1,
                   std::vector<double>& last, Vector& u)
{
   s.Initialize(0.0);
   const long long n = std::lround(t1 / dt);
   double sum = 0.0;
   int m = 0;
   last.clear();
   for (long long k = 1; k <= n; ++k)
   {
      EXPECT_TRUE(s.Step(dt)) << "step " << k;
      if (static_cast<double>(k) * dt > t0 + 1e-9 && s.StepError() >= 0.0)
      {
         sum += s.StepError() * s.StepError();
         ++m;
         last.push_back(s.StepError());
      }
   }
   if (last.size() > 3) { last.erase(last.begin(), last.end() - 3); }
   s.Displacement().GetTrueDofs(u);
   return m ? std::sqrt(sum / m) : -1.0;
}

// A1
TEST(SolidAdaptive, A1_EstimatesHaveTheirOrder)
{
   for (const Scheme& s : kSchemes)
   {
      double e[2];
      std::vector<double> last[2];
      for (int i = 0; i < 2; ++i)
      {
         const double dt = i == 0 ? 0.02 : 0.01;
         const incns::Parameters p =
            incns::Parameters::LoadYAMLString(Deck(s.equation, s.keys, dt, 0.4,
                                              "", ""));
         std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
         incns::ElasticityCase c(*mesh, p);
         Vector u;
         e[i] = p.equation == incns::Equation::Hyperelasticity ?
                WindowError(c.HyperelasticSolver(), dt, 0.3, 0.4, last[i], u) :
                WindowError(c.Solver(), dt, 0.3, 0.4, last[i], u);
         ASSERT_GT(e[i], 0.0) << s.name;
         // The deck's march (Run()) takes the same steps: the same solution
         // and the same last estimate.
         const std::string path = "solid_adaptive_a1_np" + std::to_string(Np()) +
                                  ".json";
         const Result r = RunDeck(Deck(s.equation, s.keys, dt, 0.4, "", ""), path);
         ASSERT_TRUE(r.ok) << s.name;
         const double du = MaxDiff(u, r.u);
         if (Mpi::Root())
         {
            std::cout << "[A1] " << s.name << ", dt " << dt << ": Run() against "
                      "stepping: |du| " << du << ", last estimate " << r.error
                      << " against " << last[i].back() << ", "
                      << JsonNumber(r.summary, "rejected_steps") << " rejected"
                      << std::endl;
         }
         // To roundoff: the march's clock k dt against a sum of dt.
         EXPECT_LE(du, 1e-12 * std::max(1.0, MaxAbs(u))) << s.name;
         if (Mpi::Root())
         {
            EXPECT_EQ(JsonNumber(r.summary, "rejected_steps"), 0.0) << s.name;
         }
      }
      const double ratio = e[0] / e[1], expect = std::pow(2.0, s.order);
      if (Mpi::Root())
      {
         std::cout << "[A1] " << s.name << ": RMS estimate over t in (0.3, 0.4] "
                   << e[0] << " (dt 0.02) / " << e[1] << " (dt 0.01) = " << ratio
                   << " (2^" << s.order << " = " << expect << "); last steps";
         for (int i = 0; i < 2; ++i)
         {
            std::cout << (i ? " | dt 0.01:" : " dt 0.02:");
            for (double v : last[i]) { std::cout << " " << v; }
         }
         std::cout << std::endl;
      }
      EXPECT_GT(ratio, 0.7 * expect) << s.name;
      EXPECT_LT(ratio, 1.4 * expect) << s.name;
   }
}

// A2
TEST(SolidAdaptive, A2_ErrorControlConverges)
{
   for (const Scheme& s : kSchemes)
   {
      if (s.order != 3.0) { continue; } // BDF3 is not A-stable: no long runs
      const double tf = 1.0;
      const Result ref = RunDeck(Deck(s.equation, s.keys, 5e-4, tf, "", ""));
      ASSERT_TRUE(ref.ok) << s.name;
      const double scale = MaxAbs(ref.u);
      double err[2];
      int steps[2];
      const double rtols[2] = {1e-4, 1e-6};
      for (int i = 0; i < 2; ++i)
      {
         std::ostringstream t;
         t << "  step_control: error\n  rtol: " << rtols[i] << "\n  atol: 1.0e-14\n";
         const std::string path = "solid_adaptive_a2_np" + std::to_string(Np()) +
                                  ".json";
         const Result r = RunDeck(Deck(s.equation, s.keys, 0.02, tf, t.str(), ""), path);
         ASSERT_TRUE(r.ok) << s.name;
         EXPECT_NEAR(r.time, tf, 1e-12) << s.name;
         err[i] = MaxDiff(r.u, ref.u) / scale;
         steps[i] = r.steps;
         if (Mpi::Root())
         {
            EXPECT_NE(r.summary.find("\"step_control\": \"error\""),
                      std::string::npos) << s.name;
            std::cout << "[A2] " << s.name << ", rtol " << rtols[i] << ": "
                      << r.steps << " steps, " << JsonNumber(r.summary, "rejected_steps")
                      << " rejected, error " << err[i] << " of max |u|" << std::endl;
         }
      }
      EXPECT_GT(steps[1], steps[0]) << s.name;
      EXPECT_GT(err[0] / err[1], 5.0) << s.name;
      EXPECT_LT(err[1], 1e-4) << s.name;
   }
}

// A3
TEST(SolidAdaptive, A3_CutStepRetriesAFailedNewton)
{
   // The sudden load's most Newton iterations in a step at time.dt, N: with
   // newton_max_iter N - 1 those steps fail at time.dt.
   const std::string path =
      "solid_adaptive_a3_np" + std::to_string(Np()) + ".json";
   const Result full = RunDeck(Deck("hyperelasticity", "  rho_inf: 1.0\n", 0.02,
                                    0.25, "", "", true), path);
   ASSERT_TRUE(full.ok);
   int most = 0;
   if (Mpi::Root()) { most = static_cast<int>(JsonNumber(full.summary, "newton_iterations_max")); }
   MPI_Bcast(&most, 1, MPI_INT, 0, MPI_COMM_WORLD);
   ASSERT_GE(most, 3);
   const std::string keys = "  rho_inf: 1.0\n  newton_max_iter: " +
                            std::to_string(most - 1) + "\n";
   const Result r = RunDeck(Deck("hyperelasticity", keys, 0.02, 0.25, "", "",
                                 true), path);
   ASSERT_TRUE(r.ok);
   EXPECT_NEAR(r.time, 0.25, 1e-12);
   const double d = MaxDiff(r.u, full.u) / MaxAbs(full.u);
   if (Mpi::Root())
   {
      const double rejected = JsonNumber(r.summary, "rejected_steps");
      EXPECT_GT(rejected, 0.0);
      std::cout << "[A3] newton_max_iter " << most - 1 << " (dt 0.02 needs " << most
                << "): " << r.steps << " steps, " << rejected << " rejected; against "
                << "dt 0.02 throughout " << d << std::endl;
   }
   EXPECT_LT(d, 0.05);
   // Under abort the march stops at the failed step.
   const Result a = RunDeck(Deck("hyperelasticity", keys, 0.02, 0.25,
                                 "  on_solver_failure: abort\n", "", true));
   EXPECT_FALSE(a.ok);
   EXPECT_LT(a.time, 0.25);
}

// A4
TEST(SolidAdaptive, A4_ErrorControlledRestart)
{
   for (const Scheme& s : kSchemes)
   {
      if (s.order != 3.0) { continue; }
      const std::string chk = "solid_adaptive_chk_np" + std::to_string(Np());
      const std::string t =
         "  step_control: error\n  rtol: 1.0e-5\n  atol: 1.0e-14\n";
      const Result a = RunDeck(Deck(s.equation, s.keys, 0.02, 1.0, t, ""));
      // The checkpoint once t >= 0.3; the run stops at 0.6.
      RunDeck(Deck(s.equation, s.keys, 0.02, 0.6, t,
                   "checkpoint:\n  at_time: 0.3\n  path: " + chk + "\n"));
      const Result c = RunDeck(Deck(s.equation, s.keys, 0.02, 1.0, t,
                                    "restart: " + chk + "\n"));
      ASSERT_TRUE(a.ok && c.ok) << s.name;
      EXPECT_EQ(c.steps, a.steps) << s.name;
      EXPECT_EQ(c.time, a.time) << s.name;
      EXPECT_LE(MaxDiff(a.u, c.u), incns_test::ReproTol(0.0, 1e-9) *
                std::max(1.0, MaxAbs(a.u))) << s.name;
      MPI_Barrier(MPI_COMM_WORLD);
      if (Mpi::Root()) { std::filesystem::remove_all(chk); }
   }
}
