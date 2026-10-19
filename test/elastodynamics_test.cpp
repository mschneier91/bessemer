// Elastodynamics (src/solid, LinearElasticity::Initialize/Step): the time
// schemes on problems with exact answers.
//  D1 generalized-alpha parameters (Chung & Hulbert 1993) in closed form;
//  D2 temporal order on a separable motion u = s(x) T(t) with s in the space,
//     so every error is the integrator's: generalized-alpha (rho_inf 1, 0.5,
//     0) and BDF2 order 2, BDF3 order 3, displacement and velocity, 2D/3D;
//  D3 energy in free vibration: conserved by generalized-alpha at rho_inf = 1
//     (the trapezoidal rule), dissipated by rho_inf < 1 and by BDF2;
//  D4 the fixed-fixed bar's first longitudinal mode (nu = 0: an exact
//     standing wave, omega = pi sqrt(E / rho)) after 1.25 periods, second
//     order in dt;
//  D5 a dynamic deck through ElasticityCase: a load applied suddenly at
//     t = 0, the energy balance E(t) = f . u(t) of the trapezoidal rule, the
//     last step landing on t_final, the summary and the history file.

#include <gtest/gtest.h>

#include "elasticity_test_util.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/linear_elasticity.hpp"
#include "mfem.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace elasticity_test;

namespace
{

incns::LinearElasticity::Options Opts(int order, double lambda, double mu,
                                      incns::SolidTimeScheme scheme,
                                      int bdf_order, double rho_inf)
{
   incns::LinearElasticity::Options o;
   o.order = order;
   o.lambda = lambda;
   o.mu = mu;
   o.density = 1.3;
   o.rtol = 1e-13;
   o.scheme = scheme;
   o.bdf_order = bdf_order;
   o.rho_inf = rho_inf;
   return o;
}

std::unique_ptr<ParMesh> Box(int dim, int n)
{
   Mesh m = dim == 2 ?
            Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0) :
            Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, m);
}

/// L2 norm of a grid function minus a coefficient at time t.
double L2Error(ParGridFunction& g, VectorCoefficient& exact, double t)
{
   exact.SetTime(t);
   return g.ComputeL2Error(exact);
}

/// Errors at t = 1 in u and v of the separable motion with n steps.
std::pair<double, double> MotionErrors(int dim,
                                       const incns::LinearElasticity::Options& o,
                                       int n)
{
   auto mesh = Box(dim, 2);
   SeparableMotion ex(dim, o.lambda, o.mu, o.density);
   incns::LinearElasticity s(*mesh, o);
   const std::vector<int> held =
   AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; });
   EXPECT_EQ(held.size(), 1u);
   for (int a : held) { s.AddFixed(a); }
   for (int a : OtherAttributes(*mesh, held)) { s.AddTraction(a, *ex.traction); }
   s.SetBodyForce(*ex.f);
   s.Initialize(0.0, ex.u.get(), ex.v.get());
   const double dt = 1.0 / n;
   for (int k = 0; k < n; ++k) { EXPECT_TRUE(s.Step(dt)); }
   EXPECT_NEAR(s.Time(), 1.0, 1e-12);
   return {L2Error(s.Displacement(), *ex.u, s.Time()),
           L2Error(s.Velocity(), *ex.v, s.Time())};
}

/// The observed orders of u and v over steps n, 2n, 4n (the last pair).
void ExpectOrder(int dim, const incns::LinearElasticity::Options& o,
                 double order, const std::vector<int>& steps, const char* name)
{
   std::vector<std::pair<double, double>> e;
   for (int n : steps) { e.push_back(MotionErrors(dim, o, n)); }
   const std::size_t L = e.size() - 1;
   const double ru = std::log2(e[L - 1].first / e[L].first);
   const double rv = std::log2(e[L - 1].second / e[L].second);
   if (Mpi::Root())
   {
      std::cout << "[ info     ] D2 " << name << " dim " << dim << ": u errors";
      for (const auto& p : e) { std::cout << " " << p.first; }
      std::cout << ", order u " << ru << ", v " << rv << std::endl;
   }
   EXPECT_GT(ru, order - 0.15) << name;
   EXPECT_GT(rv, order - 0.15) << name;
   EXPECT_LT(ru, order + 0.5) << name;
}

} // namespace

TEST(Elastodynamics, D1_GeneralizedAlphaParameters)
{
   const incns::GeneralizedAlphaCoefficients one = incns::GeneralizedAlpha(1.0);
   EXPECT_DOUBLE_EQ(one.alpha_m, 0.5);
   EXPECT_DOUBLE_EQ(one.alpha_f, 0.5);
   EXPECT_DOUBLE_EQ(one.gamma, 0.5);
   EXPECT_DOUBLE_EQ(one.beta, 0.25);
   const incns::GeneralizedAlphaCoefficients zero = incns::GeneralizedAlpha(0.0);
   EXPECT_DOUBLE_EQ(zero.alpha_m, -1.0);
   EXPECT_DOUBLE_EQ(zero.alpha_f, 0.0);
   EXPECT_DOUBLE_EQ(zero.gamma, 1.5);
   EXPECT_DOUBLE_EQ(zero.beta, 1.0);
   const incns::GeneralizedAlphaCoefficients h = incns::GeneralizedAlpha(0.5);
   EXPECT_NEAR(h.alpha_m, 0.0, 1e-15);
   EXPECT_NEAR(h.alpha_f, 1.0 / 3.0, 1e-15);
   EXPECT_NEAR(h.gamma, 0.5 - h.alpha_m + h.alpha_f, 1e-15);
   EXPECT_NEAR(h.beta, 0.25 * std::pow(1.0 - h.alpha_m + h.alpha_f, 2), 1e-15);
}

TEST(Elastodynamics, D2_TemporalOrders)
{
   using incns::SolidTimeScheme;
   for (int dim : {2, 3})
   {
      const std::vector<int> steps = {20, 40, 80};
      for (double r : {1.0, 0.5, 0.0})
      {
         std::ostringstream name;
         name << "generalized_alpha rho_inf " << r;
         ExpectOrder(dim, Opts(2, 1.2, 0.8, SolidTimeScheme::GeneralizedAlpha, 2, r),
                     2.0, steps, name.str().c_str());
      }
      ExpectOrder(dim, Opts(2, 1.2, 0.8, SolidTimeScheme::Bdf, 2, 0.0), 2.0, steps,
                  "bdf2");
      ExpectOrder(dim, Opts(2, 1.2, 0.8, SolidTimeScheme::Bdf, 3, 0.0), 3.0, steps,
                  "bdf3");
   }
}

TEST(Elastodynamics, D3_EnergyInFreeVibration)
{
   using incns::SolidTimeScheme;
   auto mesh = Box(2, 3);
   VectorFunctionCoefficient u0(2, [](const Vector & x, Vector & u)
   {
      u.SetSize(2);
      u(0) = 0.1 * std::sin(M_PI * x(0)) * x(1);
      u(1) = 0.05 * std::sin(2 * M_PI * x(0)) * (1.0 + x(1) * x(1));
   });
   struct Case
   {
      SolidTimeScheme scheme;
      int order;
      double rho_inf;
      bool conserves;
   };
   for (const Case& c :
        {
           Case{SolidTimeScheme::GeneralizedAlpha, 2, 1.0, true},
           Case{SolidTimeScheme::GeneralizedAlpha, 2, 0.5, false},
           Case{SolidTimeScheme::Bdf, 2, 0.0, false}
        })
   {
      incns::LinearElasticity s(*mesh, Opts(3, 1.0, 1.0, c.scheme, c.order,
                                            c.rho_inf));
      // Held on x = 0 and x = 1, free on the sides.
      for (int a : AttributesWhere(*mesh, [](const Vector & p)
   {
      return p(0) < 1e-12 || p(0) > 1.0 - 1e-12;
      }))
      {
         s.AddFixed(a);
      }
      s.Initialize(0.0, &u0, nullptr);
      const double e0 = s.KineticEnergy() + s.StrainEnergy();
      ASSERT_GT(e0, 0.0);
      double e = e0, emax = 0.0;
      for (int k = 0; k < 60; ++k)
      {
         ASSERT_TRUE(s.Step(0.05));
         e = s.KineticEnergy() + s.StrainEnergy();
         emax = std::max(emax, std::abs(e - e0) / e0);
      }
      if (Mpi::Root())
      {
         std::cout << "[ info     ] D3 scheme " << static_cast<int>(c.scheme)
                   << " rho_inf " << c.rho_inf << ": E/E0 at the end " << e / e0
                   << ", max |E - E0| / E0 " << emax << std::endl;
      }
      if (c.conserves) { EXPECT_LT(emax, 1e-10); }
      else
      {
         EXPECT_LT(e, (1.0 - 1e-3) * e0);
         EXPECT_GT(e, 0.0);
      }
   }
}

TEST(Elastodynamics, D4_FixedFixedBarMode)
{
   using incns::SolidTimeScheme;
   // nu = 0 (lambda = 0): u = (sin(pi x) cos(omega t), 0) is exact with
   // traction-free sides, omega = pi sqrt((lambda + 2 mu) / rho).
   Mesh serial = Mesh::MakeCartesian2D(6, 2, Element::QUADRILATERAL, true, 1.0,
                                       0.25);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   const double mu = 0.5, rho = 1.3;
   const double omega = M_PI * std::sqrt(2.0 * mu / rho);
   const double period = 2.0 * M_PI / omega;
   VectorFunctionCoefficient exact(2, [omega](const Vector & x, double t,
                                   Vector & u)
   {
      u.SetSize(2);
      u(0) = std::sin(M_PI * x(0)) * std::cos(omega * t);
      u(1) = 0.0;
   });
   for (SolidTimeScheme scheme :
        {
           SolidTimeScheme::GeneralizedAlpha,
           SolidTimeScheme::Bdf
        })
   {
      std::vector<double> err;
      for (int n : {50, 100})
      {
         incns::LinearElasticity::Options o = Opts(5, 0.0, mu, scheme, 2, 1.0);
         o.density = rho;
         incns::LinearElasticity s(mesh, o);
         for (int a : AttributesWhere(mesh, [](const Vector & p)
      {
         return p(0) < 1e-12 || p(0) > 1.0 - 1e-12;
         }))
         {
            s.AddFixed(a);
         }
         exact.SetTime(0.0);
         s.Initialize(0.0, &exact, nullptr);
         // 1.25 periods: there cos(omega t) crosses zero, so a phase error
         // shows at first order (after a whole period only at second).
         for (int k = 0; k < 5 * n / 4; ++k) { ASSERT_TRUE(s.Step(period / n)); }
         // Relative to ||sin(pi x)||_L2 = sqrt(0.25 / 2).
         err.push_back(L2Error(s.Displacement(), exact, s.Time()) /
                       std::sqrt(0.125));
      }
      if (Mpi::Root())
      {
         std::cout << "[ info     ] D4 scheme " << static_cast<int>(scheme)
                   << ": relative error after 1.25 periods " << err[0] << ", "
                   << err[1] << " (order " << std::log2(err[0] / err[1]) << ")"
                   << std::endl;
      }
      EXPECT_GT(std::log2(err[0] / err[1]), 1.85);
      if (scheme == SolidTimeScheme::GeneralizedAlpha)
      {
         // rho_inf = 1 is the trapezoidal rule: a mode turns by 2 atan(omega
         // dt / 2) per step, so at t = 1.25 T, where cos(omega t) = 0, the
         // error is |cos(omega_h t)| = |sin((omega - omega_h) t)|.
         const double dt = period / 100, t = 1.25 * period;
         const double omega_h = 2.0 / dt * std::atan(0.5 * omega * dt);
         EXPECT_NEAR(err[1], std::abs(std::sin((omega - omega_h) * t)),
                     0.02 * err[1]);
      }
      EXPECT_LT(err[1], scheme == SolidTimeScheme::GeneralizedAlpha ? 5e-3 : 2e-2);
   }
}

TEST(Elastodynamics, D5_DynamicDeckThroughElasticityCase)
{
   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   const std::string dir = "elastodynamics_test_out_np" + std::to_string(np);
   std::ostringstream deck;
   deck << "equation: linear_elasticity\n"
        << "mesh:\n  dim: 2\n  elements: [6, 2]\n  lengths: [3.0, 0.5]\n"
        << "  periodic: [false, false]\n"
        << "solid:\n  youngs_modulus: 100.0\n  poisson_ratio: 0.3\n"
        << "  plane: stress\n  order: 3\n  rtol: 1.0e-13\n"
        << "  analysis: dynamic\n  density: 2.0\n  rho_inf: 1.0\n"
        << "  probe: [3.0, 0.25]\n"
        << "time:\n  dt: 0.02\n  t_final: 0.25\n"
        << "boundary_conditions:\n"
        << "  - {select: [xmin], type: fixed}\n"
        << "  - {select: [xmax], type: traction, value: [0.0, -0.5]}\n"
        << "  - {select: [ymin, ymax], type: free}\n"
        << "output:\n  path: " << dir << "\n  name: beam\n  history: true\n";
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck.str());
   ASSERT_EQ(p.solid.analysis, incns::SolidAnalysis::Dynamic);
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   incns::LinearElasticity& s = c.Solver();
   // 0.25 / 0.02: 12 steps of 0.02 and one of 0.01.
   EXPECT_EQ(s.StepCount(), 13);
   EXPECT_NEAR(s.Time(), 0.25, 1e-14);
   // The trapezoidal rule's energy balance under a constant load from rest:
   // E(t) = f . u(t), exactly (to the solver tolerance).
   const double e = s.KineticEnergy() + s.StrainEnergy();
   const double w = s.ExternalWork();
   EXPECT_GT(w, 0.0);
   EXPECT_NEAR(e, w, 1e-9 * w);
   Vector x(2), u;
   x(0) = 3.0;
   x(1) = 0.25;
   ASSERT_TRUE(s.DisplacementAt(x, u));
   EXPECT_LT(u(1), 0.0); // the tip moves with the load

   const std::string summary = dir + "/beam_summary.json";
   c.WriteSummary(summary, "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::ifstream f(summary);
      std::stringstream ss;
      ss << f.rdbuf();
      const std::string j = ss.str();
      for (const char* key :
           {"\"dynamics\"", "\"generalized_alpha\"",
            "\"total_energy\"", "\"steps\": 13"
           })
      {
         EXPECT_NE(j.find(key), std::string::npos) << key << " in\n" << j;
      }
      std::ifstream h(dir + "/beam_history.csv");
      std::string line;
      int rows = 0;
      std::getline(h, line);
      EXPECT_EQ(line.rfind("step,t,dt,cg_iterations,kinetic_energy", 0), 0u);
      while (std::getline(h, line)) { ++rows; }
      EXPECT_EQ(rows, 14); // the initial state and 13 steps
      std::filesystem::remove_all(dir);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}
