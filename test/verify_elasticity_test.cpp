// Verification tier (label `verify`): linear elasticity (src/solid) against
// exact solutions and a benchmark from the literature.
//  V1 2D manufactured solution u_i = sin(pi x) sin(pi y) (div u != 0), unit
//     square, held on x = 0, the exact traction on the other faces;
//  V2 the same in 3D;
//  V3 Lame's thick-walled cylinder under internal pressure (Timoshenko &
//     Goodier 1970, section 28; plane strain): a quarter annulus on a curved,
//     isoparametric mesh, exact displacement on the symmetry edges, the
//     pressure inside, traction-free outside;
//  V4 Cook's membrane (Cook 1974; E = 1, nu = 1/3, total shear load 1): the
//     vertical displacement at the loaded edge's midpoint against the
//     literature's values in plane stress and in plane strain;
//  V5 elastodynamics: the temporal rate of the deck's default time scheme on
//     a separable motion u = s(x) T(t) with s in the space;
//  V6 a clamped-free beam's first natural frequency against Euler-Bernoulli
//     beam theory, started from its first mode shape.
// Each study runs at the orders listed in baselines.yaml `verification.
// elasticity` and at the deck default (solid.order unset: whatever the
// default is when the suite runs). The material and solver settings come from
// a minimal deck, so their defaults are what is verified.
// Theory: ||u - u_h||_L2 at k + 1, |u - u_h|_H1 at k. Pass: the rate fitted
// over all levels >= theory - rate_margin.

#include <gtest/gtest.h>

#include "elasticity_test_util.hpp"
#include "verify_util.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/linear_elasticity.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace elasticity_test;

namespace
{

YAML::Node Settings() { return verify::Baselines()["elasticity"]; }

/// A minimal solid deck: dimension, material, order (k <= 0: the default),
/// plane model, solver tolerance.
incns::Parameters Deck(int dim, int k, double E, double nu, const char* plane)
{
   std::ostringstream d;
   d << std::setprecision(17) << "equation: linear_elasticity\n"
     << "mesh:\n  dim: " << dim << "\n"
     << "solid:\n  youngs_modulus: " << E << "\n  poisson_ratio: " << nu
     << "\n  plane: " << plane << "\n  rtol: "
     << Settings()["rtol"].as<double>() << "\n";
   if (k > 0) { d << "  order: " << k << "\n"; }
   return incns::Parameters::LoadYAMLString(d.str());
}

/// Deck() with analysis dynamic and a density.
incns::Parameters DynamicDeck(int dim, double E, double nu, const char* plane,
                              double rho)
{
   std::ostringstream d;
   d << std::setprecision(17) << "equation: linear_elasticity\n"
     << "mesh:\n  dim: " << dim << "\n"
     << "solid:\n  youngs_modulus: " << E << "\n  poisson_ratio: " << nu
     << "\n  plane: " << plane << "\n  rtol: "
     << Settings()["rtol"].as<double>() << "\n  analysis: dynamic\n"
     << "  density: " << rho << "\n";
   return incns::Parameters::LoadYAMLString(d.str());
}

/// @return The order a time scheme of @p o converges at.
double SchemeOrder(const incns::LinearElasticity::Options& o)
{
   return o.scheme == incns::SolidTimeScheme::Bdf ? o.bdf_order : 2.0;
}

/// @return A short name of @p o's time scheme.
std::string SchemeName(const incns::LinearElasticity::Options& o)
{
   if (o.scheme == incns::SolidTimeScheme::Bdf)
   {
      return "bdf" + std::to_string(o.bdf_order);
   }
   std::ostringstream s;
   s << "generalized_alpha_rho_inf" << o.rho_inf;
   return s.str();
}

std::unique_ptr<ParMesh> Partition(Mesh& serial)
{
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

/// Solve one level and return its errors; @p order receives the order used.
Errors Level(ParMesh& mesh, const incns::Parameters& p,
             const std::function<std::unique_ptr<Exact>(double, double)>& make,
             const std::function<bool(const Vector&)>& held, int& order)
{
   const incns::LinearElasticity::Options o = incns::ElasticityOptions(p);
   order = o.order;
   incns::LinearElasticity s(mesh, o);
   auto ex = make(o.lambda, o.mu);
   const std::vector<int> h = AttributesWhere(mesh, held);
   EXPECT_FALSE(h.empty());
   for (int a : h) { s.AddDisplacement(a, *ex->u); }
   for (int a : OtherAttributes(mesh, h)) { s.AddTraction(a, *ex->traction); }
   if (ex->f) { s.SetBodyForce(*ex->f); }
   EXPECT_TRUE(s.Solve());
   return ComputeErrors(s.Displacement(), *ex, s.Rules(), 2 * order + 4);
}

/// A rate study over @p levels at order @p k (k <= 0: the default).
void Study(verify::Record& rec, const std::string& label, int dim, int k,
           const std::vector<int>& levels, const char* plane,
           const std::function<std::unique_ptr<Mesh>(int n, int order)>& mesh_of,
           const std::function<std::unique_ptr<Exact>(double, double)>& make,
           const std::function<bool(const Vector&)>& held)
{
   const YAML::Node b = verify::Baselines();
   const double margin = b["rate_margin"].as<double>();
   const double floor = b["noise_floor"].as<double>();
   const incns::Parameters p = Deck(dim, k,
                                    Settings()["youngs_modulus"].as<double>(),
                                    Settings()["poisson_ratio"].as<double>(), plane);
   const int order = p.solid.Order(p.order_u);
   const std::string variant = (k > 0 ? "Q" : "default_Q") + std::to_string(order);
   std::vector<double> hs, l2, h1;
   for (std::size_t i = 0; i < levels.size(); ++i)
   {
      const int n = levels[i];
      std::unique_ptr<Mesh> serial = mesh_of(n, order);
      auto mesh = Partition(*serial);
      int used = 0;
      const Errors e = Level(*mesh, p, make, held, used);
      const double h = 1.0 / n;
      std::ostringstream s;
      s << "Q" << used << " E=" << p.solid.youngs_modulus << " nu="
        << p.solid.poisson_ratio;
      rec.Error(variant, static_cast<int>(i), h, "u_l2", e.l2, s.str());
      rec.Error(variant, static_cast<int>(i), h, "u_h1", e.h1, s.str());
      hs.push_back(h);
      l2.push_back(e.l2);
      h1.push_back(e.h1);
   }
   rec.Rate(variant, "u_l2", hs, l2, order + 1.0, margin, floor);
   rec.Rate(variant, "u_h1", hs, h1, order, margin, floor);
   (void)label;
}

std::vector<int> Ints(const YAML::Node& n) { return n.as<std::vector<int>>(); }

std::unique_ptr<Mesh> Box(int dim, int n)
{
   return std::make_unique<Mesh>(dim == 2 ?
                                 Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0) :
                                 Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0));
}

void TrigStudy(int dim)
{
   const YAML::Node s = Settings()[dim == 2 ? "mms_2d" : "mms_3d"];
   verify::Record rec(dim == 2 ? "elasticity_mms_2d" : "elasticity_mms_3d");
   auto mesh_of = [dim](int n, int) { return Box(dim, n); };
   auto make = [dim](double lam, double mu) { return Trigonometric(dim, lam, mu); };
   auto held = [](const Vector & c) { return c(0) < 1e-12; };
   for (int k : Ints(s["orders"]))
   {
      Study(rec, "mms", dim, k, Ints(s["levels"]), "strain", mesh_of, make, held);
   }
   Study(rec, "mms", dim, 0, Ints(s["default_levels"]), "strain", mesh_of, make,
         held);
}

} // namespace

TEST(VerifyElasticity, V1_ManufacturedSolution2D) { TrigStudy(2); }

TEST(VerifyElasticity, V2_ManufacturedSolution3D) { TrigStudy(3); }

TEST(VerifyElasticity, V3_LameCylinderOnACurvedMesh)
{
   const YAML::Node s = Settings()["lame_cylinder"];
   const double a = s["inner_radius"].as<double>();
   const double b = s["outer_radius"].as<double>();
   const double pr = s["pressure"].as<double>();
   const double E = Settings()["youngs_modulus"].as<double>();
   const double nu = Settings()["poisson_ratio"].as<double>();
   verify::Record rec("elasticity_lame_cylinder");
   auto mesh_of = [a, b](int n, int order)
   {
      return std::make_unique<Mesh>(QuarterAnnulus(n, a, b, order));
   };
   auto make = [E, nu, pr, a, b](double, double)
   {
      return LameCylinder(E, nu, pr, a, b);
   };
   // Exact displacement on the two symmetry edges (x = 0, y = 0).
   auto held = [](const Vector & c) { return std::abs(c(0)) < 1e-9 || std::abs(c(1)) < 1e-9; };
   for (int k : Ints(s["orders"]))
   {
      Study(rec, "lame", 2, k, Ints(s["levels"]), "strain", mesh_of, make, held);
   }
   Study(rec, "lame", 2, 0, Ints(s["default_levels"]), "strain", mesh_of, make,
         held);
}

TEST(VerifyElasticity, V4_CookMembrane)
{
   const YAML::Node s = Settings()["cook_membrane"];
   const double tol = s["tolerance"].as<double>();
   const std::vector<double> point = s["point"].as<std::vector<double>>();
   verify::Record rec("elasticity_cook_membrane");
   const std::vector<int> levels = Ints(s["levels"]);
   for (const char* plane : {"stress", "strain"})
   {
      const double reference = s[std::string("reference_uy_plane_") + plane]
                               .as<double>();
      // E = 1, nu = 1/3, the deck's default order and solver.
      const incns::Parameters p = Deck(2, 0, 1.0, 1.0 / 3.0, plane);
      const incns::LinearElasticity::Options o = incns::ElasticityOptions(p);
      std::vector<double> uy;
      for (std::size_t i = 0; i < levels.size(); ++i)
      {
         Mesh serial = CookMembrane(levels[i]);
         auto mesh = Partition(serial);
         incns::LinearElasticity solver(*mesh, o);
         const std::vector<int> clamped =
         AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-9; });
         const std::vector<int> loaded =
         AttributesWhere(*mesh, [](const Vector & c) { return c(0) > 48.0 - 1e-9; });
         ASSERT_EQ(clamped.size(), 1u);
         ASSERT_EQ(loaded.size(), 1u);
         Vector t(2);
         t(0) = 0.0;
         t(1) = 1.0 / 16.0; // total 1 over the 16-long edge
         VectorConstantCoefficient tc(t);
         for (int a : clamped) { solver.AddFixed(a); }
         solver.AddTraction(loaded[0], tc);
         ASSERT_TRUE(solver.Solve());
         Vector x(2), u;
         x(0) = point[0];
         x(1) = point[1];
         ASSERT_TRUE(solver.DisplacementAt(x, u));
         uy.push_back(u(1));
         std::ostringstream st;
         st << "Q" << o.order << " plane " << plane << " E=1 nu=1/3 u_y="
            << std::setprecision(10) << u(1);
         rec.Error(std::string("default_Q") + std::to_string(o.order) + "_" + plane,
                   static_cast<int>(i), 48.0 / levels[i], "uy_error",
                   std::abs(u(1) - reference), st.str());
      }
      // Converging (successive changes shrink) and at the literature's value.
      for (std::size_t i = 2; i < uy.size(); ++i)
      {
         EXPECT_LT(std::abs(uy[i] - uy[i - 1]), std::abs(uy[i - 1] - uy[i - 2]))
               << plane << ", level " << i;
      }
      EXPECT_NEAR(uy.back(), reference, tol * reference) << "plane " << plane;
   }
}

TEST(VerifyElasticity, V5_DefaultTimeSchemeOrder)
{
   const YAML::Node b = verify::Baselines();
   const double margin = b["rate_margin"].as<double>();
   const double floor = b["noise_floor"].as<double>();
   const std::vector<int> steps = Ints(Settings()["dynamic_mms"]["steps"]);
   verify::Record rec("elasticity_dynamic_mms");
   const incns::Parameters p = DynamicDeck(2,
                                           Settings()["youngs_modulus"].as<double>(),
                                           Settings()["poisson_ratio"].as<double>(),
                                           "strain", 1.3);
   const incns::LinearElasticity::Options o = incns::ElasticityOptions(p);
   const std::string variant = "default_" + SchemeName(o) + "_Q" +
                               std::to_string(o.order);
   std::vector<double> dts, eu, ev;
   for (std::size_t i = 0; i < steps.size(); ++i)
   {
      Mesh serial = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true,
                                          1.0, 1.0);
      auto mesh = Partition(serial);
      SeparableMotion ex(2, o.lambda, o.mu, o.density);
      incns::LinearElasticity s(*mesh, o);
      const std::vector<int> held =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; });
      for (int a : held) { s.AddFixed(a); }
      for (int a : OtherAttributes(*mesh, held)) { s.AddTraction(a, *ex.traction); }
      s.SetBodyForce(*ex.f);
      s.Initialize(0.0, ex.u.get(), ex.v.get());
      const double dt = 1.0 / steps[i];
      for (int k = 0; k < steps[i]; ++k) { ASSERT_TRUE(s.Step(dt)); }
      ex.u->SetTime(s.Time());
      ex.v->SetTime(s.Time());
      const double e_u = s.Displacement().ComputeL2Error(*ex.u);
      const double e_v = s.Velocity().ComputeL2Error(*ex.v);
      std::ostringstream st;
      st << SchemeName(o) << " Q" << o.order << " 2x2, t = 1";
      rec.Error(variant, static_cast<int>(i), dt, "u_l2", e_u, st.str());
      rec.Error(variant, static_cast<int>(i), dt, "v_l2", e_v, st.str());
      dts.push_back(dt);
      eu.push_back(e_u);
      ev.push_back(e_v);
   }
   rec.Rate(variant, "u_l2", dts, eu, SchemeOrder(o), margin, floor);
   rec.Rate(variant, "v_l2", dts, ev, SchemeOrder(o), margin, floor);
}

TEST(VerifyElasticity, V6_CantileverFirstFrequency)
{
   const YAML::Node s = Settings()["cantilever_frequency"];
   const double L = s["length"].as<double>(), D = s["depth"].as<double>();
   const std::vector<int> ne = Ints(s["elements"]);
   const double E = s["youngs_modulus"].as<double>();
   const double rho = s["density"].as<double>();
   const int per = s["steps_per_period"].as<int>();
   const int periods = s["periods"].as<int>();
   verify::Record rec("elasticity_cantilever_frequency");
   const incns::Parameters p = DynamicDeck(2, E, s["poisson_ratio"].as<double>(),
                                           "stress", rho);
   const incns::LinearElasticity::Options o = incns::ElasticityOptions(p);

   // Euler-Bernoulli: omega_1 = (beta L)^2 / L^2 sqrt(E I / (rho A)), the
   // mode phi = cosh bx - cos bx - k (sinh bx - sin bx), u_x = -(y - D/2) phi'.
   const double bL = 1.8751040687119611, beta = bL / L;
   const double I = D * D * D / 12.0, A = D;
   const double omega = beta * beta * std::sqrt(E * I / (rho * A));
   const double k = (std::cosh(bL) + std::cos(bL)) / (std::sinh(bL) + std::sin(
                       bL));
   VectorFunctionCoefficient mode(2, [ = ](const Vector & x, Vector & u)
   {
      const double b = beta * x(0), amp = 1e-3;
      const double phi = std::cosh(b) - std::cos(b) - k * (std::sinh(b) - std::sin(
                            b));
      const double dphi = beta * (std::sinh(b) + std::sin(b) -
                                  k * (std::cosh(b) - std::cos(b)));
      u.SetSize(2);
      u(0) = -amp * (x(1) - 0.5 * D) * dphi;
      u(1) = amp * phi;
   });

   Mesh serial = Mesh::MakeCartesian2D(ne[0], ne[1], Element::QUADRILATERAL,
                                       true, L, D);
   auto mesh = Partition(serial);
   incns::LinearElasticity solver(*mesh, o);
   for (int a : AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; }))
   {
      solver.AddFixed(a);
   }
   solver.Initialize(0.0, &mode, nullptr);
   const double dt = 2.0 * M_PI / omega / per;
   Vector tip(2), u;
   tip(0) = L;
   tip(1) = 0.5 * D;
   ASSERT_TRUE(solver.DisplacementAt(tip, u));
   double prev = u(1), t_prev = 0.0;
   std::vector<double> crossings;
   for (int n = 0; n < periods * per + per / 4; ++n)
   {
      ASSERT_TRUE(solver.Step(dt));
      ASSERT_TRUE(solver.DisplacementAt(tip, u));
      const double t = solver.Time();
      if ((prev > 0.0) != (u(1) > 0.0))
      {
         crossings.push_back(t_prev + dt * prev / (prev - u(1)));
      }
      prev = u(1);
      t_prev = t;
   }
   ASSERT_GE(crossings.size(), 2u * periods);
   const double period = 2.0 * (crossings.back() - crossings.front()) /
                         (crossings.size() - 1);
   const double rel = 2.0 * M_PI / period / omega - 1.0;
   std::ostringstream st;
   st << SchemeName(o) << " Q" << o.order << " " << ne[0] << "x" << ne[1]
      << " L/D " << L / D << ": omega_h / omega_EB - 1 = " << rel;
   rec.Error("default_" + SchemeName(o) + "_Q" + std::to_string(o.order), 0,
             L / ne[0], "frequency_rel_error", std::abs(rel), st.str());
   if (Mpi::Root()) { std::cout << "[ info     ] V6 " << st.str() << std::endl; }
   EXPECT_LT(std::abs(rel), s["tolerance"].as<double>());
}
