// Verification tier (label `verify`): large-deformation elasticity (the static
// St. Venant-Kirchhoff solver, solid/hyperelasticity.hpp) against an exact
// solution and the Turek-Hron structural benchmark.
//  V1 a manufactured finite deformation u_i = a prod_d sin(pi x_d) on the unit
//     square (2D) and cube (3D), held on the whole boundary, under the body
//     force -Div P(F(u)) (P evaluated here, independently of the kernels);
//     the displacement gradient reaches a pi, so strains are finite. a stays
//     where SVK is strongly elliptic: at a = 0.1 (2D, J down to 0.69) the
//     acoustic tensor's least eigenvalue is -0.025, the problem ill-posed,
//     and Newton diverges once the mesh resolves the unstable modes;
//  V2 Turek & Hron's CSM1 and CSM2 (2006): the flag clamped to the cylinder
//     under gravity, through a deck and ElasticityCase; the displacement of
//     point A against the benchmark's reference values (FeatFlow), converging
//     as the flag mesh is refined. CSM3: the flag released from rest, dynamic;
//     the mean, amplitude and frequency of point A's oscillation against the
//     benchmark's at its time step;
//  V3 mesh motion: the DFG cylinder turned about its centre by a large angle
//     and back, the walls held, by the linear, St. Venant-Kirchhoff and
//     neo-Hookean models: each one's reach.
// Settings and thresholds: baselines.yaml `verification.hyperelasticity`.
// Theory (V1): ||u - u_h||_L2 at k + 1, |u - u_h|_H1 at k. Pass: the rate
// fitted over all levels >= theory - rate_margin.

#include <gtest/gtest.h>

#include "elasticity_test_util.hpp"
#include "turned_cylinder.hpp"
#include "verify_util.hpp"

#include "ale/hyperelastic_motion.hpp"
#include "ale/linear_elastic_motion.hpp"
#include "ale/mesh_motion.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "quadrature/rule_book.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/hyperelasticity.hpp"
#include "mfem.hpp"

#include <algorithm>
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

YAML::Node Settings() { return verify::Baselines()["hyperelasticity"]; }

std::vector<int> Ints(const YAML::Node& n) { return n.as<std::vector<int>>(); }

/// SVK at F: S = lambda tr(E) I + 2 mu E, E = (F^T F - I) / 2, and P = F S.
void Svk(const DenseMatrix& F, double lambda, double mu, DenseMatrix& S,
         DenseMatrix& P)
{
   const int dim = F.Height();
   DenseMatrix E(dim);
   MultAtB(F, F, E);
   for (int i = 0; i < dim; ++i) { E(i, i) -= 1.0; }
   E *= 0.5;
   S = E;
   S *= 2.0 * mu;
   const double tr = E.Trace();
   for (int i = 0; i < dim; ++i) { S(i, i) += lambda * tr; }
   P.SetSize(dim);
   Mult(F, S, P);
}

/// The derivative of P at F in the direction dF: dF S + F dS, dS = lambda
/// tr(dE) I + 2 mu dE, dE = sym(F^T dF).
void SvkDerivative(const DenseMatrix& F, const DenseMatrix& S,
                   const DenseMatrix& dF, double lambda, double mu,
                   DenseMatrix& dP)
{
   const int dim = F.Height();
   DenseMatrix FtdF(dim), dE(dim), dS(dim), t(dim);
   MultAtB(F, dF, FtdF);
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < dim; ++j) { dE(i, j) = 0.5 * (FtdF(i, j) + FtdF(j, i)); }
   }
   dS = dE;
   dS *= 2.0 * mu;
   const double tr = dE.Trace();
   for (int i = 0; i < dim; ++i) { dS(i, i) += lambda * tr; }
   dP.SetSize(dim);
   Mult(dF, S, dP);
   Mult(F, dS, t);
   dP += t;
}

/**
 * u_i = a prod_d sin(pi x_d) for every component i: zero on the unit box's
 * boundary, f = -Div P(I + grad u) with Div P_i = sum_J dP(F; d_J F)_iJ.
 */
std::unique_ptr<Exact> FiniteTrigonometric(int dim, double a, double lambda,
      double mu)
{
   const double pi = M_PI;
   // prod_{d != s1, s2} sin(pi x_d).
   auto ps = [dim, pi](const Vector & x, int s1, int s2)
   {
      double s = 1.0;
      for (int d = 0; d < dim; ++d)
      {
         if (d != s1 && d != s2) { s *= std::sin(pi * x(d)); }
      }
      return s;
   };
   VecFn uf = [dim, a, ps](const Vector & x, Vector & v)
   {
      v.SetSize(dim);
      v = a * ps(x, -1, -1);
   };
   MatFn gf = [dim, a, pi, ps](const Vector & x, DenseMatrix & g)
   {
      g.SetSize(dim);
      for (int j = 0; j < dim; ++j)
      {
         const double dj = a * pi * std::cos(pi * x(j)) * ps(x, j, -1);
         for (int i = 0; i < dim; ++i) { g(i, j) = dj; }
      }
   };
   VecFn ff = [dim, a, pi, ps, gf, lambda, mu](const Vector & x, Vector & v)
   {
      DenseMatrix F(dim), S, P, dF(dim), dP;
      gf(x, F);
      for (int i = 0; i < dim; ++i) { F(i, i) += 1.0; }
      Svk(F, lambda, mu, S, P);
      v.SetSize(dim);
      v = 0.0;
      for (int J = 0; J < dim; ++J)
      {
         // d_J F: (d_J F)_iK = d_J d_K u_i, the same for every i.
         for (int K = 0; K < dim; ++K)
         {
            const double h = (K == J) ? -a * pi * pi * ps(x, -1, -1) :
                             a * pi * pi * std::cos(pi * x(K)) * std::cos(pi * x(J)) *
                             ps(x, K, J);
            for (int i = 0; i < dim; ++i) { dF(i, K) = h; }
         }
         SvkDerivative(F, S, dF, lambda, mu, dP);
         for (int i = 0; i < dim; ++i) { v(i) -= dP(i, J); }
      }
   };
   return std::make_unique<Exact>(dim, uf, gf, ff, lambda, mu);
}

/// A minimal hyperelastic deck: dimension, material, order (k <= 0: the
/// default), and the tolerances of baselines.yaml.
incns::Parameters Deck(int dim, int k, double E, double nu)
{
   std::ostringstream d;
   d << std::setprecision(17) << "equation: hyperelasticity\n"
     << "mesh:\n  dim: " << dim << "\n"
     << "solid:\n  youngs_modulus: " << E << "\n  poisson_ratio: " << nu
     << "\n  rtol: " << Settings()["rtol"].as<double>()
     << "\n  newton_rtol: " << Settings()["newton_rtol"].as<double>() << "\n";
   if (k > 0) { d << "  order: " << k << "\n"; }
   return incns::Parameters::LoadYAMLString(d.str());
}

std::unique_ptr<Mesh> Box(int dim, int n)
{
   return std::make_unique<Mesh>(dim == 2 ?
                                 Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0) :
                                 Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0));
}

/// A rate study of V1 over @p levels at order @p k (k <= 0: the default).
void Study(verify::Record& rec, int dim, int k, const std::vector<int>& levels)
{
   const YAML::Node b = verify::Baselines();
   const double margin = b["rate_margin"].as<double>();
   const double floor = b["noise_floor"].as<double>();
   const double amplitude = Settings()["mms_amplitude"].as<double>();
   const incns::Parameters p = Deck(dim, k,
                                    Settings()["youngs_modulus"].as<double>(),
                                    Settings()["poisson_ratio"].as<double>());
   const incns::Hyperelasticity::Options o = incns::HyperelasticOptions(p);
   const std::string variant = (k > 0 ? "Q" : "default_Q") + std::to_string(
                                  o.order);
   std::vector<double> hs, l2, h1;
   for (std::size_t i = 0; i < levels.size(); ++i)
   {
      const int n = levels[i];
      std::unique_ptr<Mesh> serial = Box(dim, n);
      ParMesh mesh(MPI_COMM_WORLD, *serial);
      serial.reset();
      incns::RuleBook rules;
      Errors e;
      int newton = 0;
      {
         incns::Hyperelasticity s(mesh, o);
         auto ex = FiniteTrigonometric(dim, amplitude, o.lambda, o.mu);
         for (int a = 1; a <= mesh.bdr_attributes.Max(); ++a) { s.AddFixed(a); }
         s.SetBodyForce(*ex->f);
         EXPECT_TRUE(s.Solve()) << variant << " level " << n;
         newton = s.NewtonIterations();
         e = ComputeErrors(s.Displacement(), *ex, rules, 2 * o.order + 4);
      }
      const double h = 1.0 / n;
      std::ostringstream st;
      st << "Q" << o.order << " a=" << amplitude << " E=" << p.solid.youngs_modulus
         << " nu=" << p.solid.poisson_ratio << " newton=" << newton;
      rec.Error(variant, static_cast<int>(i), h, "u_l2", e.l2, st.str());
      rec.Error(variant, static_cast<int>(i), h, "u_h1", e.h1, st.str());
      hs.push_back(h);
      l2.push_back(e.l2);
      h1.push_back(e.h1);
   }
   rec.Rate(variant, "u_l2", hs, l2, o.order + 1.0, margin, floor);
   rec.Rate(variant, "u_h1", hs, h1, o.order, margin, floor);
}

void TrigStudy(int dim)
{
   const YAML::Node s = Settings()[dim == 2 ? "mms_2d" : "mms_3d"];
   verify::Record rec(dim == 2 ? "hyperelasticity_mms_2d" :
                      "hyperelasticity_mms_3d");
   for (int k : Ints(s["orders"])) { Study(rec, dim, k, Ints(s["levels"])); }
   Study(rec, dim, 0, Ints(s["default_levels"]));
}

/**
 * The Turek-Hron flag deck of a CSM test on an n_flag x n_flag_y mesh at
 * @p order; @p solid_extra is appended to the solid section, @p tail to the
 * deck.
 */
incns::Parameters CsmDeck(const YAML::Node& csm, const YAML::Node& test,
                          int n_flag, int n_flag_y, int order,
                          const std::string& solid_extra = "",
                          const std::string& tail = "")
{
   const std::vector<double> g = csm["body_force"].as<std::vector<double>>();
   std::ostringstream d;
   d << std::setprecision(17)
     << "equation: hyperelasticity\n"
     << "mesh:\n  dim: 2\n  geometry: cylinder_channel\n  cylinder_channel:\n"
     << "    length: 2.5\n    flag_length: 0.35\n    flag_thickness: 0.02\n"
     << "    n_flag: " << n_flag << "\n    n_flag_y: " << n_flag_y << "\n"
     << "    part: solid\n"
     << "solid:\n  model: st_venant_kirchhoff\n"
     << "  youngs_modulus: " << test["youngs_modulus"].as<double>() << "\n"
     << "  poisson_ratio: " << csm["poisson_ratio"].as<double>() << "\n"
     << "  plane: strain\n  order: " << order << "\n"
     << "  body_force: [" << g[0] << ", " << g[1] << "]\n"
     << "  rtol: " << Settings()["rtol"].as<double>() << "\n"
     << "  probe: [0.6, 0.2]\n" << solid_extra
     << "boundary_conditions:\n"
     << "  - {select: [cylinder], type: fixed}\n"
     << "  - {select: [flag], type: free}\n" << tail;
   return incns::Parameters::LoadYAMLString(d.str());
}

void CsmStudy(const std::string& name)
{
   const YAML::Node csm = Settings()["turek_hron_csm"];
   const YAML::Node test = csm[name];
   const std::vector<double> ref = test["reference_u_a"].as<std::vector<double>>();
   const double tol = csm["tolerance"].as<double>();
   const std::vector<std::vector<int>> levels =
      csm["levels"].as<std::vector<std::vector<int>>>();
   verify::Record rec("hyperelasticity_" + name);
   const std::string variant = "Q" + std::to_string(csm["order"].as<int>());
   std::vector<double> ux, uy;
   for (std::size_t i = 0; i < levels.size(); ++i)
   {
      const incns::Parameters p = CsmDeck(csm, test, levels[i][0], levels[i][1],
                                          csm["order"].as<int>());
      auto mesh = incns::MakeCaseMesh(p);
      incns::ElasticityCase c(*mesh, p);
      ASSERT_TRUE(c.Solve()) << name << " level " << i;
      Vector x(2), u;
      x(0) = 0.6;
      x(1) = 0.2;
      ASSERT_TRUE(c.HyperelasticSolver().DisplacementAt(x, u));
      ux.push_back(u(0));
      uy.push_back(u(1));
      std::ostringstream st;
      st << std::setprecision(8) << variant << " flag " << levels[i][0] << " x "
         << levels[i][1] << " u(A) = (" << u(0) << ", " << u(1) << "), Newton "
         << c.HyperelasticSolver().NewtonIterations();
      const double h = 0.35 / levels[i][0];
      rec.Error(variant, static_cast<int>(i), h, "ux_error",
                std::abs(u(0) - ref[0]), st.str());
      rec.Error(variant, static_cast<int>(i), h, "uy_error",
                std::abs(u(1) - ref[1]), st.str());
   }
   // At the benchmark's values on the finest mesh, and converging there.
   EXPECT_NEAR(ux.back(), ref[0], tol * std::abs(ref[0])) << name;
   EXPECT_NEAR(uy.back(), ref[1], tol * std::abs(ref[1])) << name;
   const std::size_t n = uy.size();
   if (n >= 2)
   {
      EXPECT_LE(std::abs(uy[n - 1] - ref[1]), std::abs(uy[n - 2] - ref[1]) +
                tol * std::abs(ref[1]) / 10.0) << name;
   }
}

/// An oscillation's mean and amplitude (from its extremes) and frequency (the
/// mean period between upward crossings of the mean, interpolated).
struct Oscillation
{
   double mean = 0.0, amplitude = 0.0, frequency = 0.0;
};

Oscillation Analyze(const std::vector<double>& t, const std::vector<double>& u)
{
   const auto lohi = std::minmax_element(u.begin(), u.end());
   Oscillation o;
   o.mean = 0.5 * (*lohi.second + *lohi.first);
   o.amplitude = 0.5 * (*lohi.second - *lohi.first);
   std::vector<double> ups;
   for (std::size_t i = 0; i + 1 < u.size(); ++i)
   {
      if (u[i] < o.mean && u[i + 1] >= o.mean)
      {
         ups.push_back(t[i] + (o.mean - u[i]) * (t[i + 1] - t[i]) / (u[i + 1] - u[i]));
      }
   }
   if (ups.size() > 1)
   {
      o.frequency = static_cast<double>(ups.size() - 1) / (ups.back() - ups.front());
   }
   return o;
}

} // namespace

TEST(VerifyHyperelasticity, V1_FiniteManufacturedDeformation2D) { TrigStudy(2); }

TEST(VerifyHyperelasticity, V1_FiniteManufacturedDeformation3D) { TrigStudy(3); }

TEST(VerifyHyperelasticity, V2_TurekHronCSM1) { CsmStudy("csm1"); }

TEST(VerifyHyperelasticity, V2_TurekHronCSM2) { CsmStudy("csm2"); }

TEST(VerifyHyperelasticity, V2_TurekHronCSM3)
{
   const YAML::Node csm = Settings()["turek_hron_csm"];
   const YAML::Node c3 = csm["csm3"];
   const double dt = c3["dt"].as<double>();
   const double tf = c3["t_final"].as<double>();
   const double t0 = c3["window_start"].as<double>();
   const double tol = c3["tolerance"].as<double>();
   const double tol_f = c3["tolerance_frequency"].as<double>();
   const std::vector<int> level = c3["level"].as<std::vector<int>>();
   const int order = c3["order"].as<int>();
   std::ostringstream extra, tail;
   extra << std::setprecision(17) << "  analysis: dynamic\n  density: "
         << c3["density"].as<double>() << "\n  rho_inf: "
         << c3["rho_inf"].as<double>() << "\n";
   tail << std::setprecision(17) << "time:\n  dt: " << dt << "\n  t_final: " << tf
        << "\n";
   const incns::Parameters p = CsmDeck(csm, c3, level[0], level[1], order,
                                       extra.str(), tail.str());
   auto mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   incns::Hyperelasticity& s = c.HyperelasticSolver();
   s.Initialize(0.0);
   Vector x(2), u;
   x(0) = 0.6;
   x(1) = 0.2;
   std::vector<double> t, ux, uy;
   int most = 0;
   const long n = std::lround(tf / dt);
   for (long k = 1; k <= n; ++k)
   {
      ASSERT_TRUE(s.Step(dt)) << "step " << k;
      most = std::max(most, s.NewtonIterations());
      ASSERT_TRUE(s.DisplacementAt(x, u));
      if (s.Time() >= t0 - 1e-12)
      {
         t.push_back(s.Time());
         ux.push_back(u(0));
         uy.push_back(u(1));
      }
   }
   const Oscillation ox = Analyze(t, ux), oy = Analyze(t, uy);
   const std::vector<double> rx = c3["reference_u_x"].as<std::vector<double>>();
   const std::vector<double> ry = c3["reference_u_y"].as<std::vector<double>>();
   const double rf = c3["reference_frequency"].as<double>();
   verify::Record rec("hyperelasticity_csm3");
   std::ostringstream st;
   st << std::setprecision(6) << "Q" << order << " flag " << level[0] << " x "
      << level[1] << " dt " << dt << " u_x " << 1e3 * ox.mean << " +- " << 1e3 *
      ox.amplitude << " [" << ox.frequency << "] u_y " << 1e3 * oy.mean << " +- "
      << 1e3 * oy.amplitude << " [" << oy.frequency << "] Newton max " << most;
   const std::string variant = "Q" + std::to_string(order);
   rec.Error(variant, 0, dt, "ux_mean_error", std::abs(ox.mean - rx[0]), st.str());
   rec.Error(variant, 0, dt, "ux_amplitude_error", std::abs(ox.amplitude - rx[1]),
             st.str());
   rec.Error(variant, 0, dt, "uy_mean_error", std::abs(oy.mean - ry[0]), st.str());
   rec.Error(variant, 0, dt, "uy_amplitude_error", std::abs(oy.amplitude - ry[1]),
             st.str());
   rec.Error(variant, 0, dt, "frequency_error", std::abs(oy.frequency - rf),
             st.str());
   EXPECT_NEAR(ox.mean, rx[0], tol * std::abs(rx[0])) << st.str();
   EXPECT_NEAR(ox.amplitude, rx[1], tol * rx[1]) << st.str();
   EXPECT_NEAR(oy.mean, ry[0], tol * std::abs(ry[0])) << st.str();
   EXPECT_NEAR(oy.amplitude, ry[1], tol * ry[1]) << st.str();
   EXPECT_NEAR(ox.frequency, rf, tol_f * rf) << st.str();
   EXPECT_NEAR(oy.frequency, rf, tol_f * rf) << st.str();
}

TEST(VerifyHyperelasticity, V3_CylinderTurnedFar)
{
   // The DFG cylinder turned about its centre in steps of step_degrees up to
   // top_degrees and back, every other boundary held, stiffening chi = 1: the
   // linear, St. Venant-Kirchhoff and neo-Hookean models. Each one's reach,
   // the largest angle on the way up with every move converged and det F > 0.
   // The neo-Hookean model must reach at least as far as the linear one and
   // reach_degrees, converge at every move, and return to rest
   // (docs/hyperelasticity.md section 4).
   using namespace turned_cylinder;
   const YAML::Node s = Settings()["mesh_motion"];
   const double step = s["step_degrees"].as<double>() * M_PI / 180.0;
   const double top = s["top_degrees"].as<double>() * M_PI / 180.0;
   const int n = static_cast<int>(std::lround(top / step));
   const incns::CylinderChannelSpec spec;
   VectorFunctionCoefficient g(2, [spec, top](const Vector & X, double t,
                               Vector & d)
   {
      Turn(spec, TurnAngle(t, top), X, d);
   });
   const char* names[3] = {"linear_elastic", "st_venant_kirchhoff", "neo_hookean"};
   double reach[3] = {0.0, 0.0, 0.0};
   for (int m = 0; m < 3; ++m)
   {
      auto mesh = CylinderChannel(spec);
      incns::MeshMotionModelFactory factory;
      if (m == 0)
      {
         factory = incns::LinearElasticMotion::Factory(
                      incns::LinearElasticMotion::Options());
      }
      else
      {
         incns::HyperelasticMotion::Options h;
         h.material = m == 1 ? incns::HyperelasticMaterial::StVenantKirchhoff :
                      incns::HyperelasticMaterial::NeoHookean;
         factory = incns::HyperelasticMotion::Factory(h);
      }
      incns::MeshMotion motion(*mesh, incns::MeshMotion::Options(), factory);
      motion.AddDisplacement(incns::kCylinderBody, g);
      bool valid = true, all = true;
      for (int k = 1; k <= 2 * n; ++k)
      {
         const bool ok = motion.Move(k * step);
         all = all && ok;
         double jmin, jmax;
         motion.JacobianRange(jmin, jmax);
         const double deg = TurnAngle(k * step, top) * 180.0 / M_PI;
         if (k <= n)
         {
            valid = valid && ok && jmin > 0.0;
            if (valid) { reach[m] = deg; }
         }
         int newton = 0;
         if (m > 0)
         {
            newton = static_cast<incns::HyperelasticMotion&>(motion.Model())
                     .Solver().NewtonIterations();
         }
         if (Mpi::Root())
         {
            std::cout << "[V3] " << names[m] << " " << std::lround(deg)
                      << " deg: det F in [" << jmin << ", " << jmax << "], "
                      << (ok ? "converged" : "NOT converged") << ", Newton "
                      << newton << ", CG " << motion.Model().Iterations()
                      << std::endl;
         }
      }
      const double rest = RestError(*mesh, motion.ReferenceNodes());
      if (Mpi::Root())
      {
         std::cout << "[V3] " << names[m] << ": reach " << reach[m]
                   << " deg; back at rest: max |x - X| " << rest << ", every move "
                   << (all ? "converged" : "NOT converged") << std::endl;
      }
      if (m == 2)
      {
         EXPECT_TRUE(all) << "a neo-Hookean move did not converge";
         EXPECT_LT(rest, 1e-10);
      }
   }
   EXPECT_GE(reach[2], reach[0]) << "the neo-Hookean model turns the cylinder "
                                 "less far than the linear one";
   EXPECT_GE(reach[2], s["reach_degrees"].as<double>());
}
