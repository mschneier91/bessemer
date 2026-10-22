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
//     as the flag mesh is refined.
// Settings and thresholds: baselines.yaml `verification.hyperelasticity`.
// Theory (V1): ||u - u_h||_L2 at k + 1, |u - u_h|_H1 at k. Pass: the rate
// fitted over all levels >= theory - rate_margin.

#include <gtest/gtest.h>

#include "elasticity_test_util.hpp"
#include "verify_util.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "quadrature/rule_book.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/hyperelasticity.hpp"
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

/// The Turek-Hron flag deck of CSM test @p name on an n_flag x n_flag_y mesh.
incns::Parameters CsmDeck(const YAML::Node& csm, const YAML::Node& test,
                          int n_flag, int n_flag_y)
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
     << "  plane: strain\n  order: " << csm["order"].as<int>() << "\n"
     << "  body_force: [" << g[0] << ", " << g[1] << "]\n"
     << "  rtol: " << Settings()["rtol"].as<double>() << "\n"
     << "  probe: [0.6, 0.2]\n"
     << "boundary_conditions:\n"
     << "  - {select: [cylinder], type: fixed}\n"
     << "  - {select: [flag], type: free}\n";
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
      const incns::Parameters p = CsmDeck(csm, test, levels[i][0], levels[i][1]);
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

} // namespace

TEST(VerifyHyperelasticity, V1_FiniteManufacturedDeformation2D) { TrigStudy(2); }

TEST(VerifyHyperelasticity, V1_FiniteManufacturedDeformation3D) { TrigStudy(3); }

TEST(VerifyHyperelasticity, V2_TurekHronCSM1) { CsmStudy("csm1"); }

TEST(VerifyHyperelasticity, V2_TurekHronCSM2) { CsmStudy("csm2"); }
