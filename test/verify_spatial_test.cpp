// Verification tier (label `verify`): spatial convergence rates of steady
// Stokes through incns::Case, against an exact trigonometric solution.
//  V1 2D, Taylor-Hood Q_k/Q_{k-1} for every k in the degree sweep;
//  V2 3D, the same for the 3D degrees;
//  V3 2D on a curved (isoparametric) mesh of the unit square: the interior
//     nodes moved by a smooth map that leaves the boundary in place;
//  V4 2D on nonconforming meshes: each level is the uniform box with the AMR
//     tests' refined bands (hanging nodes inside and on the walls).
// Each level marches the unsteady Stokes equations from the interpolated exact
// solution with large steps to the discrete steady state (BDF2 damps every
// mode by ~0.15 per step at nu dt = 1), so the error is purely spatial. Every
// key the study is not about is the case-level default (verify_util.hpp), and
// every study runs with grad-div at its default and switched off.
// Theory, Taylor-Hood: |u - u_h|_H1 and ||p - p_h|| converge at
// min(k_u, k_p + 1), ||u - u_h|| one order higher. Pass: the rate fitted over
// all levels >= theory - rate_margin (baselines.yaml `verification`).

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "verify_util.hpp"

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/periodic_box.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

using VecFn = std::function<void(const Vector&, Vector&)>;
using MatFn = std::function<void(const Vector&, DenseMatrix&)>;
using ScalarFn = std::function<double(const Vector&)>;

/// The exact steady Stokes solution on the unit square / cube.
///  2D: u = curl(sin pi x sin pi y) = pi (sin pi x cos pi y, -cos pi x sin pi y),
///      p = cos pi x cos pi y;
///  3D: u = (s c c, c s c, -2 c c s) (s = sin pi ., c = cos pi .; divergence
///      free), p = cos pi x cos pi y cos pi z.
/// Both u are eigenfunctions of the Laplacian (-lap u = dim pi^2 u) and both p
/// have zero mean, so f = dim pi^2 nu u + grad p.
struct Fields
{
   std::unique_ptr<VectorFunctionCoefficient> u;    ///< Exact velocity.
   std::unique_ptr<MatrixFunctionCoefficient> grad; ///< Its gradient du_i/dx_j.
   std::unique_ptr<FunctionCoefficient> p;          ///< Exact pressure.
   std::unique_ptr<VectorFunctionCoefficient> f;    ///< Forcing.

   Fields(int dim, double nu)
   {
      const double pi = M_PI;
      VecFn vel, rhs;
      MatFn dvel;
      ScalarFn pres;
      if (dim == 2)
      {
         vel = [pi](const Vector & x, Vector & v)
         {
            v(0) = pi * std::sin(pi * x(0)) * std::cos(pi * x(1));
            v(1) = -pi * std::cos(pi * x(0)) * std::sin(pi * x(1));
         };
         dvel = [pi](const Vector & x, DenseMatrix & g)
         {
            const double sx = std::sin(pi * x(0)), cx = std::cos(pi * x(0));
            const double sy = std::sin(pi * x(1)), cy = std::cos(pi * x(1));
            g(0, 0) = pi * pi * cx * cy;
            g(0, 1) = -pi * pi * sx * sy;
            g(1, 0) = pi * pi * sx * sy;
            g(1, 1) = -pi * pi * cx * cy;
         };
         pres = [pi](const Vector & x)
         {
            return std::cos(pi * x(0)) * std::cos(pi * x(1));
         };
         rhs = [pi, nu, vel](const Vector & x, Vector & v)
         {
            vel(x, v);
            v *= 2.0 * pi * pi * nu;
            v(0) -= pi * std::sin(pi * x(0)) * std::cos(pi * x(1));
            v(1) -= pi * std::cos(pi * x(0)) * std::sin(pi * x(1));
         };
      }
      else
      {
         vel = [pi](const Vector & x, Vector & v)
         {
            const double sx = std::sin(pi * x(0)), cx = std::cos(pi * x(0));
            const double sy = std::sin(pi * x(1)), cy = std::cos(pi * x(1));
            const double sz = std::sin(pi * x(2)), cz = std::cos(pi * x(2));
            v(0) = sx * cy * cz;
            v(1) = cx * sy * cz;
            v(2) = -2.0 * cx * cy * sz;
         };
         dvel = [pi](const Vector & x, DenseMatrix & g)
         {
            const double sx = std::sin(pi * x(0)), cx = std::cos(pi * x(0));
            const double sy = std::sin(pi * x(1)), cy = std::cos(pi * x(1));
            const double sz = std::sin(pi * x(2)), cz = std::cos(pi * x(2));
            g(0, 0) = pi * cx * cy * cz;
            g(0, 1) = -pi * sx * sy * cz;
            g(0, 2) = -pi * sx * cy * sz;
            g(1, 0) = -pi * sx * sy * cz;
            g(1, 1) = pi * cx * cy * cz;
            g(1, 2) = -pi * cx * sy * sz;
            g(2, 0) = 2.0 * pi * sx * cy * sz;
            g(2, 1) = 2.0 * pi * cx * sy * sz;
            g(2, 2) = -2.0 * pi * cx * cy * cz;
         };
         pres = [pi](const Vector & x)
         {
            return std::cos(pi * x(0)) * std::cos(pi * x(1)) * std::cos(pi * x(2));
         };
         rhs = [pi, nu, vel](const Vector & x, Vector & v)
         {
            const double sx = std::sin(pi * x(0)), cx = std::cos(pi * x(0));
            const double sy = std::sin(pi * x(1)), cy = std::cos(pi * x(1));
            const double sz = std::sin(pi * x(2)), cz = std::cos(pi * x(2));
            vel(x, v);
            v *= 3.0 * pi * pi * nu;
            v(0) -= pi * sx * cy * cz;
            v(1) -= pi * cx * sy * cz;
            v(2) -= pi * cx * cy * sz;
         };
      }
      u = std::make_unique<VectorFunctionCoefficient>(dim, vel);
      grad = std::make_unique<MatrixFunctionCoefficient>(dim, dvel);
      p = std::make_unique<FunctionCoefficient>(pres);
      f = std::make_unique<VectorFunctionCoefficient>(dim, rhs);
   }
};

/// How a level's mesh is built from the deck's box.
enum class MeshKind { Box, Curved, Nonconforming };

/// "[a, a, a]" with @p dim entries.
template <typename T>
std::string List(int dim, T a)
{
   std::ostringstream s;
   s << std::boolalpha << std::setprecision(17) << "[";
   for (int d = 0; d < dim; ++d) { s << (d ? ", " : "") << a; }
   s << "]";
   return s.str();
}

/// The study's deck for one level: unit box with n elements per direction,
/// the given orders (k <= 0: the default orders), the pseudo-time march and
/// tolerances from baselines.yaml, and the grad-div variant's lines.
std::string Deck(int dim, int n, int k, const verify::GradDivVariant& v)
{
   const YAML::Node b = verify::Baselines()["steady_stokes"];
   std::ostringstream d;
   d << std::setprecision(17)
     << "equation: stokes\n"
     << "physics:\n  nu: " << b["nu"].as<double>() << "\n" << v.physics;
   if (k > 0)
   {
      d << "discretization:\n  order_u: " << k << "\n  order_p: " << k - 1
        << "\n";
   }
   d << "mesh:\n  dim: " << dim << "\n  elements: " << List(dim, n)
     << "\n  lengths: " << List(dim, 1.0) << "\n  periodic: "
     << List(dim, false) << "\n"
     << "time:\n  dt: " << b["dt"].as<double>() << "\n  t_final: "
     << b["t_final"].as<double>() << "\n"
     << "solver:\n  rtol: " << b["rtol"].as<double>() << "\n  atol: "
     << b["atol"].as<double>() << "\n";
   return d.str();
}

/// One level's mesh.
std::unique_ptr<ParMesh> BuildMesh(const incns::Parameters& p, MeshKind kind)
{
   switch (kind)
   {
      case MeshKind::Box:
         return incns::MakeCaseMesh(p);
      case MeshKind::Curved:
      {
         // Isoparametric: geometry of the velocity order. The displacement
         // a sin(pi x) sin(pi y) (1, 1) vanishes on the boundary, so the
         // domain stays the unit square; det J = 1 + a pi sin(pi (x + y)).
         const double a =
            verify::Baselines()["curved_2d"]["amplitude"].as<double>();
         Mesh serial = incns::MakeBoxMesh(p.mesh);
         serial.SetCurvature(p.order_u);
         serial.Transform([a](const Vector & x, Vector & y)
         {
            const double s = a * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
            y = x;
            y(0) += s;
            y(1) += s;
         });
         return incns::PartitionMesh(serial, false);
      }
      case MeshKind::Nonconforming:
      {
         Mesh serial = incns::MakeBoxMesh(p.mesh);
         auto mesh = incns::PartitionMesh(serial, true);
         amr_test::RefineBands(*mesh, false);
         return mesh;
      }
   }
   return nullptr;
}

/// March one level to its steady state; the errors against the exact
/// solution. @p settings receives the run's resolved defaults (provenance).
verify::Errors RunLevel(const std::string& deck, MeshKind kind,
                        std::string& settings, int& order_u, int& order_p)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   const int dim = p.mesh.dim;
   Fields fields(dim, p.nu);
   incns::RuleBook rules; // before the mesh, so it outlives it
   std::unique_ptr<ParMesh> mesh = BuildMesh(p, kind);
   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   for (int a : mesh->bdr_attributes) { bc.AddVelocityDirichlet(a, *fields.u); }
   flow.SetBoundaryConditions(bc);
   flow.SetInitialVelocity(*fields.u);
   flow.SetForcing(*fields.f);
   flow.Run();
   EXPECT_FALSE(flow.Diverged());
   EXPECT_NEAR(flow.Time(), p.t_final, 1e-12);
   std::ostringstream s;
   s << "Q" << p.order_u << "/Q" << p.order_p << " grad_div_h="
     << p.grad_div_h << " grad_div_nu=" << p.grad_div_nu << " steps="
     << flow.Integrator().StepCount();
   settings = s.str();
   order_u = p.order_u;
   order_p = p.order_p;
   return verify::ComputeErrors(flow.Velocity(), flow.Pressure(), *fields.u,
                                *fields.grad, *fields.p, rules,
                                2 * p.order_u + 4);
}

/// One convergence study: every level of @p levels at velocity order @p k
/// (k <= 0: the default), both grad-div variants; rates checked.
void Study(verify::Record& rec, const std::string& label, int dim, int k,
           const std::vector<int>& levels, MeshKind kind)
{
   const YAML::Node b = verify::Baselines();
   const double margin = b["rate_margin"].as<double>();
   const double floor = b["noise_floor"].as<double>();
   for (const verify::GradDivVariant& v : verify::GradDivVariants())
   {
      SCOPED_TRACE(label + " " + v.name);
      const std::string variant = label + "/" + v.name;
      std::vector<double> h, eu, eh1, ep;
      int ku = 0, kp = 0;
      for (std::size_t l = 0; l < levels.size(); ++l)
      {
         std::string settings;
         const verify::Errors e =
            RunLevel(Deck(dim, levels[l], k, v), kind, settings, ku, kp);
         h.push_back(1.0 / levels[l]);
         eu.push_back(e.u_l2);
         eh1.push_back(e.u_h1);
         ep.push_back(e.p_l2);
         rec.Error(variant, l, h.back(), "u_l2", e.u_l2, settings);
         rec.Error(variant, l, h.back(), "u_h1", e.u_h1, settings);
         rec.Error(variant, l, h.back(), "p_l2", e.p_l2, settings);
         rec.Error(variant, l, h.back(), "div_l2", e.div_l2, settings);
      }
      const double energy = std::min(ku, kp + 1); // Taylor-Hood
      rec.Rate(variant, "u_l2", h, eu, energy + 1.0, margin, floor);
      rec.Rate(variant, "u_h1", h, eh1, energy, margin, floor);
      rec.Rate(variant, "p_l2", h, ep, energy, margin, floor);
   }
}

std::vector<int> Ints(const YAML::Node& n) { return n.as<std::vector<int>>(); }

} // namespace

TEST(VerifySpatial, V1_StokesDegrees2D)
{
   const YAML::Node b = verify::Baselines()["stokes_2d"];
   verify::Record rec("stokes_2d");
   for (int k : Ints(b["orders"]))
   {
      Study(rec, "Q" + std::to_string(k), 2, k, Ints(b["levels"]),
            MeshKind::Box);
   }
}

TEST(VerifySpatial, V2_StokesDegrees3D)
{
   const YAML::Node b = verify::Baselines()["stokes_3d"];
   verify::Record rec("stokes_3d");
   for (int k : Ints(b["orders"]))
   {
      Study(rec, "Q" + std::to_string(k), 3, k, Ints(b["levels"]),
            MeshKind::Box);
   }
}

TEST(VerifySpatial, V3_CurvedMesh2D)
{
   const YAML::Node b = verify::Baselines()["curved_2d"];
   verify::Record rec("curved_2d");
   Study(rec, "default_order", 2, 0, Ints(b["levels"]), MeshKind::Curved);
}

TEST(VerifySpatial, V4_NonconformingMesh2D)
{
   const YAML::Node b = verify::Baselines()["nonconforming_2d"];
   verify::Record rec("nonconforming_2d");
   Study(rec, "default_order", 2, 0, Ints(b["levels"]),
         MeshKind::Nonconforming);
}
