// Verification tier (label `verify`): spatial convergence of the steady
// Navier-Stokes equations through incns::Case, against Kovasznay's exact
// solution (Proc. Camb. Phil. Soc. 44 (1948) 58-62), the standard NSE oracle:
//   u = 1 - e^{lambda x} cos(2 pi y),  v = lambda / (2 pi) e^{lambda x} sin(2 pi y),
//   p = (1 - e^{2 lambda x}) / 2,  lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2),
// on [-0.5, 1] x [-0.5, 1.5] (the box [0, 1.5] x [0, 2] shifted by 0.5),
// exact velocity on the whole boundary, no forcing. Each level starts from the
// interpolated solution and marches with the case-level defaults (CFL-
// controlled steps, grad-div, ...) until the error stops changing, except the
// convection, pinned to IMEX: the steady state of the IMEX scheme is the
// discrete steady solution, so what remains is the spatial error. OIFS's is
// not: it carries the splitting error (with the OIFS default the errors stayed
// near 1e-3 at every level, fitted rates ~1.4, 2026-10-16; see "When IMEX is
// the better choice" in docs/oifs_implementation.md). Theory, Taylor-Hood:
// |u - u_h|_H1 and
// ||p - p_h|| at min(k_u, k_p + 1), ||u - u_h|| one order higher; checked
// with grad-div at its default and switched off.

#include <gtest/gtest.h>

#include "verify_util.hpp"

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
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

/// Kovasznay's solution in box coordinates (x - 0.5, y - 0.5).
struct Kovasznay
{
   std::unique_ptr<VectorFunctionCoefficient> u;    ///< Velocity.
   std::unique_ptr<MatrixFunctionCoefficient> grad; ///< du_i/dx_j.
   std::unique_ptr<FunctionCoefficient> p;          ///< Pressure.

   explicit Kovasznay(double re)
   {
      const double lam = 0.5 * re - std::sqrt(0.25 * re * re + 4.0 * M_PI * M_PI);
      const double k = 2.0 * M_PI;
      std::function<void(const Vector&, Vector&)> vel =
         [lam, k](const Vector & x, Vector & v)
      {
         const double e = std::exp(lam * (x(0) - 0.5));
         const double y = x(1) - 0.5;
         v(0) = 1.0 - e * std::cos(k * y);
         v(1) = lam / k * e * std::sin(k * y);
      };
      std::function<void(const Vector&, DenseMatrix&)> dvel =
         [lam, k](const Vector & x, DenseMatrix & g)
      {
         const double e = std::exp(lam * (x(0) - 0.5));
         const double y = x(1) - 0.5;
         g(0, 0) = -lam * e * std::cos(k * y);
         g(0, 1) = k * e * std::sin(k * y);
         g(1, 0) = lam * lam / k * e * std::sin(k * y);
         g(1, 1) = lam * e * std::cos(k * y);
      };
      std::function<double(const Vector&)> pres = [lam](const Vector & x)
      {
         return 0.5 * (1.0 - std::exp(2.0 * lam * (x(0) - 0.5)));
      };
      u = std::make_unique<VectorFunctionCoefficient>(2, vel);
      grad = std::make_unique<MatrixFunctionCoefficient>(2, dvel);
      p = std::make_unique<FunctionCoefficient>(pres);
   }
};

/// The level's deck: Re, the box, the march's first step and end time, IMEX
/// convection (see above), the solver tolerances; everything else is the
/// case-level default.
std::string Deck(const YAML::Node& b, int n, const verify::GradDivVariant& v)
{
   std::ostringstream d;
   d << std::setprecision(17)
     << "equation: navier_stokes\n"
     << "physics:\n  nu: " << 1.0 / b["re"].as<double>() << "\n" << v.physics
     << "mesh:\n  dim: 2\n  elements: [" << 3 * n << ", " << 4 * n << "]\n"
     << "  lengths: [1.5, 2.0]\n  periodic: [false, false]\n"
     << "time:\n  dt: " << b["dt"].as<double>() << "\n  t_final: "
     << b["t_max"].as<double>() << "\n  convection: imex\n"
     << "solver:\n  rtol: " << b["rtol"].as<double>() << "\n  atol: "
     << b["atol"].as<double>() << "\n";
   return d.str();
}

} // namespace

TEST(VerifyKovasznay, K1_SteadyNavierStokes2D)
{
   const YAML::Node all = verify::Baselines();
   const YAML::Node b = all["kovasznay"];
   const double margin = all["rate_margin"].as<double>();
   const double floor = all["noise_floor"].as<double>();
   const double check_every = b["check_every"].as<double>();
   const double steady_tol = b["steady_tol"].as<double>();
   const std::vector<int> levels = b["levels"].as<std::vector<int>>();
   const Kovasznay exact(b["re"].as<double>());
   verify::Record rec("kovasznay");

   for (const verify::GradDivVariant& v : verify::GradDivVariants())
   {
      SCOPED_TRACE(v.name);
      std::vector<double> h, eu, eh1, ep;
      int ku = 0, kp = 0;
      for (std::size_t l = 0; l < levels.size(); ++l)
      {
         const incns::Parameters p =
            incns::Parameters::LoadYAMLString(Deck(b, levels[l], v));
         ku = p.order_u;
         kp = p.order_p;
         incns::RuleBook rules; // before the mesh, so it outlives it
         auto mesh = incns::MakeCaseMesh(p);
         incns::Case flow(*mesh, p);
         incns::BoundaryConditions bc(flow.Spaces().Velocity());
         for (int a : mesh->bdr_attributes) { bc.AddVelocityDirichlet(a, *exact.u); }
         flow.SetBoundaryConditions(bc);
         flow.SetInitialVelocity(*exact.u);
         const int order = 2 * p.order_u + 4;

         // March until the velocity error changes by less than steady_tol
         // (relative) over check_every time units, twice in a row.
         double next = check_every, prev = -1.0;
         int settled = 0;
         verify::Errors e;
         while (!flow.Done() && !flow.Diverged() && settled < 2)
         {
            flow.Step();
            if (flow.Time() < next - 1e-12) { continue; }
            next += check_every;
            e = verify::ComputeErrors(flow.Velocity(), flow.Pressure(),
                                      *exact.u, *exact.grad, *exact.p, rules,
                                      order);
            settled = (prev > 0.0 && std::fabs(e.u_l2 - prev) <= steady_tol * e.u_l2)
                      ? settled + 1 : 0;
            prev = e.u_l2;
         }
         EXPECT_FALSE(flow.Diverged());
         EXPECT_EQ(settled, 2) << "level " << levels[l]
                               << " did not reach a steady state by t = "
                               << flow.Time();
         std::ostringstream s;
         s << "Q" << p.order_u << "/Q" << p.order_p << " grad_div_h="
           << p.grad_div_h << " grad_div_nu=" << p.grad_div_nu << " t="
           << flow.Time() << " steps=" << flow.Integrator().StepCount();
         h.push_back(0.5 / levels[l]);
         eu.push_back(e.u_l2);
         eh1.push_back(e.u_h1);
         ep.push_back(e.p_l2);
         rec.Error(v.name, l, h.back(), "u_l2", e.u_l2, s.str());
         rec.Error(v.name, l, h.back(), "u_h1", e.u_h1, s.str());
         rec.Error(v.name, l, h.back(), "p_l2", e.p_l2, s.str());
         rec.Error(v.name, l, h.back(), "div_l2", e.div_l2, s.str());
      }
      const double energy = std::min(ku, kp + 1);
      rec.Rate(v.name, "u_l2", h, eu, energy + 1.0, margin, floor);
      rec.Rate(v.name, "u_h1", h, eh1, energy, margin, floor);
      rec.Rate(v.name, "p_l2", h, ep, energy, margin, floor);
   }
}
