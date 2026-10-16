// Verification tier (label `verify`): temporal order of the Navier-Stokes
// time schemes in 3D, through incns::Case.
//  T1 the case-level default scheme (whatever it is: today IMEX BDF2/EXT2);
//  T2 IMEX BDF3/EXT3;
//  T3 OIFS at its default BDF order (today BDF3).
// Manufactured solution u = G(t) (y^2, z^2, x^2), p = G(t) (x + y + z - 3/2),
// G = 1 + sin(2t)/2, on the unit cube with exact (time-dependent) velocity on
// every face. The spatial fields lie in the default discrete spaces and G is
// not a polynomial, so every term of the time error (BDF, EXT, the OIFS
// substeps) is present. Errors are measured against a run of the same scheme
// at a much smaller step (self-convergence: the spatial error cancels, which
// the collocated mass of OIFS needs), and every run starts from the exact
// history at t = 0, -dt, ... (the trapezoidal starter's O(dt^2) local error
// would otherwise cap BDF3 at order 2: docs/developing.md §5.2). Theory: the
// BDF order, and under IMEX also the extrapolation order (the smaller one).
// Pass: the rate fitted over all steps >= theory - rate_margin.

#include <gtest/gtest.h>

#include "verify_util.hpp"

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "time/integrator_state.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

double G(double t) { return 1.0 + 0.5 * std::sin(2.0 * t); }
double Gp(double t) { return std::cos(2.0 * t); }

/// A time scheme under test: deck lines under `time:`.
struct Scheme
{
   const char* name; ///< CSV label.
   const char* time; ///< Lines added under the deck's `time:` key.
};

const Scheme kSchemes[] =
{
   {"default", ""},
   {"imex_bdf3_ext3", "  order: 3\n  ext_order: 3\n"},
   {"oifs", "  convection: oifs\n"},
};

std::string Deck(const YAML::Node& b, const Scheme& s, double dt)
{
   const int n = b["elements"].as<int>();
   std::ostringstream d;
   d << std::setprecision(17)
     << "equation: navier_stokes\n"
     << "physics:\n  nu: " << b["nu"].as<double>() << "\n"
     << "mesh:\n  dim: 3\n  elements: [" << n << ", " << n << ", " << n
     << "]\n  lengths: [1.0, 1.0, 1.0]\n  periodic: [false, false, false]\n"
     << "time:\n  step_control: fixed\n  dt: " << dt << "\n  t_final: "
     << b["t_final"].as<double>() << "\n" << s.time
     << "solver:\n  rtol: " << b["rtol"].as<double>() << "\n  atol: "
     << b["atol"].as<double>() << "\n";
   return d.str();
}

/// One run's final velocity (true dofs) and its L2 error against the exact
/// solution; @p theory receives the scheme's design order.
struct Result
{
   Vector u_true;        ///< Final velocity, true dofs.
   double u_err = 0.0;   ///< ||u_h - u|| at t_final.
   double theory = 0.0;  ///< Design order of the scheme.
   std::string settings; ///< Resolved scheme (provenance).
};

Result RunScheme(const YAML::Node& b, const Scheme& s, double dt)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(Deck(b, s, dt));
   const double nu = p.nu;
   VectorFunctionCoefficient u_exact(3, [](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x(1) * x(1);
      v(1) = G(t) * x(2) * x(2);
      v(2) = G(t) * x(0) * x(0);
   });
   // f = u_t + (u.grad)u - nu lap u + grad p, with lap(y^2, z^2, x^2) = 2,
   // grad(x + y + z) = 1 and (u_s.grad)u_s = 2 (y z^2, z x^2, x y^2).
   auto rhs = [nu](const Vector & x, double t, Vector & f)
   {
      const double g = G(t), gp = Gp(t);
      f(0) = gp * x(1) * x(1) + 2.0 * g * g * x(1) * x(2) * x(2);
      f(1) = gp * x(2) * x(2) + 2.0 * g * g * x(2) * x(0) * x(0);
      f(2) = gp * x(0) * x(0) + 2.0 * g * g * x(0) * x(1) * x(1);
      for (int i = 0; i < 3; ++i) { f(i) += g * (1.0 - 2.0 * nu); }
   };
   VectorFunctionCoefficient forcing(3, rhs);

   incns::RuleBook rules; // before the mesh, so it outlives it
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   for (int a : mesh->bdr_attributes) { bc.AddVelocityDirichlet(a, u_exact); }
   flow.SetBoundaryConditions(bc);
   flow.SetForcing(forcing);
   u_exact.SetTime(0.0);
   flow.SetInitialVelocity(u_exact);

   // The exact history: as many levels as the BDF and EXT orders need.
   const bool oifs = p.Convection() == incns::ConvectionTreatment::Oifs;
   const int bdf = p.BdfOrder();
   const int levels = std::max(bdf, p.ext_order);
   ParFiniteElementSpace& V = flow.Spaces().Velocity();
   incns::IntegratorState st;
   for (int j = 0; j < levels; ++j)
   {
      const double tj = -j * dt;
      u_exact.SetTime(tj);
      ParGridFunction g(&V);
      g.ProjectCoefficient(u_exact);
      Vector tv(V.GetTrueVSize());
      g.GetTrueDofs(tv);
      st.u_hist.push_back(tv);
      st.times.push_back(tj);
   }
   st.completed_steps = levels - 1; // the first step is already BDF-k
   st.next_dt = dt;
   flow.Integrator().ImportState(st);

   while (!flow.Done() && !flow.Diverged()) { flow.Step(); }
   EXPECT_FALSE(flow.Diverged());
   EXPECT_NEAR(flow.Time(), p.t_final, 1e-12);

   Result r;
   r.theory = oifs ? bdf : std::min(bdf, p.ext_order);
   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::CUBE] = &rules.Get(Geometry::CUBE, 2 * p.order_u + 4);
   u_exact.SetTime(flow.Time());
   r.u_err = flow.Velocity().ComputeL2Error(u_exact, irs);
   r.u_true.SetSize(V.GetTrueVSize());
   flow.Velocity().GetTrueDofs(r.u_true);
   std::ostringstream s;
   s << (oifs ? "OIFS" : "IMEX") << " BDF" << bdf << "/EXT" << p.ext_order
     << " Q" << p.order_u << "/Q" << p.order_p << " mass="
     << (p.CollocatedMass() ? "collocated" : "consistent") << " grad_div_h="
     << p.grad_div_h;
   r.settings = s.str();
   return r;
}

} // namespace

TEST(VerifyTemporal, T1_T2_T3_Order3D)
{
   const YAML::Node all = verify::Baselines();
   const YAML::Node b = all["temporal_3d"];
   const double margin = all["rate_margin"].as<double>();
   const double floor = all["noise_floor"].as<double>();
   const std::vector<double> dts = b["dts"].as<std::vector<double>>();
   verify::Record rec("temporal_3d");
   for (const Scheme& s : kSchemes)
   {
      SCOPED_TRACE(s.name);
      const Result ref = RunScheme(b, s, b["dt_ref"].as<double>());
      const double ref_norm =
         std::sqrt(InnerProduct(MPI_COMM_WORLD, ref.u_true, ref.u_true));
      rec.Error(s.name, -1, b["dt_ref"].as<double>(), "u_l2_vs_exact",
                ref.u_err, ref.settings);
      std::vector<double> e;
      for (std::size_t l = 0; l < dts.size(); ++l)
      {
         Result r = RunScheme(b, s, dts[l]);
         r.u_true -= ref.u_true;
         e.push_back(std::sqrt(InnerProduct(MPI_COMM_WORLD, r.u_true,
                                            r.u_true)) / ref_norm);
         rec.Error(s.name, l, dts[l], "u_rel_vs_ref", e.back(), r.settings);
         rec.Error(s.name, l, dts[l], "u_l2_vs_exact", r.u_err, r.settings);
      }
      rec.Rate(s.name, "u_rel_vs_ref", dts, e, ref.theory, margin, floor);
   }
}
