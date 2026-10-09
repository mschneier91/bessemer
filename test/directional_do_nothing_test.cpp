// Directional do-nothing (DDN) outflow condition, Braack & Mucha, J. Comput.
// Math. 32 (2014) 507-521: T(u,p) n - 1/2 (u.n)_- u = 0 on the outflow,
// added to the IMEX convection (operators/directional_do_nothing).
//  D1 the boundary term's value and outward normal: for the polynomial
//     u = (1 - 2y, xy) on [0,1]^2 (element edges at y = 1/2, where u.n
//     changes sign on x = 0, 1), -1/2 int (u.n)_- u.phi matches the integrals
//     by hand face by face (x = 1: -1/12 and 5/48 for phi = e_x, e_y; x = 0:
//     +1/12 and 0; pure outflow faces: 0);
//  D2 the energy identity the condition exists for, 2D and 3D: with every
//     face marked, <N(u) + D(u), u> = 1/2 int (u.n)_+ |u|^2 - 1/2 int div(u)
//     |u|^2 -- the backflow part of the convective boundary flux is gone --
//     while <N(u), u> keeps the full 1/2 int (u.n) |u|^2 (exact quadrature,
//     so to roundoff);
//  D3 the paper's steady test (its section 5.1: unit square, do-nothing at
//     x = 0, no-slip elsewhere, f = (sin x + sin y, 0), nu = 0.05) marched to
//     steady state through the Case: the inflow flux j1 = int (u.n)_- and the
//     outflow energy flux j2 = int (u.n)_+ |u|^2 on the outflow match the
//     paper's Table 5.1 for BOTH conditions (DDN: -4.269e-2 / 5.318e-4,
//     classical: -4.498e-2 / 6.109e-4), and DDN reduces both;
//  D4 D3 with OIFS convection (BDF3, CFL 2), where the term is explicit and
//     extrapolated in the BDF step: the same Table 5.1 values for both
//     conditions (measured 2026-10-09: directional j1 -4.2705e-2, j2
//     5.3215e-4 -- IMEX's to 5e-5; classical = IMEX's to 2e-5). Stable to
//     CFL 8 here (j1 within 0.5%). Placed INSIDE the substeps instead
//     (linearized, energy-exact) it gave j1 -4.171e-2 at CFL 2 (-2.3%),
//     converging only as CFL -> 0.25: stiff at the backflow nodes.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/convection.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/case.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

using namespace mfem;

namespace
{

incns::BoxSpec UnitBox(int dim, int n)
{
   incns::BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// The boundary attribute of the box face {x_axis = value} (geometric lookup,
// reduced over ranks).
int FaceAttribute(ParMesh& mesh, int axis, double value)
{
   int attr = -1;
   Array<int> v;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      mesh.GetBdrElementVertices(be, v);
      bool on = true;
      for (int k : v) { on = on && std::abs(mesh.GetVertex(k)[axis] - value) < 1e-12; }
      if (on) { attr = mesh.GetBdrAttribute(be); break; }
   }
   int out = attr;
   MPI_Allreduce(&attr, &out, 1, MPI_INT, MPI_MAX, mesh.GetComm());
   return out;
}

// Integral over the marked boundary faces of g(x, n) with the outward unit
// normal n (high-order rule; reduced over ranks).
double BoundaryIntegral(ParMesh& mesh, const Array<int>& attrs,
                        const std::function<double(const Vector&, const Vector&)>& g)
{
   const int dim = mesh.Dimension();
   const Geometry::Type face = (dim == 3) ? Geometry::SQUARE : Geometry::SEGMENT;
   const IntegrationRule& ir = IntRules.Get(face, 16);
   double s = 0.0;
   Vector x(dim), nor(dim);
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      if (attrs.Find(mesh.GetBdrAttribute(be)) < 0) { continue; }
      FaceElementTransformations* tr = mesh.GetBdrFaceTransformations(be);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         tr->SetAllIntPoints(&ip);
         CalcOrtho(tr->Jacobian(), nor);
         const double len = nor.Norml2();
         nor /= len;
         tr->Transform(ip, x);
         s += ip.weight * len * g(x, nor);
      }
   }
   double out = s;
   MPI_Allreduce(&s, &out, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return out;
}

double DomainIntegral(ParMesh& mesh,
                      const std::function<double(const Vector&)>& g)
{
   const int dim = mesh.Dimension();
   const IntegrationRule& ir =
      IntRules.Get(dim == 3 ? Geometry::CUBE : Geometry::SQUARE, 16);
   double s = 0.0;
   Vector x(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         T.Transform(ip, x);
         s += ip.weight * T.Weight() * g(x);
      }
   }
   double out = s;
   MPI_Allreduce(&s, &out, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return out;
}

} // namespace

TEST(DirectionalDoNothing, D1_ValueAndOutwardNormal)
{
   incns::RuleBook rules; // before the mesh (rule-keyed caches)
   Mesh serial = incns::MakeBoxMesh(UnitBox(2, 4));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::MixedSpaces spaces(mesh, 3, 2);
   ParFiniteElementSpace& V = spaces.Velocity();

   VectorFunctionCoefficient uc(2, [](const Vector & x, Vector & u)
   {
      u(0) = 1.0 - 2.0 * x(1);
      u(1) = x(0) * x(1);
   });
   ParGridFunction ug(&V);
   ug.ProjectCoefficient(uc);
   Vector u(V.GetTrueVSize());
   ug.GetTrueDofs(u);
   auto unit = [&](int c)
   {
      VectorFunctionCoefficient ec(2, [c](const Vector&, Vector & e)
      {
         e = 0.0;
         e(c) = 1.0;
      });
      ParGridFunction g(&V);
      g.ProjectCoefficient(ec);
      Vector t(V.GetTrueVSize());
      g.GetTrueDofs(t);
      return t;
   };
   const Vector ex = unit(0), ey = unit(1);

   incns::Convection plain(spaces, rules);
   Vector n_only(V.GetTrueVSize());
   plain.Mult(u, n_only);
   // The term alone: (N + D)(u) - N(u), tested against e_x and e_y.
   auto term = [&](int axis, double value, double & dx, double & dy)
   {
      incns::Convection ddn(spaces, rules);
      Array<int> attrs({FaceAttribute(mesh, axis, value)});
      ddn.EnableDirectionalDoNothing(attrs);
      Vector y(V.GetTrueVSize());
      ddn.Mult(u, y);
      y -= n_only;
      dx = InnerProduct(MPI_COMM_WORLD, y, ex);
      dy = InnerProduct(MPI_COMM_WORLD, y, ey);
   };
   double dx, dy;
   term(0, 1.0, dx, dy); // x = 1: n = +e_x, backflow on y > 1/2
   EXPECT_NEAR(dx, -1.0 / 12.0, 1e-13);
   EXPECT_NEAR(dy, 5.0 / 48.0, 1e-13);
   term(0, 0.0, dx, dy); // x = 0: n = -e_x, backflow on y < 1/2
   EXPECT_NEAR(dx, 1.0 / 12.0, 1e-13);
   EXPECT_NEAR(dy, 0.0, 1e-13);
   term(1, 1.0, dx, dy); // y = 1: u.n = x >= 0, pure outflow
   EXPECT_NEAR(dx, 0.0, 1e-14);
   EXPECT_NEAR(dy, 0.0, 1e-14);
}

TEST(DirectionalDoNothing, D2_CancelsTheBackflowEnergyFlux)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      incns::RuleBook rules;
      Mesh serial = incns::MakeBoxMesh(UnitBox(dim, dim == 2 ? 3 : 2));
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, 3, 2);
      ParFiniteElementSpace& V = spaces.Velocity();
      // u.n has one sign per face: inflow through x = 0, y = 0 (and z = 0),
      // outflow elsewhere -- so (u.n)_- is polynomial on every face and the
      // 3k rules are exact.
      auto ufun = [dim](const Vector & x, Vector & u)
      {
         u(0) = 1.0 + x(0);
         u(1) = 0.5 + x(0) * x(1);
         if (dim == 3) { u(2) = 0.25 + x(2) * x(0); }
      };
      auto divu = [dim](const Vector & x)
      {
         return 1.0 + x(0) + (dim == 3 ? x(0) : 0.0);
      };
      VectorFunctionCoefficient uc(dim, ufun);
      ParGridFunction ug(&V);
      ug.ProjectCoefficient(uc);
      Vector u(V.GetTrueVSize());
      ug.GetTrueDofs(u);

      Array<int> all;
      for (int a : mesh.bdr_attributes) { all.Append(a); }
      incns::Convection plain(spaces, rules), ddn(spaces, rules);
      ddn.EnableDirectionalDoNothing(all);
      Vector y(V.GetTrueVSize());
      plain.Mult(u, y);
      const double e_plain = InnerProduct(MPI_COMM_WORLD, y, u);
      ddn.Mult(u, y);
      const double e_ddn = InnerProduct(MPI_COMM_WORLD, y, u);

      Vector uv(dim);
      auto flux = [&](bool positive_part)
      {
         return BoundaryIntegral(mesh, all, [&](const Vector & x, const Vector & n)
         {
            ufun(x, uv);
            const double un = uv * n;
            return 0.5 * (positive_part ? std::max(un, 0.0) : un) * (uv * uv);
         });
      };
      const double vol = DomainIntegral(mesh, [&](const Vector & x)
      {
         ufun(x, uv);
         return 0.5 * divu(x) * (uv * uv);
      });
      const double full = flux(false), outflow_part = flux(true);
      if (Mpi::Root())
      {
         mfem::out << "[ddn energy] dim " << dim << ": <N u,u> = " << e_plain
                   << " (1/2 int (u.n)|u|^2 - 1/2 int div u |u|^2 = "
                   << full - vol << "), <(N+D) u,u> = " << e_ddn
                   << " (outflow part only: " << outflow_part - vol << ")\n";
      }
      EXPECT_NEAR(e_plain, full - vol, 1e-11 * std::abs(full));
      EXPECT_NEAR(e_ddn, outflow_part - vol, 1e-11 * std::abs(full));
      // The test has real backflow: 1/2 int (u.n)_- |u|^2 < 0.
      EXPECT_GT(outflow_part - full, 0.1 * std::abs(full))
            << "the inflow faces must carry a real backflow flux";
   }
}

namespace
{

struct Fluxes
{
   double j1 = 0.0; ///< int_{S1} (u.n)_-
   double j2 = 0.0; ///< int_{S1} (u.n)_+ |u|^2
};

// Braack & Mucha section 5.1, marched to steady state.
Fluxes PaperSquare(incns::OutflowCondition outflow, int n, double t_final,
                   incns::ConvectionTreatment conv =
                      incns::ConvectionTreatment::Imex, double cfl = 0.5)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.outflow = outflow;
   p.convection_treatment = conv;
   p.cfl_target = cfl;
   p.nu = 0.05;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = UnitBox(2, n);
   p.dt = 0.05;
   p.t_final = t_final;
   p.krylov_rtol = 1e-10;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   const int s1 = FaceAttribute(*mesh, 0, 0.0); // the do-nothing side x = 0
   for (int a : mesh->bdr_attributes)
   {
      if (a == s1) { bc.AddOutflow(a); }
      else { bc.AddNoSlip(a); }
   }
   flow.SetBoundaryConditions(bc);
   VectorFunctionCoefficient f(2, [](const Vector & x, Vector & v)
   {
      v(0) = std::sin(x(0)) + std::sin(x(1));
      v(1) = 0.0;
   });
   flow.SetForcing(f);
   flow.Run();

   const ParGridFunction& u = flow.Velocity();
   u.HostRead(); // evaluated on the host below
   Vector uv(2);
   Fluxes out;
   // u at a boundary point: evaluated in the adjacent element.
   const IntegrationRule& ir = IntRules.Get(Geometry::SEGMENT, 16);
   double loc[2] = {0.0, 0.0};
   Vector nor(2);
   for (int be = 0; be < mesh->GetNBE(); ++be)
   {
      if (mesh->GetBdrAttribute(be) != s1) { continue; }
      FaceElementTransformations* tr = mesh->GetBdrFaceTransformations(be);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         tr->SetAllIntPoints(&ip);
         CalcOrtho(tr->Jacobian(), nor);
         const double len = nor.Norml2();
         u.GetVectorValue(*tr->Elem1, tr->GetElement1IntPoint(), uv);
         const double un = (uv * nor) / len;
         loc[0] += ip.weight * len * std::min(un, 0.0);
         loc[1] += ip.weight * len * std::max(un, 0.0) * (uv * uv);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, loc, 2, MPI_DOUBLE, MPI_SUM, mesh->GetComm());
   out.j1 = loc[0];
   out.j2 = loc[1];
   return out;
}

} // namespace

TEST(DirectionalDoNothing, D3_ReproducesBraackMuchaTable51)
{
   const int n = 8;
   const double t_final = 15.0;
   const Fluxes cdn = PaperSquare(incns::OutflowCondition::Classical, n, t_final);
   const Fluxes ddn = PaperSquare(incns::OutflowCondition::Directional, n,
                                  t_final);
   if (Mpi::Root())
   {
      mfem::out << "[ddn paper 5.1] nu 0.05: classical j1 " << cdn.j1 << " j2 "
                << cdn.j2 << " (paper -4.498e-2, 6.109e-4); directional j1 "
                << ddn.j1 << " j2 " << ddn.j2 << " (paper -4.269e-2, 5.318e-4)\n";
   }
   // Measured 2026-10-08 (8x8 Q3/Q2, steady by t = 15): within 0.06% of the
   // paper's values for both conditions, which differ by 5% (j1) and 13%
   // (j2) -- so 0.5% tells them apart.
   EXPECT_NEAR(cdn.j1, -4.498e-2, 5e-3 * 4.498e-2);
   EXPECT_NEAR(cdn.j2, 6.109e-4, 5e-3 * 6.109e-4);
   EXPECT_NEAR(ddn.j1, -4.269e-2, 5e-3 * 4.269e-2);
   EXPECT_NEAR(ddn.j2, 5.318e-4, 5e-3 * 5.318e-4);
   EXPECT_GT(ddn.j1, cdn.j1); // less inflow
   EXPECT_LT(ddn.j2, cdn.j2);
}

TEST(DirectionalDoNothing, D4_OifsReproducesBraackMuchaTable51)
{
   const int n = 8;
   const double t_final = 15.0;
   using incns::ConvectionTreatment;
   const Fluxes cdn = PaperSquare(incns::OutflowCondition::Classical, n, t_final,
                                  ConvectionTreatment::Oifs, 2.0);
   const Fluxes ddn = PaperSquare(incns::OutflowCondition::Directional, n,
                                  t_final, ConvectionTreatment::Oifs, 2.0);
   if (Mpi::Root())
   {
      mfem::out << "[ddn paper 5.1, OIFS CFL 2] classical j1 " << cdn.j1
                << " j2 " << cdn.j2 << " (paper -4.498e-2, 6.109e-4); "
                << "directional j1 " << ddn.j1 << " j2 " << ddn.j2
                << " (paper -4.269e-2, 5.318e-4)\n";
   }
   EXPECT_NEAR(cdn.j1, -4.498e-2, 5e-3 * 4.498e-2);
   EXPECT_NEAR(cdn.j2, 6.109e-4, 5e-3 * 6.109e-4);
   EXPECT_NEAR(ddn.j1, -4.269e-2, 5e-3 * 4.269e-2);
   EXPECT_NEAR(ddn.j2, 5.318e-4, 5e-3 * 5.318e-4);
   EXPECT_GT(ddn.j1, cdn.j1); // less inflow
   EXPECT_LT(ddn.j2, cdn.j2);
}
