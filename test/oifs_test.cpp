// OIFS (time/oifs.hpp): the BDF history advected by RK4 substeps.
//  O1 the advector alone: a smooth periodic field carried by a constant wind
//     matches the exact translation (spatial accuracy of the collocated-mass,
//     dealiased advection), and its time error against a tiny-substep
//     reference falls like the 4th power of the substep (RK4);
//  O2 the point of OIFS: the periodic NSE Taylor-Green vortex (exact, nu =
//     0.01, coarse 8x8 Q3 over 2 pi) under CFL-controlled steps at CFL 2 and
//     4 -- beyond the IMEX limit (~0.7) -- runs stably in 3.5-4x fewer steps
//     than IMEX at CFL 0.5, AS ACCURATELY, with the OIFS production scheme
//     (BDF3 + the collocated mass, both automatic under OIFS; human
//     2026-10-09). Measured 2026-10-09: IMEX 0.097 (38 steps); OIFS 0.074
//     (11) / 0.072 (9); with the consistent BDF mass 0.085 / 0.079; BDF2 +
//     consistent mass was 0.164 / 0.204 (its trajectory error at steps 4-8x
//     larger);
//  O3 hanging nodes, and O4 boundary data separable in time (below).

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "operators/stokes_operator.hpp"
#include "time/oifs.hpp"
#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

incns::BoxSpec PeriodicUnitBox(int n)
{
   incns::BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {true, true, true};
   return s;
}

} // namespace

TEST(Oifs, O1_AdvectionTranslatesWithRk4TimeOrder)
{
   incns::RuleBook rules; // before the mesh (rule-keyed caches)
   Mesh serial = incns::MakeBoxMesh(PeriodicUnitBox(12));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::MixedSpaces spaces(mesh, 3, 2);
   incns::BoundaryConditions bc(spaces.Velocity()); // fully periodic
   ParFiniteElementSpace& V = spaces.Velocity();
   const double wx = 1.0, wy = 0.5, T = 0.5;

   auto field = [](const Vector & x, double t, Vector & f, double ax, double ay)
   {
      const double X = x(0) - ax * t, Y = x(1) - ay * t;
      f(0) = std::sin(2.0 * M_PI * X) * std::cos(2.0 * M_PI * Y);
      f(1) = 0.5 * std::cos(2.0 * M_PI * X) * std::sin(4.0 * M_PI * Y);
   };
   auto to_true = [&](VectorCoefficient & coef)
   {
      ParGridFunction g(&V);
      g.ProjectCoefficient(coef);
      Vector t(V.GetTrueVSize());
      g.GetTrueDofs(t);
      return t;
   };
   VectorFunctionCoefficient wind_c(2, [&](const Vector&, Vector & w)
   {
      w(0) = wx;
      w(1) = wy;
   });
   VectorFunctionCoefficient f0_c(2, [&](const Vector & x, Vector & f)
   {
      field(x, 0.0, f, wx, wy);
   });
   VectorFunctionCoefficient fT_c(2, [&](const Vector & x, Vector & f)
   {
      field(x, T, f, wx, wy);
   });
   const std::deque<Vector> wind = {to_true(wind_c)};
   const std::deque<Vector> fields = {to_true(f0_c)};

   // The mass a BDF step would use: the collocated GLL mass (OIFS's default).
   incns::StokesOperatorOptions mo;
   mo.nu = 1.0;
   mo.collocated_mass = true;
   incns::StokesOperator blocks(spaces, rules, mo);
   auto advect = [&](double sub_cfl, int& substeps)
   {
      incns::OifsAdvector adv(spaces, rules, bc, sub_cfl, blocks.Mass(),
                              blocks.MassDiagonal());
      EXPECT_TRUE(adv.MassIsDiagonal()); // conforming: pointwise inverse
      Vector phi;
      adv.AdvectWith(wind, {0.0}, {0.0, 1.0}, fields, {0.0}, T, phi);
      substeps = adv.LastSubsteps();
      return phi;
   };
   int n_ref = 0;
   const Vector ref = advect(0.02, n_ref);
   const double ref_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, ref, ref));

   // Spatial accuracy: the reference against the exact translation.
   ParGridFunction g(&V);
   g.SetFromTrueDofs(ref);
   const IntegrationRule* irs[Geometry::NumGeom] = {};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 12);
   const double l2 = g.ComputeL2Error(fT_c, irs);
   if (Mpi::Root())
   {
      mfem::out << "[oifs O1] reference (" << n_ref << " substeps) L2 error vs "
                "exact translation: " << l2 << "\n";
   }
   EXPECT_LT(l2, 2e-3);

   // Time order: errors against the reference at three substep sizes.
   std::vector<double> err, ds;
   for (double cfl : {0.8, 0.4, 0.2})
   {
      int n = 0;
      Vector phi = advect(cfl, n);
      phi -= ref;
      err.push_back(std::sqrt(InnerProduct(MPI_COMM_WORLD, phi, phi)) / ref_norm);
      ds.push_back(T / n);
   }
   for (std::size_t i = 1; i < err.size(); ++i)
   {
      const double order = std::log(err[i - 1] / err[i]) / std::log(
                              ds[i - 1] / ds[i]);
      if (Mpi::Root())
      {
         mfem::out << "[oifs O1] ds " << ds[i] << " time error " << err[i]
                   << " observed order " << order << "\n";
      }
      EXPECT_GT(order, 3.5);
   }
}

namespace
{

struct TgvRun
{
   double err = 0.0; ///< velocity L2 error vs the exact TGV at t_final
   int steps = 0;
};

TgvRun Tgv(incns::ConvectionTreatment conv, double cfl)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convection_treatment = conv;
   p.nu = 0.01;
   // No grad-div: this compares the two convection treatments where the
   // SPATIAL error of the 8x8 mesh dominates (IMEX 0.097, OIFS 0.074 / 0.072).
   // With the default gamma = 2h the spatial error falls to 0.003 and the
   // comparison becomes one of time error at 4x and 8x the step, which OIFS
   // loses by construction (0.018 / 0.045; measured 2026-10-12).
   p.grad_div_h = 0.0;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = PeriodicUnitBox(8);
   p.mesh.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   p.step_control = incns::StepControl::Cfl;
   p.cfl_target = cfl;
   p.dt = 0.25;      // the first step is capped by the target anyway
   p.t_final = 4.0;  // long enough that the x1.2 growth limit is not the story
   p.krylov_rtol = 1e-12;
   p.Normalize();
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
   flow.SetInitialVelocity(u0);
   TgvRun r;
   while (!flow.Done()) { flow.Step(); ++r.steps; }
   VectorFunctionCoefficient ue = incns::tgv2d::VelocityCoefficient(p.nu);
   ue.SetTime(flow.Time());
   const IntegrationRule* irs[Geometry::NumGeom] = {};
   irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 14);
   r.err = flow.Velocity().ComputeL2Error(ue, irs);
   return r;
}

} // namespace

TEST(Oifs, O2_TgvBeyondTheImexCflLimit)
{
   using incns::ConvectionTreatment;
   const TgvRun imex = Tgv(ConvectionTreatment::Imex, 0.5);
   const TgvRun oifs2 = Tgv(ConvectionTreatment::Oifs, 2.0);
   const TgvRun oifs4 = Tgv(ConvectionTreatment::Oifs, 4.0);
   if (Mpi::Root())
   {
      mfem::out << "[oifs O2] TGV nu 0.01, t = 4: IMEX CFL 0.5 err " << imex.err
                << " (" << imex.steps << " steps); OIFS CFL 2 err " << oifs2.err
                << " (" << oifs2.steps << "); OIFS CFL 4 err " << oifs4.err
                << " (" << oifs4.steps << ")\n";
   }
   EXPECT_LT(oifs2.err, 1.2 * imex.err);
   EXPECT_LT(oifs4.err, 1.2 * imex.err);
   EXPECT_LE(3 * oifs2.steps, imex.steps);
   EXPECT_LE(4 * oifs4.steps, imex.steps);
}


// O3 OIFS on a NONCONFORMING (AMR) mesh: the periodic translation of O1 on
// the 12x12 mesh with a band of elements refined once (hanging nodes on both
// band edges) is no less accurate than on the conforming 12x12 mesh. There
// the collocated mass P^T D P is not diagonal, so the substeps solve with it
// (CG) -- the same mass as the BDF step, as on conforming meshes, where it
// is diagonal (human 2026-10-10). Measured 2026-10-09 with the old row-sum
// lumping: 4.08e-4 vs 4.23e-4 (uniform 24x24: 2.4e-5). Guards hanging-node
// meshes, where the square-cylinder runs live.
TEST(Oifs, O3_NonconformingAdvection)
{
   if (amr_test::DebugDeviceSkipsPeriodicNcSolves())
   {
      GTEST_SKIP() << "periodic NC mesh: MFEM debug-device alias false "
                   "positive (CLAUDE.md section 6)";
   }
   const double wx = 1.0, wy = 0.5, T = 0.5;
   auto field = [](const Vector & x, double t, Vector & f, double ax, double ay)
   {
      const double X = x(0) - ax * t, Y = x(1) - ay * t;
      f(0) = std::sin(2.0 * M_PI * X) * std::cos(2.0 * M_PI * Y);
      f(1) = 0.5 * std::cos(2.0 * M_PI * X) * std::sin(4.0 * M_PI * Y);
   };
   // 0 conforming; a band refined (hanging nodes) with the collocated mass
   // 1 lumped (the default: diagonal, b = P^T d) or 2 exact (P^T D P, CG).
   double err[3] = {0.0, 0.0, 0.0};
   for (int mode : {0, 1, 2})
   {
      incns::RuleBook rules;
      Mesh serial = incns::MakeBoxMesh(PeriodicUnitBox(12));
      if (mode >= 1)
      {
         serial.EnsureNCMesh();
         Array<int> els;
         for (int e = 0; e < serial.GetNE(); ++e)
         {
            Vector c(2);
            serial.GetElementCenter(e, c);
            if (c(0) > 0.3 && c(0) < 0.7) { els.Append(e); }
         }
         serial.GeneralRefinement(els, 0, 1);
      }
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, 3, 2);
      incns::BoundaryConditions bc(spaces.Velocity());
      ParFiniteElementSpace& V = spaces.Velocity();
      auto to_true = [&](VectorCoefficient & coef)
      {
         ParGridFunction g(&V);
         g.ProjectCoefficient(coef);
         Vector t(V.GetTrueVSize());
         g.GetTrueDofs(t);
         return t;
      };
      VectorFunctionCoefficient wind_c(2, [&](const Vector&, Vector & w) { w(0) = wx; w(1) = wy; });
      VectorFunctionCoefficient f0_c(2, [&](const Vector & x, Vector & f) { field(x, 0.0, f, wx, wy); });
      VectorFunctionCoefficient fT_c(2, [&](const Vector & x, Vector & f) { field(x, T, f, wx, wy); });
      const std::deque<Vector> wind = {to_true(wind_c)};
      const std::deque<Vector> fields = {to_true(f0_c)};
      // The BDF step's collocated mass: diagonal on the conforming mesh and,
      // lumped at the hanging nodes (the default), on the refined one; the
      // exact P^T D P is not, and the substeps then solve with it (CG).
      incns::StokesOperatorOptions mo;
      mo.nu = 1.0;
      mo.collocated_mass = true;
      mo.lumped_hanging_mass = (mode != 2);
      incns::StokesOperator blocks(spaces, rules, mo);
      incns::OifsAdvector adv(spaces, rules, bc, 0.05, blocks.Mass(),
                              blocks.MassDiagonal(),
                              nullptr, blocks.MassDiagonalInverse());
      EXPECT_EQ(adv.MassIsDiagonal(), mode != 2);
      Vector phi;
      adv.AdvectWith(wind, {0.0}, {0.0, 1.0}, fields, {0.0}, T, phi);
      if (mode >= 1 && Mpi::Root())
      {
         mfem::out << "[oifs O3] hanging nodes, mass "
                   << (mode == 1 ? "lumped" : "exact") << ": "
                   << adv.MassSolveIterations() << " mass-CG iterations over "
                   << 4 * adv.LastSubsteps() << " RK stages\n";
      }
      if (mode == 1) { EXPECT_EQ(adv.MassSolveIterations(), 0); }
      if (mode == 2) { EXPECT_GT(adv.MassSolveIterations(), 0); }
      ParGridFunction g(&V);
      g.SetFromTrueDofs(phi);
      const IntegrationRule* irs[Geometry::NumGeom] = {};
      irs[Geometry::SQUARE] = &IntRules.Get(Geometry::SQUARE, 12);
      const double l2 = g.ComputeL2Error(fT_c, irs);
      err[mode] = l2;
   }
   if (Mpi::Root())
   {
      mfem::out << "[oifs O3] translation L2 error: conforming " << err[0]
                << ", band refined (hanging nodes) with the mass lumped "
                << err[1] << ", exact " << err[2] << "\n";
   }
   EXPECT_LT(err[1], 1.1 * err[0]);
   EXPECT_LT(err[2], 1.1 * err[0]);
}

// O4 separable boundary data: the DFG 2D channel's parabolic inflow (walls
// and cylinder no-slip), the history carried by that profile over t in
// [1, 1.05]. The deck's coefficient is separable (u_D = s(t) g): the
// substeps scale its shapes, projected once. The same data as a plain
// function coefficient is projected at every stage (host-only, only the
// imposed values gathered). The two advected fields agree to roundoff under
// the 2D-3 sine (the projection averages s g where the shapes are averaged,
// then scaled), and bit for bit with a constant time profile (s = 1).
TEST(Oifs, O4_SeparableInflowMatchesTheProjection)
{
   for (const bool sine : {true, false})
   {
      const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                     std::string("equation: navier_stokes\nphysics:\n  nu: 0.001\n"
                                        "mesh:\n  dim: 2\n  geometry: cylinder_channel\n"
                                        "boundary_conditions:\n"
                                        "  - {select: [inflow], type: velocity, profile: parabolic, "
                                        "u_max: 1.5, height: 0.41") +
                                     (sine ? ", time_profile: sine, time_scale: 8.0" : "") +
                                     "}\n"
                                     "  - {select: [walls, cylinder], type: no_slip}\n"
                                     "  - {select: [outflow], type: outflow}\n");
      incns::RuleBook rules; // before the mesh (rule-keyed caches)
      auto mesh = incns::MakeCaseMesh(p);
      incns::MixedSpaces spaces(*mesh, 3, 2);
      ParFiniteElementSpace& V = spaces.Velocity();
      incns::BoundaryConditions bc_sep(V);
      incns::DeckBoundaryConditions deck(p, *mesh);
      deck.Apply(bc_sep, nullptr, true);
      // The deck's formula as a plain coefficient, the entries in the deck's
      // order (the last wins where they meet).
      VectorFunctionCoefficient inflow(2, [sine](const Vector & x, double t,
                                       Vector & u)
      {
         const double sc = sine ? std::sin(M_PI * t / 8.0) : 1.0;
         const double h = 0.41, y = x(1);
         u = 0.0;
         u(0) = sc * 4.0 * 1.5 * y * (h - y) / (h * h);
      });
      incns::BoundaryConditions bc_plain(V);
      bc_plain.AddVelocityDirichlet(incns::NamedBoundaryAttribute(p, *mesh, "inflow"),
                                    inflow);
      bc_plain.AddNoSlip(incns::NamedBoundaryAttribute(p, *mesh, "walls"));
      bc_plain.AddNoSlip(incns::NamedBoundaryAttribute(p, *mesh, "cylinder"));
      bc_plain.AddOutflow(incns::NamedBoundaryAttribute(p, *mesh, "outflow"));
      ASSERT_TRUE(bc_sep.DirichletIsSeparable());
      ASSERT_FALSE(bc_plain.DirichletIsSeparable());

      VectorFunctionCoefficient profile(2, [](const Vector & x, Vector & w)
      {
         const double h = 0.41, y = x(1);
         w(0) = 4.0 * 1.5 * y * (h - y) / (h * h);
         w(1) = 0.0;
      });
      ParGridFunction g(&V);
      g.ProjectCoefficient(profile);
      Vector w0(V.GetTrueVSize());
      g.GetTrueDofs(w0);
      const std::deque<Vector> wind = {w0};
      incns::StokesOperatorOptions mo;
      mo.nu = 1.0;
      mo.collocated_mass = true;
      incns::StokesOperator blocks(spaces, rules, mo);
      Vector phi[2];
      int imposed[2] = {0, 0};
      for (int k : {0, 1})
      {
         incns::OifsAdvector adv(spaces, rules, k == 0 ? bc_sep : bc_plain, 0.5,
                                 blocks.Mass(), blocks.MassDiagonal());
         adv.AdvectWith(wind, {1.0}, {0.0, 1.0}, wind, {1.0}, 1.05, phi[k]);
         imposed[k] = adv.ImposedDofs().Size();
      }
      EXPECT_EQ(imposed[0], imposed[1]);
      phi[1] -= phi[0];
      double v[2] = {phi[1].Normlinf(), phi[0].Normlinf()};
      MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         mfem::out << "[oifs O4] " << (sine ? "sine" : "constant")
                   << " inflow: max |separable - projected| " << v[0]
                   << " (max |phi| " << v[1] << ")\n";
      }
      EXPECT_GT(v[1], 0.5);
      if (sine) { EXPECT_LE(v[0], 1e-13 * v[1]); }
      else { EXPECT_EQ(v[0], 0.0); }
   }
}
