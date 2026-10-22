// Robin boundaries (BoundaryConditions::AddRobin): the traction on the Robin
// faces is sigma(u, p) n = g + alpha (w - u), with alpha (u, v) in the
// momentum block and alpha (w, v) + g on the right-hand side. The interface
// condition of the Robin-Neumann FSI coupling (docs/fsi.md).
//
//  R1. Unsteady channel, x-periodic, no-slip bottom, Robin top with a
//      time-dependent data velocity w(t) and no load: u = G(t) phi(y) with
//      phi quadratic, p = 0, w chosen so that nu u_y(1) = alpha (w - u(1)).
//      In the discrete spaces (Q2/Q1) and quadratic in t: exact at every
//      BDF2 step. The history is seeded exactly: the trapezoidal starter
//      takes the Robin term at t^1 alone (not split like its viscous term),
//      an O(dt^2) local error in that step.
//  R2. The same steady (G = 1) with w = 0 and the load g = (nu phi'(1) +
//      alpha phi(1)) on the top: exact; and MomentumResidual() at the Robin
//      rows is the boundary's load g - alpha M_G u (the residual without the
//      Robin terms: what the solid of a Robin-Neumann coupling is given).
//  R3. alpha -> infinity approaches the Dirichlet condition u = w: the
//      error against the no-slip solution falls like 1/alpha.
//
// Boundary attributes are found geometrically.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesTimeIntegrator;
using incns::TimeIntegratorOptions;

namespace
{
constexpr double kLen = 2.0; // channel length (x); height 1

double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }

int FindBoundaryAttrAt(ParMesh& mesh, int ci, double value)
{
   int attr = 0;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      Vector c;
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      if (std::abs(c[ci] - value) < 1e-10) { attr = mesh.GetBdrAttribute(be); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MFEM_VERIFY(attr > 0, "no boundary attribute found at the requested plane");
   return attr;
}

void ElevatedRules(const RuleBook& rules, const IntegrationRule* irs[])
{
   for (int g = 0; g < Geometry::NumGeom; ++g) { irs[g] = nullptr; }
   irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, 8);
}

/// The channel: x-periodic, Q2/Q1.
struct Channel
{
   Mesh serial;
   ParMesh mesh;
   MixedSpaces spaces;
   RuleBook rules;
   int bottom = 0;
   int top = 0;

   static Mesh Make()
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {4, 3, 0};
      s.lengths = {kLen, 1.0, 1.0};
      s.periodic = {true, false, false};
      return MakeBoxMesh(s);
   }

   Channel()
      : serial(Make()), mesh(MPI_COMM_WORLD, serial), spaces(mesh, 2, 1)
   {
      bottom = FindBoundaryAttrAt(mesh, 1, 0.0);
      top = FindBoundaryAttrAt(mesh, 1, 1.0);
   }
};

TimeIntegratorOptions Options(double nu)
{
   TimeIntegratorOptions opts;
   opts.nu = nu;
   opts.dt = 0.02;
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   return opts;
}
} // namespace

// --- R1. Time-dependent Robin data velocity --------------------------------
TEST(Robin, R1_UnsteadyDataVelocity)
{
   const double nu = 0.7, alpha = 3.0;
   // phi = y (c - y) with phi(0) = 0 and an arbitrary c; then w makes the
   // Robin condition hold: nu phi'(1) = alpha (W - phi(1)).
   const double c = 1.6;
   const double phi1 = c - 1.0, dphi1 = c - 2.0;
   const double W = phi1 + nu * dphi1 / alpha;
   VectorFunctionCoefficient u_exact(2, [c](const Vector & x, double t, Vector & v)
   {
      v(0) = G(t) * x[1] * (c - x[1]);
      v(1) = 0.0;
   });
   VectorFunctionCoefficient w(2, [W](const Vector&, double t, Vector & v)
   {
      v(0) = G(t) * W;
      v(1) = 0.0;
   });
   // u_t - nu u_yy = G' phi + 2 nu G.
   VectorFunctionCoefficient forcing(2, [nu, c](const Vector & x, double t,
                                     Vector & f)
   {
      f(0) = Gp(t) * x[1] * (c - x[1]) + 2.0 * nu * G(t);
      f(1) = 0.0;
   });

   Channel ch;
   BoundaryConditions bc(ch.spaces.Velocity());
   bc.AddNoSlip(ch.bottom);
   bc.AddRobin(ch.top, alpha, w);
   ASSERT_TRUE(bc.HasRobin());
   ASSERT_FALSE(bc.PressureNullspaceExists()); // the Robin top fixes p

   const TimeIntegratorOptions opts = Options(nu);
   StokesTimeIntegrator stepper(ch.spaces, ch.rules, bc, forcing, opts);
   stepper.SetInitialVelocity(u_exact);
   {
      incns::IntegratorState st;
      for (int j = 0; j < 2; ++j)
      {
         const double tj = -j * opts.dt;
         u_exact.SetTime(tj);
         ParGridFunction gf(&ch.spaces.Velocity());
         gf.ProjectCoefficient(u_exact);
         Vector tv(ch.spaces.Velocity().GetTrueVSize());
         gf.GetTrueDofs(tv);
         st.u_hist.push_back(tv);
         st.times.push_back(tj);
      }
      st.completed_steps = 1;
      st.next_dt = opts.dt;
      stepper.ImportState(st);
   }
   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(ch.rules, irs);
   FunctionCoefficient zero([](const Vector&) { return 0.0; });
   for (int step = 1; step <= 4; ++step)
   {
      stepper.Step();
      u_exact.SetTime(stepper.Time());
      EXPECT_LE(stepper.Velocity().ComputeL2Error(u_exact, irs), 1e-8)
            << "step " << step;
      EXPECT_LE(stepper.Pressure().ComputeL2Error(zero, irs), 1e-7)
            << "step " << step;
   }
}

// --- R2. A boundary load, and the residual at the Robin rows ----------------
TEST(Robin, R2_LoadAndResidual)
{
   const double nu = 0.7, alpha = 3.0, c = 1.6;
   const double phi1 = c - 1.0, dphi1 = c - 2.0;
   const double tau = nu * dphi1 + alpha * phi1; // the load's x component
   VectorFunctionCoefficient u_exact(2, [c](const Vector & x, Vector & v)
   {
      v(0) = x[1] * (c - x[1]);
      v(1) = 0.0;
   });
   Vector zv(2);
   zv = 0.0;
   VectorConstantCoefficient w(zv);
   Vector f0(2);
   f0 = 0.0;
   f0(0) = 2.0 * nu;
   VectorConstantCoefficient forcing(f0);

   Channel ch;
   BoundaryConditions bc(ch.spaces.Velocity());
   bc.AddNoSlip(ch.bottom);
   bc.AddRobin(ch.top, alpha, w);

   // g = (tau, 0) on the top, as a load vector.
   ParFiniteElementSpace& vfes = ch.spaces.Velocity();
   Vector tv(2);
   tv(0) = tau;
   tv(1) = 0.0;
   VectorConstantCoefficient tc(tv);
   Array<int> top_marker(ch.mesh.bdr_attributes.Max());
   top_marker = 0;
   top_marker[ch.top - 1] = 1;
   ParLinearForm lf(&vfes);
   lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tc), top_marker);
   lf.Assemble();
   Vector g(vfes.GetTrueVSize());
   lf.ParallelAssemble(g);

   TimeIntegratorOptions opts = Options(nu);
   opts.keep_rhs = true;
   StokesTimeIntegrator stepper(ch.spaces, ch.rules, bc, forcing, opts);
   stepper.SetBoundaryLoad(&g);
   stepper.SetInitialVelocity(u_exact);
   const IntegrationRule* irs[Geometry::NumGeom];
   ElevatedRules(ch.rules, irs);
   for (int step = 1; step <= 3; ++step)
   {
      stepper.Step();
      EXPECT_LE(stepper.Velocity().ComputeL2Error(u_exact, irs), 1e-8)
            << "step " << step;
   }

   // The residual at the free Robin rows: the boundary's load,
   // sigma n = g - alpha u (w = 0), tested against the basis.
   Vector r;
   stepper.MomentumResidual(r);
   ParBilinearForm m(&vfes);
   ConstantCoefficient a(alpha);
   m.AddBoundaryIntegrator(new VectorMassIntegrator(a), top_marker);
   m.Assemble();
   m.Finalize();
   std::unique_ptr<HypreParMatrix> M(m.ParallelAssemble());
   Vector u_true(vfes.GetTrueVSize()), expect(vfes.GetTrueVSize());
   stepper.Velocity().GetTrueDofs(u_true);
   M->Mult(u_true, expect);
   expect.Neg();
   expect += g;
   Array<int> top_tdofs, ess;
   vfes.GetEssentialTrueDofs(top_marker, top_tdofs);
   Array<int> bottom_marker(top_marker.Size());
   bottom_marker = 0;
   bottom_marker[ch.bottom - 1] = 1;
   vfes.GetEssentialTrueDofs(bottom_marker, ess);
   std::vector<char> is_ess(vfes.GetTrueVSize(), 0);
   for (int i : ess) { is_ess[i] = 1; }
   double err = 0.0, ref = 0.0;
   r.HostRead();
   expect.HostRead();
   for (int i : top_tdofs)
   {
      if (is_ess[i]) { continue; } // the corners are the wall's
      err = std::max(err, std::abs(r(i) - expect(i)));
      ref = std::max(ref, std::abs(expect(i)));
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &ref, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   ASSERT_GT(ref, 0.0);
   EXPECT_LE(err, 1e-8 * ref);
}

// --- R3. Large alpha: Dirichlet in the limit --------------------------------
TEST(Robin, R3_LargeAlphaApproachesDirichlet)
{
   // Steady forced channel, no-slip bottom, Robin top with w = 0: the
   // no-slip solution is y (1 - y) (nu u_yy = -2 nu); with Robin the top
   // slips by O(nu / alpha).
   const double nu = 0.7;
   VectorFunctionCoefficient u_noslip(2, [](const Vector & x, Vector & v)
   {
      v(0) = x[1] * (1.0 - x[1]);
      v(1) = 0.0;
   });
   Vector zv(2);
   zv = 0.0;
   VectorConstantCoefficient w(zv);
   Vector f0(2);
   f0 = 0.0;
   f0(0) = 2.0 * nu;
   VectorConstantCoefficient forcing(f0);
   const IntegrationRule* irs[Geometry::NumGeom];
   double err_prev = 0.0;
   for (double alpha : {1e2, 1e3, 1e4})
   {
      Channel ch;
      ElevatedRules(ch.rules, irs);
      BoundaryConditions bc(ch.spaces.Velocity());
      bc.AddNoSlip(ch.bottom);
      bc.AddRobin(ch.top, alpha, w);
      StokesTimeIntegrator stepper(ch.spaces, ch.rules, bc, forcing,
                                   Options(nu));
      // Exact Robin solution y (c - y), c = (2 nu + alpha) / (nu + alpha):
      // start there, so the steps keep it.
      const double c = (2.0 * nu + alpha) / (nu + alpha);
      VectorFunctionCoefficient u_robin(2, [c](const Vector & x, Vector & v)
      {
         v(0) = x[1] * (c - x[1]);
         v(1) = 0.0;
      });
      stepper.SetInitialVelocity(u_robin);
      for (int step = 0; step < 2; ++step) { stepper.Step(); }
      EXPECT_LE(stepper.Velocity().ComputeL2Error(u_robin, irs), 1e-8);
      const double err = stepper.Velocity().ComputeL2Error(u_noslip, irs);
      if (err_prev > 0.0)
      {
         // 1 / alpha: a factor 10 per decade, within 5 %.
         EXPECT_NEAR(err_prev / err, 10.0, 0.5) << "alpha " << alpha;
      }
      err_prev = err;
   }
}
