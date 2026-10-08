// Lift/drag by John's volume-integral formulation (post/body_force).
//  F1 EXACT: channel periodic in x (and z), no-slip walls at y = 0, 1, exact
//     solution u = (g(t) y(1-y), 0[, 0]), p = q(t)(y - 1/2), g quadratic and q
//     linear in time (both in the discrete spaces, BDF2-exact). The force on a
//     wall is known in closed form: bottom (y = 0): F = (nu g, q/2) L, top
//     (y = 1): F = (nu g, q/2) L (L = wall length, or area in 3D). The
//     computed force must match at every step from the second on (the
//     trapezoidal starter's pressure is a time average). Stokes and
//     convective NSE ((u.grad)u = 0 here), 2D and 3D, and an AMR-refined NC
//     mesh with hanging nodes on the body and refinement events during the
//     march (the force queried right after an event).
//  F2 choice independence: F computed with v_i and with v_i + w (w any vector
//     vanishing at the Dirichlet dofs) agree to Krylov tolerance -- the
//     residual of the scheme vanishes at every free dof, so only v_i's
//     boundary values matter (the discrete form of the continuous identity).

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "bc/boundary_conditions.hpp"
#include "mesh/case_mesh.hpp"
#include "post/body_force.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::Case;
using incns::Parameters;

namespace
{

double G(double t) { return 1.0 + t + 0.5 * t * t; }
double Gp(double t) { return 1.0 + t; }
double Q(double t) { return 2.0 + t; }

// Boundary attribute of the real face y = @p y0 (geometric, global).
int WallAttribute(ParMesh& mesh, double y0)
{
   int attr = 0;
   Vector c;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      if (std::abs(c(1) - y0) < 1e-9) { attr = mesh.GetBdrAttribute(be); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   return attr;
}

struct ChannelFlow
{
   int dim;
   double nu = 0.3;
   std::unique_ptr<VectorFunctionCoefficient> u, f;
   explicit ChannelFlow(int d) : dim(d)
   {
      const double n = nu;
      u = std::make_unique<VectorFunctionCoefficient>(dim, [](const Vector & x,
          double t, Vector & v)
      {
         v = 0.0;
         v(0) = G(t) * x(1) * (1.0 - x(1));
      });
      f = std::make_unique<VectorFunctionCoefficient>(dim, [n](const Vector & x,
          double t, Vector & v)
      {
         v = 0.0;
         v(0) = Gp(t) * x(1) * (1.0 - x(1)) + 2.0 * n * G(t);
         v(1) = Q(t); // grad p
      });
   }
};

Parameters ChannelParams(int dim, bool nse)
{
   Parameters p;
   p.equation = nse ? incns::Equation::NavierStokes : incns::Equation::Stokes;
   p.nu = 0.3;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh = amr_test::Box(dim, 3, false);
   p.mesh.num_elems = {4, 3, 3};
   p.mesh.lengths = {2.0, 1.0, 1.5};
   p.mesh.periodic = {true, false, true};
   p.dt = 0.02;
   p.t_final = 0.1;
   p.step_control = incns::StepControl::Fixed;
   p.krylov_rtol = 1e-12;
   p.max_iter = 5000;
   p.kdim = 400;
   return p;
}

void RunExactForce(int dim, bool nse, bool amr)
{
   SCOPED_TRACE("dim=" + std::to_string(dim) + (nse ? " nse" : " stokes") +
                (amr ? " amr" : ""));
   Parameters p = ChannelParams(dim, nse);
   if (amr)
   {
      p.amr.enabled = true;
      // Refine where the IC varies (at the walls), and once more during the
      // march (step 3): right after an event the rebuilt integrator has not
      // stepped, and the force must still be available (Case caches the
      // pre-event step's force).
      p.amr.interval = 3;
      p.amr.initial_passes = 2;
      p.amr.theta = 0.5;
   }
   ChannelFlow flow_ex(dim);
   const double area = p.mesh.lengths[0] * (dim == 3 ? p.mesh.lengths[2] : 1.0);
   for (double y0 : {0.0, 1.0})
   {
      p.forces.enabled = true;
      p.forces.interval = 0; // no CSV log
      auto mesh = incns::MakeCaseMesh([&]() { Parameters q = p; q.Normalize(); return q; }());
      p.forces.attributes = {WallAttribute(*mesh, y0)};
      Parameters q = p;
      q.Normalize();
      Case flow(*mesh, q);
      incns::BoundaryConditions bc(flow.Spaces().Velocity());
      bc.AddVelocityDirichlet(WallAttribute(*mesh, 0.0), *flow_ex.u);
      bc.AddVelocityDirichlet(WallAttribute(*mesh, 1.0), *flow_ex.u);
      flow.SetBoundaryConditions(bc);
      flow.SetInitialVelocity(*flow_ex.u);
      flow.SetForcing(*flow_ex.f);
      int step = 0;
      while (!flow.Done())
      {
         flow.Step();
         ++step;
         if (step < 2) { continue; }
         const double t = flow.Time();
         const Vector F = flow.BodyForceVector();
         const double fx = p.nu * G(t) * area, fy = 0.5 * Q(t) * area;
         EXPECT_NEAR(F(0), fx, 1e-8 * fx) << "wall y=" << y0 << " step " << step;
         EXPECT_NEAR(F(1), fy, 1e-8 * fy) << "wall y=" << y0 << " step " << step;
         if (dim == 3) { EXPECT_NEAR(F(2), 0.0, 1e-8 * fx); }
      }
      if (amr) { EXPECT_TRUE(mesh->Nonconforming()); }
   }
}

} // namespace

TEST(BodyForce, F1_ExactChannelForce)
{
   RunExactForce(2, false, false);
   RunExactForce(2, true, false);
   RunExactForce(3, false, false);
   RunExactForce(2, false, true);
}

namespace
{

void CheckInteriorIndependence(incns::ConvectiveForm form, double grad_div)
{
   SCOPED_TRACE(std::string(form == incns::ConvectiveForm::Rotational ?
                            "rotational" : "convective") +
                (grad_div > 0 ? " + grad-div" : ""));
   // A non-polynomial forcing: the discrete solution is NOT exact, so the
   // residual's interior part is the Krylov residual, not exactly zero -- and
   // a term missing from the residual (rotation, grad-div, ...) would leave
   // an O(1) interior residual and make the force depend on w.
   Parameters p = ChannelParams(2, true);
   p.convective_form = form;
   p.grad_div = grad_div;
   p.forces.enabled = true;
   p.forces.interval = 0; // no CSV log
   Parameters q0 = p;
   q0.Normalize();
   auto mesh = incns::MakeCaseMesh(q0);
   p.forces.attributes = {WallAttribute(*mesh, 0.0)};
   p.Normalize();
   Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   bc.AddNoSlip(WallAttribute(*mesh, 0.0));
   bc.AddNoSlip(WallAttribute(*mesh, 1.0));
   flow.SetBoundaryConditions(bc);
   VectorFunctionCoefficient f(2, [](const Vector & x, double t, Vector & v)
   {
      v(0) = 1.0 + std::sin(3.0 * x(0)) * std::cos(2.0 * x(1)) + t;
      v(1) = std::cos(x(0) + x(1));
   });
   flow.SetForcing(f);
   for (int s = 0; s < 3; ++s) { flow.Step(); }

   Vector r;
   flow.Integrator().MomentumResidual(r);
   incns::BodyForce bf(flow.Spaces().Velocity(), p.forces.attributes);
   for (int i = 0; i < 2; ++i)
   {
      Vector w(r.Size());
      w.Randomize(11 + i);
      const Array<int>& ess = bc.EssentialTrueDofs();
      w.SetSubVector(ess, 0.0); // vanishes on the whole Dirichlet boundary
      Vector v2(bf.TestFunction(i));
      v2 += w;
      const double f1 = bf.ForceWith(r, bf.TestFunction(i));
      const double f2 = bf.ForceWith(r, v2);
      EXPECT_NEAR(f2, f1, 1e-8 * std::abs(f1)) << "component " << i;
      EXPECT_GT(std::abs(f1), 1e-3); // a real force, not a vacuous zero
   }
}

} // namespace

TEST(BodyForce, F2_IndependentOfTheInteriorExtension)
{
   CheckInteriorIndependence(incns::ConvectiveForm::Convective, 0.0);
   CheckInteriorIndependence(incns::ConvectiveForm::Rotational, 0.0);
   CheckInteriorIndependence(incns::ConvectiveForm::Convective, 0.5);
}
