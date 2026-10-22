// The ALE fluid (docs/ale_fluid.md): a Case whose mesh moves at every step
// (Case::SetMeshMotion), IMEX, the non-conservative ALE form with the mesh
// velocity w the BDF derivative of the node positions. At np 1/2/4:
//  A1 zero motion: the moving-mesh path with d = 0 reproduces the fixed-mesh
//     run (Navier-Stokes Taylor-Green) to the solver tolerance;
//  A2 free stream (the geometric conservation test): a uniform flow on a
//     mesh whose interior nodes oscillate stays uniform, and the pressure
//     constant, to the solver tolerance, 2D and 3D, Navier-Stokes;
//  A3 a box moved rigidly with no-slip walls (u = w): the fluid moves with
//     it, u = w uniform and the pressure gradient balances the acceleration;
//     with a constant velocity u = V exactly, with an oscillation u equals
//     the discrete mesh velocity w_h;
//  A4 a linear Stokes flow on a mesh moved at constant rates (every node
//     path linear in time, so BDF2 and EXT2 are exact): translated, exact to
//     the solver tolerance; stretched too, the trapezoidal starter's O(dt^2)
//     error, falling at rate 2;
//  A5 a step whose solve fails on a moving mesh is cut: the mesh is moved
//     back, the step retried, and the run ends where the motion says;
//  A6, A7 Navier-Stokes Taylor-Green on a mesh whose interior oscillates:
//     the velocity's L2 error at Q3 converges at order 4 in h, and at Q5 at
//     order 2 in dt (BDF2/EXT2).
// These are unit tests: they pin their own small orders and tolerances.

#include <gtest/gtest.h>

#include "ale/mesh_motion.hpp"
#include "ale/mesh_motion_model.hpp"
#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem;
using incns::Case;
using incns::MeshMotion;
using incns::Parameters;

namespace
{

using Field = std::function<void(const Vector& X, double t, Vector& d)>;

/// A test model: the displacement is a given field of the reference
/// coordinates and the time, interpolated on the whole domain (no solve).
class PrescribedMotion : public incns::MeshMotionModel
{
public:
   PrescribedMotion(ParMesh& ref, int order, Field f)
      : fec_(order, ref.Dimension()),
        fes_(&ref, &fec_, ref.Dimension(), Ordering::byNODES), d_(&fes_),
        coeff_(ref.Dimension(), f)
   {
      d_ = 0.0;
   }
   const char* Name() const override { return "prescribed"; }
   ParFiniteElementSpace& Space() override { return fes_; }
   void AddFixed(int) override {}
   void AddDisplacement(int, VectorCoefficient&) override {}
   bool Solve(double t) override
   {
      coeff_.SetTime(t);
      d_.ProjectCoefficient(coeff_);
      return true;
   }
   const ParGridFunction& Displacement() const override { return d_; }
   int Iterations() const override { return 0; }

private:
   H1_FECollection fec_;
   ParFiniteElementSpace fes_;
   ParGridFunction d_;
   VectorFunctionCoefficient coeff_;
};

incns::MeshMotionModelFactory Prescribed(Field f)
{
   return [f](ParMesh & ref, int order)
   {
      return std::make_unique<PrescribedMotion>(ref, order, f);
   };
}

MeshMotion::Options MotionOptions(int order)
{
   MeshMotion::Options o;
   o.order = order;
   o.geometry_order = order;
   return o;
}

/// Max over the true dofs of |a - b| (global).
double MaxDiff(const Vector& a, const Vector& b)
{
   a.HostRead(); // on a device the solver's results are current there only
   b.HostRead();
   double m = 0.0;
   for (int i = 0; i < a.Size(); ++i) { m = std::max(m, std::abs(a(i) - b(i))); }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

/// Max over the velocity's true dofs of |u - value of component c|.
double MaxDeviation(const ParGridFunction& u, const Vector& value)
{
   Vector ut(u.ParFESpace()->GetTrueVSize());
   u.GetTrueDofs(ut);
   ut.HostRead();
   const int n = ut.Size() / value.Size();
   double m = 0.0;
   for (int c = 0; c < value.Size(); ++c)
   {
      for (int i = 0; i < n; ++i)
      {
         m = std::max(m, std::abs(ut(c * n + i) - value(c)));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

/// Max |p - mean| over the pressure's true dofs (global).
double PressureSpread(const ParGridFunction& p)
{
   Vector pt(p.ParFESpace()->GetTrueVSize());
   p.GetTrueDofs(pt);
   pt.HostRead();
   double mm[2] = {-1e300, -1e300};
   for (int i = 0; i < pt.Size(); ++i)
   {
      mm[0] = std::max(mm[0], pt(i));
      mm[1] = std::max(mm[1], -pt(i));
   }
   MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return mm[0] + mm[1];
}

/// Every boundary attribute of the mesh, each with Dirichlet data @p g.
void DirichletEverywhere(const ParMesh& mesh, incns::BoundaryConditions& bc,
                         VectorCoefficient& g)
{
   const Array<int>& a = mesh.bdr_attributes;
   for (int i = 0; i < a.Size(); ++i) { bc.AddVelocityDirichlet(a[i], g); }
}

std::string Deck(const std::string& equation, int dim, int n,
                 const std::string& extra)
{
   std::string elems = dim == 2 ? "[" + std::to_string(n) + ", " +
                       std::to_string(n) + "]" :
                       "[" + std::to_string(n) + ", " + std::to_string(n) +
                       ", " + std::to_string(n) + "]";
   std::string per = dim == 2 ? "[false, false]" : "[false, false, false]";
   std::string len = dim == 2 ? "[1.0, 1.0]" : "[1.0, 1.0, 1.0]";
   return "equation: " + equation + "\n"
          "mesh:\n  dim: " + std::to_string(dim) + "\n  elements: " + elems +
          "\n  periodic: " + per + "\n  lengths: " + len + "\n"
          "discretization:\n  order_u: 3\n"
          "solver:\n  rtol: 1.0e-12\n  atol: 1.0e-13\n  initial_guess: previous\n"
          "  tolerance_reference: initial_residual\n" + extra;
}

} // namespace

// A1: d = 0 through the moving-mesh path equals the fixed mesh.
TEST(AleFluid, A1_ZeroMotionMatchesFixedMesh)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("navier_stokes", 2, 4,
                                "initial_velocity: taylor_green_2d\n"
                                "physics:\n  nu: 0.05\n"
                                "time:\n  dt: 0.02\n  t_final: 0.1\n"
                                "  step_control: fixed\n"
                                "  convection: imex\n"));
   VectorFunctionCoefficient tg(2, [](const Vector & x, double t, Vector & u)
   {
      const double f = std::exp(-2.0 * 0.05 * 4.0 * M_PI * M_PI * t);
      u(0) = std::sin(2 * M_PI * x(0)) * std::cos(2 * M_PI * x(1)) * f;
      u(1) = -std::cos(2 * M_PI * x(0)) * std::sin(2 * M_PI * x(1)) * f;
   });
   Vector u_fixed, u_moving;
   for (int moving = 0; moving < 2; ++moving)
   {
      auto mesh = incns::MakeCaseMesh(p);
      std::unique_ptr<MeshMotion> motion;
      if (moving)
      {
         motion = std::make_unique<MeshMotion>(*mesh, MotionOptions(3),
                                               Prescribed([](const Vector&,
         double, Vector & d) { d = 0.0; }));
      }
      Case flow(*mesh, p);
      incns::BoundaryConditions bc(flow.Spaces().Velocity());
      DirichletEverywhere(*mesh, bc, tg);
      flow.SetBoundaryConditions(bc);
      flow.SetInitialVelocity(tg);
      if (moving) { flow.SetMeshMotion(*motion); }
      flow.Run();
      Vector& u = moving ? u_moving : u_fixed;
      u.SetSize(flow.Spaces().Velocity().GetTrueVSize());
      flow.Velocity().GetTrueDofs(u);
      if (moving) { EXPECT_EQ(motion->MoveCount(), 5); }
   }
   const double scale = u_fixed.Normlinf();
   EXPECT_LT(MaxDiff(u_fixed, u_moving), 1e-9 * std::max(scale, 1.0));
}

// A2: a uniform flow stays uniform on a mesh whose interior moves.
TEST(AleFluid, A2_FreeStreamOnAnOscillatingMesh)
{
   for (int dim = 2; dim <= 3; ++dim)
   {
      const int n = dim == 2 ? 4 : 2;
      const Parameters p = Parameters::LoadYAMLString(
                              Deck("navier_stokes", dim, n,
                                   "physics:\n  nu: 0.01\n"
                                   "time:\n  dt: 0.05\n  t_final: 0.3\n"
                                   "  step_control: fixed\n"));
      auto mesh = incns::MakeCaseMesh(p);
      // Zero on the unit box's boundary, a large interior motion.
      MeshMotion motion(*mesh, MotionOptions(3), Prescribed(
                           [dim](const Vector & X, double t, Vector & d)
      {
         double b = 1.0;
         for (int i = 0; i < dim; ++i) { b *= std::sin(M_PI * X(i)); }
         d = 0.0;
         // Period 0.4: still displaced at t_final = 0.3.
         d(0) = 0.15 * b * std::sin(2 * M_PI * t / 0.4);
         d(1) = 0.10 * b * (1.0 - std::cos(2 * M_PI * t / 0.4));
      }));
      Vector U(dim);
      U = 0.0;
      U(0) = 1.0;
      U(1) = -0.5;
      VectorConstantCoefficient u_inf(U);
      Case flow(*mesh, p);
      incns::BoundaryConditions bc(flow.Spaces().Velocity());
      DirichletEverywhere(*mesh, bc, u_inf);
      flow.SetBoundaryConditions(bc);
      flow.SetInitialVelocity(u_inf);
      flow.SetMeshMotion(motion);
      flow.Run();
      EXPECT_EQ(motion.MoveCount(), flow.Integrator().StepCount());
      double dmin = 0.0, dmax = 0.0;
      motion.JacobianRange(dmin, dmax);
      EXPECT_LT(dmin, 0.999); // the mesh did move
      EXPECT_LT(MaxDeviation(flow.Velocity(), U), 1e-9) << "dim " << dim;
      EXPECT_LT(PressureSpread(flow.Pressure()), 1e-8) << "dim " << dim;
   }
}

// A3: a rigidly moved box with no-slip walls (u = w) carries the fluid.
TEST(AleFluid, A3_RigidBoxCarriesTheFluid)
{
   for (int oscillate = 0; oscillate < 2; ++oscillate)
   {
      for (const std::string eq : {"stokes", "navier_stokes"})
      {
         const Parameters p = Parameters::LoadYAMLString(
                                 Deck(eq, 2, 4, "physics:\n  nu: 0.1\n"
                                      "time:\n  dt: 0.05\n  t_final: 0.5\n"
                                      "  step_control: fixed\n"));
         auto mesh = incns::MakeCaseMesh(p);
         MeshMotion motion(*mesh, MotionOptions(3), Prescribed(
                              [oscillate](const Vector&, double t, Vector & d)
         {
            if (oscillate)
            {
               d(0) = 0.3 * std::sin(2.0 * t);
               d(1) = 0.2 * (1.0 - std::cos(3.0 * t));
            }
            else
            {
               d(0) = 0.4 * t;
               d(1) = -0.3 * t;
            }
         }));
         Case flow(*mesh, p);
         incns::BoundaryConditions bc(flow.Spaces().Velocity());
         flow.SetMeshMotion(motion);
         DirichletEverywhere(*mesh, bc, flow.MeshVelocityCoefficient());
         flow.SetBoundaryConditions(bc);
         flow.Run();
         ASSERT_EQ(motion.MoveCount(), 10);
         // u = w_h, a uniform field: the box's (discrete) velocity.
         Vector w(2);
         {
            Vector wt(flow.Spaces().Velocity().GetTrueVSize());
            flow.MeshVelocity().GetTrueDofs(wt);
            wt.HostRead();
            const int nd = wt.Size() / 2;
            w = 0.0;
            if (nd > 0) { w(0) = wt(0); w(1) = wt(nd); }
            MPI_Allreduce(MPI_IN_PLACE, w.GetData(), 2, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            int owners = nd > 0 ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
            w /= owners;
         }
         EXPECT_LT(MaxDeviation(flow.MeshVelocity(), w), 1e-12);
         if (!oscillate)
         {
            EXPECT_NEAR(w(0), 0.4, 1e-12);
            EXPECT_NEAR(w(1), -0.3, 1e-12);
         }
         EXPECT_LT(MaxDeviation(flow.Velocity(), w), 1e-9)
               << eq << (oscillate ? ", oscillating" : ", constant velocity");
         if (!oscillate)
         {
            EXPECT_LT(PressureSpread(flow.Pressure()), 1e-8) << eq;
         }
      }
   }
}

namespace
{

/// A4's run: the linear Stokes flow u = (y + 1, 2 x - 0.5) (divergence-free,
/// f = 0, p constant) on a mesh moved by x = X + t (A X + b), every node path
/// linear in t; A = 0 with @p stretch false. @return the velocity's max
/// nodal error at t = 0.4 (and @p spread, the pressure's spread).
double LinearFlowError(bool stretch, double dt, double& spread)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("stokes", 2, 4, "physics:\n  nu: 0.01\n"
                                "time:\n  dt: " + std::to_string(dt) +
                                "\n  t_final: 0.4\n  step_control: fixed\n"));
   auto mesh = incns::MakeCaseMesh(p);
   const double s = stretch ? 1.0 : 0.0;
   MeshMotion motion(*mesh, MotionOptions(3), Prescribed(
                        [s](const Vector & X, double t, Vector & d)
   {
      d(0) = t * (s * (0.3 * X(0) + 0.1 * X(1)) + 0.2);
      d(1) = t * (s * (-0.1 * X(0) + 0.2 * X(1)) - 0.1);
   }));
   VectorFunctionCoefficient exact(2, [](const Vector & x, Vector & u)
   {
      u(0) = x(1) + 1.0;
      u(1) = 2.0 * x(0) - 0.5;
   });
   Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   DirichletEverywhere(*mesh, bc, exact);
   flow.SetBoundaryConditions(bc);
   flow.SetInitialVelocity(exact);
   flow.SetMeshMotion(motion);
   flow.Run();
   EXPECT_EQ(motion.MoveCount(), flow.Integrator().StepCount());
   ParGridFunction ue(&flow.Spaces().Velocity());
   ue.ProjectCoefficient(exact);
   Vector a(ue.ParFESpace()->GetTrueVSize()), b(a.Size());
   ue.GetTrueDofs(a);
   flow.Velocity().GetTrueDofs(b);
   spread = PressureSpread(flow.Pressure());
   return MaxDiff(a, b);
}

} // namespace

// A4: a linear Stokes flow on a mesh moved at constant rates. Translated,
// every term is exact (w uniform): the solution is exact to the solver
// tolerance. Stretched too, the BDF2 steps are exact but the trapezoidal
// starter's explicit C(w) u^0 is not (u^0 on the moved nodes is off by
// dt (grad u) w, and grad w != 0): an O(dt^2) error that the march carries,
// so the error falls 4x when dt halves.
TEST(AleFluid, A4_LinearFlowOnAMeshMovedAtConstantRates)
{
   double spread = 0.0;
   EXPECT_LT(LinearFlowError(false, 0.05, spread), 1e-9);
   EXPECT_LT(spread, 1e-8);
   const double e1 = LinearFlowError(true, 0.05, spread);
   const double e2 = LinearFlowError(true, 0.025, spread);
   if (Mpi::Root())
   {
      std::cout << "[ale] stretched linear flow: errors " << e1 << " " << e2
                << ", rate " << std::log2(e1 / e2) << std::endl;
   }
   EXPECT_GT(std::log2(e1 / e2), 1.8);
}

// A5: a rejected solve on a moving mesh: the mesh goes back, the step is
// retried at half the step, and the march ends on the motion's final mesh.
TEST(AleFluid, A5_CutStepMovesTheMeshBack)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("navier_stokes", 2, 4, "physics:\n  nu: 0.1\n"
                                "time:\n  dt: 0.05\n  t_final: 0.3\n"
                                "  step_control: fixed\n"
                                "  on_solver_failure: cut_step\n"));
   auto mesh = incns::MakeCaseMesh(p);
   MeshMotion motion(*mesh, MotionOptions(3), Prescribed(
                        [](const Vector&, double t, Vector & d)
   {
      d(0) = 0.4 * t;
      d(1) = 0.1 * t;
   }));
   Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   flow.SetMeshMotion(motion);
   DirichletEverywhere(*mesh, bc, flow.MeshVelocityCoefficient());
   flow.SetBoundaryConditions(bc);
   // Reject the first attempt of the third step.
   flow.Step(); // builds the integrator
   flow.Integrator().SetSolveVeto([](int step, int attempt)
   {
      return step == 2 && attempt == 0;
   });
   flow.Run();
   EXPECT_EQ(flow.Integrator().RejectedSteps(), 1);
   // Fixed steps end within half a step of t_final once a cut has moved
   // them off the grid (StokesTimeIntegrator::Done).
   EXPECT_NEAR(flow.Time(), 0.3, 0.5 * 0.05);
   EXPECT_NEAR(motion.Time(), flow.Time(), 1e-12);
   const double t_end = flow.Time();
   EXPECT_EQ(motion.MoveCount(), flow.Integrator().StepCount());
   // The nodes are the motion's at t_final: X + (0.4, 0.1) t.
   const GridFunction& x = *mesh->GetNodes();
   const Vector& X = motion.ReferenceNodes();
   x.HostRead();
   X.HostRead();
   const int nn = x.Size() / 2;
   double err = 0.0;
   for (int i = 0; i < nn; ++i)
   {
      err = std::max(err, std::abs(x(i) - X(i) - 0.4 * t_end));
      err = std::max(err, std::abs(x(nn + i) - X(nn + i) - 0.1 * t_end));
   }
   EXPECT_LT(err, 1e-12);
   Vector V(2);
   V(0) = 0.4;
   V(1) = 0.1;
   EXPECT_LT(MaxDeviation(flow.Velocity(), V), 1e-9);
}

namespace
{

/// The 2D Taylor-Green vortex on the unit box (wavenumber 2 pi), an exact
/// Navier-Stokes solution: the velocity's L2 error at t_final on a mesh whose
/// interior oscillates (the boundary is held; exact data on it).
double TaylorGreenError(int n, double dt, double t_final, int order)
{
   const double nu = 0.05, k = 2.0 * M_PI;
   const Parameters p = Parameters::LoadYAMLString(
                           "equation: navier_stokes\n"
                           "mesh:\n  dim: 2\n  elements: [" + std::to_string(n) +
                           ", " + std::to_string(n) + "]\n"
                           "  periodic: [false, false]\n  lengths: [1.0, 1.0]\n"
                           "discretization:\n  order_u: " + std::to_string(order) +
                           "\nphysics:\n  nu: 0.05\n"
                           "solver:\n  rtol: 1.0e-12\n  atol: 1.0e-10\n  initial_guess: previous\n"
                           "  tolerance_reference: initial_residual\n"
                           "time:\n  dt: " + std::to_string(dt) + "\n  t_final: " +
                           std::to_string(t_final) + "\n  step_control: fixed\n");
   auto mesh = incns::MakeCaseMesh(p);
   MeshMotion motion(*mesh, MotionOptions(order), Prescribed(
                        [](const Vector & X, double t, Vector & d)
   {
      // Period 0.4: the mesh moves at speeds comparable to the flow's.
      const double b = std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
      d(0) = 0.12 * b * std::sin(2 * M_PI * t / 0.4);
      d(1) = 0.08 * b * std::sin(M_PI * t / 0.4);
   }));
   VectorFunctionCoefficient tg(2, [nu, k](const Vector & x, double t, Vector & u)
   {
      const double f = std::exp(-2.0 * nu * k * k * t);
      u(0) = std::sin(k * x(0)) * std::cos(k * x(1)) * f;
      u(1) = -std::cos(k * x(0)) * std::sin(k * x(1)) * f;
   });
   Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   DirichletEverywhere(*mesh, bc, tg);
   flow.SetBoundaryConditions(bc);
   flow.SetInitialVelocity(tg);
   flow.SetMeshMotion(motion);
   flow.Run();
   EXPECT_NEAR(flow.Time(), t_final, 1e-10);
   tg.SetTime(flow.Time());
   const IntegrationRule* irs[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; ++g)
   {
      irs[g] = &IntRules.Get(g, 2 * order + 6);
   }
   return flow.Velocity().ComputeL2Error(tg, irs);
}

double Rate(double e_coarse, double e_fine, double ratio)
{
   return std::log(e_coarse / e_fine) / std::log(ratio);
}

} // namespace

// A6: spatial order on the moving mesh (Q3: velocity L2 at order 4).
TEST(AleFluid, A6_SpatialOrderOnAMovingMesh)
{
   std::vector<double> e;
   for (int n : {2, 4, 8}) { e.push_back(TaylorGreenError(n, 5e-4, 0.1, 3)); }
   if (Mpi::Root())
   {
      std::cout << "[ale] spatial errors " << e[0] << " " << e[1] << " "
                << e[2] << ", rates " << Rate(e[0], e[1], 2) << " "
                << Rate(e[1], e[2], 2) << std::endl;
   }
   EXPECT_GT(Rate(e[1], e[2], 2.0), 3.7);
}

// A7: temporal order on the moving mesh (BDF2/EXT2: 2).
TEST(AleFluid, A7_TemporalOrderOnAMovingMesh)
{
   std::vector<double> e;
   for (double dt : {0.04, 0.02, 0.01}) { e.push_back(TaylorGreenError(8, dt, 0.4, 5)); }
   if (Mpi::Root())
   {
      std::cout << "[ale] temporal errors " << e[0] << " " << e[1] << " "
                << e[2] << ", rates " << Rate(e[0], e[1], 2) << " "
                << Rate(e[1], e[2], 2) << std::endl;
   }
   EXPECT_GT(Rate(e[1], e[2], 2.0), 1.7);
}

namespace
{

/// A deck whose whole boundary moves by @p motion (two expressions in the
/// reference x, y and t), a no-slip wall there (u = w): Navier-Stokes on a
/// 4 x 4 unit box, Q3, fixed steps.
std::string MovingWallDeck(const std::string& motion, double t_final,
                           const std::string& extra = "")
{
   return Deck("navier_stokes", 2, 4,
               "physics:\n  nu: 0.1\n"
               "time:\n  dt: 0.05\n  t_final: " + std::to_string(t_final) +
               "\n  step_control: fixed\n"
               "boundary_conditions:\n"
               "  - {select: all, type: no_slip, motion: " + motion +
               ", parameters: {a: 0.3, b: 0.2}}\n" + extra);
}

} // namespace

// A8: the deck surface. A boundary group with `motion` moves the mesh (the
// case builds the linear elastic model of section mesh_motion), a moving
// no_slip wall takes u = w, and convection auto is IMEX. A rigid translation
// of the whole box carries the fluid exactly, as A3 does by hand.
TEST(AleFluid, A8_DeckMovesTheMesh)
{
   const Parameters p = Parameters::LoadYAMLString(
                           MovingWallDeck("[\"a*t\", \"-b*t\"]", 0.3));
   EXPECT_TRUE(p.moving_mesh);
   EXPECT_EQ(p.Convection(), incns::ConvectionTreatment::Imex);
   auto mesh = incns::MakeCaseMesh(p);
   Case flow(*mesh, p);
   ASSERT_TRUE(flow.MovingMesh());
   flow.Run();
   EXPECT_EQ(flow.Motion().MoveCount(), 6);
   Vector V(2);
   V(0) = 0.3;
   V(1) = -0.2;
   EXPECT_LT(MaxDeviation(flow.MeshVelocity(), V), 1e-9);
   EXPECT_LT(MaxDeviation(flow.Velocity(), V), 1e-9);
   // The nodes are the reference ones translated by V t.
   const GridFunction& x = *mesh->GetNodes();
   const Vector& X = flow.Motion().ReferenceNodes();
   x.HostRead();
   X.HostRead();
   const int nn = x.Size() / 2;
   double err = 0.0;
   for (int i = 0; i < nn; ++i)
   {
      err = std::max(err, std::abs(x(i) - X(i) - 0.3 * flow.Time()));
      err = std::max(err, std::abs(x(nn + i) - X(nn + i) + 0.2 * flow.Time()));
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LT(err, 1e-9);
}

// A9: a checkpoint of a moving-mesh run carries the node history: the
// restarted march equals the uninterrupted one (same rank count), with the
// interior moved by the elastic model. Bitwise with the solvers rebuilt
// after every move; with the in-place refresh to the solver tolerance (the
// restarted solvers' AMG hierarchies are built on the restart's mesh, the
// uninterrupted run's on an earlier one).
static void RestartMatchesUninterrupted(const std::string& refresh, double tol)
{
   const std::string motion =
      "[\"a*sin(2*pi*t)*x*(1-x)*y\", \"b*sin(2*pi*t)*y*(1-y)*x\"]";
   const std::string dir = "ale_chk_" + refresh + "_np" +
                           std::to_string(Mpi::WorldSize());
   const std::string extra = "mesh_motion:\n  refresh: " + refresh + "\n";
   Vector u_ref, u_restart;
   double t_ref = 0.0;
   {
      const Parameters p = Parameters::LoadYAMLString(MovingWallDeck(motion,
                           0.4, extra));
      auto mesh = incns::MakeCaseMesh(p);
      Case flow(*mesh, p);
      flow.Run();
      u_ref.SetSize(flow.Spaces().Velocity().GetTrueVSize());
      flow.Velocity().GetTrueDofs(u_ref);
      t_ref = flow.Time();
   }
   {
      const Parameters p = Parameters::LoadYAMLString(MovingWallDeck(motion,
                           0.4, extra));
      auto mesh = incns::MakeCaseMesh(p);
      Case flow(*mesh, p);
      for (int i = 0; i < 4; ++i) { flow.Step(); }
      flow.WriteCheckpoint(dir);
   }
   {
      Parameters p = Parameters::LoadYAMLString(MovingWallDeck(motion, 0.4,
                     extra));
      p.restart_from = dir;
      auto mesh = incns::MakeCaseMesh(p);
      Case flow(*mesh, p);
      flow.Run();
      EXPECT_NEAR(flow.Time(), t_ref, 1e-12);
      EXPECT_EQ(flow.Motion().MoveCount(), 8);
      u_restart.SetSize(flow.Spaces().Velocity().GetTrueVSize());
      flow.Velocity().GetTrueDofs(u_restart);
   }
   EXPECT_LT(MaxDiff(u_ref, u_restart), tol * std::max(1.0, u_ref.Normlinf()))
         << refresh;
   EXPECT_GT(u_ref.Normlinf(), 1e-3); // the wall moved the fluid
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::filesystem::remove_all(dir); }
}

TEST(AleFluid, A9_RestartMatchesUninterrupted)
{
   RestartMatchesUninterrupted("rebuild", 1e-13);
   RestartMatchesUninterrupted("in_place", 1e-10);
}

// A10: the in-place refresh after a move (mesh_motion.refresh: in_place,
// StokesSolver::UpdateGeometry) marches to the answer of building every
// solver anew (rebuild), to the solver tolerance: with the AMG hierarchies
// rebuilt at every move and kept for three, on the default solver path, with
// a LOR-AMG velocity block, and with the Schur (Uzawa) outer method.
TEST(AleFluid, A10_InPlaceRefreshMatchesRebuild)
{
   const std::string motion =
      "[\"a*sin(2*pi*t)*x*(1-x)*y\", \"b*sin(2*pi*t)*y*(1-y)*x\"]";
   struct Variant
   {
      const char* name;
      const char* refresh;
      int interval;
      bool loramg;
      bool schur;
   };
   const Variant variants[] =
   {
      {"rebuild", "rebuild", 1, false, false},
      {"in_place/1", "in_place", 1, false, false},
      {"in_place/3", "in_place", 3, false, false},
      {"rebuild loramg", "rebuild", 1, true, false},
      {"in_place/3 loramg", "in_place", 3, true, false},
      {"rebuild schur", "rebuild", 1, false, true},
      {"in_place/3 schur", "in_place", 3, false, true},
   };
   Vector u_ref;
   for (const Variant& v : variants)
   {
      Parameters p = Parameters::LoadYAMLString(MovingWallDeck(motion, 0.4,
                     std::string("mesh_motion:\n  refresh: ") + v.refresh +
                     "\n  amg_interval: " + std::to_string(v.interval) + "\n"));
      if (v.loramg) { p.cc.a_pc = incns::APC::LORAMG; }
      if (v.schur) { p.outer = incns::OuterMethod::Schur; }
      auto mesh = incns::MakeCaseMesh(p);
      Case flow(*mesh, p);
      flow.Run();
      EXPECT_EQ(flow.Motion().MoveCount(), 8) << v.name;
      Vector u(flow.Spaces().Velocity().GetTrueVSize());
      flow.Velocity().GetTrueDofs(u);
      if (u_ref.Size() == 0)
      {
         u_ref = u;
         EXPECT_GT(u_ref.Normlinf(), 1e-3); // the wall moved the fluid
         continue;
      }
      const double d = MaxDiff(u, u_ref);
      if (Mpi::Root()) { std::cout << "[ale] " << v.name << ": " << d << std::endl; }
      EXPECT_LT(d, 1e-9 * u_ref.Normlinf()) << v.name;
   }
}
