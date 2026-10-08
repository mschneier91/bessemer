// DFG benchmark 2D-1 (Schaefer & Turek 1996): steady flow around a cylinder at
// Re = 20, marched from rest (ramped parabolic inflow) to steady state on the
// default cylinder-channel mesh (mesh/cylinder_channel, Q3/Q2). Drag and lift
// by John's volume-integral formulation (post/body_force), the pressure
// difference between the cylinder's front and back points, all compared with
// John's reference values (test/baselines.yaml `dfg_2d1`, with the measured
// relative-error ceilings). Two runs: the convective form (the benchmark's)
// and the rotational form with grad-div. `slow` label, np 4 only, ~9.5 min --
// run on demand: `scripts/test.sh cpu -L slow`.
//
// What it pins that the fast tier cannot: the residual-based force is the
// right force on a curved, non-trivial body in a real flow (sign, scaling,
// every term of the residual), to the accuracy the discretization supports.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <memory>

using namespace mfem;

namespace
{

constexpr double kH = 0.41;

// Pressure at a point (collective): the owning rank's value, reduced.
double PointValue(ParMesh& mesh, ParGridFunction& p, double x, double y)
{
   DenseMatrix pts(2, 1);
   pts(0, 0) = x;
   pts(1, 0) = y;
   Array<int> elems;
   Array<IntegrationPoint> ips;
   mesh.FindPoints(pts, elems, ips, false);
   double buf[2] = {0.0, 0.0};
   if (elems[0] >= 0)
   {
      buf[0] = p.GetValue(elems[0], ips[0]);
      buf[1] = 1.0;
   }
   MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return buf[0] / buf[1];
}

struct Result
{
   double c_d, c_l, dp;
};

// March 2D-1 from rest to t_final in the given form; c_D, c_L and dp there.
Result RunSteadyRe20(bool rotational, double grad_div, double t_final,
                     double cfl_max)
{
   const double U_m = 0.3, U_mean = 2.0 * U_m / 3.0, D = 0.1;
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = rotational ? incns::ConvectiveForm::Rotational :
                       incns::ConvectiveForm::Convective;
   if (rotational) { p.rotation_pc = incns::RotationVelocityPC::PbjKrylov; }
   p.nu = 1e-3;
   p.grad_div = grad_div;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh.dim = 2;
   p.dt = 0.01;
   p.t_final = t_final;
   p.step_control = incns::StepControl::Error;
   p.controller.atol = 1e-6;
   p.controller.rtol = 1e-5;
   p.cfl_max = cfl_max;
   p.krylov_rtol = 1e-10;
   p.forces.enabled = true;
   p.forces.attributes = {incns::kCylinderBody};
   p.forces.reference_velocity = U_mean;
   p.forces.reference_area = D;
   p.forces.interval = 0; // no CSV log
   p.Normalize();

   incns::CylinderChannelSpec spec; // DFG geometry, default resolution
   Mesh serial = incns::MakeCylinderChannelMesh(spec);
   std::unique_ptr<ParMesh> mesh = incns::PartitionMesh(serial, false);

   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   VectorFunctionCoefficient inflow(2, [U_m](const Vector & x, double t,
                                    Vector & u)
   {
      const double s = (t < 1.0) ? std::sin(0.5 * M_PI * t) : 1.0;
      u(0) = s * s * 4.0 * U_m * x(1) * (kH - x(1)) / (kH * kH);
      u(1) = 0.0;
   });
   bc.AddVelocityDirichlet(incns::kCylinderInflow, inflow);
   bc.AddNoSlip(incns::kCylinderWalls);
   bc.AddNoSlip(incns::kCylinderBody);
   bc.AddOutflow(incns::kCylinderOutflow);
   flow.SetBoundaryConditions(bc);
   flow.Run();

   const Vector C = flow.ForceCoefficients();
   const double dp = PointValue(*mesh, flow.Pressure(), 0.15, 0.2) -
                     PointValue(*mesh, flow.Pressure(), 0.25, 0.2);
   return {C(0), C(1), dp};
}

void Check(const char* name)
{
   const YAML::Node root = YAML::LoadFile(INCNS_BASELINES_FILE)["dfg_2d1"];
   ASSERT_TRUE(root) << "baselines.yaml: missing dfg_2d1";
   const YAML::Node ref = root["reference"], b = root[name];
   ASSERT_TRUE(ref && b) << "baselines.yaml: missing dfg_2d1." << name;
   const Result r = RunSteadyRe20(b["rotational"].as<bool>(),
                                  b["grad_div"].as<double>(),
                                  b["t_final"].as<double>(),
                                  b["cfl_max"].as<double>());
   const double cd = ref["c_d"].as<double>(), cl = ref["c_l"].as<double>(),
                dp = ref["dp"].as<double>();
   const double e_d = std::abs(r.c_d - cd) / cd, e_l = std::abs(r.c_l - cl) / cl,
         e_p = std::abs(r.dp - dp) / dp;
   if (Mpi::Root())
   {
      std::printf("DFG 2D-1 %s: c_D = %.10f (rel err %.3e), c_L = %.10f "
                  "(%.3e), dp = %.10f (%.3e)\n", name, r.c_d, e_d, r.c_l, e_l,
                  r.dp, e_p);
   }
   EXPECT_LT(e_d, b["max_rel_err_c_d"].as<double>());
   EXPECT_LT(e_l, b["max_rel_err_c_l"].as<double>());
   EXPECT_LT(e_p, b["max_rel_err_dp"].as<double>());
}

} // namespace

// The benchmark's own form: convective (IMEX, CFL-capped steps). ~8 min.
TEST(DfgCylinder, D1_ConvectiveForm)
{
   Check("convective");
}

// Rotational (semi-implicit, PBJ) with grad-div: without grad-div the
// rotational form's c_D error is ~9e-4 on this mesh and stalls under
// refinement (the classical rotation-form accuracy loss); grad-div brings it
// to the convective form's ~1e-4, so this also guards the rotation term in
// the residual. ~2 min.
TEST(DfgCylinder, D2_RotationalFormWithGradDiv)
{
   Check("rotational_grad_div");
}
