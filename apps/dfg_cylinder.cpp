// DFG "flow around a cylinder" benchmark (Schaefer & Turek 1996), with drag
// and lift by John's volume-integral formulation (post/body_force) and the
// pressure difference between the cylinder's front and back points.
//
//   Case 1 = 2D-1: steady, Re = 20 (U_m = 0.3, U_mean = 0.2, D = 0.1,
//            nu = 1e-3). Reference values (John 2004 / Nabh): c_D =
//            5.57953523384, c_L = 0.010618948146, dp = 0.11752016697. The
//            flow is marched from rest (smoothly ramped inflow) to steady state.
//   Case 2 = 2D-2: periodic vortex shedding, Re = 100 (U_m = 1.5). Reference
//            (John 2004): c_D,max = 3.22662, c_L,max = 1.00 (approx.); a
//            long run -- expensive tier, launch it by hand.
//
// Usage: mpirun -np <=4 build/cpu/apps/dfg_cylinder [-c 1] [-o 3] [-ns 4]
//        [-nr 3] [-nd 16] [-ip 0] [-tf 12] [-dt 0.01] [-rot [-pbj]] [-dout]
//        [-out dir]
//
// Prints the force coefficients every reported step and, for case 1, the
// relative errors against the reference values at the end.

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
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
   double val = 0.0, found = 0.0;
   if (elems[0] >= 0)
   {
      val = p.GetValue(elems[0], ips[0]);
      found = 1.0;
   }
   double buf[2] = {val, found};
   MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return buf[1] > 0.0 ? buf[0] / buf[1] : std::nan("");
}

} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int bench = 1, order = 3, n_side = 4, n_ring = 3, n_down = 16;
   int initial_passes = 0, amr_interval = 0;
   double t_final = 12.0, dt = 0.01, cfl = 0.6, grad_div = 0.0;
   bool rotational = false, pbj = false, dirichlet_out = false;
   const char* out = "dfg_out";
   OptionsParser args(argc, argv);
   args.AddOption(&bench, "-c", "--case", "1 = 2D-1 (steady, Re 20), 2 = 2D-2.");
   args.AddOption(&order, "-o", "--order", "Velocity order k_u (k_p = k_u - 1).");
   args.AddOption(&n_side, "-ns", "--n-side", "O-grid cells per square side.");
   args.AddOption(&n_ring, "-nr", "--n-ring", "Radial ring layers.");
   args.AddOption(&n_down, "-nd", "--n-down", "Downstream cells.");
   args.AddOption(&initial_passes, "-ip", "--initial-passes",
                  "AMR passes on the initial (ramped) flow.");
   args.AddOption(&amr_interval, "-ai", "--amr-interval",
                  "AMR event every N steps (0 = none).");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&dt, "-dt", "--dt", "Initial step size (adaptive).");
   args.AddOption(&cfl, "-cfl", "--cfl", "Convective CFL ceiling.");
   args.AddOption(&rotational, "-rot", "--rotational", "-conv", "--convective",
                  "Rotational (semi-implicit) or convective (IMEX) form.");
   args.AddOption(&pbj, "-pbj", "--pbj-krylov", "-sym", "--symmetric-pc",
                  "Rotational form: PBJ-preconditioned GMRES velocity PC.");
   args.AddOption(&grad_div, "-gd", "--grad-div",
                  "Grad-div scale c_gd (gamma = c_gd h_K; 0 = off).");
   args.AddOption(&dirichlet_out, "-dout", "--dirichlet-outflow", "-nout",
                  "--natural-outflow", "Outflow: the inflow profile imposed "
                  "(Dirichlet) or do-nothing (natural, the benchmark's).");
   args.AddOption(&out, "-out", "--output", "Output directory (forces CSV).");
   args.ParseCheck();

   const double U_m = (bench == 1) ? 0.3 : 1.5, U_mean = 2.0 * U_m / 3.0;
   const double D = 0.1;

   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = rotational ? incns::ConvectiveForm::Rotational
                       : incns::ConvectiveForm::Convective;
   p.rotation_pc = pbj ? incns::RotationVelocityPC::PbjKrylov
                   : incns::RotationVelocityPC::Symmetric;
   p.nu = 1e-3;
   p.grad_div = grad_div;
   p.order_u = order;
   p.order_p = order - 1;
   p.mesh.dim = 2;
   p.dt = dt;
   p.t_final = t_final;
   p.adaptive = true;
   p.controller.atol = 1e-6;
   p.controller.rtol = 1e-5;
   p.cfl_max = cfl;
   p.krylov_rtol = 1e-10;
   p.forces.enabled = true;
   p.forces.attributes = {incns::kCylinderBody};
   p.forces.reference_velocity = U_mean;
   p.forces.reference_area = D;
   p.forces.interval = 10;
   p.output.path = out;
   p.output.name = (bench == 1) ? "dfg_2d1" : "dfg_2d2";
   p.amr.enabled = (initial_passes > 0 || amr_interval > 0);
   p.amr.initial_passes = initial_passes;
   p.amr.interval = amr_interval;
   p.amr.theta = 0.3;
   p.amr.max_elements = 20000;
   p.Normalize();
   incns::ConfigureDevice(p.device);

   incns::CylinderChannelSpec spec;
   spec.n_side = n_side;
   spec.n_ring = n_ring;
   spec.n_down = n_down;
   spec.order = std::max(order, 2);
   Mesh serial = incns::MakeCylinderChannelMesh(spec);
   std::unique_ptr<ParMesh> mesh = incns::PartitionMesh(serial, p.amr.enabled);

   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   // Parabolic inflow, smoothly ramped from rest over t in [0, 1].
   VectorFunctionCoefficient inflow(2, [U_m](const Vector & x, double t,
                                    Vector & u)
   {
      const double ramp = (t < 1.0) ? std::sin(0.5 * M_PI * t) * std::sin(
                             0.5 * M_PI * t) : 1.0;
      u(0) = ramp * 4.0 * U_m * x(1) * (kH - x(1)) / (kH * kH);
      u(1) = 0.0;
   });
   bc.AddVelocityDirichlet(incns::kCylinderInflow, inflow);
   bc.AddNoSlip(incns::kCylinderWalls);
   bc.AddNoSlip(incns::kCylinderBody);
   // Do-nothing acts on the solver's pressure: the Bernoulli head in the
   // rotational form, a different outflow condition than the benchmark's. The
   // developed profile imposed at the outlet removes that modelling difference
   // (for 2D-1 it changes nothing measurable: the outlet is far downstream).
   if (dirichlet_out) { bc.AddVelocityDirichlet(incns::kCylinderOutflow, inflow); }
   else { bc.AddOutflow(incns::kCylinderOutflow); }
   flow.SetBoundaryConditions(bc);

   int steps = 0;
   double cd = 0.0, cl = 0.0, cd_prev = 0.0;
   while (!flow.Done())
   {
      flow.Step();
      ++steps;
      if (steps % 20 == 0 || flow.Done())
      {
         const Vector C = flow.ForceCoefficients();
         cd_prev = cd;
         cd = C(0);
         cl = C(1);
         const long long ne = mesh->GetGlobalNE();
         if (Mpi::Root())
         {
            std::printf("t = %8.4f  dt = %.3e  NE = %lld  c_D = %.8f  c_L = %+.8f\n",
                        flow.Time(), flow.Integrator().CurrentDt(), ne, cd, cl);
            std::fflush(stdout);
         }
      }
   }
   const double dp = PointValue(*mesh, flow.Pressure(), 0.15, 0.2) -
                     PointValue(*mesh, flow.Pressure(), 0.25, 0.2);
   if (Mpi::Root())
   {
      std::printf("final: t = %.4f, %d steps, c_D = %.10f, c_L = %.10f, "
                  "dp = %.10f (last c_D change %.2e)\n", flow.Time(), steps, cd,
                  cl, dp, std::abs(cd - cd_prev));
      if (bench == 1)
      {
         const double cd_ref = 5.57953523384, cl_ref = 0.010618948146,
                      dp_ref = 0.11752016697;
         std::printf("2D-1 relative errors: c_D %.3e, c_L %.3e, dp %.3e\n",
                     std::abs(cd - cd_ref) / cd_ref, std::abs(cl - cl_ref) / cl_ref,
                     std::abs(dp - dp_ref) / dp_ref);
      }
   }
   return 0;
}
