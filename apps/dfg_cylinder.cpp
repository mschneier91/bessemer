// DFG "flow around a cylinder" benchmarks (Schaefer & Turek 1996), with drag
// and lift by John's volume-integral formulation (post/body_force) and the
// pressure difference between the cylinder's front and back points.
//
//   Case 1 = 2D-1: steady, Re = 20 (U_m = 0.3, U_mean = 0.2, D = 0.1,
//            nu = 1e-3). Reference values (John 2004 / Nabh): c_D =
//            5.57953523384, c_L = 0.010618948146, dp = 0.11752016697. The
//            flow is marched from rest (smoothly ramped inflow) to steady
//            state; adaptive steps unless -fixed.
//   Case 2 = 2D-2: periodic vortex shedding, Re = 100 (U_m = 1.5). A long
//            run -- expensive tier, launch it by hand.
//   Case 3 = 2D-3: time-dependent inflow U_m(t) = 1.5 sin(pi t / 8) on
//            t in [0, 8] from rest, Re up to 100; c_D, c_L referred to the
//            maximal mean inflow velocity 1. Reference values (V. John,
//            "Reference values for drag and lift of a two-dimensional
//            time-dependent flow around a cylinder", IJNMF 44, 2004):
//            c_D,max = 2.950921575 at t = 3.93625, c_L,max = 0.47795 at
//            t = 5.693125, dp(t = 8) = -0.1116. Fixed steps; the forces are
//            evaluated every step and the extrema located by parabolic
//            interpolation through the three samples around the discrete max.
//
// Usage: mpirun -np <=4 build/cpu/apps/dfg_cylinder [-c 1|2|3] [-o 3]
//        [-ns 4] [-nr 3] [-nd 16] [-dt 0.01] [-tf T] [-fixed]
//        [-rot [-pbj] [-schur cc|tensor|auto] [-rlog N]]
//        [-gd c_gd]
//        [-ip N] [-ai N | -at T] [-maxe N] [-theta t] [-aniso] [-cfl c]
//        [-rtol r] [-apc loramg|jacobi_chebyshev|jacobi_pcg] [-ext 2|3]
//        [-cflt c [-dtmax d]] [-dout] [-mref L] [-out dir]
//
// -mref L regenerates the mesh with every cell count times 2^L and every
// grading ratio to the power 2^-L: the cells of level L split those of level 0
// exactly (geometric sequences nest), with the cylinder exact at each level.
//
// Case 3 ends with one machine-readable "RESULT key=value ..." line (errors
// against John's values, wall time, steps, final element count, the largest
// convective CFL number reached, iteration statistics) -- what the DFG 2D-3
// study (docs/imex_vs_semi_implicit.md) collects -- and writes the per-step
// history <out>/dfg_2d3_history.csv (t, dt, c_D, c_L, outer and velocity-inner
// iterations, step wall time). step_wall is the time spent in Case::Step()
// alone (the scheme's cost); wall adds the force and CFL bookkeeping. A run
// whose forces blow up stops with status=diverged.

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

// Maximum of samples f(t) located by the parabola through the discrete
// maximum and its two neighbours (any spacing -- CFL-controlled steps vary).
// Returns {t*, f*}.
std::pair<double, double> PeakOf(const std::vector<double>& t,
                                 const std::vector<double>& f)
{
   const std::size_t n = f.size();
   if (n == 0) { return {std::nan(""), std::nan("")}; }
   const std::size_t k = static_cast<std::size_t>(
                            std::max_element(f.begin(), f.end()) - f.begin());
   if (k == 0 || k + 1 >= n) { return {t[k], f[k]}; }
   const double t0 = t[k - 1], t1 = t[k], t2 = t[k + 1];
   const double d01 = (f[k] - f[k - 1]) / (t1 - t0);
   const double d12 = (f[k + 1] - f[k]) / (t2 - t1);
   const double a = (d12 - d01) / (t2 - t0); // second divided difference
   if (a >= 0.0) { return {t[k], f[k]}; }
   // f(t) = f0 + d01 (t - t0) + a (t - t0)(t - t1); f'(t*) = 0.
   const double ts = 0.5 * (t0 + t1) - d01 / (2.0 * a);
   return {ts, f[k - 1] + d01* (ts - t0) + a* (ts - t0)* (ts - t1)};
}

} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int bench = 1, order = 3, n_side = 4, n_ring = 3, n_down = 16;
   int initial_passes = 0, amr_interval = 0, max_elements = 20000;
   int rotation_log = 0, ext = 2; // ext: the case-level default (BDF2/EXT2)
   int mesh_level = 0;
   // cfl: Nek-scale CFL ceiling (time/cfl.hpp); 0.24 = the 0.6 this app used
   // in the old k^2 measure.
   double t_final = -1.0, dt = 0.01, cfl = 0.24, grad_div = 0.0;
   double amr_time = 0.0, theta = 0.3, rtol = 1e-10;
   double cfl_target = 0.0, dt_max = 0.0;
   bool rotational = false, pbj = false, dirichlet_out = false, fixed = false;
   bool aniso = false;
   const char* schur = "cc";
   const char* apc = "jacobi_pcg"; // the case-level default
   const char* out = "dfg_out";
   OptionsParser args(argc, argv);
   args.AddOption(&bench, "-c", "--case",
                  "1 = 2D-1 (steady, Re 20), 2 = 2D-2, 3 = 2D-3 (unsteady).");
   args.AddOption(&order, "-o", "--order", "Velocity order k_u (k_p = k_u - 1).");
   args.AddOption(&n_side, "-ns", "--n-side", "O-grid cells per square side.");
   args.AddOption(&n_ring, "-nr", "--n-ring", "Radial ring layers.");
   args.AddOption(&n_down, "-nd", "--n-down", "Downstream cells.");
   args.AddOption(&mesh_level, "-mref", "--mesh-level",
                  "Mesh level L: every cell count x 2^L (nested refinement).");
   args.AddOption(&initial_passes, "-ip", "--initial-passes",
                  "AMR passes on the initial (ramped) flow.");
   args.AddOption(&amr_interval, "-ai", "--amr-interval",
                  "AMR event every N steps (0 = none).");
   args.AddOption(&amr_time, "-at", "--amr-time",
                  "AMR event every T time units (overrides -ai; 0 = none).");
   args.AddOption(&max_elements, "-maxe", "--max-elements",
                  "AMR global element cap.");
   args.AddOption(&theta, "-theta", "--theta",
                  "AMR relative marking threshold (theta * max eta).");
   args.AddOption(&aniso, "-aniso", "--anisotropic", "-iso", "--isotropic",
                  "AMR: anisotropic or isotropic refinement.");
   args.AddOption(&rtol, "-rtol", "--rtol", "Outer FGMRES relative tolerance.");
   args.AddOption(&ext, "-ext", "--ext-order",
                  "Extrapolation order of the nonlinear term: 2 or 3.");
   args.AddOption(&cfl_target, "-cflt", "--cfl-target",
                  "CFL-controlled steps at this CFL number (0 = off); -dt is "
                  "the first step.");
   args.AddOption(&dt_max, "-dtmax", "--dt-max", "CFL mode: dt cap (0 = none).");
   args.AddOption(&apc, "-apc", "--a-pc",
                  "Velocity-block PC of the CC path (convective form, or "
                  "rotational without -pbj): loramg|jacobi_chebyshev|"
                  "jacobi_pcg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time (default 12 / 30 / 8 for cases 1 / 2 / 3).");
   args.AddOption(&dt, "-dt", "--dt",
                  "Step size (fixed) or initial step (adaptive).");
   args.AddOption(&fixed, "-fixed", "--fixed-step", "-adapt", "--adaptive",
                  "Fixed or error-controlled steps (case 3: fixed); -cflt "
                  "overrides both.");
   args.AddOption(&cfl, "-cfl", "--cfl",
                  "Convective CFL ceiling (adaptive convective form only; "
                  "fixed steps never abort on it -- it is reported).");
   args.AddOption(&rotational, "-rot", "--rotational", "-conv", "--convective",
                  "Rotational (semi-implicit) or convective (IMEX) form.");
   args.AddOption(&pbj, "-pbj", "--pbj-krylov", "-sym", "--symmetric-pc",
                  "Rotational form: PBJ-preconditioned GMRES velocity PC.");
   args.AddOption(&schur, "-schur", "--rotation-schur",
                  "Rotational form: pressure Schur PC cc|tensor|auto.");
   args.AddOption(&rotation_log, "-rlog", "--rotation-log",
                  "Rotational form: per-step rotation log every N steps.");
   args.AddOption(&grad_div, "-gd", "--grad-div",
                  "Grad-div scale c_gd (gamma = c_gd h_K; 0 = off).");
   args.AddOption(&dirichlet_out, "-dout", "--dirichlet-outflow", "-nout",
                  "--natural-outflow", "Outflow: the inflow profile imposed "
                  "(Dirichlet) or do-nothing (natural, the benchmark's).");
   args.AddOption(&out, "-out", "--output", "Output directory (CSV logs).");
   args.ParseCheck();
   if (bench == 3) { fixed = true; }
   if (t_final <= 0.0) { t_final = (bench == 1) ? 12.0 : (bench == 2) ? 30.0 : 8.0; }

   const double U_m = (bench == 1) ? 0.3 : 1.5;
   // Coefficient reference velocity: the mean inflow velocity (2/3 U_m); for
   // 2D-3 its maximum over time, 1.
   const double U_ref = 2.0 * U_m / 3.0;
   const double D = 0.1;

   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = rotational ? incns::ConvectiveForm::Rotational
                       : incns::ConvectiveForm::Convective;
   p.rotation_pc = pbj ? incns::RotationVelocityPC::PbjKrylov
                   : incns::RotationVelocityPC::Symmetric;
   {
      using RSP = incns::RotationalSchurPreconditioner;
      if (!std::strcmp(schur, "cc")) { p.rotation_schur.mode = RSP::Mode::CahouetChabard; }
      else if (!std::strcmp(schur, "tensor")) { p.rotation_schur.mode = RSP::Mode::Tensor; }
      else if (!std::strcmp(schur, "auto")) { p.rotation_schur.mode = RSP::Mode::Auto; }
      else { MFEM_ABORT("dfg_cylinder: -schur must be cc, tensor or auto"); }
   }
   if (!std::strcmp(apc, "loramg")) { p.cc.a_pc = incns::APC::LORAMG; }
   else if (!std::strcmp(apc, "jacobi_chebyshev"))
   {
      p.cc.a_pc = incns::APC::JacobiChebyshev;
   }
   else if (!std::strcmp(apc, "jacobi_pcg")) { p.cc.a_pc = incns::APC::JacobiPCG; }
   else
   {
      MFEM_ABORT("dfg_cylinder: -apc must be loramg, jacobi_chebyshev or "
                 "jacobi_pcg");
   }
   p.rotation_log_interval = rotational ? rotation_log : 0;
   p.nu = 1e-3;
   p.grad_div = grad_div;
   p.order_u = order;
   p.order_p = order - 1;
   p.mesh.dim = 2;
   p.dt = dt;
   p.t_final = t_final;
   if (cfl_target > 0.0)
   {
      p.step_control = incns::StepControl::Cfl;
      p.cfl_target = cfl_target;
   }
   else
   {
      p.step_control = fixed ? incns::StepControl::Fixed
                       : incns::StepControl::Error;
   }
   p.dt_max = dt_max;
   p.ext_order = ext;
   p.controller.atol = 1e-6;
   p.controller.rtol = 1e-5;
   // Fixed steps never change dt, and the study runs the convective form past
   // its CFL limit on purpose: the ceiling only steers adaptive runs.
   p.cfl_max = (fixed || cfl_target > 0.0) ? 0.0 : cfl;
   p.krylov_rtol = rtol;
   p.forces.enabled = true;
   p.forces.attributes = {incns::kCylinderBody};
   p.forces.reference_velocity = U_ref;
   p.forces.reference_area = D;
   // Case 3 evaluates the forces itself every step (below); the Case's own
   // CSV log would evaluate them a second time.
   p.forces.interval = (bench == 3) ? 0 : 10;
   p.output.path = out;
   p.output.name = (bench == 1) ? "dfg_2d1" : (bench == 2) ? "dfg_2d2" : "dfg_2d3";
   if (amr_time > 0.0)
   {
      amr_interval = std::max(1, static_cast<int>(std::lround(amr_time / dt)));
   }
   p.amr.enabled = (initial_passes > 0 || amr_interval > 0);
   p.amr.initial_passes = initial_passes;
   p.amr.interval = amr_interval;
   p.amr.theta = theta;
   p.amr.anisotropic = aniso;
   p.amr.max_elements = max_elements;
   p.Normalize();
   incns::ConfigureDevice(p.device);

   incns::CylinderChannelSpec spec;
   MFEM_VERIFY(mesh_level >= 0 && mesh_level <= 4, "dfg_cylinder: -mref 0..4");
   {
      // Level L: counts x 2^L, gradings g -> g^(2^-L). A geometric sequence
      // of n cells with ratio g is split exactly by 2n cells with ratio
      // sqrt(g), so the levels nest.
      const int m = 1 << mesh_level;
      const double root = 1.0 / m;
      spec.n_side = n_side * m;
      spec.n_ring = n_ring * m;
      spec.n_down = n_down * m;
      spec.n_up *= m;
      spec.n_below *= m;
      spec.n_above *= m;
      spec.ring_grading = std::pow(spec.ring_grading, root);
      spec.down_grading = std::pow(spec.down_grading, root);
   }
   spec.order = std::max(order, 2);
   Mesh serial = incns::MakeCylinderChannelMesh(spec);
   std::unique_ptr<ParMesh> mesh = incns::PartitionMesh(serial, p.amr.enabled);

   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   // Parabolic inflow. Cases 1/2: smoothly ramped from rest over t in [0, 1].
   // Case 3: U_m sin(pi t / 8), the benchmark's own time dependence.
   VectorFunctionCoefficient inflow(2, [U_m, bench](const Vector & x, double t,
                                    Vector & u)
   {
      double s = 1.0;
      if (bench == 3) { s = std::sin(M_PI * t / 8.0); }
      else if (t < 1.0) { s = std::sin(0.5 * M_PI * t) * std::sin(0.5 * M_PI * t); }
      u(0) = s * 4.0 * U_m * x(1) * (kH - x(1)) / (kH * kH);
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
   std::vector<double> ts, cds, cls;
   // Per-step history (case 3): dt, outer / velocity-inner iterations, and the
   // wall time of Case::Step() alone.
   std::vector<double> hdt, hstep;
   std::vector<int> houter, hinner;
   double step_wall = 0.0;
   double cd = 0.0, cl = 0.0, cd_prev = 0.0, cfl_seen = 0.0, max_mu = 0.0;
   long long outer_sum = 0, inner_sum = 0, cap_hits = 0;
   int outer_max = 0, tensor_steps = 0;
   bool diverged = false;
   const double wall0 = MPI_Wtime();
   double next_report = 0.0;
   while (!flow.Done())
   {
      const double step0 = MPI_Wtime();
      flow.Step();
      const double step_t = MPI_Wtime() - step0;
      step_wall += step_t;
      ++steps;
      const int its = flow.Integrator().LastIterations();
      outer_sum += its;
      outer_max = std::max(outer_max, its);
      int inner_its = 0;
      if (const incns::StokesSolver* s = flow.Integrator().LastSolver())
      {
         const incns::SolveStats& st = s->Stats();
         inner_its = static_cast<int>(st.vel_inner_iterations);
         inner_sum += st.vel_inner_iterations;
         cap_hits += st.vel_cap_hits;
         tensor_steps += st.tensor_active ? 1 : 0;
         max_mu = std::max(max_mu, st.rotation.max_mu);
      }
      const bool report = (bench == 3) || steps % 20 == 0 || flow.Done();
      if (!report) { continue; }
      const Vector C = flow.ForceCoefficients();
      cd_prev = cd;
      cd = C(0);
      cl = C(1);
      if (!std::isfinite(cd) || !std::isfinite(cl) || std::abs(cd) > 1e3 ||
          std::abs(cl) > 1e3)
      {
         diverged = true;
         if (Mpi::Root())
         {
            std::printf("DIVERGED at t = %.5f (step %d): c_D = %g, c_L = %g\n",
                        flow.Time(), steps, cd, cl);
         }
         break;
      }
      ts.push_back(flow.Time());
      cds.push_back(cd);
      cls.push_back(cl);
      if (bench == 3)
      {
         hdt.push_back(flow.Integrator().CurrentDt());
         hstep.push_back(step_t);
         houter.push_back(its);
         hinner.push_back(inner_its);
      }
      if (steps % 5 == 0) { cfl_seen = std::max(cfl_seen, flow.ConvectiveCflNumber()); }
      if (flow.Time() >= next_report - 1e-12 || flow.Done())
      {
         next_report += (bench == 3) ? 0.5 : 0.0;
         const long long ne = mesh->GetGlobalNE();
         if (Mpi::Root())
         {
            std::printf("t = %8.4f  dt = %.3e  NE = %lld  c_D = %.8f  "
                        "c_L = %+.8f  its = %d  CFL so far = %.2f\n", flow.Time(),
                        flow.Integrator().CurrentDt(), ne, cd, cl, its, cfl_seen);
            std::fflush(stdout);
         }
      }
   }
   const double wall = MPI_Wtime() - wall0;
   const double dp = diverged ? std::nan("") :
                     PointValue(*mesh, flow.Pressure(), 0.15, 0.2) -
                     PointValue(*mesh, flow.Pressure(), 0.25, 0.2);
   const long long ne = mesh->GetGlobalNE();
   if (!Mpi::Root()) { return diverged ? 2 : 0; }

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
   if (bench == 3)
   {
      const double cd_ref = 2.950921575, tcd_ref = 3.93625;
      const double cl_ref = 0.47795, tcl_ref = 5.693125, dp_ref = -0.1116;
      const auto [tcd, cdm] = PeakOf(ts, cds);
      const auto [tcl, clm] = PeakOf(ts, cls);
      const double n = std::max(steps, 1);
      std::printf("RESULT case=3 form=%s schur=%s pc=%s gd=%g dt=%g status=%s "
                  "t=%.6f steps=%d ne=%lld wall=%.2f cd_max=%.8f t_cd=%.6f "
                  "cl_max=%.8f t_cl=%.6f dp8=%.8f err_cd=%.3e err_tcd=%.3e "
                  "err_cl=%.3e err_tcl=%.3e err_dp=%.3e cfl_max=%.3f "
                  "outer_mean=%.2f outer_max=%d vel_inner_mean=%.2f "
                  "cap_hits=%lld tensor_frac=%.3f mu_max=%.3f rtol=%g "
                  "ext=%d cfl_target=%g mref=%d dout=%d step_wall=%.2f\n",
                  rotational ? "rotational" : "convective",
                  rotational ? schur : "-", (rotational && pbj) ? "pbj_krylov"
                  : apc,
                  grad_div, dt, diverged ? "diverged" : "ok", flow.Time(), steps,
                  ne, wall, cdm, tcd, clm, tcl, dp,
                  std::abs(cdm - cd_ref) / cd_ref, std::abs(tcd - tcd_ref),
                  std::abs(clm - cl_ref) / cl_ref, std::abs(tcl - tcl_ref),
                  std::abs(dp - dp_ref) / std::abs(dp_ref), cfl_seen,
                  outer_sum / n, outer_max, inner_sum / n, cap_hits,
                  tensor_steps / n, max_mu, rtol, ext,
                  cfl_target, mesh_level, dirichlet_out ? 1 : 0, step_wall);
      std::filesystem::create_directories(out);
      const std::string hpath = std::string(out) + "/dfg_2d3_history.csv";
      std::ofstream h(hpath);
      h << "t,dt,c_d,c_l,outer,vel_inner,step_wall\n";
      h.precision(16);
      for (std::size_t i = 0; i < hdt.size(); ++i)
      {
         h << ts[i] << "," << hdt[i] << "," << cds[i] << "," << cls[i] << ","
           << houter[i] << "," << hinner[i] << "," << hstep[i] << "\n";
      }
   }
   std::fflush(stdout);
   return diverged ? 2 : 0;
}
