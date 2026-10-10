// Flow past a square cylinder in a large domain (Joly, Etienne & Pelletier,
// J. Fluids Struct. 28 (2012) 232-243): the inlet 60 D upstream of the
// square's centre, the outlet 120 D downstream, the sides 60 D away (their
// Fig. 3). Free-stream U on the inlet and sides, do-nothing outflow
// (directional by default with the IMEX convective form), no-slip square.
// Re = U D / nu (U = D = 1). Their 2D values at Re = 200, zero incidence:
// C_D = 1.44, mean C_L = 0.001 (Table 2), C_L,rms = 0.42 and St = 0.151
// (Table 1).
//
// IMEX convective form, CFL-controlled steps, AMR: the run starts on a coarse
// graded mesh (mesh/square_cylinder) and refines by the velocity-gradient
// indicator at fixed SIMULATED-time intervals (-at; step-count events would
// drift as dt adapts), with an absolute tolerance, a minimum cell size and an
// element cap. An asymmetric bump in the initial velocity starts the
// shedding early.
//
// Live tracking: every -pi time units one progress line (t, current dt and
// its min/max since the last line, measured CFL, C_D, C_L, iterations,
// elements, wall time, ETA), and the per-step history <out>/history.csv
// (t, dt, c_d, c_l, outer, ne, step_wall) is flushed. At the end, the
// coefficients are averaged over the last -np shedding periods (between
// up-crossings of C_L through its mean, an integer number of periods as the
// paper does) and one machine-readable RESULT line is printed.
//
// AMR window: events only for -amr-start <= t <= -amr-end. Spin-up only has
// to reach the shedding state, not accurately, so it can run on the coarse
// mesh (few cells, large steps); -amr-burst N runs up to N refinement passes
// at the first event so resolution arrives quickly.
//
// Checkpoints: -chk-at T writes <out>/chk once t >= T; -restart dir continues
// from one (same np; the AMR refinement history is replayed) -- e.g. one
// spin-up, then branches with different CFL targets for a time-step study.
//
// Usage: mpirun -np <=4 build/cpu/apps/square_cylinder [-re 200] [-tf 150]
//        [-cflt 0.5] [-at 1] [-amr-start 0] [-amr-burst 1] [-amr-end 90]
//        [-tol 0.25] [-minh 0.2]
//        [-maxe 3000] [-np 10] [-pi 1] [-chk-at T] [-restart dir] [-out dir]

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/square_cylinder.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

struct Sample
{
   double t, dt, cd, cl;
   int outer;
   long long ne;
   double step_wall;
};

// Time integral of f over [ta, tb] by the trapezoid rule on the samples
// (non-uniform steps), interpolating linearly at the window ends.
double Integral(const std::vector<Sample>& s, double ta, double tb,
                double (*f)(const Sample&))
{
   double sum = 0.0;
   for (std::size_t k = 1; k < s.size(); ++k)
   {
      const double t0 = std::max(s[k - 1].t, ta), t1 = std::min(s[k].t, tb);
      if (t1 <= t0) { continue; }
      const double h = s[k].t - s[k - 1].t;
      auto at = [&](double t)
      {
         const double w = (t - s[k - 1].t) / h;
         return (1.0 - w) * f(s[k - 1]) + w * f(s[k]);
      };
      sum += 0.5 * (t1 - t0) * (at(t0) + at(t1));
   }
   return sum;
}

double Cd(const Sample& s) { return s.cd; }
double Cl(const Sample& s) { return s.cl; }
double Cl2(const Sample& s) { return s.cl * s.cl; }

} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   double re = 200.0, t_final = 150.0, cfl_target = 0.5, dt0 = 0.01;
   double amr_dt = 1.0, amr_start = 0.0, amr_end = 90.0, tol = 0.25, min_h = 0.2;
   double rtol = 1e-8;
   int amr_burst = 1;
   double pert = 0.1, progress_dt = 1.0, grad_div = 0.0;
   int order = 3, max_elements = 3000, n_periods = 10, ext = 2, bdf = 0;
   bool aniso = true, classical_out = false, oifs = false, cmass = false;
   double oifs_cfl = 0.5;
   incns::SquareCylinderSpec spec;
   // Coarse base mesh: AMR does the refinement.
   spec.n_face = 2;
   spec.corner_ratio = 1.0;
   spec.far_ratio = 1.5;
   spec.wake_ratio = 1.3;
   spec.wake_h = 1.0;
   spec.wake_end = 25.0;
   const char* out = "square_out";
   const char* restart = "";
   double chk_at = 0.0;
   OptionsParser args(argc, argv);
   args.AddOption(&re, "-re", "--reynolds", "Reynolds number U D / nu.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time (D/U units).");
   args.AddOption(&cfl_target, "-cflt", "--cfl-target", "CFL target (Nek scale).");
   args.AddOption(&dt0, "-dt", "--dt", "First step (capped by the CFL target).");
   args.AddOption(&ext, "-ext", "--ext-order", "Extrapolation order 2 or 3.");
   args.AddOption(&bdf, "-bdf", "--bdf-order",
                  "BDF order 2 or 3; 0 = auto (3 with OIFS, 2 with IMEX).");
   args.AddOption(&order, "-o", "--order", "Velocity order k_u (k_p = k_u - 1).");
   args.AddOption(&grad_div, "-gd", "--grad-div",
                  "Grad-div scale c_gd (0 = off).");
   args.AddOption(&rtol, "-rtol", "--rtol", "Outer FGMRES relative tolerance.");
   args.AddOption(&amr_dt, "-at", "--amr-time",
                  "AMR event every T time units (0 = none).");
   args.AddOption(&amr_start, "-amr-start", "--amr-start",
                  "No AMR events before this time (spin up on the coarse mesh).");
   args.AddOption(&amr_burst, "-amr-burst", "--amr-burst",
                  "Refinement passes at the first AMR event (until nothing changes).");
   args.AddOption(&amr_end, "-amr-end", "--amr-end",
                  "No AMR events after this time.");
   args.AddOption(&tol, "-tol", "--amr-tol",
                  "AMR absolute tolerance (velocity units).");
   args.AddOption(&min_h, "-minh", "--min-size",
                  "AMR: do not split below this extent.");
   args.AddOption(&max_elements, "-maxe", "--max-elements", "AMR element cap.");
   args.AddOption(&aniso, "-aniso", "--anisotropic", "-iso", "--isotropic",
                  "AMR: anisotropic or isotropic refinement.");
   args.AddOption(&spec.n_face, "-nf", "--n-face",
                  "Base mesh: cells per face (even).");
   args.AddOption(&spec.far_ratio, "-fr", "--far-ratio",
                  "Base mesh: far-field growth.");
   args.AddOption(&spec.wake_h, "-wh", "--wake-h",
                  "Base mesh: near-wake cell width.");
   args.AddOption(&spec.wake_end, "-we", "--wake-end",
                  "Base mesh: near-wake extent.");
   args.AddOption(&pert, "-pert", "--perturbation",
                  "Asymmetric initial bump amplitude (fraction of U).");
   args.AddOption(&oifs, "-oifs", "--oifs", "-imex", "--imex",
                  "Convection: OIFS (advected BDF history, CFL > 1) or IMEX.");
   args.AddOption(&oifs_cfl, "-ocfl", "--oifs-cfl", "OIFS substep CFL number.");
   args.AddOption(&cmass, "-cm", "--collocated-mass", "-gm", "--default-mass",
                  "Velocity mass: GLL collocated (diagonal), or the default "
                  "(consistent under IMEX, collocated under OIFS).");
   args.AddOption(&classical_out, "-cdn", "--classical-do-nothing", "-ddn",
                  "--directional-do-nothing", "Outflow condition.");
   args.AddOption(&n_periods, "-np", "--n-periods",
                  "Shedding periods to average over.");
   args.AddOption(&progress_dt, "-pi", "--progress",
                  "Progress line every T time units.");
   args.AddOption(&chk_at, "-chk-at", "--checkpoint-at",
                  "Write <out>/chk once t >= T (0 = never).");
   args.AddOption(&restart, "-restart", "--restart",
                  "Restart from this checkpoint.");
   args.AddOption(&out, "-out", "--output", "Output directory.");
   args.ParseCheck();
   const double U = 1.0, D = spec.side;

   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.convective_form = incns::ConvectiveForm::Convective;
   p.convection_treatment = oifs ? incns::ConvectionTreatment::Oifs
                            : incns::ConvectionTreatment::Imex;
   p.oifs_cfl = oifs_cfl;
   p.mass = cmass ? incns::VelocityMass::Collocated : incns::VelocityMass::Auto;
   p.outflow = classical_out ? incns::OutflowCondition::Classical
               : incns::OutflowCondition::Directional;
   p.nu = U * D / re;
   p.grad_div = grad_div;
   p.order_u = order;
   p.order_p = order - 1;
   p.mesh.dim = 2;
   p.dt = dt0;
   p.t_final = t_final;
   p.step_control = incns::StepControl::Cfl;
   p.cfl_target = cfl_target;
   p.ext_order = ext;
   p.time_order = bdf;
   // BDF3 extrapolates at order 3 whatever ext says.
   const int ext_eff = std::max(ext, p.BdfOrder());
   p.krylov_rtol = rtol;
   p.forces.enabled = true;
   p.forces.attributes = {incns::kSquareBody};
   p.forces.reference_velocity = U;
   p.forces.reference_area = D;
   p.forces.interval = 0; // evaluated below, every step
   p.amr.enabled = (amr_dt > 0.0);
   p.amr.interval = 0; // events are triggered below, in simulated time
   p.amr.threshold_mode = incns::AmrThreshold::Absolute;
   p.amr.tolerance = tol;
   p.amr.min_size = min_h;
   p.amr.max_elements = max_elements;
   p.amr.anisotropic = aniso;
   p.restart_from = restart;
   p.Normalize();
   incns::ConfigureDevice(p.device);

   Mesh serial = incns::MakeSquareCylinderMesh(spec);
   std::unique_ptr<ParMesh> mesh = incns::PartitionMesh(serial, p.amr.enabled);
   incns::Case flow(*mesh, p);
   incns::BoundaryConditions bc(flow.Spaces().Velocity());
   VectorFunctionCoefficient free_stream(2, [U](const Vector&, Vector & u)
   {
      u(0) = U;
      u(1) = 0.0;
   });
   bc.AddVelocityDirichlet(incns::kSquareInflow, free_stream);
   bc.AddVelocityDirichlet(incns::kSquareSides, free_stream);
   bc.AddOutflow(incns::kSquareOutflow);
   bc.AddNoSlip(incns::kSquareBody);
   flow.SetBoundaryConditions(bc);
   // Impulsive start plus an asymmetric bump behind the square: the
   // symmetric wake would otherwise wait for round-off to start shedding.
   VectorFunctionCoefficient u0(2, [U, pert](const Vector & x, Vector & u)
   {
      u(0) = U;
      u(1) = pert * U * std::exp(-((x(0) - 2.0) * (x(0) - 2.0) + x(1) * x(1)));
   });
   flow.SetInitialVelocity(u0);

   const bool root = Mpi::Root();
   // GetGlobalNE() is collective: every rank calls it, only root prints.
   const long long ne0 = mesh->GetGlobalNE();
   if (root)
   {
      std::filesystem::create_directories(out);
      std::printf("square cylinder: Re %g, nu %g, Q%d/Q%d, base mesh %lld cells, "
                  "CFL target %g (%s), BDF%d/EXT%d, %s mass, outflow %s, AMR every %g for t in "
                  "[%g, %g] (tol %g, min size %g, cap %d, first-event burst %d), "
                  "t_final %g, np %d\n",
                  re, p.nu, order, order - 1, ne0, cfl_target,
                  oifs ? "OIFS" : "IMEX", p.BdfOrder(), ext_eff,
                  p.CollocatedMass() ? "collocated" : "consistent",
                  classical_out ? "classical" : "directional", amr_dt,
                  amr_start, amr_end, tol, min_h, max_elements, amr_burst,
                  t_final, Mpi::WorldSize());
      std::fflush(stdout);
   }
   const std::string hpath = std::string(out) + "/history.csv";
   std::ofstream hist;
   if (root)
   {
      hist.open(hpath);
      hist << "t,dt,c_d,c_l,outer,ne,step_wall\n";
      hist.precision(12);
   }

   std::vector<Sample> s;
   std::size_t flushed = 0;
   double next_amr = amr_dt, next_report = progress_dt;
   double dt_lo = std::numeric_limits<double>::infinity(), dt_hi = 0.0;
   double step_wall = 0.0;
   int events = 0;
   bool diverged = false, first = true, chk_written = false, amr_started = false;
   const double wall0 = MPI_Wtime();
   while (!flow.Done())
   {
      const double w0 = MPI_Wtime();
      flow.Step();
      const double sw = MPI_Wtime() - w0;
      step_wall += sw;
      const Vector C = flow.ForceCoefficients();
      const double t = flow.Time(), dt = flow.Integrator().CurrentDt();
      if (first)
      {
         // A restart starts mid-run: the event and report clocks start there.
         first = false;
         if (amr_dt > 0.0) { next_amr = (std::floor(t / amr_dt) + 1.0) * amr_dt; }
         next_report = (std::floor(t / progress_dt) + 1.0) * progress_dt;
      }
      long long ne = mesh->GetGlobalNE(); // collective (all ranks)
      s.push_back({t, dt, C(0), C(1), flow.Integrator().LastIterations(), ne, sw});
      dt_lo = std::min(dt_lo, dt);
      dt_hi = std::max(dt_hi, dt);
      if (!std::isfinite(C(0)) || std::abs(C(0)) > 1e3)
      {
         diverged = true;
         if (root) { std::printf("DIVERGED at t = %.4f: c_D = %g\n", t, C(0)); }
         break;
      }
      if (chk_at > 0.0 && !chk_written && t >= chk_at)
      {
         chk_written = true;
         flow.WriteCheckpoint(std::string(out) + "/chk");
         if (root) { std::printf("checkpoint written at t = %.4f\n", t); }
      }
      if (amr_dt > 0.0 && t >= next_amr - 1e-12)
      {
         next_amr += amr_dt;
         if (t >= amr_start - 1e-12 && t <= amr_end)
         {
            // The first event may run a burst of passes (until one changes
            // nothing); later events run one.
            const int passes = amr_started ? 1 : std::max(1, amr_burst);
            amr_started = true;
            for (int k = 0; k < passes; ++k)
            {
               const incns::AdaptStats st = flow.Adapt();
               if (st.ne_after == st.ne_before) { break; }
               ++events;
            }
            ne = mesh->GetGlobalNE();
            if (root && passes > 1)
            {
               std::printf("AMR started at t = %.3f: %lld cells\n", t, ne);
            }
         }
      }
      if (t >= next_report - 1e-12 || flow.Done())
      {
         next_report += progress_dt;
         Vector where;
         const double cfl = flow.ConvectiveCflNumber(&where);
         const double wall = MPI_Wtime() - wall0;
         const double eta = (t > 0.0) ? wall * (t_final - t) / t : 0.0;
         if (root)
         {
            std::printf("t %7.2f  dt %.3e [%.3e, %.3e]  CFL %.2f @(%.2f,%.2f)  "
                        "C_D %.4f  C_L %+.4f  its %2d  sub %d  NE %lld  steps %zu  "
                        "wall %.0f s  ETA %.0f min\n", t, dt, dt_lo, dt_hi, cfl,
                        where(0), where(1), C(0), C(1), s.back().outer,
                        flow.Integrator().LastOifsSubsteps(), ne, s.size(), wall,
                        eta / 60.0);
            std::fflush(stdout);
            for (; flushed < s.size(); ++flushed)
            {
               const Sample& q = s[flushed];
               hist << q.t << "," << q.dt << "," << q.cd << "," << q.cl << ","
                    << q.outer << "," << q.ne << "," << q.step_wall << "\n";
            }
            hist.flush();
         }
         dt_lo = std::numeric_limits<double>::infinity();
         dt_hi = 0.0;
      }
   }
   const double wall = MPI_Wtime() - wall0;
   const long long ne_final = mesh->GetGlobalNE(); // collective

   // Averages over the last n_periods shedding periods: up-crossings of C_L
   // through its mean over the second half of the run.
   double cl_level = 0.0;
   {
      const double ta = 0.5 * flow.Time();
      cl_level = Integral(s, ta, flow.Time(), Cl) / (flow.Time() - ta);
   }
   std::vector<double> up;
   for (std::size_t k = 1; k < s.size(); ++k)
   {
      const double a = s[k - 1].cl - cl_level, b = s[k].cl - cl_level;
      if (a < 0.0 && b >= 0.0)
      {
         up.push_back(s[k - 1].t + (s[k].t - s[k - 1].t) * (-a) / (b - a));
      }
   }
   const int nper = std::min<int>(n_periods, static_cast<int>(up.size()) - 1);
   double cd_mean = NAN, cl_mean = NAN, cl_rms = NAN, st = NAN, ta = NAN, tb = NAN;
   double period_spread = NAN;
   if (!diverged && nper >= 2)
   {
      ta = up[up.size() - 1 - nper];
      tb = up.back();
      const double T = tb - ta;
      cd_mean = Integral(s, ta, tb, Cd) / T;
      cl_mean = Integral(s, ta, tb, Cl) / T;
      cl_rms = std::sqrt(Integral(s, ta, tb, Cl2) / T);
      st = D / (U * T / nper);
      double pmin = 1e30, pmax = 0.0;
      for (std::size_t k = up.size() - nper; k < up.size(); ++k)
      {
         pmin = std::min(pmin, up[k] - up[k - 1]);
         pmax = std::max(pmax, up[k] - up[k - 1]);
      }
      period_spread = (pmax - pmin) / (T / nper);
   }
   // dt statistics: whole run and the averaging window.
   double dmin = 1e30, dmax = 0.0, wmin = 1e30, wmax = 0.0;
   for (const Sample& q : s)
   {
      dmin = std::min(dmin, q.dt);
      dmax = std::max(dmax, q.dt);
      if (q.t >= ta && q.t <= tb) { wmin = std::min(wmin, q.dt); wmax = std::max(wmax, q.dt); }
   }
   if (root)
   {
      const double cd_ref = 1.44, clrms_ref = 0.42, st_ref = 0.151;
      std::printf("averaged over %d periods, t in [%.2f, %.2f] (period spread "
                  "%.2f%%): C_D %.4f (paper 1.44, %+.1f%%), mean C_L %+.4f "
                  "(paper 0.001), C_L,rms %.4f (paper 0.42, %+.1f%%), St %.4f "
                  "(paper 0.151, %+.1f%%)\n", nper, ta, tb, 100.0 * period_spread,
                  cd_mean, 100.0 * (cd_mean / cd_ref - 1.0), cl_mean, cl_rms,
                  100.0 * (cl_rms / clrms_ref - 1.0), st,
                  100.0 * (st / st_ref - 1.0));
      std::printf("dt: whole run [%.3e, %.3e], mean %.3e over %zu steps; "
                  "averaging window [%.3e, %.3e]\n", dmin, dmax,
                  s.empty() ? 0.0 : flow.Time() / s.size(), s.size(), wmin, wmax);
      std::printf("RESULT case=square re=%g status=%s t=%.4f steps=%zu ne=%lld "
                  "events=%d wall=%.1f step_wall=%.1f cd=%.5f cl_mean=%.5f "
                  "cl_rms=%.5f st=%.5f err_cd=%.3e err_clrms=%.3e err_st=%.3e "
                  "periods=%d dt_min=%.4e dt_max=%.4e cfl_target=%g bdf=%d "
                  "ext=%d conv=%s mass=%s\n", re,
                  diverged ? "diverged" : "ok", flow.Time(), s.size(),
                  ne_final, events, wall, step_wall, cd_mean, cl_mean,
                  cl_rms, st, std::abs(cd_mean / cd_ref - 1.0),
                  std::abs(cl_rms / clrms_ref - 1.0), std::abs(st / st_ref - 1.0),
                  nper, dmin, dmax, cfl_target, p.BdfOrder(), ext_eff,
                  oifs ? "oifs" : "imex", p.CollocatedMass() ? "collocated" : "consistent");
      std::fflush(stdout);
      for (; flushed < s.size(); ++flushed)
      {
         const Sample& q = s[flushed];
         hist << q.t << "," << q.dt << "," << q.cd << "," << q.cl << ","
              << q.outer << "," << q.ne << "," << q.step_wall << "\n";
      }
   }
   return diverged ? 2 : 0;
}
