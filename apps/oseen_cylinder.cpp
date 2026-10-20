// DFG 2D-3 (Schaefer & Turek; V. John, Int. J. Numer. Meth. Fluids 44 (2004))
// with the semi-implicit or implicit BDF2 Oseen schemes (src/oseen/,
// docs/design/oseen_bdf2.md): the channel (0, 2.2) x (0, 0.41) without the
// disk of radius 0.05 at (0.2, 0.2), nu = 1e-3, inflow 6 y (0.41 - y) / 0.41^2
// sin(pi t / 8) (U_m = 1.5), no-slip walls and cylinder, do-nothing outflow,
// t in (0, 8]. Reference: c_D,max 2.950921575 at t 3.93625, c_L,max 0.47795 at
// t 5.693125, dp(8) = p(0.15, 0.2) - p(0.25, 0.2) = -0.1116. Fixed steps.
//
//   mpirun -np 1 build/cpu/apps/oseen_cylinder -o 3 -dt 0.01 -s semi
//   INCNS_DEVICE=cuda mpirun -np 1 build/cuda/apps/oseen_cylinder -s implicit -nlit 2 -nl 0

#include "bc/boundary_conditions.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "oseen/oseen_stepper.hpp"
#include "post/body_force.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/cfl.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
      p.HostRead();
      val = p.GetValue(elems[0], ips[0]);
      found = 1.0;
   }
   double buf[2] = {val, found};
   MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return buf[1] > 0.0 ? buf[0] / buf[1] : std::nan("");
}

// The maximum of samples f(t), refined by the parabola through the discrete
// maximum and its neighbours. Returns {t*, f*}.
std::pair<double, double> PeakOf(const std::vector<double>& t,
                                 const std::vector<double>& f)
{
   const std::size_t n = f.size();
   if (n == 0) { return {std::nan(""), std::nan("")}; }
   const std::size_t k = static_cast<std::size_t>(
                            std::max_element(f.begin(), f.end()) - f.begin());
   if (k == 0 || k + 1 >= n) { return {t[k], f[k]}; }
   const double t0 = t[k - 1], t1 = t[k], t2 = t[k + 1];
   const double f0 = f[k - 1], f1 = f[k], f2 = f[k + 1];
   const double d01 = (f1 - f0) / (t1 - t0), d12 = (f2 - f1) / (t2 - t1);
   const double a = (d12 - d01) / (t2 - t0);
   if (a >= 0.0) { return {t1, f1}; }
   const double b = d01 - a * (t0 + t1);
   const double ts = -b / (2.0 * a);
   return {ts, f1 + (ts - t1)* (b + a * (ts + t1))};
}

} // namespace

int main(int argc, char* argv[])
{
   incns::ConfigureMpiTransport();
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 3, mref = 0, nlit = 10, reuse = 10, print_every = 50, n_inner = 2;
   double dt = 0.01, t_final = 8.0, nl = 1e-3, plin = 1e-1, lin = 1e-6, gd = 1.0;
   const char* scheme = "semi";
   const char* amg = "auto";
   const char* device = "cpu";
   bool oss = true;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Velocity order k (pressure k - 1).");
   args.AddOption(&mref, "-mref", "--mesh-level",
                  "Mesh level 0..4 (counts x 2^L).");
   args.AddOption(&dt, "-dt", "--dt", "Step size.");
   args.AddOption(&t_final, "-tf", "--t-final", "End time.");
   args.AddOption(&scheme, "-s", "--scheme", "semi | implicit.");
   args.AddOption(&nl, "-nl", "--nl-rtol",
                  "Picard tolerance (<= 0: -nlit iterations).");
   args.AddOption(&nlit, "-nlit", "--nl-maxit", "Picard iteration cap.");
   args.AddOption(&plin, "-plin", "--picard-lin-rtol",
                  "FGMRES tolerance per Picard step.");
   args.AddOption(&lin, "-lin", "--lin-rtol", "FGMRES tolerance (semi-implicit).");
   args.AddOption(&reuse, "-reuse", "--lor-reuse",
                  "LOR rebuild interval (steps).");
   args.AddOption(&gd, "-gd", "--grad-div-nu", "Grad-div gamma = gd * nu.");
   args.AddOption(&oss, "-oss", "--oss", "-no-oss", "--no-oss",
                  "OSS stabilization.");
   args.AddOption(&amg, "-amg", "--amg", "auto | air | amg (velocity V-cycle).");
   args.AddOption(&n_inner, "-ni", "--n-inner", "PCD: CG iterations on A_p.");
   args.AddOption(&print_every, "-pe", "--print-every", "Progress every N steps.");
   args.AddOption(&device, "-d", "--device", "cpu | cuda (INCNS_DEVICE wins).");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }
   const char* env_dev = std::getenv("INCNS_DEVICE");
   incns::ConfigureDevice(env_dev ? env_dev : device);

   incns::CylinderChannelSpec spec;
   {
      const int m = 1 << mref;
      const double root = 1.0 / m;
      spec.n_side *= m;
      spec.n_ring *= m;
      spec.n_down *= m;
      spec.n_up *= m;
      spec.n_below *= m;
      spec.n_above *= m;
      spec.ring_grading = std::pow(spec.ring_grading, root);
      spec.down_grading = std::pow(spec.down_grading, root);
   }
   spec.order = std::max(order, 2);
   Mesh serial = incns::MakeCylinderChannelMesh(spec);
   std::unique_ptr<ParMesh> mesh = incns::PartitionMesh(serial, false);

   incns::MixedSpaces spaces(*mesh, order, order - 1);
   incns::RuleBook rules;
   incns::BoundaryConditions bc(spaces.Velocity());
   const double U_m = 1.5;
   VectorFunctionCoefficient inflow(2, [U_m](const Vector & x, double t,
                                    Vector & u)
   {
      u(0) = std::sin(M_PI * t / 8.0) * 4.0 * U_m * x(1) * (kH - x(1)) / (kH * kH);
      u(1) = 0.0;
   });
   bc.AddVelocityDirichlet(incns::kCylinderInflow, inflow);
   bc.AddNoSlip(incns::kCylinderWalls);
   bc.AddNoSlip(incns::kCylinderBody);
   bc.AddOutflow(incns::kCylinderOutflow);

   incns::OseenStepperOptions so;
   so.nu = 1e-3;
   so.dt = dt;
   so.scheme = std::strcmp(scheme, "implicit") == 0 ? incns::OseenScheme::Implicit
               : incns::OseenScheme::SemiImplicit;
   so.nl_rtol = nl;
   so.nl_maxit = nlit;
   so.picard_lin_rtol = plin;
   so.lin_rtol = lin;
   so.ns_reuse = reuse;
   so.grad_div_nu = gd;
   so.oss = oss;
   so.n_inner = n_inner;
   so.amg_mode = std::strcmp(amg, "air") == 0 ? incns::OseenAmgMode::Air
                 : std::strcmp(amg, "amg") == 0 ? incns::OseenAmgMode::Amg
                 : incns::OseenAmgMode::Auto;
   incns::OseenStepper stepper(spaces, rules, bc, so);
   VectorFunctionCoefficient zero(2, [](const Vector&, Vector & u) { u = 0.0; });
   stepper.SetInitialVelocity(zero);

   incns::BodyForce body(spaces.Velocity(), {incns::kCylinderBody});
   // bessemer's convective CFL (Nek5000's definition, as dfg_cylinder reports).
   incns::ConvectiveCfl cfl(spaces.Velocity(), rules);
   double cfl_max = 0.0;
   const HYPRE_BigInt ndofs = spaces.GlobalVelocityTDofs() +
                              spaces.GlobalPressureTDofs();
   const long long nelem = mesh->GetGlobalNE(); // collective: outside Root()
   if (Mpi::Root())
   {
      std::printf("oseen_cylinder: Q%d/Q%d, level %d, %lld elements, %lld dofs, dt %g, %s",
                  order, order - 1, mref, nelem, (long long) ndofs, dt,
                  so.scheme == incns::OseenScheme::Implicit ? "implicit" : "semi-implicit");
      if (so.scheme == incns::OseenScheme::Implicit)
      {
         std::printf(" (Picard: %s %g, max %d, linear %g)", nl > 0 ? "tol" : "fixed",
                     nl, nlit, plin);
      }
      std::printf(", oss %d, grad-div %g nu, np %d\n", oss ? 1 : 0, gd,
                  Mpi::WorldSize());
      std::fflush(stdout);
   }

   std::vector<double> ts, cds, cls;
   long lin_total = 0, picard_total = 0;
   int failed = 0;
   double wall = 0.0, wall_lor = 0.0;
   Vector r;
   const int nsteps = static_cast<int>(std::lround(t_final / dt));
   for (int s = 0; s < nsteps; s++)
   {
      stepper.Step();
      const auto& st = stepper.LastStats();
      lin_total += st.linear_iterations;
      picard_total += st.picard_iterations;
      failed += st.converged ? 0 : 1;
      wall += st.t_step;
      wall_lor += st.t_lor;
      stepper.MomentumResidual(r);
      const Vector F = body.Force(r);
      const Vector C = incns::BodyForce::Coefficients(F, 1.0, 0.1);
      const double cfl_now = dt * cfl.Rate(stepper.Velocity()); // collective
      cfl_max = std::max(cfl_max, cfl_now);
      ts.push_back(stepper.Time());
      cds.push_back(C(0));
      cls.push_back(C(1));
      if (Mpi::Root() && ((s + 1) % std::max(print_every, 1) == 0 || s + 1 == nsteps))
      {
         std::printf("t %7.4f  c_D %9.6f  c_L %9.6f  its %3d  picard %d  r0 %.2e "
                     "r %.2e  %s Re_h %.1f  step %.3f s (LOR %.3f s)%s\n",
                     stepper.Time(), C(0), C(1), st.linear_iterations,
                     st.picard_iterations, st.r0, st.r_final, st.air ? "AIR" : "AMG",
                     st.cell_re, st.t_step, st.t_lor, st.converged ? "" : "  NOT CONVERGED");
         std::fflush(stdout);
      }
   }
   const double dp = PointValue(*mesh, stepper.Pressure(), 0.15, 0.2) -
                     PointValue(*mesh, stepper.Pressure(), 0.25, 0.2);
   const auto cdm = PeakOf(ts, cds);
   const auto clm = PeakOf(ts, cls);
   if (Mpi::Root())
   {
      std::printf("RESULT dt %g: c_D,max %.9f at t %.5f (ref 2.950921575 at 3.93625, "
                  "err %.2e) | c_L,max %.6f at t %.5f (ref 0.47795 at 5.693125, err %.2e) | "
                  "dp(8) %.6f (ref -0.1116) | steps %d, FGMRES its %.2f/step, Picard %.2f/step, "
                  "unconverged %d | CFL max %.2f | wall %.1f s (%.4f s/step, LOR %.1f s)\n",
                  dt, cdm.second, cdm.first, (cdm.second - 2.950921575) / 2.950921575,
                  clm.second, clm.first, (clm.second - 0.47795) / 0.47795, dp, nsteps,
                  double(lin_total) / nsteps, double(picard_total) / nsteps, failed, cfl_max,
                  wall, wall / nsteps, wall_lor);
   }
   return 0;
}
