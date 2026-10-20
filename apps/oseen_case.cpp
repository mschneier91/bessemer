// A deck through the semi-implicit / implicit BDF2 Oseen schemes (src/oseen/,
// docs/design/oseen_bdf2.md): the deck's mesh (AMR off), orders, viscosity,
// boundary conditions, initial velocity and forces; the step, the scheme and
// the end time from the command line (the deck's time section is not used).
// Reports the force coefficients' extremes and the convective CFL.
//
//   mpirun -np 16 build/cpu/apps/oseen_case -deck benchmarks/dfg_3d/dfg_3d_3z.yaml
//          -dt 0.005 -tf 8 -s implicit -nl 0 -nlit 2   (one command line)

#include "bc/boundary_conditions.hpp"
#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
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

// The extremum of samples f(t) (max, or min with sign -1), refined by the
// parabola through it and its neighbours. Returns {t*, f*}.
std::pair<double, double> PeakOf(const std::vector<double>& t,
                                 const std::vector<double>& f, double sign)
{
   const std::size_t n = f.size();
   if (n == 0) { return {std::nan(""), std::nan("")}; }
   std::size_t k = 0;
   for (std::size_t i = 1; i < n; i++) { if (sign * f[i] > sign * f[k]) { k = i; } }
   if (k == 0 || k + 1 >= n) { return {t[k], f[k]}; }
   const double t0 = t[k - 1], t1 = t[k], t2 = t[k + 1];
   const double f0 = f[k - 1], f1 = f[k], f2 = f[k + 1];
   const double d01 = (f1 - f0) / (t1 - t0), d12 = (f2 - f1) / (t2 - t1);
   const double a = (d12 - d01) / (t2 - t0);
   if (sign * a >= 0.0) { return {t1, f1}; }
   const double b = d01 - a * (t0 + t1);
   const double ts = -b / (2.0 * a);
   return {ts, f1 + (ts - t1) * (b + a * (ts + t1))};
}

} // namespace

int main(int argc, char* argv[])
{
   incns::ConfigureMpiTransport();
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char* deck = "";
   const char* scheme = "semi";
   const char* amg = "auto";
   int nlit = 10, reuse = 10, print_every = 50, n_inner = 2;
   double dt = 0.01, t_final = -1.0, nl = 1e-3, plin = 1e-1, lin = 1e-6, gd = 1.0;
   bool oss = true, pcd = true;
   OptionsParser args(argc, argv);
   args.AddOption(&deck, "-deck", "--deck", "The deck (YAML).", true);
   args.AddOption(&dt, "-dt", "--dt", "Step size.");
   args.AddOption(&t_final, "-tf", "--t-final", "End time (< 0: the deck's).");
   args.AddOption(&scheme, "-s", "--scheme", "semi | implicit.");
   args.AddOption(&nl, "-nl", "--nl-rtol", "Picard tolerance (<= 0: -nlit iterations).");
   args.AddOption(&nlit, "-nlit", "--nl-maxit", "Picard iteration cap.");
   args.AddOption(&plin, "-plin", "--picard-lin-rtol", "FGMRES tolerance per Picard step.");
   args.AddOption(&lin, "-lin", "--lin-rtol", "FGMRES tolerance (semi-implicit).");
   args.AddOption(&reuse, "-reuse", "--lor-reuse", "LOR rebuild interval (steps).");
   args.AddOption(&gd, "-gd", "--grad-div-nu", "Grad-div gamma = gd * nu.");
   args.AddOption(&oss, "-oss", "--oss", "-no-oss", "--no-oss", "OSS stabilization.");
   args.AddOption(&amg, "-amg", "--amg", "auto | air | amg (velocity V-cycle).");
   args.AddOption(&n_inner, "-ni", "--n-inner", "PCD: CG iterations on A_p.");
   args.AddOption(&pcd, "-pcd", "--pcd", "-cc", "--cahouet-chabard",
                  "Schur: PCD, or Cahouet-Chabard (no N_p).");
   args.AddOption(&print_every, "-pe", "--print-every", "Progress every N steps.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(std::cout); }
      return 1;
   }

   incns::Parameters p = incns::Parameters::LoadYAML(deck);
   const char* env_dev = std::getenv("INCNS_DEVICE");
   incns::ConfigureDevice(env_dev ? env_dev : p.device);
   if (t_final < 0.0) { t_final = p.t_final; }
   p.amr.enabled = false;
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   incns::MixedSpaces spaces(*mesh, p.order_u, p.order_p);
   incns::RuleBook rules;
   incns::BoundaryConditions bc(spaces.Velocity());
   incns::DeckBoundaryConditions deck_bc(p, *mesh);
   deck_bc.Apply(bc, nullptr, /*require_coverage=*/true);
   std::vector<int> body = p.forces.attributes;
   for (const std::string& name : p.forces.boundaries)
   {
      body.push_back(incns::NamedBoundaryAttribute(p, *mesh, name));
   }

   incns::OseenStepperOptions so;
   so.nu = p.nu;
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
   so.pcd_convection = pcd;
   so.amg_mode = std::strcmp(amg, "air") == 0 ? incns::OseenAmgMode::Air
                 : std::strcmp(amg, "amg") == 0 ? incns::OseenAmgMode::Amg
                 : incns::OseenAmgMode::Auto;
   incns::OseenStepper stepper(spaces, rules, bc, so);
   std::unique_ptr<VectorCoefficient> u0 = incns::MakeInitialVelocity(p);
   stepper.SetInitialVelocity(*u0);

   std::unique_ptr<incns::BodyForce> force;
   if (!body.empty())
   {
      force = std::make_unique<incns::BodyForce>(spaces.Velocity(), body);
   }
   incns::ConvectiveCfl cfl(spaces.Velocity(), rules);
   const HYPRE_BigInt ndofs = spaces.GlobalVelocityTDofs() +
                              spaces.GlobalPressureTDofs();
   const long long nelem = mesh->GetGlobalNE(); // collective: outside Root()
   if (Mpi::Root())
   {
      std::printf("oseen_case %s: dim %d, Q%d/Q%d, %lld elements, %lld dofs, nu %g, "
                  "dt %g to t %g, %s", deck, spaces.Dim(), p.order_u, p.order_p,
                  nelem, (long long) ndofs, p.nu, dt, t_final,
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
   double wall = 0.0, wall_lor = 0.0, cfl_max = 0.0;
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
      cfl_max = std::max(cfl_max, dt * cfl.Rate(stepper.Velocity()));
      double cd = 0.0, cl = 0.0;
      if (force)
      {
         stepper.MomentumResidual(r);
         const Vector C = incns::BodyForce::Coefficients(
                             force->Force(r), p.forces.reference_velocity,
                             p.forces.reference_area);
         cd = C(0);
         cl = C(1);
      }
      ts.push_back(stepper.Time());
      cds.push_back(cd);
      cls.push_back(cl);
      if (Mpi::Root() && ((s + 1) % std::max(print_every, 1) == 0 || s + 1 == nsteps))
      {
         std::printf("t %7.4f  c_D %10.6f  c_L %10.6f  its %3d  picard %d  r0 %.2e "
                     "r %.2e  %s Re_h %.1f  step %.3f s (LOR %.3f s)%s\n",
                     stepper.Time(), cd, cl, st.linear_iterations, st.picard_iterations,
                     st.r0, st.r_final, st.air ? "AIR" : "AMG", st.cell_re, st.t_step,
                     st.t_lor, st.converged ? "" : "  NOT CONVERGED");
         std::fflush(stdout);
      }
   }
   const auto cdmax = PeakOf(ts, cds, 1.0);
   const auto clmin = PeakOf(ts, cls, -1.0);
   const auto clmax = PeakOf(ts, cls, 1.0);
   if (Mpi::Root())
   {
      std::printf("RESULT dt %g: c_D,max %.6f at t %.4f | c_L,min %.6f at t %.4f | "
                  "c_L,max %.6f at t %.4f | steps %d, FGMRES its %.2f/step, Picard "
                  "%.2f/step, unconverged %d | CFL max %.2f | wall %.1f s (%.4f s/step, "
                  "LOR %.1f s)\n", dt, cdmax.second, cdmax.first, clmin.second,
                  clmin.first, clmax.second, clmax.first, nsteps,
                  double(lin_total) / nsteps, double(picard_total) / nsteps, failed,
                  cfl_max, wall, wall / nsteps, wall_lor);
   }
   return 0;
}
