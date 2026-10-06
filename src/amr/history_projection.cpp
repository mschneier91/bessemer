#include "amr/history_projection.hpp"

#include "solver/stokes_solver.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

int ProjectHistoryDivergenceFree(MixedSpaces& spaces, const RuleBook& rules,
                                 BoundaryConditions& bc, IntegratorState& state,
                                 const HistoryProjectionOptions& opts)
{
   INCNS_PROFILE("amr::project_history");
   MFEM_VERIFY(state.u_hist.size() == state.times.size() && !state.times.empty(),
               "history_projection: empty or inconsistent history");

   StokesSolverOptions so;
   so.nu = 1e-12;          // mass-dominated block; the operator needs nu > 0
   so.mass_coeff = 1.0;
   so.schur = SchurBlockType::CahouetChabard;
   so.rtol = opts.rtol;
   so.max_iter = opts.max_iter;
   so.kdim = opts.kdim;
   so.print_level = opts.print_level;
   StokesSolver solver(spaces, rules, bc, so);

   const int n_u = spaces.Velocity().GetTrueVSize();
   ParGridFunction v(&spaces.Velocity()), lambda(&spaces.Pressure());
   Vector b(n_u);
   b.UseDevice(true);
   int iterations = 0;
   for (std::size_t j = 0; j < state.u_hist.size(); ++j)
   {
      MFEM_VERIFY(state.u_hist[j].Size() == n_u,
                  "history_projection: history level " << j
                  << " is not on the velocity space");
      bc.SetTime(state.times[j]);
      solver.Blocks().Mass().Mult(state.u_hist[j], b); // M u_j
      // COLD start: FGMRES's relative tolerance is measured against the
      // initial residual, and u_j itself is often already (nearly) the
      // answer -- a warm start would ask for rtol times a roundoff residual.
      v = 0.0;
      lambda = 0.0;
      solver.SolveTrue(b, v, lambda);
      MFEM_VERIFY(solver.Converged(), "history_projection: the projection "
                  "solve for history level " << j << " did not converge");
      iterations += solver.Iterations();
      v.GetTrueDofs(state.u_hist[j]);
   }
   bc.SetTime(state.times[0]);
   return iterations;
}

} // namespace incns
