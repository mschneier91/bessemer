#include "solver/stokes_solver.hpp"

#include "post/pressure_mean.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

namespace
{
// Remove the component along the constant vector: v <- v - mean(v), plain l2
// over the GLOBAL vector. Used ONCE per solve on the constraint RHS: the
// operator's range excludes the constant mode, so a constant component in b_p
// would sit in the residual forever and block convergence to tight tolerances.
// (The per-apply projection lives in the OrthoSolver wrap of the Schur block;
// the physical mean-zero output shift is post/pressure_mean.)
void SubtractGlobalMean(Vector& v, MPI_Comm comm)
{
   double local[2] = { v.Sum(), static_cast<double>(v.Size()) };
   double global[2] = { 0.0, 0.0 };
   MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm);
   MFEM_VERIFY(global[1] > 0.0, "SubtractGlobalMean: empty global vector");
   v -= global[0] / global[1];
}
} // namespace

StokesSolver::StokesSolver(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc,
                           const StokesSolverOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), opts_(opts),
     nullspace_(bc.PressureNullspaceExists()),
     op_(spaces, rules,
         StokesOperatorOptions{opts.nu, opts.collocated_mass, opts.mass_coeff, opts.grad_div},
         &bc.EssentialTrueDofs()),
     block_op_(const_cast<Array<int>&>(spaces.BlockTrueOffsets())),
     schur_(spaces.Pressure(), rules, opts.nu),
     ortho_schur_(spaces.Velocity().GetComm()),
     fgmres_(spaces.Velocity().GetComm())
{
   INCNS_PROFILE("stokes_solver::setup");

   // Block system [nu*K, -B^T; B, 0] (non-symmetric sign choice; FGMRES).
   BT_ = std::make_unique<TransposeOperator>(&op_.Divergence());
   block_op_.SetBlock(0, 0, &op_.Momentum());
   block_op_.SetBlock(0, 1, BT_.get(), -1.0);
   block_op_.SetBlock(1, 0, &op_.Divergence());

   // With the constant null space, the pressure block is P * S^{-1} * P
   // (mfem::OrthoSolver): iterates never accumulate the constant mode.
   Solver* pressure_block = &schur_;
   if (nullspace_)
   {
      ortho_schur_.SetSolver(schur_);
      pressure_block = &ortho_schur_;
   }
   prec_ = std::make_unique<StokesBlockPreconditioner>(
              spaces_.BlockTrueOffsets(), op_.MomentumDiagonal(),
              bc_.EssentialTrueDofs(), *pressure_block);

   fgmres_.SetOperator(block_op_);
   fgmres_.SetPreconditioner(*prec_);
   fgmres_.SetRelTol(opts_.rtol);
   fgmres_.SetAbsTol(opts_.atol);
   fgmres_.SetMaxIter(opts_.max_iter);
   fgmres_.SetKDim(opts_.kdim);
   fgmres_.SetPrintLevel(opts_.print_level);
   fgmres_.iterative_mode = true; // start from the Dirichlet-seeded guess
}

void StokesSolver::Solve(VectorCoefficient& forcing, ParGridFunction& u,
                         ParGridFunction& p)
{
   INCNS_PROFILE("stokes_solver::solve");

   // Momentum forcing (f, v). Fast assembly per project policy: the device
   // (GPU) path is exercised from day one even though CPU is the target.
   ParLinearForm f_form(&spaces_.Velocity());
   auto* fi = new VectorDomainLFIntegrator(forcing);
   const int dim = spaces_.Dim();
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   fi->SetIntRule(&rules_.Get(geom, 2 * spaces_.OrderU() + 2));
   f_form.AddDomainIntegrator(fi);
   f_form.UseFastAssembly(true);
   f_form.Assemble();
   std::unique_ptr<HypreParVector> f_true(f_form.ParallelAssemble());

   // One-shot semantics: cold start from zero.
   u = 0.0;
   p = 0.0;
   SolveTrue(*f_true, u, p);
}

void StokesSolver::SolveTrue(const Vector& b_mom, ParGridFunction& u,
                             ParGridFunction& p)
{
   INCNS_PROFILE("stokes_solver::solve_true");

   const Array<int>& offsets = spaces_.BlockTrueOffsets();
   BlockVector x(const_cast<Array<int>&>(offsets));
   BlockVector b(const_cast<Array<int>&>(offsets));
   b = 0.0;

   // Dirichlet data (at the BC's current time) -> u's boundary; the incoming
   // u/p act as the Krylov warm start. The eliminated (identity) rows of the
   // block system reproduce the boundary values.
   bc_.ProjectDirichlet(u);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));

   {
      INCNS_PROFILE("rhs");
      b.GetBlock(0) = b_mom;

      // Dirichlet elimination on both RHS blocks:
      //   momentum: b_u -= A u_D, then b_u[ess] = u_D[ess];
      //   constraint: b_p = -B u_D  (so that B u_0 + B u_D = 0).
      auto* Ac = dynamic_cast<ConstrainedOperator*>(&op_.Momentum());
      MFEM_VERIFY(Ac, "stokes_solver: momentum block is not constrained");
      Ac->EliminateRHS(x.GetBlock(0), b.GetBlock(0));

      auto* Bc = dynamic_cast<RectangularConstrainedOperator*>(&op_.Divergence());
      MFEM_VERIFY(Bc, "stokes_solver: divergence block is not constrained");
      Bc->EliminateRHS(x.GetBlock(0), b.GetBlock(1));

      // Compatibility: with the constant null space present, the constraint RHS
      // must be orthogonal to the constant mode (any residue is quadrature/
      // roundoff or an incompatible net boundary flux).
      if (nullspace_)
      {
         SubtractGlobalMean(b.GetBlock(1), spaces_.Velocity().GetComm());
      }
   }

   {
      INCNS_PROFILE("fgmres");
      fgmres_.Mult(b, x);
   }
   iterations_ = fgmres_.GetNumIterations();
   converged_ = fgmres_.GetConverged();

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   // Output normalization (mass-weighted mean-zero) -- only meaningful when the
   // pressure level is otherwise undetermined.
   if (nullspace_)
   {
      SubtractMean(p, rules_);
   }
}

} // namespace incns
