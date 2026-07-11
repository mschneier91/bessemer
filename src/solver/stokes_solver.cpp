#include "solver/stokes_solver.hpp"

#include "post/pressure_mean.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesSolver::StokesSolver(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc,
                           const StokesSolverOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), opts_(opts),
     nullspace_(bc.PressureNullspaceExists()),
     op_(spaces, rules,
         StokesOperatorOptions{opts.nu, opts.collocated_mass, opts.mass_coeff},
         &bc.EssentialTrueDofs()),
     block_op_(const_cast<Array<int>&>(spaces.BlockTrueOffsets())),
     schur_(spaces.Pressure(), rules, opts.nu),
     fgmres_(spaces.Velocity().GetComm())
{
   INCNS_PROFILE("stokes_solver::setup");

   // Block system [nu*K, -B^T; B, 0] (non-symmetric sign choice; FGMRES).
   BT_ = std::make_unique<TransposeOperator>(&op_.Divergence());
   block_op_.SetBlock(0, 0, &op_.Momentum());
   block_op_.SetBlock(0, 1, BT_.get(), -1.0);
   block_op_.SetBlock(1, 0, &op_.Divergence());

   prec_ = std::make_unique<StokesBlockPreconditioner>(
              spaces_.BlockTrueOffsets(), op_.MomentumDiagonal(),
              bc_.EssentialTrueDofs(), schur_, nullspace_,
              spaces_.Velocity().GetComm());

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

   const Array<int>& offsets = spaces_.BlockTrueOffsets();
   BlockVector x(const_cast<Array<int>&>(offsets));
   BlockVector b(const_cast<Array<int>&>(offsets));
   x = 0.0;
   b = 0.0;

   // Dirichlet data -> u's boundary, then into the true-dof initial guess. The
   // eliminated (identity) rows of the block system reproduce these values.
   u = 0.0;
   bc_.ProjectDirichlet(u);
   u.GetTrueDofs(x.GetBlock(0));

   {
      INCNS_PROFILE("rhs");

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
      b.GetBlock(0) = *f_true;

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
