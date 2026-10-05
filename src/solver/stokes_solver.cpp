#include "solver/stokes_solver.hpp"

#include "post/pressure_mean.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

// Map the solver options to the operator's assembly options. A LOR source is
// built when the velocity block PC is (LOR-)AMG on either Schur path;
// amg_reuse freezes it so its hierarchy survives Delta-t.
static StokesOperatorOptions MakeOpOptions(const StokesSolverOptions& o)
{
   StokesOperatorOptions so;
   so.nu = o.nu;
   so.collocated_mass = o.collocated_mass;
   so.mass_coeff = o.mass_coeff;
   so.grad_div = o.grad_div;
   so.grad_div_scale = o.grad_div_scale;
   const bool cc = (o.schur == SchurBlockType::CahouetChabard);
   so.lagged_velocity = o.lagged_velocity;
   so.rotation_alpha = o.rotation_alpha;
   // Point-block Jacobi replaces the velocity PC: no LOR source then.
   const bool pbj = o.lagged_velocity &&
                    o.rotation_pc != RotationVelocityPC::Symmetric;
   so.lor_momentum =
      !pbj && (cc ? (o.cc.a_pc == APC::LORAMG)
               : (o.velocity_prec == VelocityPreconditioner::LORAMG));
   // Rotation in LOR: the LOR operator changes every step, so it is never
   // frozen (amg_reuse does not apply).
   so.rotation_in_lor = o.rotation_in_lor && so.lor_momentum &&
                        o.lagged_velocity != nullptr;
   so.lor_frozen = so.lor_momentum && o.amg_reuse && !so.rotation_in_lor;
   return so;
}

StokesSolver::StokesSolver(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc,
                           const StokesSolverOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), opts_(opts),
     nullspace_(bc.PressureNullspaceExists()),
     cc_mode_(opts.schur == SchurBlockType::CahouetChabard),
     op_(spaces, rules, MakeOpOptions(opts), &bc.EssentialTrueDofs()),
     block_op_(const_cast<Array<int>&>(spaces.BlockTrueOffsets())),
     schur_(spaces.Pressure(), rules, opts.nu),
     ortho_schur_(spaces.Velocity().GetComm()),
     fgmres_(spaces.Velocity().GetComm())
{
   INCNS_PROFILE("stokes_solver::setup");
   {
      const bool lor_amg = (opts.schur == SchurBlockType::CahouetChabard)
                           ? (opts.cc.a_pc == APC::LORAMG)
                           : (opts.velocity_prec == VelocityPreconditioner::LORAMG);
      MFEM_VERIFY(!opts.rotation_in_lor || !opts.lagged_velocity ||
                  (lor_amg && opts.rotation_pc == RotationVelocityPC::Symmetric),
                  "stokes_solver: rotation_in_lor needs the LOR-AMG velocity PC "
                  "(solver.preconditioner: loramg, or cc a_pc LORAMG) and "
                  "rotation_pc = symmetric");
   }

   // Block system. Mass path (Sprint-1 default): [A, -B^T; B, 0] on physical p
   // (non-symmetric sign choice) -- numerically untouched, baselines intact.
   // CC path: the canonical SYMMETRIC [A, +B^T; B, 0] on the internal pressure
   // p~ = -p_physical (SPEC par.2.2); same equations (substitute p~ = -p), the
   // sign flip to physical pressure happens exactly once at output. B and B^T
   // are Delta-t independent, so they are wired once and never refreshed.
   BT_ = std::make_unique<TransposeOperator>(&op_.Divergence());
   // (0,0) is the full momentum block: with a rotation term that is the
   // nonsymmetric A + N (FGMRES is the outer solver on both paths).
   block_op_.SetBlock(0, 0, &op_.FullMomentum());
   block_op_.SetBlock(0, 1, BT_.get(), cc_mode_ ? 1.0 : -1.0);
   block_op_.SetBlock(1, 0, &op_.Divergence());

   // With the constant null space, the pressure block is P * S^{-1} * P
   // (mfem::OrthoSolver): iterates never accumulate the constant mode. The
   // Sprint-1 Schur block is Delta-t independent, so it too is built once.
   // (Mass path only; the CC PC owns its own nullspace treatment.)
   pressure_block_ = &schur_;
   if (nullspace_)
   {
      ortho_schur_.SetSolver(schur_);
      pressure_block_ = &ortho_schur_;
   }

   if (cc_mode_)
   {
      // Single source of truth: sigma and nu come from THIS solver's options
      // (mass_coeff IS gamma0/dt). Validation happens in the PC constructor.
      CahouetChabardConfig cc = opts_.cc;
      cc.sigma = opts_.mass_coeff;
      cc.nu = opts_.nu;
      cc_pc_ = std::make_unique<CahouetChabardSchurPC>(
                  cc, spaces_.Velocity(), spaces_.Pressure(), rules_,
                  op_.Divergence(), op_.Mass(), op_.MassDiagonal(),
                  bc_.OutflowAttributes(), nullspace_);
   }

   BuildVelocityPreconditioner();

   fgmres_.SetOperator(block_op_);
   fgmres_.SetPreconditioner(*prec_);
   fgmres_.SetRelTol(opts_.rtol);
   fgmres_.SetAbsTol(opts_.atol);
   fgmres_.SetMaxIter(opts_.max_iter);
   fgmres_.SetKDim(opts_.kdim);
   fgmres_.SetPrintLevel(opts_.print_level);
   fgmres_.iterative_mode = true; // start from the Dirichlet-seeded guess
}

void StokesSolver::BuildVelocityPreconditioner()
{
   // Velocity block A-hat^-1. Mass path: matrix-free Jacobi (default) or
   // LOR-BoomerAMG via velocity_prec. CC path: cc.a_pc -- LORAMG (default) or
   // JacobiChebyshev. AMG on the dense high-order operator coarsens poorly, so
   // it is built on a low-order-refined (Q1-on-GLL-nodes) rediscretization that
   // is spectrally equivalent -- LOR tdofs match the HO velocity tdofs, so the
   // same ess-dof list applies. The block system FGMRES applies is the same
   // either way; only this preconditioner block changes.
   //
   // With the rotation term, rotation_pc may replace all of that with
   // point-block Jacobi (it sees N's nodal blocks; Jacobi/LOR-AMG cannot):
   // applied once, or as the PC of a loose inner GMRES on the full block.
   lor_rot_ = nullptr; // set again below if this build carries N in LOR
   const bool use_pbj = op_.Rotation() &&
                        opts_.rotation_pc != RotationVelocityPC::Symmetric;
   const bool want_lor_amg = !use_pbj &&
                             (cc_mode_ ? (opts_.cc.a_pc == APC::LORAMG)
                              : (opts_.velocity_prec == VelocityPreconditioner::LORAMG));
   if (use_pbj)
   {
      vel_prec_.reset(); // a PbjKrylov GMRES borrows pbj_: drop it first
      pbj_ = std::make_unique<PointBlockJacobi>(
                spaces_.Velocity(), *op_.Rotation(), bc_.EssentialTrueDofs());
      pbj_->SetDiagonal(op_.MomentumDiagonal()); // N's diagonal is exactly 0
      pbj_->UpdateSkew();
      if (opts_.rotation_pc == RotationVelocityPC::PbjKrylov)
      {
         auto gmres = std::make_unique<GMRESSolver>(spaces_.Velocity().GetComm());
         gmres->SetKDim(opts_.pbj_krylov_kdim);
         gmres->SetRelTol(opts_.pbj_krylov_rtol);
         gmres->SetAbsTol(0.0);
         gmres->SetMaxIter(opts_.pbj_krylov_max_iter);
         gmres->SetPrintLevel(-1);
         gmres->SetOperator(op_.FullMomentum());
         gmres->SetPreconditioner(*pbj_);
         gmres->iterative_mode = false;
         vel_prec_ = std::move(gmres);
      }
   }
   else if (want_lor_amg && op_.RotationInLor())
   {
      // AMG on the LOR c0 M + nu K + N, on op_'s own LOR discretization (the
      // one its LOR-space w lives on). Drop the old PC first: re-assembly
      // replaces the matrix it was set up on.
      vel_prec_.reset();
      auto lor = std::make_unique<LORSolver<HypreBoomerAMG>>(
                    op_.AssembleLorMomentum(), op_.LorDiscretization());
      lor->GetSolver().SetSystemsOptions(spaces_.Dim(), /*order_bynodes=*/true);
      lor->GetSolver().SetPrintLevel(0);
      lor->GetSolver().iterative_mode = false;
      if (cc_mode_) { lor->GetSolver().SetMaxIter(opts_.cc.a_vcycles); }
      lor_rot_ = lor.get();
      vel_prec_ = std::move(lor);
   }
   else if (want_lor_amg)
   {
      auto lor = std::make_unique<LORSolver<HypreBoomerAMG>>(
                    op_.MomentumLORForm(), bc_.EssentialTrueDofs());
      lor->GetSolver().SetSystemsOptions(spaces_.Dim(), /*order_bynodes=*/true);
      lor->GetSolver().SetPrintLevel(0);
      lor->GetSolver().iterative_mode = false;
      if (cc_mode_) { lor->GetSolver().SetMaxIter(opts_.cc.a_vcycles); }
      vel_prec_ = std::move(lor);
   }
   else if (cc_mode_ && opts_.cc.a_pc == APC::JacobiChebyshev)
   {
      // Fixed-order Chebyshev with the momentum diagonal: cheap, for
      // sigma-dominated regimes (SPEC par.6.3).
      vel_prec_ = std::make_unique<OperatorChebyshevSmoother>(
                     op_.Momentum(), op_.MomentumDiagonal(),
                     bc_.EssentialTrueDofs(), 4,
                     spaces_.Velocity().GetComm());
   }
   else
   {
      vel_prec_ = std::make_unique<OperatorJacobiSmoother>(
                     op_.MomentumDiagonal(), bc_.EssentialTrueDofs());
   }

   // PbjOnly leaves vel_prec_ empty: point-block Jacobi is the velocity PC.
   Solver& vel = vel_prec_ ? *vel_prec_ : *pbj_;
   if (cc_mode_)
   {
      // Block Diag/LowerTri/UpperTri on the symmetric system; the minus of the
      // pressure row lives inside BlockStokesPC (exactly once).
      prec_ = std::make_unique<BlockStokesPC>(
                 spaces_.BlockTrueOffsets(), vel, *cc_pc_,
                 op_.Divergence(), opts_.cc.block_shape);
   }
   else
   {
      prec_ = std::make_unique<StokesBlockPreconditioner>(
                 spaces_.BlockTrueOffsets(), vel, *pressure_block_);
   }
   fgmres_.SetPreconditioner(*prec_);
}

void StokesSolver::Refresh(double c0)
{
   INCNS_PROFILE("stokes_solver::refresh");

   // Only the Delta-t-dependent momentum block changes: reassemble it (cheap --
   // geometric factors are cached) and re-point the block operator (its
   // underlying pointer moved). B, B^T, the Schur block structure, and the
   // FGMRES object all persist untouched.
   op_.SetMassCoeff(c0);
   block_op_.SetBlock(0, 0, &op_.FullMomentum());

   // CC Schur block: sigma tracks the BDF factor; everything structural inside
   // (masses, B, the L_p AMG hierarchy) is reused always (SPEC par.9).
   if (cc_pc_) { cc_pc_->Reset(c0, opts_.nu); }

   // Refresh the velocity preconditioner unless it is a FROZEN AMG hierarchy
   // (lor_frozen: built once at the reference operator, reused across Delta-t).
   // For Jacobi/Chebyshev the new diagonal is near-free; for non-frozen AMG the
   // LOR hierarchy rebuilds. Rebuilding vel_prec_ re-wires the block PC (it
   // borrows the velocity solver), which BuildVelocityPreconditioner does.
   const bool lor_vel =
      cc_mode_ ? (opts_.cc.a_pc == APC::LORAMG)
      : (opts_.velocity_prec == VelocityPreconditioner::LORAMG);
   const bool use_pbj = op_.Rotation() &&
                        opts_.rotation_pc != RotationVelocityPC::Symmetric;
   const bool frozen_amg = !use_pbj && lor_vel && opts_.amg_reuse;
   if (!frozen_amg) { BuildVelocityPreconditioner(); }
}

void StokesSolver::UpdateRotation()
{
   if (!op_.Rotation()) { return; }
   INCNS_PROFILE("stokes_solver::update_rotation");
   op_.Rotation()->UpdateVorticity();
   if (pbj_) { pbj_->UpdateSkew(); }
   if (lor_rot_) { lor_rot_->SetOperator(op_.AssembleLorMomentum()); }
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
   // MFEM-owned true-dof vector, not ParallelAssemble()'s hypre-backed
   // HypreParVector -- see MassWeightedMean (post/pressure_mean.cpp).
   Vector f_true(spaces_.Velocity().GetTrueVSize());
   f_form.ParallelAssemble(f_true);

   // One-shot semantics: cold start from zero.
   u = 0.0;
   p = 0.0;
   SolveTrue(f_true, u, p);
}

void StokesSolver::SolveTrue(const Vector& b_mom, ParGridFunction& u,
                             ParGridFunction& p)
{
   INCNS_PROFILE("stokes_solver::solve_true");

   // Allocate the saddle-point block vectors in the device memory space (no-op
   // on CPU: Device::GetMemoryType() is HOST there) and mark them device-aware,
   // so BlockOperator::Mult and the Krylov work vectors it spawns stay resident
   // on the device instead of triggering per-apply host<->device copies. Found
   // via the debug device (scripts/debug_device.sh), which faulted here.
   const Array<int>& offsets = spaces_.BlockTrueOffsets();
   const MemoryType mt = Device::GetMemoryType();
   BlockVector x(const_cast<Array<int>&>(offsets), mt);
   BlockVector b(const_cast<Array<int>&>(offsets), mt);
   x.UseDevice(true);
   b.UseDevice(true);
   b = 0.0;

   // Dirichlet data (at the BC's current time) -> u's boundary; the incoming
   // u/p act as the Krylov warm start. The eliminated (identity) rows of the
   // block system reproduce the boundary values.
   bc_.ProjectDirichlet(u);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));
   // CC path iterates on the internal pressure p~ = -p_physical (the canonical
   // symmetric convention): flip the warm start in, flip the answer out.
   if (cc_mode_) { x.GetBlock(1).Neg(); }

   {
      INCNS_PROFILE("rhs");
      b.GetBlock(0) = b_mom;

      // Dirichlet elimination on both RHS blocks:
      //   momentum: b_u -= A u_D, then b_u[ess] = u_D[ess];
      //   constraint: b_p = -B u_D  (so that B u_0 + B u_D = 0).
      auto* Ac = dynamic_cast<ConstrainedOperator*>(&op_.Momentum());
      MFEM_VERIFY(Ac, "stokes_solver: momentum block is not constrained");
      // Rotation term FIRST: it subtracts N u_D (its essential rows end up
      // overwritten); the momentum elimination then sets b[ess] = u_D.
      if (op_.Rotation())
      {
         op_.RotationConstrained().EliminateRHS(x.GetBlock(0), b.GetBlock(0));
      }
      Ac->EliminateRHS(x.GetBlock(0), b.GetBlock(0));

      auto* Bc = dynamic_cast<RectangularConstrainedOperator*>(&op_.Divergence());
      MFEM_VERIFY(Bc, "stokes_solver: divergence block is not constrained");
      Bc->EliminateRHS(x.GetBlock(0), b.GetBlock(1));

      // Compatibility: with the constant null space present, the constraint RHS
      // must be orthogonal to the constant mode (any residue is quadrature/
      // roundoff or an incompatible net boundary flux).
      // No RHS projection: the constraint RHS b_p = -B u_D is compatible to
      // machine precision by construction (measured ~1e-17 even for
      // interpolated non-polynomial data -- the discrete boundary flux of
      // admissible Dirichlet data telescopes to roundoff). Genuinely
      // incompatible data (net flux != 0 on an enclosed domain) is an
      // ill-posed input and SHOULD fail loudly (residual floor,
      // Converged() = false) rather than be silently projected away.
   }

   {
      INCNS_PROFILE("fgmres");
      fgmres_.Mult(b, x);
   }
   iterations_ = fgmres_.GetNumIterations();
   converged_ = fgmres_.GetConverged();

   u.SetFromTrueDofs(x.GetBlock(0));
   if (cc_mode_) { x.GetBlock(1).Neg(); } // p = -p~: the ONE output sign flip
   p.SetFromTrueDofs(x.GetBlock(1));

   // Output normalization (mass-weighted mean-zero) -- only meaningful when the
   // pressure level is otherwise undetermined.
   if (nullspace_)
   {
      SubtractMean(p, rules_);
   }
}

} // namespace incns
