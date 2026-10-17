#include "solver/stokes_solver.hpp"

#include "post/pressure_mean.hpp"
#include "precond/hypre_device.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

namespace
{
// x is split into blocks of the sizes @p offsets gives.
bool HasBlockLayout(const BlockVector& x, const Array<int>& offsets)
{
   if (x.NumBlocks() != offsets.Size() - 1) { return false; }
   for (int i = 0; i < x.NumBlocks(); ++i)
   {
      if (x.GetBlock(i).Size() != offsets[i + 1] - offsets[i]) { return false; }
   }
   return true;
}
} // namespace

// Map the solver options to the operator's assembly options. A LOR source is
// built when the velocity block PC is (LOR-)AMG on either Schur path;
// amg_reuse freezes it so its hierarchy survives Delta-t.
static StokesOperatorOptions MakeOpOptions(const StokesSolverOptions& o)
{
   StokesOperatorOptions so;
   so.nu = o.nu;
   so.collocated_mass = o.collocated_mass;
   so.scratch = o.scratch;
   so.mass_coeff = o.mass_coeff;
   so.grad_div = o.grad_div;
   so.grad_div_scale = o.grad_div_scale;
   so.grad_div_nu = o.grad_div_nu;
   so.grad_div_nu_ref = o.grad_div_nu_ref;
   so.compact_diffusion = o.compact_diffusion;
   so.fuse_grad_div = o.fuse_grad_div;
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

void MonitoredSolver::Mult(const Vector& x, Vector& y) const
{
   // The (1,1) block's preconditioner, or the Schur complement's.
   INCNS_PROFILE(vel_ ? "velocity_block::apply" : "schur_block::apply");
   const double t0 = MPI_Wtime();
   s_.Mult(x, y);
   const double dt = MPI_Wtime() - t0;
   if (vel_)
   {
      ++stats_.vel_applications;
      stats_.vel_time += dt;
      if (it_)
      {
         stats_.vel_inner_iterations += it_->GetNumIterations();
         if (!it_->GetConverged()) { ++stats_.vel_cap_hits; }
      }
   }
   else
   {
      ++stats_.schur_applications;
      stats_.schur_time += dt;
   }
}

StokesSolver::StokesSolver(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc,
                           const StokesSolverOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), opts_(opts),
     nullspace_(bc.PressureNullspaceExists()), pressure_mean_(rules),
     cc_mode_(opts.schur == SchurBlockType::CahouetChabard),
     op_(spaces, rules, MakeOpOptions(opts), &bc.EssentialTrueDofs()),
     block_op_(const_cast<Array<int>&>(spaces.BlockTrueOffsets())),
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
      MFEM_VERIFY(opts.outer == OuterMethod::Coupled ||
                  opts.schur == SchurBlockType::CahouetChabard,
                  "stokes_solver: solver.outer schur needs the Cahouet-Chabard "
                  "Schur preconditioner (solver.schur: cc or laplacian_legacy)");
      MFEM_VERIFY(opts.outer == OuterMethod::Coupled || !opts.lagged_velocity,
                  "stokes_solver: solver.outer schur needs a symmetric momentum "
                  "block: not with the rotational form (use solver.outer: "
                  "coupled)");
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
   // (DeviceOrthoSolver, precond/nullspace.hpp): iterates never accumulate
   // the constant mode. The Sprint-1 Schur block is Delta-t independent, so it too is built once.
   // (Mass path only; the CC PC owns its own nullspace treatment.)
   if (!cc_mode_)
   {
      schur_ = std::make_unique<PressureMassSchur>(spaces.Pressure(), rules,
               opts.nu);
      pressure_block_ = schur_.get();
      if (nullspace_)
      {
         ortho_schur_.SetSolver(*schur_);
         pressure_block_ = &ortho_schur_;
      }
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
      // Grad-div raises the Schur complement's plateau from nu to nu + gamma.
      if (opts_.cc.schur_grad_div && op_.GradDivCoefficient())
      {
         cc_pc_->SetGradDiv(*op_.GradDivCoefficient());
         // The plain pressure mass is then unused, unless the rotation-aware
         // Schur PC (below) needs it.
         if (!op_.Rotation()) { cc_pc_->ReleasePlainMass(); }
      }
   }

   // Rotation term: the rotation-number statistics and viscous ratio (Auto
   // switching / per-step log) and the rotation-aware Schur PC.
   if (op_.Rotation())
   {
      using RSP = RotationalSchurPreconditioner;
      const int dim = spaces_.Dim();
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const bool switching = (opts_.rotation_schur.mode == RSP::Mode::Auto);
      if (opts_.rotation_diagnostics || switching)
      {
         // The spec's sample rule: Gauss-Legendre of order 2 k_u.
         rot_number_ = std::make_unique<RotationNumber>(
                          spaces_.Velocity(),
                          rules_.Get(geom, 2 * spaces_.OrderU()));
      }
      if (opts_.rotation_diagnostics)
      {
         vratio_ = std::make_unique<ViscousRatioDiagnostic>(spaces_.Velocity(),
                   rules_);
      }
      if (opts_.rotation_schur.mode != RSP::Mode::CahouetChabard)
      {
         MFEM_VERIFY(cc_mode_, "stokes_solver: rotation_schur tensor/auto needs "
                     "the Cahouet-Chabard Schur path (solver.schur: cc)");
         RSP::Options ro = opts_.rotation_schur;
         ro.remove_mean = cc_pc_->Singular();
         // L_T on the pressure space at the viscous-rule convention 2k + dim-1,
         // with the Laplacian's own Dirichlet (outflow) dofs; the tensor mode
         // reuses CC's mass inverse and LOR-AMG L_p V-cycle (spec 0).
         rot_schur_ = std::make_unique<RSP>(
                         spaces_.Pressure(), cc_pc_->Laplacian().EssentialTrueDofs(),
                         rules_.Get(geom, 2 * spaces_.OrderP() + dim - 1),
                         *opts_.lagged_velocity, op_.Rotation()->GetAlpha(),
                         cc_pc_->NuPc(), *cc_pc_, cc_pc_->PressureMassInverse(),
                         cc_pc_->Laplacian(), ro);
      }
   }

   BuildVelocityPreconditioner();

   fgmres_.SetOperator(block_op_);
   if (prec_) { fgmres_.SetPreconditioner(*prec_); } // none under Schur
   fgmres_.SetRelTol(opts_.rtol);
   fgmres_.SetAbsTol(opts_.atol);
   fgmres_.SetMaxIter(opts_.max_iter);
   fgmres_.SetKDim(opts_.kdim);
   kdim_ = opts_.kdim;
   fgmres_.SetPrintLevel(opts_.print_level);
   fgmres_.iterative_mode = true; // start from the Dirichlet-seeded guess
}

void StokesSolver::BuildVelocityPreconditioner()
{
   // Velocity block A-hat^-1. Mass path: matrix-free Jacobi (default) or
   // LOR-BoomerAMG via velocity_prec. CC path: cc.a_pc -- LORAMG, JacobiChebyshev
   // or JacobiPCG. AMG on the dense high-order operator coarsens poorly, so
   // it is built on a low-order-refined (Q1-on-GLL-nodes) rediscretization that
   // is spectrally equivalent -- LOR tdofs match the HO velocity tdofs, so the
   // same ess-dof list applies. The block system FGMRES applies is the same
   // either way; only this preconditioner block changes.
   //
   // With the rotation term, rotation_pc may replace all of that with
   // point-block Jacobi (it sees N's nodal blocks; Jacobi/LOR-AMG cannot):
   // applied once, or as the PC of a loose inner GMRES on the full block.
   INCNS_PROFILE("stokes_solver::velocity_pc");
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
                spaces_.Velocity(), *op_.Rotation(), op_.EssentialTrueDofs());
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
      VerifyHypreAmgOnDevice(lor->GetSolver(),
                             "stokes_solver: the velocity LOR-AMG");
      lor_rot_ = lor.get();
      vel_prec_ = std::move(lor);
   }
   else if (want_lor_amg)
   {
      auto lor = std::make_unique<LORSolver<HypreBoomerAMG>>(
                    op_.MomentumLORForm(), op_.EssentialTrueDofs());
      lor->GetSolver().SetSystemsOptions(spaces_.Dim(), /*order_bynodes=*/true);
      lor->GetSolver().SetPrintLevel(0);
      lor->GetSolver().iterative_mode = false;
      if (cc_mode_) { lor->GetSolver().SetMaxIter(opts_.cc.a_vcycles); }
      VerifyHypreAmgOnDevice(lor->GetSolver(),
                             "stokes_solver: the velocity LOR-AMG");
      vel_prec_ = std::move(lor);
   }
   else if (cc_mode_ && opts_.cc.a_pc == APC::JacobiPCG)
   {
      // CG with Jacobi on the SPD momentum block (the symmetric part with the
      // rotation term) to a loose tolerance. A tolerance-based inner solve is
      // nonlinear -- legal because the outer solver is FGMRES. SetOperator
      // before SetPreconditioner: MFEM forwards SetOperator to the PC.
      // Its preconditioner: scalar Jacobi, or point-block Jacobi with the
      // grad-div term's coupling of the components at each node in the blocks
      // (the default when there is a grad-div term). The blocks are symmetric
      // positive definite here -- no skew part -- so CG stays legal.
      const bool pbj_pc =
         opts_.cc.a_pcg_pc == APcgPC::PointBlockJacobi ||
         (opts_.cc.a_pcg_pc == APcgPC::Auto && op_.GradDivTerm());
      // The Schur method runs its own velocity CG on vel_inner_pc_ (see
      // SchurVelocityPreconditioner): this one, and its three velocity
      // vectors, only for the coupled method.
      const bool own_cg = opts_.outer == OuterMethod::Coupled;
      auto* kept_pbj = dynamic_cast<PointBlockJacobi*>(vel_inner_pc_.get());
      auto* kept_cg = dynamic_cast<CGSolver*>(vel_prec_.get());
      if (pbj_pc && kept_pbj && (kept_cg || !own_cg))
      {
         // A refresh (Delta-t or nu): only the diagonal moved. The grad-div
         // off-diagonals depend on neither and cost ~0.8 s to rebuild at
         // 64^3 Q3 on an H100 (2026-10-14), so the blocks are updated in
         // place and the CG re-pointed at the rebuilt momentum operator.
         kept_pbj->SetDiagonal(op_.MomentumDiagonal());
         if (kept_cg) { kept_cg->SetOperator(op_.Momentum()); }
      }
      else
      {
         vel_prec_.reset(); // it borrows vel_inner_pc_
         vel_inner_pc_.reset(); // before its replacement is built, not after
         if (pbj_pc)
         {
            auto pbj = std::make_unique<PointBlockJacobi>(
                          spaces_.Velocity(), op_.EssentialTrueDofs());
            pbj->SetDiagonal(op_.MomentumDiagonal());
            if (op_.GradDivTerm()) { pbj->SetGradDiv(*op_.GradDivTerm()); }
            vel_inner_pc_ = std::move(pbj);
         }
         else
         {
            vel_inner_pc_ = std::make_unique<OperatorJacobiSmoother>(
                               op_.MomentumDiagonal(), op_.EssentialTrueDofs());
         }
         if (own_cg)
         {
            auto cg = std::make_unique<CGSolver>(spaces_.Velocity().GetComm());
            cg->SetOperator(op_.Momentum());
            cg->SetPreconditioner(*vel_inner_pc_);
            cg->SetRelTol(opts_.cc.a_pcg_rtol);
            cg->SetAbsTol(0.0);
            cg->SetMaxIter(opts_.cc.a_pcg_max_iter);
            cg->SetPrintLevel(-1);
            cg->iterative_mode = false;
            vel_prec_ = std::move(cg);
         }
      }
   }
   else if (cc_mode_ && opts_.cc.a_pc == APC::JacobiChebyshev)
   {
      // Fixed-order Chebyshev with the momentum diagonal: cheap, for
      // sigma-dominated regimes (SPEC par.6.3).
      vel_prec_ = std::make_unique<OperatorChebyshevSmoother>(
                     op_.Momentum(), op_.MomentumDiagonal(),
                     op_.EssentialTrueDofs(), 4,
                     spaces_.Velocity().GetComm());
   }
   else
   {
      vel_prec_ = std::make_unique<OperatorJacobiSmoother>(
                     op_.MomentumDiagonal(), op_.EssentialTrueDofs());
   }

   // The Schur method applies the velocity PC and the Cahouet-Chabard PC
   // itself: no block preconditioner (it holds a velocity-sized scratch
   // vector) and no velocity-block monitor.
   if (opts_.outer == OuterMethod::Schur)
   {
      vel_mon_.reset();
      prec_.reset();
      if (!schur_mon_)
      {
         schur_mon_ = std::make_unique<MonitoredSolver>(*cc_pc_, stats_, false);
      }
      WireSchurSolver();
      return;
   }

   // PbjOnly leaves vel_prec_ empty: point-block Jacobi is the velocity PC.
   // Both blocks go through MonitoredSolver (counts and wall time only -- no
   // numerics; see SolveStats).
   prec_.reset(); // it borrows the monitors being replaced
   Solver& vel = vel_prec_ ? *vel_prec_ : *pbj_;
   vel_mon_ = std::make_unique<MonitoredSolver>(vel, stats_, true);
   if (cc_mode_)
   {
      // Block Diag/LowerTri/UpperTri on the symmetric system; the minus of the
      // pressure row lives inside BlockStokesPC (exactly once). With the
      // rotation term the Schur PC may be the rotation-aware one.
      Solver& schur = rot_schur_ ? static_cast<Solver&>(*rot_schur_)
                      : static_cast<Solver&>(*cc_pc_);
      schur_mon_ = std::make_unique<MonitoredSolver>(schur, stats_, false);
      prec_ = std::make_unique<BlockStokesPC>(
                 spaces_.BlockTrueOffsets(), *vel_mon_, *schur_mon_,
                 op_.Divergence(), opts_.cc.block_shape);
   }
   else
   {
      schur_mon_ = std::make_unique<MonitoredSolver>(*pressure_block_, stats_,
                   false);
      prec_ = std::make_unique<StokesBlockPreconditioner>(
                 spaces_.BlockTrueOffsets(), *vel_mon_, *schur_mon_);
   }
   fgmres_.SetPreconditioner(*prec_);
}

Solver& StokesSolver::SchurVelocityPreconditioner()
{
   // JacobiPCG: its CG's preconditioner, not the CG.
   if (opts_.cc.a_pc == APC::JacobiPCG) { return *vel_inner_pc_; }
   return *vel_prec_; // Jacobi, Chebyshev or a LOR-AMG V-cycle
}

void StokesSolver::WireSchurSolver()
{
   // The Schur PC is the monitored one (counts and time in SolveStats).
   Solver& a_pc = SchurVelocityPreconditioner();
   if (schur_solver_)
   {
      schur_solver_->Rewire(op_.FullMomentum(), a_pc, *schur_mon_);
      return;
   }
   if (cc_pc_->Singular()) // the CC PC's resolved null-space flag
   {
      schur_proj_ = std::make_unique<ConstantPressureProjector>(
                       spaces_.Pressure());
   }
   schur_solver_ = std::make_unique<SchurSolver>(
                      spaces_.Velocity().GetComm(), op_.FullMomentum(),
                      op_.Divergence(), a_pc, *schur_mon_, schur_proj_.get(),
                      opts_.uzawa);
   schur_solver_->SetMaxIter(opts_.max_iter);
   schur_solver_->SetPrintLevel(opts_.print_level);
   // The CC PC is applied only between velocity solves: its B M_v^-1 B^T
   // runs in the velocity CG's two work vectors instead of two of its own.
   cc_pc_->BorrowVelocityScratch(schur_solver_->VelocityWorkD(),
                                 schur_solver_->VelocityWorkZ());
   // initial_guess project: the pressure-space projection, inside the Schur
   // solver (2 projection_k + 2 pressure vectors, no system vector).
   if (opts_.initial_guess == InitialGuess::Project)
   {
      schur_solver_->EnablePressureProjection(opts_.projection_k);
   }
}

void StokesSolver::Refresh(double c0)
{
   INCNS_PROFILE("stokes_solver::refresh");

   // Only the Delta-t-dependent momentum block changes: reassemble it (cheap --
   // geometric factors are cached) and re-point the block operator (its
   // underlying pointer moved). B, B^T, the Schur block structure, and the
   // FGMRES object all persist untouched.
   op_.SetMassCoeff(c0);
   opts_.mass_coeff = c0; // sigma of the rotation statistics / Schur update
   // The Schur method's pressure projection holds images of the old S.
   if (schur_solver_) { schur_solver_->ResetGuess(); }
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
   else if (schur_solver_) { WireSchurSolver(); } // the momentum block moved
}

void StokesSolver::WarmUp()
{
   // The Schur method has no block preconditioner, and no Krylov basis for
   // the warm-up's measurement to size (CapKrylovDim skips it).
   if (warmed_up_ || !prec_) { return; }
   INCNS_PROFILE("stokes_solver::warm_up");
   const Array<int>& offsets = spaces_.BlockTrueOffsets();
   const MemoryType mt = Device::GetMemoryType();
   BlockVector r(const_cast<Array<int>&>(offsets), mt);
   BlockVector z(const_cast<Array<int>&>(offsets), mt);
   r.UseDevice(true);
   z.UseDevice(true);
   // Random, not constant: with a pressure null space a constant pressure
   // is projected away and the pressure AMG would never run.
   r.Randomize(1 + Mpi::WorldRank());
   block_op_.Mult(r, z);
   prec_->Mult(r, z);
   warmed_up_ = true;
}

int StokesSolver::GuessVectors() const
{
   // ToleranceReference::Step keeps A x_prev and works in one more system
   // vector and two velocity-sized ones: counted as four.
   const int step =
      (opts_.tolerance_reference == ToleranceReference::Step) ? 4 : 0;
   switch (opts_.initial_guess)
   {
      case InitialGuess::Project: return 2 * opts_.projection_k + step;
      case InitialGuess::Extrapolate: return 2 + step;
      default: return step;
   }
}

void StokesSolver::PrepareGuess(const Vector& b, BlockVector& x)
{
   if (opts_.initial_guess == InitialGuess::Previous) { return; }
   // The Schur method projects the pressure itself (SchurSolver).
   if (schur_solver_ && opts_.initial_guess == InitialGuess::Project)
   {
      return;
   }
   INCNS_PROFILE("stokes_solver::initial_guess");
   // The elimination put the Dirichlet values in x: keep them.
   const Array<int>& ess = op_.EssentialTrueDofs();
   Vector& u_dirichlet = guess_dirichlet_;
   u_dirichlet.UseDevice(true);
   if (ess.Size() > 0) { x.GetBlock(0).GetSubVector(ess, u_dirichlet); }
   bool guessed = false;
   const double c0 = opts_.mass_coeff;
   if (opts_.initial_guess == InitialGuess::Project && projection_ &&
       projection_->Size() > 0)
   {
      // Images of an operator whose mass factor moved by more than
      // projection_rebuild_threshold (default 5 %), or whose viscosity
      // changed, are recomputed: m operator applications.
      if (projection_stale_ ||
          (projection_c0_ > 0.0 && std::abs(c0 / projection_c0_ - 1.0) >
           opts_.projection_rebuild_threshold))
      {
         projection_->Rebuild(block_op_);
         projection_c0_ = c0;
         projection_stale_ = false;
      }
      guessed = projection_->Guess(b, x);
   }
   else if (opts_.initial_guess == InitialGuess::Extrapolate)
   {
      // x holds x_n (the warm start). x_0 = x_n + r (x_n - x_{n-1}) with
      // r = dt_new / dt_old, taken as c0_old / c0_new.
      Vector x_n(x.Size(), Device::GetDeviceMemoryType());
      x_n.UseDevice(true);
      x_n = x;
      if (previous_x_.Size() == x.Size())
      {
         const double r = (c0 > 0.0 && previous_c0_ > 0.0) ? previous_c0_ / c0
                          : 1.0;
         x *= 1.0 + r;
         x.Add(-r, previous_x_);
         guessed = true;
      }
      else
      {
         previous_x_.SetSize(x.Size(), Device::GetDeviceMemoryType());
         previous_x_.UseDevice(true);
      }
      previous_x_ = x_n;
      previous_x_valid_ = true;
   }
   if (guessed && ess.Size() > 0) { x.GetBlock(0).SetSubVector(ess, u_dirichlet); }
}

void StokesSolver::SetTolerance(const Vector& b, const Vector& x)
{
   // Step from the warm start itself: the solver's initial residual IS
   // b - A x_prev then. So too under the Schur method with the projection,
   // which guesses inside SchurSolver after computing that residual in its
   // own work vectors: no system-sized residual vector, no extra apply.
   const bool step_as_initial =
      opts_.tolerance_reference == ToleranceReference::Step &&
      (opts_.initial_guess == InitialGuess::Previous ||
       (schur_solver_ && opts_.initial_guess == InitialGuess::Project));
   if (opts_.tolerance_reference == ToleranceReference::Rhs)
   {
      const double b_norm =
         std::sqrt(InnerProduct(spaces_.Velocity().GetComm(), b, b));
      tol_scale_ = b_norm;
      tol_rel_ = 0.0;
      tol_abs_ = std::max(opts_.atol, opts_.rtol * b_norm);
   }
   else if (opts_.tolerance_reference == ToleranceReference::Step &&
            !step_as_initial)
   {
      const double r_norm = StepResidualNorm(b, x);
      tol_scale_ = r_norm;
      tol_rel_ = 0.0;
      tol_abs_ = std::max(opts_.atol, opts_.rtol * r_norm);
   }
   else
   {
      // InitialResidual, or Step from the warm start itself (above).
      tol_scale_ = 0.0;
      tol_rel_ = opts_.rtol;
      tol_abs_ = opts_.atol;
   }
   // The same rule for either outer method (the Schur one computes ||r_0||
   // itself when tol_rel_ > 0).
   fgmres_.SetRelTol(tol_rel_);
   fgmres_.SetAbsTol(tol_abs_);
   if (schur_solver_) { schur_solver_->SetTolerance(tol_rel_, tol_abs_); }
}

double StokesSolver::StepResidualNorm(const Vector& b, const Vector& x)
{
   // ||b - A x_prev||: how far this step moves the solution -- small at a
   // small dt, so the tolerance follows the step, unlike ||b|| (which holds
   // c0 M u_n). With the projection, A x_prev is the image the projection
   // made when it recorded x_prev (no operator application); if the mass
   // factor moved since (the step size changed), (c0 - c0_prev) M x_u is
   // added on the free velocity rows -- the momentum block's other terms and
   // B, B^T do not depend on dt. x_prev's Dirichlet values are the previous
   // step's, so the essential rows hold their change. Otherwise (the first
   // solve, a new viscosity, the rotation term, other initial guesses) the
   // operator is applied to the warm start @p x.
   Vector& r = step_r_;
   r.SetSize(b.Size());
   r.UseDevice(true);
   const bool reuse = opts_.step_reference_reuse && step_ax_valid_ &&
                      step_ax_.Size() == b.Size() && step_nu_ == opts_.nu &&
                      !op_.Rotation();
   if (reuse)
   {
      r = step_ax_;
      const double dc0 = opts_.mass_coeff - step_c0_;
      if (dc0 != 0.0)
      {
         INCNS_PROFILE("stokes_solver::step_reference");
         const Array<int>& ess = op_.EssentialTrueDofs();
         const int n_u = spaces_.Velocity().GetTrueVSize();
         Vector xu;
         xu.MakeRef(const_cast<Vector&>(x), 0, n_u);
         step_xu_.SetSize(n_u);
         step_xu_.UseDevice(true);
         step_xu_ = xu;
         if (ess.Size() > 0) { step_xu_.SetSubVector(ess, 0.0); }
         step_mx_.SetSize(n_u);
         step_mx_.UseDevice(true);
         op_.Mass().Mult(step_xu_, step_mx_);
         if (ess.Size() > 0) { step_mx_.SetSubVector(ess, 0.0); }
         Vector ru;
         ru.MakeRef(r, 0, n_u);
         ru.Add(dc0, step_mx_);
      }
   }
   else
   {
      INCNS_PROFILE("stokes_solver::step_reference");
      block_op_.Mult(x, r);
   }
   subtract(b, r, r);
   return std::sqrt(InnerProduct(spaces_.Velocity().GetComm(), r, r));
}

void StokesSolver::RecordSolution(const Vector& x)
{
   const double c0 = opts_.mass_coeff;
   if (opts_.initial_guess == InitialGuess::Project && !schur_solver_)
   {
      INCNS_PROFILE("stokes_solver::initial_guess");
      if (!projection_)
      {
         projection_ = std::make_unique<SolutionProjection>(
                          opts_.projection_k, spaces_.Velocity().GetComm());
      }
      const bool keep_ax =
         opts_.tolerance_reference == ToleranceReference::Step;
      projection_->Add(block_op_, x, keep_ax ? &step_ax_ : nullptr);
      if (keep_ax)
      {
         step_ax_valid_ = true;
         step_c0_ = c0;
         step_nu_ = opts_.nu;
      }
      if (projection_->Size() <= 1) // a new basis
      {
         projection_c0_ = c0;
         projection_stale_ = false;
      }
   }
   previous_c0_ = c0;
}

namespace
{
// A non-owning alias of v.
void Alias(const Vector& v, Vector& a)
{
   a.MakeRef(const_cast<Vector&>(v), 0, v.Size());
}
} // namespace

void StokesSolver::ExportGuessState(SolverGuessState& s) const
{
   s.projection_x.clear();
   s.projection_w.clear();
   // The Schur method's projection is on the pressure space; it travels in
   // the same fields, flagged (projection_pressure).
   const SolutionProjection* proj =
      schur_solver_ ? schur_solver_->Projection() : projection_.get();
   s.projection_pressure = schur_solver_ != nullptr;
   if (proj)
   {
      const std::vector<Vector>& xs = proj->Solutions();
      const std::vector<Vector>& ws = proj->Images();
      s.projection_x.resize(xs.size());
      s.projection_w.resize(ws.size());
      for (std::size_t i = 0; i < xs.size(); ++i)
      {
         Alias(xs[i], s.projection_x[i]);
         Alias(ws[i], s.projection_w[i]);
      }
   }
   s.projection_c0 = projection_c0_;
   s.projection_stale = projection_stale_;
   s.previous_x.Destroy();
   s.has_previous_x = previous_x_valid_;
   if (previous_x_valid_) { Alias(previous_x_, s.previous_x); }
   s.previous_c0 = previous_c0_;
   s.step_ax.Destroy();
   s.has_step_ax = step_ax_valid_;
   if (step_ax_valid_) { Alias(step_ax_, s.step_ax); }
   s.step_c0 = step_c0_;
   s.step_nu = step_nu_;
}

void StokesSolver::ImportGuessState(SolverGuessState& s)
{
   const int n = SystemSize();
   auto check = [n](const Vector & v)
   {
      MFEM_VERIFY(v.Size() == 0 || v.Size() == n,
                  "stokes_solver: a restored guess vector holds " << v.Size()
                  << " entries, the system " << n);
   };
   // The Schur method's pressure projection (see ExportGuessState).
   SolutionProjection* pressure_proj =
      schur_solver_ ? schur_solver_->Projection() : nullptr;
   const int n_p = spaces_.Pressure().GetTrueVSize();
   const bool pressure_basis = s.projection_pressure;
   if (pressure_basis)
   {
      for (const Vector& v : s.projection_x)
      {
         MFEM_VERIFY(v.Size() == n_p, "stokes_solver: a mixed restored basis");
      }
      for (const Vector& v : s.projection_w)
      {
         MFEM_VERIFY(v.Size() == n_p, "stokes_solver: a mixed restored basis");
      }
   }
   else
   {
      for (const Vector& v : s.projection_x) { check(v); }
      for (const Vector& v : s.projection_w) { check(v); }
   }
   check(s.previous_x);
   check(s.step_ax);

   if (pressure_proj)
   {
      // A coupled run's basis does not fit the pressure projection: the
      // restarted run starts it afresh.
      if (pressure_basis) { pressure_proj->Restore(s.projection_x, s.projection_w); }
      else { pressure_proj->Reset(); }
   }
   if (opts_.initial_guess == InitialGuess::Project && !schur_solver_ &&
       !pressure_basis && !s.projection_x.empty())
   {
      if (!projection_)
      {
         projection_ = std::make_unique<SolutionProjection>(
                          opts_.projection_k, spaces_.Velocity().GetComm());
      }
      projection_->Restore(s.projection_x, s.projection_w);
   }
   else if (projection_) { projection_->Reset(); }
   projection_c0_ = s.projection_c0;
   projection_stale_ = s.projection_stale;

   if (opts_.initial_guess == InitialGuess::Extrapolate)
   {
      previous_x_.Swap(s.previous_x);
      previous_x_.UseDevice(true);
   }
   else { previous_x_.Destroy(); }
   previous_x_valid_ = opts_.initial_guess == InitialGuess::Extrapolate &&
                       s.has_previous_x;
   previous_c0_ = s.previous_c0;

   // The flag, not the size: an empty rank must take the same path as the
   // others (StepResidualNorm may apply the operator otherwise).
   const bool step = opts_.tolerance_reference == ToleranceReference::Step &&
                     s.has_step_ax;
   if (step)
   {
      step_ax_.Swap(s.step_ax);
      step_ax_.UseDevice(true);
   }
   step_ax_valid_ = step;
   step_c0_ = s.step_c0;
   step_nu_ = s.step_nu;
}

void StokesSolver::SetKrylovDim(int k)
{
   MFEM_VERIFY(k >= 1, "stokes_solver: the restart length must be >= 1");
   fgmres_.SetKDim(k);
   kdim_ = k;
}

bool StokesSolver::CanChangeViscosity()
{
   // gamma follows nu only through its nu-proportional part, and not at all
   // when that part is tied to a reference viscosity.
   const bool gamma_fixed =
      opts_.grad_div_nu_ref > 0.0 ||
      (opts_.grad_div_nu == 0.0 &&
       (opts_.grad_div == 0.0 || opts_.grad_div_scale == GradDivScale::OrderH));
   return cc_mode_ && !op_.Rotation() && !op_.LorFrozen() && gamma_fixed;
}

void StokesSolver::Refresh(double c0, double nu)
{
   if (nu != opts_.nu)
   {
      MFEM_VERIFY(CanChangeViscosity(), "stokes_solver: this solver cannot "
                  "change its viscosity (see CanChangeViscosity)");
      op_.SetViscosity(nu);
      opts_.nu = nu; // the Schur preconditioner's nu, in Refresh(c0) below
      projection_stale_ = true; // its images are of the old operator
   }
   Refresh(c0);
}

void StokesSolver::UpdateRotation()
{
   if (!op_.Rotation()) { return; }
   INCNS_PROFILE("stokes_solver::update_rotation");
   op_.Rotation()->UpdateVorticity();
   if (pbj_) { pbj_->UpdateSkew(); }
   // S = B (A + N)^-1 B^T moved with N: the pressure projection's images are
   // of the old one.
   if (schur_solver_) { schur_solver_->ResetGuess(); }
   if (lor_rot_) { lor_rot_->SetOperator(op_.AssembleLorMomentum()); }

   // Diagnostics and the Schur mode of this step (sigma = 0, steady: none).
   const double sigma = opts_.mass_coeff;
   stats_.sigma = sigma;
   if (sigma <= 0.0) { return; }
   if (rot_number_)
   {
      stats_.rotation = rot_number_->Compute(*opts_.lagged_velocity,
                                             op_.Rotation()->GetAlpha(), sigma,
                                             opts_.rotation_schur.mu_on);
   }
   if (vratio_) { stats_.vhat_max = vratio_->VHatMax(sigma, opts_.nu); }
   if (rot_schur_)
   {
      rot_schur_->Update(sigma, stats_.rotation);
      stats_.tensor_active = rot_schur_->TensorActive();
      stats_.schur_switches = rot_schur_->NumSwitches();
   }
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
   // Kept between solves (locals cost an allocation and a synchronizing free
   // per solve on a GPU), rebuilt when the block layout changes. Compared
   // block by block, not by total size: on a rank that owns no elements the
   // offsets are {0, 0, 0}, the size of a never-built BlockVector, which has
   // no blocks to return.
   const Array<int>& offsets = spaces_.BlockTrueOffsets();
   if (!HasBlockLayout(solve_x_, offsets))
   {
      const MemoryType mt = Device::GetMemoryType();
      solve_x_.Update(const_cast<Array<int>&>(offsets), mt);
      solve_b_.Update(const_cast<Array<int>&>(offsets), mt);
      solve_x_.UseDevice(true);
      solve_b_.UseDevice(true);
   }
   BlockVector& x = solve_x_;
   BlockVector& b = solve_b_;
   b = 0.0;

   // Dirichlet data (at the BC's current time) -> u's boundary; the incoming
   // u/p act as the Krylov warm start. The eliminated (identity) rows of the
   // block system reproduce the boundary values. Constant data (no-slip) is
   // projected once (on the host: a device round trip of u) and then written
   // on the essential true dofs, the only ones the projection sets.
   const Array<int>& ess = op_.EssentialTrueDofs();
   if (bc_.DirichletIsConstant() && dirichlet_cached_)
   {
      u.GetTrueDofs(x.GetBlock(0));
      if (ess.Size() > 0) { x.GetBlock(0).SetSubVector(ess, dirichlet_vals_); }
   }
   else
   {
      bc_.ProjectDirichlet(u);
      u.GetTrueDofs(x.GetBlock(0));
      if (bc_.DirichletIsConstant())
      {
         dirichlet_vals_.UseDevice(true);
         if (ess.Size() > 0) { x.GetBlock(0).GetSubVector(ess, dirichlet_vals_); }
         dirichlet_cached_ = true;
      }
   }
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

   SetTolerance(b, x); // before PrepareGuess: Step reads the warm start
   PrepareGuess(b, x);

   // Per-solve counters (the rotation fields persist from UpdateRotation).
   stats_.vel_applications = stats_.vel_inner_iterations = 0;
   stats_.vel_cap_hits = stats_.schur_applications = 0;
   stats_.vel_time = stats_.schur_time = 0.0;
   const double t0 = MPI_Wtime();
   if (schur_solver_)
   {
      // Krylov on the pressure Schur complement; same eliminated system,
      // same warm start, same coupled-residual target.
      schur_solver_->Mult(b, x);
      const SchurSolveStats& s = schur_solver_->Stats();
      iterations_ = s.iterations;
      converged_ = schur_solver_->Converged();
      stats_.final_residual = s.final_residual;
      stats_.tolerance = s.tolerance;
      stats_.schur_outer = true;
      stats_.uzawa = s;
   }
   else
   {
      {
         INCNS_PROFILE("fgmres");
         fgmres_.Mult(b, x);
      }
      iterations_ = fgmres_.GetNumIterations();
      converged_ = fgmres_.GetConverged();
      stats_.final_residual = fgmres_.GetFinalNorm();
      stats_.tolerance = std::max(tol_abs_, tol_rel_ * fgmres_.GetInitialNorm());
      stats_.schur_outer = false;
   }
   stats_.solve_time = MPI_Wtime() - t0;
   ++solves_;
   stats_.outer_iterations = iterations_;

   RecordSolution(x); // in the solver's variables, before the sign flip
   u.SetFromTrueDofs(x.GetBlock(0));
   if (cc_mode_) { x.GetBlock(1).Neg(); } // p = -p~: the ONE output sign flip
   p.SetFromTrueDofs(x.GetBlock(1));

   // Output normalization (mass-weighted mean-zero) -- only meaningful when the
   // pressure level is otherwise undetermined.
   if (nullspace_)
   {
      pressure_mean_.Subtract(p);
   }
}

} // namespace incns
