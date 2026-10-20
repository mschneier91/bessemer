/**
 * @file oseen_stepper.cpp
 * @brief OseenStepper.
 */
#include "oseen/oseen_stepper.hpp"

#include <cmath>

namespace incns
{

using namespace mfem;

namespace
{

/// The velocity (block 0) or pressure (block 1) part of a block true-dof
/// vector, as an alias.
void Block(const Vector& x, const Array<int>& offsets, int b, Vector& part)
{
   part.MakeRef(const_cast<Vector&>(x), offsets[b], offsets[b + 1] - offsets[b]);
}

} // namespace

OseenStepper::OseenStepper(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc, const OseenStepperOptions& opts)
   : spaces_(spaces), rules_(rules), bc_(bc), opts_(opts),
     u_(&spaces.Velocity()), p_(&spaces.Pressure()), wind_gf_(&spaces.Velocity())
{
   ParFiniteElementSpace& vfes = spaces.Velocity();
   ParFiniteElementSpace& pfes = spaces.Pressure();
   dim_ = spaces.Dim();
   offsets_ = spaces.BlockTrueOffsets();
   nu_ = offsets_[1] - offsets_[0];
   np_ = offsets_[2] - offsets_[1];
   ess_ = bc.EssentialTrueDofs();

   StokesOperatorOptions sop;
   sop.nu = opts.nu;
   sop.collocated_mass = true;
   sop.mass_coeff = 1.0 / opts.dt;
   sop.grad_div = 0.0;
   sop.grad_div_nu = opts.grad_div_nu;
   op_ = std::make_unique<StokesOperator>(spaces, rules, sop, &ess_);
   c0_ = sop.mass_coeff;

   sfes_ = std::make_unique<ParFiniteElementSpace>(vfes.GetParMesh(),
           vfes.FEColl());
   wind_ = std::make_unique<OseenWindData>(vfes, rules, opts.nu);
   nscalar_ = std::make_unique<ScalarOseenOperator>(*sfes_, *wind_, opts.oss);
   N_ = std::make_unique<ComponentwiseOperator>(*nscalar_, dim_);
   Nc_ = std::make_unique<ConstrainedOperator>(N_.get(), ess_, false,
         Operator::DIAG_ZERO);

   OseenLorPreconditioner::Options lo;
   lo.nu = opts.nu;
   lo.mode = opts.amg_mode;
   lor_pc_ = std::make_unique<OseenLorPreconditioner>(vfes, rules, ess_, lo);
   OseenPcdSchurPC::Options po;
   po.nu = opts.nu;
   po.nu_tilde = opts.nu * (1.0 + opts.grad_div_nu);
   po.n_inner = opts.n_inner;
   po.oss = opts.oss;
   pcd_ = std::make_unique<OseenPcdSchurPC>(vfes, pfes, rules, op_->Divergence(),
          bc.OutflowAttributes(),
          bc.PressureNullspaceExists(), po);
   pc_ = std::make_unique<BlockStokesPC>(offsets_, *lor_pc_, *pcd_,
                                         op_->Divergence(), BlockPCShape::UpperTri);
   fgmres_ = std::make_unique<FGMRESSolver>(vfes.GetComm());
   fgmres_->SetKDim(opts.kdim);
   fgmres_->SetMaxIter(opts.max_iter);
   fgmres_->SetAbsTol(0.0);
   fgmres_->SetPrintLevel(opts.print_level);
   fgmres_->iterative_mode = false;
   fgmres_->SetPreconditioner(*pc_);
   BuildSystem();

   const int n = offsets_.Last();
   for (Vector* v : {&X_, &Xn_, &Xnm1_, &R_, &D_})
   {
      v->SetSize(n);
      v->UseDevice(true);
      *v = 0.0;
   }
   for (Vector* v : {&b_, &hist_, &W_})
   {
      v->SetSize(nu_);
      v->UseDevice(true);
      *v = 0.0;
   }
}

OseenStepper::~OseenStepper() = default;

void OseenStepper::BuildSystem()
{
   F_ = std::make_unique<SumOperator>(&op_->Momentum(), 1.0, Nc_.get(), 1.0,
                                      false, false);
   Func_ = std::make_unique<SumOperator>(&op_->MomentumUnconstrained(), 1.0,
                                         N_.get(), 1.0, false, false);
   Bt_ = std::make_unique<TransposeOperator>(&op_->Divergence());
   sys_ = std::make_unique<BlockOperator>(offsets_);
   sys_->SetBlock(0, 0, F_.get());
   sys_->SetBlock(0, 1, Bt_.get());
   sys_->SetBlock(1, 0, &op_->Divergence());
   fgmres_->SetOperator(*sys_);
}

void OseenStepper::SetInitialVelocity(VectorCoefficient& u0)
{
   t_ = 0.0;
   step_ = 0;
   bc_.SetTime(0.0);
   u_.ProjectCoefficient(u0);
   bc_.ProjectDirichlet(u_);
   Vector U;
   u_.GetTrueDofs(U);
   Xn_ = 0.0;
   Vector xu;
   Block(Xn_, offsets_, 0, xu);
   Xn_.ReadWrite();
   xu = U;
   Xnm1_ = Xn_;
   p_ = 0.0;
}

double OseenStepper::Norm(const Vector& v) const
{
   return std::sqrt(InnerProduct(spaces_.Velocity().GetComm(), v, v));
}

void OseenStepper::SetWind(const Vector& W)
{
   wind_gf_.SetFromTrueDofs(W);
   wind_->Update(wind_gf_);
   nscalar_->Assemble();
   pcd_->Update(wind_gf_, c0_);
}

void OseenStepper::Residual(const Vector& X, Vector& R) const
{
   Vector xu, xp, ru, rp;
   X.Read();
   R.Write();
   Block(X, offsets_, 0, xu);
   Block(X, offsets_, 1, xp);
   Block(R, offsets_, 0, ru);
   Block(R, offsets_, 1, rp);
   // r_u = b - (F u + B^T p~), r_p = -B u, the Dirichlet rows of r_u zeroed.
   Func_->Mult(xu, ru);
   op_->DivergenceUnconstrained().AddMultTranspose(xp, ru);
   ru -= b_;
   ru.Neg();
   op_->DivergenceUnconstrained().Mult(xu, rp);
   rp.Neg();
   ru.SetSubVector(ess_, 0.0);
}

void OseenStepper::MomentumResidual(Vector& r) const
{
   Vector xu, xp;
   Block(Xn_, offsets_, 0, xu);
   Block(Xn_, offsets_, 1, xp);
   r.SetSize(nu_);
   r.UseDevice(true);
   Func_->Mult(xu, r);
   op_->DivergenceUnconstrained().AddMultTranspose(xp, r);
   r -= b_;
}

void OseenStepper::Step()
{
   StopWatch sw;
   sw.Start();
   stats_ = Stats();
   const double dt = opts_.dt;
   const double t_new = t_ + dt;
   const bool first = (step_ == 0);
   const bool semi = (opts_.scheme == OseenScheme::SemiImplicit);
   const double c0 = first ? 1.0 / dt : 1.5 / dt;
   if (c0 != c0_)
   {
      op_->SetMassCoeff(c0);
      c0_ = c0;
      BuildSystem();
      lor_age_ = 0;
   }

   // b = M (u^n / dt) or M (4 u^n - u^{n-1}) / (2 dt).
   Vector un, unm1;
   Block(Xn_, offsets_, 0, un);
   Block(Xnm1_, offsets_, 0, unm1);
   if (first) { hist_.Set(1.0 / dt, un); }
   else { add(2.0 / dt, un, -0.5 / dt, unm1, hist_); }
   op_->Mass().Mult(hist_, b_);

   // The guess X = 2 X^n - X^{n-1} (X^n on the first step), Dirichlet data
   // at t^{n+1}.
   if (first) { X_ = Xn_; }
   else { add(2.0, Xn_, -1.0, Xnm1_, X_); }
   Vector xu;
   Block(X_, offsets_, 0, xu);
   bc_.SetTime(t_new);
   {
      u_.SetFromTrueDofs(xu);
      bc_.ProjectDirichlet(u_);
      Vector ud, vals;
      u_.GetTrueDofs(ud);
      ud.GetSubVector(ess_, vals);
      X_.ReadWrite();
      xu.SetSubVector(ess_, vals);
   }

   // The wind: extrapolated (semi-implicit, with its boundary values
   // extrapolated alike), or the guess itself (implicit).
   if (semi && !first) { add(2.0, un, -1.0, unm1, W_); }
   else if (semi) { W_ = un; }
   else { W_ = xu; }
   SetWind(W_);
   if (lor_age_ % std::max(opts_.ns_reuse, 1) == 0)
   {
      StopWatch sl;
      sl.Start();
      lor_pc_->Update(W_, c0);
      stats_.t_lor = sl.RealTime();
   }
   lor_age_++;
   stats_.air = lor_pc_->UsingAir();
   stats_.cell_re = lor_pc_->CellReynolds();

   Residual(X_, R_);
   stats_.r0 = Norm(R_);
   double rn = stats_.r0;
   for (int it = 0; it < std::max(opts_.nl_maxit, 1); it++)
   {
      D_ = 0.0;
      fgmres_->SetRelTol(semi ? opts_.lin_rtol : opts_.picard_lin_rtol);
      fgmres_->Mult(R_, D_);
      stats_.linear_iterations += fgmres_->GetNumIterations();
      stats_.converged = stats_.converged && fgmres_->GetConverged();
      X_ += D_;
      stats_.picard_iterations = it + 1;
      if (semi) { break; }
      SetWind(xu);
      Residual(X_, R_);
      rn = Norm(R_);
      if (opts_.nl_rtol > 0.0 && rn <= opts_.nl_rtol * stats_.r0) { break; }
   }
   if (semi)
   {
      Residual(X_, R_);
      rn = Norm(R_);
   }
   stats_.r_final = rn;
   MFEM_VERIFY(std::isfinite(rn), "OseenStepper: non-finite residual at t = "
               << t_new);

   Xnm1_ = Xn_;
   Xn_ = X_;
   t_ = t_new;
   step_++;
   Vector xp;
   Block(Xn_, offsets_, 1, xp);
   u_.SetFromTrueDofs(xu);
   Vector p_phys(xp);
   p_phys.Neg();
   p_.SetFromTrueDofs(p_phys);
   MFEM_DEVICE_SYNC;
   stats_.t_step = sw.RealTime();
}

} // namespace incns
