#include "solver/schur_complement.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

SchurComplementOperator::SchurComplementOperator(const Operator& B,
                                                 IterativeSolver& a_inv,
                                                 const ConstantPressureProjector* proj)
   : Operator(B.Height()), B_(B), a_inv_(&a_inv), proj_(proj),
     w_(B.Width()), v_(B.Width())
{
   w_.UseDevice(true);
   v_.UseDevice(true);
   a_inv_->iterative_mode = false;
   a_inv_->SetRelTol(rel_);
}

void SchurComplementOperator::SetVelocitySolver(IterativeSolver& a_inv)
{
   a_inv_ = &a_inv;
   a_inv_->iterative_mode = false;
   a_inv_->SetRelTol(rel_);
}

void SchurComplementOperator::ResetCounters() const
{
   applications_ = inner_iterations_ = cap_hits_ = last_iterations_ = 0;
   inner_time_ = 0.0;
}

void SchurComplementOperator::Mult(const Vector& d, Vector& q) const
{
   INCNS_PROFILE("schur_operator::apply");
   B_.MultTranspose(d, w_);
   v_ = 0.0;
   const double t0 = MPI_Wtime();
   {
      INCNS_PROFILE("schur_operator::velocity_solve");
      a_inv_->Mult(w_, v_);
   }
   inner_time_ += MPI_Wtime() - t0;
   last_iterations_ = a_inv_->GetNumIterations();
   inner_iterations_ += last_iterations_;
   if (!a_inv_->GetConverged()) { ++cap_hits_; }
   ++applications_;
   B_.Mult(v_, q);
   // B v is orthogonal to the constants in exact arithmetic (B^T 1 = 0 on
   // these domains); the projection removes the roundoff.
   if (proj_) { proj_->Project(q); }
}

} // namespace incns
