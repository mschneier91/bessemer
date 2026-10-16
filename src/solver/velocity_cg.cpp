#include "solver/velocity_cg.hpp"

#include <algorithm>

namespace incns
{

using namespace mfem;

VelocityCG::VelocityCG(MPI_Comm comm, int n)
   : comm_(comm), d_(n), z_(n)
{
   d_.UseDevice(true);
   z_.UseDevice(true);
}

void VelocityCG::Solve(Vector& b, Vector& x)
{
   MFEM_VERIFY(A_ && pc_, "velocity_cg: operator and preconditioner not set");
   // mfem::CGSolver::Mult with iterative_mode false, its residual r being b.
   Vector& r = b;
   x = 0.0;
   iterations_ = 0;
   converged_ = false;
   pc_->Mult(r, z_);
   d_ = z_;
   double nom = Dot(d_, r);
   if (!(nom >= 0.0)) { return; } // the preconditioner is not positive
   const double r0 = std::max(nom * rel_tol_ * rel_tol_, abs_tol_ * abs_tol_);
   if (nom <= r0)
   {
      converged_ = true;
      return;
   }
   A_->Mult(d_, z_);
   double den = Dot(z_, d_);
   if (!(den > 0.0)) { return; } // the operator is not positive
   for (int i = 1; ; ++i)
   {
      const double alpha = nom / den;
      x.Add(alpha, d_);
      r.Add(-alpha, z_);
      pc_->Mult(r, z_);
      const double betanom = Dot(r, z_);
      iterations_ = i;
      if (!(betanom >= 0.0)) { return; }
      if (betanom <= r0)
      {
         converged_ = true;
         return;
      }
      if (i >= max_iter_) { return; }
      const double beta = betanom / nom;
      add(z_, beta, d_, d_); // d = z + beta d
      A_->Mult(d_, z_);
      den = Dot(d_, z_);
      if (!(den > 0.0)) { return; }
      nom = betanom;
   }
}

} // namespace incns
