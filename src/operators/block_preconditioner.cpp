#include "operators/block_preconditioner.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

void SubtractGlobalMean(Vector& v, MPI_Comm comm)
{
   double local[2] = { v.Sum(), static_cast<double>(v.Size()) };
   double global[2] = { 0.0, 0.0 };
   MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm);
   MFEM_VERIFY(global[1] > 0.0, "SubtractGlobalMean: empty global vector");
   v -= global[0] / global[1];
}

StokesBlockPreconditioner::StokesBlockPreconditioner(
   const Array<int>& offsets, const Vector& velocity_diag,
   const Array<int>& ess_tdofs, Solver& schur, bool orthogonalize,
   MPI_Comm comm)
   : Solver(offsets.Last()), offsets_(offsets),
     vel_jacobi_(velocity_diag, ess_tdofs), schur_(schur),
     orthogonalize_(orthogonalize), comm_(comm)
{
   MFEM_VERIFY(offsets_.Size() == 3, "block_preconditioner: need 3 offsets");
   MFEM_VERIFY(velocity_diag.Size() == offsets_[1] - offsets_[0],
               "block_preconditioner: velocity diagonal size mismatch");
   MFEM_VERIFY(schur_.Height() == offsets_[2] - offsets_[1],
               "block_preconditioner: Schur block size mismatch");
}

void StokesBlockPreconditioner::Mult(const Vector& x, Vector& y) const
{
   INCNS_PROFILE("block_prec::apply");

   const BlockVector xb(const_cast<Vector&>(x), offsets_);
   BlockVector yb(y, offsets_);

   vel_jacobi_.Mult(xb.GetBlock(0), yb.GetBlock(0));
   schur_.Mult(xb.GetBlock(1), yb.GetBlock(1));

   if (orthogonalize_)
   {
      SubtractGlobalMean(yb.GetBlock(1), comm_);
   }

   // BlockVector views may hold stale data pointers after block writes; make
   // sure y sees the updates (no-op on CPU).
   yb.SyncFromBlocks();
   y.SyncMemory(yb);
}

} // namespace incns
