#include "operators/block_preconditioner.hpp"

#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

StokesBlockPreconditioner::StokesBlockPreconditioner(
   const Array<int>& offsets, const Vector& velocity_diag,
   const Array<int>& ess_tdofs, Solver& pressure)
   : Solver(offsets.Last()), offsets_(offsets),
     vel_jacobi_(velocity_diag, ess_tdofs), pressure_(pressure)
{
   MFEM_VERIFY(offsets_.Size() == 3, "block_preconditioner: need 3 offsets");
   MFEM_VERIFY(velocity_diag.Size() == offsets_[1] - offsets_[0],
               "block_preconditioner: velocity diagonal size mismatch");
}

void StokesBlockPreconditioner::Mult(const Vector& x, Vector& y) const
{
   INCNS_PROFILE("block_prec::apply");

   const BlockVector xb(const_cast<Vector&>(x), offsets_);
   BlockVector yb(y, offsets_);

   vel_jacobi_.Mult(xb.GetBlock(0), yb.GetBlock(0));
   pressure_.Mult(xb.GetBlock(1), yb.GetBlock(1));

   // BlockVector views may hold stale data pointers after block writes; make
   // sure y sees the updates (no-op on CPU).
   yb.SyncFromBlocks();
   y.SyncMemory(yb);
}

} // namespace incns
