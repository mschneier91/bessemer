#include "util/profiled_operator.hpp"

#include "util/profiler.hpp"

namespace incns
{

ProfiledOperator::ProfiledOperator(const mfem::Operator& op, const char* region,
                                   const char* transpose_region)
   : mfem::Solver(op.Height(), op.Width()), op_(op), region_(region),
     transpose_region_(transpose_region ? transpose_region : region)
{
}

void ProfiledOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   INCNS_PROFILE(region_);
   op_.Mult(x, y);
}

void ProfiledOperator::MultTranspose(const mfem::Vector& x,
                                     mfem::Vector& y) const
{
   INCNS_PROFILE(transpose_region_);
   op_.MultTranspose(x, y);
}

} // namespace incns
