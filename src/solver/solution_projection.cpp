#include "solver/solution_projection.hpp"

#include <cmath>

namespace incns
{

using namespace mfem;

namespace
{

Vector DeviceVector(int n)
{
   Vector v(n, Device::GetDeviceMemoryType());
   v.UseDevice(true);
   return v;
}

} // namespace

SolutionProjection::SolutionProjection(int k, MPI_Comm comm)
   : k_(k), comm_(comm)
{
   MFEM_VERIFY(k >= 1, "solution projection: k must be >= 1");
}

void SolutionProjection::Reset()
{
   x_.clear();
   w_.clear();
}

bool SolutionProjection::Guess(const Vector& b, Vector& x0) const
{
   if (x_.empty()) { return false; }
   x0 = 0.0;
   for (std::size_t i = 0; i < x_.size(); ++i)
   {
      x0.Add(InnerProduct(comm_, w_[i], b), x_[i]);
   }
   return true;
}

void SolutionProjection::Append(Vector& x, Vector& w)
{
   const double norm0 = std::sqrt(InnerProduct(comm_, w, w));
   if (norm0 == 0.0) { return; }
   for (int pass = 0; pass < 2; ++pass)
   {
      for (std::size_t i = 0; i < w_.size(); ++i)
      {
         const double h = InnerProduct(comm_, w_[i], w);
         w.Add(-h, w_[i]);
         x.Add(-h, x_[i]);
      }
   }
   const double norm = std::sqrt(InnerProduct(comm_, w, w));
   // Nothing new: the solution is (to roundoff) in the span already.
   if (norm <= 1e-12 * norm0) { return; }
   w *= 1.0 / norm;
   x *= 1.0 / norm;
   x_.push_back(std::move(x));
   w_.push_back(std::move(w));
}

void SolutionProjection::Add(const Operator& A, const Vector& x)
{
   if (Size() >= k_) { Reset(); } // full: restart from the newest
   Vector xn = DeviceVector(x.Size());
   Vector wn = DeviceVector(A.Height());
   xn = x;
   A.Mult(xn, wn);
   Append(xn, wn);
}

void SolutionProjection::Rebuild(const Operator& A)
{
   std::vector<Vector> xs = std::move(x_);
   Reset();
   for (Vector& x : xs)
   {
      Vector w = DeviceVector(A.Height());
      A.Mult(x, w);
      Append(x, w);
   }
}

} // namespace incns
