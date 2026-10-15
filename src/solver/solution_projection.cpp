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
   x_.reserve(k_);
   w_.reserve(k_);
}

void SolutionProjection::Take(int n, Vector& v)
{
   for (std::size_t i = spare_.size(); i-- > 0;)
   {
      if (spare_[i]->Size() == n)
      {
         v.Swap(*spare_[i]);
         spare_.erase(spare_.begin() + i);
         return;
      }
   }
   Vector fresh = DeviceVector(n);
   v.Swap(fresh);
}

void SolutionProjection::Give(Vector& v)
{
   if (v.Size() == 0) { return; }
   spare_.push_back(std::make_unique<Vector>());
   spare_.back()->Swap(v);
}

void SolutionProjection::Reset()
{
   for (Vector& v : x_) { Give(v); }
   for (Vector& v : w_) { Give(v); }
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
   x_.emplace_back();
   x_.back().Swap(x);
   w_.emplace_back();
   w_.back().Swap(w);
}

void SolutionProjection::Add(const Operator& A, const Vector& x)
{
   if (Size() >= k_) { Reset(); } // full: restart from the newest
   Vector xn, wn;
   Take(x.Size(), xn);
   Take(A.Height(), wn);
   xn = x;
   A.Mult(xn, wn);
   Append(xn, wn);
   Give(xn); // not appended: kept for the next one
   Give(wn);
}

void SolutionProjection::Rebuild(const Operator& A)
{
   std::vector<Vector> xs = std::move(x_);
   x_.clear();
   x_.reserve(k_);
   Reset();
   for (Vector& x : xs)
   {
      Vector w;
      Take(A.Height(), w);
      A.Mult(x, w);
      Append(x, w);
      Give(x);
      Give(w);
   }
}

} // namespace incns
