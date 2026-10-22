#include "fsi/interface_accelerator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace incns
{

using namespace mfem;

AitkenRelaxation::AitkenRelaxation(double omega0, double omega_max, bool aitken,
                                   MPI_Comm comm)
   : omega0_(omega0), omega_max_(omega_max), aitken_(aitken), comm_(comm)
{
}

void AitkenRelaxation::NewStep() { has_prev_ = false; }

double AitkenRelaxation::Next(const Vector& d, const Vector& dtilde,
                              Vector& next)
{
   Vector r(dtilde);
   r -= d;
   double omega = omega0_;
   if (aitken_ && has_prev_)
   {
      Vector dr(r);
      dr -= r_prev_;
      const double dd = InnerProduct(comm_, dr, dr);
      const double rd = InnerProduct(comm_, r_prev_, dr);
      omega = dd > 0.0 ? -omega_ * rd / dd : omega_;
      if (!std::isfinite(omega)) { omega = omega0_; }
      omega = std::max(-omega_max_, std::min(omega_max_, omega));
   }
   omega_ = omega;
   r_prev_ = r;
   has_prev_ = true;
   next = d;
   next.Add(omega, r);
   return omega;
}

IqnIls::IqnIls(double omega0, int reuse, MPI_Comm comm)
   : omega0_(omega0), reuse_(reuse), comm_(comm)
{
}

void IqnIls::NewStep()
{
   if (reuse_ > 0 && !current_.v.empty())
   {
      past_.push_front(std::move(current_));
      while (static_cast<int>(past_.size()) > reuse_) { past_.pop_back(); }
   }
   current_ = Columns_();
   has_prev_ = false;
}

double IqnIls::Next(const Vector& d, const Vector& dtilde, Vector& next)
{
   Vector r(dtilde);
   r -= d;
   if (has_prev_)
   {
      Vector dv(r), dw(dtilde);
      dv -= r_prev_;
      dw -= dt_prev_;
      current_.v.push_front(std::move(dv));
      current_.w.push_front(std::move(dw));
   }
   r_prev_ = r;
   dt_prev_ = dtilde;
   has_prev_ = true;

   // The columns, newest first: this step's, then the kept steps'.
   std::vector<const Vector*> V, W;
   for (std::size_t i = 0; i < current_.v.size(); ++i)
   {
      V.push_back(&current_.v[i]);
      W.push_back(&current_.w[i]);
   }
   for (const Columns_& c : past_)
   {
      for (std::size_t i = 0; i < c.v.size(); ++i)
      {
         V.push_back(&c.v[i]);
         W.push_back(&c.w[i]);
      }
   }
   if (V.empty())
   {
      columns_ = 0;
      next = d;
      next.Add(omega0_, r);
      return omega0_;
   }

   // Modified Gram-Schmidt on V; a column whose part orthogonal to the
   // newer ones is below eps of its norm is dropped (with its W column).
   const double eps = 1e-4;
   std::vector<Vector> Q;
   std::vector<int> kept;
   std::vector<std::vector<double>> R; // R[j] = column j of R (kept order)
   for (std::size_t i = 0; i < V.size(); ++i)
   {
      Vector q(*V[i]);
      const double n0 = std::sqrt(InnerProduct(comm_, q, q));
      if (!(n0 > 0.0)) { continue; }
      std::vector<double> col(Q.size() + 1, 0.0);
      for (std::size_t j = 0; j < Q.size(); ++j)
      {
         col[j] = InnerProduct(comm_, Q[j], q);
         q.Add(-col[j], Q[j]);
      }
      const double n1 = std::sqrt(InnerProduct(comm_, q, q));
      if (n1 < eps * n0) { continue; }
      q /= n1;
      col.back() = n1;
      Q.push_back(std::move(q));
      kept.push_back(static_cast<int>(i));
      R.push_back(std::move(col));
   }
   const int m = static_cast<int>(Q.size());
   columns_ = m;
   // R c = -Q^T r, back substitution.
   std::vector<double> c(m, 0.0);
   for (int j = 0; j < m; ++j) { c[j] = -InnerProduct(comm_, Q[j], r); }
   for (int j = m - 1; j >= 0; --j)
   {
      for (int l = j + 1; l < m; ++l) { c[j] -= R[l][j] * c[l]; }
      c[j] /= R[j][j];
   }
   next = dtilde;
   for (int j = 0; j < m; ++j) { next.Add(c[j], *W[kept[j]]); }
   return std::numeric_limits<double>::quiet_NaN();
}

std::unique_ptr<InterfaceAccelerator> MakeInterfaceAccelerator(
   const FsiParameters& f, MPI_Comm comm)
{
   switch (f.acceleration)
   {
      case FsiAcceleration::IqnIls:
         return std::make_unique<IqnIls>(f.relaxation, f.reuse, comm);
      case FsiAcceleration::Aitken:
      case FsiAcceleration::Constant:
         break;
   }
   return std::make_unique<AitkenRelaxation>(f.relaxation, f.omega_max,
          f.acceleration == FsiAcceleration::Aitken, comm);
}

} // namespace incns
