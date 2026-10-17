#include "time/cfl.hpp"

#include "mfem/general/forall.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// This TU defines a device kernel (mfem::forall). A host compiler would build
// it as a host loop over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "cfl.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernel in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace cfl_kernels
{

// rate[e] = max over the GLL nodes q = (i, j[, l]) (lexicographic, x
// fastest) of sum_d |(J_q^{-1} u_q)_d| * inv_dxi[index_d(q)], from u at the
// nodes U(q, c, e) (byNODES) and Jacobians J(q, i, j, e) = dx_i/dxi_j.
void ElementRates(int ne, int d1, int dim, const Vector& uq, const Vector& jac,
                  const Vector& inv_dxi, Vector& rate)
{
   const int nq = (dim == 3) ? d1 * d1 * d1 : d1 * d1;
   const auto U = Reshape(uq.Read(), nq, dim, ne);
   const auto J = Reshape(jac.Read(), nq, dim, dim, ne);
   const auto X = inv_dxi.Read();
   auto R = rate.Write();
   const real_t inf = std::numeric_limits<real_t>::infinity();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t rmax = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         const int ix = q % d1, iy = (q / d1) % d1, iz = q / (d1 * d1);
         real_t s = 0.0;
         if (dim == 2)
         {
            const real_t a = J(q, 0, 0, e), b = J(q, 0, 1, e);
            const real_t c = J(q, 1, 0, e), d = J(q, 1, 1, e);
            const real_t det = a * d - b * c;
            const real_t u0 = U(q, 0, e), u1 = U(q, 1, e);
            // J^{-1} = [d -b; -c a] / det
            s = fabs((d * u0 - b * u1) / det) * X[ix] +
                fabs((-c * u0 + a * u1) / det) * X[iy];
         }
         else
         {
            real_t m[3][3];
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j) { m[i][j] = J(q, i, j, e); }
            const real_t a00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            const real_t a01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
            const real_t a02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            const real_t a10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
            const real_t a11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
            const real_t a12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
            const real_t a20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
            const real_t a21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
            const real_t a22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const real_t det = m[0][0] * a00 + m[0][1] * a10 + m[0][2] * a20;
            const real_t u0 = U(q, 0, e), u1 = U(q, 1, e), u2 = U(q, 2, e);
            s = fabs((a00 * u0 + a01 * u1 + a02 * u2) / det) * X[ix] +
                fabs((a10 * u0 + a11 * u1 + a12 * u2) / det) * X[iy] +
                fabs((a20 * u0 + a21 * u1 + a22 * u2) / det) * X[iz];
         }
         // A NaN velocity reports an infinite rate: a max skips a NaN (every
         // comparison with it is false), and so would MPI's.
         if (!(s == s)) { s = inf; }
         rmax = (s > rmax) ? s : rmax;
      }
      R[e] = rmax;
   });
}

// ElementRates on straight-sided elements, nothing per point read from
// memory: u at node q through the gather map (byNODES, component stride
// ndofs), J at q from the element's vertices XV(v, i, e), v lexicographic (x
// fastest), the multilinear map on MFEM's [0, 1] reference element.
void ElementRatesLinear(int ne, int d1, int dim, const Array<int>& gmap,
                        int ndofs, const Vector& u, const Vector& xv,
                        const Vector& z, const Vector& inv_dxi, Vector& rate)
{
   const int nq = (dim == 3) ? d1 * d1 * d1 : d1 * d1;
   const int nv = (dim == 3) ? 8 : 4;
   const auto Gm = Reshape(gmap.Read(), nq, ne);
   const auto U = u.Read();
   const auto XV = Reshape(xv.Read(), nv, dim, ne);
   const auto Z = z.Read();
   const auto X = inv_dxi.Read();
   auto R = rate.Write();
   const real_t inf = std::numeric_limits<real_t>::infinity();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t rmax = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         const int ix = q % d1, iy = (q / d1) % d1, iz = q / (d1 * d1);
         const int g = Gm(q, e);
         real_t s = 0.0;
         if (dim == 2)
         {
            const real_t x = Z[ix], y = Z[iy];
            real_t m[2][2];
            for (int i = 0; i < 2; ++i)
            {
               m[i][0] = (1.0 - y) * (XV(1, i, e) - XV(0, i, e)) +
                         y * (XV(3, i, e) - XV(2, i, e));
               m[i][1] = (1.0 - x) * (XV(2, i, e) - XV(0, i, e)) +
                         x * (XV(3, i, e) - XV(1, i, e));
            }
            const real_t a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
            const real_t det = a * d - b * c;
            const real_t u0 = U[g], u1 = U[g + ndofs];
            s = fabs((d * u0 - b * u1) / det) * X[ix] +
                fabs((-c * u0 + a * u1) / det) * X[iy];
         }
         else
         {
            const real_t x = Z[ix], y = Z[iy], w = Z[iz];
            real_t m[3][3];
            for (int i = 0; i < 3; ++i)
            {
               m[i][0] = (1.0 - y) * (1.0 - w) * (XV(1, i, e) - XV(0, i, e)) +
                         y * (1.0 - w) * (XV(3, i, e) - XV(2, i, e)) +
                         (1.0 - y) * w * (XV(5, i, e) - XV(4, i, e)) +
                         y * w * (XV(7, i, e) - XV(6, i, e));
               m[i][1] = (1.0 - x) * (1.0 - w) * (XV(2, i, e) - XV(0, i, e)) +
                         x * (1.0 - w) * (XV(3, i, e) - XV(1, i, e)) +
                         (1.0 - x) * w * (XV(6, i, e) - XV(4, i, e)) +
                         x * w * (XV(7, i, e) - XV(5, i, e));
               m[i][2] = (1.0 - x) * (1.0 - y) * (XV(4, i, e) - XV(0, i, e)) +
                         x * (1.0 - y) * (XV(5, i, e) - XV(1, i, e)) +
                         (1.0 - x) * y * (XV(6, i, e) - XV(2, i, e)) +
                         x * y * (XV(7, i, e) - XV(3, i, e));
            }
            const real_t a00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            const real_t a01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
            const real_t a02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            const real_t a10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
            const real_t a11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
            const real_t a12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
            const real_t a20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
            const real_t a21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
            const real_t a22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            const real_t det = m[0][0] * a00 + m[0][1] * a10 + m[0][2] * a20;
            const real_t u0 = U[g], u1 = U[g + ndofs], u2 = U[g + 2 * ndofs];
            s = fabs((a00 * u0 + a01 * u1 + a02 * u2) / det) * X[ix] +
                fabs((a10 * u0 + a11 * u1 + a12 * u2) / det) * X[iy] +
                fabs((a20 * u0 + a21 * u1 + a22 * u2) / det) * X[iz];
         }
         if (!(s == s)) { s = inf; } // as ElementRates: NaN -> infinite rate
         rmax = (s > rmax) ? s : rmax;
      }
      R[e] = rmax;
   });
}

} // namespace cfl_kernels

void RequireFiniteRate(double rate, double t)
{
   MFEM_VERIFY(std::isfinite(rate), "the velocity is not finite (NaN or Inf) "
               "at t = " << t << ": the run diverged in the step that reached "
               "it (the usual remedy is a smaller time step or CFL target)");
}

Vector ConvectiveCfl::InverseNodeSpacing(int k)
{
   // GLL points on [0,1] (MFEM's Poly_1D); Nek's getdr on the same points.
   const real_t* z = poly1d.GetPoints(k, BasisType::GaussLobatto);
   Vector inv(k + 1);
   for (int i = 0; i <= k; ++i)
   {
      const real_t dz = (i == 0) ? z[1] - z[0]
                        : (i == k) ? z[k] - z[k - 1]
                        : 0.5 * (z[i + 1] - z[i - 1]);
      inv(i) = 1.0 / dz;
   }
   return inv;
}

ConvectiveCfl::ConvectiveCfl(const ParFiniteElementSpace& vfes,
                             const RuleBook& rules)
   : vfes_(vfes)
{
   ParMesh& mesh = *vfes.GetParMesh();
   dim_ = mesh.Dimension();
   ne_ = vfes.GetNE();
   order_ = vfes.GetTypicalFE()->GetOrder();
   MFEM_VERIFY(order_ >= 1, "cfl: velocity order must be >= 1");
   MFEM_VERIFY(vfes.GetVDim() == dim_, "cfl: velocity vdim must equal dim");
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   // The k+1 GLL points per direction: the H1 nodes, where Nek evaluates.
   ir_ = &rules.Get(geom, 2 * order_ - 1, Rule1D::GaussLobatto);
   MFEM_VERIFY(ir_->GetNPoints() == ((dim_ == 3) ? (order_ + 1) * (order_ + 1) *
                                     (order_ + 1) : (order_ + 1) * (order_ + 1)),
               "cfl: expected k+1 GLL points per direction");
   restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   qi_ = vfes.GetQuadratureInterpolator(*ir_);
   qi_->SetOutputLayout(QVectorLayout::byNODES);
   mesh.EnsureNodes(); // Rate() computes the Jacobians from them
   inv_dxi_ = InverseNodeSpacing(order_);
   inv_dxi_.UseDevice(true);
   const auto* h1 = dynamic_cast<const H1_FECollection*>(vfes.FEColl());
   nodal_gll_ = h1 && h1->GetBasisType() == BasisType::GaussLobatto;
   rate_.SetSize(ne_);
   rate_.UseDevice(true);
   // On the fly where the elements are straight-sided (linear mesh nodes).
   const GridFunction& nodes = *mesh.GetNodes();
   const FiniteElementSpace& nfes = *nodes.FESpace();
   const auto* er = dynamic_cast<const ElementRestriction*>(restr_);
   fly_able_ = nodal_gll_ && er && vfes.GetOrdering() == Ordering::byNODES &&
               ne_ > 0 && nfes.GetTypicalFE()->GetOrder() == 1 &&
               dynamic_cast<const TensorBasisElement*>(nfes.GetTypicalFE());
   if (fly_able_)
   {
      gather_ = &er->GatherMap();
      const int* gm = gather_->HostRead();
      for (int i = 0; i < gather_->Size(); ++i)
      {
         MFEM_VERIFY(gm[i] >= 0, "cfl: an H1 gather map has no orientation signs");
      }
      ndofs_ = vfes.GetNDofs();
      const Operator* nr =
         nfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      xv_.UseDevice(true);
      xv_.SetSize(nr->Height());
      nr->Mult(nodes, xv_);
      const real_t* z = poly1d.GetPoints(order_, BasisType::GaussLobatto);
      zgll_.SetSize(order_ + 1);
      for (int i = 0; i <= order_; ++i) { zgll_(i) = z[i]; }
      zgll_.UseDevice(true);
      fly_ = true;
   }
}

double ConvectiveCfl::Rate(const ParGridFunction& u) const
{
   return MaxRate({&u});
}

double ConvectiveCfl::MaxRate(const std::vector<const ParGridFunction*>& us)
const
{
   MFEM_VERIFY(!us.empty(), "cfl: no velocity");
   double local = 0.0;
   if (ne_ > 0 && fly_)
   {
      for (const ParGridFunction* up : us)
      {
         MFEM_VERIFY(up->ParFESpace() == &vfes_, "cfl: u must live on the CFL space");
         cfl_kernels::ElementRatesLinear(ne_, order_ + 1, dim_, *gather_, ndofs_,
                                         *up, xv_, zgll_, inv_dxi_, rate_);
         local = std::max(local, rate_.Max()); // device-aware reduction
      }
   }
   else if (ne_ > 0)
   {
      // The element copies of the velocity live for this call only: kept as
      // members they held 0.8 GB for 64^3 Q3 elements (2026-10-13).
      Vector ue(restr_->Height());
      ue.UseDevice(true);
      // So do the Jacobians at the GLL nodes: cached in the mesh they held
      // 1.2 GB at 64^3 Q3 for the whole run, and the rate is evaluated
      // between solves, away from the memory peak (2026-10-14). MFEM's own
      // computation, not its cache: the same values. Once per call for all
      // the velocities (OIFS: one per history level).
      const GeometricFactors geom(*vfes_.GetParMesh()->GetNodes(), *ir_,
                                  GeometricFactors::JACOBIANS,
                                  Device::GetDeviceMemoryType());
      for (const ParGridFunction* up : us)
      {
         const ParGridFunction& u = *up;
         MFEM_VERIFY(u.ParFESpace() == &vfes_, "cfl: u must live on the CFL space");
         restr_->Mult(u, ue);
         if (nodal_gll_)
         {
            // (dofs, component, element), lexicographic: the layout the kernel
            // reads, with the dofs at the GLL points.
            cfl_kernels::ElementRates(ne_, order_ + 1, dim_, ue, geom.J,
                                      inv_dxi_, rate_);
         }
         else
         {
            Vector uq(ir_->GetNPoints() * dim_ * ne_);
            uq.UseDevice(true);
            qi_->SetOutputLayout(QVectorLayout::byNODES); // shared object: set it
            qi_->Values(ue, uq);
            cfl_kernels::ElementRates(ne_, order_ + 1, dim_, uq, geom.J,
                                      inv_dxi_, rate_);
         }
         local = std::max(local, rate_.Max()); // device-aware reduction
      }
   }
   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, vfes_.GetComm());
   return global;
}

void ConvectiveCfl::Rates(const std::vector<const ParGridFunction*>& us,
                          std::vector<double>& rates) const
{
   // MaxRate's evaluation, keeping each velocity's maximum apart.
   rates.assign(us.size(), 0.0);
   if (ne_ > 0 && fly_)
   {
      for (std::size_t l = 0; l < us.size(); ++l)
      {
         MFEM_VERIFY(us[l]->ParFESpace() == &vfes_,
                     "cfl: u must live on the CFL space");
         cfl_kernels::ElementRatesLinear(ne_, order_ + 1, dim_, *gather_, ndofs_,
                                         *us[l], xv_, zgll_, inv_dxi_, rate_);
         rates[l] = std::max(0.0, rate_.Max());
      }
   }
   else if (ne_ > 0)
   {
      Vector ue(restr_->Height());
      ue.UseDevice(true);
      const GeometricFactors geom(*vfes_.GetParMesh()->GetNodes(), *ir_,
                                  GeometricFactors::JACOBIANS,
                                  Device::GetDeviceMemoryType());
      for (std::size_t l = 0; l < us.size(); ++l)
      {
         const ParGridFunction& u = *us[l];
         MFEM_VERIFY(u.ParFESpace() == &vfes_, "cfl: u must live on the CFL space");
         restr_->Mult(u, ue);
         if (nodal_gll_)
         {
            cfl_kernels::ElementRates(ne_, order_ + 1, dim_, ue, geom.J,
                                      inv_dxi_, rate_);
         }
         else
         {
            Vector uq(ir_->GetNPoints() * dim_ * ne_);
            uq.UseDevice(true);
            qi_->SetOutputLayout(QVectorLayout::byNODES);
            qi_->Values(ue, uq);
            cfl_kernels::ElementRates(ne_, order_ + 1, dim_, uq, geom.J,
                                      inv_dxi_, rate_);
         }
         rates[l] = std::max(0.0, rate_.Max());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, rates.data(), static_cast<int>(rates.size()),
                 MPI_DOUBLE, MPI_MAX, vfes_.GetComm());
}

double ConvectiveCfl::MaxRateAt(const std::vector<const ParGridFunction*>& us,
                                const std::vector<double>& times) const
{
   MFEM_VERIFY(!us.empty() && us.size() == times.size(),
               "cfl: one time per velocity");
   if (level_seq_ != vfes_.GetSequence())
   {
      level_rates_.clear();
      level_seq_ = vfes_.GetSequence();
   }
   double global = 0.0;
   std::vector<const ParGridFunction*> todo;
   std::vector<double> todo_t;
   for (std::size_t i = 0; i < us.size(); ++i)
   {
      const auto it = std::find_if(level_rates_.begin(), level_rates_.end(),
                                   [&](const std::pair<double, double>& p)
      { return p.first == times[i]; });
      if (it != level_rates_.end()) { global = std::max(global, it->second); }
      else
      {
         todo.push_back(us[i]);
         todo_t.push_back(times[i]);
      }
   }
   if (!todo.empty()) // the same on every rank: the keys are global
   {
      std::vector<double> r;
      Rates(todo, r);
      for (std::size_t k = 0; k < todo.size(); ++k)
      {
         global = std::max(global, r[k]);
         level_rates_.emplace_back(todo_t[k], r[k]);
      }
      // Keep the newest few levels (OIFS uses up to three).
      while (level_rates_.size() > 8)
      {
         level_rates_.erase(std::min_element(
                               level_rates_.begin(), level_rates_.end()));
      }
   }
   return global;
}

double ConvectiveCfl::RateAndLocation(const ParGridFunction& u,
                                      Vector& where) const
{
   const double rate = Rate(u); // fills rate_ per element
   struct { double val; int rank; } loc{-1.0, 0}, glob{0.0, 0};
   MPI_Comm_rank(vfes_.GetComm(), &loc.rank);
   int arg = -1;
   if (ne_ > 0)
   {
      const double* r = rate_.HostRead();
      arg = static_cast<int>(std::max_element(r, r + ne_) - r);
      loc.val = r[arg];
   }
   MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC, vfes_.GetComm());
   where.SetSize(dim_);
   where = 0.0;
   if (loc.rank == glob.rank && arg >= 0)
   {
      vfes_.GetParMesh()->GetElementCenter(arg, where);
   }
   MPI_Bcast(where.HostReadWrite(), dim_, MPI_DOUBLE, glob.rank, vfes_.GetComm());
   return rate;
}

} // namespace incns
