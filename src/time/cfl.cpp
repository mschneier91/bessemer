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

// Mesh-node 1D sizes the curved kernel takes (mesh orders up to 9): its tiles
// and its per-thread partial sums are sized by it.
constexpr int kCurvedMaxND1 = 10;

// ElementRates on curved elements in 3D, nothing per point read from memory:
// u at node q through the gather map (byNODES, component stride ndofs); J at
// q contracted from the element's mesh nodes (through their gather map, value
// c of node j at XN[j * nds + c * ncs]) with the mesh basis B and its
// derivative Gd at the GLL nodes, in the order of MFEM's Derivatives3D (x,
// then y, then z, each sum from index 0, the same products), so J -- and the
// rate -- are the stored path's bit for bit where both are compiled alike (on
// a GPU; the CPU build differs from MFEM's in FMA). Threads: a (qx, qy)
// column each, qz looped; the element's maximum from the columns' at the end.
void ElementRatesCurved3D(int ne, int d1, int nd1, const Array<int>& gmap,
                          int ndofs, const Vector& u, const Array<int>& ngmap,
                          int nds, int ncs, const Vector& xn,
                          const Array<real_t>& b, const Array<real_t>& g,
                          const Vector& inv_dxi, Vector& rate)
{
   const int nq = d1 * d1 * d1, nn = nd1 * nd1 * nd1;
   const auto Gm = Reshape(gmap.Read(), nq, ne);
   const auto Nm = Reshape(ngmap.Read(), nn, ne);
   const auto U = u.Read();
   const auto XN = xn.Read();
   const auto B = Reshape(b.Read(), d1, nd1);
   const auto Gd = Reshape(g.Read(), d1, nd1);
   const auto X = inv_dxi.Read();
   auto R = rate.Write();
   const real_t inf = std::numeric_limits<real_t>::infinity();
   mfem::forall_2D(ne, d1, d1, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MN = kCurvedMaxND1;
      constexpr int MQ = DofQuadLimits::MAX_D1D;
      MFEM_SHARED real_t sX[3][MN][MN][MN]; // [c][dz][dy][dx]
      MFEM_SHARED real_t sB[MQ][MN], sG[MQ][MN];
      MFEM_SHARED real_t smax[MQ][MQ];
      MFEM_FOREACH_THREAD(dy, y, nd1)
      {
         MFEM_FOREACH_THREAD(dx, x, nd1)
         {
            for (int dz = 0; dz < nd1; ++dz)
            {
               const int j = Nm(dx + nd1 * (dy + nd1 * dz), e);
               for (int c = 0; c < 3; ++c) { sX[c][dz][dy][dx] = XN[j * nds + c * ncs]; }
            }
         }
      }
      MFEM_FOREACH_THREAD(d, y, nd1)
      {
         MFEM_FOREACH_THREAD(q, x, d1)
         {
            sB[q][d] = B(q, d);
            sG[q][d] = Gd(q, d);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy, y, d1)
      {
         MFEM_FOREACH_THREAD(qx, x, d1)
         {
            // x and y contracted for every dz: r[c][d][dz] (MFEM's DQQ0-2).
            real_t r[3][3][MN];
            for (int c = 0; c < 3; ++c)
            {
               for (int dz = 0; dz < nd1; ++dz)
               {
                  real_t a0 = 0.0, a1 = 0.0, a2 = 0.0;
                  for (int dy = 0; dy < nd1; ++dy)
                  {
                     real_t bu = 0.0, gu = 0.0; // MFEM's DDQ0, DDQ1
                     for (int dx = 0; dx < nd1; ++dx)
                     {
                        const real_t in = sX[c][dz][dy][dx];
                        bu += in * sB[qx][dx];
                        gu += in * sG[qx][dx];
                     }
                     a0 += gu * sB[qy][dy];
                     a1 += bu * sG[qy][dy];
                     a2 += bu * sB[qy][dy];
                  }
                  r[c][0][dz] = a0;
                  r[c][1][dz] = a1;
                  r[c][2][dz] = a2;
               }
            }
            real_t rmax = 0.0;
            for (int qz = 0; qz < d1; ++qz)
            {
               real_t m[3][3]; // dx_c / dxi_d
               for (int c = 0; c < 3; ++c)
               {
                  real_t j0 = 0.0, j1 = 0.0, j2 = 0.0;
                  for (int dz = 0; dz < nd1; ++dz)
                  {
                     j0 += r[c][0][dz] * sB[qz][dz];
                     j1 += r[c][1][dz] * sB[qz][dz];
                     j2 += r[c][2][dz] * sG[qz][dz];
                  }
                  m[c][0] = j0;
                  m[c][1] = j1;
                  m[c][2] = j2;
               }
               // ElementRates' arithmetic, as written there.
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
               const int gq = Gm(qx + d1 * (qy + d1 * qz), e);
               const real_t u0 = U[gq], u1 = U[gq + ndofs], u2 = U[gq + 2 * ndofs];
               real_t s = fabs((a00 * u0 + a01 * u1 + a02 * u2) / det) * X[qx] +
                          fabs((a10 * u0 + a11 * u1 + a12 * u2) / det) * X[qy] +
                          fabs((a20 * u0 + a21 * u1 + a22 * u2) / det) * X[qz];
               if (!(s == s)) { s = inf; } // as ElementRates: NaN -> infinite rate
               rmax = (s > rmax) ? s : rmax;
            }
            smax[qy][qx] = rmax;
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0)
      {
         real_t rmax = 0.0;
         for (int qy = 0; qy < d1; ++qy)
         {
            for (int qx = 0; qx < d1; ++qx)
            {
               rmax = (smax[qy][qx] > rmax) ? smax[qy][qx] : rmax;
            }
         }
         R[e] = rmax;
      }
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
   // On the fly where the elements are straight-sided (linear mesh nodes),
   // and on curved elements in 3D (the Jacobians from the mesh nodes).
   const GridFunction& nodes = *mesh.GetNodes();
   const FiniteElementSpace& nfes = *nodes.FESpace();
   const auto* er = dynamic_cast<const ElementRestriction*>(restr_);
   const bool gather_able = nodal_gll_ && er &&
                            vfes.GetOrdering() == Ordering::byNODES && ne_ > 0;
   const int nk = nfes.GetTypicalFE()->GetOrder();
   const bool tensor_nodes =
      dynamic_cast<const TensorBasisElement*>(nfes.GetTypicalFE()) != nullptr;
   fly_able_ = gather_able && nk == 1 && tensor_nodes;
   const auto* ner = dynamic_cast<const ElementRestriction*>(
                        nfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   curved_able_ = gather_able && nk >= 2 && tensor_nodes && dim_ == 3 && ner &&
                  nfes.GetVDim() == 3 && nk + 1 <= cfl_kernels::kCurvedMaxND1 &&
                  order_ + 1 <= DeviceDofQuadLimits::Get().MAX_D1D;
   if (fly_able_ || curved_able_)
   {
      gather_ = &er->GatherMap();
      const int* gm = gather_->HostRead();
      for (int i = 0; i < gather_->Size(); ++i)
      {
         MFEM_VERIFY(gm[i] >= 0, "cfl: an H1 gather map has no orientation signs");
      }
      ndofs_ = vfes.GetNDofs();
   }
   if (curved_able_)
   {
      ngather_ = &ner->GatherMap();
      const int* gm = ngather_->HostRead();
      for (int i = 0; i < ngather_->Size(); ++i)
      {
         MFEM_VERIFY(gm[i] >= 0, "cfl: an H1 gather map has no orientation signs");
      }
      // Value c of mesh node j at nodes[j * node_stride_ + c * comp_stride_].
      const bool byvdim = nfes.GetOrdering() == Ordering::byVDIM;
      node_stride_ = byvdim ? 3 : 1;
      comp_stride_ = byvdim ? 1 : nfes.GetNDofs();
      nodes_ = &nodes;
      // The maps MFEM's QuadratureInterpolator uses for the mesh nodes at
      // this rule (GeometricFactors' Jacobians).
      nmaps_ = &nfes.GetTypicalFE()->GetDofToQuad(*ir_, DofToQuad::TENSOR);
      curved_fly_ = true;
   }
   if (fly_able_)
   {
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

void ConvectiveCfl::CurvedRates(const ParGridFunction& u) const
{
   cfl_kernels::ElementRatesCurved3D(ne_, order_ + 1, nmaps_->ndof, *gather_,
                                     ndofs_, u, *ngather_, node_stride_,
                                     comp_stride_, *nodes_, nmaps_->B,
                                     nmaps_->G, inv_dxi_, rate_);
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
   else if (ne_ > 0 && curved_fly_)
   {
      for (const ParGridFunction* up : us)
      {
         MFEM_VERIFY(up->ParFESpace() == &vfes_, "cfl: u must live on the CFL space");
         CurvedRates(*up);
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
   else if (ne_ > 0 && curved_fly_)
   {
      for (std::size_t l = 0; l < us.size(); ++l)
      {
         MFEM_VERIFY(us[l]->ParFESpace() == &vfes_,
                     "cfl: u must live on the CFL space");
         CurvedRates(*us[l]);
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
