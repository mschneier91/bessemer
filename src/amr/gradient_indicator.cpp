#include "amr/gradient_indicator.hpp"

#include "operators/element_scratch.hpp"
#include "util/profiler.hpp"
#include "mfem/general/forall.hpp"

#include <cmath>

// This TU defines device kernels (mfem::forall, forall_2D + MFEM_SHARED
// tiles). A host compiler would build them as host loops over device
// pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "gradient_indicator.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernel in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace amr_indicator
{

// g[d + dim e] = sqrt( sum_q w_q detJ_q sum_c du_c/dxi_d(q)^2 / sum_q w_q detJ_q )
// from reference gradients D(q, c, d, e) (QVectorLayout::byNODES). One thread
// per element: an adaptation-event kernel, not a hot path.
void Reduce(int ne, int nq, int dim, const Array<real_t>& w,
            const Vector& detj, const Vector& qd, Vector& g)
{
   const auto W = w.Read();
   const auto DJ = Reshape(detj.Read(), nq, ne);
   const auto D = Reshape(qd.Read(), nq, dim, dim, ne);
   auto G = Reshape(g.Write(), dim, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t vol = 0.0;
      real_t s[3] = {0.0, 0.0, 0.0};
      for (int q = 0; q < nq; ++q)
      {
         const real_t wq = W[q] * DJ(q, e);
         vol += wq;
         for (int d = 0; d < dim; ++d)
         {
            real_t a = 0.0;
            for (int c = 0; c < dim; ++c) { a += D(q, c, d, e) * D(q, c, d, e); }
            s[d] += wq * a;
         }
      }
      for (int d = 0; d < dim; ++d) { G(d, e) = sqrt(s[d] / vol); }
   });
}

// The same g in 3D straight from the velocity E-vector U(dx, dy, dz, c, e),
// with no quadrature-point buffer. Per component c and direction d, the
// sum-factorized contraction with the 1D derivative Gd along d and the basis
// B along the other two: x into t1, then each thread's (qx, qy) column
// contracts y into r and z point by point, adding w_q detJ_q to the column's
// volume (once) and w_q detJ_q (du_c/dxi_d)^2 to its sum for d. The columns
// are summed in a fixed order at the end: deterministic.
void Fused3D(int ne, int D1D, int Q1D, const Array<real_t>& b,
             const Array<real_t>& gd, const Array<real_t>& w,
             const Vector& detj, const Vector& ue, Vector& g)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto Gd = Reshape(gd.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto DJ = Reshape(detj.Read(), Q1D, Q1D, Q1D, ne);
   const auto U = Reshape(ue.Read(), D1D, D1D, D1D, 3, ne);
   auto G = Reshape(g.Write(), 3, ne);
   const int T = (D1D > Q1D) ? D1D : Q1D;
   mfem::forall_2D(ne, T, T, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD = DofQuadLimits::MAX_D1D;
      constexpr int MQ = DofQuadLimits::MAX_Q1D;
      MFEM_SHARED real_t sB[MQ][MD], sG[MQ][MD];
      MFEM_SHARED real_t t1[MD][MD][MQ];  // [dz][dy][qx]
      MFEM_SHARED real_t col[4][MQ][MQ]; // volume, d = 0, 1, 2; [qy][qx]
      MFEM_FOREACH_THREAD(d, y, D1D)
      {
         MFEM_FOREACH_THREAD(q, x, Q1D)
         {
            sB[q][d] = B(q, d);
            sG[q][d] = Gd(q, d);
         }
      }
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            for (int k = 0; k < 4; ++k) { col[k][qy][qx] = 0.0; }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         for (int d = 0; d < 3; ++d)
         {
            const real_t (*bx)[MD] = (d == 0) ? sG : sB;
            const real_t (*by)[MD] = (d == 1) ? sG : sB;
            const real_t (*bz)[MD] = (d == 2) ? sG : sB;
            // x: threads (qx, dy), loop dz, dx.
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     real_t s = 0.0;
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        s += bx[qx][dx] * U(dx, dy, dz, c, e);
                     }
                     t1[dz][dy][qx] = s;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // y, then z point by point: threads (qx, qy).
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  real_t r[MD];
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     real_t s = 0.0;
                     for (int dy = 0; dy < D1D; ++dy) { s += by[qy][dy] * t1[dz][dy][qx]; }
                     r[dz] = s;
                  }
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     real_t v = 0.0;
                     for (int dz = 0; dz < D1D; ++dz) { v += bz[qz][dz] * r[dz]; }
                     const real_t wq = W(qx, qy, qz) * DJ(qx, qy, qz, e);
                     if (c == 0 && d == 0) { col[0][qy][qx] += wq; }
                     col[1 + d][qy][qx] += wq * v * v;
                  }
               }
            }
            MFEM_SYNC_THREAD; // t1 is rewritten by the next pass
         }
      }
      if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0)
      {
         real_t s[4] = {0.0, 0.0, 0.0, 0.0};
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int k = 0; k < 4; ++k) { s[k] += col[k][qy][qx]; }
            }
         }
         for (int d = 0; d < 3; ++d) { G(d, e) = sqrt(s[1 + d] / s[0]); }
      }
   });
}

} // namespace amr_indicator

GradientIndicator::GradientIndicator(const ParFiniteElementSpace& vfes,
                                     const RuleBook& rules)
   : vfes_(vfes)
{
   ParMesh& mesh = *vfes.GetParMesh();
   dim_ = mesh.Dimension();
   ne_ = vfes.GetNE();
   MFEM_VERIFY(dim_ == 2 || dim_ == 3, "gradient_indicator: dim must be 2 or 3");
   MFEM_VERIFY(vfes.GetVDim() == dim_,
               "gradient_indicator: velocity vdim must equal dim");
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const int ku = vfes.GetTypicalFE()->GetOrder();
   ir_ = &rules.Get(geom, 2 * ku + dim_ - 1);
   restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   geom_ = mesh.GetGeometricFactors(*ir_, GeometricFactors::DETERMINANTS,
                                    Device::GetDeviceMemoryType());
   // The 1D maps are the ones MFEM's QuadratureInterpolator uses for the
   // same rule (tensor products of the 1D rule of its order).
   const DofToQuad& maps =
      vfes.GetTypicalFE()->GetDofToQuad(*ir_, DofToQuad::TENSOR);
   const DeviceDofQuadLimits& lim = DeviceDofQuadLimits::Get();
   if (dim_ == 3 && maps.ndof <= lim.MAX_D1D && maps.nqpt <= lim.MAX_Q1D)
   {
      maps_ = &maps;
   }
   else
   {
      qi_ = vfes.GetQuadratureInterpolator(*ir_);
      qi_->SetOutputLayout(QVectorLayout::byNODES);
      qd_.SetSize(ir_->GetNPoints() * dim_ * dim_ * ne_);
      qd_.UseDevice(true);
   }
}

void GradientIndicator::UseScratch(ElementScratch& scratch)
{
   const int n = restr_->Height();
   if (n <= scratch.Size()) { scratch.Lend(0, n, ue_); }
}

void GradientIndicator::Compute(const ParGridFunction& u, Vector& g) const
{
   INCNS_PROFILE("amr::indicator");
   MFEM_VERIFY(u.ParFESpace() == &vfes_,
               "gradient_indicator: u must live on the indicator's space");
   if (ue_.Size() != restr_->Height())
   {
      ue_.SetSize(restr_->Height()); // no scratch lent: an own buffer
      ue_.UseDevice(true);
   }
   restr_->Mult(u, ue_);
   g.SetSize(dim_ * ne_);
   g.UseDevice(true);
   if (ne_ == 0) { return; }
   if (maps_)
   {
      amr_indicator::Fused3D(ne_, maps_->ndof, maps_->nqpt, maps_->B, maps_->G,
                             ir_->GetWeights(), geom_->detJ, ue_, g);
      return;
   }
   qi_->Derivatives(ue_, qd_);
   amr_indicator::Reduce(ne_, ir_->GetNPoints(), dim_, ir_->GetWeights(),
                         geom_->detJ, qd_, g);
}

void GradientIndicator::Eta(const Vector& g, int dim, Vector& eta)
{
   const int ne = g.Size() / dim;
   eta.SetSize(ne);
   const real_t* G = g.HostRead();
   real_t* E = eta.HostWrite();
   for (int e = 0; e < ne; ++e)
   {
      real_t s = 0.0;
      for (int d = 0; d < dim; ++d) { s += G[d + dim * e] * G[d + dim * e]; }
      E[e] = std::sqrt(s);
   }
}

} // namespace incns
