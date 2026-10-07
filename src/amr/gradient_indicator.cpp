#include "amr/gradient_indicator.hpp"

#include "util/profiler.hpp"
#include "mfem/general/forall.hpp"

#include <cmath>

// This TU defines a device kernel (mfem::forall). A host compiler would build
// it as a host loop over device pointers, which segfaults on a GPU;
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
   qi_ = vfes.GetQuadratureInterpolator(*ir_);
   qi_->SetOutputLayout(QVectorLayout::byNODES);
   geom_ = mesh.GetGeometricFactors(*ir_, GeometricFactors::DETERMINANTS,
                                    Device::GetDeviceMemoryType());
   ue_.SetSize(restr_->Height());
   ue_.UseDevice(true);
   qd_.SetSize(ir_->GetNPoints() * dim_ * dim_ * ne_);
   qd_.UseDevice(true);
}

void GradientIndicator::Compute(const ParGridFunction& u, Vector& g) const
{
   INCNS_PROFILE("amr::indicator");
   MFEM_VERIFY(u.ParFESpace() == &vfes_,
               "gradient_indicator: u must live on the indicator's space");
   restr_->Mult(u, ue_);
   qi_->Derivatives(ue_, qd_);
   g.SetSize(dim_ * ne_);
   g.UseDevice(true);
   if (ne_ == 0) { return; }
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
