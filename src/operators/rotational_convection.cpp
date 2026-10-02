#include "operators/rotational_convection.hpp"

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

#include <cmath>
#include <set>
#include <tuple>

// This TU defines device kernels (forall_2D + MFEM_SHARED tiles). A host
// compiler would build them as host loops over device pointers, which
// segfaults on a GPU; src/CMakeLists.txt marks this file LANGUAGE CUDA, and
// this line turns a lost entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "rotational_convection.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernels live in a NAMED namespace: nvcc rejects extended lambdas whose
// enclosing function has internal linkage, so an anonymous namespace is out.
namespace rotconv
{

using kernels::internal::SetMaxOf;

// (dim, d1d, q1d) keys with compile-time kernels; the fallback-size check in
// AssemblePA applies only outside this set (the dispatch table is private).
std::set<std::tuple<int, int, int>>& Specialized()
{
   static std::set<std::tuple<int, int, int>> keys;
   return keys;
}

// ---------------------------------------------------------------------------
// Setup: d_q = alpha w_q (H_ij - H_ji), H = grad_xi(w) adj(J) = detJ grad_x(w)
// (spec 1.3, 5.2). Mirrors MFEM's SmemPAConvectionNLApply: the vd register
// tiles hold every component's reference gradient at once, g[c][d] =
// dw_c/dxi_d. GeometricFactors J(q, i, j, e) = dx_i/dxi_j, column-major.
// ---------------------------------------------------------------------------
template <int T_D1D = 0, int T_Q1D = 0>
void Setup2D(const int NE, const real_t alpha, const real_t* b,
             const real_t* g, const real_t* w, const real_t* j,
             const real_t* we, real_t* d, const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto W = Reshape(w, Q1D, Q1D);
   const auto J = Reshape(j, Q1D, Q1D, DIM, DIM, NE);
   const auto WE = Reshape(we, D1D, D1D, VDIM, NE);
   auto D = Reshape(d, Q1D, Q1D, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);
      kernels::internal::LoadDofs2d(e, D1D, WE, r0);
      kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1);
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t J00 = J(qx, qy, 0, 0, e), J01 = J(qx, qy, 0, 1, e);
            const real_t J10 = J(qx, qy, 1, 0, e), J11 = J(qx, qy, 1, 1, e);
            // A = adj(J) = detJ inv(J), A[d][k].
            const real_t A00 = J11, A01 = -J01, A10 = -J10, A11 = J00;
            // H[c][k] = sum_d g[c][d] A[d][k]; only the off-diagonals.
            const real_t H10 = r1[1][0][qy][qx] * A00 + r1[1][1][qy][qx] * A10;
            const real_t H01 = r1[0][0][qy][qx] * A01 + r1[0][1][qy][qx] * A11;
            D(qx, qy, e) = alpha * W(qx, qy) * (H10 - H01);
         }
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void Setup3D(const int NE, const real_t alpha, const real_t* b,
             const real_t* g, const real_t* w, const real_t* j,
             const real_t* we, real_t* d, const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto W = Reshape(w, Q1D, Q1D, Q1D);
   const auto J = Reshape(j, Q1D, Q1D, Q1D, DIM, DIM, NE);
   const auto WE = Reshape(we, D1D, D1D, D1D, VDIM, NE);
   auto D = Reshape(d, Q1D, Q1D, Q1D, DIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, b, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, g, sG);
      kernels::internal::LoadDofs3d(e, D1D, WE, r0);
      kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               real_t Jm[3][3];
               for (int i = 0; i < 3; ++i)
                  for (int jj = 0; jj < 3; ++jj)
                  {
                     Jm[i][jj] = J(qx, qy, qz, i, jj, e);
                  }
               // A = adj(J) = detJ inv(J), A[d][k].
               real_t A[3][3];
               A[0][0] = Jm[1][1] * Jm[2][2] - Jm[1][2] * Jm[2][1];
               A[0][1] = Jm[0][2] * Jm[2][1] - Jm[0][1] * Jm[2][2];
               A[0][2] = Jm[0][1] * Jm[1][2] - Jm[0][2] * Jm[1][1];
               A[1][0] = Jm[1][2] * Jm[2][0] - Jm[1][0] * Jm[2][2];
               A[1][1] = Jm[0][0] * Jm[2][2] - Jm[0][2] * Jm[2][0];
               A[1][2] = Jm[0][2] * Jm[1][0] - Jm[0][0] * Jm[1][2];
               A[2][0] = Jm[1][0] * Jm[2][1] - Jm[1][1] * Jm[2][0];
               A[2][1] = Jm[0][1] * Jm[2][0] - Jm[0][0] * Jm[2][1];
               A[2][2] = Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0];
               // H[c][k] = sum_d g[c][d] A[d][k]; only the off-diagonals.
               auto H = [&](int c, int k)
               {
                  real_t h = 0.0;
                  for (int dd = 0; dd < 3; ++dd)
                  {
                     h += r1[c][dd][qz][qy][qx] * A[dd][k];
                  }
                  return h;
               };
               const real_t aw = alpha * W(qx, qy, qz);
               D(qx, qy, qz, 0, e) = aw * (H(2, 1) - H(1, 2));
               D(qx, qy, qz, 1, e) = aw * (H(0, 2) - H(2, 0));
               D(qx, qy, qz, 2, e) = aw * (H(1, 0) - H(0, 1));
            }
         }
      }
   });
}

// ---------------------------------------------------------------------------
// Apply: MFEM's SmemPAVectorMassApply{2,3}D with the coefficient block
// replaced by the pointwise [d_q]_x (omega x u). sign = +1 for N, -1 for N^T;
// it multiplies a finished product, so the transpose is bitwise -N x.
// ---------------------------------------------------------------------------
template <int T_D1D = 0, int T_Q1D = 0>
void Apply2D(const int NE, const real_t sign, const Array<real_t>& b,
             const Vector& d, const Vector& x, Vector& y,
             const int d1d = 0, const int q1d = 0)
{
   static constexpr int VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read();
   const auto D = Reshape(d.Read(), Q1D, Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::v_regs2d_t<VDIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadDofs2d(e, D1D, X, r0);
      kernels::internal::Eval2d(D1D, Q1D, smem, sB, r0, r1);
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            const real_t Qx = r1[0][qy][qx];
            const real_t Qy = r1[1][qy][qx];
            const real_t D0 = D(qx, qy, e);
            r0[0][qy][qx] = -sign * (D0 * Qy);
            r0[1][qy][qx] = sign * (D0 * Qx);
         }
      }
      kernels::internal::EvalTranspose2d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs2d(e, D1D, r1, Y);
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void Apply3D(const int NE, const real_t sign, const Array<real_t>& b,
             const Vector& d, const Vector& x, Vector& y,
             const int d1d = 0, const int q1d = 0)
{
   static constexpr int VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read();
   const auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, VDIM, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::v_regs3d_t<VDIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadDofs3d(e, D1D, X, r0);
      kernels::internal::Eval3d(D1D, Q1D, smem, sB, r0, r1);
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t Qx = r1[0][qz][qy][qx];
               const real_t Qy = r1[1][qz][qy][qx];
               const real_t Qz = r1[2][qz][qy][qx];
               const real_t D0 = D(qx, qy, qz, 0, e);
               const real_t D1 = D(qx, qy, qz, 1, e);
               const real_t D2 = D(qx, qy, qz, 2, e);
               r0[0][qz][qy][qx] = sign * (D1 * Qz - D2 * Qy);
               r0[1][qz][qy][qx] = sign * (D2 * Qx - D0 * Qz);
               r0[2][qz][qy][qx] = sign * (D0 * Qy - D1 * Qx);
            }
         }
      }
      kernels::internal::EvalTranspose3d(D1D, Q1D, smem, sB, r0, r1);
      kernels::internal::WriteDofs3d(e, D1D, r1, Y);
   });
}

} // namespace rotconv

/// \cond DO_NOT_DOCUMENT
// Registry plumbing: compile-time specializations + untemplated fallbacks.
// (Generated by MFEM_REGISTER_KERNELS, so doxygen cannot match these.)
template <int DIM, int D1D, int Q1D>
VectorRotationalConvectionIntegrator::ApplyType
VectorRotationalConvectionIntegrator::RotConvApplyPA::Kernel()
{
   if constexpr(DIM == 2) { return rotconv::Apply2D<D1D, Q1D>; }
   else if constexpr(DIM == 3) { return rotconv::Apply3D<D1D, Q1D>; }
   MFEM_ABORT("VectorRotationalConvectionIntegrator: unsupported dim");
}

VectorRotationalConvectionIntegrator::ApplyType
VectorRotationalConvectionIntegrator::RotConvApplyPA::Fallback(int dim, int,
      int)
{
   if (dim == 2) { return rotconv::Apply2D; }
   else if (dim == 3) { return rotconv::Apply3D; }
   MFEM_ABORT("VectorRotationalConvectionIntegrator: unsupported dim");
}

template <int DIM, int D1D, int Q1D>
VectorRotationalConvectionIntegrator::SetupType
VectorRotationalConvectionIntegrator::RotConvSetupPA::Kernel()
{
   if constexpr(DIM == 2) { return rotconv::Setup2D<D1D, Q1D>; }
   else if constexpr(DIM == 3) { return rotconv::Setup3D<D1D, Q1D>; }
   MFEM_ABORT("VectorRotationalConvectionIntegrator: unsupported dim");
}

VectorRotationalConvectionIntegrator::SetupType
VectorRotationalConvectionIntegrator::RotConvSetupPA::Fallback(int dim, int,
      int)
{
   if (dim == 2) { return rotconv::Setup2D; }
   else if (dim == 3) { return rotconv::Setup3D; }
   MFEM_ABORT("VectorRotationalConvectionIntegrator: unsupported dim");
}

// Specializations (spec 5.6) for p = 1..8 (D1D = p + 1), both dims, at every
// Q1D a production rule can produce -- measured on the installed MFEM:
// collocated/Gauss p+1; VectorConvectionNLFIntegrator::GetRule at mesh order
// 1 and 2 (2D and 3D); the 3/2 rule ceil(3(p+1)/2); and the RuleBook's
// dealiased order 3p (Convection::DealiasedOrder). Union per D1D below.
namespace
{
template <int DIM, int D1D, int Q1D>
void AddRotConvSpecialization()
{
   using I = VectorRotationalConvectionIntegrator;
   I::RotConvApplyPA::Specialization<DIM, D1D, Q1D>::Add();
   I::RotConvSetupPA::Specialization<DIM, D1D, Q1D>::Add();
   rotconv::Specialized().insert(std::make_tuple(DIM, D1D, Q1D));
}

template <int DIM>
void AddRotConvSpecializations()
{
   AddRotConvSpecialization<DIM, 2, 2>();   // p = 1
   AddRotConvSpecialization<DIM, 2, 3>();
   AddRotConvSpecialization<DIM, 2, 4>();
   AddRotConvSpecialization<DIM, 3, 3>();   // p = 2
   AddRotConvSpecialization<DIM, 3, 4>();
   AddRotConvSpecialization<DIM, 3, 5>();
   AddRotConvSpecialization<DIM, 4, 4>();   // p = 3
   AddRotConvSpecialization<DIM, 4, 5>();
   AddRotConvSpecialization<DIM, 4, 6>();
   AddRotConvSpecialization<DIM, 4, 7>();
   AddRotConvSpecialization<DIM, 5, 5>();   // p = 4
   AddRotConvSpecialization<DIM, 5, 7>();
   AddRotConvSpecialization<DIM, 5, 8>();
   AddRotConvSpecialization<DIM, 6, 6>();   // p = 5
   AddRotConvSpecialization<DIM, 6, 8>();
   AddRotConvSpecialization<DIM, 6, 9>();
   AddRotConvSpecialization<DIM, 6, 10>();
   AddRotConvSpecialization<DIM, 7, 7>();   // p = 6
   AddRotConvSpecialization<DIM, 7, 10>();
   AddRotConvSpecialization<DIM, 7, 11>();
   AddRotConvSpecialization<DIM, 8, 8>();   // p = 7
   AddRotConvSpecialization<DIM, 8, 11>();
   AddRotConvSpecialization<DIM, 8, 12>();
   AddRotConvSpecialization<DIM, 8, 13>();
   AddRotConvSpecialization<DIM, 9, 9>();   // p = 8
   AddRotConvSpecialization<DIM, 9, 13>();
   AddRotConvSpecialization<DIM, 9, 14>();
}

struct RotConvRegistrar
{
   RotConvRegistrar()
   {
      AddRotConvSpecializations<2>();
      AddRotConvSpecializations<3>();
   }
};
const RotConvRegistrar rotconv_registrar;
} // namespace
/// \endcond

const IntegrationRule& VectorRotationalConvectionIntegrator::GetRule(
   const FiniteElement& fe, const ElementTransformation& T)
{
   return VectorConvectionNLFIntegrator::GetRule(fe, T);
}

void VectorRotationalConvectionIntegrator::AssemblePA(
   const FiniteElementSpace& fes)
{
   MFEM_VERIFY(!DeviceCanUseCeed(), "VectorRotationalConvectionIntegrator: "
               "libCEED backends are not supported (they pass L-vectors)");
   Mesh* mesh = fes.GetMesh();
   const FiniteElement& el = *fes.GetTypicalFE();
   dim_ = mesh->Dimension();
   MFEM_VERIFY(dim_ == 2 || dim_ == 3,
               "VectorRotationalConvectionIntegrator: dim must be 2 or 3");
   MFEM_VERIFY(mesh->SpaceDimension() == dim_,
               "VectorRotationalConvectionIntegrator: sdim must equal dim");
   MFEM_VERIFY(fes.GetVDim() == dim_,
               "VectorRotationalConvectionIntegrator: vdim must equal dim");
   MFEM_VERIFY(dynamic_cast<const TensorBasisElement*>(&el) &&
               el.GetMapType() == FiniteElement::VALUE,
               "VectorRotationalConvectionIntegrator: tensor-product H1 "
               "elements (quads/hexes) required");
   MFEM_VERIFY(mesh->GetNumGeometries(dim_) == 1 && !fes.IsVariableOrder() &&
               fes.GetNURBSext() == nullptr,
               "VectorRotationalConvectionIntegrator: one geometry, uniform "
               "order, no NURBS");

   const IntegrationRule* ir =
      IntRule ? IntRule
      : &GetRule(el, *mesh->GetTypicalElementTransformation());
   pa_ir_ = ir;
   pa_mesh_ = mesh;
   trial_fec_ = fes.FEColl()->Name();

   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                         ? Device::GetDeviceMemoryType() : pa_mt;
   ne_ = fes.GetNE();
   geom_ = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps_ = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   d1d_ = maps_->ndof;
   q1d_ = maps_->nqpt;
   // The register tiles are sized by MQ1 and hold dof values before
   // interpolation, so q1d < d1d would overflow them.
   MFEM_VERIFY(d1d_ <= q1d_, "VectorRotationalConvectionIntegrator: needs "
               "q1d >= d1d, got d1d = " << d1d_ << ", q1d = " << q1d_);
   if (!rotconv::Specialized().count(std::make_tuple(dim_, d1d_, q1d_)))
   {
      const DeviceDofQuadLimits& lim = DeviceDofQuadLimits::Get();
      MFEM_VERIFY(d1d_ <= lim.MAX_D1D && q1d_ <= lim.MAX_Q1D,
                  "VectorRotationalConvectionIntegrator: (dim, d1d, q1d) = ("
                  << dim_ << ", " << d1d_ << ", " << q1d_ << ") has no "
                  "specialization and exceeds the fallback limits (" <<
                  lim.MAX_D1D << ", " << lim.MAX_Q1D << "); add a "
                  "specialization in rotational_convection.cpp");
   }

   const int nq = ir->GetNPoints();
   pa_data_.SetSize((dim_ == 3 ? 3 : 1) * nq * ne_, mt);
   pa_data_.UseDevice(true);

   w_fes_ = nullptr; // force a fresh bind: the trial space may have changed
   BindLaggedSpace();
   UpdateVorticity();
}

void VectorRotationalConvectionIntegrator::BindLaggedSpace()
{
   const FiniteElementSpace& wfes = *w_->FESpace();
   if (&wfes == w_fes_) { return; }
   MFEM_VERIFY(wfes.GetMesh() == pa_mesh_,
               "VectorRotationalConvectionIntegrator: w must live on the "
               "trial space's mesh");
   MFEM_VERIFY(wfes.GetVDim() == dim_,
               "VectorRotationalConvectionIntegrator: w must have vdim == dim");
   MFEM_VERIFY(trial_fec_ == wfes.FEColl()->Name(),
               "VectorRotationalConvectionIntegrator: w must use the trial "
               "space's FE collection (" << trial_fec_ << "), got "
               << wfes.FEColl()->Name());
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                         ? Device::GetDeviceMemoryType() : pa_mt;
   w_restr_ = wfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   w_e_.SetSize(w_restr_->Height(), mt);
   w_e_.UseDevice(true);
   w_fes_ = &wfes;
}

void VectorRotationalConvectionIntegrator::UpdateVorticity()
{
   MFEM_VERIFY(maps_, "VectorRotationalConvectionIntegrator: "
               "UpdateVorticity() before AssemblePA()");
   BindLaggedSpace(); // no-op unless SetLaggedVelocity changed the space
   w_restr_->Mult(*w_, w_e_);
   RotConvSetupPA::Run(dim_, d1d_, q1d_, ne_, alpha_, maps_->B.Read(),
                       maps_->G.Read(), pa_ir_->GetWeights().Read(),
                       geom_->J.Read(), w_e_.Read(), pa_data_.Write(), d1d_,
                       q1d_);
}

void VectorRotationalConvectionIntegrator::AddMultPA(const Vector& x,
      Vector& y) const
{
   RotConvApplyPA::Run(dim_, d1d_, q1d_, ne_, 1.0, maps_->B, pa_data_, x, y,
                       d1d_, q1d_);
}

void VectorRotationalConvectionIntegrator::AddMultTransposePA(
   const Vector& x, Vector& y) const
{
   RotConvApplyPA::Run(dim_, d1d_, q1d_, ne_, -1.0, maps_->B, pa_data_, x, y,
                       d1d_, q1d_);
}

void VectorRotationalConvectionIntegrator::AssembleDiagonalPA(Vector&)
{
   // diag(N) = 0 exactly (P2): nothing to add.
}

// ---------------------------------------------------------------------------
// Nodal skew (spec 5.9): s(a, c, e) += sum_q B_a(q)^2 d_q[c], the diagonal of
// a scalar mass matrix weighted by d[c]. Sum-factorized z -> y -> x with
// forall_2D(ne, Q1D, Q1D), NOT MFEM's Q1D^3-thread diagonal layout (capped at
// q1d <= 10). Runs once per step, so no specializations: the shared tiles are
// sized by the device's DofQuadLimits (CUDA 14: two 14^3 tiles, ~44 KB).
// ---------------------------------------------------------------------------
void VectorRotationalConvectionIntegrator::AddNodalSkewPA(Vector& s_e) const
{
   MFEM_VERIFY(maps_, "VectorRotationalConvectionIntegrator: "
               "AddNodalSkewPA() before AssemblePA()");
   const DeviceDofQuadLimits& lim = DeviceDofQuadLimits::Get();
   MFEM_VERIFY(d1d_ <= lim.MAX_D1D && q1d_ <= lim.MAX_Q1D,
               "VectorRotationalConvectionIntegrator::AddNodalSkewPA: d1d = "
               << d1d_ << ", q1d = " << q1d_ << " exceed the device limits ("
               << lim.MAX_D1D << ", " << lim.MAX_Q1D << ")");
   const int NE = ne_, D1D = d1d_, Q1D = q1d_;
   MFEM_VERIFY(s_e.Size() == (dim_ == 3 ? D1D * D1D* D1D : D1D * D1D) *
               dim_ * NE, "VectorRotationalConvectionIntegrator::AddNodalSkewPA: "
               "s_e is not an E-vector of the assembled velocity space");
   const auto B = Reshape(maps_->B.Read(), Q1D, D1D);

   if (dim_ == 2)
   {
      const auto D = Reshape(pa_data_.Read(), Q1D, Q1D, NE);
      auto S = Reshape(s_e.ReadWrite(), D1D, D1D, 2, NE);
      mfem::forall_2D(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD = DofQuadLimits::MAX_D1D;
         constexpr int MQ = DofQuadLimits::MAX_Q1D;
         MFEM_SHARED real_t sB2[MD][MQ], t1[MD][MQ];
         MFEM_FOREACH_THREAD(d, y, D1D)
         {
            MFEM_FOREACH_THREAD(q, x, Q1D) { sB2[d][q] = B(q, d) * B(q, d); }
         }
         MFEM_SYNC_THREAD;
         // y: t1[dy][qx] = sum_qy B2[dy][qy] D(qx, qy)
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy) { u += sB2[dy][qy] * D(qx, qy, e); }
               t1[dy][qx] = u;
            }
         }
         MFEM_SYNC_THREAD;
         // x: S(dx, dy) += sum_qx B2[dx][qx] t1[dy][qx]
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx) { u += sB2[dx][qx] * t1[dy][qx]; }
               S(dx, dy, 0, e) += u;
            }
         }
      });
   }
   else
   {
      const auto D = Reshape(pa_data_.Read(), Q1D, Q1D, Q1D, 3, NE);
      auto S = Reshape(s_e.ReadWrite(), D1D, D1D, D1D, 3, NE);
      mfem::forall_2D(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MD = DofQuadLimits::MAX_D1D;
         constexpr int MQ = DofQuadLimits::MAX_Q1D;
         MFEM_SHARED real_t sB2[MD][MQ];
         MFEM_SHARED real_t t1[MD][MQ][MQ]; // [dz][qy][qx]
         MFEM_SHARED real_t t2[MD][MD][MQ]; // [dz][dy][qx]
         MFEM_FOREACH_THREAD(d, y, D1D)
         {
            MFEM_FOREACH_THREAD(q, x, Q1D) { sB2[d][q] = B(q, d) * B(q, d); }
         }
         MFEM_SYNC_THREAD;
         for (int c = 0; c < 3; ++c)
         {
            // z: threads (qx, qy), loop dz, qz.
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     real_t u = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                     {
                        u += sB2[dz][qz] * D(qx, qy, qz, c, e);
                     }
                     t1[dz][qy][qx] = u;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // y: threads (qx, dy), loop dz, qy.
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     real_t u = 0.0;
                     for (int qy = 0; qy < Q1D; ++qy)
                     {
                        u += sB2[dy][qy] * t1[dz][qy][qx];
                     }
                     t2[dz][dy][qx] = u;
                  }
               }
            }
            MFEM_SYNC_THREAD;
            // x: threads (dx, dy), loop dz, qx; add to the output.
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     real_t u = 0.0;
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        u += sB2[dx][qx] * t2[dz][dy][qx];
                     }
                     S(dx, dy, dz, c, e) += u;
                  }
               }
            }
            MFEM_SYNC_THREAD; // t1, t2 are reused by the next component
         }
      });
   }
}

// ---------------------------------------------------------------------------
// Rotation number (spec 5.10): vol_q = W_q detJ_q, |omega_q| = |d_q| /
// (alpha vol_q), mu_q = |omega_q| / sigma. Device reductions via
// Vector::Max/Sum, then MPI over the ParMesh if there is one.
// ---------------------------------------------------------------------------
RotationNumberStats
VectorRotationalConvectionIntegrator::GetRotationNumberStats(
   real_t sigma, real_t threshold) const
{
   MFEM_VERIFY(maps_, "VectorRotationalConvectionIntegrator: "
               "GetRotationNumberStats() before AssemblePA()");
   MFEM_VERIFY(alpha_ != 0.0, "VectorRotationalConvectionIntegrator: "
               "GetRotationNumberStats() needs alpha != 0");
   const int nq = pa_ir_->GetNPoints(), NE = ne_, dim = dim_;
   const MemoryType mt = pa_data_.GetMemory().GetMemoryType();
   if (diag_tmp0_.Size() != nq * NE)
   {
      diag_tmp0_.SetSize(nq * NE, mt);
      diag_tmp1_.SetSize(nq * NE, mt);
      diag_tmp0_.UseDevice(true);
      diag_tmp1_.UseDevice(true);
   }
   const real_t alpha = alpha_;
   const auto W = Reshape(pa_ir_->GetWeights().Read(), nq);
   const auto J = Reshape(geom_->J.Read(), nq, dim, dim, NE);
   const auto D = Reshape(pa_data_.Read(), nq, dim == 3 ? 3 : 1, NE);
   auto MU = Reshape(diag_tmp0_.Write(), nq, NE);
   auto V = Reshape(diag_tmp1_.Write(), nq, NE);

   // Pass 1: volume per point (for the total).
   mfem::forall(nq * NE, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % nq, e = i / nq;
      real_t det;
      if (dim == 2)
      {
         det = J(q, 0, 0, e) * J(q, 1, 1, e) - J(q, 0, 1, e) * J(q, 1, 0, e);
      }
      else
      {
         det = J(q, 0, 0, e) * (J(q, 1, 1, e) * J(q, 2, 2, e) -
                                J(q, 1, 2, e) * J(q, 2, 1, e)) -
               J(q, 0, 1, e) * (J(q, 1, 0, e) * J(q, 2, 2, e) -
                                J(q, 1, 2, e) * J(q, 2, 0, e)) +
               J(q, 0, 2, e) * (J(q, 1, 0, e) * J(q, 2, 1, e) -
                                J(q, 1, 1, e) * J(q, 2, 0, e));
      }
      const real_t vol = W(q) * det;
      real_t dn;
      if (dim == 2) { dn = fabs(D(q, 0, e)); }
      else
      {
         dn = sqrt(D(q, 0, e) * D(q, 0, e) + D(q, 1, e) * D(q, 1, e) +
                   D(q, 2, e) * D(q, 2, e));
      }
      MU(q, e) = dn / (fabs(alpha) * vol) / sigma;
      V(q, e) = vol;
   });
   real_t vol_total = diag_tmp1_.Size() ? diag_tmp1_.Sum() : 0.0;
   real_t max_mu = diag_tmp0_.Size() ? diag_tmp0_.Max() : 0.0;

   // Pass 2: volume above the threshold.
   const auto MUr = Reshape(diag_tmp0_.Read(), nq, NE);
   auto Vw = Reshape(diag_tmp1_.ReadWrite(), nq, NE);
   mfem::forall(nq * NE, [ = ] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % nq, e = i / nq;
      if (!(MUr(q, e) > threshold)) { Vw(q, e) = 0.0; }
   });
   real_t vol_above = diag_tmp1_.Size() ? diag_tmp1_.Sum() : 0.0;

   if (auto* pmesh = dynamic_cast<const ParMesh*>(pa_mesh_))
   {
      MPI_Comm comm = pmesh->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &max_mu, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, comm);
      real_t sums[2] = {vol_above, vol_total};
      MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
      vol_above = sums[0];
      vol_total = sums[1];
   }
   RotationNumberStats st;
   st.max_mu = max_mu;
   st.vol_fraction = vol_total > 0.0 ? vol_above / vol_total : 0.0;
   st.threshold = threshold;
   return st;
}

} // namespace incns
