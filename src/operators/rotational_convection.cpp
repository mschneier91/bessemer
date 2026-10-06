#include "operators/rotational_convection.hpp"

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

#include <cmath>
#include <set>
#include <tuple>
#include <utility>

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

// Specializations: the DEALIASED Gauss-Legendre rule only, p = 1..5 (human
// decision 2026-10-05: the rotational term is always over-integrated with
// Gauss-Legendre -- under-integrating the nonlinear term has gone badly on
// energy). The solver's rule is the RuleBook's order-3p GL rule
// (Convection::DealiasedOrder), ceil((3p+1)/2) = (3p+2)/2 points per
// direction, so the list is generated from that formula: (D1D, Q1D) = (2,2),
// (3,4), (4,5), (5,7), (6,8). Every other rule or order still runs, through
// the generic fallback kernel (correct, ~2.5x slower measured);
// MFEM_REPORT_KERNELS=1 prints each fallback. HasSpecialization() lets a test
// pin the solver's actual rule to this list.
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

// Order p at the dealiased rule: D1D = p + 1, Q1D = (3p + 2) / 2.
template <int DIM, int P>
void AddDealiasedSpecialization()
{
   constexpr int D1D = P + 1;
   constexpr int Q1D = (3 * P + 2) / 2;
   AddRotConvSpecialization<DIM, D1D, Q1D>();
}

template <int DIM, int... P>
void AddDealiasedSpecializations(std::integer_sequence<int, P...>)
{
   const int expand[] = {(AddDealiasedSpecialization<DIM, P>(), 0)...};
   static_cast<void>(expand);
}

struct RotConvRegistrar
{
   RotConvRegistrar()
   {
      AddDealiasedSpecializations<2>(std::integer_sequence<int, 1, 2, 3, 4, 5> {});
      AddDealiasedSpecializations<3>(std::integer_sequence<int, 1, 2, 3, 4, 5> {});
   }
};
const RotConvRegistrar rotconv_registrar;
} // namespace
/// \endcond

bool VectorRotationalConvectionIntegrator::HasSpecialization(int dim, int d1d,
      int q1d)
{
   return rotconv::Specialized().count(std::make_tuple(dim, d1d, q1d)) > 0;
}

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

// ---------------------------------------------------------------------------
// Legacy full assembly (CPU): curl w from the physical gradient of w at each
// point (GridFunction::GetVectorGradient -- independent of the PA setup
// path), then elmat(a + i nd, b + j nd) += alpha w detJ phi_a phi_b
// [omega]_x(i, j). w is read through Trans's element number, so Trans must
// belong to w's mesh -- MFEM's legacy LOR assembly calls this on LOR
// elements, where a high-order w would be read from the wrong element.
// ---------------------------------------------------------------------------
void VectorRotationalConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "VectorRotationalConvectionIntegrator: dim must be 2 or 3");
   MFEM_VERIFY(Trans.GetSpaceDim() == dim,
               "VectorRotationalConvectionIntegrator: sdim must equal dim");
   MFEM_VERIFY(Trans.mesh == w_->FESpace()->GetMesh(),
               "VectorRotationalConvectionIntegrator: the lagged velocity must "
               "live on the mesh being assembled -- for LOR, give the "
               "integrator a w on the LOR space (same true dofs as the "
               "high-order field)");
   MFEM_VERIFY(w_->FESpace()->GetVDim() == dim,
               "VectorRotationalConvectionIntegrator: w must have vdim == dim");

   const IntegrationRule* ir = IntRule ? IntRule : &GetRule(el, Trans);
   Vector shape(nd);
   DenseMatrix G, K(dim);
   elmat.SetSize(dim * nd);
   elmat = 0.0;
   for (int q = 0; q < ir->GetNPoints(); ++q)
   {
      const IntegrationPoint& ip = ir->IntPoint(q);
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      w_->GetVectorGradient(Trans, G); // G(c, k) = dw_c/dx_k
      const real_t wgt = alpha_ * ip.weight * Trans.Weight();
      K = 0.0;
      if (dim == 2)
      {
         const real_t om = G(1, 0) - G(0, 1);
         K(0, 1) = -wgt * om;
         K(1, 0) = wgt * om;
      }
      else
      {
         const real_t o0 = G(2, 1) - G(1, 2);
         const real_t o1 = G(0, 2) - G(2, 0);
         const real_t o2 = G(1, 0) - G(0, 1);
         K(0, 1) = -wgt * o2; K(0, 2) = wgt * o1;
         K(1, 0) = wgt * o2;  K(1, 2) = -wgt * o0;
         K(2, 0) = -wgt * o1; K(2, 1) = wgt * o0;
      }
      for (int i = 0; i < dim; ++i)
         for (int j = 0; j < dim; ++j)
         {
            if (i == j) { continue; }
            for (int a = 0; a < nd; ++a)
               for (int b = 0; b < nd; ++b)
               {
                  elmat(a + i * nd, b + j * nd) += K(i, j) * shape(a) * shape(b);
               }
         }
   }
}

// ---------------------------------------------------------------------------
// Component-block EA: N^{ij}(a, b) = sign_ij sum_q D(q, c_ij) B_a(q) B_b(q),
// with D the PA quadrature data (alpha w detJ omega, from the same setup
// kernel) and [omega]_x(i, j) = sign_ij omega_{c_ij}: in 3D c = 3 - i - j and
// sign = -1 when j == i + 1 (mod 3), else +1; in 2D c = 0, sign -1 for (0,1).
// One thread per (e, a, b) entry, reference basis from the 1D tables, so dofs
// are lexicographic (MFEM's EA order); row-major per element: in a
// column-major Reshape(nd, nd, ne), entry (b, a, e) is row a, column b.
// ---------------------------------------------------------------------------
void VectorRotationalConvectionComponentIntegrator::AssembleEA(
   const FiniteElementSpace& fes, Vector& emat, const bool add)
{
   MFEM_VERIFY(!DeviceCanUseCeed(),
               "VectorRotationalConvectionComponentIntegrator: "
               "libCEED backends are not supported");
   Mesh* mesh = fes.GetMesh();
   const FiniteElement& el = *fes.GetTypicalFE();
   const int dim = mesh->Dimension();
   const FiniteElementSpace& wfes = *w_->FESpace();
   MFEM_VERIFY((dim == 2 || dim == 3) && mesh->SpaceDimension() == dim,
               "VectorRotationalConvectionComponentIntegrator: dim == sdim in "
               "{2, 3} required");
   MFEM_VERIFY(fes.GetVDim() == 1,
               "VectorRotationalConvectionComponentIntegrator: "
               "blocks live on a scalar (vdim == 1) space");
   MFEM_VERIFY(el.GetGeomType() == Geometry::SQUARE ||
               el.GetGeomType() == Geometry::CUBE,
               "VectorRotationalConvectionComponentIntegrator: quads/hexes only");
   MFEM_VERIFY(wfes.GetMesh() == mesh && wfes.GetVDim() == dim &&
               std::string(wfes.FEColl()->Name()) == fes.FEColl()->Name(),
               "VectorRotationalConvectionComponentIntegrator: w must be a "
               "vdim == dim field on the assembly mesh with the space's FE "
               "collection");
   MFEM_VERIFY(0 <= i_block_ && i_block_ < dim && 0 <= j_block_ && j_block_ < dim,
               "VectorRotationalConvectionComponentIntegrator: block index out "
               "of range");

   const int ne = fes.GetNE();
   const int nd = el.GetDof();
   MFEM_VERIFY(emat.Size() == nd * nd * ne,
               "VectorRotationalConvectionComponentIntegrator: emat must be "
               "nd*nd*ne");
   if (i_block_ == j_block_) // [omega]_x has a zero diagonal
   {
      if (!add) { emat = 0.0; }
      return;
   }

   const IntegrationRule* ir =
      IntRule ? IntRule
      : &VectorRotationalConvectionIntegrator::GetRule(
         el, *mesh->GetTypicalElementTransformation());
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors* geom =
      mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   const DofToQuad& maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps.ndof, Q1D = maps.nqpt;
   MFEM_VERIFY(D1D <= Q1D, "VectorRotationalConvectionComponentIntegrator: "
               "needs q1d >= d1d, got d1d = " << D1D << ", q1d = " << Q1D);
   const int nq = ir->GetNPoints();

   // Quadrature data from the integrator's own setup kernel (one call; the
   // buffers are temporaries -- EA runs once per assembly, not per apply).
   const Operator* w_restr =
      wfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   Vector w_e(w_restr->Height(), mt), qdata((dim == 3 ? 3 : 1) * nq * ne, mt);
   w_e.UseDevice(true);
   qdata.UseDevice(true);
   w_restr->Mult(*w_, w_e);
   VectorRotationalConvectionIntegrator::RotConvSetupPA::Run(
      dim, D1D, Q1D, ne, alpha_, maps.B.Read(), maps.G.Read(),
      ir->GetWeights().Read(), geom->J.Read(), w_e.Read(), qdata.Write(), D1D,
      Q1D);

   const int c = (dim == 3) ? 3 - i_block_ - j_block_ : 0;
   const real_t sign = (dim == 3)
                       ? ((j_block_ == (i_block_ + 1) % 3) ? -1.0 : 1.0)
                       : (i_block_ == 0 ? -1.0 : 1.0);
   const int ncomp = (dim == 3) ? 3 : 1;
   const auto B = Reshape(maps.B.Read(), Q1D, D1D);
   const auto D = Reshape(qdata.Read(), nq, ncomp, ne);
   auto E = Reshape(add ? emat.ReadWrite() : emat.Write(), nd, nd, ne);

   mfem::forall(ne * nd * nd, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / (nd * nd);
      const int a = (idx / nd) % nd; // row
      const int b = idx % nd;        // column (fastest -> coalesced writes)
      const int ax = a % D1D, ay = (a / D1D) % D1D, az = a / (D1D * D1D);
      const int bx = b % D1D, by = (b / D1D) % D1D, bz = b / (D1D * D1D);
      const int NQZ = (dim == 3) ? Q1D : 1;
      real_t acc = 0.0;
      for (int qz = 0; qz < NQZ; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + Q1D * qz);
               real_t ba = B(qx, ax) * B(qy, ay);
               real_t bb = B(qx, bx) * B(qy, by);
               if (dim == 3)
               {
                  ba *= B(qz, az);
                  bb *= B(qz, bz);
               }
               acc += D(q, c, e) * ba * bb;
            }
      if (add) { E(b, a, e) += sign * acc; }
      else { E(b, a, e) = sign * acc; }
   });
}

} // namespace incns
