#include "operators/vecdivdiv_integrator.hpp"

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

namespace incns
{

using namespace mfem;

// ---------------------------------------------------------------------------
// Full assembly (the oracle). Mirrors the lambda-only part of
// ElasticityIntegrator::AssembleElementMatrix: physical gradients, GradToDiv
// into the byNODES-flattened divergence vector, rank-1 update.
// ---------------------------------------------------------------------------
void VectorDivDivIntegrator::AssembleElementMatrix(
   const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();
   MFEM_VERIFY(dim == Trans.GetSpaceDim(),
               "VectorDivDivIntegrator: vdim == dim == sdim required");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape_(nd, dim), gshape_(nd, dim);
   Vector divshape_(dim * nd);
#else
   dshape_.SetSize(nd, dim);
   gshape_.SetSize(nd, dim);
   divshape_.SetSize(dim * nd);
#endif
   elmat.SetSize(dim * nd);
   elmat = 0.0;

   const IntegrationRule* ir =
      IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
   for (int q = 0; q < ir->GetNPoints(); ++q)
   {
      const IntegrationPoint& ip = ir->IntPoint(q);
      el.CalcDShape(ip, dshape_);
      Trans.SetIntPoint(&ip);
      Mult(dshape_, Trans.InverseJacobian(), gshape_); // physical gradients
      gshape_.GradToDiv(divshape_); // length dim*nd, byNODES blocks
      real_t w = ip.weight * Trans.Weight();
      if (Q) { w *= Q->Eval(Trans, ip); }
      AddMult_a_VVt(w, divshape_, elmat); // elmat += w * div div^T
   }
}

// ---------------------------------------------------------------------------
// PA kernels. Data layout: D(q..., k + i*dim, e) = A[k][i] (adj(J): row k =
// reference direction, column i = physical component), slot dim*dim = alpha =
// Q w / detJ. The quadrature-point operator is rank 1 (spec par.2.2):
//   S = sum_{i,k} Ghat[i][k] A[k][i]   (unscaled divergence)
//   Yhat[i][k] = alpha S A[k][i]
// so A is loaded once into registers and used for both the contraction and the
// distribution. Tensor machinery is identical to SmemPAVectorDiffusionApply*:
// the only structural change is that ALL components are interpolated before
// the (coupling) pointwise stage.
// ---------------------------------------------------------------------------
namespace vecdivdiv
{

using kernels::internal::SetMaxOf;

// Quadrature data for PA and component EA; defined after the kernels.
void SetupQuadratureData(Mesh& mesh, const IntegrationRule& ir,
                         const GeometricFactors& geom, Coefficient* Q,
                         MemoryType mt, Vector& qdata);

template <int T_D1D = 0, int T_Q1D = 0>
void SmemApply2D(const int NE, const Array<real_t>& b, const Array<real_t>& g,
                 const Vector& d, const Vector& x, Vector& y,
                 const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 2, VDIM = 2;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, DIM * DIM + 1, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs2d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      // Scalar-coupled two-pass form (see the 3D kernel comment): pass 1
      // accumulates S per point component by component; pass 2 rebuilds each
      // output field as t*A[:,c] -- the input gradients are never stored.
      kernels::internal::vd_regs2d_t<1, 1, MQ1> Sreg; // same thread-dim trick
      MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            Sreg[0][0][qy][qx] = 0.0;
         }
      }
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs2d(e, D1D, c, XE, r0);
         kernels::internal::Grad2d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               for (int k = 0; k < DIM; ++k)
               {
                  Sreg[0][0][qy][qx] += r1[c][k][qy][qx] *
                                        DE(qx, qy, k + c * DIM, e);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               const real_t t = DE(qx, qy, DIM * DIM, e) * Sreg[0][0][qy][qx];
               for (int k = 0; k < DIM; ++k)
               {
                  r0[c][k][qy][qx] = t * DE(qx, qy, k + c * DIM, e);
               }
            }
         }
         kernels::internal::GradTranspose2d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs2d(e, D1D, c, c, r1, YE);
         MFEM_SYNC_THREAD; // smem reused by the next GradTranspose
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void SmemApply3D(const int NE, const Array<real_t>& b, const Array<real_t>& g,
                 const Vector& d, const Vector& x, Vector& y,
                 const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D, Q1D, Q1D, DIM * DIM + 1, NE);
   const auto XE = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto YE = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D* T_Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      // The rank-1 structure couples the components through ONE SCALAR per
      // quadrature point: S = sum_{i,k} Ghat[i][k] A[k][i], and the output
      // reference gradients are t*A[:,c] with t = alpha*S -- they do not need
      // the INPUT gradients at all. So pass 1 accumulates S component by
      // component (only one component's register tensor is ever hot -- much
      // better cache behavior than holding all VDIM gradient fields live),
      // and pass 2 reconstructs each output field from (t, A) alone.
      // S as a register FIELD with the same [qy][qx] thread-dim trick as the
      // gradient tensors (0-extent on GPU, full extents on CPU where one
      // thread sweeps all points -- a plain per-thread scalar would wrongly
      // accumulate across points there).
      kernels::internal::vd_regs3d_t<1, 1, MQ1> Sreg;
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               Sreg[0][0][qz][qy][qx] = 0.0;
            }
         }
      }
      // pass 1: for each component, interpolate and fold into S.
      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadDofs3d(e, D1D, c, XE, r0);
         kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  real_t acc = 0.0;
                  for (int k = 0; k < DIM; ++k)
                  {
                     acc += r1[c][k][qz][qy][qx] * DE(qx, qy, qz, k + c * DIM, e);
                  }
                  Sreg[0][0][qz][qy][qx] += acc;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // pass 2: rebuild each component's field as t*A[:,c] and contract back.
      for (int c = 0; c < VDIM; ++c)
      {
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  const real_t t = DE(qx, qy, qz, DIM * DIM, e) *
                                   Sreg[0][0][qz][qy][qx];
                  for (int k = 0; k < DIM; ++k)
                  {
                     r0[c][k][qz][qy][qx] = t * DE(qx, qy, qz, k + c * DIM, e);
                  }
               }
            }
         }
         kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs3d(e, D1D, c, c, r1, YE);
         MFEM_SYNC_THREAD; // smem reused by the next GradTranspose
      }
   });
}

} // namespace vecdivdiv

/// \cond DO_NOT_DOCUMENT
// Kernel registry: compile-time specializations + untemplated fallback. (The
// ApplyPAKernels class is generated by MFEM_REGISTER_KERNELS, so doxygen
// cannot match these definitions to a declaration it can see.)
template <int DIM, int D1D, int Q1D>
VectorDivDivIntegrator::ApplyKernelType
VectorDivDivIntegrator::ApplyPAKernels::Kernel()
{
   if constexpr(DIM == 2) { return vecdivdiv::SmemApply2D<D1D, Q1D>; }
   else if constexpr(DIM == 3) { return vecdivdiv::SmemApply3D<D1D, Q1D>; }
   MFEM_ABORT("VectorDivDivIntegrator: unsupported kernel dim");
}

VectorDivDivIntegrator::ApplyKernelType
VectorDivDivIntegrator::ApplyPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return vecdivdiv::SmemApply2D; }
   else if (dim == 3) { return vecdivdiv::SmemApply3D; }
   MFEM_ABORT("VectorDivDivIntegrator: unsupported kernel dim");
}

// Compile-time (DIM, D1D, Q1D) specializations for the order range bessemer
// runs (through Q6 velocity: D1D in 3..7; GL rules give Q1D ~ D1D+1,
// collocated GLL gives Q1D == D1D). These are NOT just a GPU concern: the
// untemplated fallback sizes its register tensors at the MAX_T1D compile-time
// bound, which wrecks CPU cache behavior (measured 2.6x at p=6 before the
// (7,8) pair was added). Extend this list when running above Q6. Registered once via a static initializer (the same
// pattern the in-tree integrators use).
namespace
{
struct VddRegistrar
{
   VddRegistrar()
   {
      using I = VectorDivDivIntegrator;
      I::AddSpecialization<2, 3, 3>(); I::AddSpecialization<2, 3, 4>();
      I::AddSpecialization<2, 4, 4>(); I::AddSpecialization<2, 4, 5>();
      I::AddSpecialization<2, 5, 5>(); I::AddSpecialization<2, 5, 6>();
      I::AddSpecialization<2, 6, 6>(); I::AddSpecialization<2, 6, 7>();
      I::AddSpecialization<3, 3, 3>(); I::AddSpecialization<3, 3, 4>();
      I::AddSpecialization<3, 4, 4>(); I::AddSpecialization<3, 4, 5>();
      I::AddSpecialization<3, 5, 5>(); I::AddSpecialization<3, 5, 6>();
      I::AddSpecialization<3, 6, 6>(); I::AddSpecialization<3, 6, 7>();
      I::AddSpecialization<3, 7, 7>(); I::AddSpecialization<3, 7, 8>();
      I::AddSpecialization<2, 7, 7>(); I::AddSpecialization<2, 7, 8>();
   }
};
const VddRegistrar vdd_registrar;
} // namespace

/// \endcond

// ---------------------------------------------------------------------------
// Quadrature data shared by the PA setup and the component-block EA:
// (nq, dim*dim + 1, ne) with adj(J) in the A[k][i] slot order k + i*dim, then
// alpha = Q w / detJ (spec par.2.3 formulas, the same expressions the
// vecdiffusion setup uses).
// ---------------------------------------------------------------------------
void vecdivdiv::SetupQuadratureData(Mesh& mesh, const IntegrationRule& ir,
                                    const GeometricFactors& geom,
                                    Coefficient* Q, MemoryType mt,
                                    Vector& qdata)
{
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   QuadratureSpace qs(mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }
   MFEM_VERIFY(coeff.GetVDim() == 1,
               "VectorDivDivIntegrator: scalar coefficient required");

   const int nq = ir.GetNPoints();
   const int pa_size = dim * dim + 1; // adj(J) entries + alpha
   qdata.SetSize(nq * pa_size * ne, mt);
   qdata.UseDevice(true);

   const auto W = Reshape(ir.GetWeights().Read(), nq);
   const auto C = Reshape(coeff.Read(), nq, ne);
   const auto J = Reshape(geom.J.Read(), nq, dim, dim, ne);
   auto D = Reshape(qdata.Write(), nq, pa_size, ne);

   mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / nq, q = idx % nq;
      if (dim == 2)
      {
         const real_t J11 = J(q, 0, 0, e), J12 = J(q, 0, 1, e);
         const real_t J21 = J(q, 1, 0, e), J22 = J(q, 1, 1, e);
         const real_t detJ = J11 * J22 - J12 * J21;
         // A[k][i]: row k = reference direction, column i = physical component.
         D(q, 0, e) = J22;  // A[0][0]
         D(q, 1, e) = -J21; // A[1][0]
         D(q, 2, e) = -J12; // A[0][1]
         D(q, 3, e) = J11;  // A[1][1]
         D(q, 4, e) = C(q, e) * W(q) / detJ; // alpha
      }
      else
      {
         const real_t J11 = J(q, 0, 0, e), J12 = J(q, 0, 1, e), J13 = J(q, 0, 2, e);
         const real_t J21 = J(q, 1, 0, e), J22 = J(q, 1, 1, e), J23 = J(q, 1, 2, e);
         const real_t J31 = J(q, 2, 0, e), J32 = J(q, 2, 1, e), J33 = J(q, 2, 2, e);
         const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         const real_t A11 = (J22 * J33) - (J23 * J32);
         const real_t A12 = (J32 * J13) - (J12 * J33);
         const real_t A13 = (J12 * J23) - (J22 * J13);
         const real_t A21 = (J31 * J23) - (J21 * J33);
         const real_t A22 = (J11 * J33) - (J13 * J31);
         const real_t A23 = (J21 * J13) - (J11 * J23);
         const real_t A31 = (J21 * J32) - (J31 * J22);
         const real_t A32 = (J31 * J12) - (J11 * J32);
         const real_t A33 = (J11 * J22) - (J12 * J21);
         // Columns i of A hold the i-th physical component's pullback: the
         // (k + i*dim) slot order below is A[k][i] with A(row=ref k, col=phys i)
         // = adj(J) as defined by the identities above (J^-1 = A/detJ, and
         // A11..A33 here are the entries of adj(J) in (row, col) = (k, i)).
         D(q, 0, e) = A11; D(q, 1, e) = A21; D(q, 2, e) = A31; // column i = 0
         D(q, 3, e) = A12; D(q, 4, e) = A22; D(q, 5, e) = A32; // column i = 1
         D(q, 6, e) = A13; D(q, 7, e) = A23; D(q, 8, e) = A33; // column i = 2
         D(q, 9, e) = C(q, e) * W(q) / detJ; // alpha
      }
   });
}

// ---------------------------------------------------------------------------
// PA setup: adjugate + alpha per quadrature point.
// ---------------------------------------------------------------------------
void VectorDivDivIntegrator::AssemblePA(const FiniteElementSpace& fes)
{
   Mesh* mesh = fes.GetMesh();
   const FiniteElement& el = *fes.GetTypicalFE();
   const IntegrationRule* ir =
      IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
   dim_ = mesh->Dimension();
   MFEM_VERIFY(dim_ == 2 || dim_ == 3,
               "VectorDivDivIntegrator: dim must be 2 or 3");
   MFEM_VERIFY(mesh->SpaceDimension() == dim_,
               "VectorDivDivIntegrator: sdim must equal dim");
   MFEM_VERIFY(fes.GetVDim() == dim_,
               "VectorDivDivIntegrator: vdim must equal dim");

   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                         ? Device::GetDeviceMemoryType() : pa_mt;
   ne_ = fes.GetNE();
   geom_ = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   // Tensor basis required for PA; use full assembly on non-tensor elements.
   maps_ = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D_ = maps_->ndof;
   quad1D_ = maps_->nqpt;

   vecdivdiv::SetupQuadratureData(*mesh, *ir, *geom_, Q, mt, pa_data_);
}

void VectorDivDivIntegrator::AddMultPA(const Vector& x, Vector& y) const
{
   ApplyPAKernels::Run(dim_, dofs1D_, quad1D_, ne_, maps_->B, maps_->G,
                       pa_data_, x, y, dofs1D_, quad1D_);
}

// ---------------------------------------------------------------------------
// PA diagonal: diag(a, c) = sum_q alpha (Ghat_a . A[:,c])^2, evaluated
// directly per (dof, qpt) pair from the 1D basis tables. Deviation from the
// spec's par.5.5 sweep structure (documented): the sum-factorized QQD/QDD
// sweeps live in an uninstalled MFEM source file; this direct form is
// verifiable against the assembled diagonal and runs ONCE per assembly, so
// setup-time performance is not on the critical path. A sum-factorized
// version is a later optimization if profiling ever demands it.
// ---------------------------------------------------------------------------
void VectorDivDivIntegrator::AssembleDiagonalPA(Vector& diag)
{
   const int D1D = dofs1D_, Q1D = quad1D_, dim = dim_;
   const auto B = Reshape(maps_->B.Read(), Q1D, D1D);
   const auto G = Reshape(maps_->G.Read(), Q1D, D1D);

   if (dim == 2)
   {
      const auto DE = Reshape(pa_data_.Read(), Q1D, Q1D, 5, ne_);
      auto Y = Reshape(diag.ReadWrite(), D1D, D1D, 2, ne_);
      mfem::forall(ne_, [ = ] MFEM_HOST_DEVICE(int e)
      {
         for (int c = 0; c < 2; ++c)
            for (int dy = 0; dy < D1D; ++dy)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  real_t acc = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                     for (int qx = 0; qx < Q1D; ++qx)
                     {
                        // Ghat_a = (G B, B G) at (qx, qy) for dof (dx, dy).
                        const real_t g0 = G(qx, dx) * B(qy, dy);
                        const real_t g1 = B(qx, dx) * G(qy, dy);
                        const real_t proj = g0 * DE(qx, qy, 0 + c * 2, e) +
                                            g1 * DE(qx, qy, 1 + c * 2, e);
                        acc += DE(qx, qy, 4, e) * proj * proj;
                     }
                  Y(dx, dy, c, e) += acc;
               }
      });
   }
   else
   {
      const auto DE = Reshape(pa_data_.Read(), Q1D, Q1D, Q1D, 10, ne_);
      auto Y = Reshape(diag.ReadWrite(), D1D, D1D, D1D, 3, ne_);
      mfem::forall(ne_, [ = ] MFEM_HOST_DEVICE(int e)
      {
         for (int c = 0; c < 3; ++c)
            for (int dz = 0; dz < D1D; ++dz)
               for (int dy = 0; dy < D1D; ++dy)
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     real_t acc = 0.0;
                     for (int qz = 0; qz < Q1D; ++qz)
                        for (int qy = 0; qy < Q1D; ++qy)
                           for (int qx = 0; qx < Q1D; ++qx)
                           {
                              const real_t g0 = G(qx, dx) * B(qy, dy) * B(qz, dz);
                              const real_t g1 = B(qx, dx) * G(qy, dy) * B(qz, dz);
                              const real_t g2 = B(qx, dx) * B(qy, dy) * G(qz, dz);
                              const real_t proj =
                                 g0 * DE(qx, qy, qz, 0 + c * 3, e) +
                                 g1 * DE(qx, qy, qz, 1 + c * 3, e) +
                                 g2 * DE(qx, qy, qz, 2 + c * 3, e);
                              acc += DE(qx, qy, qz, 9, e) * proj * proj;
                           }
                     Y(dx, dy, dz, c, e) += acc;
                  }
      });
   }
}

// ---------------------------------------------------------------------------
// Component-block EA: K^{ij}(a, b) = sum_q alpha (Ghat_a . A[:,i])
// (Ghat_b . A[:,j]), row a (test, component i), column b (trial, component
// j). One thread per (e, a, b) entry, reference gradients rebuilt from the 1D
// tables as in the PA diagonal, so dofs are lexicographic -- the order MFEM's
// EA restriction uses. Direct O(nd^2 nq) per element: fine for the Q1 LOR
// matrices this exists for, not a high-order path (spec par.10).
// MFEM's EA storage is row-major per element: in a column-major
// Reshape(nd, nd, ne), entry (b, a, e) is row a, column b.
// ---------------------------------------------------------------------------
void VectorDivDivComponentIntegrator::AssembleEA(const FiniteElementSpace& fes,
      Vector& emat, const bool add)
{
   Mesh* mesh = fes.GetMesh();
   const FiniteElement& el = *fes.GetTypicalFE();
   const int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "VectorDivDivComponentIntegrator: dim must be 2 or 3");
   MFEM_VERIFY(mesh->SpaceDimension() == dim,
               "VectorDivDivComponentIntegrator: sdim must equal dim");
   MFEM_VERIFY(fes.GetVDim() == 1,
               "VectorDivDivComponentIntegrator: blocks live on a scalar "
               "(vdim == 1) space");
   MFEM_VERIFY(el.GetGeomType() == Geometry::SQUARE ||
               el.GetGeomType() == Geometry::CUBE,
               "VectorDivDivComponentIntegrator: quads/hexes only");
   MFEM_VERIFY(0 <= i_block_ && i_block_ < dim && 0 <= j_block_ && j_block_ < dim,
               "VectorDivDivComponentIntegrator: block index out of range");

   const IntegrationRule* ir =
      IntRule ? IntRule
      : parent_.GetIntRule() ? parent_.GetIntRule()
      : &DiffusionIntegrator::GetRule(el, el);
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors* geom =
      mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   Vector qdata;
   vecdivdiv::SetupQuadratureData(*mesh, *ir, *geom, parent_.Q, mt, qdata);

   const DofToQuad& maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int D1D = maps.ndof, Q1D = maps.nqpt;
   const int ne = fes.GetNE();
   const int nd = el.GetDof();
   const int nq = ir->GetNPoints();
   MFEM_VERIFY(emat.Size() == nd * nd * ne,
               "VectorDivDivComponentIntegrator: emat must be nd*nd*ne");

   const int ib = i_block_, jb = j_block_;
   const auto B = Reshape(maps.B.Read(), Q1D, D1D);
   const auto G = Reshape(maps.G.Read(), Q1D, D1D);
   const auto DE = Reshape(qdata.Read(), nq, dim * dim + 1, ne);
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
               // reference gradients of dofs a, b ([2] unused in 2D)
               real_t ga[3] = {0.0, 0.0, 0.0}, gb[3] = {0.0, 0.0, 0.0};
               if (dim == 2)
               {
                  ga[0] = G(qx, ax) * B(qy, ay);
                  ga[1] = B(qx, ax) * G(qy, ay);
                  gb[0] = G(qx, bx) * B(qy, by);
                  gb[1] = B(qx, bx) * G(qy, by);
               }
               else
               {
                  ga[0] = G(qx, ax) * B(qy, ay) * B(qz, az);
                  ga[1] = B(qx, ax) * G(qy, ay) * B(qz, az);
                  ga[2] = B(qx, ax) * B(qy, ay) * G(qz, az);
                  gb[0] = G(qx, bx) * B(qy, by) * B(qz, bz);
                  gb[1] = B(qx, bx) * G(qy, by) * B(qz, bz);
                  gb[2] = B(qx, bx) * B(qy, by) * G(qz, bz);
               }
               real_t pa = 0.0, pb = 0.0;
               for (int k = 0; k < dim; ++k)
               {
                  pa += ga[k] * DE(q, k + ib * dim, e);
                  pb += gb[k] * DE(q, k + jb * dim, e);
               }
               acc += DE(q, dim * dim, e) * pa * pb;
            }
      if (add) { E(b, a, e) += acc; }
      else { E(b, a, e) = acc; }
   });
}

} // namespace incns
