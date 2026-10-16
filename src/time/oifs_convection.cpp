#include "time/oifs_convection.hpp"

#include "operators/component_kernels.hpp"

#include "mfem/fem/kernels.hpp"
#include "mfem/general/forall.hpp"

// The kernels below must be compiled by nvcc in a CUDA build, or they run on
// the host while the data is on the device. See src/CMakeLists.txt's nvcc
// list: this line turns a lost entry into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "oifs_convection.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Device kernels in a NAMED namespace: nvcc rejects extended lambdas whose
// enclosing function has internal linkage.
namespace oifsconv
{

// op(q, i, e) = W(q) (adj(J) u)_i at the points of element e: MFEM's
// PAConvectionSetup3D with alpha = 1, written into one wind's slice of op_.
void SetupWind(const int nq, const int ne, const Array<real_t>& w,
               const Vector& j, const Vector& vel, real_t* op)
{
   const auto W = Reshape(w.Read(), nq);
   const auto J = Reshape(j.Read(), nq, 3, 3, ne);
   const auto V = Reshape(vel.Read(), 3, nq, ne);
   auto y = Reshape(op, nq, 3, ne);
   mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int q_global)
   {
      const int e = q_global / nq;
      const int q = q_global % nq;
      const real_t J11 = J(q, 0, 0, e);
      const real_t J12 = J(q, 0, 1, e);
      const real_t J13 = J(q, 0, 2, e);
      const real_t J21 = J(q, 1, 0, e);
      const real_t J22 = J(q, 1, 1, e);
      const real_t J23 = J(q, 1, 2, e);
      const real_t J31 = J(q, 2, 0, e);
      const real_t J32 = J(q, 2, 1, e);
      const real_t J33 = J(q, 2, 2, e);
      const real_t wq = 1.0 * W(q);
      const real_t wx = wq * V(0, q, e);
      const real_t wy = wq * V(1, q, e);
      const real_t wz = wq * V(2, q, e);
      const real_t A11 = (J22 * J33) - (J23 * J32);
      const real_t A12 = (J32 * J13) - (J12 * J33);
      const real_t A13 = (J12 * J23) - (J22 * J13);
      const real_t A21 = (J31 * J23) - (J21 * J33);
      const real_t A22 = (J11 * J33) - (J13 * J31);
      const real_t A23 = (J21 * J13) - (J11 * J23);
      const real_t A31 = (J21 * J32) - (J31 * J22);
      const real_t A32 = (J31 * J12) - (J11 * J32);
      const real_t A33 = (J11 * J22) - (J12 * J21);
      y(q, 0, e) = wx * A11 + wy * A12 + wz * A13;
      y(q, 1, e) = wx * A21 + wy * A22 + wz * A23;
      y(q, 2, e) = wx * A31 + wy * A32 + wz * A33;
   });
}

// y(:, c, e) = C x(:, c, e) for c = 0, 1, 2, with the quadrature data
// sum_{l < nw} w_l op(:, :, e, l) formed once per point: MFEM's
// SmemPAConvectionApply3D (one thread per point, shared-memory sum
// factorization) with that data and a loop over the components.
template <int D1D, int Q1D>
void Apply3D(const int NE, const int nw, const real_t w0, const real_t w1,
             const real_t w2, const Array<real_t>& b, const Array<real_t>& g,
             const Array<real_t>& bt, const Vector& op_, const Vector& x_,
             Vector& y_)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Bt = Reshape(bt.Read(), D1D, Q1D);
   const auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, 3, NE, 3);
   const auto x = Reshape(x_.Read(), D1D, D1D, D1D, 3, NE);
   auto y = Reshape(y_.Write(), D1D, D1D, D1D, 3, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int DQ = (Q1D > D1D) ? Q1D : D1D;
      MFEM_SHARED real_t sD[3][Q1D][Q1D][Q1D];
      MFEM_SHARED real_t sm0[DQ * DQ * DQ];
      MFEM_SHARED real_t sm1[DQ * DQ * DQ];
      MFEM_SHARED real_t sm2[DQ * DQ * DQ];
      MFEM_SHARED real_t sm3[DQ * DQ * DQ];
      MFEM_SHARED real_t sm4[DQ * DQ * DQ];
      MFEM_SHARED real_t sm5[DQ * DQ * DQ];
      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               for (int i = 0; i < 3; ++i)
               {
                  real_t v = w0 * op(qx, qy, qz, i, e, 0);
                  if (nw > 1) { v += w1 * op(qx, qy, qz, i, e, 1); }
                  if (nw > 2) { v += w2 * op(qx, qy, qz, i, e, 2); }
                  sD[i][qz][qy][qx] = v;
               }
            }
         }
      }
      for (int c = 0; c < 3; ++c)
      {
         real_t (*u)[D1D][D1D] = (real_t (*)[D1D][D1D]) sm0;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  u[dz][dy][dx] = x(dx, dy, dz, c, e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*Bu)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm1;
         real_t (*Gu)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm2;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  real_t Bu_ = 0.0;
                  real_t Gu_ = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const real_t bx = B(qx, dx);
                     const real_t gx = G(qx, dx);
                     const real_t xv = u[dz][dy][dx];
                     Bu_ += bx * xv;
                     Gu_ += gx * xv;
                  }
                  Bu[dz][dy][qx] = Bu_;
                  Gu[dz][dy][qx] = Gu_;
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*BBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm3;
         real_t (*GBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm4;
         real_t (*BGu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm5;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               MFEM_FOREACH_THREAD(qy, y, Q1D)
               {
                  real_t BBu_ = 0.0;
                  real_t GBu_ = 0.0;
                  real_t BGu_ = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     const real_t bx = B(qy, dy);
                     const real_t gx = G(qy, dy);
                     BBu_ += bx * Bu[dz][dy][qx];
                     GBu_ += gx * Bu[dz][dy][qx];
                     BGu_ += bx * Gu[dz][dy][qx];
                  }
                  BBu[dz][qy][qx] = BBu_;
                  GBu[dz][qy][qx] = GBu_;
                  BGu[dz][qy][qx] = BGu_;
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*GBBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm0;
         real_t (*BGBu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm1;
         real_t (*BBGu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm2;
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qz, z, Q1D)
               {
                  real_t GBBu_ = 0.0;
                  real_t BGBu_ = 0.0;
                  real_t BBGu_ = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     const real_t bx = B(qz, dz);
                     const real_t gx = G(qz, dz);
                     GBBu_ += gx * BBu[dz][qy][qx];
                     BGBu_ += bx * GBu[dz][qy][qx];
                     BBGu_ += bx * BGu[dz][qy][qx];
                  }
                  GBBu[qz][qy][qx] = GBBu_;
                  BGBu[qz][qy][qx] = BGBu_;
                  BBGu[qz][qy][qx] = BBGu_;
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*DGu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm3;
         MFEM_FOREACH_THREAD(qz, z, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(qx, x, Q1D)
               {
                  const real_t O1 = sD[0][qz][qy][qx];
                  const real_t O2 = sD[1][qz][qy][qx];
                  const real_t O3 = sD[2][qz][qy][qx];
                  const real_t gradX = BBGu[qz][qy][qx];
                  const real_t gradY = BGBu[qz][qy][qx];
                  const real_t gradZ = GBBu[qz][qy][qx];
                  DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*BDGu)[Q1D][Q1D] = (real_t (*)[Q1D][Q1D]) sm4;
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            MFEM_FOREACH_THREAD(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD(dz, z, D1D)
               {
                  real_t BDGu_ = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     BDGu_ += Bt(dz, qz) * DGu[qz][qy][qx];
                  }
                  BDGu[dz][qy][qx] = BDGu_;
               }
            }
         }
         MFEM_SYNC_THREAD;
         real_t (*BBDGu)[D1D][Q1D] = (real_t (*)[D1D][Q1D]) sm5;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               MFEM_FOREACH_THREAD(dy, y, D1D)
               {
                  real_t BBDGu_ = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     BBDGu_ += Bt(dy, qy) * BDGu[dz][qy][qx];
                  }
                  BBDGu[dz][dy][qx] = BBDGu_;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz, z, D1D)
         {
            MFEM_FOREACH_THREAD(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD(dx, x, D1D)
               {
                  real_t BBBDGu = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     BBBDGu += Bt(dx, qx) * BBDGu[dz][dy][qx];
                  }
                  y(dx, dy, dz, c, e) = BBBDGu;
               }
            }
         }
         MFEM_SYNC_THREAD; // the next component reuses the shared arrays
      }
   });
}

// Apply3D at Q1D 10 and 11 (velocity orders 6 and 7), where a thread per
// point no longer fits: Q1D^3 threads exceed CUDA's 1024 per block, its
// shared arrays the 48 KB of static shared memory. A thread per (x, y)
// column, looping over z (the layout of MFEM's kernels::internal): the
// combined wind stays in shared memory (3 Q1D^3 values, 32 KB at Q1D 11),
// each component's gradient and the transposed interpolation go through
// registers one component at a time (operators/component_kernels.hpp). The
// same terms as Apply3D, contracted in the same order.
template <int D1D, int Q1D>
void Apply3DColumns(const int NE, const int nw, const real_t w0,
                    const real_t w1, const real_t w2, const Array<real_t>& b,
                    const Array<real_t>& g, const Array<real_t>& bt,
                    const Vector& op_, const Vector& x_, Vector& y_)
{
   namespace ki = kernels::internal;
   const auto B = b.Read(), G = g.Read(), Bt = bt.Read();
   const auto op = Reshape(op_.Read(), Q1D, Q1D, Q1D, 3, NE, 3);
   const auto x = Reshape(x_.Read(), D1D, D1D, D1D, 3, NE);
   auto y = Reshape(y_.Write(), D1D, D1D, D1D, 3, NE);
   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t sD[3][MQ1][MQ1][MQ1];
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1], sBt[MQ1][MQ1];
      ki::vd_regs3d_t<1, 3, MQ1> g0, g1;
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::LoadMatrix(D1D, Q1D, B, sB);
      ki::LoadMatrix(D1D, Q1D, G, sG);
      ki::LoadMatrix<MQ1, true>(D1D, Q1D, Bt, sBt);
      for (int qz = 0; qz < Q1D; ++qz)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
            {
               for (int i = 0; i < 3; ++i)
               {
                  real_t v = w0 * op(qx, qy, qz, i, e, 0);
                  if (nw > 1) { v += w1 * op(qx, qy, qz, i, e, 1); }
                  if (nw > 2) { v += w2 * op(qx, qy, qz, i, e, 2); }
                  sD[i][qz][qy][qx] = v;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_UNROLL(3)
      for (int c = 0; c < 3; ++c)
      {
         kernels3d::LoadComponent(e, D1D, c, x, g0);
         kernels3d::GradComponent<false>(D1D, Q1D, smem, sB, sG, g0, g1);
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  r0[0][qz][qy][qx] =
                     (sD[0][qz][qy][qx] * g1[0][0][qz][qy][qx]) +
                     (sD[1][qz][qy][qx] * g1[0][1][qz][qy][qx]) +
                     (sD[2][qz][qy][qx] * g1[0][2][qz][qy][qx]);
               }
            }
         }
         MFEM_SYNC_THREAD;
         ki::EvalTranspose3d(D1D, Q1D, smem, sBt, r0, r1);
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  y(dx, dy, dz, c, e) = r1[0][dz][dy][dx];
               }
            }
         }
         MFEM_SYNC_THREAD; // the next component reuses the shared arrays
      }
   });
}

// J(q, i, k, e) = d x_i / d xi_k at the points (GeometricFactors' layout,
// nq x 3 x 3 x ne) from the element nodes xe (nd1^3 x 3 x ne), a thread per
// (x, y) column: GeometricFactors goes through the quadrature interpolator,
// which does not serve these sizes (OifsConvection::Columns()).
template <int Q1D>
void Jacobians3DColumns(const int NE, const int nd1, const Array<real_t>& b,
                        const Array<real_t>& g, const Vector& xe_, Vector& j_)
{
   namespace ki = kernels::internal;
   const auto B = b.Read(), G = g.Read();
   const auto X = Reshape(xe_.Read(), nd1, nd1, nd1, 3, NE);
   auto J = Reshape(j_.Write(), Q1D, Q1D, Q1D, 3, 3, NE);
   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1], sG[MQ1][MQ1];
      ki::vd_regs3d_t<1, 3, MQ1> g0, g1;
      ki::LoadMatrix(nd1, Q1D, B, sB);
      ki::LoadMatrix(nd1, Q1D, G, sG);
      for (int i = 0; i < 3; ++i)
      {
         kernels3d::LoadComponent(e, nd1, i, X, g0);
         kernels3d::GradComponent<false>(nd1, Q1D, smem, sB, sG, g0, g1);
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  J(qx, qy, qz, i, 0, e) = g1[0][0][qz][qy][qx];
                  J(qx, qy, qz, i, 1, e) = g1[0][1][qz][qy][qx];
                  J(qx, qy, qz, i, 2, e) = g1[0][2][qz][qy][qx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// The velocity's values at the points, byVDIM (3 x nq x ne), from its element
// dofs (D1D^3 x 3 x ne), a thread per (x, y) column: the quadrature
// interpolator's job where it cannot (OifsConvection::Columns()).
template <int D1D, int Q1D>
void Values3DColumns(const int NE, const Array<real_t>& b, const Vector& ue_,
                     Vector& uq_)
{
   namespace ki = kernels::internal;
   const auto B = b.Read();
   const auto X = Reshape(ue_.Read(), D1D, D1D, D1D, 3, NE);
   auto Y = Reshape(uq_.Write(), 3, Q1D, Q1D, Q1D, NE);
   mfem::forall_2D<Q1D* Q1D>(NE, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int MQ1 = Q1D;
      MFEM_SHARED real_t smem[MQ1][MQ1];
      MFEM_SHARED real_t sB[MQ1][MQ1];
      ki::v_regs3d_t<1, MQ1> r0, r1;
      ki::LoadMatrix(D1D, Q1D, B, sB);
      for (int c = 0; c < 3; ++c)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            MFEM_FOREACH_THREAD_DIRECT(dy, y, D1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(dx, x, D1D)
               {
                  r0[0][dz][dy][dx] = X(dx, dy, dz, c, e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         ki::Eval3d(D1D, Q1D, smem, sB, r0, r1);
         for (int qz = 0; qz < Q1D; ++qz)
         {
            MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
            {
               MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
               {
                  Y(c, qx, qy, qz, e) = r1[0][qz][qy][qx];
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace oifsconv

bool OifsConvection::Supported(const ParFiniteElementSpace& vfes,
                               const IntegrationRule& ir)
{
   const ParMesh* mesh = vfes.GetParMesh();
   if (mesh->Dimension() != 3 || vfes.GetVDim() != 3 ||
       vfes.GetOrdering() != Ordering::byNODES)
   {
      return false;
   }
   const FiniteElement* fe = vfes.GetTypicalFE();
   if (fe == nullptr || dynamic_cast<const TensorBasisElement*>(fe) == nullptr)
   {
      return false;
   }
   const DofToQuad& m = fe->GetDofToQuad(ir, DofToQuad::TENSOR);
   const int d1 = m.ndof, q1 = m.nqpt;
   // Velocity orders 2 to 7 at the 3k rule.
   return (d1 == 3 && q1 == 4) || (d1 == 4 && q1 == 5) || (d1 == 5 && q1 == 7) ||
          (d1 == 6 && q1 == 8) || (d1 == 7 && q1 == 10) || (d1 == 8 && q1 == 11);
}

OifsConvection::OifsConvection(ParFiniteElementSpace& vfes,
                               const IntegrationRule& ir)
   : vfes_(vfes), ir_(ir)
{
   MFEM_VERIFY(Supported(vfes, ir), "oifs_convection: no kernel for this "
               "space and rule (Supported())");
   ParMesh& mesh = *vfes.GetParMesh();
   ne_ = vfes.GetNE();
   maps_ = &vfes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   d1_ = maps_->ndof;
   q1_ = maps_->nqpt;
   restr_ = vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   if (Columns())
   {
      MFEM_VERIFY(q1_ == 10 || q1_ == 11,
                  "oifs_convection: no column kernels for Q1D " << q1_);
      mesh.EnsureNodes();
      const GridFunction& nodes = *mesh.GetNodes();
      const FiniteElementSpace& nfes = *nodes.FESpace();
      const DofToQuad& nm =
         nfes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
      MFEM_VERIFY(nm.ndof <= q1_, "oifs_convection: mesh order too high");
      const Operator* nr =
         nfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      Vector xe(nr->Height());
      xe.UseDevice(true);
      nr->Mult(nodes, xe);
      jac_.SetSize(9 * ir.GetNPoints() * ne_);
      jac_.UseDevice(true);
      if (ne_ > 0 && q1_ == 10)
      {
         oifsconv::Jacobians3DColumns<10>(ne_, nm.ndof, nm.B, nm.G, xe, jac_);
      }
      else if (ne_ > 0)
      {
         oifsconv::Jacobians3DColumns<11>(ne_, nm.ndof, nm.B, nm.G, xe, jac_);
      }
      j_ = &jac_;
   }
   else
   {
      geom_ = mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
      j_ = &geom_->J;
   }
   prolong_ = vfes.GetProlongationMatrix();
   const int nq = ir.GetNPoints();
   for (Vector* v : {&op_, &ue_, &uq_, &xl_, &xe_, &ye_, &yl_})
   {
      v->UseDevice(true);
   }
   op_.SetSize(3 * 3 * nq * ne_);
   ue_.SetSize(restr_->Height());
   uq_.SetSize(3 * nq * ne_);
   xl_.SetSize(vfes.GetVSize());
   xe_.SetSize(restr_->Height());
   ye_.SetSize(restr_->Height());
   yl_.SetSize(vfes.GetVSize());
}

void OifsConvection::SetWind(int l, const ParGridFunction& u)
{
   MFEM_VERIFY(l >= 0 && l < 3, "oifs_convection: wind index out of range");
   MFEM_VERIFY(u.ParFESpace() == &vfes_, "oifs_convection: the wind must live "
               "on the bound velocity space");
   if (ne_ == 0) { set_ |= (1 << l); return; }
   restr_->Mult(u, ue_);
   if (Columns() && q1_ == 10)
   {
      oifsconv::Values3DColumns<7, 10>(ne_, maps_->B, ue_, uq_);
   }
   else if (Columns())
   {
      oifsconv::Values3DColumns<8, 11>(ne_, maps_->B, ue_, uq_);
   }
   else
   {
      const QuadratureInterpolator* qi = vfes_.GetQuadratureInterpolator(ir_);
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      qi->Values(ue_, uq_);
   }
   const int nq = ir_.GetNPoints();
   real_t* op = op_.ReadWrite() + static_cast<std::size_t>(l) * 3 * nq * ne_;
   oifsconv::SetupWind(nq, ne_, ir_.GetWeights(), *j_, uq_, op);
   set_ |= (1 << l);
}

void OifsConvection::Mult(const std::vector<double>& w, const Vector& x,
                          Vector& y)
{
   const int nw = static_cast<int>(w.size());
   MFEM_VERIFY(nw >= 1 && nw <= 3, "oifs_convection: 1 to 3 weights");
   for (int l = 0; l < nw; ++l)
   {
      MFEM_VERIFY(set_ & (1 << l), "oifs_convection: wind " << l << " not set");
   }
   const real_t w0 = w[0];
   const real_t w1 = (nw > 1) ? w[1] : 0.0;
   const real_t w2 = (nw > 2) ? w[2] : 0.0;
   const Vector* xl = &x;
   if (prolong_)
   {
      prolong_->Mult(x, xl_);
      xl = &xl_;
   }
   if (ne_ > 0)
   {
      restr_->Mult(*xl, xe_);
      const Array<real_t>& B = maps_->B;
      const Array<real_t>& G = maps_->G;
      const Array<real_t>& Bt = maps_->Bt;
      if (d1_ == 3 && q1_ == 4)
      {
         oifsconv::Apply3D<3, 4>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_, ye_);
      }
      else if (d1_ == 4 && q1_ == 5)
      {
         oifsconv::Apply3D<4, 5>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_, ye_);
      }
      else if (d1_ == 5 && q1_ == 7)
      {
         oifsconv::Apply3D<5, 7>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_, ye_);
      }
      else if (d1_ == 6 && q1_ == 8)
      {
         // 512 threads and 37 KB of shared memory: a thread per point fits.
         oifsconv::Apply3D<6, 8>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_, ye_);
      }
      else if (d1_ == 7 && q1_ == 10)
      {
         oifsconv::Apply3DColumns<7, 10>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_,
                                         ye_);
      }
      else
      {
         oifsconv::Apply3DColumns<8, 11>(ne_, nw, w0, w1, w2, B, G, Bt, op_, xe_,
                                         ye_);
      }
   }
   else { ye_ = 0.0; }
   if (prolong_)
   {
      restr_->MultTranspose(ye_, yl_);
      prolong_->MultTranspose(yl_, y);
   }
   else
   {
      restr_->MultTranspose(ye_, y);
   }
}

} // namespace incns
