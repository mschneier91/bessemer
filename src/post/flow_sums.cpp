#include "post/flow_sums.hpp"

#include <mfem/general/forall.hpp>

// The kernels below must be compiled by nvcc in a CUDA build, or they run on
// the host while the data is on the device. See src/CMakeLists.txt's nvcc
// list: this line turns a lost entry into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "flow_sums.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Device kernels live in free functions in a NAMED namespace: nvcc rejects
// extended lambdas in functions with internal linkage or in private members.
namespace flowsums
{

// out(0..2, e): the element's int |u|^2, int |grad u|^2, int (div u)^2.
void Sums2D(int ne, int D1D, int Q1D, const Array<real_t>& b,
            const Array<real_t>& g, const Array<real_t>& w, const Vector& jac,
            const Vector& xe, Vector& out)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D);
   const auto J = Reshape(jac.Read(), Q1D, Q1D, 2, 2, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, 2, ne);
   auto Y = Reshape(out.Write(), 3, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int qy = 0; qy < Q1D; ++qy)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            real_t u[2] = {0.0, 0.0};
            real_t gr[2][2] = {{0.0, 0.0}, {0.0, 0.0}}; // d u_c / d xi_j
            for (int dy = 0; dy < D1D; ++dy)
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t bb = B(qx, dx) * B(qy, dy);
                  const real_t g0 = G(qx, dx) * B(qy, dy);
                  const real_t g1 = B(qx, dx) * G(qy, dy);
                  for (int c = 0; c < 2; ++c)
                  {
                     const real_t v = X(dx, dy, c, e);
                     u[c] += bb * v;
                     gr[c][0] += g0 * v;
                     gr[c][1] += g1 * v;
                  }
               }
            const real_t j00 = J(qx, qy, 0, 0, e), j01 = J(qx, qy, 0, 1, e);
            const real_t j10 = J(qx, qy, 1, 0, e), j11 = J(qx, qy, 1, 1, e);
            const real_t det = j00 * j11 - j01 * j10;
            // inv(J)(j, i) = d xi_j / d x_i.
            const real_t i00 = j11 / det, i01 = -j01 / det;
            const real_t i10 = -j10 / det, i11 = j00 / det;
            real_t grad_sq = 0.0, div = 0.0;
            for (int c = 0; c < 2; ++c)
            {
               const real_t dx0 = gr[c][0] * i00 + gr[c][1] * i10; // d u_c / dx
               const real_t dx1 = gr[c][0] * i01 + gr[c][1] * i11; // d u_c / dy
               grad_sq += dx0 * dx0 + dx1 * dx1;
               div += (c == 0) ? dx0 : dx1;
            }
            const real_t wq = W(qx, qy) * (det < 0.0 ? -det : det);
            s0 += wq * (u[0] * u[0] + u[1] * u[1]);
            s1 += wq * grad_sq;
            s2 += wq * div * div;
         }
      Y(0, e) = s0;
      Y(1, e) = s1;
      Y(2, e) = s2;
   });
}

void Sums3D(int ne, int D1D, int Q1D, const Array<real_t>& b,
            const Array<real_t>& g, const Array<real_t>& w, const Vector& jac,
            const Vector& xe, Vector& out)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(jac.Read(), Q1D, Q1D, Q1D, 3, 3, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, D1D, 3, ne);
   auto Y = Reshape(out.Write(), 3, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      real_t s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               real_t u[3] = {0.0, 0.0, 0.0};
               real_t gr[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
               for (int dz = 0; dz < D1D; ++dz)
                  for (int dy = 0; dy < D1D; ++dy)
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        const real_t bb = B(qx, dx) * B(qy, dy) * B(qz, dz);
                        const real_t g0 = G(qx, dx) * B(qy, dy) * B(qz, dz);
                        const real_t g1 = B(qx, dx) * G(qy, dy) * B(qz, dz);
                        const real_t g2 = B(qx, dx) * B(qy, dy) * G(qz, dz);
                        for (int c = 0; c < 3; ++c)
                        {
                           const real_t v = X(dx, dy, dz, c, e);
                           u[c] += bb * v;
                           gr[c][0] += g0 * v;
                           gr[c][1] += g1 * v;
                           gr[c][2] += g2 * v;
                        }
                     }
               const real_t j00 = J(qx, qy, qz, 0, 0, e), j01 = J(qx, qy, qz, 0, 1, e);
               const real_t j02 = J(qx, qy, qz, 0, 2, e), j10 = J(qx, qy, qz, 1, 0, e);
               const real_t j11 = J(qx, qy, qz, 1, 1, e), j12 = J(qx, qy, qz, 1, 2, e);
               const real_t j20 = J(qx, qy, qz, 2, 0, e), j21 = J(qx, qy, qz, 2, 1, e);
               const real_t j22 = J(qx, qy, qz, 2, 2, e);
               const real_t det = j00 * (j11 * j22 - j12 * j21) -
                                  j01 * (j10 * j22 - j12 * j20) +
                                  j02 * (j10 * j21 - j11 * j20);
               // inv(J)(j, i) = d xi_j / d x_i.
               real_t iv[3][3];
               iv[0][0] = (j11 * j22 - j12 * j21) / det;
               iv[0][1] = (j02 * j21 - j01 * j22) / det;
               iv[0][2] = (j01 * j12 - j02 * j11) / det;
               iv[1][0] = (j12 * j20 - j10 * j22) / det;
               iv[1][1] = (j00 * j22 - j02 * j20) / det;
               iv[1][2] = (j02 * j10 - j00 * j12) / det;
               iv[2][0] = (j10 * j21 - j11 * j20) / det;
               iv[2][1] = (j01 * j20 - j00 * j21) / det;
               iv[2][2] = (j00 * j11 - j01 * j10) / det;
               real_t grad_sq = 0.0, div = 0.0;
               for (int c = 0; c < 3; ++c)
               {
                  for (int i = 0; i < 3; ++i)
                  {
                     const real_t d = gr[c][0] * iv[0][i] + gr[c][1] * iv[1][i] +
                                      gr[c][2] * iv[2][i]; // d u_c / d x_i
                     grad_sq += d * d;
                     if (i == c) { div += d; }
                  }
               }
               const real_t wq = W(qx, qy, qz) * (det < 0.0 ? -det : det);
               s0 += wq * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
               s1 += wq * grad_sq;
               s2 += wq * div * div;
            }
      Y(0, e) = s0;
      Y(1, e) = s1;
      Y(2, e) = s2;
   });
}

// Sums3D with one thread per quadrature point (Q1D^3 per element) instead of
// one per element: the same arithmetic at each point, from shared memory,
// and the element's sums added in the same order (qx fastest) by one thread.
// Sums3D's one thread per element left the GPU mostly idle: 4.0 ms per call
// at TGV 32^3 Q3, 9 % of the kernel time with diagnostics every step (Nsight
// Systems, 2026-10-15). Compile-time sizes: the solver's (k+1, k+2).
template <int D1D, int Q1D>
void Sums3DPoints(int ne, const Array<real_t>& b, const Array<real_t>& g,
                  const Array<real_t>& w, const Vector& jac, const Vector& xe,
                  Vector& out)
{
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(jac.Read(), Q1D, Q1D, Q1D, 3, 3, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, D1D, 3, ne);
   auto Y = Reshape(out.Write(), 3, ne);
   mfem::forall_3D(ne, Q1D, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      constexpr int NQ = Q1D * Q1D * Q1D;
      MFEM_SHARED real_t sB[Q1D][D1D], sG[Q1D][D1D];
      MFEM_SHARED real_t sX[3][D1D][D1D][D1D];
      MFEM_SHARED real_t sR[3][NQ];
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(d, y, D1D)
         {
            MFEM_FOREACH_THREAD(q, x, Q1D)
            {
               sB[q][d] = B(q, d);
               sG[q][d] = G(q, d);
            }
         }
      }
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               for (int c = 0; c < 3; ++c) { sX[c][dz][dy][dx] = X(dx, dy, dz, c, e); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[3] = {0.0, 0.0, 0.0};
               real_t gr[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
               for (int dz = 0; dz < D1D; ++dz)
                  for (int dy = 0; dy < D1D; ++dy)
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        const real_t bb = sB[qx][dx] * sB[qy][dy] * sB[qz][dz];
                        const real_t g0 = sG[qx][dx] * sB[qy][dy] * sB[qz][dz];
                        const real_t g1 = sB[qx][dx] * sG[qy][dy] * sB[qz][dz];
                        const real_t g2 = sB[qx][dx] * sB[qy][dy] * sG[qz][dz];
                        for (int c = 0; c < 3; ++c)
                        {
                           const real_t v = sX[c][dz][dy][dx];
                           u[c] += bb * v;
                           gr[c][0] += g0 * v;
                           gr[c][1] += g1 * v;
                           gr[c][2] += g2 * v;
                        }
                     }
               const real_t j00 = J(qx, qy, qz, 0, 0, e), j01 = J(qx, qy, qz, 0, 1, e);
               const real_t j02 = J(qx, qy, qz, 0, 2, e), j10 = J(qx, qy, qz, 1, 0, e);
               const real_t j11 = J(qx, qy, qz, 1, 1, e), j12 = J(qx, qy, qz, 1, 2, e);
               const real_t j20 = J(qx, qy, qz, 2, 0, e), j21 = J(qx, qy, qz, 2, 1, e);
               const real_t j22 = J(qx, qy, qz, 2, 2, e);
               const real_t det = j00 * (j11 * j22 - j12 * j21) -
                                  j01 * (j10 * j22 - j12 * j20) +
                                  j02 * (j10 * j21 - j11 * j20);
               real_t iv[3][3];
               iv[0][0] = (j11 * j22 - j12 * j21) / det;
               iv[0][1] = (j02 * j21 - j01 * j22) / det;
               iv[0][2] = (j01 * j12 - j02 * j11) / det;
               iv[1][0] = (j12 * j20 - j10 * j22) / det;
               iv[1][1] = (j00 * j22 - j02 * j20) / det;
               iv[1][2] = (j02 * j10 - j00 * j12) / det;
               iv[2][0] = (j10 * j21 - j11 * j20) / det;
               iv[2][1] = (j01 * j20 - j00 * j21) / det;
               iv[2][2] = (j00 * j11 - j01 * j10) / det;
               real_t grad_sq = 0.0, div = 0.0;
               for (int c = 0; c < 3; ++c)
               {
                  for (int i = 0; i < 3; ++i)
                  {
                     const real_t d = gr[c][0] * iv[0][i] + gr[c][1] * iv[1][i] +
                                      gr[c][2] * iv[2][i];
                     grad_sq += d * d;
                     if (i == c) { div += d; }
                  }
               }
               const real_t wq = W(qx, qy, qz) * (det < 0.0 ? -det : det);
               const int q = qx + Q1D * (qy + Q1D * qz);
               sR[0][q] = wq * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
               sR[1][q] = wq * grad_sq;
               sR[2][q] = wq * div * div;
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) == 0 && MFEM_THREAD_ID(y) == 0 &&
          MFEM_THREAD_ID(z) == 0)
      {
         real_t s0 = 0.0, s1 = 0.0, s2 = 0.0;
         for (int q = 0; q < NQ; ++q)
         {
            s0 += sR[0][q];
            s1 += sR[1][q];
            s2 += sR[2][q];
         }
         Y(0, e) = s0;
         Y(1, e) = s1;
         Y(2, e) = s2;
      }
   });
}

// The three columns of out(3, ne), summed on the device.
void Reduce(int ne, const Vector& out, double sums[3])
{
   for (int k = 0; k < 3; ++k)
   {
      Vector col(ne);
      col.UseDevice(true);
      const auto Y = Reshape(out.Read(), 3, ne);
      auto C = col.Write();
      mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e) { C[e] = Y(k, e); });
      sums[k] = col.Sum();
   }
}

} // namespace flowsums

void LocalFlowSums(const ParGridFunction& u, const IntegrationRule& ir,
                   double sums[3])
{
   sums[0] = sums[1] = sums[2] = 0.0;
   ParFiniteElementSpace& fes = *u.ParFESpace();
   ParMesh& mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   const int ne = fes.GetNE();
   MFEM_VERIFY(dim == 2 || dim == 3, "flow_sums: dim must be 2 or 3");
   MFEM_VERIFY(mesh.SpaceDimension() == dim && fes.GetVDim() == dim,
               "flow_sums: needs sdim == dim == vdim");
   if (ne == 0) { return; }

   const FiniteElement& el = *fes.GetTypicalFE();
   const DofToQuad& maps = el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const GeometricFactors* geom =
      mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   const Operator* restrict_op =
      fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restrict_op, "flow_sums: no element restriction");
   Vector xe(restrict_op->Height());
   xe.UseDevice(true);
   restrict_op->Mult(u, xe);
   Vector out(3 * ne);
   out.UseDevice(true);
   if (dim == 2)
   {
      flowsums::Sums2D(ne, maps.ndof, maps.nqpt, maps.B, maps.G, ir.GetWeights(),
                       geom->J, xe, out);
   }
   else
   {
      const int d1 = maps.ndof, q1 = maps.nqpt;
      const Array<real_t>& w = ir.GetWeights();
      if (d1 == 3 && q1 == 4)
      {
         flowsums::Sums3DPoints<3, 4>(ne, maps.B, maps.G, w, geom->J, xe, out);
      }
      else if (d1 == 4 && q1 == 5)
      {
         flowsums::Sums3DPoints<4, 5>(ne, maps.B, maps.G, w, geom->J, xe, out);
      }
      else if (d1 == 5 && q1 == 6)
      {
         flowsums::Sums3DPoints<5, 6>(ne, maps.B, maps.G, w, geom->J, xe, out);
      }
      else
      {
         flowsums::Sums3D(ne, d1, q1, maps.B, maps.G, w, geom->J, xe, out);
      }
   }
   flowsums::Reduce(ne, out, sums);
}

} // namespace incns
