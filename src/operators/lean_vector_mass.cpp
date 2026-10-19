#include "operators/lean_vector_mass.hpp"

#include "mfem/general/forall.hpp"

using namespace mfem;

namespace incns
{

void MassWeightDet(ParFiniteElementSpace& vfes, const IntegrationRule& ir,
                   Vector& wdet)
{
   ParMesh& mesh = *vfes.GetParMesh();
   const int ne = mesh.GetNE();
   const int nq = ir.GetNPoints();
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   wdet.UseDevice(true);
   wdet.SetSize(nq * ne);
   if (ne == 0) { return; }
   MFEM_VERIFY(dim == sdim && (dim == 2 || dim == 3),
               "lean_vector_mass: 2D or 3D, space dimension = dimension");
   // MFEM's computation, not its cache (Mesh::GetGeometricFactors does the
   // same EnsureNodes first).
   mesh.EnsureNodes();
   const GeometricFactors geom(*mesh.GetNodes(), ir,
                               GeometricFactors::JACOBIANS,
                               Device::GetDeviceMemoryType());
   const auto W = ir.GetWeights().Read();
   const auto J = Reshape(geom.J.Read(), nq, sdim, dim, ne);
   auto D = Reshape(wdet.Write(), nq, ne);
   if (dim == 2)
   {
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq;
         const int e = i / nq;
         const real_t J11 = J(q, 0, 0, e), J12 = J(q, 1, 0, e);
         const real_t J21 = J(q, 0, 1, e), J22 = J(q, 1, 1, e);
         const real_t detJ = (J11 * J22) - (J21 * J12);
         D(q, e) = W[q] * detJ;
      });
   }
   else
   {
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq;
         const int e = i / nq;
         const real_t J11 = J(q, 0, 0, e), J12 = J(q, 0, 1, e),
                      J13 = J(q, 0, 2, e);
         const real_t J21 = J(q, 1, 0, e), J22 = J(q, 1, 1, e),
                      J23 = J(q, 1, 2, e);
         const real_t J31 = J(q, 2, 0, e), J32 = J(q, 2, 1, e),
                      J33 = J(q, 2, 2, e);
         const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                             J21 * (J12 * J33 - J32 * J13) +
                             J31 * (J12 * J23 - J22 * J13);
         D(q, e) = W[q] * detJ;
      });
   }
}

namespace leanmass
{

// y += B^T D B x per component, a thread per point: a Q1D^3 block per element,
// the three components together, two shared buffers in turn (each stage reads
// one and writes the other).
template <int D1D, int Q1D>
void Apply3DPoints(const int NE, const Array<real_t>& b, const Vector& d,
                   const Vector& x, Vector& y)
{
   static_assert(D1D <= Q1D, "a thread per point covers the dofs");
   static constexpr int VDIM = 3;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   const auto X = Reshape(x.Read(), D1D, D1D, D1D, VDIM, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, VDIM, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [ = ] MFEM_HOST_DEVICE(int e)
   {
      MFEM_SHARED real_t s0[VDIM][Q1D][Q1D][Q1D];
      MFEM_SHARED real_t s1[VDIM][Q1D][Q1D][Q1D];
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               for (int c = 0; c < VDIM; ++c) { s0[c][dz][dy][dx] = X(dx, dy, dz, c, e); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // x: dofs -> points.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const real_t bx = B(qx, dx);
                  for (int c = 0; c < VDIM; ++c) { u[c] += bx * s0[c][dz][dy][dx]; }
               }
               for (int c = 0; c < VDIM; ++c) { s1[c][dz][dy][qx] = u[c]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // y.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int dy = 0; dy < D1D; ++dy)
               {
                  const real_t by = B(qy, dy);
                  for (int c = 0; c < VDIM; ++c) { u[c] += by * s1[c][dz][dy][qx]; }
               }
               for (int c = 0; c < VDIM; ++c) { s0[c][dz][qy][qx] = u[c]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // z, and C W det J at the point.
      MFEM_FOREACH_THREAD(qz, z, Q1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int dz = 0; dz < D1D; ++dz)
               {
                  const real_t bz = B(qz, dz);
                  for (int c = 0; c < VDIM; ++c) { u[c] += bz * s0[c][dz][qy][qx]; }
               }
               const real_t w = D(qx, qy, qz, e);
               for (int c = 0; c < VDIM; ++c) { s1[c][qz][qy][qx] = w * u[c]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Back: z.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(qy, y, Q1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const real_t bz = B(qz, dz);
                  for (int c = 0; c < VDIM; ++c) { u[c] += bz * s1[c][qz][qy][qx]; }
               }
               for (int c = 0; c < VDIM; ++c) { s0[c][dz][qy][qx] = u[c]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // y.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const real_t by = B(qy, dy);
                  for (int c = 0; c < VDIM; ++c) { u[c] += by * s0[c][dz][qy][qx]; }
               }
               for (int c = 0; c < VDIM; ++c) { s1[c][dz][dy][qx] = u[c]; }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // x, added to y.
      MFEM_FOREACH_THREAD(dz, z, D1D)
      {
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(dx, x, D1D)
            {
               real_t u[VDIM] = {0.0, 0.0, 0.0};
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const real_t bx = B(qx, dx);
                  for (int c = 0; c < VDIM; ++c) { u[c] += bx * s1[c][dz][dy][qx]; }
               }
               for (int c = 0; c < VDIM; ++c) { Y(dx, dy, dz, c, e) += u[c]; }
            }
         }
      }
   });
}

} // namespace leanmass

bool VectorMassApply3DPoints(int d1d, int q1d, int ne, const Array<real_t>& b,
                             const Vector& d, const Vector& x, Vector& y)
{
   if (d1d != q1d) { return false; }
   switch (d1d)
   {
      case 2: leanmass::Apply3DPoints<2, 2>(ne, b, d, x, y); return true;
      case 3: leanmass::Apply3DPoints<3, 3>(ne, b, d, x, y); return true;
      case 4: leanmass::Apply3DPoints<4, 4>(ne, b, d, x, y); return true;
      case 5: leanmass::Apply3DPoints<5, 5>(ne, b, d, x, y); return true;
      case 6: leanmass::Apply3DPoints<6, 6>(ne, b, d, x, y); return true;
      case 7: leanmass::Apply3DPoints<7, 7>(ne, b, d, x, y); return true;
      case 8: leanmass::Apply3DPoints<8, 8>(ne, b, d, x, y); return true;
   }
   return false;
}

void LeanVectorMassIntegrator::AddMultPA(const Vector& x, Vector& y) const
{
   // On a device, a thread per point: MFEM's kernel runs a Q1D^2 block per
   // element and the components in turn (1.9 ms per apply at Q6 on 32^3
   // elements, ~10 % of the memory bandwidth). The host keeps MFEM's apply.
   if (dim == 3 && coeff_vdim == 1 && !DeviceCanUseCeed() &&
       Device::Allows(Backend::DEVICE_MASK) &&
       pa_data.Size() == quad1D * quad1D * quad1D * ne &&
       VectorMassApply3DPoints(dofs1D, quad1D, ne, maps->B, pa_data, x, y))
   {
      return;
   }
   mfem::VectorMassIntegrator::AddMultPA(x, y);
}

void LeanVectorMassIntegrator::AssemblePA(const FiniteElementSpace& fes)
{
   Mesh* mesh = fes.GetMesh();
   const int ne_fes = mesh->GetNE();
   if (!IntRule || VQ || MQ || DeviceCanUseCeed() || ne_fes == 0 ||
       wdet_.Size() != IntRule->GetNPoints() * ne_fes)
   {
      mfem::VectorMassIntegrator::AssemblePA(fes);
      return;
   }
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                         ? Device::GetDeviceMemoryType() : pa_mt;
   const FiniteElement& el = *fes.GetTypicalFE();
   ne = ne_fes;
   dim = mesh->Dimension();
   geom = nullptr; // no Jacobians: the kept W det J instead
   maps = &el.GetDofToQuad(*IntRule, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   QuadratureSpace qs(*mesh, *IntRule);
   CoefficientVector coeff(qs);
   if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }
   coeff_vdim = coeff.GetVDim();
   MFEM_VERIFY(coeff_vdim == 1 && coeff.Size() == wdet_.Size(),
               "lean_vector_mass: a scalar coefficient at every point");
   // MFEM's D = C * w_det, point by point.
   pa_data.SetSize(wdet_.Size(), mt);
   pa_data.UseDevice(true);
   pa_data = coeff;
   pa_data *= wdet_;
}

} // namespace incns
