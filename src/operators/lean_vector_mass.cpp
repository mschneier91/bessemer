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
