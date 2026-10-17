#include "operators/collocated_mass.hpp"

#include "mfem/general/forall.hpp"

using namespace mfem;

namespace incns
{

namespace
{

/// y += c (d .* x). A free function: nvcc rejects an extended device lambda
/// inside a private member function.
void AddScaledProduct(real_t c, const Vector& d, const Vector& x, Vector& y)
{
   if (c == 0.0) { return; }
   const int n = y.Size();
   const auto D = d.Read();
   const auto X = x.Read();
   auto Y = y.ReadWrite();
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int i)
   {
      Y[i] += c * D[i] * X[i];
   });
}

} // namespace

void AssembleCollocatedMassDiagonal(ParFiniteElementSpace& vfes,
                                    const IntegrationRule& ir, Vector& diag)
{
   ParMesh& mesh = *vfes.GetParMesh();
   const int ne = mesh.GetNE();
   const int nq = ir.GetNPoints();
   const int vdim = vfes.GetVDim();
   const Operator* restr =
      vfes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restr && restr->Height() == nq * vdim * ne,
               "collocated_mass: the rule is not collocated with the velocity "
               "nodes (" << nq << " points per element)");
   // (nodes, component, element), lexicographic: the GLL rule's points are
   // the nodes in the same order.
   Vector de(restr->Height());
   de.UseDevice(true);
   if (ne > 0)
   {
      // MFEM's computation, not its cache (Mesh::GetGeometricFactors does the
      // same EnsureNodes first).
      mesh.EnsureNodes();
      const GeometricFactors geom(*mesh.GetNodes(), ir,
                                  GeometricFactors::DETERMINANTS,
                                  Device::GetDeviceMemoryType());
      const auto W = ir.GetWeights().Read();
      const auto detJ = Reshape(geom.detJ.Read(), nq, ne);
      auto D = Reshape(de.Write(), nq, vdim, ne);
      mfem::forall(nq * ne, [ = ] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % nq;
         const int e = i / nq;
         const real_t m = W[q] * detJ(q, e);
         for (int c = 0; c < vdim; ++c) { D(q, c, e) = m; }
      });
   }
   Vector dl(vfes.GetVSize());
   dl.UseDevice(true);
   restr->MultTranspose(de, dl);
   diag.UseDevice(true);
   diag.SetSize(vfes.GetTrueVSize());
   const Operator* P = vfes.GetProlongationMatrix();
   if (IsIdentityProlongation(P)) { diag = dl; }
   else { P->MultTranspose(dl, diag); }
}

DiagonalOperator::DiagonalOperator(const Vector& d)
   : Operator(d.Size()), d_(d)
{
}

void DiagonalOperator::Mult(const Vector& x, Vector& y) const
{
   const auto d = d_.Read();
   const auto X = x.Read();
   auto Y = y.Write();
   mfem::forall(height, [ = ] MFEM_HOST_DEVICE(int i)
   {
      Y[i] = d[i] * X[i];
   });
}

DiagonalShiftOperator::DiagonalShiftOperator(const Operator& A,
      const Vector& d, const double& c)
   : Operator(A.Height(), A.Width()), A_(A), d_(d), c_(c)
{
   MFEM_VERIFY(A.Height() == A.Width() && d.Size() == A.Height(),
               "collocated_mass: the shift needs a square block of the "
               "diagonal's size");
}

void DiagonalShiftOperator::Mult(const Vector& x, Vector& y) const
{
   MFEM_ASSERT(&x != &y, "collocated_mass: the shift cannot run in place");
   A_.Mult(x, y);
   AddScaledProduct(c_, d_, x, y);
}

void DiagonalShiftOperator::MultTranspose(const Vector& x, Vector& y) const
{
   MFEM_ASSERT(&x != &y, "collocated_mass: the shift cannot run in place");
   A_.MultTranspose(x, y);
   AddScaledProduct(c_, d_, x, y);
}

} // namespace incns
