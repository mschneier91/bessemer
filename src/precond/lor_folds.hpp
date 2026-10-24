/**
 * @file lor_folds.hpp
 * @brief Folded LOR sub-elements: their count, and integrators that assemble
 *        them with |det J| so that an LOR operator stays positive definite.
 *
 * On a moved curved mesh the GLL-node LOR refinement can fold sub-elements
 * (a Q1 corner Jacobian <= 0) inside high-order elements that are still
 * valid: FSI2's flag tip (2026-10-23). Integrators divide or multiply by the
 * signed det J, so a folded sub-element gives a negative element matrix and
 * the LOR operator loses definiteness; the AMG on it then diverges. The
 * pressure L_p (LpSurrogate) and the velocity LOR-AMG (StokesSolver) check for
 * folds at every setup and take |det J| there; without folds they are
 * unchanged.
 */
#pragma once

#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Index of LOR vertex (i, j, k) of element e in the batched LOR vertex
/// coordinates (byVDIM: sdim x n1^dim x ne, lexicographic, x fastest).
MFEM_HOST_DEVICE inline int LorVertex(int sdim, int n1, int nv, int e, int i,
                                      int j, int k)
{
   return sdim * (nv * e + i + n1 * (j + n1 * k));
}

/**
 * @brief The number of LOR sub-elements, over all ranks, of @p fes's
 *        GLL-node refinement whose Q1 map has a non-positive Jacobian at a
 *        vertex. The batched LOR assembly integrates there, so such a
 *        sub-element gives a negative (semi)definite element matrix.
 *
 * A moved curved mesh can fold sub-elements that its high-order elements
 * survive: FSI2's flag tip (2026-10-23), where L_p's diagonal went
 * non-positive and the V-cycles stopped preconditioning. Collective. 0 for
 * spaces the check does not cover (non-tensor elements, DG, a surface).
 */
inline long long CountFoldedLorElements(mfem::ParFiniteElementSpace& fes)
{
   mfem::ParMesh& mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   int covered = (mfem::UsesTensorBasis(fes) && !fes.IsDGSpace() &&
                  mesh.SpaceDimension() == dim && (dim == 2 || dim == 3))
                 ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &covered, 1, MPI_INT, MPI_MIN, fes.GetComm());
   if (!covered) { return 0; }
   const int p = fes.GetMaxElementOrder(), n1 = p + 1;
   const int nv = dim == 2 ? n1 * n1 : n1 * n1 * n1;
   const int nsub = dim == 2 ? p * p : p * p * p;
   const int ne = fes.GetNE(), sdim = dim;
   // The LOR vertices: the mesh's nodes at the n1-point Gauss-Lobatto
   // tensor rule (lexicographic, x fastest), as the batched assembly's
   // FormLORVertexCoordinates finds them -- but on a rule with a fixed
   // address. MFEM caches quadrature maps by the rule's address, and that
   // function's rule is on its stack: the pressure's and the velocity's
   // orders met each other's stale maps there (a 25-point map wrote into a
   // 16-point buffer).
   static mfem::IntegrationRules lobatto(0, mfem::Quadrature1D::GaussLobatto);
   const mfem::IntegrationRule& ir =
      lobatto.Get(mesh.GetTypicalElementGeometry(), 2 * n1 - 3);
   MFEM_VERIFY(ir.GetNPoints() == nv, "lor_folds: the vertex rule has "
               << ir.GetNPoints() << " points, not " << nv);
   mesh.EnsureNodes();
   const mfem::GridFunction& nodes = *mesh.GetNodes();
   const mfem::FiniteElementSpace& nfes = *nodes.FESpace();
   const mfem::Operator* restriction =
      nfes.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
   mfem::Vector nodal_e(restriction->Height());
   nodal_e.UseDevice(true);
   restriction->Mult(nodes, nodal_e);
   mfem::Vector x_vert(sdim * nv * ne);
   x_vert.UseDevice(true);
   const mfem::QuadratureInterpolator* qi = nfes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
   qi->Values(nodal_e, x_vert);
   mfem::Vector folded(ne * nsub);
   folded.UseDevice(true);
   const double* X = x_vert.Read();
   double* F = folded.Write();
   if (dim == 2)
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub, ix = s % p, iy = s / p;
         double fold = 0.0;
         for (int b = 0; b < 2; ++b)
         {
            for (int a = 0; a < 2; ++a)
            {
               // The edges through vertex (a, b) along xi and eta.
               const int x0 = LorVertex(sdim, n1, nv, e, ix, iy + b, 0);
               const int x1 = LorVertex(sdim, n1, nv, e, ix + 1, iy + b, 0);
               const int y0 = LorVertex(sdim, n1, nv, e, ix + a, iy, 0);
               const int y1 = LorVertex(sdim, n1, nv, e, ix + a, iy + 1, 0);
               const double det = (X[x1] - X[x0]) * (X[y1 + 1] - X[y0 + 1]) -
                                  (X[x1 + 1] - X[x0 + 1]) * (X[y1] - X[y0]);
               if (!(det > 0.0)) { fold = 1.0; }
            }
         }
         F[t] = fold;
      });
   }
   else
   {
      mfem::forall(ne * nsub, [ = ] MFEM_HOST_DEVICE(int t)
      {
         const int e = t / nsub, s = t % nsub;
         const int ix = s % p, iy = (s / p) % p, iz = s / (p * p);
         double fold = 0.0;
         for (int c = 0; c < 2; ++c)
         {
            for (int b = 0; b < 2; ++b)
            {
               for (int a = 0; a < 2; ++a)
               {
                  // The edges through vertex (a, b, c) along xi, eta, zeta.
                  double u[3], v[3], w[3];
                  const int x0 = LorVertex(sdim, n1, nv, e, ix, iy + b, iz + c);
                  const int x1 = LorVertex(sdim, n1, nv, e, ix + 1, iy + b, iz + c);
                  const int y0 = LorVertex(sdim, n1, nv, e, ix + a, iy, iz + c);
                  const int y1 = LorVertex(sdim, n1, nv, e, ix + a, iy + 1, iz + c);
                  const int z0 = LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz);
                  const int z1 = LorVertex(sdim, n1, nv, e, ix + a, iy + b, iz + 1);
                  for (int d = 0; d < 3; ++d)
                  {
                     u[d] = X[x1 + d] - X[x0 + d];
                     v[d] = X[y1 + d] - X[y0 + d];
                     w[d] = X[z1 + d] - X[z0 + d];
                  }
                  const double det = u[0] * (v[1] * w[2] - v[2] * w[1]) -
                                     u[1] * (v[0] * w[2] - v[2] * w[0]) +
                                     u[2] * (v[0] * w[1] - v[1] * w[0]);
                  if (!(det > 0.0)) { fold = 1.0; }
               }
            }
         }
         F[t] = fold;
      });
   }
   long long n = static_cast<long long>(folded.Sum() + 0.5);
   MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG_LONG, MPI_SUM, fes.GetComm());
   return n;
}

/**
 * @brief An integrator that assembles @p inner with |det J| on folded
 *        elements (non-owning).
 *
 * Where det J > 0 at every point of the rule it is @p inner, unchanged. On
 * an element where it is not, the element matrix is the sum of @p inner's
 * one-point contributions, each times the sign of det J there: for the mass
 * (W det J), the diffusion (W adj J adj J^T / det J) and the grad-div
 * (W det J div div, div ~ 1 / det J) terms that is exactly |det J| in place of
 * det J. Boundary elements (a non-square Jacobian) pass through. Needs a rule
 * (the LOR assembly sets the vertex rule), else it passes through too.
 */
class FoldSafeIntegrator : public mfem::BilinearFormIntegrator
{
public:
   /// @param inner The integrator (borrowed: must outlive this).
   explicit FoldSafeIntegrator(mfem::BilinearFormIntegrator& inner) : inner_
      (inner) {}

   void AssembleElementMatrix(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& T,
                              mfem::DenseMatrix& elmat) override
   {
      const mfem::IntegrationRule* saved = inner_.GetIntRule();
      const mfem::IntegrationRule* ir = IntRule ? IntRule : saved;
      inner_.SetIntRule(ir);
      bool folded = false;
      if (ir && T.GetDimension() == T.GetSpaceDim())
      {
         for (int q = 0; q < ir->GetNPoints() && !folded; ++q)
         {
            T.SetIntPoint(&ir->IntPoint(q));
            folded = !(T.Jacobian().Det() > 0.0);
         }
      }
      if (!folded)
      {
         inner_.AssembleElementMatrix(el, T, elmat);
         inner_.SetIntRule(saved);
         return;
      }
      mfem::IntegrationRule one(1);
      bool first = true;
      for (int q = 0; q < ir->GetNPoints(); ++q)
      {
         const mfem::IntegrationPoint& ip = ir->IntPoint(q);
         T.SetIntPoint(&ip);
         const double det = T.Jacobian().Det();
         if (det == 0.0) { continue; } // a degenerate point: left out
         one.IntPoint(0) = ip;
         inner_.SetIntRule(&one);
         inner_.AssembleElementMatrix(el, T, part_);
         if (first)
         {
            elmat = part_;
            if (det < 0.0) { elmat.Neg(); }
            first = false;
         }
         else { elmat.Add(det > 0.0 ? 1.0 : -1.0, part_); }
      }
      if (first) // every point degenerate: the plain matrix (sized)
      {
         inner_.SetIntRule(ir);
         inner_.AssembleElementMatrix(el, T, elmat);
      }
      inner_.SetIntRule(saved);
   }

private:
   mfem::BilinearFormIntegrator& inner_; ///< The wrapped integrator.
   mfem::DenseMatrix part_;              ///< One point's contribution.
};

/**
 * @return A form on @p src's space with @p src's domain and boundary
 *         integrators (and markers) each wrapped in a FoldSafeIntegrator, for
 *         an LOR assembly on folded sub-elements. @p src's integrators must
 *         outlive it. Face integrators are not covered (verified absent).
 */
inline std::unique_ptr<mfem::ParBilinearForm> MakeFoldSafeForm(
   mfem::ParBilinearForm& src)
{
   MFEM_VERIFY(src.GetFBFI()->Size() == 0 && src.GetBFBFI()->Size() == 0,
               "lor_folds: MakeFoldSafeForm() does not cover face integrators");
   auto f = std::make_unique<mfem::ParBilinearForm>(src.ParFESpace());
   for (mfem::BilinearFormIntegrator* bi : *src.GetDBFI())
   {
      f->AddDomainIntegrator(new FoldSafeIntegrator(*bi));
   }
   mfem::Array<mfem::BilinearFormIntegrator*>& bb = *src.GetBBFI();
   mfem::Array<mfem::Array<int>*>& bm = *src.GetBBFI_Marker();
   for (int i = 0; i < bb.Size(); ++i)
   {
      if (bm[i]) { f->AddBoundaryIntegrator(new FoldSafeIntegrator(*bb[i]), *bm[i]); }
      else { f->AddBoundaryIntegrator(new FoldSafeIntegrator(*bb[i])); }
   }
   return f;
}

} // namespace incns
