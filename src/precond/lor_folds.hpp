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
long long CountFoldedLorElements(mfem::ParFiniteElementSpace& fes);

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
