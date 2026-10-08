#include "operators/directional_do_nothing.hpp"

#include <algorithm>

namespace incns
{

using namespace mfem;

DirectionalDoNothingIntegrator::DirectionalDoNothingIntegrator(
   const GridFunction& u, double beta)
   : u_(u), beta_(beta)
{
   MFEM_VERIFY(u.FESpace()->GetVDim() == u.FESpace()->GetMesh()->Dimension(),
               "directional_do_nothing: u must be a vector field with "
               "vdim = dim");
}

void DirectionalDoNothingIntegrator::AssembleRHSElementVect(
   const mfem::FiniteElement& el, mfem::ElementTransformation& Tr,
   mfem::Vector& elvect)
{
   (void)el;
   (void)Tr;
   (void)elvect;
   MFEM_ABORT("directional_do_nothing: a boundary-face integrator "
              "(LinearForm::AddBdrFaceIntegrator), not a domain one");
}

void DirectionalDoNothingIntegrator::AssembleRHSElementVect(
   const mfem::FiniteElement& el, mfem::FaceElementTransformations& Tr,
   mfem::Vector& elvect)
{
   const int dim = el.GetDim();
   MFEM_ASSERT(dim == 2 || dim == 3, "directional_do_nothing: 2D/3D only");
   const int ndof = el.GetDof();
   elvect.SetSize(ndof * dim);
   elvect = 0.0;
   u_.FESpace()->GetElementVDofs(Tr.Elem1No, vdofs_);
   u_.GetSubVector(vdofs_, u_el_); // component-major: [c * ndof + i]
   shape_.SetSize(ndof);
   nor_.SetSize(dim);
   u_q_.SetSize(dim);

   const IntegrationRule* ir = IntRule;
   if (!ir)
   {
      // The integrand (u.n)_- (u.phi) is cubic in degree-k fields: 3k, the
      // same dealiasing order as the interior convection.
      ir = &IntRules.Get(Tr.GetGeometryType(), 3 * el.GetOrder());
   }
   for (int q = 0; q < ir->GetNPoints(); ++q)
   {
      const IntegrationPoint& ip = ir->IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      // Area-scaled outward normal: |nor| is the face measure density, so
      // (u.nor)_- ds_ref = (u.n)_- ds.
      CalcOrtho(Tr.Jacobian(), nor_);
      el.CalcShape(Tr.GetElement1IntPoint(), shape_);
      for (int c = 0; c < dim; ++c)
      {
         double s = 0.0;
         for (int i = 0; i < ndof; ++i) { s += shape_(i) * u_el_(c * ndof + i); }
         u_q_(c) = s;
      }
      const double un = std::min(u_q_ * nor_, 0.0);
      if (un == 0.0) { continue; } // outflow point: classical do-nothing
      const double w = -beta_ * ip.weight * un;
      for (int c = 0; c < dim; ++c)
      {
         const double wc = w * u_q_(c);
         for (int i = 0; i < ndof; ++i) { elvect(c * ndof + i) += wc * shape_(i); }
      }
   }
}

} // namespace incns
