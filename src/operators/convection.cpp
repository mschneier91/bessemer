#include "operators/convection.hpp"

#include "operators/directional_do_nothing.hpp"
#include "operators/kernel_specializations.hpp"
#include "time/oifs_convection.hpp"

#include <algorithm>
#include <vector>

namespace incns
{

using namespace mfem;

Convection::~Convection() = default;

Convection::Convection(MixedSpaces& spaces, const RuleBook& rules,
                       ElementScratch* scratch, bool volume)
   : spaces_(spaces), rules_(rules)
{
   // MFEM's form asks the mesh for the Jacobians at the dealiased rule (10
   // points per direction at order 6): on a GPU only the registered
   // compile-time kernels hold them (AddInterpolationKernels; idempotent).
   // Case and StokesOperator register them too; a Convection built on its
   // own (tests, apps) must not depend on that.
   AddInterpolationKernels();
   // Quad/hex only, same as every other operator here (StokesOperator does the
   // identical mapping) -- the tensor-product structure the PA kernels assume.
   const Geometry::Type geom =
      (spaces_.Dim() == 3) ? Geometry::CUBE : Geometry::SQUARE;

   // THE dealiasing choice. Elevated from the linear blocks' ~2k to 3k so the
   // cubic (u.grad)u . v integrand is integrated exactly -- see the header for
   // why, and do not lower it. Verified against MFEM's PA path, which selects
   // `IntRule ? IntRule : &GetRule(...)` (fem/integ/nonlininteg_vecconvection_pa.cpp),
   // so this rule is honored under partial assembly rather than silently
   // replaced by the integrator's default.
   rule_ = &rules_.Get(geom, DealiasedOrder(spaces_.OrderU()));
   if (!volume) { return; } // the boundary term alone (EnableDirectionalDoNothing)

   // Straight-sided elements at velocity orders 6 and 7: the same term,
   // C(u) u, through the OIFS on-the-fly kernel, which keeps nothing per point.
   // Applied once per step here (IMEX's explicit term; OIFS's wall nodes), so
   // the on-the-fly kernel's extra arithmetic costs little -- unlike the OIFS
   // substeps (time.oifs_on_the_fly).
   if (OifsConvection::OnTheFlyEligible(spaces_.Velocity(), *rule_))
   {
      fly_ = std::make_unique<OifsConvection>(spaces_.Velocity(), *rule_,
                                              scratch, /*on_the_fly=*/true,
                                              /*self_only=*/true);
      return;
   }
   // Curved elements at orders 6 and 7: the OIFS kernel's self apply by
   // element chunks, keeping nothing per point (MFEM's form keeps 18 numbers
   // per point, and at 11 points per direction cannot run on a GPU). On every
   // device, so a configuration has one path.
   ParFiniteElementSpace& vfes = spaces_.Velocity();
   if (OifsConvection::Supported(vfes, *rule_) &&
       vfes.GetTypicalFE()->GetDofToQuad(*rule_, DofToQuad::TENSOR).nqpt > 8)
   {
      fly_ = std::make_unique<OifsConvection>(vfes, *rule_, scratch,
                                              /*on_the_fly=*/false,
                                              /*self_only=*/true);
      return;
   }

   form_ = std::make_unique<ParNonlinearForm>(&spaces_.Velocity());
   // No coefficient: the integrator returns +(u.grad)u, and the sign stays with
   // the caller (see Mult's doc comment). MFEM's navier miniapp folds a -1 in
   // here instead; keeping it out makes the operator directly testable against
   // the analytic term.
   auto* nlfi = new VectorConvectionNLFIntegrator();
   nlfi->SetIntRule(rule_);
   form_->AddDomainIntegrator(nlfi);

   // Partial assembly, matching every other operator in this codebase (the
   // nonlinear form is applied every step and never needs its matrix).
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->Setup();
}

void Convection::EnableDirectionalDoNothing(const Array<int>& outflow_attrs)
{
   MFEM_VERIFY(!ddn_form_, "convection: directional do-nothing already enabled");
   ParMesh& mesh = *spaces_.Velocity().GetParMesh();
   const int n_attr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   ddn_marker_.SetSize(n_attr);
   ddn_marker_ = 0;
   for (int a : outflow_attrs)
   {
      MFEM_VERIFY(a >= 1 && a <= n_attr, "convection: outflow attribute " << a
                  << " not on the mesh");
      ddn_marker_[a - 1] = 1;
   }
   ddn_u_ = std::make_unique<ParGridFunction>(&spaces_.Velocity());
   *ddn_u_ = 0.0;
   ddn_form_ = std::make_unique<ParLinearForm>(&spaces_.Velocity());
   ddn_form_->UseDevice(true);
   auto* integ = new DirectionalDoNothingIntegrator(*ddn_u_, 0.5);
   ddn_integ_ = integ;
   // Same exactness as the interior term: (u.n)_- (u.phi) is cubic in u.
   const Geometry::Type face =
      (spaces_.Dim() == 3) ? Geometry::SQUARE : Geometry::SEGMENT;
   integ->SetIntRule(&rules_.Get(face, DealiasedOrder(spaces_.OrderU())));
   ddn_form_->AddBdrFaceIntegrator(integ, ddn_marker_);
   ddn_true_.SetSize(spaces_.Velocity().GetTrueVSize());
   ddn_true_.UseDevice(true);
}

void Convection::Mult(const Vector& u, Vector& y) const
{
   MFEM_VERIFY(fly_ || form_, "convection: built without the volume term "
               "(volume = false): only MultDirectionalDoNothing applies");
   if (fly_)
   {
      y.SetSize(spaces_.Velocity().GetTrueVSize());
      y.UseDevice(true);
      fly_->MultSelf(u, y);
   }
   else { form_->Mult(u, y); }
   if (!ddn_form_) { return; }
   MultDirectionalDoNothing(u, ddn_true_);
   y += ddn_true_;
}

void Convection::MultDirectionalDoNothing(const Vector& u, Vector& y) const
{
   MFEM_VERIFY(ddn_form_, "convection: directional do-nothing not enabled");
   // Host assembly over the marked faces (few), as LinearForm::Assemble does
   // it -- the same faces in the same order, the same element vectors added
   // the same way -- but only the elements next to them move between the
   // device and the host: a whole velocity vector each way per call before
   // (~90 ms per step on the DFG 3D cylinder at 22M DOFs, 2026-10-18).
   CacheDirectionalDoNothingDofs();
   ddn_u_->SetFromTrueDofs(u);
   {
      // A device view of ddn_u_ (the integrator reads ddn_u_ on the host).
      Vector lu;
      lu.MakeRef(*ddn_u_, 0, ddn_u_->Size());
      lu.UseDevice(true);
      lu.GetSubVector(ddn_dofs_, ddn_vals_);
   }
   const int nd = ddn_dofs_.Size();
   const int* d = ddn_dofs_.HostRead();
   {
      // The host copy is current at ddn_dofs_ only: all the faces read.
      real_t* hu = ddn_u_->HostWrite();
      const real_t* v = ddn_vals_.HostRead();
      for (int k = 0; k < nd; ++k) { hu[d[k]] = v[k]; }
   }
   ParFiniteElementSpace& fes = spaces_.Velocity();
   ParMesh& mesh = *fes.GetParMesh();
   ddn_b_.UseDevice(true);
   ddn_b_.SetSize(nd);
   real_t* b = ddn_b_.HostWrite();
   for (int k = 0; k < nd; ++k) { b[k] = 0.0; }
   for (const int i : ddn_bdr_)
   {
      FaceElementTransformations* tr = mesh.GetBdrFaceTransformations(i);
      fes.GetElementVDofs(tr->Elem1No, ddn_vdofs_);
      ddn_integ_->AssembleRHSElementVect(*fes.GetFE(tr->Elem1No), *tr,
                                         ddn_elvect_);
      for (int j = 0; j < ddn_vdofs_.Size(); ++j)
      {
         const int v = ddn_vdofs_[j];
         const int k = static_cast<int>(
                          std::lower_bound(d, d + nd, v >= 0 ? v : -1 - v) - d);
         if (v >= 0) { b[k] += ddn_elvect_(j); }
         else { b[k] -= ddn_elvect_(j); }
      }
   }
   *ddn_form_ = 0.0;
   ddn_form_->SetSubVector(ddn_dofs_, ddn_b_);
   y.SetSize(fes.GetTrueVSize());
   y.UseDevice(true);
   ddn_form_->ParallelAssemble(y); // MFEM-owned buffer (CLAUDE.md §6)
}

void Convection::CacheDirectionalDoNothingDofs() const
{
   ParFiniteElementSpace& fes = spaces_.Velocity();
   if (ddn_seq_ == fes.GetSequence()) { return; }
   ParMesh& mesh = *fes.GetParMesh();
   std::vector<int> dofs;
   ddn_bdr_.SetSize(0);
   for (int i = 0; i < fes.GetNBE(); ++i)
   {
      if (ddn_marker_[mesh.GetBdrAttribute(i) - 1] == 0) { continue; }
      FaceElementTransformations* tr = mesh.GetBdrFaceTransformations(i);
      if (tr == nullptr) { continue; }
      ddn_bdr_.Append(i);
      fes.GetElementVDofs(tr->Elem1No, ddn_vdofs_);
      for (const int v : ddn_vdofs_) { dofs.push_back(v >= 0 ? v : -1 - v); }
   }
   std::sort(dofs.begin(), dofs.end());
   dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
   ddn_dofs_.SetSize(static_cast<int>(dofs.size()));
   for (int k = 0; k < ddn_dofs_.Size(); ++k) { ddn_dofs_[k] = dofs[k]; }
   ddn_seq_ = fes.GetSequence();
}

} // namespace incns
