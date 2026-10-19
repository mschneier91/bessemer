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

bool Convection::RestrictToTrueDofs(const Array<int>& tdofs)
{
   ParFiniteElementSpace& V = spaces_.Velocity();
   // Hanging nodes: the prolongation spreads a true dof over constrained
   // local dofs with weights of either sign; keep the whole mesh there. The
   // same choice on every rank (the mesh's type), so the collective holds.
   if (!fly_ || V.Nonconforming()) { return false; }
   fly_->RestrictToElements(OifsConvection::ElementsTouching(V, tdofs));
   return true;
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
   // The integrator's arithmetic on the marked faces, in the same order of
   // faces and points, from data cached per mesh (weights, normals, the face
   // dofs' shapes): no face transformation and no element shapes per call
   // (~50 ms per call on the DFG 3D cylinder at 22M DOFs, 2026-10-18), and
   // only u on the faces' dofs comes to the host. The element's other shapes
   // vanish on the face: the same term to roundoff.
   CacheDirectionalDoNothingDofs();
   ddn_u_->SetFromTrueDofs(u);
   {
      Vector lu;
      lu.MakeRef(*ddn_u_, 0, ddn_u_->Size());
      lu.UseDevice(true);
      lu.GetSubVector(ddn_dofs_, ddn_vals_);
   }
   const int nd = ddn_dofs_.Size();
   const int dim = spaces_.Dim();
   MFEM_ASSERT(dim <= 3, "convection: 2D or 3D");
   const real_t* v = ddn_vals_.HostRead();
   ddn_b_.UseDevice(true);
   ddn_b_.SetSize(nd);
   real_t* b = ddn_b_.HostWrite();
   for (int k = 0; k < nd; ++k) { b[k] = 0.0; }
   const double beta = ddn_integ_->Beta();
   for (int f = 0; f < ddn_bdr_.Size(); ++f)
   {
      const int q0 = ddn_qoff_[f], nq = ddn_qoff_[f + 1] - q0;
      const int nfd = ddn_doff_[f + 1] - ddn_doff_[f];
      const int* fk = &ddn_fk_[dim * ddn_doff_[f]];
      ddn_e_.assign(static_cast<std::size_t>(dim) * nfd, 0.0);
      double* e = ddn_e_.data();
      for (int q = 0; q < nq; ++q)
      {
         const double* sh = &ddn_shape_[ddn_soff_[f] + q * nfd];
         const double* nor = &ddn_nor_[dim * (q0 + q)];
         double uq[3];
         double un = 0.0;
         for (int c = 0; c < dim; ++c)
         {
            double s = 0.0;
            for (int j = 0; j < nfd; ++j)
            {
               const int k = fk[c * nfd + j];
               s += sh[j] * ((k >= 0) ? v[k] : -v[-1 - k]);
            }
            uq[c] = s;
            un += s * nor[c];
         }
         un = std::min(un, 0.0);
         if (un == 0.0) { continue; } // outflow point: classical do-nothing
         const double w = -beta * ddn_w_[q0 + q] * un;
         for (int c = 0; c < dim; ++c)
         {
            const double wc = w * uq[c];
            for (int j = 0; j < nfd; ++j) { e[c * nfd + j] += wc * sh[j]; }
         }
      }
      for (int m = 0; m < dim * nfd; ++m)
      {
         const int k = fk[m];
         if (k >= 0) { b[k] += e[m]; }
         else { b[-1 - k] -= e[m]; }
      }
   }
   *ddn_form_ = 0.0;
   ddn_form_->SetSubVector(ddn_dofs_, ddn_b_);
   y.SetSize(spaces_.Velocity().GetTrueVSize());
   y.UseDevice(true);
   ddn_form_->ParallelAssemble(y); // MFEM-owned buffer (CLAUDE.md §6)
}

void Convection::CacheDirectionalDoNothingDofs() const
{
   ParFiniteElementSpace& fes = spaces_.Velocity();
   if (ddn_seq_ == fes.GetSequence()) { return; }
   ParMesh& mesh = *fes.GetParMesh();
   const int dim = spaces_.Dim();
   auto decode = [](int d) { return d >= 0 ? d : -1 - d; };
   // The marked faces in mesh order; per face its own dofs (the boundary
   // element's, found among the neighbour element's) and their quadrature
   // data.
   ddn_bdr_.SetSize(0);
   ddn_w_.clear();
   ddn_nor_.clear();
   ddn_shape_.clear();
   ddn_qoff_.assign(1, 0);
   ddn_doff_.assign(1, 0);
   ddn_soff_.assign(1, 0);
   std::vector<int> fv; // per face, component-major: the face dofs' vdofs
   Array<int> edofs, bdofs, pos;
   for (int i = 0; i < fes.GetNBE(); ++i)
   {
      if (ddn_marker_[mesh.GetBdrAttribute(i) - 1] == 0) { continue; }
      FaceElementTransformations* tr = mesh.GetBdrFaceTransformations(i);
      if (tr == nullptr) { continue; }
      ddn_bdr_.Append(i);
      const FiniteElement& el = *fes.GetFE(tr->Elem1No);
      fes.GetElementDofs(tr->Elem1No, edofs);
      fes.GetBdrElementDofs(i, bdofs);
      fes.GetElementVDofs(tr->Elem1No, ddn_vdofs_);
      const int ndof = edofs.Size();
      pos.SetSize(bdofs.Size());
      for (int j = 0; j < bdofs.Size(); ++j)
      {
         pos[j] = -1;
         for (int e = 0; e < ndof && pos[j] < 0; ++e)
         {
            if (decode(edofs[e]) == decode(bdofs[j])) { pos[j] = e; }
         }
         MFEM_VERIFY(pos[j] >= 0, "convection: a boundary face's dof is not "
                     "one of its element's");
      }
      for (int c = 0; c < dim; ++c)
      {
         for (int j = 0; j < pos.Size(); ++j)
         {
            fv.push_back(ddn_vdofs_[c * ndof + pos[j]]);
         }
      }
      const int nq = ddn_integ_->FaceData(el, *tr, pos, ddn_w_, ddn_nor_,
                                          ddn_shape_);
      ddn_qoff_.push_back(ddn_qoff_.back() + nq);
      ddn_doff_.push_back(ddn_doff_.back() + pos.Size());
      ddn_soff_.push_back(ddn_soff_.back() + nq * pos.Size());
   }
   std::vector<int> dofs(fv.size());
   for (std::size_t m = 0; m < fv.size(); ++m) { dofs[m] = decode(fv[m]); }
   std::sort(dofs.begin(), dofs.end());
   dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
   // Through an explicit host write: the list is read on the device.
   ddn_dofs_.SetSize(static_cast<int>(dofs.size()));
   int* d = ddn_dofs_.HostWrite();
   for (std::size_t k = 0; k < dofs.size(); ++k) { d[k] = dofs[k]; }
   ddn_fk_.resize(fv.size());
   for (std::size_t m = 0; m < fv.size(); ++m)
   {
      const int k = static_cast<int>(std::lower_bound(dofs.begin(), dofs.end(),
                                     decode(fv[m])) - dofs.begin());
      ddn_fk_[m] = (fv[m] >= 0) ? k : -1 - k;
   }
   ddn_seq_ = fes.GetSequence();
}

} // namespace incns
