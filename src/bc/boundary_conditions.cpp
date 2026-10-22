#include "bc/boundary_conditions.hpp"

// The separable data's kernel must be compiled by nvcc in a CUDA build, or it
// runs on the host while the data is on the device: see the nvcc TU list in
// src/CMakeLists.txt.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "boundary_conditions.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Device kernels live in free functions in a NAMED namespace: nvcc rejects
// extended lambdas in private members.
namespace bckernels
{

// vals[k] = s_e(k) g(k) (fac: the entries' factors; entry: each dof's entry).
void ScaleEntries(const Vector& fac, const Vector& shape, const Vector& entry,
                  Vector& vals)
{
   const int n = shape.Size();
   vals.SetSize(n);
   const auto F = fac.Read();
   const auto G = shape.Read();
   const auto E = entry.Read();
   auto U = vals.Write();
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int k)
   {
      U[k] = F[static_cast<int>(E[k])] * G[k];
   });
}

} // namespace bckernels

void TrueDofLocalDofs(const ParFiniteElementSpace& fes, const Array<int>& tdofs,
                      std::vector<int>& ldofs)
{
   std::vector<int> t2l(fes.GetTrueVSize(), -1);
   if (fes.Conforming())
   {
      // R is the transpose of P's diagonal block: the owned local dof of each
      // true-dof number.
      for (int l = 0; l < fes.GetVSize(); ++l)
      {
         const int t = fes.GetLocalTDofNumber(l);
         if (t >= 0) { t2l[t] = l; }
      }
   }
   else
   {
      const SparseMatrix* R = fes.GetRestrictionMatrix();
      const int* ri = R->HostReadI();
      const int* rj = R->HostReadJ();
      const real_t* ra = R->HostReadData();
      for (int t = 0; t < R->Height(); ++t)
      {
         MFEM_VERIFY(ri[t + 1] - ri[t] == 1 && ra[ri[t]] == 1.0,
                     "TrueDofLocalDofs: the restriction is not a selection");
         t2l[t] = rj[ri[t]];
      }
   }
   const int* td = tdofs.HostRead();
   ldofs.resize(tdofs.Size());
   for (int k = 0; k < tdofs.Size(); ++k)
   {
      ldofs[k] = t2l[td[k]];
      MFEM_VERIFY(ldofs[k] >= 0, "TrueDofLocalDofs: a true dof without a "
                  "local dof");
   }
}

BoundaryConditions::BoundaryConditions(ParFiniteElementSpace& vfes)
   : vfes_(vfes)
{
   ParMesh& mesh = *vfes_.GetParMesh();
   // Attribute values are global mesh data, but a rank that owns no boundary
   // elements has an empty local list -- reduce for the global maximum.
   int local_max = (mesh.bdr_attributes.Size() > 0) ? mesh.bdr_attributes.Max() :
                   0;
   MPI_Allreduce(&local_max, &max_attr_, 1, MPI_INT, MPI_MAX, mesh.GetComm());

   dirichlet_marker_.SetSize(max_attr_);
   outflow_marker_.SetSize(max_attr_);
   robin_marker_.SetSize(max_attr_);
   real_bdr_marker_.SetSize(max_attr_);
   dirichlet_marker_ = 0;
   outflow_marker_ = 0;
   robin_marker_ = 0;
   real_bdr_marker_ = 0;
   ess_tdofs_.SetSize(0);

   // Which attributes own REAL boundary faces? MakePeriodic keeps the base
   // mesh's boundary elements, but their faces are topologically interior
   // (periodic wrap), so bdr_attributes alone cannot distinguish a periodic
   // pseudo-boundary from an actual one. Classify by face topology instead.
   // ExchangeFaceNbrData is required first: without it, a wrap face whose
   // partner element lives on another rank is misreported as Boundary.
   if (max_attr_ > 0)
   {
      mesh.ExchangeFaceNbrData();
      for (int be = 0; be < mesh.GetNBE(); ++be)
      {
         const int f = mesh.GetBdrElementFaceIndex(be);
         if (mesh.GetFaceInformation(f).IsBoundary())
         {
            real_bdr_marker_[mesh.GetBdrAttribute(be) - 1] = 1;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, real_bdr_marker_.GetData(), max_attr_,
                    MPI_INT, MPI_MAX, mesh.GetComm());
   }
}

void BoundaryConditions::AddVelocityDirichlet(int attr,
      VectorCoefficient& coeff)
{
   MFEM_VERIFY(attr >= 1 && attr <= max_attr_,
               "BoundaryConditions: boundary attribute out of range");
   MFEM_VERIFY(real_bdr_marker_[attr - 1] == 1,
               "BoundaryConditions: attribute has no real boundary faces "
               "(periodic pseudo-boundary?) -- eliminating its dofs would "
               "corrupt the periodic coupling");
   MFEM_VERIFY(dirichlet_marker_[attr - 1] == 0 && outflow_marker_[attr - 1] == 0
               &&
               robin_marker_[attr - 1] == 0,
               "BoundaryConditions: attribute already has a condition");
   MFEM_VERIFY(coeff.GetVDim() == vfes_.GetVDim(),
               "BoundaryConditions: coefficient vdim does not match velocity");

   dirichlet_marker_[attr - 1] = 1;
   dirichlet_.emplace_back(attr, &coeff);
   UpdateEssentialTrueDofs();
}

void BoundaryConditions::AddNoSlip(int attr)
{
   if (!zero_coeff_)
   {
      Vector zero(vfes_.GetVDim());
      zero = 0.0;
      zero_coeff_ = std::make_unique<VectorConstantCoefficient>(zero);
   }
   AddVelocityDirichlet(attr, *zero_coeff_);
}

void BoundaryConditions::AddOutflow(int attr)
{
   MFEM_VERIFY(attr >= 1 && attr <= max_attr_,
               "BoundaryConditions: boundary attribute out of range");
   MFEM_VERIFY(real_bdr_marker_[attr - 1] == 1,
               "BoundaryConditions: attribute has no real boundary faces "
               "(periodic pseudo-boundary?)");
   MFEM_VERIFY(dirichlet_marker_[attr - 1] == 0 && outflow_marker_[attr - 1] == 0
               &&
               robin_marker_[attr - 1] == 0,
               "BoundaryConditions: attribute already has a condition");

   outflow_marker_[attr - 1] = 1;
   has_outflow_ = true;
   // Outflow is natural: nothing to assemble, nothing to eliminate.
}

void BoundaryConditions::AddRobin(int attr, double alpha,
                                  mfem::VectorCoefficient& w)
{
   MFEM_VERIFY(alpha > 0.0, "BoundaryConditions: the Robin coefficient must be "
               "positive");
   if (!robin_constant_)
   {
      robin_constant_ = std::make_unique<ConstantCoefficient>(alpha);
   }
   MFEM_VERIFY(robin_constant_->constant == alpha,
               "BoundaryConditions: one Robin coefficient for all attributes");
   AddRobin(attr, *robin_constant_, alpha, w);
}

void BoundaryConditions::AddRobin(int attr, mfem::Coefficient& alpha,
                                  double alpha_max,
                                  mfem::VectorCoefficient& w)
{
   MFEM_VERIFY(attr >= 1 && attr <= max_attr_,
               "BoundaryConditions: boundary attribute out of range");
   MFEM_VERIFY(real_bdr_marker_[attr - 1] == 1,
               "BoundaryConditions: attribute has no real boundary faces "
               "(periodic pseudo-boundary?)");
   MFEM_VERIFY(dirichlet_marker_[attr - 1] == 0 && outflow_marker_[attr - 1] == 0
               &&
               robin_marker_[attr - 1] == 0,
               "BoundaryConditions: attribute already has a condition");
   MFEM_VERIFY(alpha_max > 0.0, "BoundaryConditions: the Robin coefficient "
               "must be positive");
   MFEM_VERIFY(!robin_alpha_ || robin_alpha_ == &alpha,
               "BoundaryConditions: one Robin coefficient for all attributes");
   MFEM_VERIFY(!robin_w_ || robin_w_ == &w,
               "BoundaryConditions: one Robin velocity for all attributes");
   MFEM_VERIFY(w.GetVDim() == vfes_.GetVDim(),
               "BoundaryConditions: coefficient vdim does not match velocity");
   robin_marker_[attr - 1] = 1;
   robin_alpha_ = &alpha;
   robin_alpha_max_ = alpha_max;
   robin_w_ = &w;
   // Natural as well: the operator and the right-hand side carry it.
}

bool BoundaryConditions::DirichletIsConstant() const
{
   for (const auto& entry : dirichlet_)
   {
      if (!dynamic_cast<const VectorConstantCoefficient*>(entry.second))
      {
         return false;
      }
   }
   return true;
}

bool BoundaryConditions::DirichletIsSeparable() const
{
   for (const auto& entry : dirichlet_)
   {
      if (!dynamic_cast<const VectorConstantCoefficient*>(entry.second) &&
          !dynamic_cast<const SeparableVectorCoefficient*>(entry.second))
      {
         return false;
      }
   }
   return true;
}

double BoundaryConditions::DirichletFactor(int e, double t) const
{
   MFEM_ASSERT(e >= 0 && e < NumDirichlet(),
               "BoundaryConditions: Dirichlet entry out of range");
   const auto* s = dynamic_cast<const SeparableVectorCoefficient*>(
                      dirichlet_[e].second);
   return s ? s->TimeFactor(t) : 1.0;
}

void BoundaryConditions::ProjectDirichletShapes(ParGridFunction& g,
      ParGridFunction& entry) const
{
   MFEM_VERIFY(DirichletIsSeparable(), "BoundaryConditions: the Dirichlet "
               "data is not separable in time");
   Array<int> marker(max_attr_);
   Vector id(vfes_.GetVDim());
   for (std::size_t e = 0; e < dirichlet_.size(); ++e)
   {
      marker = 0;
      marker[dirichlet_[e].first - 1] = 1;
      VectorCoefficient* c = dirichlet_[e].second;
      auto* s = dynamic_cast<SeparableVectorCoefficient*>(c);
      g.ProjectBdrCoefficient(s ? s->Shape() : *c, marker);
      // The same projection of the index: an average of equal whole
      // numbers, exact.
      id = static_cast<double>(e);
      VectorConstantCoefficient ec(id);
      entry.ProjectBdrCoefficient(ec, marker);
   }
}

void BoundaryConditions::CacheShapes() const
{
   if (shape_seq_ == vfes_.GetSequence()) { return; }
   MFEM_VERIFY(DirichletIsSeparable(), "BoundaryConditions: the Dirichlet "
               "data is not separable in time");
   // Host-only scratch: the projections run on the host, and only the values
   // on the essential dofs are kept.
   ParGridFunction g(&vfes_), entry(&vfes_);
   g.UseDevice(false);
   entry.UseDevice(false);
   g = 0.0;
   entry = -1.0;
   ProjectDirichletShapes(g, entry);
   std::vector<int> ldof;
   TrueDofLocalDofs(vfes_, ess_tdofs_, ldof);
   const int n = static_cast<int>(ldof.size());
   ess_shape_.SetSize(n);
   ess_entry_.SetSize(n);
   const double* gh = g.HostRead();
   const double* eh = entry.HostRead();
   double* sh = ess_shape_.HostWrite();
   double* nh = ess_entry_.HostWrite();
   for (int k = 0; k < n; ++k)
   {
      sh[k] = gh[ldof[k]]; // GetTrueDofs' value: a copy of the local dof
      nh[k] = eh[ldof[k]];
      MFEM_VERIFY(nh[k] >= 0.0 && nh[k] < NumDirichlet(), "BoundaryConditions: "
                  "a Dirichlet dof without a projected shape");
   }
   ess_shape_.UseDevice(true);
   ess_entry_.UseDevice(true);
   shape_seq_ = vfes_.GetSequence();
}

const Vector& BoundaryConditions::EssentialShapes() const
{
   CacheShapes();
   return ess_shape_;
}

const Vector& BoundaryConditions::EssentialEntries() const
{
   CacheShapes();
   return ess_entry_;
}

void BoundaryConditions::SeparableEssentialValues(Vector& vals) const
{
   CacheShapes();
   const int ne = NumDirichlet();
   fac_.UseDevice(true);
   fac_.SetSize(ne);
   double* f = fac_.HostWrite();
   for (int e = 0; e < ne; ++e)
   {
      // Each coefficient's own time: what ProjectDirichlet() would evaluate.
      auto* s = dynamic_cast<SeparableVectorCoefficient*>(dirichlet_[e].second);
      f[e] = s ? s->TimeFactor(s->GetTime()) : 1.0;
   }
   vals.UseDevice(true);
   bckernels::ScaleEntries(fac_, ess_shape_, ess_entry_, vals);
}

void BoundaryConditions::SetTime(double t)
{
   for (auto& entry : dirichlet_) { entry.second->SetTime(t); }
}

void BoundaryConditions::ProjectDirichlet(ParGridFunction& u) const
{
   Array<int> marker(max_attr_);
   for (const auto& entry : dirichlet_)
   {
      marker = 0;
      marker[entry.first - 1] = 1;
      u.ProjectBdrCoefficient(*entry.second, marker);
   }
}

bool BoundaryConditions::PressureNullspaceExists() const
{
   if (has_outflow_) { return false; }
   // Every attribute with REAL boundary faces must be Dirichlet: an unassigned
   // real attribute is a natural (do-nothing) boundary, which fixes the
   // pressure level. Periodic pseudo-boundary attributes are ignored, so a
   // fully periodic mesh (no real boundary) is vacuously enclosed.
   for (int a = 0; a < max_attr_; ++a)
   {
      if (real_bdr_marker_[a] == 1 && dirichlet_marker_[a] == 0) { return false; }
   }
   return true;
}

void BoundaryConditions::UpdateEssentialTrueDofs()
{
   Array<int> marker(dirichlet_marker_); // copy: GetEssentialTrueDofs is non-const
   vfes_.GetEssentialTrueDofs(marker, ess_tdofs_);
   shape_seq_ = -1; // the shapes are re-gathered on the new list
}

} // namespace incns
