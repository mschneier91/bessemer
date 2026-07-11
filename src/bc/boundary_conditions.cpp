#include "bc/boundary_conditions.hpp"

namespace incns
{

using namespace mfem;

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
   real_bdr_marker_.SetSize(max_attr_);
   dirichlet_marker_ = 0;
   outflow_marker_ = 0;
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
   MFEM_VERIFY(dirichlet_marker_[attr - 1] == 0 && outflow_marker_[attr - 1] == 0,
               "BoundaryConditions: attribute already has a condition");
   MFEM_VERIFY(coeff.GetVDim() == vfes_.GetVDim(),
               "BoundaryConditions: coefficient vdim does not match velocity");

   dirichlet_marker_[attr - 1] = 1;
   dirichlet_.emplace_back(attr, &coeff);
   UpdateEssentialTrueDofs();
}

void BoundaryConditions::AddOutflow(int attr)
{
   MFEM_VERIFY(attr >= 1 && attr <= max_attr_,
               "BoundaryConditions: boundary attribute out of range");
   MFEM_VERIFY(real_bdr_marker_[attr - 1] == 1,
               "BoundaryConditions: attribute has no real boundary faces "
               "(periodic pseudo-boundary?)");
   MFEM_VERIFY(dirichlet_marker_[attr - 1] == 0 && outflow_marker_[attr - 1] == 0,
               "BoundaryConditions: attribute already has a condition");

   outflow_marker_[attr - 1] = 1;
   has_outflow_ = true;
   // Outflow is natural: nothing to assemble, nothing to eliminate.
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
}

} // namespace incns
