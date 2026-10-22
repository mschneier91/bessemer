#include "fsi/interface_transfer.hpp"

#include <algorithm>

namespace incns
{

using namespace mfem;

InterfaceTransfer::InterfaceTransfer(ParMesh& source,
                                     ParFiniteElementSpace& destination,
                                     const std::vector<int>& attributes)
   : source_(source), dst_(destination), vdim_(destination.GetVDim())
{
   MFEM_VERIFY(!attributes.empty(), "interface_transfer: no boundary attributes");
   MPI_Comm comm = destination.GetComm();
   int rank = 0;
   int size = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   const int dim = source.SpaceDimension();
   MFEM_VERIFY(destination.GetMesh()->SpaceDimension() == dim,
               "interface_transfer: the two meshes differ in dimension");

   // The destination's nodes on the interface, each local scalar dof once.
   ParMesh& dmesh = *destination.GetParMesh();
   std::vector<char> seen(destination.GetNDofs(), 0);
   std::vector<int> dofs_list;
   std::vector<double> coords; // dim per node
   Array<int> dofs;
   Vector x(dim);
   for (int be = 0; be < dmesh.GetNBE(); ++be)
   {
      if (std::find(attributes.begin(), attributes.end(),
                    dmesh.GetBdrAttribute(be)) == attributes.end())
      {
         continue;
      }
      destination.GetBdrElementDofs(be, dofs);
      const FiniteElement* fe = destination.GetBE(be);
      ElementTransformation* T = dmesh.GetBdrElementTransformation(be);
      const IntegrationRule& nodes = fe->GetNodes();
      MFEM_VERIFY(nodes.GetNPoints() == dofs.Size(),
                  "interface_transfer: the destination space must be nodal");
      for (int j = 0; j < dofs.Size(); ++j)
      {
         const int d = dofs[j] >= 0 ? dofs[j] : -1 - dofs[j];
         if (seen[d]) { continue; }
         seen[d] = 1;
         T->SetIntPoint(&nodes.IntPoint(j));
         T->Transform(nodes.IntPoint(j), x);
         dofs_list.push_back(d);
         for (int c = 0; c < dim; ++c) { coords.push_back(x(c)); }
      }
   }
   n_local_ = static_cast<int>(dofs_list.size());
   vdofs_.SetSize(n_local_ * vdim_);
   for (int c = 0; c < vdim_; ++c)
   {
      for (int k = 0; k < n_local_; ++k)
      {
         vdofs_[c * n_local_ + k] = destination.DofToVDof(dofs_list[k], c);
      }
   }

   // Every rank's nodes, everywhere.
   counts_.assign(size, 0);
   MPI_Allgather(&n_local_, 1, MPI_INT, counts_.data(), 1, MPI_INT, comm);
   offsets_.assign(size + 1, 0);
   for (int r = 0; r < size; ++r) { offsets_[r + 1] = offsets_[r] + counts_[r]; }
   n_global_ = offsets_[size];
   std::vector<int> ccounts(size);
   std::vector<int> coffsets(size);
   for (int r = 0; r < size; ++r)
   {
      ccounts[r] = counts_[r] * dim;
      coffsets[r] = offsets_[r] * dim;
   }
   std::vector<double> all(static_cast<std::size_t>(n_global_) * dim);
   MPI_Allgatherv(coords.data(), n_local_ * dim, MPI_DOUBLE, all.data(),
                  ccounts.data(), coffsets.data(), MPI_DOUBLE, comm);

   // A local search on this rank's elements (ParMesh::FindPoints is
   // collective and expects the same points everywhere: not wanted here).
   DenseMatrix pts(dim, n_global_);
   for (int k = 0; k < n_global_; ++k)
   {
      for (int c = 0; c < dim; ++c) { pts(c, k) = all[k * dim + c]; }
   }
   Array<int> elem;
   Array<IntegrationPoint> ips;
   if (n_global_ > 0) { source.Mesh::FindPoints(pts, elem, ips, false); }
   std::vector<int> owner(n_global_);
   for (int k = 0; k < n_global_; ++k) { owner[k] = elem[k] >= 0 ? rank : size; }
   MPI_Allreduce(MPI_IN_PLACE, owner.data(), n_global_, MPI_INT, MPI_MIN, comm);
   for (int k = 0; k < n_global_; ++k)
   {
      MFEM_VERIFY(owner[k] < size, "interface_transfer: interface node ("
                  << pts(0, k) << ", " << pts(1, k) << ") lies in no element "
                  "of the source mesh (do the two interfaces coincide?)");
      if (owner[k] == rank)
      {
         owned_.push_back(k);
         owned_elem_.push_back(elem[k]);
         owned_ip_.push_back(ips[k]);
      }
   }
}

void InterfaceTransfer::Transfer(const ParGridFunction& in,
                                 ParGridFunction& out) const
{
   MFEM_VERIFY(in.ParFESpace()->GetParMesh() == &source_,
               "interface_transfer: the field is not on the source mesh");
   MFEM_VERIFY(out.ParFESpace() == &dst_,
               "interface_transfer: the output is not on the destination space");
   MFEM_VERIFY(in.VectorDim() == vdim_, "interface_transfer: the field has "
               << in.VectorDim() << " components, the destination " << vdim_);
   std::vector<double> buf(static_cast<std::size_t>(n_global_) * vdim_, 0.0);
   Vector v(vdim_);
   in.HostRead();
   for (std::size_t j = 0; j < owned_.size(); ++j)
   {
      in.GetVectorValue(owned_elem_[j], owned_ip_[j], v);
      for (int c = 0; c < vdim_; ++c) { buf[owned_[j] * vdim_ + c] = v(c); }
   }
   MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()),
                 MPI_DOUBLE, MPI_SUM, dst_.GetComm());
   int rank = 0;
   MPI_Comm_rank(dst_.GetComm(), &rank);
   double* o = out.HostReadWrite();
   for (int c = 0; c < vdim_; ++c)
   {
      for (int k = 0; k < n_local_; ++k)
      {
         o[vdofs_[c * n_local_ + k]] = buf[(offsets_[rank] + k) * vdim_ + c];
      }
   }
}

} // namespace incns
