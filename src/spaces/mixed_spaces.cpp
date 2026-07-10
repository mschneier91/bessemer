#include "spaces/mixed_spaces.hpp"

#include "mesh/periodic_box.hpp" // AssertTensorProductGeometry

namespace incns
{

using namespace mfem;

MixedSpaces::MixedSpaces(ParMesh& mesh, int order_u, int order_p)
   : dim_(mesh.Dimension()), order_u_(order_u), order_p_(order_p)
{
   MFEM_VERIFY(order_u_ >= 1 && order_p_ >= 1,
               "mixed_spaces: velocity/pressure orders must be >= 1");
   AssertTensorProductGeometry(mesh);

   if (order_u_ == order_p_ && Mpi::Root())
   {
      mfem::out << "[mixed_spaces] WARNING: k_u == k_p (" << order_u_
                << ") is inf-sup unstable without pressure stabilization "
                "(PSPG), which is not available in Sprint 1.\n";
   }

   // H1_FECollection defaults to the Gauss-Lobatto nodal basis, which the
   // collocated-mass option relies on later (Sprint 1.4).
   fec_u_ = std::make_unique<H1_FECollection>(order_u_, dim_);
   fec_p_ = std::make_unique<H1_FECollection>(order_p_, dim_);

   // Velocity is vector-valued (vdim = dim); byNODES keeps each component's
   // dofs contiguous. Pressure is scalar.
   vfes_ = std::make_unique<ParFiniteElementSpace>(&mesh, fec_u_.get(), dim_,
           Ordering::byNODES);
   pfes_ = std::make_unique<ParFiniteElementSpace>(&mesh, fec_p_.get());

   block_true_offsets_.SetSize(3);
   block_true_offsets_[0] = 0;
   block_true_offsets_[1] = vfes_->GetTrueVSize();
   block_true_offsets_[2] = pfes_->GetTrueVSize();
   block_true_offsets_.PartialSum();
}

} // namespace incns
