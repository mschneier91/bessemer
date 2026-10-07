#include "post/body_force.hpp"

namespace incns
{

using namespace mfem;

BodyForce::BodyForce(ParFiniteElementSpace& vfes,
                     const std::vector<int>& attributes)
   : comm_(vfes.GetComm())
{
   ParMesh& mesh = *vfes.GetParMesh();
   int max_attr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   MPI_Allreduce(MPI_IN_PLACE, &max_attr, 1, MPI_INT, MPI_MAX, comm_);
   MFEM_VERIFY(!attributes.empty(), "body_force: no body attributes given");
   Array<int> marker(max_attr);
   marker = 0;
   for (int a : attributes)
   {
      MFEM_VERIFY(a >= 1 && a <= max_attr, "body_force: boundary attribute "
                  << a << " is not in the mesh (1.." << max_attr << ")");
      marker[a - 1] = 1;
   }
   const int dim = vfes.GetVDim();
   const int n = vfes.GetTrueVSize();
   v_.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      // e_i at the body's velocity nodes, zero at every other node (true dofs:
      // hanging nodes on S follow from their masters, which are also on S).
      Array<int> dofs;
      vfes.GetEssentialTrueDofs(marker, dofs, i);
      v_[i].SetSize(n);
      v_[i].UseDevice(true);
      v_[i] = 0.0;
      v_[i].SetSubVector(dofs, 1.0);
   }
}

double BodyForce::ForceWith(const Vector& r, const Vector& v) const
{
   return -InnerProduct(comm_, r, v);
}

Vector BodyForce::Force(const Vector& r) const
{
   Vector F(static_cast<int>(v_.size()));
   for (std::size_t i = 0; i < v_.size(); ++i)
   {
      F(static_cast<int>(i)) = ForceWith(r, v_[i]);
   }
   return F;
}

Vector BodyForce::Coefficients(const Vector& F, double U, double A)
{
   MFEM_VERIFY(U > 0.0 && A > 0.0, "body_force: reference velocity and area "
               "must be positive");
   Vector C(F);
   C *= 2.0 / (U * U * A);
   return C;
}

} // namespace incns
