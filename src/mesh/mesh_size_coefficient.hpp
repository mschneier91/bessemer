/**
 * @file mesh_size_coefficient.hpp
 * @brief Per-element mesh-size coefficient gamma(x) = scale * h_K.
 */
#ifndef INCNS_MESH_MESH_SIZE_COEFFICIENT_HPP
#define INCNS_MESH_MESH_SIZE_COEFFICIENT_HPP

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Coefficient equal to @c scale * h_K on each element K -- the order-h,
 *        spatially varying grad-div coefficient gamma(x).
 *
 * The size measure is mfem::Mesh::GetElementSize type 0, @c det(J)^{1/dim} at
 * the element center (the volume-equivalent edge length); on Cartesian boxes
 * this is exactly the edge length. Piecewise constant per element, so any
 * quadrature rule integrates it exactly.
 */
class MeshSizeCoefficient : public mfem::Coefficient
{
public:
   /**
    * @brief Create the coefficient over a mesh.
    * @param mesh  Mesh whose local element sizes are queried (borrowed).
    * @param scale Multiplier @c c_gd: the coefficient value is scale * h_K.
    */
   MeshSizeCoefficient(mfem::ParMesh& mesh, double scale)
      : mesh_(mesh), scale_(scale) {}

   /**
    * @brief Evaluate @c scale * h at the transformation's element.
    * @param T Element transformation (identifies the local element).
    * @return The per-element value; constant within each element.
    */
   double Eval(mfem::ElementTransformation& T,
               const mfem::IntegrationPoint&) override
   {
      return scale_ * mesh_.GetElementSize(T.ElementNo);
   }

private:
   mfem::ParMesh& mesh_; ///< Mesh (borrowed).
   double scale_;        ///< Multiplier c_gd.
};

} // namespace incns

#endif // INCNS_MESH_MESH_SIZE_COEFFICIENT_HPP
