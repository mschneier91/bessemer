/**
 * @file mesh_size_coefficient.hpp
 * @brief Per-element mesh-size coefficient gamma(x) = scale * h_K.
 */
#pragma once

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
 *
 * LOR use (@p children_per_element > 1): evaluated on a low-order-refined
 * mesh built by mfem::Mesh::MakeRefined (each element of @p mesh split into
 * @p children_per_element sub-elements, stored consecutively in element
 * order), it returns the value of the PARENT high-order element,
 * @c scale * h_{K(e / children)} -- the gamma of the operator being
 * preconditioned, not the sub-element's own (p times smaller) size. Built on
 * the high-order mesh, evaluated with LOR element transformations.
 */
class MeshSizeCoefficient : public mfem::Coefficient
{
public:
   /**
    * @brief Create the coefficient over a mesh.
    * @param mesh  Mesh whose local element sizes are queried (borrowed).
    * @param scale Multiplier @c c_gd: the coefficient value is
    *              scale * h_K + offset.
    * @param children_per_element 1 on @p mesh itself; p^dim when evaluated on
    *        the order-p LOR mesh of @p mesh (see the class note).
    * @param offset A constant added to scale * h_K (e.g. a part of the
    *        grad-div coefficient proportional to the viscosity).
    */
   MeshSizeCoefficient(mfem::ParMesh& mesh, double scale,
                       int children_per_element = 1, double offset = 0.0)
      : mesh_(mesh), scale_(scale), children_(children_per_element),
        offset_(offset)
   {
      MFEM_VERIFY(children_ >= 1, "mesh_size_coefficient: children_per_element "
                  "must be >= 1");
   }

   /**
    * @brief Evaluate @c scale * h + offset at the transformation's element.
    * @param T Element transformation (identifies the local element).
    * @return The per-element value; constant within each element.
    */
   double Eval(mfem::ElementTransformation& T,
               const mfem::IntegrationPoint&) override
   {
      const int parent = T.ElementNo / children_;
      MFEM_ASSERT(parent < mesh_.GetNE(), "mesh_size_coefficient: element "
                  << T.ElementNo << " has no parent (wrong children count?)");
      // Element sizes are computed once per mesh state: GetElementSize
      // builds a Jacobian per call, and this is evaluated at every quadrature
      // point of every form that carries gamma.
      if (sequence_ != mesh_.GetSequence())
      {
         h_.SetSize(mesh_.GetNE());
         for (int e = 0; e < mesh_.GetNE(); ++e) { h_(e) = mesh_.GetElementSize(e); }
         sequence_ = mesh_.GetSequence();
      }
      return scale_ * h_(parent) + offset_;
   }

   /**
    * @brief The value on every element of the mesh (one number each), as
    *        Eval returns it there.
    * @param v Output, resized to the local element count (host write).
    * @pre children_per_element == 1.
    */
   void ElementValues(mfem::Vector& v) const
   {
      MFEM_VERIFY(children_ == 1, "mesh_size_coefficient: ElementValues needs "
                  "children_per_element == 1");
      v.SetSize(mesh_.GetNE());
      double* h = v.HostWrite();
      for (int e = 0; e < mesh_.GetNE(); ++e)
      {
         h[e] = scale_ * mesh_.GetElementSize(e) + offset_;
      }
   }

private:
   mfem::ParMesh& mesh_; ///< Mesh (borrowed).
   double scale_;        ///< Multiplier c_gd.
   int children_;        ///< Sub-elements per element of mesh_ (1 = itself).
   double offset_;       ///< Constant added to scale_ * h_K.
   mfem::Vector h_;      ///< h_K of every element of mesh_ (host; cached).
   long sequence_ = -1;  ///< The mesh sequence h_ was computed for.
};

} // namespace incns
