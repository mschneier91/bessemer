/**
 * @file element_scaled_coefficient.hpp
 * @brief A coefficient constant on each element: a value times a per-element
 *        factor (the Jacobian stiffening of mesh motion, docs/mesh_motion.md).
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief value * scale(K) on each element K of a mesh.
 *
 * LOR use (@p children_per_element > 1): evaluated on the low-order-refined
 * mesh MFEM's LOR discretization builds (each element split into
 * children_per_element sub-elements, stored consecutively in element order,
 * as MeshSizeCoefficient relies on), it returns the PARENT element's value,
 * so the LOR operator carries the coefficient of the operator it
 * preconditions.
 */
class ElementScaledCoefficient : public mfem::Coefficient
{
public:
   /**
    * @param value The value multiplied by the element factor.
    * @param scale One factor per local element (borrowed; host readable;
    *              must outlive the coefficient).
    * @param children_per_element 1 on the mesh itself; k^dim when evaluated
    *        on the order-k LOR mesh.
    */
   ElementScaledCoefficient(double value, const mfem::Vector& scale,
                            int children_per_element = 1)
      : value_(value), scale_(scale), children_(children_per_element)
   {
      MFEM_VERIFY(children_ >= 1, "element_scaled_coefficient: "
                  "children_per_element must be >= 1");
   }

   /**
    * @param T Element transformation (identifies the local element).
    * @return value * scale of the (parent) element.
    */
   double Eval(mfem::ElementTransformation& T,
               const mfem::IntegrationPoint&) override
   {
      const int parent = T.ElementNo / children_;
      MFEM_ASSERT(parent < scale_.Size(), "element_scaled_coefficient: element "
                  << T.ElementNo << " has no factor (wrong children count?)");
      return value_ * scale_.HostRead()[parent];
   }

private:
   double value_;               ///< The value.
   const mfem::Vector& scale_;  ///< Per-element factors (borrowed).
   int children_;               ///< Sub-elements per element (LOR).
};

} // namespace incns
