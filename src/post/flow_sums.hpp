/**
 * @file flow_sums.hpp
 * @brief The integrals behind the flow diagnostics, straight from the field.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief @f$ \int |u|^2 @f$, @f$ \int |\nabla u|^2 @f$ and
 *        @f$ \int (\nabla\cdot u)^2 @f$ over THIS rank's elements, on the
 *        device, in one pass and with nothing stored.
 *
 * At every quadrature point the velocity and its gradient are interpolated
 * from the element's dofs and the three integrands are summed: no assembled
 * form, so no quadrature data (three partially assembled forms for the same
 * values hold 38 numbers per quadrature point: about 10 GB for 64^3 Q3
 * elements). One thread per element, no work arrays.
 *
 * @param u    Velocity (vdim == dim, quads or hexes, consistent across ranks).
 * @param ir   Tensor-product integration rule (borrowed).
 * @param sums Output: the three integrals, in the order above.
 */
void LocalFlowSums(const mfem::ParGridFunction& u,
                   const mfem::IntegrationRule& ir, double sums[3]);

/// Per-element work vectors of LocalFlowSums, kept by a caller that sums
/// often (FlowDiagnostics): the element sums and a column (4 NE numbers; the
/// element copy of the velocity is not kept, see ConvectiveCfl).
struct FlowSumsWork
{
   mfem::Vector out; ///< (3, NE) element sums.
   mfem::Vector col; ///< One column of out.
   /// The velocity's element vector, when lent (an ElementScratch alias of
   /// the restriction's height): used instead of one allocated per call.
   mfem::Vector xe;
};

/// As above, with the per-element vectors in @p work.
void LocalFlowSums(const mfem::ParGridFunction& u,
                   const mfem::IntegrationRule& ir, double sums[3],
                   FlowSumsWork& work);

} // namespace incns
