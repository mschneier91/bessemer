/**
 * @file lean_vector_mass.hpp
 * @brief The vector mass whose partial assembly keeps W det J (one number per
 *        point) instead of the mesh's cached Jacobians (nine) at its rule.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief W det J at every point of @p ir on every element, (nq, ne): the
 *        expressions of MFEM's VectorMassIntegrator::AssemblePA, from
 *        Jacobians computed for the call and freed (not the mesh's cache).
 * @param vfes The space (its mesh).
 * @param ir   The rule.
 * @param wdet Output (resized, device-enabled).
 */
void MassWeightDet(mfem::ParFiniteElementSpace& vfes,
                   const mfem::IntegrationRule& ir, mfem::Vector& wdet);

/**
 * @brief mfem::VectorMassIntegrator whose partial assembly reads a kept
 *        W det J (MassWeightDet()) instead of the mesh's cached Jacobians.
 *
 * MFEM's AssemblePA asks the mesh for the Jacobians at the rule (9 numbers
 * per point, cached for the run: 1.7 GiB at Q6 on 42^3 elements) and forms
 * pa_data = C W det J. Here W det J is computed once with its expressions and
 * kept (one number per point), and every AssemblePA (a new mass factor in the
 * momentum block) forms C W det J from it: the same PA data, bitwise, and
 * MFEM's apply and diagonal kernels unchanged. A scalar coefficient or none;
 * anything else, or no rule set, is MFEM's path.
 */
class LeanVectorMassIntegrator : public mfem::VectorMassIntegrator
{
public:
   /// @param wdet W det J at the rule's points (borrowed: must outlive this).
   explicit LeanVectorMassIntegrator(const mfem::Vector& wdet)
      : wdet_(wdet) {}

   /// @param wdet W det J (borrowed). @param q Scalar coefficient (borrowed).
   LeanVectorMassIntegrator(const mfem::Vector& wdet, mfem::Coefficient& q)
      : mfem::VectorMassIntegrator(q), wdet_(wdet) {}

   using mfem::BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const mfem::FiniteElementSpace& fes) override;

private:
   const mfem::Vector& wdet_; ///< W det J, (nq, ne) (borrowed).
};

} // namespace incns
