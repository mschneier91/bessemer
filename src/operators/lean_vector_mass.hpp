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
 * @brief The 3D vector mass's apply on E-vectors, y += B^T D B x for each of
 *        the three components, with a thread per quadrature point (a Q1D^3
 *        block per element, the components together): the same sums as
 *        MFEM's kernel in another order (roundoff).
 * @param d1d Dofs per direction.
 * @param q1d Points per direction (= d1d; 2 to 8).
 * @param ne  Elements.
 * @param b   Basis at the points, (q1d, d1d).
 * @param d   C W det J at the points, (q1d^3, ne).
 * @param x   Input E-vector, (d1d^3, 3, ne).
 * @param y   Output E-vector (added to).
 * @return False, and nothing done, for an unsupported size.
 */
bool VectorMassApply3DPoints(int d1d, int q1d, int ne,
                             const mfem::Array<double>& b, const mfem::Vector& d,
                             const mfem::Vector& x, mfem::Vector& y);

/**
 * @brief mfem::VectorMassIntegrator whose partial assembly reads a kept
 *        W det J (MassWeightDet()) instead of the mesh's cached Jacobians.
 *
 * MFEM's AssemblePA asks the mesh for the Jacobians at the rule (9 numbers
 * per point, cached for the run: 1.7 GiB at Q6 on 42^3 elements) and forms
 * pa_data = C W det J. Here W det J is computed once with its expressions and
 * kept (one number per point), and every AssemblePA (a new mass factor in the
 * momentum block) forms C W det J from it: the same PA data, bitwise, and
 * MFEM's diagonal kernel. A scalar coefficient or none; anything else, or no
 * rule set, is MFEM's path. On a device the 3D apply is
 * VectorMassApply3DPoints (MFEM's kernel took ~10 % of the memory bandwidth
 * at Q6); the host keeps MFEM's apply, so CPU results are unchanged.
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

   /**
    * @brief y += M x on E-vectors: VectorMassApply3DPoints on a device (3D,
    *        a scalar coefficient, a supported size), MFEM's apply otherwise.
    * @param x Input E-vector.
    * @param y Output E-vector (added to).
    */
   void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const override;

private:
   const mfem::Vector& wdet_; ///< W det J, (nq, ne) (borrowed).
};

} // namespace incns
