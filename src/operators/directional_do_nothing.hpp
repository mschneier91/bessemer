/**
 * @file directional_do_nothing.hpp
 * @brief The directional do-nothing (DDN) outflow term of Braack & Mucha
 *        (J. Comput. Math. 32 (2014) 507-521) as a boundary-face integrator.
 */
#pragma once

#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief Boundary-face linear form of the directional do-nothing condition:
 *        @f$ \phi \mapsto -\beta \int_{S_1} (u\cdot n)_-\, u\cdot\phi \, ds
 *        @f$, @f$ (u\cdot n)_- = \min(u\cdot n, 0) @f$, for a GIVEN velocity
 *        @c u (the term is nonlinear in u; the caller supplies the state).
 *
 * Braack & Mucha's condition on an outflow boundary @f$ S_1 @f$ is
 * @f$ T(u,p)\,n - \tfrac12 (u\cdot n)_-\, u = 0 @f$ with @f$ T = \nu\nabla u
 * - pI @f$ (the full gradient: exactly bessemer's natural condition for the
 * vector-Laplacian viscous term), i.e. the weak form gains
 * @f$ -\tfrac12\int_{S_1}(u\cdot n)_-\,u\cdot\phi @f$ on its LEFT-hand side
 * (beta = 1/2, their eq. 2.9). Where the flow leaves the domain, @f$ u\cdot n
 * \ge 0 @f$, the term vanishes and the condition IS the classical do-nothing
 * one. Where it enters (backflow), it cancels exactly the energy the
 * convective boundary flux @f$ \tfrac12\int (u\cdot n)|u|^2 @f$ would inject,
 * so @f$ ((u\cdot\nabla)u, u) - \tfrac12\int_{S_1}(u\cdot n)_-|u|^2 @f$ keeps
 * only the outflow part @f$ \tfrac12\int_{S_1}(u\cdot n)_+|u|^2 \ge 0 @f$.
 *
 * Host-side assembly over the marked boundary faces only -- a few faces,
 * evaluated once per explicit convection apply; no device kernel. The
 * outward normal comes from the boundary face transformation (MFEM's
 * convention: from element 1 outward), so it does not depend on how the mesh
 * orients its boundary elements. Element vectors are component-major
 * (@c [c * ndof + i]), matching MFEM's element vdofs. Convection applies the
 * term from FaceData cached per mesh, not through AssembleRHSElementVect.
 *
 * @warning Reads @c u on the host: the caller syncs it (HostRead) first.
 */
class DirectionalDoNothingIntegrator : public mfem::LinearFormIntegrator
{
public:
   /**
    * @param u    Velocity the term is evaluated at (borrowed; vector H1 with
    *             vdim = dim; must outlive this integrator).
    * @param beta Coefficient of the term (Braack & Mucha: 1/2).
    */
   DirectionalDoNothingIntegrator(const mfem::GridFunction& u, double beta = 0.5);

   /**
    * @brief Not a domain term: aborts (use as a boundary-FACE integrator).
    * @param el     Unused.
    * @param Tr     Unused.
    * @param elvect Unused.
    */
   void AssembleRHSElementVect(const mfem::FiniteElement& el,
                               mfem::ElementTransformation& Tr,
                               mfem::Vector& elvect) override;

   /**
    * @brief The boundary-face element vector of the term.
    * @param el     Finite element of the face's (only) neighbour element.
    * @param Tr     Boundary face transformations (Elem1 = the neighbour).
    * @param elvect Output, size ndof * dim, component-major.
    */
   void AssembleRHSElementVect(const mfem::FiniteElement& el,
                               mfem::FaceElementTransformations& Tr,
                               mfem::Vector& elvect) override;

   /// @cond INTERNAL
   using mfem::LinearFormIntegrator::AssembleRHSElementVect;
   /// @endcond

   /**
    * @brief One boundary face's quadrature data, to apply the term without
    *        its transformation (Convection caches it per mesh): per point of
    *        AssembleRHSElementVect's face rule, its weight, the area-scaled
    *        outward normal and the element's shapes at the given dofs.
    * @param el      Finite element of the face's neighbour element.
    * @param Tr      Boundary face transformations (Elem1 = the neighbour).
    * @param pos     The element dofs whose shapes to keep (the face's own:
    *                an H1 element's other shapes vanish on the face).
    * @param weights Output, appended: one per point.
    * @param normals Output, appended: dim per point.
    * @param shapes  Output, appended: pos.Size() per point.
    * @return The number of points.
    */
   int FaceData(const mfem::FiniteElement& el,
                mfem::FaceElementTransformations& Tr,
                const mfem::Array<int>& pos, std::vector<double>& weights,
                std::vector<double>& normals, std::vector<double>& shapes);

   /// @return The coefficient of the term (Braack & Mucha: 1/2).
   double Beta() const { return beta_; }

private:
   /**
    * @brief The face rule: the one set (SetIntRule), else 3k on the face.
    * @param el Finite element of the face's neighbour element.
    * @param Tr Boundary face transformations.
    * @return The rule.
    */
   const mfem::IntegrationRule& FaceRule(const mfem::FiniteElement& el,
                                         mfem::FaceElementTransformations& Tr) const;

   const mfem::GridFunction& u_; ///< Velocity the term is evaluated at.
   double beta_;                 ///< Coefficient (1/2 for DDN).
   mfem::Array<int> vdofs_;      ///< Scratch: element vdofs of u.
   mfem::Vector u_el_;           ///< Scratch: element dof values of u.
   mfem::Vector shape_;          ///< Scratch: basis values at a point.
   mfem::Vector nor_;            ///< Scratch: area-scaled outward normal.
   mfem::Vector u_q_;            ///< Scratch: u at a quadrature point.
};

} // namespace incns
