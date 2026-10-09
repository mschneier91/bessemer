/**
 * @file cfl.hpp
 * @brief Directional convective CFL number of a velocity field on its mesh,
 *        Nek5000's definition (the stability ceiling / step control for the
 *        explicit transport term).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief The convective CFL rate, Nek5000's definition (`compute_cfl`):
 *        @f$ c = \max_K \max_{\text{GLL nodes } n} \sum_d
 *        |(J_n^{-1} u_n)_d| \,/\, \Delta\xi_d(n) @f$,
 *        so that the CFL number of a step is @f$ c\,\Delta t @f$.
 *
 * Evaluated at the k_u+1 Gauss-Lobatto points per direction (the H1 nodes);
 * @f$ J^{-1}u @f$ is the velocity in reference coordinates and
 * @f$ \Delta\xi_d(n) @f$ the LOCAL reference spacing of the GLL points at the
 * node's index along direction d: one-sided at the element's end nodes, half
 * the central difference at interior nodes (Nek's `getdr`). Reference and
 * spacing are both on [0,1], so the ratio equals Nek's on [-1,1]. Directional:
 * a cell stretched along the flow allows a larger step than one stretched
 * across it.
 *
 * Replaced (2026-10-07, human decision: "whatever is most accurate") a
 * uniform @f$ k_u^2 \max_q \sum_d |(J^{-1}u)_d| @f$ at Gauss-Legendre points,
 * which is ~2.5x this value at element edges (Q3: 9 vs 3.62 per unit |u|/h)
 * and ~3.3x in element interiors. Global over ranks. Built per mesh (it
 * caches the mesh's Jacobians); rebuild after refinement.
 */
class ConvectiveCfl
{
public:
   /**
    * @param vfes  Velocity space (vector H1, vdim = dim, tensor elements).
    * @param rules Quadrature source (its GLL rule with k_u + 1 points per
    *              direction is used); must outlive this object and the mesh.
    */
   ConvectiveCfl(const mfem::ParFiniteElementSpace& vfes, const RuleBook& rules);

   /**
    * @brief The rate c above (collective).
    * @param u Velocity on the space given at construction.
    * @return c >= 0; the step's CFL number is c * dt.
    */
   double Rate(const mfem::ParGridFunction& u) const;

   /**
    * @brief Rate() and where it peaks (collective; diagnostics): the centre
    *        of the element with the largest rate, on every rank.
    * @param u     Velocity on the space given at construction.
    * @param where Output: that element's centre (size dim).
    * @return The rate c, as Rate().
    */
   double RateAndLocation(const mfem::ParGridFunction& u,
                          mfem::Vector& where) const;

   /**
    * @brief Nek's inverse local reference spacing at the k+1 GLL points on
    *        [0,1] (index 0..k).
    * @param k Polynomial order.
    * @return 1 / Delta-xi(i): 1/(z1 - z0) at the ends, 2/(z_{i+1} - z_{i-1})
    *         inside.
    */
   static mfem::Vector InverseNodeSpacing(int k);

private:
   const mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int dim_ = 0;                             ///< Spatial dimension.
   int ne_ = 0;                              ///< Local element count.
   int order_ = 0;                           ///< Velocity order k_u.
   const mfem::IntegrationRule* ir_ = nullptr; ///< GLL rule (the nodes).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic restriction.
   const mfem::QuadratureInterpolator* qi_ = nullptr; ///< Values at points.
   const mfem::GeometricFactors* geom_ = nullptr;     ///< Jacobians.
   mfem::Vector inv_dxi_;    ///< 1 / Delta-xi per 1D node index (device).
   mutable mfem::Vector ue_; ///< Velocity E-vector buffer.
   mutable mfem::Vector uq_; ///< Velocity at the nodes.
   mutable mfem::Vector rate_; ///< Per-element rate.
};

} // namespace incns
