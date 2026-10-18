/**
 * @file cfl.hpp
 * @brief Directional convective CFL number of a velocity field on its mesh,
 *        Nek5000's definition (the stability ceiling / step control for the
 *        explicit transport term).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <utility>
#include <vector>

namespace incns
{

/**
 * @brief Abort with a clear message when a CFL rate is not finite: the
 *        velocity holds a NaN or an Inf (the rate kernel reports a NaN node as
 *        an infinite rate), so the run has diverged. Collective callers pass
 *        the same (global) rate on every rank.
 * @param rate A rate from ConvectiveCfl::Rate or MaxRate.
 * @param t    The time of the velocity it was computed from (for the message).
 */
void RequireFiniteRate(double rate, double t);

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
 * and ~3.3x in element interiors. Global over ranks. Built per mesh; rebuild
 * after refinement. Nothing per point is kept: the Jacobians at the nodes
 * are formed inside the rate kernel (straight-sided elements, and curved ones
 * in 3D), or else computed at every call and freed (1.2 GB at 64^3 Q3).
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
    * @brief The largest Rate() of several velocities (collective), with the
    *        Jacobians and the element vector allocated once for all of them.
    * @param us Velocities on the space given at construction (>= 1).
    * @return max over us of Rate(), the same value.
    */
   double MaxRate(const std::vector<const mfem::ParGridFunction*>& us) const;

   /**
    * @brief MaxRate of velocities that are the solution at @p times (a step's
    *        history levels), each time's rate kept and reused while the space
    *        is unchanged (its sequence number): a level is evaluated once --
    *        by the step-size choice, or the first OIFS step that sees it --
    *        instead of at every step it stays in the history. The max of the
    *        levels' maxima is MaxRate's value exactly.
    * @param us    The velocities.
    * @param times Their times: the keys, one velocity per time.
    * @return The largest rate; collective.
    */
   double MaxRateAt(const std::vector<const mfem::ParGridFunction*>& us,
                    const std::vector<double>& times) const;

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

   /**
    * @brief Use the on-the-fly evaluation where it applies (the default):
    *        straight-sided elements, or curved ones in 3D (mesh orders up to
    *        9), the GLL nodal basis, byNODES, ElementRestrictions. false
    *        forces the element copy of the velocity and MFEM's Jacobians at
    *        the nodes (tests compare the two).
    * @param on Whether to evaluate on the fly where possible.
    */
   void SetOnTheFly(bool on)
   {
      fly_ = on && fly_able_;
      curved_fly_ = on && curved_able_;
   }
   /// @return Whether the rate is evaluated on the fly.
   bool OnTheFly() const { return fly_ || curved_fly_; }

private:
   const mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   int dim_ = 0;                             ///< Spatial dimension.
   int ne_ = 0;                              ///< Local element count.
   int order_ = 0;                           ///< Velocity order k_u.
   const mfem::IntegrationRule* ir_ = nullptr; ///< GLL rule (the nodes).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic restriction.
   const mfem::QuadratureInterpolator* qi_ = nullptr; ///< Values at points.
   mfem::Vector inv_dxi_;    ///< 1 / Delta-xi per 1D node index (device).
   /// The velocity basis is the GLL nodal one, so the element dofs ARE the
   /// values at the GLL points and nothing is interpolated.
   bool nodal_gll_ = false;
   /// Straight-sided elements (linear mesh nodes), nodal_gll_, byNODES and an
   /// ElementRestriction: the rate is evaluated on the fly, the velocity read
   /// through the restriction's gather map and each node's Jacobian formed
   /// from the element's 2^dim vertices, with nothing allocated per call.
   /// The element copy of the velocity and MFEM's Jacobians at the nodes were
   /// ~42 B/DOF at Q6 and sat on the run's memory peak (the allocation
   /// census, 2026-10-17).
   bool fly_able_ = false;
   bool fly_ = false;
   /// Curved elements in 3D (mesh nodes of order 2-9, a tensor basis), with
   /// the conditions above: on the fly too, each node's Jacobian contracted
   /// from the element's mesh nodes inside the kernel in the order of MFEM's
   /// Derivatives3D, so the rate is the stored path's bit for bit on a GPU
   /// (on the CPU build to roundoff: MFEM there is compiled with the
   /// target's FMA, bessemer without). That path
   /// allocated the velocity's element copy, MFEM's Jacobians at the nodes
   /// and its element copy of the mesh nodes for every call: 2.96 GB at 72k
   /// Q6 elements, 2.26 GB above the rest of the step on the curved cylinder
   /// with hanging nodes (the allocation trace, 2026-10-18).
   bool curved_able_ = false;
   bool curved_fly_ = false;
   const mfem::Array<int>* gather_ = nullptr; ///< The restriction's map.
   int ndofs_ = 0;      ///< Scalar local dofs: the component stride.
   mfem::Vector xv_;    ///< Element vertices, (2^dim, dim, NE) (device).
   mfem::Vector zgll_;  ///< The k+1 GLL points on [0, 1] (device).
   /// The mesh nodes' gather map, values and their layout (curved path).
   const mfem::Array<int>* ngather_ = nullptr;
   const mfem::GridFunction* nodes_ = nullptr;
   int node_stride_ = 1; ///< Value c of node j at j * node_stride_ ...
   int comp_stride_ = 0; ///< ... + c * comp_stride_.
   /// The mesh basis and its derivative at the GLL nodes (curved path).
   const mfem::DofToQuad* nmaps_ = nullptr;
   mutable mfem::Vector rate_; ///< Per-element rate.
   /// rate_ from @p u on curved elements (curved_fly_).
   void CurvedRates(const mfem::ParGridFunction& u) const;
   /// The rate of each of @p us (each a max over the ranks), with one
   /// evaluation of the Jacobians for all of them.
   void Rates(const std::vector<const mfem::ParGridFunction*>& us,
              std::vector<double>& rates) const;
   /// (time, rate) of the levels MaxRateAt has seen, newest few kept.
   mutable std::vector<std::pair<double, double>> level_rates_;
   mutable long level_seq_ = -1; ///< The space sequence they belong to.
};

} // namespace incns
