/**
 * @file mesh_motion_parameters.hpp
 * @brief Settings of a mesh deformation (deck section `mesh_motion`, equation
 *        mesh_motion): the model that extends the boundary displacements into
 *        the domain, its order and linear solver, and the moves to make.
 */
#pragma once

#include "solid/elasticity_parameters.hpp"

namespace incns
{

/// The model of a deck-driven mesh deformation (deck `mesh_motion.model`).
enum class MeshMotionModelType
{
   /// Linear elasticity with Jacobian stiffening (LinearElasticMotion).
   LinearElastic
};

/**
 * @brief How a deck deforms its mesh: the boundary displacements are the
 *        `displacement` groups of boundary_conditions, every other boundary is
 *        `fixed`; the moves are at t_k = k time.t_final / steps.
 */
struct MeshMotionParameters
{
   /// The model.
   MeshMotionModelType model = MeshMotionModelType::LinearElastic;
   double poisson_ratio = 0.3; ///< The pseudo-solid's Poisson ratio.
   double stiffening = 1.0;    ///< Jacobian stiffening exponent chi (>= 0).
   /// Displacement and geometry order; <= 0: discretization.order_u. A mesh
   /// that already has nodes (a curved geometry) keeps its own order.
   int order = 0;
   int steps = 1;         ///< Moves, evenly spaced in (0, time.t_final].
   double rtol = 1e-10;   ///< CG relative tolerance.
   double atol = 0.0;     ///< CG absolute tolerance.
   int max_iter = 2000;   ///< CG iteration cap.
   ElasticityPC preconditioner = ElasticityPC::LorAmg; ///< CG preconditioner.

   /// @param order_u The fluid's velocity order. @return The order in effect.
   int Order(int order_u) const { return order > 0 ? order : order_u; }
};

} // namespace incns
