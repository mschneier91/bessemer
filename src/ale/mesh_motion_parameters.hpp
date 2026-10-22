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

/// How a moving-mesh flow refreshes its solvers after each move (deck
/// `mesh_motion.refresh`).
enum class AleRefresh
{
   /// Keep the solvers and re-assemble their operators on the moved mesh;
   /// the AMG hierarchies are rebuilt every `amg_interval` moves.
   InPlace,
   /// Build every solver anew after each move.
   Rebuild
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
   /// The pseudo-solid's Poisson ratio. 0.3 suits most meshes; 0.4-0.45
   /// resists compression at the cost of more shear and CG iterations; near
   /// 0.5 the displacement formulation locks (docs/mesh_motion.md §1).
   double poisson_ratio = 0.3;
   /// Jacobian stiffening exponent chi (>= 0): 1 for a mesh graded toward a
   /// moving body or wall (keeps its small elements nearly rigid, pushes the
   /// distortion into the large ones), 0 for the best overall det F, 2 only
   /// when the near-body elements must stay almost undeformed. No effect on a
   /// uniform mesh. Measurements: docs/mesh_motion.md §1, "Choosing the
   /// settings".
   double stiffening = 1.0;
   /// Displacement and geometry order; <= 0: discretization.order_u. A mesh
   /// that already has nodes (a curved geometry) keeps its own order.
   int order = 0;
   int steps = 1;         ///< Moves, evenly spaced in (0, time.t_final].
   double rtol = 1e-10;   ///< CG relative tolerance.
   double atol = 0.0;     ///< CG absolute tolerance.
   int max_iter = 2000;   ///< CG iteration cap.
   ElasticityPC preconditioner = ElasticityPC::LorAmg; ///< CG preconditioner.
   /// Earlier solutions kept for reuse (0: none): boundary data in the span
   /// of earlier data (rigid or modal motion) need no solve.
   int basis_size = 12;
   /// A moving-mesh flow: how the solvers follow a move.
   AleRefresh refresh = AleRefresh::InPlace;
   /// A moving-mesh flow, refresh in place: the AMG hierarchies and the
   /// Chebyshev eigenvalue estimates are rebuilt every this many moves.
   int amg_interval = 10;

   /// @param order_u The fluid's velocity order. @return The order in effect.
   int Order(int order_u) const { return order > 0 ? order : order_u; }
};

} // namespace incns
