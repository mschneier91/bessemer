/**
 * @file mesh_motion_factory.hpp
 * @brief The deck's mesh-motion model (mesh_motion.model and its settings) as
 *        a MeshMotion factory, for the mesh-motion case and moving-mesh flows.
 */
#pragma once

#include "ale/mesh_motion_model.hpp"
#include "ale/mesh_motion_parameters.hpp"

namespace incns
{

/**
 * @brief The factory of the deck's model: LinearElasticMotion or the
 *        neo-Hookean HyperelasticMotion, with the deck's Poisson ratio,
 *        stiffening and CG settings.
 * @param mm The deck's mesh_motion section.
 * @return The factory.
 */
MeshMotionModelFactory DeckMeshMotionFactory(const MeshMotionParameters& mm);

} // namespace incns
