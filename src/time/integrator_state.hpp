/**
 * @file integrator_state.hpp
 * @brief Everything needed to continue a march exactly: the state an AMR event
 *        carries across a mesh change and a checkpoint writes to disk.
 */
#pragma once

#include "time/adaptive_controller.hpp"
#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief The marching state of a StokesTimeIntegrator.
 *
 * One definition, two transports: Checkpoint writes it to disk (same-np
 * restart), and an AMR event holds it in memory while the mesh is refined
 * (each velocity vector is transferred to the refined space in between).
 */
struct IntegratorState
{
   /// Velocity true-dof history, NEWEST first (2 levels; 3 in adaptive or
   /// BDF3 mode).
   std::vector<mfem::Vector> u_hist;
   std::vector<double> times; ///< Times of u_hist, strictly decreasing.
   int completed_steps = 0;   ///< Accepted steps so far.
   double next_dt = 0.0;      ///< Step size to attempt next.
   /// The solver's own pressure variable on true dofs (the Bernoulli head
   /// P = p + 1/2|u|^2 in the rotational form): the Krylov warm start.
   mfem::Vector pressure;
   bool has_controller = false;        ///< Adaptive mode.
   AdaptiveController::State controller; ///< Adaptive controller memory.
};

} // namespace incns
