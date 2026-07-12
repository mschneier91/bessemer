/**
 * @file checkpoint.hpp
 * @brief Checkpoint/restart of the unsteady Stokes march (same-np contract).
 */
#ifndef INCNS_POST_CHECKPOINT_HPP
#define INCNS_POST_CHECKPOINT_HPP

#include "config/parameters.hpp"
#include "time/time_integrator.hpp"

#include <string>

namespace incns
{

/**
 * @brief Writes and restores the integrator's marching state.
 *
 * Contract and format (deliberately simple -- the "Route A" design):
 *  - SAME-NP RESTART ONLY: the mesh is regenerated from the deck (the
 *    partition is deterministic at a fixed rank count), so the checkpoint
 *    holds no mesh -- just per-rank BINARY dumps of the velocity true-dof
 *    history (bitwise-exact; no text-precision loss), the pressure (Krylov
 *    warm start), and one YAML metadata file (rank 0): np, t, next dt,
 *    completed steps, history times, adaptive PI memory, reference scales.
 *    A mismatched rank count or partition is rejected at read (metadata np
 *    check + per-rank size checks).
 *  - Exact continuation: the restored history/times/dt reproduce the
 *    uninterrupted BDF march (multistep state is the whole point -- saving
 *    only u(t) would force a restart ramp); the adaptive controller resumes
 *    with its PI memory.
 *  - One ROLLING checkpoint per directory (each write overwrites).
 *
 * Layout: `<dir>/meta.yaml`, `<dir>/u_hist<j>.r<rank>.bin`,
 * `<dir>/pressure.r<rank>.bin`.
 */
class Checkpoint
{
public:
   /**
    * @brief Write the integrator's current marching state. Collective.
    * @param dir        Checkpoint directory (created if absent).
    * @param integrator Source of the state.
    * @param params     Case parameters (reference scales recorded).
    */
   static void Write(const std::string& dir, StokesTimeIntegrator& integrator,
                     const Parameters& params);

   /**
    * @brief Restore a marching state written by Write(). Collective.
    *
    * Verifies the rank count matches the checkpoint and that the per-rank
    * vector sizes match the (regenerated) velocity space, then calls
    * StokesTimeIntegrator::SetHistory (skipping the startup ramp) and restores
    * the adaptive PI memory when present.
    *
    * @param dir        Checkpoint directory.
    * @param integrator Target integrator (built over the regenerated mesh).
    */
   static void Read(const std::string& dir, StokesTimeIntegrator& integrator);
};

} // namespace incns

#endif // INCNS_POST_CHECKPOINT_HPP
