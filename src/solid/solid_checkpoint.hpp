/**
 * @file solid_checkpoint.hpp
 * @brief A solid solver's state in a checkpoint directory, the flow's
 *        same-np contract (post/checkpoint.hpp): the dynamic state
 *        (LinearElasticity::State, Hyperelasticity::State) and the
 *        displacement as per-rank binary vectors (bitwise), its scalars in
 *        `<name>.yaml` (rank 0).
 */
#pragma once

#include <string>

namespace incns
{

/**
 * @brief Write @p solver's state as `<dir>/<name>.yaml` and
 *        `<dir>/<name>_*.r<rank>.bin`: ExportState() (what a dynamic step
 *        advances from) and the displacement (where a static solve starts).
 *        Collective; @p dir exists.
 * @tparam Solver LinearElasticity or Hyperelasticity.
 * @param dir    Checkpoint directory.
 * @param name   The files' prefix.
 * @param solver The solver.
 */
template <class Solver>
void WriteSolidState(const std::string& dir, const std::string& name,
                     Solver& solver);

/**
 * @brief Restore a state WriteSolidState() wrote into @p solver, built the
 *        same way on the same partition (a dynamic one Initialize()d): its
 *        next Step() or Solve() is the uninterrupted run's. Collective.
 * @tparam Solver LinearElasticity or Hyperelasticity.
 * @param dir    Checkpoint directory.
 * @param name   The files' prefix.
 * @param solver The solver.
 */
template <class Solver>
void ReadSolidState(const std::string& dir, const std::string& name,
                    Solver& solver);

} // namespace incns
