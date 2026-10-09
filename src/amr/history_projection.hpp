/**
 * @file history_projection.hpp
 * @brief Project transferred velocity history onto the refined mesh's
 *        discretely divergence-free subspace (docs/design/amr_spec.md, Section 5.4).
 */
#pragma once

#include "bc/boundary_conditions.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/integrator_state.hpp"

namespace incns
{

/// Krylov settings for the projection solves.
struct HistoryProjectionOptions
{
   double rtol = 1e-12;  ///< FGMRES relative tolerance.
   int max_iter = 2000;  ///< FGMRES iteration cap.
   int kdim = 200;       ///< FGMRES restart size.
   int print_level = -1; ///< Solver print level.
};

/**
 * @brief Replace each velocity history level u_j (at time t_j) by the
 *        M-orthogonal projection onto { v : (q, div v) = 0 for every q in the
 *        refined pressure space, v = Dirichlet data at t_j on the boundary }:
 *        @f$ \begin{pmatrix} M & B^T \\ B & 0 \end{pmatrix}
 *        \begin{pmatrix} v \\ \lambda \end{pmatrix} =
 *        \begin{pmatrix} M u_j \\ 0 \end{pmatrix} @f$.
 *
 * Why: a transferred u_j is exactly the old function, divergence-free against
 * the old pressures but not the refined ones; the BDF history puts it on the
 * right-hand side with weight 1/dt, so the pressure would absorb
 * ~||B u_j|| / dt as a transient on the first steps after an event. A field
 * that is already discretely divergence-free (e.g. a polynomial exact
 * solution) is returned unchanged to solver tolerance.
 *
 * Solved with the existing StokesSolver: mass-dominated block (mass_coeff 1,
 * viscosity 1e-12 -- the operator requires nu > 0) on the Cahouet-Chabard
 * path regardless of the run's Schur choice (the mass-path Schur block
 * nu^{-1} M_p is degenerate at nu -> 0; CC with sigma = 1 is the exact Schur
 * model of a mass block). The BC time is left at state.times[0].
 *
 * @param spaces Mixed spaces (on the refined mesh).
 * @param rules  Quadrature source.
 * @param bc     Boundary conditions over spaces.Velocity() (time is moved).
 * @param state  State whose u_hist is projected in place.
 * @param opts   Krylov settings.
 * @return Total FGMRES iterations over the levels.
 */
int ProjectHistoryDivergenceFree(MixedSpaces& spaces, const RuleBook& rules,
                                 BoundaryConditions& bc, IntegratorState& state,
                                 const HistoryProjectionOptions& opts = {});

} // namespace incns
