/**
 * @file running_moments.hpp
 * @brief The in-place weighted update of nodal running means and second
 *        moments: one device pass over the nodes (post/flow_statistics).
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Fold one weighted sample into running nodal means and covariances.
 *
 * At every node i, with x the sample's components there and m the means:
 *   d = x - m,   m <- m + f d,
 *   N_a <- g (N_a + f d_a d_a)           (a < vdim),
 *   S_k <- g (S_k + f d_a d_b)           (k = 0: (0, 1); in 3D 1: (0, 2), 2: (1, 2)),
 * the weighted Welford update with f = w / W' and g = W / W' (W the weight
 * so far, w the sample's, W' = W + w): m, N and S stay the weighted means
 * and covariances. One thread per node; nothing is shared or reduced, so the
 * result is bitwise deterministic.
 *
 * Component a of node i is at a * stride_c + i * stride_n in x, m, N and S
 * (by nodes: stride_c = n, stride_n = 1; by vdim: 1, vdim). In 2D S's
 * component 1 is left alone.
 *
 * @param n        Nodes.
 * @param vdim     Components per node (1, 2 or 3).
 * @param stride_c Stride between the components of a node.
 * @param stride_n Stride between nodes.
 * @param f        w / W' (1 for the first sample: m becomes x).
 * @param g        W / W' (0 for the first sample: N and S become 0).
 * @param x        The sample.
 * @param m        The means, updated.
 * @param normal   The variances N, updated; null for none (then also no S).
 * @param shear    The covariances S, updated; null for none. Needs vdim >= 2.
 */
void UpdateRunningMoments(int n, int vdim, int stride_c, int stride_n,
                          double f, double g, const mfem::Vector& x,
                          mfem::Vector& m, mfem::Vector* normal,
                          mfem::Vector* shear);

} // namespace incns
