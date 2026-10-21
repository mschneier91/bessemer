/**
 * @file mesh_motion_case.hpp
 * @brief A deck-driven mesh deformation (equation mesh_motion): MeshMotion
 *        with the deck's model, the boundary groups applied (displacement:
 *        constant or an expression of x, y, z, t; fixed), the moves, mesh
 *        quality, the run summary, the history CSV and ParaView output of the
 *        deformed mesh (docs/mesh_motion.md).
 */
#pragma once

#include "ale/mesh_motion.hpp"
#include "config/parameters.hpp"
#include "mfem.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @param mesh  A mesh.
 * @param order Gauss-Lobatto points per direction minus one.
 * @return The smallest element Jacobian ratio min det J / max det J over the
 *         elements (1 for a parallelogram or parallelepiped; <= 0 for an
 *         inverted element), sampled at the Gauss-Lobatto points.
 *         Collective.
 */
double MinJacobianRatio(mfem::ParMesh& mesh, int order);

/**
 * @brief The mesh-motion counterpart of ElasticityCase: deforms the case
 *        mesh by the deck's prescribed boundary displacements.
 *
 * Every real boundary is in exactly one group, of type displacement (moved
 * to X + g(X, t)) or fixed. The moves are at t_k = k t_final / steps;
 * expressions see the reference coordinates X as (x, y, z) and t_k.
 */
class MeshMotionCase
{
public:
   /**
    * @brief Build MeshMotion and apply the boundary groups. Collective.
    *        The mesh gets nodes of the motion's order if it has none.
    * @param mesh   The case mesh (borrowed; moved in place by Run()).
    * @param params Normalized parameters with equation mesh_motion.
    */
   MeshMotionCase(mfem::ParMesh& mesh, const Parameters& params);

   /**
    * @brief Make the deck's moves; after each one, the mesh quality, a
    *        history row (`<path>/<name>_history.csv` when output.history) and
    *        ParaView output (output.enabled: the reference at cycle 0, then
    *        every output.interval moves and the last). Collective.
    * @return Whether every solve converged and the mesh stayed valid
    *         (min det F > 0).
    */
   bool Run();

   /// @return The mesh motion.
   MeshMotion& Motion() { return *motion_; }

   /**
    * @return The largest |x - X - g(X, t)| over the nodes of the displaced
    *         boundaries at the current state: how well the moved mesh
    *         reproduces the prescribed displacement. Collective.
    */
   double BoundaryError() const;

   /// @return The largest node displacement |x - X|. Collective.
   double MaxDisplacement() const;

   /**
    * @brief Write the run summary (JSON) on rank 0. Collective.
    * @param path   File path.
    * @param status "running", "ok" or "diverged".
    */
   void WriteSummary(const std::string& path, const std::string& status);

private:
   /// d(X, t) of a displaced group, evaluated at a reference point.
   using Displacement = std::function<void(const mfem::Vector&, double,
                                           mfem::Vector&)>;

   /// A displaced group: its attributes and data.
   struct Moving
   {
      std::vector<int> attributes; ///< Boundary attributes.
      Displacement g;              ///< The prescribed displacement.
   };

   /// Project det F onto detf_ (ParaView output).
   void UpdateDetF();

   mfem::ParMesh& mesh_;      ///< The mesh (borrowed).
   const Parameters& params_; ///< The deck's parameters (borrowed).
   std::unique_ptr<MeshMotion> motion_; ///< The motion.
   /// The boundary data (outlive the solves).
   std::vector<std::unique_ptr<mfem::VectorCoefficient>> coeffs_;
   std::vector<Moving> moving_; ///< The displaced groups.
   std::unique_ptr<mfem::L2_FECollection> detf_fec_; ///< det F's space.
   std::unique_ptr<mfem::ParFiniteElementSpace> detf_fes_; ///< det F's space.
   std::unique_ptr<mfem::ParGridFunction> detf_; ///< det F (output field).
   double ref_quality_ = 1.0;  ///< MinJacobianRatio of the reference mesh.
   double quality_ = 1.0;      ///< MinJacobianRatio of the current mesh.
   double detf_min_ = 1.0;     ///< min det F of the current state.
   double detf_max_ = 1.0;     ///< max det F of the current state.
   double detf_worst_ = 1.0;   ///< Smallest min det F over the moves.
   int iterations_ = 0;        ///< CG iterations of the last move.
   long long iterations_total_ = 0; ///< CG iterations of every move.
   double solve_seconds_ = 0.0; ///< Wall time of Run().
   bool solved_ = false;       ///< Whether Run() made a move.
   bool converged_ = true;     ///< Whether every solve converged.
};

} // namespace incns
