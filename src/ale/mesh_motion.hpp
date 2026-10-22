/**
 * @file mesh_motion.hpp
 * @brief Moves a high-order mesh by a displacement that a pluggable model
 *        (MeshMotionModel) extends from the moving boundaries into the
 *        domain: the mesh-motion half of an ALE formulation
 *        (docs/mesh_motion.md).
 */
#pragma once

#include "ale/mesh_motion_model.hpp"
#include "mfem.hpp"

#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @brief The mesh's nodes follow x = X + d(X, t), d found by a model on a
 *        copy of the initial (reference) mesh.
 *
 * At construction the mesh is copied (the copy never moves) and the model is
 * built on the copy. Move(t) solves the model with the boundary data at t,
 * interpolates d onto the mesh's own nodal space and sets the nodes to X + d,
 * then invalidates MFEM's cached geometric factors (Mesh::NodesUpdated). The
 * displacement is total, from the reference configuration, so a boundary
 * that returns to its place brings every node back exactly and nothing
 * drifts over a long run.
 *
 * Boundaries: AddDisplacement() for the moving ones (data as functions of
 * the reference coordinates X and t), AddFixed() for any others; a boundary
 * attribute given neither is fixed. There are no sliding boundaries yet.
 *
 * The mesh must have a continuous nodal space for a curved motion; a mesh
 * without nodes is given nodes (Options::geometry_order) at construction, so
 * MeshMotion must be built before anything caches the mesh's geometry.
 * Refinement after construction is not supported (Move() aborts).
 *
 * Mesh velocity: the node displacements of the last Options::history states
 * are kept, so MeshVelocity() gives the BDF derivative of the node positions
 * at the newest state, the discrete velocity a BDF-ALE fluid step uses.
 */
class MeshMotion
{
public:
   /// Settings.
   struct Options
   {
      /// Displacement order of the model; <= 0: the mesh's nodal order.
      int order = 0;
      /// Nodal order given to a mesh without nodes; <= 0: order, or 1 when
      /// order is also <= 0. Ignored when the mesh has nodes.
      int geometry_order = 0;
      /// Node-displacement states kept (the newest included); a mesh
      /// velocity of BDF order q needs q + 1. At least 2.
      int history = 4;
      double t0 = 0.0; ///< Time of the reference configuration.
   };

   /**
    * @brief Copy the mesh as the reference and build the model on the copy.
    *        Collective.
    * @param mesh    The mesh to move (borrowed; must outlive this object).
    * @param opts    Settings.
    * @param factory Builds the model on the reference mesh.
    */
   MeshMotion(mfem::ParMesh& mesh, const Options& opts,
              const MeshMotionModelFactory& factory);

   MeshMotion(const MeshMotion&) = delete;
   MeshMotion& operator=(const MeshMotion&) = delete;

   /// @param attr Boundary attribute held in place.
   void AddFixed(int attr);

   /**
    * @param attr Boundary attribute that moves.
    * @param g    Its displacement d(X, t) (X reference coordinates; borrowed,
    *             must outlive the moves).
    */
   void AddDisplacement(int attr, mfem::VectorCoefficient& g);

   /**
    * @brief Solve the model at time @p t and move the nodes to X + d.
    *        Collective. The first call fixes the boundary set.
    * @param t Time of the new state (greater than the last one).
    * @return Whether the model's solve converged (the nodes move anyway).
    */
   bool Move(double t);

   /**
    * @brief Undo the last Move(): drop the newest state and put the nodes
    *        back at the previous one (a fluid step that is retried with a
    *        smaller step, or a coupling subiteration that moves again).
    *        Collective.
    * @pre MoveCount() > 0 and more than one state kept.
    */
   void Revert();

   /**
    * @brief Write the state (the kept node displacements, their times, the
    *        move count) to @p path, one file per rank (`<path>.r<rank>`), for
    *        a restart at the same rank count. Collective only in that every
    *        rank writes its own file.
    * @param path File prefix.
    */
   void Save(const std::string& path) const;

   /**
    * @brief Restore a state Save() wrote and put the nodes there. The
    *        boundary set must be the same as when it was saved. Collective.
    * @param path File prefix.
    */
   void Load(const std::string& path);

   /**
    * @brief The BDF derivative of the node positions at the newest state,
    *        w = sum_j c_j x^{n+1-j} with the variable-step weights of the
    *        kept states' times (time/multistep_coeffs). Exact for node
    *        motions polynomial in time of degree <= @p order.
    * @param order BDF order (1 <= order <= States() - 1, and <= 3).
    * @param w     Output on the mesh's nodal space (resized).
    */
   void MeshVelocity(int order, mfem::Vector& w) const;

   /**
    * @brief The range of det F = det(dx/dX), the ratio of the current to the
    *        reference Jacobian determinant, over the Gauss-Lobatto points of
    *        every element (nodal order + 1 per direction). min > 0 is a
    *        necessary condition for a valid mesh (sampled, not a bound).
    *        Collective.
    * @param min Smallest det F.
    * @param max Largest det F.
    */
   void JacobianRange(double& min, double& max) const;

   /// @return The node displacement of the current state (nodal space).
   const mfem::ParGridFunction& NodeDisplacement() const { return *disp_; }
   /// @return The reference node positions X (nodal space layout).
   const mfem::Vector& ReferenceNodes() const { return x0_; }
   /// @return The reference mesh (the model's; never moves).
   mfem::ParMesh& ReferenceMesh() { return *ref_; }
   /// @return The model.
   MeshMotionModel& Model() { return *model_; }
   /// @return The displacement order of the model.
   int Order() const { return order_; }
   /// @return The time of the current state.
   double Time() const { return times_.front(); }
   /// @return Moves since construction.
   int MoveCount() const { return moves_; }
   /// @return States kept (the reference state counts until it is dropped).
   int States() const { return static_cast<int>(times_.size()); }

private:
   /// Pass the boundary set to the model: unnamed attributes are fixed.
   void Finalize();

   mfem::ParMesh& mesh_; ///< The moving mesh (borrowed).
   Options opts_;        ///< Settings.
   int order_ = 1;       ///< Model displacement order.
   long sequence_ = 0;   ///< The mesh's sequence at construction.
   std::unique_ptr<mfem::ParMesh> ref_;        ///< The reference mesh.
   std::unique_ptr<MeshMotionModel> model_;    ///< The model (on ref_).
   mfem::Vector x0_;                           ///< Reference node positions.
   std::unique_ptr<mfem::ParGridFunction> disp_; ///< Node displacement.
   std::vector<int> fixed_;                    ///< Fixed attributes.
   std::vector<int> moving_;                   ///< Displaced attributes.
   bool finalized_ = false;                    ///< Whether the set is fixed.
   int moves_ = 0;                             ///< Moves since construction.
   /// Node displacements of the kept states, newest first.
   std::vector<mfem::Vector> history_;
   std::vector<double> times_;                 ///< Their times.
   /// The state the last Move() dropped from a full history (Revert()).
   mfem::Vector dropped_;
   double dropped_time_ = 0.0; ///< Its time.
   bool has_dropped_ = false;  ///< Whether dropped_ holds one.
   /// Gauss-Lobatto rules of JacobianRange().
   mutable mfem::IntegrationRules gll_rules_;
};

} // namespace incns
