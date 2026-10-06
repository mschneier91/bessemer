/**
 * @file mixed_spaces.hpp
 * @brief Velocity/pressure finite element spaces and block offsets for the
 *        coupled mixed method.
 */
#pragma once

#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Owns the velocity and pressure parallel finite element spaces for the
 *        coupled (monolithic) mixed method, plus the saddle-point block offsets.
 *
 * The spaces are continuous H1:
 *   - velocity: vector-valued @c Q_{k_u}, vdim = dim, mfem::Ordering::byNODES;
 *   - pressure: scalar @c Q_{k_p}.
 *
 * The default pairing is inf-sup-stable Taylor-Hood, @c k_u = k_p + 1 (library
 * default Q3/Q2). Equal orders (@c k_u == k_p) are inf-sup unstable without
 * pressure stabilization, which is not available in Sprint 1, so the constructor
 * emits a warning.
 */
class MixedSpaces
{
public:
   /**
    * @brief Build the velocity and pressure spaces on a partitioned mesh.
    * @param mesh    Partitioned quad/hex mesh (validated as tensor-product).
    * @param order_u Velocity polynomial order @c k_u (>= 1).
    * @param order_p Pressure polynomial order @c k_p (>= 1).
    */
   MixedSpaces(mfem::ParMesh& mesh, int order_u, int order_p);

   /// @return The (mutable) velocity space.
   mfem::ParFiniteElementSpace& Velocity() { return *vfes_; }
   /// @return The (mutable) pressure space.
   mfem::ParFiniteElementSpace& Pressure() { return *pfes_; }
   /// @return The velocity space.
   const mfem::ParFiniteElementSpace& Velocity() const { return *vfes_; }
   /// @return The pressure space.
   const mfem::ParFiniteElementSpace& Pressure() const { return *pfes_; }

   /// @return The velocity polynomial order @c k_u.
   int OrderU() const { return order_u_; }
   /// @return The pressure polynomial order @c k_p.
   int OrderP() const { return order_p_; }
   /// @return The spatial dimension.
   int Dim() const { return dim_; }

   /**
    * @brief Local (per-rank) true-dof block offsets @c [0, n_u, n_u + n_p].
    * @return Array of size 3 for building BlockVectors/BlockOperators over the
    *         coupled velocity-pressure system.
    */
   const mfem::Array<int>& BlockTrueOffsets() const { return block_true_offsets_; }

   /**
    * @brief After the mesh changed (adaptive refinement or rebalancing):
    *        update both spaces and recompute the block offsets.
    *
    * MFEM then holds a transfer operator per space: call Update() on every
    * GridFunction living on these spaces BEFORE UpdatesFinished(), and after
    * EACH mesh change (refinement and rebalancing are two changes).
    */
   void Update();

   /// Release the spaces' transfer operators (every GridFunction updated).
   void UpdatesFinished();

   /// @return Global velocity true-dof count (collective reduction).
   HYPRE_BigInt GlobalVelocityTDofs() const { return vfes_->GlobalTrueVSize(); }
   /// @return Global pressure true-dof count (collective reduction).
   HYPRE_BigInt GlobalPressureTDofs() const { return pfes_->GlobalTrueVSize(); }

private:
   int dim_;      ///< Spatial dimension.
   int order_u_;  ///< Velocity order k_u.
   int order_p_;  ///< Pressure order k_p.
   std::unique_ptr<mfem::H1_FECollection> fec_u_;         ///< Velocity collection.
   std::unique_ptr<mfem::H1_FECollection> fec_p_;         ///< Pressure collection.
   std::unique_ptr<mfem::ParFiniteElementSpace> vfes_;    ///< Velocity space.
   std::unique_ptr<mfem::ParFiniteElementSpace> pfes_;    ///< Pressure space.
   /// Local (per-rank) true-dof block offsets.
   mfem::Array<int> block_true_offsets_;
};

} // namespace incns
