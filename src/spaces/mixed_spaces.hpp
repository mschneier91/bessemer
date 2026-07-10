#ifndef INCNS_SPACES_MIXED_SPACES_HPP
#define INCNS_SPACES_MIXED_SPACES_HPP

#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Owns the velocity and pressure parallel finite element spaces for the mixed
/// (coupled) method, plus the saddle-point block offsets.
///
///   velocity : vector H1, Q_{k_u}, vdim = dim   (Ordering::byNODES)
///   pressure : scalar H1, Q_{k_p}
///
/// Default pairing is inf-sup-stable Taylor-Hood, k_u = k_p + 1 (library default
/// Q3/Q2). Equal orders (k_u == k_p) are inf-sup unstable without pressure
/// stabilization -- not available in Sprint 1 -- so the constructor warns.
class MixedSpaces
{
public:
   MixedSpaces(mfem::ParMesh& mesh, int order_u, int order_p);

   mfem::ParFiniteElementSpace& Velocity() { return *vfes_; }
   mfem::ParFiniteElementSpace& Pressure() { return *pfes_; }
   const mfem::ParFiniteElementSpace& Velocity() const { return *vfes_; }
   const mfem::ParFiniteElementSpace& Pressure() const { return *pfes_; }

   int OrderU() const { return order_u_; }
   int OrderP() const { return order_p_; }
   int Dim() const { return dim_; }

   /// Local (per-rank) true-dof block offsets [0, n_u, n_u + n_p], sized 3, for
   /// building BlockVectors / BlockOperators over the coupled system.
   const mfem::Array<int>& BlockTrueOffsets() const { return block_true_offsets_; }

   /// Global true-dof counts (collective reductions).
   HYPRE_BigInt GlobalVelocityTDofs() const { return vfes_->GlobalTrueVSize(); }
   HYPRE_BigInt GlobalPressureTDofs() const { return pfes_->GlobalTrueVSize(); }

private:
   int dim_;
   int order_u_;
   int order_p_;
   std::unique_ptr<mfem::H1_FECollection> fec_u_;
   std::unique_ptr<mfem::H1_FECollection> fec_p_;
   std::unique_ptr<mfem::ParFiniteElementSpace> vfes_;
   std::unique_ptr<mfem::ParFiniteElementSpace> pfes_;
   mfem::Array<int> block_true_offsets_;
};

} // namespace incns

#endif // INCNS_SPACES_MIXED_SPACES_HPP
