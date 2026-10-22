/**
 * @file hyperelastic_operator.hpp
 * @brief The internal force R(u) = int P(F(u)) : grad_X v of a hyperelastic
 *        solid on the reference configuration and its tangent, matrix-free in
 *        partial assembly on the device (docs/design/hyperelasticity_spec.md
 *        section 3).
 */
#pragma once

#include "operators/element_scratch.hpp"
#include "solid/hyperelastic_material.hpp"
#include "mfem.hpp"

#include <cstddef>
#include <memory>

namespace incns
{

/// What the tangent keeps from its linearization point: memory against
/// work per apply (per quadrature point in 3D: 18, 9 or about 3 numbers).
enum class TangentStorage
{
   /// F and S at every point: the least arithmetic per apply.
   StressAndDeformation,
   /// F at every point; S recomputed in every apply. The default: half the
   /// data of StressAndDeformation and, on an H100 at Q6, 5 % faster (less to
   /// read); as fast on the CPU (docs/hyperelasticity.md section 3).
   Deformation,
   /// The displacement's element vector (its L-vector where the gather is
   /// fused, 3D on a device); F and S recomputed in every apply (one more
   /// gradient pass).
   Displacement
};

/**
 * @brief R(u) = int_Omega0 P(F(u)) : grad_X v dX on true dofs, F = I +
 *        grad_X u, for a vector H1 space ordered by nodes, without boundary
 *        conditions (the solver constrains it).
 *
 * Each apply is one fused kernel per element: the reference gradients of all
 * components by sum factorization, the material law at every point, the
 * transposed gradient back to the dofs (MFEM's kernels::internal helpers, as
 * GradDivIntegrator's apply; in 3D on a device by lines, the gradients kept in
 * shared memory, as the momentum apply of change 56, and the kernels read the
 * L-vector through the restriction's gather map and write it back themselves,
 * with no E-vectors). Tensor-product elements (quads, hexes) with as
 * many points per direction as dofs (the Gauss rule of order 2k + 1, the
 * linear solver's), k from 1 to 7. Lambda and mu are per element (the mesh
 * motion's stiffening).
 *
 * The geometry (J^-1 and w det J at every point) is computed once from the
 * mesh's geometric factors at the rule and kept; the mesh's cache is left as
 * it is (the owner of the rule clears it, as LinearElasticity does).
 */
class HyperelasticOperator : public mfem::Operator
{
public:
   /**
    * @param fes      Vector H1 space, vdim = dim, ordered by nodes (borrowed).
    * @param ir       Tensor rule with as many points per direction as the
    *                 space has dofs (borrowed; must outlive this).
    * @param material The material model.
    * @param lambda   First Lame parameter per local element.
    * @param mu       Shear modulus per local element.
    * @param storage  What the tangent keeps (TangentStorage).
    * @param scratch  Work buffers to borrow (sized for @p fes; borrowed, must
    *                 outlive this; null: own them). No apply of this
    *                 operator may run inside another borrower's.
    */
   HyperelasticOperator(mfem::ParFiniteElementSpace& fes,
                        const mfem::IntegrationRule& ir,
                        HyperelasticMaterial material,
                        const mfem::Vector& lambda, const mfem::Vector& mu,
                        TangentStorage storage = TangentStorage::Deformation,
                        ElementScratch* scratch = nullptr);
   ~HyperelasticOperator() override;

   /// r = R(u) on true dofs.
   void Mult(const mfem::Vector& u, mfem::Vector& r) const override;

   /**
    * @brief The tangent K_T(u) w = int dP(F(u))[grad_X w] : grad_X v, from
    *        what TangentStorage keeps of @p u. Valid until the next call (one
    *        tangent per operator).
    * @param u Linearization point (true dofs).
    */
   mfem::Operator& GetGradient(const mfem::Vector& u) const override;

   /// @return The strain energy int W(F(u)) dX (true dofs). Collective.
   double Energy(const mfem::Vector& u) const;

   /// @return The vector space.
   mfem::ParFiniteElementSpace& Space() const { return fes_; }

   /// @return The tangent storage in use.
   TangentStorage Storage() const { return storage_; }

   /// @return Bytes of device data this operator owns now (geometry, the
   ///         linearization, the energy partial sums, its own work vectors;
   ///         not borrowed buffers).
   std::size_t MemoryBytes() const;

private:
   class Tangent;

   /// u (true dofs) to the E-vector ue_.
   void ToElements(const mfem::Vector& u) const;
   /// The E-vector ye_ added into y (true dofs; y is overwritten).
   void FromElements(mfem::Vector& y) const;

   /// The by-lines path's maps from the restriction's gather map (lines_
   /// false where the restriction is not an ElementRestriction).
   void SetupLines(ElementScratch* scratch);
   /// u (true dofs) as an L-vector: u itself or P u in ul_.
   const mfem::Vector& LocalInput(const mfem::Vector& u) const;
   /// The L-vector the by-lines kernels write for y (true dofs): y or yl_.
   mfem::Vector& LocalOutput(mfem::Vector& y) const;
   /// The shared dofs' slots summed into @p yl, then y = P^T yl if needed.
   void FinishOutput(mfem::Vector& yl, mfem::Vector& y) const;

   mfem::ParFiniteElementSpace& fes_;
   const mfem::IntegrationRule& ir_;
   HyperelasticMaterial material_;
   TangentStorage storage_;
   int dim_ = 0, n1d_ = 0, ne_ = 0;
   const mfem::DofToQuad* maps_ = nullptr;
   const mfem::Operator* restriction_ = nullptr; ///< Lexicographic E-vectors.
   mfem::Vector lambda_, mu_; ///< Per element.
   mfem::Vector jinv_;        ///< J^-1 at the points, (nq, dim, dim, ne).
   mfem::Vector wdet_;        ///< w det J at the points, (nq, ne).
   /// The linearization: (nq, 2 dim^2, ne), (nq, dim^2, ne) or the
   /// displacement's E-vector, by storage_.
   mutable mfem::Vector lin_;
   mutable mfem::Vector qe_; ///< Energy partial sums (allocated by Energy()).
   mutable mfem::Vector ul_, ue_, ye_, yl_; ///< Work (borrowed or own).
   bool own_e_ = true, own_l_ = true; ///< Whether ue_/ye_ and ul_/yl_ are own.
   bool p_identity_ = false; ///< Whether the prolongation is the identity.

   /// 3D on a device: the by-lines kernels read the L-vector through the
   /// gather map and write it back themselves (no E-vectors): a dof one
   /// element holds directly, a shared dof through its slots, summed in
   /// element order (deterministic, no atomics).
   bool lines_ = false;
   const mfem::Array<int>* gmap_ =
      nullptr; ///< E-dof to L-dof (the restriction's).
   mfem::Array<int> dmap_; ///< Per E-dof: its L-dof, or -1 - its slot if shared.
   mfem::Array<int> sdof_, soff_, sidx_; ///< Shared L-dofs and their slots (CSR).
   mutable mfem::Vector slots_; ///< (slots, 3): the shared dofs' contributions.
   int nd_ = 0, nslots_ = 0; ///< Scalar L-dofs, slots.
   bool own_slots_ = true;
   std::unique_ptr<Tangent> tangent_;
};

} // namespace incns
