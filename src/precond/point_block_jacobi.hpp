/**
 * @file point_block_jacobi.hpp
 * @brief Matrix-free point-block Jacobi for the velocity block
 *        A = sigma M + nu K + N(omega) (rotational_convection_pa_spec.md,
 *        Part B).
 */
#ifndef INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP
#define INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP

#include "operators/rotational_convection.hpp"

#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Inverts the dim x dim block of the velocity operator at each velocity
 *        node, @f$ B_a = \mathrm{diag}(d_a) + [s_a]_\times @f$, on the device.
 *
 * @f$ d @f$ is the diagonal of A (equal to that of sigma M + nu K, since the
 * rotation term's diagonal is exactly zero) and @f$ s @f$ the off-diagonal part
 * of the rotation term's nodal blocks, summed to true dofs through MFEM's PA
 * diagonal assembly with a proxy integrator. Identical to scalar Jacobi where
 * rotation is negligible; exact for sigma M + N under GLL collocation (the
 * operator is then block diagonal). Nonsymmetric: it cannot precondition PCG.
 *
 * Essential dofs follow the constrained operator (DIAG_ONE): component c of
 * node a essential -> d_{a,c} = 1 and, in 3D, s_{a,k} = 0 for k != c; in 2D,
 * s_a = 0 if either component is essential.
 */
class PointBlockJacobi : public mfem::Solver
{
public:
   /**
    * @brief Bind the preconditioner to a velocity space and rotation term.
    * @param fes       Velocity space (vdim == dim); a ParFiniteElementSpace in
    *                  parallel.
    * @param rot       The rotation integrator of the velocity operator, after
    *                  its AssemblePA() on @a fes's mesh (not owned; must
    *                  outlive this object).
    * @param ess_tdofs Velocity essential true dofs (not copied; must outlive
    *                  this object).
    */
   PointBlockJacobi(mfem::FiniteElementSpace& fes,
                    const VectorRotationalConvectionIntegrator& rot,
                    const mfem::Array<int>& ess_tdofs);
   /// Out of line: the batched solver's type is complete only in the .cpp.
   ~PointBlockJacobi() override;

   /**
    * @brief Set the diagonal of the velocity operator and rebuild the blocks.
    *        Call at setup and whenever dt or nu changes.
    * @param d True-dof diagonal, e.g. from velocity_form.AssembleDiagonal().
    */
   void SetDiagonal(const mfem::Vector& d);

   /// Recompute s from rot's current quadrature data and rebuild the blocks.
   /// Call every step after rot.UpdateVorticity().
   void UpdateSkew();

   /**
    * @brief z = B^{-1} r, node by node.
    * @param r True-dof residual.
    * @param z True-dof correction.
    */
   void Mult(const mfem::Vector& r, mfem::Vector& z) const override;

   /// No-op by design: the blocks come from SetDiagonal()/UpdateSkew().
   void SetOperator(const mfem::Operator&) override { }

   /// @return The diagonal after the essential-dof rule (true dofs).
   const mfem::Vector& GetDiagonal() const { return d_; }
   /// @return The skew vector after the essential-dof rule (true dofs; 2D
   ///         uses component 0 only).
   const mfem::Vector& GetSkew() const { return s_; }

private:
   /// Fill the nodal blocks from d_ and s_ and invert them in one batch.
   void RebuildBlocks();

   int dim_;      ///< Spatial dimension (2 or 3).
   int n_;        ///< True dofs per component.
   bool by_vdim_; ///< True-dof layout: byVDIM a*dim+c, byNODES c*n+a.
   const mfem::Array<int>& ess_; ///< Essential true dofs (not owned).
   mfem::Vector
   d_;              ///< Operator diagonal (dim*n), essential rule applied.
   mfem::Vector s_;              ///< Nodal skew (dim*n), essential rule applied.
   std::unique_ptr<mfem::BilinearForm> skew_form_; ///< PA form, proxy integrator.
   std::unique_ptr<mfem::BatchedDirectSolver> blocks_inv_; ///< Batched inverse.
   mutable mfem::Vector r_node_; ///< byNODES: residual gathered node-contiguous.
   mutable mfem::Vector z_node_; ///< byNODES: correction before the scatter.
};

} // namespace incns

#endif // INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP
