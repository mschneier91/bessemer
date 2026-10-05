/**
 * @file point_block_jacobi.hpp
 * @brief Matrix-free point-block Jacobi for the velocity block
 *        A = sigma M + nu K + N(omega).
 */
#ifndef INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP
#define INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP

#include "operators/rotational_convection.hpp"

#include <mfem.hpp>

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
 *
 * @par How to use it
 *
 * **Your part:**
 *  1. Once: put a VectorRotationalConvectionIntegrator in a PA form, and
 *     construct PointBlockJacobi(fes, rot, ess_tdofs).
 *  2. At setup, and whenever Delta-t or nu changes: SetDiagonal(d), where d is
 *     the diagonal of the rest of the velocity operator (e.g. AssembleDiagonal
 *     of your sigma M + nu K form). N contributes exactly zero, so it does not
 *     matter whether it is included.
 *  3. Every time step, in this order:
 *     - (a) Update w. w is the lagged velocity the rotation term is linearized
 *          about, N(w) u = alpha ((curl w) x u, v): the GridFunction you passed
 *          to the integrator's constructor in step 1. The integrator keeps a
 *          pointer to it, not a copy. It is typically the velocity history
 *          extrapolated to the new time level, e.g. second order
 *          w = 2 u^n - u^{n-1}. Overwrite it in place, or point the integrator
 *          at a different GridFunction with rot.SetLaggedVelocity(w2) (same
 *          mesh and order). A ParGridFunction must be consistent across ranks
 *          before (b) (SetFromTrueDofs or Distribute).
 *     - (b) rot.UpdateVorticity(): recomputes curl w at the quadrature points.
 *          Nothing is reassembled -- operators already formed from the
 *          integrator's form see the new vorticity.
 *     - (c) UpdateSkew(): rebuilds and re-inverts the node blocks from the new
 *          vorticity. It reads the integrator's vorticity data, so it must come
 *          after (b); before (b) it would use the previous step's vorticity.
 *     @code
 *     // setup (step 1), for reference -- the integrator holds a pointer to w
 *     ParGridFunction w(&vfes);
 *     auto *rot = new VectorRotationalConvectionIntegrator(w, alpha);
 *     n_form.AddDomainIntegrator(rot);   // PA form, Assemble()d once
 *     PointBlockJacobi pbj(vfes, *rot, ess_tdofs);
 *
 *     // every time step n -> n+1
 *     Vector w_true(vfes.GetTrueVSize());
 *     add(2.0, u_n, -1.0, u_nm1, w_true); // w = 2 u^n - u^{n-1}, true dofs
 *     w.SetFromTrueDofs(w_true);          // (a) update w in place
 *     rot->UpdateVorticity();             // (b) curl w at quadrature points
 *     pbj.UpdateSkew();                   // (c) node blocks from new curl w
 *     gmres.Mult(rhs, x);                 // solve with pbj as the PC
 *     @endcode
 *  4. Using it: it is nonsymmetric, so use it with GMRES/FGMRES, never PCG.
 *     The operator you apply must include N. If you sum N with an operator
 *     constrained so its Dirichlet rows are identity rows (DIAG_ONE),
 *     constrain N with DIAG_ZERO, or those rows end up with 2 on the
 *     diagonal.
 *
 * **This class's part:** assembling s from the integrator (via the proxy),
 * summing over elements and MPI ranks, handling nonconforming meshes,
 * applying the essential-dof rule, inverting every node block, and applying
 * the inverse in either vector ordering -- all of it on the device.
 *
 * **Caveat -- it cannot tell when its inputs are stale.** If you forget
 * UpdateSkew() after the vorticity changes, or SetDiagonal() after Delta-t
 * changes, it silently preconditions with old blocks. The answer stays
 * correct, because the outer Krylov solver always applies the true operator,
 * but iteration counts climb.
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
   /// Invert the nodal blocks from d_ and s_ into inv_ (closed form, in place).
   void RebuildBlocks();

   int dim_;      ///< Spatial dimension (2 or 3).
   int n_;        ///< True dofs per component.
   bool by_vdim_; ///< True-dof layout: byVDIM a*dim+c, byNODES c*n+a.
   const mfem::Array<int>& ess_; ///< Essential true dofs (not owned).
   mfem::Vector
   d_;              ///< Operator diagonal (dim*n), essential rule applied.
   mfem::Vector s_;              ///< Nodal skew (dim*n), essential rule applied.
   std::unique_ptr<mfem::BilinearForm> skew_form_; ///< PA form, proxy integrator.
   /// Inverse nodal blocks, (dim, dim, n) column-major: inv(i, j, a).
   mfem::Vector inv_;
};

} // namespace incns

#endif // INCNS_PRECOND_POINT_BLOCK_JACOBI_HPP
