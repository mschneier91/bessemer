/**
 * @file vecdivdiv_integrator.hpp
 * @brief Sum-factorized grad-div integrator (Q div u, div v) for [H1]^d vector
 *        fields (vecdivdiv_spec.md; out-of-tree against installed MFEM).
 */
#pragma once

#include "mfem.hpp"
#include "mfem/fem/kernel_dispatch.hpp"

namespace incns
{

/**
 * @brief Integrator for the grad-div bilinear form
 *        @f$ (Q\,\nabla\!\cdot u, \nabla\!\cdot v) @f$ on [H1]^d vector fields
 *        (also searched for as: grad-div stabilization, augmented-Lagrangian
 *        term, the lambda-part of linear elasticity).
 *
 * Replaces ElasticityIntegrator(lambda = Q, mu = 0) for this term: unlike the
 * elasticity PA path (dense, non-tensor, O(p^6) reduction with a global-memory
 * gradient round trip), the PA kernels here are fused and sum-factorized
 * (O(p^4) in 3D) in the style of VectorDiffusionIntegrator, exploiting the
 * RANK-1 structure of the quadrature-point operator: D_q = alpha vec(A)vec(A)^T
 * with A = adj(J), alpha = Q w / detJ -- only d^2 + 1 reals per point.
 *
 * Requirements: vdim == dim == sdim in {2, 3}; PA requires tensor-product
 * elements (quads/hexes); full assembly works on any H1 element. Scalar
 * Coefficient only (grad-div's coefficient is physically scalar). The operator
 * is symmetric, so the PA transpose forwards to the primal apply. Element
 * assembly is per component block: see VectorDivDivComponentIntegrator.
 */
class VectorDivDivIntegrator : public mfem::BilinearFormIntegrator
{
   friend class VectorDivDivComponentIntegrator; // reads Q at assembly

protected:
   mfem::Coefficient* Q = nullptr; ///< Scalar coefficient (null = 1).

   /**
    * @brief The direct (not sum-factorized) PA diagonal, O(p^6) per element
    *        in 3D: the reference form and the fallback beyond the tile limits.
    *        Protected so a test can compare it with the factorized path.
    * @param diag Output E-vector diagonal (accumulated into).
    */
   void AssembleDiagonalPADirect(mfem::Vector& diag) const;

private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape_, gshape_; ///< Full-assembly scratch.
   mfem::Vector divshape_;             ///< Full-assembly scratch.
#endif
   // --- PA extension state ---
   /// Quadrature data, (nq, dim*dim + 1, ne): adj(J) columns then alpha.
   mfem::Vector pa_data_;
   const mfem::DofToQuad* maps_ =
      nullptr;        ///< Tensor basis maps (not owned).
   const mfem::GeometricFactors* geom_ = nullptr; ///< Jacobians (not owned).
   int dim_ = 0, ne_ = 0, dofs1D_ = 0, quad1D_ = 0; ///< PA geometry summary.

public:
   /**
    * @brief Grad-div with unit coefficient.
    * @param ir Optional integration rule (RuleBook-owned in this codebase).
    */
   VectorDivDivIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) { }

   /**
    * @brief Grad-div with a scalar coefficient.
    * @param q  Scalar coefficient (borrowed; may be negative/sign-changing).
    * @param ir Optional integration rule (RuleBook-owned in this codebase).
    */
   VectorDivDivIntegrator(mfem::Coefficient& q,
                          const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q(&q) { }

   /**
    * @brief Full assembly (the test oracle; works for any H1 element).
    * @param el    The scalar finite element (vector via vdim blocks).
    * @param Trans Element transformation.
    * @param elmat Output (dim*nd)^2 element matrix, byNODES block layout.
    */
   void AssembleElementMatrix(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

   using mfem::BilinearFormIntegrator::AssemblePA;
   /**
    * @brief PA setup: store adj(J) and alpha = Q w / detJ per quadrature point.
    * @param fes Vector H1 space with vdim == dim == sdim, tensor elements.
    */
   void AssemblePA(const mfem::FiniteElementSpace& fes) override;

   /**
    * @brief Fused sum-factorized PA apply (E-vector in, E-vector accumulate).
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const override;

   /**
    * @brief Transpose apply -- the operator is symmetric; forwards to AddMultPA.
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultTransposePA(const mfem::Vector& x, mfem::Vector& y) const override
   { AddMultPA(x, y); }

   /**
    * @brief PA diagonal: diag(a,c) = sum_q alpha (Ghat_a . A[:,c])^2,
    *        sum-factorized (O(p^4) per element in 3D, Q1D x Q1D threads per
    *        element); beyond the device's MAX_D1D / MAX_Q1D shared-tile limits
    *        it falls back to AssembleDiagonalPADirect.
    * @param diag Output E-vector diagonal (accumulated into).
    */
   void AssembleDiagonalPA(mfem::Vector& diag) override;

   /// @return The scalar coefficient (null when the default 1 is in use).
   const mfem::Coefficient* GetCoefficient() const { return Q; }

   /// Kernel signature: ne, B, G, pa_data, x, y, d1d, q1d.
   using ApplyKernelType = void (*)(const int, const mfem::Array<mfem::real_t>&,
                                    const mfem::Array<mfem::real_t>&,
                                    const mfem::Vector&, const mfem::Vector&,
                                    mfem::Vector&, const int, const int);

   /// Kernel registry keyed on (dim, d1d, q1d).
   MFEM_REGISTER_KERNELS(ApplyPAKernels, ApplyKernelType, (int, int, int));

   /// Register a compile-time (DIM, D1D, Q1D) specialization.
   template <int DIM, int D1D, int Q1D>
   static void AddSpecialization()
   { ApplyPAKernels::Specialization<DIM, D1D, Q1D>::Add(); }
};

/**
 * @brief One (i, j) block of a VectorDivDivIntegrator, viewing grad-div as a
 *        dim x dim block operator, assembled on a SCALAR H1 space:
 *        @f$ K^{ij}_{ab} = (Q\,\partial_j \phi_b, \partial_i \phi_a) @f$
 *        (row/test component i, column/trial component j).
 *
 * Exists because MFEM's element-assembly extension sizes element matrices by
 * the scalar dof count and ignores vdim (checked against the installed MFEM
 * 17d1afc: a vdim = 2 Q2 space gets 9 x 9 per element, not 18 x 18), so a
 * vector integrator cannot serve AssemblyLevel::ELEMENT / FULL. This is the
 * workaround MFEM itself uses for elasticity (ElasticityComponentIntegrator):
 * assemble each block on a scalar space -- AssemblyLevel::FULL builds the
 * sparse matrix on the device from AssembleEA -- then merge the dim^2 blocks
 * with HypreParMatrixFromBlocks, whose per-rank block ordering equals the
 * true-dof ordering of a byNODES vector space (pinned by vecdivdiv_test).
 *
 * Intended for low order -- the Q1 LOR rediscretization that feeds LOR-AMG.
 * The kernel is a direct O(nd^2 nq) sum per element, not sum-factorized, so
 * it is not meant for high-order element matrices. Quads/hexes only, like the
 * PA path. K^{ji} = (K^{ij})^T; only the diagonal blocks are symmetric.
 */
class VectorDivDivComponentIntegrator : public mfem::BilinearFormIntegrator
{
   const VectorDivDivIntegrator& parent_; ///< Coefficient source (not owned).
   const int i_block_; ///< Row (test) component.
   const int j_block_; ///< Column (trial) component.

public:
   /**
    * @brief The (i, j) block of @a parent.
    * @param parent  Supplies the coefficient; must outlive this integrator.
    * @param i_block Row (test) component, 0 <= i_block < dim.
    * @param j_block Column (trial) component, 0 <= j_block < dim.
    *
    * The integration rule is this integrator's own if set, else the parent's,
    * else the DiffusionIntegrator default -- the same fallback as the parent.
    */
   VectorDivDivComponentIntegrator(const VectorDivDivIntegrator& parent,
                                   int i_block, int j_block)
      : parent_(parent), i_block_(i_block), j_block_(j_block) { }

   /**
    * @brief Element matrices of the (i, j) block, row-major per element in
    *        lexicographic dof order (MFEM's EA layout).
    * @param fes  Scalar (vdim == 1) H1 space on a quad/hex mesh, dim == sdim.
    * @param emat Output, size nd * nd * ne.
    * @param add  Accumulate into @a emat if true, overwrite otherwise.
    */
   void AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat,
                   const bool add = true) override;
};

} // namespace incns
