/**
 * @file rotational_convection.hpp
 * @brief Lagged-vorticity (semi-implicit) rotational convection,
 *        @f$ \alpha ((\nabla\times w)\times u, v) @f$, partially assembled
 *        (docs/design/rotational_convection_pa_spec.md, Part A).
 */
#pragma once

#include "mfem.hpp"
#include "mfem/fem/kernel_dispatch.hpp"

#include <string>

namespace incns
{

/**
 * @brief @f$ a(u, v) = \alpha ((\nabla\times w)\times u, v) @f$: the rotational
 *        convection term with the vorticity lagged on a given velocity @a w.
 *
 * The term in @f$ (u\cdot\nabla)u = (\nabla\times u)\times u
 * + \nabla(\tfrac12|u|^2) @f$, linearized by lagging the first factor, so it
 * is a BILINEAR form in u and can sit in the implicit velocity block. It is
 * skew-symmetric (@f$ N^T = -N @f$, energy neutral for any w) with an exactly
 * zero diagonal. Sign convention: @f$ \omega\times u @f$, NOT
 * @f$ u\times\omega @f$; test A1 pins it.
 *
 * Zero order in u, so the apply is a vector mass apply with a pointwise skew
 * coupling; all derivative work (curl w at the quadrature points) happens once
 * per step in UpdateVorticity(). Quadrature data per point:
 * @f$ d_q = \alpha \hat w_q \det J_q\, \omega_q @f$, computed through adj(J)
 * without dividing by det J (3 reals per point in 3D, 1 in 2D).
 *
 * PA on quads/hexes, dim == sdim == vdim in {2, 3}, byNODES or byVDIM; plus
 * legacy full assembly (AssembleElementMatrix -- the path MFEM's CPU LOR
 * assembly uses) and, per component block, element assembly
 * (VectorRotationalConvectionComponentIntegrator -- the GPU route). Both
 * full/element assembly paths evaluate w on the mesh being assembled: for
 * LOR, give them a w on the LOR space (H1 LOR shares the high-order true
 * dofs, so that is a plain copy). Moving meshes: after node changes the caller must
 * call mesh->DeleteGeometricFactors() and AssemblePA() again.
 */
class VectorRotationalConvectionIntegrator : public mfem::BilinearFormIntegrator
{
public:
   /**
    * @brief Integrator for the vorticity of @a w.
    * @param w     Lagged/extrapolated velocity: H1 vector GridFunction on the
    *              same mesh and order as the trial space (not owned). In
    *              parallel it must be distributed before UpdateVorticity().
    * @param alpha Scalar multiplier.
    */
   explicit VectorRotationalConvectionIntegrator(const mfem::GridFunction& w,
         mfem::real_t alpha = 1.0)
      : w_(&w), alpha_(alpha) { }

   /**
    * @brief Replace the lagged velocity; takes effect at UpdateVorticity().
    * @param w New lagged velocity (not owned; same mesh, order and FE
    *          collection as the trial space -- any space object/ordering).
    */
   void SetLaggedVelocity(const mfem::GridFunction& w) { w_ = &w; }
   /**
    * @brief Replace alpha; takes effect at UpdateVorticity().
    * @param a New scalar multiplier.
    */
   void SetAlpha(mfem::real_t a) { alpha_ = a; }
   /// @return The current alpha.
   mfem::real_t GetAlpha() const { return alpha_; }

   /**
    * @brief Recompute the quadrature data from the current w, on the device,
    *        without allocating (after the first AssemblePA()).
    * @pre AssemblePA() has run; w is an up-to-date L-vector (a
    *      ParGridFunction needs SetFromTrueDofs()/Distribute() first).
    */
   void UpdateVorticity();

   /**
    * @brief Default rule: VectorConvectionNLFIntegrator::GetRule.
    * @param fe Trial/test element.
    * @param T  A typical element transformation (supplies the mesh order).
    * @return The rule.
    */
   static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& fe, const mfem::ElementTransformation& T);

   using mfem::BilinearFormIntegrator::AssemblePA;
   /**
    * @brief Validate, store geometry/basis data, allocate, UpdateVorticity().
    * @param fes Velocity space (vdim == dim, tensor H1, one geometry).
    */
   void AssemblePA(const mfem::FiniteElementSpace& fes) override;
   /**
    * @brief y += N x (E-vectors).
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultPA(const mfem::Vector& x, mfem::Vector& y) const override;
   /**
    * @brief y += N^T x = -N x (E-vectors).
    * @param x Input E-vector.
    * @param y Output E-vector (accumulated into).
    */
   void AddMultTransposePA(const mfem::Vector& x,
                           mfem::Vector& y) const override;
   /**
    * @brief Adds the (exactly zero) diagonal. Must exist: the base aborts and
    *        BilinearForm::AssembleDiagonal calls every integrator.
    * @param diag E-vector diagonal (unchanged).
    */
   void AssembleDiagonalPA(mfem::Vector& diag) override;

   /**
    * @brief Legacy full assembly of one element matrix (any rule; CPU), e.g.
    *        for MFEM's LOR assembly or an assembled reference.
    * @param el    Scalar finite element (vector via dim blocks, byNODES-style
    *              elmat layout: index a + i*nd).
    * @param Trans Element transformation; must belong to w's mesh (aborts
    *              otherwise -- curl w would be read from the wrong element).
    * @param elmat Output (dim*nd)^2 element matrix.
    */
   void AssembleElementMatrix(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

   /**
    * @brief Off-diagonal part of the nodal dim x dim blocks, for point-block
    *        Jacobi (spec 5.9). Adds into an E-vector of the velocity space,
    *        layout (D1D^dim, dim, NE), lexicographic: 3D component c
    *        += sum_q B_a(q)^2 d_q[c]; 2D component 0 += sum_q B_a(q)^2 d_q,
    *        component 1 untouched.
    * @param s_e Velocity E-vector (accumulated into).
    */
   void AddNodalSkewPA(mfem::Vector& s_e) const;

   /**
    * @brief Quadrature data, for tests. Column-major layout:
    *        3D (Q1D, Q1D, Q1D, 3, NE) = alpha w_q detJ_q omega_c;
    *        2D (Q1D, Q1D, NE) = alpha w_q detJ_q omega.
    * @return The data.
    */
   const mfem::Vector& GetQuadratureData() const { return pa_data_; }

   /// @return The mesh of the last AssemblePA() (null before it).
   const mfem::Mesh* GetAssembledMesh() const { return pa_mesh_; }

   /**
    * @brief Whether (dim, d1d, q1d) has compile-time-specialized apply and
    *        setup kernels (others run the slower generic fallback).
    * @param dim Spatial dimension.
    * @param d1d Dofs per direction (order + 1).
    * @param q1d Quadrature points per direction.
    * @return True if specialized.
    */
   static bool HasSpecialization(int dim, int d1d, int q1d);

   /// Apply kernel signature: ne, sign, B, d, x, y, d1d, q1d.
   using ApplyType = void (*)(int, mfem::real_t, const mfem::Array<mfem::real_t>&,
                              const mfem::Vector&, const mfem::Vector&,
                              mfem::Vector&, int, int);
   /// Apply kernel registry keyed on (dim, d1d, q1d).
   MFEM_REGISTER_KERNELS(RotConvApplyPA, ApplyType, (int, int, int));
   /// Setup kernel signature: ne, alpha, B, G, W, J, w_e, d, d1d, q1d.
   using SetupType = void (*)(int, mfem::real_t, const mfem::real_t*,
                              const mfem::real_t*, const mfem::real_t*,
                              const mfem::real_t*, const mfem::real_t*,
                              mfem::real_t*, int, int);
   /// Setup kernel registry keyed on (dim, d1d, q1d).
   MFEM_REGISTER_KERNELS(RotConvSetupPA, SetupType, (int, int, int));

private:
   /// Validate w's space against the trial space; (re)build its restriction
   /// and E-vector buffer. Allocates only when w's space object changed.
   void BindLaggedSpace();

   const mfem::GridFunction* w_;            ///< Lagged velocity (not owned).
   const mfem::FiniteElementSpace* w_fes_ = nullptr; ///< Space w_restr_ is for.
   std::string trial_fec_;                  ///< Trial FE collection name.
   mfem::real_t alpha_;                     ///< Scalar multiplier.
   int dim_ = 0; ///< Spatial dimension (set by AssemblePA).
   int ne_ = 0;  ///< Number of elements (set by AssemblePA).
   int d1d_ = 0; ///< Dofs per direction, D1D (set by AssemblePA).
   int q1d_ = 0; ///< Quadrature points per direction, Q1D (set by AssemblePA).
   const mfem::DofToQuad* maps_ = nullptr;  ///< Tensor maps (not owned).
   const mfem::GeometricFactors* geom_ = nullptr; ///< Jacobians (not owned).
   const mfem::IntegrationRule* pa_ir_ = nullptr; ///< Rule used at AssemblePA.
   const mfem::Mesh* pa_mesh_ = nullptr;    ///< Mesh at AssemblePA.
   const mfem::Operator* w_restr_ = nullptr; ///< Lexicographic restriction of w.
   mfem::Vector w_e_;                       ///< E-vector buffer for w (reused).
   mfem::Vector pa_data_;                   ///< d_q, layout above.
};

/**
 * @brief One (i, j) block of the rotation term on a SCALAR H1 space, for
 *        element assembly: @f$ N^{ij}_{ab} = \alpha \sum_q \hat w_q \det J_q
 *        \varphi_a \varphi_b [\omega_q]_{\times,ij} @f$ (row/test component
 *        i, column/trial component j); the diagonal blocks are zero.
 *
 * Exists for the same reason as GradDivComponentIntegrator: MFEM's EA
 * extension sizes element matrices by the scalar dof count and ignores vdim,
 * so a vector integrator cannot serve AssemblyLevel::ELEMENT / FULL. Assemble
 * each block on a scalar space (FULL builds the sparse matrix on the device
 * from AssembleEA) and merge with HypreParMatrixFromBlocks -- the route for a
 * device-assembled LOR operator. Direct O(nd^2 nq) kernel: meant for low
 * order (Q1 LOR), not high-order element matrices. Quads/hexes only.
 * N^{ji} = -N^{ij} entrywise-transposed (the operator is skew).
 */
class VectorRotationalConvectionComponentIntegrator :
   public mfem::BilinearFormIntegrator
{
   const mfem::GridFunction* w_; ///< Lagged velocity (not owned).
   mfem::real_t alpha_;          ///< Scalar multiplier.
   const int i_block_;           ///< Row (test) component.
   const int j_block_;           ///< Column (trial) component.

public:
   /**
    * @brief The (i, j) block for the vorticity of @a w.
    * @param w       Vector H1 GridFunction on the assembly mesh, same order
    *                and FE collection as the scalar space (not owned).
    * @param alpha   Scalar multiplier.
    * @param i_block Row (test) component, 0 <= i_block < dim.
    * @param j_block Column (trial) component, 0 <= j_block < dim.
    *
    * The integration rule is this integrator's own if set, else
    * VectorRotationalConvectionIntegrator::GetRule.
    */
   VectorRotationalConvectionComponentIntegrator(const mfem::GridFunction& w,
         mfem::real_t alpha,
         int i_block, int j_block)
      : w_(&w), alpha_(alpha), i_block_(i_block), j_block_(j_block) { }

   /**
    * @brief Element matrices of the (i, j) block, row-major per element in
    *        lexicographic dof order (MFEM's EA layout); w is read now.
    * @param fes  Scalar (vdim == 1) H1 space on w's mesh, quads/hexes.
    * @param emat Output, size nd * nd * ne.
    * @param add  Accumulate into @a emat if true, overwrite otherwise.
    */
   void AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat,
                   const bool add = true) override;
};

} // namespace incns
