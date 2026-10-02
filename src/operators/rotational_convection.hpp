/**
 * @file rotational_convection.hpp
 * @brief Lagged-vorticity (semi-implicit) rotational convection,
 *        @f$ \alpha ((\nabla\times w)\times u, v) @f$, partially assembled
 *        (rotational_convection_pa_spec.md, Part A).
 */
#ifndef INCNS_OPERATORS_ROTATIONAL_CONVECTION_HPP
#define INCNS_OPERATORS_ROTATIONAL_CONVECTION_HPP

#include "mfem.hpp"
#include "mfem/fem/kernel_dispatch.hpp"

#include <string>

namespace incns
{

/// Rotation-number diagnostic, @f$ \mu_q = |\omega_q| / \sigma @f$ (spec 5.10).
struct RotationNumberStats
{
   mfem::real_t max_mu;       ///< Max over quadrature points (global over ranks).
   mfem::real_t vol_fraction; ///< Volume fraction with mu_q > threshold (global).
   mfem::real_t threshold;    ///< The threshold the fraction was taken at.
};

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
 * PA only (no legacy/EA/MF assembly), on quads/hexes, dim == sdim == vdim in
 * {2, 3}, byNODES or byVDIM. Moving meshes: after node changes the caller must
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

   /// Replace the lagged velocity; takes effect at UpdateVorticity().
   void SetLaggedVelocity(const mfem::GridFunction& w) { w_ = &w; }
   /// Replace alpha; takes effect at UpdateVorticity().
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
    * @brief Off-diagonal part of the nodal dim x dim blocks, for point-block
    *        Jacobi (spec 5.9). Adds into an E-vector of the velocity space,
    *        layout (D1D^dim, dim, NE), lexicographic: 3D component c
    *        += sum_q B_a(q)^2 d_q[c]; 2D component 0 += sum_q B_a(q)^2 d_q,
    *        component 1 untouched.
    * @param s_e Velocity E-vector (accumulated into).
    */
   void AddNodalSkewPA(mfem::Vector& s_e) const;

   /**
    * @brief Rotation-number statistics (spec 5.10); global over ranks.
    * @param sigma     The mass coefficient (BDF2: 3/(2 dt)).
    * @param threshold mu threshold for the volume fraction.
    * @return max mu and the volume fraction above @a threshold.
    */
   RotationNumberStats GetRotationNumberStats(mfem::real_t sigma,
         mfem::real_t threshold = 1.0) const;

   /**
    * @brief Quadrature data, for tests. Column-major layout:
    *        3D (Q1D, Q1D, Q1D, 3, NE) = alpha w_q detJ_q omega_c;
    *        2D (Q1D, Q1D, NE) = alpha w_q detJ_q omega.
    * @return The data.
    */
   const mfem::Vector& GetQuadratureData() const { return pa_data_; }

   /// @return The mesh of the last AssemblePA() (null before it).
   const mfem::Mesh* GetAssembledMesh() const { return pa_mesh_; }

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
   int dim_ = 0, ne_ = 0, d1d_ = 0, q1d_ = 0; ///< PA geometry summary.
   const mfem::DofToQuad* maps_ = nullptr;  ///< Tensor maps (not owned).
   const mfem::GeometricFactors* geom_ = nullptr; ///< Jacobians (not owned).
   const mfem::IntegrationRule* pa_ir_ = nullptr; ///< Rule used at AssemblePA.
   const mfem::Mesh* pa_mesh_ = nullptr;    ///< For MPI reductions.
   const mfem::Operator* w_restr_ = nullptr; ///< Lexicographic restriction of w.
   mfem::Vector w_e_;                       ///< E-vector buffer for w (reused).
   mfem::Vector pa_data_;                   ///< d_q, layout above.
   mutable mfem::Vector diag_tmp0_, diag_tmp1_; ///< Diagnostic scratch (reused).
};

} // namespace incns

#endif // INCNS_OPERATORS_ROTATIONAL_CONVECTION_HPP
