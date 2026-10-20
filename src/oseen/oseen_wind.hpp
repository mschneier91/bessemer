/**
 * @file oseen_wind.hpp
 * @brief The wind-dependent quadrature data of the Oseen operators of the
 *        semi-implicit and implicit BDF2 schemes (docs/design/oseen_bdf2.md):
 *        the wind, 1/2 div w and the OSS stabilization's tau terms, computed on
 *        the device from the wind's grid function.
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/// A scalar coefficient backed by a quadrature function. Project() copies it
/// (partial assembly, on the device); Eval() reads it on the host at the
/// point's index (full assembly with the same rule: the tests).
class QFScalarCoefficient : public mfem::Coefficient
{
public:
   /// @param q The values (borrowed; must outlive this).
   explicit QFScalarCoefficient(const mfem::QuadratureFunction& q) : q_(q) {}
   mfem::real_t Eval(mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip) override;
   void Project(mfem::QuadratureFunction& qf) override;

private:
   const mfem::QuadratureFunction& q_; ///< The values.
};

/// A vector coefficient backed by a quadrature function (see
/// QFScalarCoefficient).
class QFVectorCoefficient : public mfem::VectorCoefficient
{
public:
   /// @param q The values, vdim = the vector dimension (borrowed).
   explicit QFVectorCoefficient(const mfem::QuadratureFunction& q)
      : mfem::VectorCoefficient(q.GetVDim()), q_(q) {}
   void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
             const mfem::IntegrationPoint& ip) override;
   using mfem::VectorCoefficient::Eval;
   void Project(mfem::QuadratureFunction& qf) override;

private:
   const mfem::QuadratureFunction& q_; ///< The values.
};

/// A symmetric matrix coefficient backed by a quadrature function of
/// dim * dim values per point, column-major (see QFScalarCoefficient).
class QFMatrixCoefficient : public mfem::MatrixCoefficient
{
public:
   /// @param q   The values, vdim = dim * dim (borrowed).
   /// @param dim The matrix dimension.
   QFMatrixCoefficient(const mfem::QuadratureFunction& q, int dim)
      : mfem::MatrixCoefficient(dim), q_(q) {}
   void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
             const mfem::IntegrationPoint& ip) override;
   /// The matrix is symmetric: @p transpose changes nothing.
   void Project(mfem::QuadratureFunction& qf, bool transpose = false) override;

private:
   const mfem::QuadratureFunction& q_; ///< The values.
};

/**
 * @brief The OSS stabilization parameter (Codina): tau = h_w xi(Pe) /
 *        (2 |w| k), Pe = |w| h_w / (2 nu k), xi(a) = coth a - 1/a, with h_w the
 *        element length along w, 1 / ||J^{-1} w/|w| ||_inf; for w -> 0,
 *        h^2 / (12 nu k^2) with h = 1 / max |(J^{-1})_ij|.
 * @param dim Dimension.
 * @param w   The wind (dim values).
 * @param Ji  J^{-1}, column-major (dim x dim).
 * @param nu  Kinematic viscosity (0: the inviscid limit xi = 1).
 * @param k   Velocity order.
 * @return tau.
 */
MFEM_HOST_DEVICE inline mfem::real_t OssTau(int dim, const mfem::real_t* w,
      const mfem::real_t* Ji,
      mfem::real_t nu, int k)
{
   mfem::real_t wn = 0.0;
   for (int d = 0; d < dim; d++) { wn += w[d] * w[d]; }
   wn = sqrt(wn);
   if (wn < 1e-14)
   {
      mfem::real_t ih = 0.0;
      for (int i = 0; i < dim * dim; i++) { ih = fmax(ih, fabs(Ji[i])); }
      const mfem::real_t h = 1.0 / ih;
      return (nu > 0.0) ? h * h / (12.0 * nu * k * k) : 0.0;
   }
   mfem::real_t ihw = 0.0; // ||J^{-1} w / |w| ||_inf = 1 / h_w
   for (int i = 0; i < dim; i++)
   {
      mfem::real_t r = 0.0;
      for (int j = 0; j < dim; j++) { r += Ji[i + dim * j] * w[j] / wn; }
      ihw = fmax(ihw, fabs(r));
   }
   const mfem::real_t hw = 1.0 / ihw;
   mfem::real_t xi = 1.0;
   if (nu > 0.0)
   {
      const mfem::real_t a = wn * hw / (2.0 * nu * k);
      xi = (a < 1e-3) ? a / 3.0 - a * a * a / 45.0
           : (a > 20.0) ? 1.0 - 1.0 / a : 1.0 / tanh(a) - 1.0 / a;
   }
   return hw * xi / (2.0 * k * wn);
}

/**
 * @brief The quadrature data one wind w gives the Oseen operators, on the
 *        device, for a space of order m (the velocity's, k, or the
 *        pressure's for the PCD preconditioner's N_p): w and 1/2 div w at the
 *        convection rule (Gauss, order k + 2m); tau_c w w^T and tau_c w at
 *        the OSS rule (Gauss, order 2m + 2, the same for both OSS terms);
 *        tau_c at the collocated GLL rule (order 2m - 1). tau always takes
 *        the velocity order k. tau_c is OssTau with the wind and J^{-1} at the element
 *        centre: one value per element keeps the OSS operator symmetric
 *        positive semidefinite (the recipe's rules).
 *
 * Update() recomputes everything from a wind grid function; the coefficients
 * keep pointing at the same quadrature functions, so a form re-assembled
 * after it sees the new wind.
 */
class OseenWindData
{
public:
   /**
    * @param vfes  The velocity space (vector, Ordering::byNODES) the wind
    *              lives on (borrowed).
    * @param rules Quadrature source (borrowed; the rules must outlive this).
    * @param nu    Kinematic viscosity (tau's Peclet number).
    * @param space_order The order m of the space the operators act on;
    *              <= 0: the velocity order k.
    */
   OseenWindData(mfem::ParFiniteElementSpace& vfes, const RuleBook& rules,
                 double nu, int space_order = 0);

   /// Recompute the data from @p wind (a consistent L-vector of vfes).
   void Update(const mfem::ParGridFunction& wind);

   /// @return The convection rule (Gauss, order k + 2m).
   const mfem::IntegrationRule& ConvRule() const { return *ir_conv_; }
   /// @return The OSS rule (Gauss, order 2m + 2).
   const mfem::IntegrationRule& OssRule() const { return *ir_oss_; }
   /// @return The collocated GLL rule (order 2m - 1).
   const mfem::IntegrationRule& GllRule() const { return *ir_gll_; }

   mfem::VectorCoefficient& Wind() { return wind_c_; }       ///< w (conv rule).
   mfem::Coefficient& HalfDivWind() { return half_divw_c_; } ///< 1/2 div w.
   mfem::MatrixCoefficient& TauWW() { return tau_ww_c_; }    ///< tau_c w w^T.
   mfem::VectorCoefficient& TauW() { return tau_w_c_; }      ///< tau_c w.
   mfem::Coefficient& TauGll() { return tau_gll_c_; }        ///< tau_c (GLL).
   /// @return tau_c, one value per element (device).
   const mfem::Vector& TauCenter() const { return tau_c_; }
   /// @return The velocity order k.
   int Order() const { return k_; }

private:
   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   double nu_;                         ///< Viscosity.
   int dim_ = 0;                       ///< Dimension.
   int k_ = 0;                         ///< Velocity order.
   int m_ = 0;                         ///< The target space's order.
   const mfem::IntegrationRule* ir_conv_ = nullptr;   ///< Convection rule.
   const mfem::IntegrationRule* ir_oss_ = nullptr;    ///< OSS rule.
   const mfem::IntegrationRule* ir_gll_ = nullptr;    ///< GLL rule.
   const mfem::IntegrationRule* ir_center_ = nullptr; ///< One point, the centre.
   mfem::QuadratureSpace qs_conv_, qs_oss_, qs_gll_, qs_center_;
   mfem::QuadratureFunction wind_conv_; ///< w at the convection rule.
   mfem::QuadratureFunction half_divw_; ///< 1/2 div w there.
   mfem::QuadratureFunction wind_oss_;  ///< w at the OSS rule (scratch).
   mfem::QuadratureFunction tau_ww_;    ///< tau_c w w^T there.
   mfem::QuadratureFunction tau_w_;     ///< tau_c w there.
   mfem::QuadratureFunction tau_gll_;   ///< tau_c at the GLL rule.
   mfem::QuadratureFunction wind_center_; ///< w at the centre (scratch).
   mfem::Vector tau_c_;   ///< tau_c per element.
   mfem::Vector e_vec_;   ///< The wind's E-vector (scratch).
   mfem::Vector q_der_;   ///< Its physical derivatives at the conv rule.
   QFVectorCoefficient wind_c_;
   QFScalarCoefficient half_divw_c_;
   QFMatrixCoefficient tau_ww_c_;
   QFVectorCoefficient tau_w_c_;
   QFScalarCoefficient tau_gll_c_;
};

} // namespace incns
