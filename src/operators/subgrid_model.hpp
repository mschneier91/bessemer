/**
 * @file subgrid_model.hpp
 * @brief Pointwise subgrid-scale eddy viscosities for large-eddy simulation:
 *        the sigma and WALE models (and a constant, for verification).
 */
#pragma once

#include "mfem.hpp"

#include <cmath>

namespace incns
{

/**
 * @brief Which subgrid-scale model closes the filtered momentum equation
 *        (deck `physics.subgrid.model: none|constant|wale|sigma`).
 *
 * Every model is an eddy viscosity @f$ \nu_t @f$ evaluated pointwise from the
 * resolved velocity gradient, carried by SubgridStress as the explicit term
 * @f$ (2\nu_t S(u), S(v)) @f$ on the right-hand side (docs/les.md).
 */
enum class SubgridModel
{
   None,     ///< No model (DNS): the default.
   /// nu_t = coefficient everywhere: an explicit extra viscosity, for
   /// verification against the implicit viscous term (test S4), never a
   /// turbulence model.
   Constant,
   /// WALE (Nicoud & Ducros, Flow Turbul. Combust. 62 (1999) 183-200):
   /// built on the traceless symmetric part of the squared gradient; vanishes
   /// in pure shear (y^3 near a wall) but not in solid rotation.
   Wale,
   /// Sigma (Nicoud, Toda, Cabrit, Bose & Lee, Phys. Fluids 23 (2011)
   /// 085106): built on the singular values of the gradient; vanishes in pure
   /// shear (y^3), solid rotation, axisymmetric and isotropic expansion, and
   /// for any two-component flow.
   Sigma
};

/// The subgrid model's settings (deck section `physics.subgrid`).
struct SubgridOptions
{
   SubgridModel model = SubgridModel::None; ///< Which model.
   /// The model constant: C_w for WALE, C_sigma for the sigma model, nu_t
   /// itself for Constant. Negative = the model's published default
   /// (DefaultCoefficient).
   double coefficient = -1.0;
   /// The filter width as a multiple of the element's resolution:
   /// Delta_K = filter_width * h_K / k_u (h_K the volume-equivalent element
   /// size, k_u the velocity order), i.e. the node spacing by default.
   double filter_width = 1.0;

   /**
    * @brief The published constant of a model.
    * @param m The model.
    * @return 0.5 for WALE (Nicoud & Ducros's 0.5 for homogeneous isotropic
    *         turbulence; 0.325 is Ducros et al.'s later value), 1.35 for the
    *         sigma model (Nicoud et al. 2011), 0 otherwise.
    */
   static double DefaultCoefficient(SubgridModel m)
   {
      switch (m)
      {
         case SubgridModel::Wale: return 0.5;
         case SubgridModel::Sigma: return 1.35;
         default: return 0.0;
      }
   }

   /// @return The coefficient in use: the given one, or the model's default.
   double Coefficient() const
   {
      return coefficient >= 0.0 ? coefficient : DefaultCoefficient(model);
   }

   /// @return True when a model is active (Mult adds a term).
   bool Enabled() const { return model != SubgridModel::None; }
};

/**
 * @brief The pointwise models, on a 3 x 3 velocity gradient
 *        @c g[i][j] = d u_i / d x_j. A 2D flow is the 3D flow with
 *        @c u_z = 0 and @c d/dz = 0 (third row and column zero), so both
 *        models keep their 3D definitions there (the sigma model is then
 *        zero, by construction: a two-component flow has sigma_3 = 0).
 *
 * MFEM_HOST_DEVICE: evaluated inside SubgridStress's kernels (an nvcc TU in
 * a CUDA build) and on the host by the tests.
 */
namespace subgrid
{

/**
 * @brief The WALE eddy viscosity (Nicoud & Ducros 1999, eq. 13):
 *        @f$ \nu_t = (C_w\Delta)^2\, \frac{(S^d:S^d)^{3/2}}{(S:S)^{5/2} +
 *        (S^d:S^d)^{5/4}} @f$ with @f$ S^d_{ij} = \tfrac12(g^2_{ij} +
 *        g^2_{ji}) - \tfrac13\delta_{ij} g^2_{kk} @f$, @f$ g^2 = g\,g @f$.
 * @param g   Velocity gradient, g[i][j] = d u_i / d x_j.
 * @param cd2 @f$ (C_w\Delta)^2 @f$.
 * @return nu_t >= 0; 0 where both invariants vanish.
 */
MFEM_HOST_DEVICE inline double Wale(const double g[3][3], double cd2)
{
   double g2[3][3];
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         g2[i][j] = g[i][0] * g[0][j] + g[i][1] * g[1][j] + g[i][2] * g[2][j];
      }
   const double tr = (g2[0][0] + g2[1][1] + g2[2][2]) / 3.0;
   double ss = 0.0, sdsd = 0.0;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         const double s = 0.5 * (g[i][j] + g[j][i]);
         const double sd = 0.5 * (g2[i][j] + g2[j][i]) - (i == j ? tr : 0.0);
         ss += s * s;
         sdsd += sd * sd;
      }
   const double den = pow(ss, 2.5) + pow(sdsd, 1.25);
   if (den <= 0.0) { return 0.0; }
   return cd2 * pow(sdsd, 1.5) / den;
}

/**
 * @brief The singular values of a 3 x 3 matrix, descending, by one-sided
 *        (Hestenes) Jacobi rotations: the columns are rotated pairwise until
 *        they are orthogonal, and their norms are the singular values.
 *
 * Why not Cardano's formula on @f$ g^T g @f$ (Nicoud et al.'s appendix): a
 * closed-form eigenvalue is accurate to about sqrt(machine epsilon) near a
 * double root, in absolute terms. Near a wall the two small singular values
 * are O(y) and O(y^2) against an O(1) shear, so the formula returned
 * sigma_3 = 0 (and nu_t = 0) already at y = 1e-3 where the true value is
 * 2e-9 (checked against LAPACK's SVD): the y^3 wall behaviour, the model's
 * point, was lost exactly where it matters. One-sided Jacobi has high
 * RELATIVE accuracy for every singular value (Demmel & Veselic 1992): the
 * same case is right to 1e-12 relative down to y = 1e-6, in 2-3 sweeps.
 * @param g Input, g[i][j]; a 2D gradient pads with zeros.
 * @param s Output, s[0] >= s[1] >= s[2] >= 0.
 */
MFEM_HOST_DEVICE inline void SingularValues3(const double g[3][3], double s[3])
{
   double a[3][3];
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) { a[i][j] = g[i][j]; }
   for (int sweep = 0; sweep < 12; ++sweep)
   {
      double off = 0.0;
      for (int p = 0; p < 2; ++p)
         for (int q = p + 1; q < 3; ++q)
         {
            double al = 0.0, be = 0.0, ga = 0.0;
            for (int i = 0; i < 3; ++i)
            {
               al += a[i][p] * a[i][p];
               be += a[i][q] * a[i][q];
               ga += a[i][p] * a[i][q];
            }
            const double ab = sqrt(al * be);
            if (ab <= 0.0 || fabs(ga) <= 1e-17 * ab) { continue; }
            off = fabs(ga) / ab > off ? fabs(ga) / ab : off;
            const double zeta = (be - al) / (2.0 * ga);
            const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                             (fabs(zeta) + sqrt(1.0 + zeta * zeta));
            const double c = 1.0 / sqrt(1.0 + t * t);
            const double sn = c * t;
            for (int i = 0; i < 3; ++i)
            {
               const double ap = a[i][p], aq = a[i][q];
               a[i][p] = c * ap - sn * aq;
               a[i][q] = sn * ap + c * aq;
            }
         }
      if (off < 1e-16) { break; }
   }
   for (int j = 0; j < 3; ++j)
   {
      s[j] = sqrt(a[0][j] * a[0][j] + a[1][j] * a[1][j] + a[2][j] * a[2][j]);
   }
   // Sort descending (three values: two passes of a swap network).
   if (s[0] < s[1]) { const double t = s[0]; s[0] = s[1]; s[1] = t; }
   if (s[1] < s[2]) { const double t = s[1]; s[1] = s[2]; s[2] = t; }
   if (s[0] < s[1]) { const double t = s[0]; s[0] = s[1]; s[1] = t; }
}

/**
 * @brief The sigma eddy viscosity (Nicoud et al. 2011, eq. 11):
 *        @f$ \nu_t = (C_\sigma\Delta)^2\, \frac{\sigma_3(\sigma_1 - \sigma_2)
 *        (\sigma_2 - \sigma_3)}{\sigma_1^2} @f$, @f$ \sigma_1 \ge \sigma_2 \ge
 *        \sigma_3 \ge 0 @f$ the singular values of g (SingularValues3).
 * @param g   Velocity gradient, g[i][j] = d u_i / d x_j.
 * @param cd2 @f$ (C_\sigma\Delta)^2 @f$.
 * @return nu_t >= 0 (roundoff in the differences is clamped at 0).
 */
MFEM_HOST_DEVICE inline double Sigma(const double g[3][3], double cd2)
{
   double s[3];
   SingularValues3(g, s);
   if (s[0] <= 0.0) { return 0.0; }
   const double d12 = s[0] - s[1] > 0.0 ? s[0] - s[1] : 0.0;
   const double d23 = s[1] - s[2] > 0.0 ? s[1] - s[2] : 0.0;
   return cd2 * s[2] * d12 * d23 / (s[0] * s[0]);
}

/**
 * @brief The eddy viscosity of a model at one point.
 * @param model    The model.
 * @param g        Velocity gradient, g[i][j] = d u_i / d x_j (3 x 3; a 2D
 *                 flow pads with zeros).
 * @param cd2      @f$ (C\Delta)^2 @f$ of the element.
 * @param constant nu_t of SubgridModel::Constant.
 * @return nu_t >= 0.
 */
MFEM_HOST_DEVICE inline double Viscosity(SubgridModel model,
      const double g[3][3],
      double cd2, double constant)
{
   switch (model)
   {
      case SubgridModel::Constant: return constant;
      case SubgridModel::Wale: return Wale(g, cd2);
      case SubgridModel::Sigma: return Sigma(g, cd2);
      default: return 0.0;
   }
}

} // namespace subgrid
} // namespace incns
