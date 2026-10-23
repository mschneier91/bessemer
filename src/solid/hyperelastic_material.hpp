/**
 * @file hyperelastic_material.hpp
 * @brief Hyperelastic materials at a point, host and device: the first
 *        Piola-Kirchhoff stress, its derivative and the strain energy
 *        (docs/design/hyperelasticity_spec.md).
 *
 * Matrices are dim x dim arrays in column-major order, A(i, j) = a[i + dim j],
 * as mfem::DenseMatrix stores them. F is the deformation gradient
 * I + grad_X u on the reference configuration.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/// The hyperelastic material models (deck `solid.model`), and the linear one.
enum class HyperelasticMaterial
{
   /// St. Venant-Kirchhoff: W = lambda/2 (tr E)^2 + mu E : E.
   StVenantKirchhoff,
   /// Compressible neo-Hookean: W = mu/2 (tr C - dim) - mu ln J + lambda/2
   /// (ln J)^2.
   NeoHookean,
   /// Small-strain linear elasticity (LinearElastic): LinearElasticity's
   /// operator, not a deck model.
   LinearElastic
};

/**
 * @brief St. Venant-Kirchhoff: W(E) = lambda/2 (tr E)^2 + mu E : E with the
 *        Green-Lagrange strain E = (F^T F - I) / 2, so S = lambda tr(E) I +
 *        2 mu E and P = F S.
 *
 * Not polyconvex: W stays finite as det F -> 0, so strong compression can
 * invert an element without the energy resisting (the spec's section 1).
 */
struct StVenantKirchhoff
{
   /**
    * @brief The second and first Piola-Kirchhoff stresses.
    * @param F      Deformation gradient (DIM x DIM).
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @param S      Output: S = lambda tr(E) I + 2 mu E.
    * @param P      Output: P = F S.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static void Stress(const mfem::real_t* F, mfem::real_t lambda,
                                       mfem::real_t mu, mfem::real_t* S,
                                       mfem::real_t* P)
   {
      mfem::real_t E[DIM * DIM] = {};
      mfem::real_t tr = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t c = 0.0; // (F^T F)(i, j)
            for (int a = 0; a < DIM; a++) { c += F[a + DIM * i] * F[a + DIM * j]; }
            E[i + DIM * j] = 0.5 * (c - (i == j ? 1.0 : 0.0));
         }
         tr += E[j + DIM * j];
      }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            S[i + DIM * j] = 2.0 * mu * E[i + DIM * j] + (i == j ? lambda* tr : 0.0);
         }
      }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t p = 0.0;
            for (int k = 0; k < DIM; k++) { p += F[i + DIM * k] * S[k + DIM * j]; }
            P[i + DIM * j] = p;
         }
      }
   }

   /**
    * @brief The derivative of P in the direction dF at F:
    *        dP = dF S + F dS, dS = lambda tr(dE) I + 2 mu dE,
    *        dE = sym(F^T dF).
    * @param F      Deformation gradient.
    * @param S      Second Piola-Kirchhoff stress at F (from Stress()).
    * @param dF     Direction.
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @param dP     Output.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static void Tangent(const mfem::real_t* F,
                                        const mfem::real_t* S,
                                        const mfem::real_t* dF, mfem::real_t lambda,
                                        mfem::real_t mu, mfem::real_t* dP)
   {
      mfem::real_t dS[DIM * DIM] = {};
      mfem::real_t tr = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t c = 0.0; // (F^T dF + dF^T F)(i, j)
            for (int a = 0; a < DIM; a++)
            {
               c += F[a + DIM * i] * dF[a + DIM * j] + dF[a + DIM * i] * F[a + DIM * j];
            }
            dS[i + DIM * j] = 0.5 * c; // dE for now
         }
         tr += dS[j + DIM * j];
      }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            dS[i + DIM * j] = 2.0 * mu * dS[i + DIM * j] + (i == j ? lambda* tr : 0.0);
         }
      }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t p = 0.0;
            for (int k = 0; k < DIM; k++)
            {
               p += dF[i + DIM * k] * S[k + DIM * j] + F[i + DIM * k] * dS[k + DIM * j];
            }
            dP[i + DIM * j] = p;
         }
      }
   }

   /**
    * @param F      Deformation gradient.
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @return The strain energy density W(F).
    */
   template <int DIM>
   MFEM_HOST_DEVICE static mfem::real_t Energy(const mfem::real_t* F,
         mfem::real_t lambda, mfem::real_t mu)
   {
      mfem::real_t tr = 0.0, ee = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t c = 0.0;
            for (int a = 0; a < DIM; a++) { c += F[a + DIM * i] * F[a + DIM * j]; }
            const mfem::real_t e = 0.5 * (c - (i == j ? 1.0 : 0.0));
            ee += e * e;
            if (i == j) { tr += e; }
         }
      }
      return 0.5 * lambda * tr * tr + mu * ee;
   }
};

/**
 * @brief Compressible neo-Hookean: W(F) = mu/2 (tr C - dim) - mu ln J +
 *        lambda/2 (ln J)^2, C = F^T F, J = det F, so P = mu F + (lambda ln J -
 *        mu) F^-T and S = mu I + (lambda ln J - mu) C^-1.
 *
 * Polyconvex, and W -> infinity as J -> 0: compression costs ever more, so
 * an element resists being folded flat -- what St. Venant-Kirchhoff lacks
 * (it lost strong ellipticity near J = 0.7, and failed as a mesh-motion
 * pseudo-solid past ~45 degrees on the turned cylinder). At small strain it is
 * linear elasticity with the same lambda and mu. J must stay positive (ln J).
 */
struct NeoHookean
{
   /**
    * @brief F^-1 and det F.
    * @param F  The deformation gradient (DIM x DIM).
    * @param Fi Output: F^-1.
    * @return det F.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static mfem::real_t Inverse(const mfem::real_t* F,
         mfem::real_t* Fi)
   {
      if constexpr(DIM == 2)
      {
         const mfem::real_t J = F[0] * F[3] - F[2] * F[1];
         Fi[0] = F[3] / J;
         Fi[1] = -F[1] / J;
         Fi[2] = -F[2] / J;
         Fi[3] = F[0] / J;
         return J;
      }
      else
      {
         // Cofactors, column-major: F(i, j) = F[i + 3 j].
         const mfem::real_t c00 = F[4] * F[8] - F[7] * F[5];
         const mfem::real_t c01 = F[7] * F[2] - F[1] * F[8];
         const mfem::real_t c02 = F[1] * F[5] - F[4] * F[2];
         const mfem::real_t J = F[0] * c00 + F[3] * c01 + F[6] * c02;
         Fi[0] = c00 / J;
         Fi[1] = c01 / J;
         Fi[2] = c02 / J;
         Fi[3] = (F[6] * F[5] - F[3] * F[8]) / J;
         Fi[4] = (F[0] * F[8] - F[6] * F[2]) / J;
         Fi[5] = (F[3] * F[2] - F[0] * F[5]) / J;
         Fi[6] = (F[3] * F[7] - F[6] * F[4]) / J;
         Fi[7] = (F[6] * F[1] - F[0] * F[7]) / J;
         Fi[8] = (F[0] * F[4] - F[3] * F[1]) / J;
         return J;
      }
   }

   /**
    * @brief The second and first Piola-Kirchhoff stresses.
    * @param F      Deformation gradient (DIM x DIM), det F > 0.
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @param S      Output: S = mu I + (lambda ln J - mu) C^-1.
    * @param P      Output: P = mu F + (lambda ln J - mu) F^-T.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static void Stress(const mfem::real_t* F, mfem::real_t lambda,
                                       mfem::real_t mu, mfem::real_t* S,
                                       mfem::real_t* P)
   {
      mfem::real_t Fi[DIM * DIM];
      const mfem::real_t J = Inverse<DIM>(F, Fi);
      const mfem::real_t a = lambda * log(J) - mu;
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t ci = 0.0; // C^-1(i, j) = (F^-1 F^-T)(i, j)
            for (int k = 0; k < DIM; k++) { ci += Fi[i + DIM * k] * Fi[j + DIM * k]; }
            S[i + DIM * j] = (i == j ? mu : 0.0) + a * ci;
            P[i + DIM * j] = mu * F[i + DIM * j] + a * Fi[j + DIM * i];
         }
      }
   }

   /**
    * @brief dP = mu dF + lambda tr(F^-1 dF) F^-T + (mu - lambda ln J) F^-T
    *        dF^T F^-T.
    * @param F      Deformation gradient (DIM x DIM).
    * @param S      Unused (the tangent is formed from F).
    * @param dF     Its increment.
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @param dP     Output.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static void Tangent(const mfem::real_t* F,
                                        const mfem::real_t* S,
                                        const mfem::real_t* dF, mfem::real_t lambda,
                                        mfem::real_t mu, mfem::real_t* dP)
   {
      MFEM_CONTRACT_VAR(S);
      mfem::real_t Fi[DIM * DIM], M[DIM * DIM];
      const mfem::real_t J = Inverse<DIM>(F, Fi);
      const mfem::real_t b = mu - lambda * log(J);
      mfem::real_t tr = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t m = 0.0; // M = F^-1 dF
            for (int k = 0; k < DIM; k++) { m += Fi[i + DIM * k] * dF[k + DIM * j]; }
            M[i + DIM * j] = m;
         }
         tr += M[j + DIM * j];
      }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            mfem::real_t a = 0.0; // (F^-T dF^T F^-T)(i, j) = (M F^-1)(j, i)
            for (int k = 0; k < DIM; k++) { a += M[j + DIM * k] * Fi[k + DIM * i]; }
            dP[i + DIM * j] = mu * dF[i + DIM * j] + lambda * tr * Fi[j + DIM * i] +
                              b * a;
         }
      }
   }

   /// @return The strain energy density W(F).
   template <int DIM>
   MFEM_HOST_DEVICE static mfem::real_t Energy(const mfem::real_t* F,
         mfem::real_t lambda, mfem::real_t mu)
   {
      mfem::real_t Fi[DIM * DIM];
      const mfem::real_t lj = log(Inverse<DIM>(F, Fi));
      mfem::real_t c = 0.0; // tr C = |F|^2
      for (int i = 0; i < DIM * DIM; i++) { c += F[i] * F[i]; }
      return 0.5 * mu * (c - DIM) - mu * lj + 0.5 * lambda * lj * lj;
   }
};

/**
 * @brief Small-strain linear elasticity: the stress lambda tr(eps) I + 2 mu
 *        eps of eps = sym(grad u) is linear in grad u, so the tangent is the
 *        stress law itself and needs no linearization point (it is St.
 *        Venant-Kirchhoff's tangent at F = I, S = 0, bit for bit).
 *
 * HyperelasticOperator applies it through the tangent only: the residual is
 * K u, from grad u itself (not from F - I, which would lose digits).
 */
struct LinearElastic
{
   /**
    * @brief dP = lambda tr(dF) I + mu (dF + dF^T).
    * @param F      Unused (the tangent does not depend on it).
    * @param S      Unused.
    * @param dF     The increment's gradient (DIM x DIM).
    * @param lambda First Lame parameter.
    * @param mu     Shear modulus.
    * @param dP     Output.
    */
   template <int DIM>
   MFEM_HOST_DEVICE static void Tangent(const mfem::real_t* F,
                                        const mfem::real_t* S,
                                        const mfem::real_t* dF, mfem::real_t lambda,
                                        mfem::real_t mu, mfem::real_t* dP)
   {
      MFEM_CONTRACT_VAR(F);
      MFEM_CONTRACT_VAR(S);
      mfem::real_t tr = 0.0;
      for (int i = 0; i < DIM; i++) { tr += dF[i + DIM * i]; }
      for (int j = 0; j < DIM; j++)
      {
         for (int i = 0; i < DIM; i++)
         {
            dP[i + DIM * j] = mu * (dF[i + DIM * j] + dF[j + DIM * i]) +
                              (i == j ? lambda* tr : 0.0);
         }
      }
   }
};

} // namespace incns
