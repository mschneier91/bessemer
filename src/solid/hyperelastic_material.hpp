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

/// The hyperelastic material models (deck `solid.model`).
enum class HyperelasticMaterial
{
   /// St. Venant-Kirchhoff: W = lambda/2 (tr E)^2 + mu E : E.
   StVenantKirchhoff
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

} // namespace incns
