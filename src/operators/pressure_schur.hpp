/**
 * @file pressure_schur.hpp
 * @brief Sprint-1 pressure Schur complement approximation: scaled pressure mass.
 */
#ifndef INCNS_OPERATORS_PRESSURE_SCHUR_HPP
#define INCNS_OPERATORS_PRESSURE_SCHUR_HPP

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

namespace incns
{

/**
 * @brief The Sprint-1 pressure Schur block: @c S_hat = (1/nu) M_p, applied as
 *        @c S_hat^{-1} = nu * M_p^{-1} with a Jacobi (diagonal) mass inverse.
 *
 * Deliberately simple: spectrally equivalent to the true Schur complement for
 * @b steady Stokes and dt-independent. Known, accepted limitation: it degrades
 * as dt shrinks in the unsteady solve -- growing iteration counts at small dt
 * are expected in Sprint 1, not a bug. The Cahouet--Chabard deep-dive replaces
 * this in Sprint 2 (do not build any of it here).
 *
 * With grad-div augmentation (sub-sprint 1.10) the scale becomes @c nu + gamma;
 * pass that as @p scale.
 */
class PressureMassSchur : public mfem::Solver
{
public:
   /**
    * @brief Assemble the pressure mass diagonal.
    * @param pfes  Pressure finite element space (borrowed).
    * @param rules Quadrature source (borrowed, must outlive this).
    * @param scale Scaling of the inverse: @c nu (plus @c gamma with grad-div).
    */
   PressureMassSchur(mfem::ParFiniteElementSpace& pfes, const RuleBook& rules,
                     double scale);

   /**
    * @brief Apply @c y = scale * diag(M_p)^{-1} x.
    * @param x Input pressure true-dof vector.
    * @param y Output pressure true-dof vector.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   /// Required by mfem::Solver; the operator is fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

private:
   double scale_;                 ///< nu (or nu + gamma with grad-div).
   mfem::ParBilinearForm mass_;   ///< Pressure mass form (partial assembly).
   mfem::Vector inv_diag_;        ///< Precomputed scale / diag(M_p).
};

} // namespace incns

#endif // INCNS_OPERATORS_PRESSURE_SCHUR_HPP
