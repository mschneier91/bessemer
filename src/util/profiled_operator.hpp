/**
 * @file profiled_operator.hpp
 * @brief An operator wrapper that times its applications under a named region.
 */
#pragma once

#include "mfem.hpp"

namespace incns
{

/**
 * @brief Forwards Mult / MultTranspose to a borrowed operator inside an
 *        INCNS_PROFILE region. No numerics.
 *
 * For the pieces a Krylov solver calls itself (its operator, its
 * preconditioner, the factors of a composed operator): they cannot be timed at
 * the call site, and several are header-only classes, where a region would be
 * compiled differently by the library and by its consumers. It is a
 * mfem::Solver so that it can stand in as a preconditioner.
 */
class ProfiledOperator : public mfem::Solver
{
public:
   /**
    * @brief Wrap an operator.
    * @param op               The operator (borrowed).
    * @param region           Region name for Mult (a string literal).
    * @param transpose_region Region name for MultTranspose; null uses @p region.
    */
   ProfiledOperator(const mfem::Operator& op, const char* region,
                    const char* transpose_region = nullptr);

   /**
    * @brief y = op(x), timed.
    * @param x Input.
    * @param y Output.
    */
   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   /**
    * @brief y = op^T(x), timed.
    * @param x Input.
    * @param y Output.
    */
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

   /// Ignored: the wrapped operator is fixed.
   void SetOperator(const mfem::Operator&) override {}

private:
   const mfem::Operator& op_;     ///< Wrapped operator (borrowed).
   const char* region_;           ///< Region name for Mult.
   const char* transpose_region_; ///< Region name for MultTranspose.
};

} // namespace incns
