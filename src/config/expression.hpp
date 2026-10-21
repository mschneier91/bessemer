/**
 * @file expression.hpp
 * @brief Analytic expressions of (x, y, z, t) read from a deck, e.g. a
 *        boundary displacement `0.1*4*x*(1 - x)*t` (docs/mesh_motion.md).
 */
#pragma once

#include "mfem.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @brief A scalar expression of the coordinates x, y, z and the time t,
 *        parsed once and evaluated many times (host only).
 *
 * Grammar: numbers (`2`, `0.5`, `1e-3`), the variables x, y, z, t, the
 * constants pi and e, named parameters given at parse time, the operators
 * `+ - * /` and `^` (or `**`; right-associative, binding tighter than a unary
 * minus: `-x^2` is -(x^2)), parentheses, and the functions sin, cos, tan,
 * asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs (one argument) and
 * min, max, pow, atan2 (two). A coordinate the mesh does not have (z in 2D)
 * is 0.
 */
class Expression
{
public:
   /**
    * @brief Parse @p text. Throws std::invalid_argument with the position of
    *        the first error (unknown name, wrong argument count, unbalanced
    *        parentheses, trailing input).
    * @param text       The expression.
    * @param parameters Named constants usable in it (their names may not be
    *                   x, y, z, t, pi, e or a function name).
    */
   explicit Expression(const std::string& text,
                       const std::map<std::string, double>& parameters = {});

   /**
    * @param x Coordinates (size 1 to 3; missing ones are 0).
    * @param t Time.
    * @return The value.
    */
   double Eval(const mfem::Vector& x, double t) const;

   /**
    * @param v (x, y, z, t).
    * @return The value.
    */
   double Eval(const double v[4]) const;

   /// @return The text it was parsed from.
   const std::string& Text() const { return text_; }

   /// @return Whether the expression reads the variable t.
   bool DependsOnTime() const { return uses_time_; }

   /// One instruction of the postfix program.
   struct Op
   {
      int code = 0;       ///< What the instruction does (expression.cpp).
      double value = 0.0; ///< A constant's value.
      int index = 0;      ///< A variable's index (x, y, z, t = 0..3).
   };

private:
   std::string text_;      ///< The source text.
   std::vector<Op> prog_;  ///< Postfix program.
   int depth_ = 0;         ///< Largest evaluation stack the program needs.
   bool uses_time_ = false; ///< Whether t appears.
};

/**
 * @brief A vector of expressions, one per component: d(X, t) on a boundary.
 */
class VectorExpression
{
public:
   /**
    * @param components One expression per component (the mesh dimension).
    * @param parameters Named constants usable in every component.
    */
   VectorExpression(const std::vector<std::string>& components,
                    const std::map<std::string, double>& parameters = {});

   /// @return The number of components.
   int Size() const { return static_cast<int>(c_.size()); }

   /**
    * @param x Coordinates.
    * @param t Time.
    * @param v Output, resized to Size().
    */
   void Eval(const mfem::Vector& x, double t, mfem::Vector& v) const;

   /**
    * @return A coefficient evaluating it at the physical point of the
    *         element transformation (it keeps this object alive).
    * @param self A shared pointer to this object.
    */
   static std::unique_ptr<mfem::VectorCoefficient>
   Coefficient(const std::shared_ptr<const VectorExpression>& self);

private:
   std::vector<Expression> c_; ///< The components.
};

} // namespace incns
