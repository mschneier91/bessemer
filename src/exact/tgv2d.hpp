/**
 * @file tgv2d.hpp
 * @brief Analytic 2D Taylor-Green vortex (shared by solvers, drivers, tests).
 */
#ifndef INCNS_EXACT_TGV2D_HPP
#define INCNS_EXACT_TGV2D_HPP

#include "mfem.hpp"

#include <cmath>

namespace incns
{
namespace tgv2d
{

/**
 * @brief TGV velocity @c u = (sin x cos y, -cos x sin y) e^{-2 nu t} on the
 *        fully periodic @c [0,2pi]^2 box.
 *
 * Exact solution of BOTH unforced incompressible Navier-Stokes and unforced
 * unsteady Stokes: @c lap(u) = -2u so @c u_t = nu*lap(u), and the convective
 * term is a pure gradient absorbed into the NSE pressure. For @b Stokes the
 * pressure is identically zero; the NSE pressure is Pressure() below.
 *
 * @param x   Evaluation point.
 * @param t   Time.
 * @param nu  Kinematic viscosity.
 * @param u   Output velocity (size 2).
 */
inline void Velocity(const mfem::Vector& x, double t, double nu,
                     mfem::Vector& u)
{
   const double decay = std::exp(-2.0 * nu * t);
   u(0) = std::sin(x[0]) * std::cos(x[1]) * decay;
   u(1) = -std::cos(x[0]) * std::sin(x[1]) * decay;
}

/**
 * @brief The NSE pressure @c p = (cos 2x + cos 2y)/4 * e^{-4 nu t} (zero mean).
 *
 * Balances the convective term of the NSE; for unsteady @b Stokes the exact
 * pressure is zero, not this.
 *
 * @param x  Evaluation point.
 * @param t  Time.
 * @param nu Kinematic viscosity.
 * @return Pressure value.
 */
inline double Pressure(const mfem::Vector& x, double t, double nu)
{
   const double decay2 = std::exp(-4.0 * nu * t);
   return 0.25 * (std::cos(2.0 * x[0]) + std::cos(2.0 * x[1])) * decay2;
}

/**
 * @brief Velocity coefficient for the TGV (time-dependent; use SetTime).
 * @param nu Kinematic viscosity.
 * @return A 2-component mfem::VectorFunctionCoefficient.
 */
inline mfem::VectorFunctionCoefficient VelocityCoefficient(double nu)
{
   return mfem::VectorFunctionCoefficient(
             2, [nu](const mfem::Vector & x, double t, mfem::Vector & u)
   { Velocity(x, t, nu, u); });
}

} // namespace tgv2d
} // namespace incns

#endif // INCNS_EXACT_TGV2D_HPP
