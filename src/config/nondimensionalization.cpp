#include "config/nondimensionalization.hpp"

namespace incns
{

using namespace mfem;

namespace
{
void VerifyScales(const Nondimensionalization& nd)
{
   MFEM_VERIFY(nd.mode == ScalingMode::Dimensional,
               "nondimensionalization: wrappers are for Dimensional mode "
               "(dimensionless data needs no wrapping)");
   MFEM_VERIFY(nd.L_ref > 0.0 && nd.U_ref > 0.0,
               "nondimensionalization: reference scales must be positive");
}
} // namespace

std::unique_ptr<VectorCoefficient>
WrapDimensionalVelocity(VectorFieldFn f, const Nondimensionalization& nd,
                        int dim)
{
   VerifyScales(nd);
   const double L = nd.L_ref, U = nd.U_ref, T = nd.TRef();
   return std::make_unique<VectorFunctionCoefficient>(
             dim, [f = std::move(f), L, U, T](const Vector & x_star,
                double t_star, Vector & u)
   {
      Vector x_dim(x_star);
      x_dim *= L;
      f(x_dim, T * t_star, u);
      u /= U;
   });
}

std::unique_ptr<VectorCoefficient>
WrapDimensionalForcing(VectorFieldFn f, const Nondimensionalization& nd,
                       int dim)
{
   VerifyScales(nd);
   const double L = nd.L_ref, U = nd.U_ref, T = nd.TRef();
   return std::make_unique<VectorFunctionCoefficient>(
             dim, [f = std::move(f), L, U, T](const Vector & x_star,
                double t_star, Vector & fval)
   {
      Vector x_dim(x_star);
      x_dim *= L;
      f(x_dim, T * t_star, fval);
      fval *= L / (U * U);
   });
}

} // namespace incns
