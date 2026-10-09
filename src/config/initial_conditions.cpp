#include "config/initial_conditions.hpp"

#include "exact/tgv2d.hpp"

#include <cmath>

namespace incns
{

using namespace mfem;

std::unique_ptr<VectorCoefficient>
MakeInitialVelocity(const Parameters& params)
{
   const int dim = params.mesh.dim;

   if (params.initial_velocity == "zero")
   {
      Vector z(dim);
      z = 0.0;
      return std::make_unique<VectorConstantCoefficient>(z);
   }
   if (params.initial_velocity == "uniform")
   {
      const InitialFlowParameters ic = params.initial;
      double speed = 0.0;
      for (int d = 0; d < dim; ++d) { speed += ic.velocity[d] * ic.velocity[d]; }
      speed = std::sqrt(speed);
      return std::make_unique<VectorFunctionCoefficient>(
                dim, [ic, dim, speed](const Vector & x, Vector & u)
      {
         double r2 = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            u(d) = ic.velocity[d];
            r2 += (x(d) - ic.perturbation_center[d]) * (x(d) - ic.perturbation_center[d]);
         }
         if (dim >= 2) { u(1) += ic.perturbation * speed * std::exp(-r2); }
      });
   }
   if (params.initial_velocity == "taylor_green_2d")
   {
      MFEM_VERIFY(dim == 2, "initial_conditions: taylor_green_2d is 2D");
      const double nu = params.nu;
      return std::make_unique<VectorFunctionCoefficient>(
                2, [nu](const Vector & x, double t, Vector & u)
      { tgv2d::Velocity(x, t, nu, u); });
   }
   MFEM_ABORT("initial_conditions: unknown initial_velocity '"
              << params.initial_velocity << "'");
   return nullptr;
}

} // namespace incns
