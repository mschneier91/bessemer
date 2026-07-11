#include "config/initial_conditions.hpp"

#include "exact/tgv2d.hpp"

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
