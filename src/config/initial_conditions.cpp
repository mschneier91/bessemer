#include "config/initial_conditions.hpp"

#include "exact/tgv2d.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace incns
{

using namespace mfem;

double ReichardtUPlus(double y_plus)
{
   const double kappa = 0.41, c = 7.8;
   return std::log(1.0 + kappa * y_plus) / kappa +
          c * (1.0 - std::exp(-y_plus / 11.0) -
               (y_plus / 11.0) * std::exp(-y_plus / 3.0));
}

std::unique_ptr<VectorCoefficient>
MakeChannelInitialVelocity(const Parameters& params)
{
   const int dim = params.mesh.dim;
   MFEM_VERIFY(params.geometry == MeshGeometry::Box && (dim == 2 || dim == 3),
               "initial_conditions: channel needs a 2D or 3D box (walls normal "
               "to y)");
   const double delta = 0.5 * params.mesh.lengths[1];
   const double fx = params.forcing.body_force[0];
   MFEM_VERIFY(fx > 0.0, "initial_conditions: channel takes u_tau = sqrt(f_x "
               "delta) from forcing.body_force, which needs f_x > 0");
   const double nu = params.nu;
   const double u_tau = std::sqrt(fx * delta);
   const double u_c = u_tau * ReichardtUPlus(u_tau * delta / nu);
   const double lx = params.mesh.lengths[0];
   const double lz = (dim == 3) ? params.mesh.lengths[2] : 1.0;

   // Divergence-free perturbation u' = curl(psi) with psi = g(eta) * (modes),
   // g = (1 - eta^2)^2: u' and its wall-normal derivative vanish at the walls.
   // Modes 1..3 in x (and z), random phases; the amplitude bounds the
   // streamwise part by perturbation * u_c (max |g'| = 8 / 3^{3/2}).
   struct Mode { double kx, kz, phase_a, phase_b; };
   std::vector<Mode> modes;
   std::mt19937 gen(static_cast<unsigned>(params.initial.seed));
   std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);
   const int nz_modes = (dim == 3) ? 3 : 1;
   for (int m = 1; m <= 3; ++m)
   {
      for (int n = 1; n <= nz_modes; ++n)
      {
         const double kz = (dim == 3) ? 2.0 * M_PI * n / lz : 0.0;
         const double a = phase(gen), b = phase(gen);
         modes.push_back({2.0 * M_PI* m / lx, kz, a, b});
      }
   }
   const double gmax = 8.0 / std::pow(3.0, 1.5);
   const double amp = params.initial.perturbation * u_c * delta /
                      (gmax * static_cast<double>(modes.size()));
   return std::make_unique<VectorFunctionCoefficient>(
             dim, [ = ](const Vector & x, Vector & u)
   {
      const double y = x(1);
      const double y_plus = (delta - std::abs(y - delta)) * u_tau / nu;
      u = 0.0;
      u(0) = u_tau * ReichardtUPlus(y_plus);
      if (amp == 0.0) { return; }
      const double eta = (y - delta) / delta;
      const double g = (1.0 - eta * eta) * (1.0 - eta * eta);
      const double dg = -4.0 * eta * (1.0 - eta * eta) / delta; // dg/dy
      for (const Mode& md : modes)
      {
         if (dim == 2)
         {
            // psi_z = g sin(kx x + a): u = d psi / dy, v = -d psi / dx.
            const double s = std::sin(md.kx * x(0) + md.phase_a);
            const double c = std::cos(md.kx * x(0) + md.phase_a);
            u(0) += amp * dg * s;
            u(1) += -amp * g * md.kx * c;
            continue;
         }
         // psi_x = g cos(kx x + a) sin(kz z + b),
         // psi_z = g sin(kx x + b) cos(kz z + a), psi_y = 0:
         // u = d psi_z/dy, v = d psi_x/dz - d psi_z/dx, w = -d psi_x/dy.
         const double xa = md.kx * x(0) + md.phase_a, xb = md.kx * x(0) + md.phase_b;
         const double zb = md.kz * x(2) + md.phase_b, za = md.kz * x(2) + md.phase_a;
         const double px = std::cos(xa) * std::sin(zb);
         const double pz = std::sin(xb) * std::cos(za);
         u(0) += amp * dg * pz;
         u(1) += amp * g * (md.kz * std::cos(xa) * std::cos(zb) -
                            md.kx * std::cos(xb) * std::cos(za));
         u(2) += -amp * dg * px;
      }
   });
}

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
   if (params.initial_velocity == "taylor_green_3d")
   {
      MFEM_VERIFY(dim == 3, "initial_conditions: taylor_green_3d is 3D");
      return std::make_unique<VectorFunctionCoefficient>(
                3, [](const Vector & x, Vector & u)
      {
         u(0) = std::sin(x(0)) * std::cos(x(1)) * std::cos(x(2));
         u(1) = -std::cos(x(0)) * std::sin(x(1)) * std::cos(x(2));
         u(2) = 0.0;
      });
   }
   if (params.initial_velocity == "channel")
   {
      return MakeChannelInitialVelocity(params);
   }
   MFEM_ABORT("initial_conditions: unknown initial_velocity '"
              << params.initial_velocity << "'");
   return nullptr;
}

} // namespace incns
