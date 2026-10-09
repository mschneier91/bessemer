// precond/viscous_ratio -- docs/design/rotational_schur_velocity_mg_spec.md 8, V1 (the
// diagnostic part; the p-multigrid parts of V1-V3 are not built):
//  - on uniform n^dim boxes, VHatMax(sigma, nu) = nu / (sigma (1/(n p))^2) to
//    1e-12 for p in {3, 4, 5}, 2D and 3D (the spec is 3D only);
//  - n = 3 -> 6 multiplies it by 4 to 1e-12;
//  - c_p is mesh-independent by construction and matches the spec's
//    calibrated 3D values (20.22, 29.53, 40.92 for p = 3, 4, 5) to 1e-2.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "precond/viscous_ratio.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cstdio>

using namespace mfem;

namespace
{

double VHat(int dim, int n, int p, double sigma, double nu, double* cp)
{
   incns::RuleBook rules; // before the mesh (rule-keyed caches)
   incns::BoxSpec box;
   box.dim = dim;
   box.num_elems = {n, n, n};
   box.lengths = {1.0, 1.0, 1.0};
   box.periodic = {false, false, false};
   Mesh serial = incns::MakeBoxMesh(box);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace vfes(&mesh, &fec, dim);
   incns::ViscousRatioDiagnostic d(vfes, rules);
   if (cp) { *cp = d.Cp(); }
   return d.VHatMax(sigma, nu);
}

} // namespace

TEST(ViscousRatio, ExactOnUniformMeshes)
{
   const double sigma = 1.5, nu = 0.01;
   const double spec_cp3d[3] = {20.22, 29.53, 40.92};
   for (int dim : {2, 3})
   {
      for (int p = 3; p <= 5; ++p)
      {
         double cp = 0.0;
         const double v3 = VHat(dim, 3, p, sigma, nu, &cp);
         const double v6 = VHat(dim, 6, p, sigma, nu, nullptr);
         const double h = 1.0 / (3.0 * p);
         const double exact = nu / (sigma * h * h);
         if (Mpi::Root())
         {
            std::printf("dim=%d p=%d: c_p = %.4f, vhat(n=3) = %.6f (exact "
                        "%.6f)\n", dim, p, cp, v3, exact);
         }
         EXPECT_NEAR(v3, exact, 1e-12 * exact) << "dim=" << dim << " p=" << p;
         EXPECT_NEAR(v6 / v3, 4.0, 1e-12) << "dim=" << dim << " p=" << p;
         if (dim == 3) { EXPECT_NEAR(cp, spec_cp3d[p - 3], 1e-2) << "p=" << p; }
      }
   }
}
