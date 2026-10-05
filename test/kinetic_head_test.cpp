// KineticHeadInterpolator: the device kernel (sum-factorized 1/2|u|^2 at the
// pressure nodes, E -> L by the restriction's left inverse) equals the host
// reference (ProjectCoefficient of 1/2|u_h|^2) on curved partitioned meshes,
// dim 2/3, velocity order ku, pressure order ku-1 (Taylor-Hood) and ku-2.

#include <gtest/gtest.h>

#include "post/kinetic_head.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::KineticHeadInterpolator;

namespace
{
void vel_fn(const Vector& x, Vector& u)
{
   const real_t X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   u(0) = sin(M_PI * X) * cos(1.3 * M_PI * Y) + 0.3 * Y + 0.1 * Z;
   u(1) = cos(0.7 * M_PI * X) * sin(M_PI * Y) - 0.2 * X * X;
   if (x.Size() == 3) { u(2) = sin(0.8 * X + 1.1 * Y) * cos(M_PI * Z); }
}
} // namespace

TEST(KineticHead, DeviceKernelMatchesHostProjection)
{
   for (int dim : {2, 3})
      for (int ku = 2; ku <= (dim == 2 ? 5 : 4); ++ku)
         for (int kp : {ku - 1, ku - 2})
         {
            if (kp < 1) { continue; }
            SCOPED_TRACE("dim=" + std::to_string(dim) + " ku=" +
                         std::to_string(ku) + " kp=" + std::to_string(kp));
            Mesh serial = (dim == 2)
                          ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
                          : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
            serial.SetCurvature(ku);
            serial.Transform([](const Vector & x, Vector & y)
            {
               y = x;
               y(0) += 0.05 * std::sin(M_PI * x(1));
               y(1) += 0.04 * std::sin(M_PI * x(0));
               if (x.Size() == 3) { y(2) += 0.03 * std::sin(M_PI * x(0)); }
            });
            ParMesh mesh(MPI_COMM_WORLD, serial);
            H1_FECollection ufec(ku, dim), pfec(kp, dim);
            ParFiniteElementSpace vfes(&mesh, &ufec, dim, Ordering::byNODES);
            ParFiniteElementSpace pfes(&mesh, &pfec);
            ParGridFunction u(&vfes), ke_dev(&pfes), ke_host(&pfes);
            VectorFunctionCoefficient uc(dim, vel_fn);
            u.ProjectCoefficient(uc);
            Vector ut(vfes.GetTrueVSize());
            u.GetTrueDofs(ut);
            u.SetFromTrueDofs(ut);

            KineticHeadInterpolator kh(vfes, pfes);
            ASSERT_TRUE(kh.UsesDevicePath());
            kh.Interpolate(u, ke_dev);
            KineticHeadInterpolator::InterpolateHost(u, ke_host);

            // Compare as true-dof vectors (owner values), globally reduced.
            Vector a(pfes.GetTrueVSize()), b(pfes.GetTrueVSize());
            ke_dev.GetTrueDofs(a);
            ke_host.GetTrueDofs(b);
            a -= b;
            real_t err = a.Normlinf(), scale = b.Normlinf();
            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_MAX, MPI_COMM_WORLD);
            EXPECT_GT(scale, 0.1);
            EXPECT_LE(err, 1e-13 * scale);
         }
}
