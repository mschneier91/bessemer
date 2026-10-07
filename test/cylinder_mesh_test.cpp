// mesh/cylinder_channel -- the DFG cylinder-channel quad mesh.
//  M1 geometry: area = L H - pi R^2 and cylinder perimeter = 2 pi R to the
//     geometry order's accuracy (an inconsistent shared node would break
//     both), inflow/outflow/wall lengths exact, det J > 0 at every quadrature
//     point, and order 1 (straight ring edges) visibly worse;
//  M2 it partitions (np 1/2/4), including nonconforming-ready for AMR, and an
//     NC refinement keeps the area (the curved geometry is refined exactly).

#include <gtest/gtest.h>

#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::CylinderChannelSpec;

namespace
{

double Area(Mesh& mesh)
{
   double a = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T.GetGeometryType(), 16);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         a += ir.IntPoint(q).weight * T.Weight();
      }
   }
   return a;
}

double BoundaryLength(Mesh& mesh, int attr)
{
   double l = 0.0;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      if (mesh.GetBdrAttribute(be) != attr) { continue; }
      ElementTransformation& T = *mesh.GetBdrElementTransformation(be);
      const IntegrationRule& ir = IntRules.Get(T.GetGeometryType(), 16);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         l += ir.IntPoint(q).weight * T.Weight();
      }
   }
   return l;
}

double Sum(double v, MPI_Comm comm = MPI_COMM_WORLD)
{
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, comm);
   return v;
}

} // namespace

TEST(CylinderMesh, M1_Geometry)
{
   CylinderChannelSpec s; // DFG defaults
   s.order = 3;
   Mesh mesh = incns::MakeCylinderChannelMesh(s);
   const double area_ex = s.length * s.height - M_PI * s.radius * s.radius;
   EXPECT_NEAR(Area(mesh), area_ex, 1e-7 * area_ex);
   EXPECT_NEAR(BoundaryLength(mesh, incns::kCylinderBody), 2 * M_PI * s.radius,
               1e-6 * 2 * M_PI * s.radius);
   EXPECT_NEAR(BoundaryLength(mesh, incns::kCylinderInflow), s.height, 1e-13);
   EXPECT_NEAR(BoundaryLength(mesh, incns::kCylinderOutflow), s.height, 1e-13);
   EXPECT_NEAR(BoundaryLength(mesh, incns::kCylinderWalls), 2 * s.length, 1e-12);

   // det J > 0 everywhere.
   double min_det = 1e300;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      const IntegrationRule& ir = IntRules.Get(T.GetGeometryType(), 10);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         min_det = std::min(min_det, T.Jacobian().Det());
      }
   }
   EXPECT_GT(min_det, 0.0);

   // Geometry order matters: order 1 (straight ring edges) is visibly wrong.
   CylinderChannelSpec s1 = s;
   s1.order = 1;
   Mesh straight = incns::MakeCylinderChannelMesh(s1);
   EXPECT_GT(std::abs(Area(straight) - area_ex), 1e-5);
}

TEST(CylinderMesh, M2_PartitionAndRefine)
{
   for (bool nc : {false, true})
   {
      CylinderChannelSpec s;
      Mesh serial = incns::MakeCylinderChannelMesh(s);
      const double area_ex = s.length * s.height - M_PI * s.radius * s.radius;
      auto mesh = incns::PartitionMesh(serial, nc);
      EXPECT_NEAR(Sum(Area(*mesh)), area_ex, 1e-7 * area_ex);
      if (nc)
      {
         // Refine every element touching the cylinder (curved), twice.
         for (int pass = 0; pass < 2; ++pass)
         {
            Array<int> refs;
            Vector c(2);
            for (int e = 0; e < mesh->GetNE(); ++e)
            {
               mesh->GetElementCenter(e, c);
               if (std::hypot(c(0) - s.cx, c(1) - s.cy) < 0.09) { refs.Append(e); }
            }
            mesh->GeneralRefinement(refs, 1, 1);
         }
         EXPECT_NEAR(Sum(Area(*mesh)), area_ex, 1e-7 * area_ex);
         double perim = 0.0;
         perim = Sum(BoundaryLength(*mesh, incns::kCylinderBody));
         EXPECT_NEAR(perim, 2 * M_PI * s.radius, 1e-6 * 2 * M_PI * s.radius);
      }
   }
}
