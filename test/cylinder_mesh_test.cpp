// mesh/cylinder_channel -- the DFG cylinder-channel quad mesh.
//  M1 geometry: area = L H - pi R^2 and cylinder perimeter = 2 pi R to the
//     geometry order's accuracy (an inconsistent shared node would break
//     both), inflow/outflow/wall lengths exact, det J > 0 at every quadrature
//     point, and order 1 (straight ring edges) visibly worse;
//  M2 it partitions (np 1/2/4), including nonconforming-ready for AMR, and an
//     NC refinement keeps the area (the curved geometry is refined exactly);
//  M3 mesh/square_cylinder (Joly et al. 2012's domain): area = 180 x 120 - 1,
//     boundary lengths exact (inflow/outflow 120, sides 360, square 4), the
//     square's faces on +-1/2, nodes increasing, the corner cells the
//     smallest along each face, the near wake uniform at wake_h, det J > 0,
//     and it partitions nonconforming-ready and refines keeping the area.

#include <gtest/gtest.h>

#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "mesh/square_cylinder.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

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

TEST(CylinderMesh, M3_SquareCylinderDomain)
{
   incns::SquareCylinderSpec spec; // the defaults: a fine graded mesh
   for (bool x : {true, false})
   {
      const std::vector<double> n = incns::SquareCylinderNodes(spec, x);
      for (std::size_t k = 1; k < n.size(); ++k) { ASSERT_GT(n[k], n[k - 1]); }
      EXPECT_DOUBLE_EQ(n.front(), x ? -spec.upstream : -spec.half_height);
      EXPECT_DOUBLE_EQ(n.back(), x ? spec.downstream : spec.half_height);
      // The square's faces are nodes; along a face the corner cells are the
      // smallest.
      const auto ia = std::find(n.begin(), n.end(), -0.5), ib = std::find(n.begin(),
                      n.end(), 0.5);
      ASSERT_NE(ia, n.end());
      ASSERT_NE(ib, n.end());
      ASSERT_EQ(ib - ia, spec.n_face);
      const double corner = *(ia + 1) - *ia;
      for (auto it = ia + 1; it != ib; ++it) { EXPECT_GE(*(it + 1) - *it, corner - 1e-14); }
      if (x)
      {
         // Near wake: the last cell before wake_end is (close to) wake_h.
         const auto we = std::lower_bound(n.begin(), n.end(), spec.wake_end - 1e-12);
         ASSERT_NE(we, n.end());
         EXPECT_NEAR(*we - * (we - 1), spec.wake_h, 0.2 * spec.wake_h);
      }
   }

   Mesh serial = incns::MakeSquareCylinderMesh(spec);
   const double W = spec.upstream + spec.downstream, H = 2.0 * spec.half_height;
   EXPECT_NEAR(Area(serial), W * H - spec.side * spec.side, 1e-9 * W * H);
   EXPECT_NEAR(BoundaryLength(serial, incns::kSquareInflow), H, 1e-11);
   EXPECT_NEAR(BoundaryLength(serial, incns::kSquareOutflow), H, 1e-11);
   EXPECT_NEAR(BoundaryLength(serial, incns::kSquareSides), 2.0 * W, 1e-10);
   EXPECT_NEAR(BoundaryLength(serial, incns::kSquareBody), 4.0 * spec.side, 1e-12);
   for (int e = 0; e < serial.GetNE(); ++e)
   {
      ElementTransformation& T = *serial.GetElementTransformation(e);
      T.SetIntPoint(&Geometries.GetCenter(T.GetGeometryType()));
      ASSERT_GT(T.Weight(), 0.0) << "element " << e;
   }

   std::unique_ptr<ParMesh> pm = incns::PartitionMesh(
                                    serial, /*nonconforming=*/true);
   Array<int> refs;
   for (int e = 0; e < pm->GetNE(); e += 3) { refs.Append(e); }
   pm->GeneralRefinement(refs, 1, 1);
   EXPECT_NEAR(Sum(Area(*pm)), W * H - spec.side * spec.side, 1e-9 * W * H);
}

TEST(CylinderMesh, M4_TurekHronFlag)
{
   // The Turek & Hron (2006) geometry: channel 2.5 x 0.41, cylinder of radius
   // 0.05 at (0.2, 0.2), flag 0.35 x 0.02 from the cylinder to x = 0.6.
   CylinderChannelSpec s;
   s.length = 2.5;
   s.flag_length = 0.35;
   s.flag_thickness = 0.02;
   s.n_flag = 6;
   s.n_flag_y = 2;
   s.order = 4;
   const double r = s.radius, b = 0.5 * s.flag_thickness;
   const double tip = s.cx + r + s.flag_length;
   const double chord = std::sqrt(r * r - b * b);
   // The flag: the strip |y - cy| < b between the circle and the tip.
   const double flag_area = 2 * b * (tip - s.cx) - (b * chord + r * r *
                            std::asin(b / r));
   const double clamp = 2 * r * std::asin(b / r);

   Mesh fluid = incns::MakeCylinderChannelMesh(s);
   s.solid = true;
   Mesh solid = incns::MakeCylinderChannelMesh(s);

   // The arcs are polynomial interpolants of the circle (order 4).
   EXPECT_NEAR(Area(fluid), s.height * s.length - M_PI * r * r - flag_area, 1e-9);
   EXPECT_NEAR(Area(solid), flag_area, 1e-12);
   EXPECT_NEAR(BoundaryLength(fluid, incns::kCylinderBody), 2 * M_PI * r - clamp,
               1e-7 * r);
   EXPECT_NEAR(BoundaryLength(solid, incns::kCylinderBody), clamp, 1e-9 * r);
   const double surface = 2 * (tip - s.cx - chord) + 2 * b;
   EXPECT_NEAR(BoundaryLength(fluid, incns::kCylinderFlag), surface, 1e-13);
   EXPECT_NEAR(BoundaryLength(solid, incns::kCylinderFlag), surface, 1e-13);
   EXPECT_NEAR(BoundaryLength(fluid, incns::kCylinderInflow), s.height, 1e-13);
   EXPECT_NEAR(BoundaryLength(fluid, incns::kCylinderWalls), 2 * s.length, 1e-12);
   EXPECT_EQ(solid.bdr_attributes.Size(), 2);
   for (Mesh* m : {&fluid, &solid})
   {
      for (int e = 0; e < m->GetNE(); ++e)
      {
         ElementTransformation& T = *m->GetElementTransformation(e);
         const IntegrationRule& ir = IntRules.Get(T.GetGeometryType(), 8);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            T.SetIntPoint(&ir.IntPoint(q));
            ASSERT_GT(T.Weight(), 0.0) << "element " << e;
         }
      }
   }

   // The interface carries the same nodes on both sides: every high-order
   // node of the solid's flag boundary is a node of the fluid's.
   auto boundary_nodes = [](Mesh & m)
   {
      std::vector<std::pair<double, double>> out;
      const GridFunction& x = *m.GetNodes();
      const FiniteElementSpace& f = *x.FESpace();
      Array<int> vd;
      for (int be = 0; be < m.GetNBE(); ++be)
      {
         if (m.GetBdrAttribute(be) != incns::kCylinderFlag) { continue; }
         f.GetBdrElementVDofs(be, vd);
         const int n = vd.Size() / 2;
         for (int j = 0; j < n; ++j) { out.push_back({x(vd[j]), x(vd[j + n])}); }
      }
      return out;
   };
   const auto fn = boundary_nodes(fluid), sn = boundary_nodes(solid);
   EXPECT_EQ(fn.size(), sn.size());
   for (const auto& p : sn)
   {
      double best = 1e300;
      for (const auto& q : fn)
      {
         best = std::min(best, std::hypot(p.first - q.first, p.second - q.second));
      }
      ASSERT_LT(best, 1e-14) << "solid node (" << p.first << ", " << p.second
                             << ") is not on the fluid's interface";
   }

   // Without a flag the mesh is the one before the flag existed.
   CylinderChannelSpec d;
   Mesh plain = incns::MakeCylinderChannelMesh(d);
   d.flag_thickness = 0.03; // ignored without a length
   Mesh same = incns::MakeCylinderChannelMesh(d);
   Vector diff(*plain.GetNodes());
   diff -= *same.GetNodes();
   EXPECT_EQ(diff.Normlinf(), 0.0);
   EXPECT_EQ(plain.GetNE(), 208);
}
