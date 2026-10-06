// AMR.2 -- RefinementMarker: indicator -> refinements. Synthetic indicators are
// functions of the element center, so the expected marks are known exactly and
// independent of the partition.
//  K1 relative / absolute thresholds mark exactly the expected elements;
//  K2 directions follow aniso_ratio; isotropic mode splits every direction;
//  K3 min_size drops directions (and unmarks); max_elements keeps the largest
//     indicators and respects the cap;
//  K4 3D: anisotropic conflicts across faces are resolved (np > 1) and the
//     refinement then succeeds with nc_limit 1 (valid mesh: exact transfer);
//  K5 with the real GradientIndicator, u varying in x only -> only X splits.

#include <gtest/gtest.h>

#include "amr/gradient_indicator.hpp"
#include "amr/refinement_marker.hpp"
#include "amr_test_util.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <functional>

using namespace mfem;
using incns::AmrParameters;
using incns::AmrThreshold;
using incns::MarkStats;
using incns::RefinementMarker;

namespace
{

using GFn = std::function<void(const Vector& c, double* g)>;

// g[d + dim e] from the element center.
Vector SyntheticG(ParMesh& mesh, const GFn& f)
{
   const int dim = mesh.Dimension();
   Vector g(dim * mesh.GetNE()), c(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      f(c, &g(dim * e));
   }
   return g;
}

// Local element index -> marked type (0 = not marked).
std::vector<int> TypeOf(const ParMesh& mesh, const Array<Refinement>& refs)
{
   std::vector<int> t(mesh.GetNE(), 0);
   for (int i = 0; i < refs.Size(); ++i) { t[refs[i].index] = refs[i].GetType(); }
   return t;
}

long long Sum(long long v)
{
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
   return v;
}

} // namespace

TEST(AmrMarker, K1_Thresholds)
{
   auto mesh = amr_test::NcBox(amr_test::Box(2, 4, false));
   const Vector g = SyntheticG(*mesh, [](const Vector & c, double * G)
   {
      G[0] = 1.0 + 4.0 * c(0); // eta = 1 + 4 x_c; max (x_c = 7/8) = 4.5
      G[1] = 0.0;
   });
   for (AmrThreshold mode : {AmrThreshold::Relative, AmrThreshold::Absolute})
   {
      AmrParameters o;
      o.threshold_mode = mode;
      o.theta = 0.5;      // relative: eta >= 2.25 <=> x_c >= 0.3125
      o.tolerance = 3.0;  // absolute: eta >= 3    <=> x_c >= 0.5
      Array<Refinement> refs;
      MarkStats st;
      RefinementMarker(o).Mark(*mesh, g, refs, &st);
      const auto t = TypeOf(*mesh, refs);
      Vector c(2);
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         mesh->GetElementCenter(e, c);
         const bool expect = (mode == AmrThreshold::Relative) ? c(0) >= 0.3125
                             : c(0) >= 0.5;
         EXPECT_EQ(t[e] != 0, expect) << "element center x = " << c(0);
         if (t[e] != 0) { EXPECT_EQ(t[e], Refinement::XY); }
      }
      EXPECT_EQ(st.marked, (mode == AmrThreshold::Relative) ? 12 : 8);
      EXPECT_DOUBLE_EQ(st.max_eta, 4.5);
      EXPECT_EQ(st.projected_ne, 16 + 3 * st.marked);
   }
}

TEST(AmrMarker, K2_Directions)
{
   auto mesh = amr_test::NcBox(amr_test::Box(2, 4, false));
   const Vector g = SyntheticG(*mesh, [](const Vector & c, double * G)
   {
      G[0] = 1.0 + c(0);
      G[1] = 0.6 * (1.0 + c(0)); // G_y / G_x = 0.6 everywhere
   });
   struct Case { bool aniso; double ratio; int type; } cases[] =
   {
      {true, 0.5, Refinement::XY}, {true, 0.7, Refinement::X},
      {false, 0.7, Refinement::XY}
   };
   for (const Case& k : cases)
   {
      AmrParameters o;
      o.anisotropic = k.aniso;
      o.aniso_ratio = k.ratio;
      o.theta = 1e-6; // mark everything
      Array<Refinement> refs;
      RefinementMarker(o).Mark(*mesh, g, refs);
      EXPECT_EQ(refs.Size(), mesh->GetNE());
      for (int i = 0; i < refs.Size(); ++i) { EXPECT_EQ(refs[i].GetType(), k.type); }
   }
}

TEST(AmrMarker, K3_MinSizeAndElementCap)
{
   // h = (0.25, 0.125): children would be (0.125, 0.0625).
   incns::BoxSpec spec = amr_test::Box(2, 4, false);
   spec.num_elems = {4, 8, 1};
   auto mesh = amr_test::NcBox(spec);
   const Vector flat = SyntheticG(*mesh, [](const Vector&, double * G)
   {
      G[0] = 1.0;
      G[1] = 1.0;
   });
   {
      AmrParameters o;
      o.theta = 1e-6;
      o.min_size = 0.1; // x child 0.125 >= 0.1 ok, y child 0.0625 < 0.1 dropped
      Array<Refinement> refs;
      RefinementMarker(o).Mark(*mesh, flat, refs);
      EXPECT_EQ(Sum(refs.Size()), 32);
      for (int i = 0; i < refs.Size(); ++i) { EXPECT_EQ(refs[i].GetType(), Refinement::X); }
      o.min_size = 0.2; // no direction may split
      RefinementMarker(o).Mark(*mesh, flat, refs);
      EXPECT_EQ(Sum(refs.Size()), 0);
   }
   {
      // Distinct indicators; cap = 32 + 3 * 5 -> exactly the 5 largest.
      const Vector g = SyntheticG(*mesh, [](const Vector & c, double * G)
      {
         G[0] = 1.0 + c(0) + 10.0 * c(1);
         G[1] = 0.5 * G[0];
      });
      AmrParameters o;
      o.theta = 1e-6;
      o.max_elements = 32 + 3 * 5;
      Array<Refinement> refs;
      MarkStats st;
      RefinementMarker(o).Mark(*mesh, g, refs, &st);
      EXPECT_EQ(st.marked, 5);
      EXPECT_LE(st.projected_ne, o.max_elements);
      // Every marked element's eta >= every unmarked element's eta.
      const auto t = TypeOf(*mesh, refs);
      double min_marked = 1e300, max_unmarked = 0.0;
      for (int e = 0; e < mesh->GetNE(); ++e)
      {
         const double eta = std::hypot(g(2 * e), g(2 * e + 1));
         if (t[e]) { min_marked = std::min(min_marked, eta); }
         else { max_unmarked = std::max(max_unmarked, eta); }
      }
      MPI_Allreduce(MPI_IN_PLACE, &min_marked, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &max_unmarked, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      EXPECT_GE(min_marked, max_unmarked);
   }
}

TEST(AmrMarker, K4_AnisotropicConflictsResolved3D)
{
   auto mesh = amr_test::NcBox(amr_test::Box(3, 4, false));
   // Lower half split in X, upper half in Y: the faces at z = 1/2 would be
   // split in different directions from the two sides.
   const Vector g = SyntheticG(*mesh, [](const Vector & c, double * G)
   {
      const bool low = c(2) < 0.5;
      G[0] = low ? 1.0 : 0.1;
      G[1] = low ? 0.1 : 1.0;
      G[2] = 0.1;
   });
   AmrParameters o;
   o.anisotropic = true;
   o.aniso_ratio = 0.9;
   o.theta = 1e-6;
   Array<Refinement> refs;
   MarkStats st;
   RefinementMarker(o).Mark(*mesh, g, refs, &st);
   EXPECT_EQ(st.marked, 64);
   if (mesh->GetNRanks() > 1)
   {
      std::set<int> conflicts;
      EXPECT_FALSE(mesh->AnisotropicConflict(refs, conflicts));
   }
   if (Mpi::Root())
   {
      mfem::out << "[marker] 3D conflict upgrades to XYZ: " << st.conflict_upgrades
                << std::endl;
   }

   // The refinement goes through, and the result is a valid NC mesh: a field
   // transferred to it is unchanged (exact integration of |u_h|^2).
   H1_FECollection fec(2, 3);
   ParFiniteElementSpace fes(mesh.get(), &fec, 3, Ordering::byNODES);
   ParGridFunction u(&fes);
   VectorFunctionCoefficient uc(3, [](const Vector & x, Vector & v)
   {
      v(0) = std::sin(x(0)) * x(2);
      v(1) = x(1) * x(1) - x(0);
      v(2) = std::cos(x(1) + x(2));
   });
   u.ProjectCoefficient(uc);
   {
      Vector t;
      u.GetTrueDofs(t);
      u.SetFromTrueDofs(t);
   }
   Vector z(3);
   z = 0.0;
   VectorConstantCoefficient zero(z);
   const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
   irs[Geometry::CUBE] = &IntRules.Get(Geometry::CUBE, 6);
   const double n0 = u.ComputeL2Error(zero, irs);
   mesh->GeneralRefinement(refs, 1, 1);
   fes.Update();
   u.Update();
   fes.UpdatesFinished();
   EXPECT_GT(mesh->GetGlobalNE(), 64);
   EXPECT_NEAR(u.ComputeL2Error(zero, irs), n0, 1e-12 * n0);
}

TEST(AmrMarker, K5_IndicatorDrivesDirections)
{
   incns::RuleBook rules; // outlives the mesh
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      auto mesh = amr_test::NcBox(amr_test::Box(dim, dim == 2 ? 4 : 3, false));
      H1_FECollection fec(3, dim);
      ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
      ParGridFunction u(&fes);
      VectorFunctionCoefficient uc(dim, [](const Vector & x, Vector & v)
      {
         v = 0.0;
         v(0) = std::tanh((x(0) - 0.5) / 0.1); // varies in x only
      });
      u.ProjectCoefficient(uc);
      incns::GradientIndicator ind(fes, rules);
      Vector g;
      ind.Compute(u, g);
      AmrParameters o;
      o.anisotropic = true;
      o.theta = 0.2;
      Array<Refinement> refs;
      MarkStats st;
      RefinementMarker(o).Mark(*mesh, g, refs, &st);
      EXPECT_GT(st.marked, 0);
      EXPECT_EQ(st.per_dir[0], st.marked);
      EXPECT_EQ(st.per_dir[1], 0);
      EXPECT_EQ(st.per_dir[2], 0);
   }
}
