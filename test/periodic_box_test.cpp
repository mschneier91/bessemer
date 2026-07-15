// Unit test for src/mesh/periodic_box: element geometry is quad/hex, and
// periodicity identifies opposite-boundary nodes (observed as a reduced vertex
// count). Mesh-level checks -- identical on every rank, so np-robust by
// construction.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "mfem.hpp"

#include <set>
#include <vector>

using namespace mfem;
using incns::BoxSpec;
using incns::MakeBoxMesh;

TEST(PeriodicBox, Elements2DAreSquares)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   Mesh m = MakeBoxMesh(s);
   ASSERT_EQ(m.GetNE(), 16);
   for (int e = 0; e < m.GetNE(); ++e)
   {
      EXPECT_EQ(m.GetElementBaseGeometry(e), Geometry::SQUARE);
   }
}

TEST(PeriodicBox, Elements3DAreCubes)
{
   BoxSpec s;
   s.dim = 3;
   s.num_elems = {3, 3, 3};
   Mesh m = MakeBoxMesh(s);
   ASSERT_EQ(m.GetNE(), 27);
   for (int e = 0; e < m.GetNE(); ++e)
   {
      EXPECT_EQ(m.GetElementBaseGeometry(e), Geometry::CUBE);
   }
}

// Fully periodic box: opposite boundaries are identified, so the vertex count
// drops from (n+1)^d to n^d while the element count is unchanged.
TEST(PeriodicBox, PeriodicIdentifiesVertices2D)
{
   BoxSpec np;
   np.dim = 2;
   np.num_elems = {4, 4, 0};
   np.periodic = {false, false, false};
   BoxSpec fp = np;
   fp.periodic = {true, true, true};

   Mesh mnp = MakeBoxMesh(np);
   Mesh mfp = MakeBoxMesh(fp);
   EXPECT_EQ(mnp.GetNV(), 25); // (4+1)^2
   EXPECT_EQ(mfp.GetNV(), 16); // 4^2
   EXPECT_EQ(mnp.GetNE(), mfp.GetNE());
}

TEST(PeriodicBox, PeriodicIdentifiesVertices3D)
{
   BoxSpec np;
   np.dim = 3;
   np.num_elems = {3, 3, 3};
   np.periodic = {false, false, false};
   BoxSpec fp = np;
   fp.periodic = {true, true, true};

   Mesh mnp = MakeBoxMesh(np);
   Mesh mfp = MakeBoxMesh(fp);
   EXPECT_EQ(mnp.GetNV(), 64); // (3+1)^3
   EXPECT_EQ(mfp.GetNV(), 27); // 3^3
   EXPECT_EQ(mnp.GetNE(), mfp.GetNE());
}

// Periodic in x only: x collapses (4 cols), y stays open (5 rows) -> 20 vertices.
TEST(PeriodicBox, PartialPeriodicityVertexCount2D)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   s.periodic = {true, false, false};
   Mesh m = MakeBoxMesh(s);
   EXPECT_EQ(m.GetNV(), 20);
}

// Wall-normal tanh stretching (H2): the endpoints are preserved, the near-wall
// element is the smallest, and the spacing grows monotonically toward the
// centre then shrinks again (symmetric). Structural, np-robust.
TEST(PeriodicBox, TwoSidedTanhStretchClustersAtWalls)
{
   const int n = 8;
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {3, n, 0};       // stretch only the y (wall-normal) direction
   s.lengths = {1.0, 1.0};
   s.periodic = {true, false, false};
   s.stretch = {incns::Stretch::None, incns::Stretch::TwoSidedTanh,
                incns::Stretch::None
               };
   s.stretch_beta = {2.0, 2.0, 2.0};
   Mesh m = MakeBoxMesh(s);

   // Collect the distinct y-node coordinates (one column suffices; the mesh is
   // tensor-product). Gather from all vertices and de-duplicate.
   std::set<double> yset;
   for (int v = 0; v < m.GetNV(); ++v) { yset.insert(m.GetVertex(v)[1]); }
   std::vector<double> y(yset.begin(), yset.end());
   ASSERT_EQ(static_cast<int>(y.size()), n + 1);

   // Endpoints preserved (so periodicity in a stretched direction would work).
   EXPECT_NEAR(y.front(), 0.0, 1e-14);
   EXPECT_NEAR(y.back(), 1.0, 1e-14);

   // Symmetric clustering: near-wall spacing is smallest, mid-channel largest.
   const double h_wall = y[1] - y[0];
   const double h_mid = y[n / 2] - y[n / 2 - 1];
   const double h_uniform = 1.0 / n;
   EXPECT_LT(h_wall, 0.75 * h_uniform);   // genuinely clustered at the wall
   EXPECT_GT(h_mid, 1.25 * h_uniform);    // coarsened in the middle
   EXPECT_NEAR(h_wall, y[n] - y[n - 1], 1e-13); // symmetric about the centre
}
