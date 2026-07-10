// Unit test for src/mesh/periodic_box: element geometry is quad/hex, and
// periodicity identifies opposite-boundary nodes (observed as a reduced vertex
// count). Mesh-level checks -- identical on every rank, so np-robust by
// construction.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "mfem.hpp"

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
