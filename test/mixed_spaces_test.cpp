// Unit test for src/spaces/mixed_spaces: global true-dof counts match the
// closed-form node count for the (k_u, k_p) pair, and the block offsets are
// self-consistent. Uses a partitioned ParMesh, so np in {2, 4} exercises the
// parallel dof assembly, not just serial counting.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

using namespace mfem;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;

// Unique scalar H1 nodes on a Cartesian n^d grid of Q_k elements: a periodic
// direction contributes n*k nodes, an open direction n*k + 1.
static long ScalarNodeCount(const BoxSpec& s, int k)
{
   long total = 1;
   for (int d = 0; d < s.dim; ++d)
   {
      total *= s.periodic[d] ? (long)s.num_elems[d] * k
               : (long)s.num_elems[d] * k + 1;
   }
   return total;
}

static void CheckCounts(const BoxSpec& s, int ku, int kp)
{
   Mesh serial = MakeBoxMesh(s);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   MixedSpaces sp(pmesh, ku, kp);

   const long scal_u = ScalarNodeCount(s, ku);
   const long scal_p = ScalarNodeCount(s, kp);
   EXPECT_EQ(sp.GlobalVelocityTDofs(), (HYPRE_BigInt)(s.dim) * scal_u);
   EXPECT_EQ(sp.GlobalPressureTDofs(), (HYPRE_BigInt)scal_p);
}

TEST(MixedSpaces, DofCounts2D_Q3Q2_FullyPeriodic)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   s.periodic = {true, true, true};
   CheckCounts(s, 3, 2);
}

TEST(MixedSpaces, DofCounts2D_Q3Q2_NonPeriodic)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   s.periodic = {false, false, false};
   CheckCounts(s, 3, 2);
}

TEST(MixedSpaces, DofCounts2D_Q3Q2_PartialPeriodic)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 5, 0};
   s.periodic = {true, false, false};
   CheckCounts(s, 3, 2);
}

TEST(MixedSpaces, DofCounts3D_Q3Q2_FullyPeriodic)
{
   BoxSpec s;
   s.dim = 3;
   s.num_elems = {3, 3, 3};
   s.periodic = {true, true, true};
   CheckCounts(s, 3, 2);
}

TEST(MixedSpaces, DofCounts3D_Q2Q1_NonPeriodic)
{
   BoxSpec s;
   s.dim = 3;
   s.num_elems = {3, 3, 3};
   s.periodic = {false, false, false};
   CheckCounts(s, 2, 1);
}

TEST(MixedSpaces, BlockOffsetsConsistent)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   Mesh serial = MakeBoxMesh(s);
   ParMesh pmesh(MPI_COMM_WORLD, serial);
   MixedSpaces sp(pmesh, 3, 2);

   const Array<int>& off = sp.BlockTrueOffsets();
   ASSERT_EQ(off.Size(), 3);
   EXPECT_EQ(off[0], 0);
   EXPECT_EQ(off[1], sp.Velocity().GetTrueVSize());
   EXPECT_EQ(off[2] - off[1], sp.Pressure().GetTrueVSize());
   EXPECT_EQ(off[2],
             sp.Velocity().GetTrueVSize() + sp.Pressure().GetTrueVSize());
}
