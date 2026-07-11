// Unit tests for src/bc/boundary_conditions: essential true-dof counts against
// closed-form boundary node counts, time-dependent Dirichlet projection
// (SetTime + re-project), and pressure null-space detection from the BC set.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <set>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;

namespace
{
BoxSpec OpenBox2D(int n)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {n, n, 0};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// Global count of essential true dofs (ess lists are disjoint across ranks).
long GlobalEssCount(const BoundaryConditions& bc, MPI_Comm comm)
{
   long local = bc.EssentialTrueDofs().Size(), global = 0;
   MPI_Allreduce(&local, &global, 1, MPI_LONG, MPI_SUM, comm);
   return global;
}
} // namespace

// Closed form for an n x n grid of Q_k quads, velocity vdim = 2:
//   whole boundary: 4nk nodes  -> 2 * 4nk essential tdofs
//   one side:       nk+1 nodes -> 2 * (nk+1)
TEST(BoundaryConditions, EssentialTrueDofCounts)
{
   const int n = 4, k = 3;
   Mesh serial = MakeBoxMesh(OpenBox2D(n));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, k, k - 1);

   Vector zero_vec(2);
   zero_vec = 0.0;
   VectorConstantCoefficient zero(zero_vec);

   // All four sides Dirichlet (MakeCartesian2D attributes are 1..4).
   BoundaryConditions bc_all(spaces.Velocity());
   for (int attr = 1; attr <= 4; ++attr)
   {
      bc_all.AddVelocityDirichlet(attr, zero);
   }
   EXPECT_EQ(GlobalEssCount(bc_all, MPI_COMM_WORLD), 2L * 4 * n * k);

   // A single side.
   BoundaryConditions bc_one(spaces.Velocity());
   bc_one.AddVelocityDirichlet(1, zero);
   EXPECT_EQ(GlobalEssCount(bc_one, MPI_COMM_WORLD), 2L * (n * k + 1));
}

// SetTime advances the Dirichlet data and ProjectDirichlet imposes it: with
// u_D = (t, 2t) (constant in space), every essential true dof carries t or 2t,
// and both components appear.
TEST(BoundaryConditions, TimeDependentDirichletProjection)
{
   Mesh serial = MakeBoxMesh(OpenBox2D(3));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);

   VectorFunctionCoefficient ud(2, [](const Vector&, double t, Vector & v)
   {
      v(0) = t;
      v(1) = 2.0 * t;
   });

   BoundaryConditions bc(spaces.Velocity());
   for (int attr = 1; attr <= 4; ++attr) { bc.AddVelocityDirichlet(attr, ud); }

   for (double t : {1.5, 4.0})
   {
      bc.SetTime(t);
      ParGridFunction u(&spaces.Velocity());
      u = 0.0;
      bc.ProjectDirichlet(u);
      Vector u_true(spaces.Velocity().GetTrueVSize());
      u.GetTrueDofs(u_true);

      std::set<int> seen; // which components appeared on this rank
      for (int i = 0; i < bc.EssentialTrueDofs().Size(); ++i)
      {
         const double val = u_true(bc.EssentialTrueDofs()[i]);
         const bool is_c0 = std::abs(val - t) < 1e-13 * t;
         const bool is_c1 = std::abs(val - 2.0 * t) < 1e-13 * t;
         EXPECT_TRUE(is_c0 || is_c1) << "unexpected boundary value " << val;
         seen.insert(is_c0 ? 0 : 1);
      }
      // Some rank may own few boundary dofs, but globally both components must
      // be present; check on ranks that own any essential dofs at all.
      if (bc.EssentialTrueDofs().Size() > 1)
      {
         EXPECT_EQ(seen.size(), 2u);
      }
   }
}

// Null-space detection: exists iff no outflow AND full Dirichlet coverage
// (or no boundary at all -- fully periodic).
TEST(BoundaryConditions, PressureNullspaceDetection)
{
   // Fully periodic box: no boundary, empty BC set => null space exists.
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {4, 4, 0};
      Mesh serial = MakeBoxMesh(s);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      MixedSpaces spaces(mesh, 2, 1);
      BoundaryConditions bc(spaces.Velocity());
      EXPECT_TRUE(bc.PressureNullspaceExists());
      EXPECT_FALSE(bc.HasOutflow());
   }

   Mesh serial = MakeBoxMesh(OpenBox2D(3));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 2, 1);
   Vector zero_vec(2);
   zero_vec = 0.0;
   VectorConstantCoefficient zero(zero_vec);

   // Enclosed cavity (all Dirichlet) => null space exists.
   {
      BoundaryConditions bc(spaces.Velocity());
      for (int attr = 1; attr <= 4; ++attr) { bc.AddVelocityDirichlet(attr, zero); }
      EXPECT_TRUE(bc.PressureNullspaceExists());
   }
   // One outflow side => the pressure level is fixed, no null space.
   {
      BoundaryConditions bc(spaces.Velocity());
      for (int attr = 1; attr <= 3; ++attr) { bc.AddVelocityDirichlet(attr, zero); }
      bc.AddOutflow(4);
      EXPECT_TRUE(bc.HasOutflow());
      EXPECT_FALSE(bc.PressureNullspaceExists());
   }
   // Partial Dirichlet with an unassigned (natural) side => no null space.
   {
      BoundaryConditions bc(spaces.Velocity());
      for (int attr = 1; attr <= 3; ++attr) { bc.AddVelocityDirichlet(attr, zero); }
      EXPECT_FALSE(bc.PressureNullspaceExists());
   }

   // x-periodic channel: only bottom (1) and top (3) own real boundary faces;
   // the left/right attributes are periodic pseudo-boundary. Dirichlet on the
   // two real walls alone encloses the domain => null space exists.
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {4, 4, 0};
      s.lengths = {1.0, 1.0, 1.0};
      s.periodic = {true, false, false};
      Mesh chan_serial = MakeBoxMesh(s);
      ParMesh chan(MPI_COMM_WORLD, chan_serial);
      MixedSpaces chan_spaces(chan, 2, 1);
      BoundaryConditions bc(chan_spaces.Velocity());
      bc.AddVelocityDirichlet(1, zero);
      EXPECT_FALSE(bc.PressureNullspaceExists()); // top wall still natural
      bc.AddVelocityDirichlet(3, zero);
      EXPECT_TRUE(bc.PressureNullspaceExists());
   }
}
