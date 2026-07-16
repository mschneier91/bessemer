// CC.2 -- constant-pressure nullspace projector (SPEC par.4):
// plain Euclidean (l2) projection on true dofs, exact on constants,
// idempotent, and a no-op on already-projected vectors. Runs on both an
// enclosed box and a fully periodic box (identified true dofs).

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "precond/nullspace.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::BoxSpec;
using incns::ConstantPressureProjector;
using incns::MakeBoxMesh;
using incns::MixedSpaces;

namespace
{

std::unique_ptr<MixedSpaces> MakeSpaces(ParMesh& mesh, bool periodic)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   if (!periodic)
   {
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
   }
   Mesh serial = MakeBoxMesh(s);
   mesh = ParMesh(MPI_COMM_WORLD, serial);
   return std::make_unique<MixedSpaces>(mesh, 3, 2);
}

void CheckProjector(bool periodic)
{
   ParMesh mesh;
   auto spaces = MakeSpaces(mesh, periodic);
   ConstantPressureProjector proj(spaces->Pressure());
   const int n = spaces->Pressure().GetTrueVSize();

   // A pure constant is annihilated (to reduction roundoff).
   Vector q(n);
   q = 3.7;
   proj.Project(q);
   const double after = std::sqrt(InnerProduct(MPI_COMM_WORLD, q, q));
   EXPECT_LE(after, 1e-12);

   // Idempotent: the constant component of a projected vector is ~0, and a
   // second projection changes nothing.
   Vector r(n), r2(n);
   r.Randomize(21);
   proj.Project(r);
   EXPECT_LE(std::abs(proj.ConstantComponent(r)), 1e-13);
   r2 = r;
   proj.Project(r2);
   r2 -= r;
   const double drift = std::sqrt(InnerProduct(MPI_COMM_WORLD, r2, r2));
   const double scale = std::sqrt(InnerProduct(MPI_COMM_WORLD, r, r));
   EXPECT_LE(drift, 1e-14 * scale);
}

} // namespace

TEST(Nullspace, ProjectorEnclosedBox) { CheckProjector(/*periodic=*/false); }

TEST(Nullspace, ProjectorFullyPeriodicBox) { CheckProjector(/*periodic=*/true); }
