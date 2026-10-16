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

namespace
{
// A device-aware SPD stand-in for the AMG: x = d .* b (+ x0 in iterative mode).
class DiagSolver : public Solver
{
public:
   explicit DiagSolver(const Vector& d) : Solver(d.Size()), d_(d) {}
   void SetOperator(const Operator&) override {}
   void Mult(const Vector& b, Vector& x) const override
   {
      Vector t(b.Size());
      t.UseDevice(true);
      t = b;
      t *= d_;
      if (iterative_mode) { x += t; }
      else { x = t; }
   }
private:
   const Vector& d_;
};
} // namespace

// incns::DeviceOrthoSolver gives mfem::OrthoSolver's results bitwise, for
// both iterative modes and repeated calls (the projection on the device
// instead of MFEM's host loop).
TEST(Nullspace, DeviceOrthoSolverMatchesMfem)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int n = 1000 + 37 * rank;
   Vector d(n);
   d.Randomize(3 + rank);
   d += 0.5;
   d.UseDevice(true);
   DiagSolver s1(d), s2(d);
   OrthoSolver ref(MPI_COMM_WORLD);
   incns::DeviceOrthoSolver dev(MPI_COMM_WORLD);
   ref.SetSolver(s1);
   dev.SetSolver(s2);
   for (int call = 0; call < 4; ++call)
   {
      const bool iterative = (call % 2 == 1);
      ref.iterative_mode = iterative;
      dev.iterative_mode = iterative;
      Vector b(n), x0(n);
      b.Randomize(10 + call + 7 * rank);
      b += 0.25 * call;
      x0.Randomize(20 + call + 7 * rank);
      Vector xr(x0), xd(x0);
      b.UseDevice(true);
      xr.UseDevice(true);
      xd.UseDevice(true);
      ref.Mult(b, xr);
      dev.Mult(b, xd);
      const double* r = xr.HostRead();
      const double* v = xd.HostRead();
      for (int i = 0; i < n; ++i) { ASSERT_EQ(r[i], v[i]) << "call " << call; }
   }
}
