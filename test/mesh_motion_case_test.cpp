// Deck-driven mesh deformation (equation mesh_motion, ale/mesh_motion_case),
// at np 1/2/4. The decks set only the geometry, the boundary groups and a
// small order and step count; everything else is the deck defaults.
//  C1 2D: a rectangle's bottom pushed up by a parabola ramped in time: after
//     every move the bottom nodes sit at X + g(X, t) (to roundoff), the other
//     sides do not move, det F > 0 and the element Jacobian ratio stays
//     positive; the peak node moved by the parabola's peak;
//  C2 3D: a cube's bottom face pushed up by a sin^2 bump: the same checks;
//  C3 a constant displacement of every boundary is a translation: det F = 1
//     and every node moves by it;
//  C4 model neo_hookean: a rotation of every boundary by 60 degrees (four
//     moves) is a rigid rotation of the mesh, det F = 1 and every node at R X,
//     which the linear model cannot give.

#include <gtest/gtest.h>

#include "ale/mesh_motion_case.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using namespace mfem;
using incns::MeshMotionCase;
using incns::Parameters;

namespace
{

/// Largest |x - X| over the nodes of the mesh not on the moving boundaries
/// that lie on the given box face (coordinate @p dir = @p value).
double FaceMotion(MeshMotionCase& c, int dir, double value)
{
   const ParGridFunction& d = c.Motion().NodeDisplacement();
   const ParFiniteElementSpace& fes = *d.ParFESpace();
   const Vector& X = c.Motion().ReferenceNodes();
   const double* dh = d.HostRead();
   const double* xh = X.HostRead();
   double m = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      if (std::abs(xh[fes.DofToVDof(i, dir)] - value) > 1e-12) { continue; }
      for (int k = 0; k < fes.GetVDim(); ++k)
      {
         m = std::max(m, std::abs(dh[fes.DofToVDof(i, k)]));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

} // namespace

TEST(MeshMotionCase, C1_RectangleParabola2D)
{
   const Parameters p = Parameters::LoadYAMLString(R"(
equation: mesh_motion
mesh: {dim: 2, elements: [8, 4], lengths: [2.0, 1.0], periodic: [false, false]}
mesh_motion: {order: 3, steps: 2}
time: {t_final: 1.0}
boundary_conditions:
  - select: [ymin]
    type: displacement
    expression: ["0", "A * t * 4 * x * (L - x) / L^2"]
    parameters: {A: 0.3, L: 2.0}
  - {select: [xmin, xmax, ymax], type: fixed}
)");
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   MeshMotionCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   EXPECT_EQ(c.Motion().MoveCount(), 2);
   EXPECT_DOUBLE_EQ(c.Motion().Time(), 1.0);
   EXPECT_LT(c.BoundaryError(), 1e-12);
   // The peak at x = 1 is a vertex: it moved by A.
   EXPECT_NEAR(c.MaxDisplacement(), 0.3, 1e-12);
   EXPECT_LT(FaceMotion(c, 1, 1.0), 1e-14); // top
   EXPECT_LT(FaceMotion(c, 0, 0.0), 1e-14); // left
   EXPECT_LT(FaceMotion(c, 0, 2.0), 1e-14); // right
   double lo, hi;
   c.Motion().JacobianRange(lo, hi);
   EXPECT_GT(lo, 0.0);
   EXPECT_LT(lo, 1.0); // the bottom elements were squeezed
   EXPECT_GT(incns::MinJacobianRatio(*mesh, c.Motion().Order()), 0.0);
   // The summary is written (and removed: np 1/2/4 run concurrently).
   const std::string summary = "mesh_motion_case_c1_np" +
                               std::to_string(Mpi::WorldSize()) + ".json";
   c.WriteSummary(summary, "ok");
   if (Mpi::Root())
   {
      std::ifstream f(summary);
      std::stringstream text;
      text << f.rdbuf();
      EXPECT_NE(text.str().find("\"boundary_error\""), std::string::npos);
      std::remove(summary.c_str());
   }
}

TEST(MeshMotionCase, C2_CubeBump3D)
{
   const Parameters p = Parameters::LoadYAMLString(R"(
equation: mesh_motion
mesh: {dim: 3, elements: [4, 4, 4], lengths: [1.0, 1.0, 1.0],
       periodic: [false, false, false]}
mesh_motion: {order: 2, steps: 2}
time: {t_final: 1.0}
boundary_conditions:
  - select: [ymin]
    type: displacement
    expression: ["0", "A * t * sin(pi * x)^2 * sin(pi * z)^2", "0"]
    parameters: {A: 0.3}
  - {select: [xmin, xmax, ymax, zmin, zmax], type: fixed}
)");
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   MeshMotionCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   EXPECT_LT(c.BoundaryError(), 1e-12);
   EXPECT_NEAR(c.MaxDisplacement(), 0.3, 1e-12); // (0.5, 0, 0.5) is a vertex
   EXPECT_LT(FaceMotion(c, 1, 1.0), 1e-14);
   EXPECT_LT(FaceMotion(c, 0, 0.0), 1e-14);
   EXPECT_LT(FaceMotion(c, 2, 1.0), 1e-14);
   double lo, hi;
   c.Motion().JacobianRange(lo, hi);
   EXPECT_GT(lo, 0.0);
   EXPECT_GT(incns::MinJacobianRatio(*mesh, c.Motion().Order()), 0.0);
}

TEST(MeshMotionCase, C3_ConstantDisplacementTranslates)
{
   const Parameters p = Parameters::LoadYAMLString(R"(
equation: mesh_motion
mesh: {dim: 2, elements: [3, 3], lengths: [1.0, 1.0], periodic: [false, false]}
mesh_motion: {order: 2}
boundary_conditions:
  - {select: [all], type: displacement, value: [0.1, -0.05]}
)");
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   MeshMotionCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   const ParGridFunction& d = c.Motion().NodeDisplacement();
   const ParFiniteElementSpace& fes = *d.ParFESpace();
   const double* dh = d.HostRead();
   double err = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      err = std::max(err, std::abs(dh[fes.DofToVDof(i, 0)] - 0.1));
      err = std::max(err, std::abs(dh[fes.DofToVDof(i, 1)] + 0.05));
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LT(err, 1e-8);
   double lo, hi;
   c.Motion().JacobianRange(lo, hi);
   EXPECT_NEAR(lo, 1.0, 1e-8);
   EXPECT_NEAR(hi, 1.0, 1e-8);
}

TEST(MeshMotionCase, C4_NeoHookeanRotatesRigidly)
{
   const Parameters p = Parameters::LoadYAMLString(R"(
equation: mesh_motion
mesh: {dim: 2, elements: [4, 4], lengths: [1.0, 1.0], periodic: [false, false]}
mesh_motion: {model: neo_hookean, order: 2, steps: 4}
time: {t_final: 1.0}
boundary_conditions:
  - select: [all]
    type: displacement
    expression: ["cos(a * t) * (x - 0.5) - sin(a * t) * (y - 0.5) + 0.5 - x",
                 "sin(a * t) * (x - 0.5) + cos(a * t) * (y - 0.5) + 0.5 - y"]
    parameters: {a: 1.0471975511965976}
)");
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   MeshMotionCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   EXPECT_STREQ(c.Motion().Model().Name(), "neo_hookean");
   const ParGridFunction& d = c.Motion().NodeDisplacement();
   const ParFiniteElementSpace& fes = *d.ParFESpace();
   const Vector& X = c.Motion().ReferenceNodes();
   const double* dh = d.HostRead();
   const double* xh = X.HostRead();
   const double a = 1.0471975511965976;
   double err = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      const double x = xh[fes.DofToVDof(i, 0)] - 0.5;
      const double y = xh[fes.DofToVDof(i, 1)] - 0.5;
      err = std::max(err, std::abs(dh[fes.DofToVDof(i, 0)] -
                                   (std::cos(a) * x - std::sin(a) * y - x)));
      err = std::max(err, std::abs(dh[fes.DofToVDof(i, 1)] -
                                   (std::sin(a) * x + std::cos(a) * y - y)));
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LT(err, 1e-8);
   double lo, hi;
   c.Motion().JacobianRange(lo, hi);
   EXPECT_NEAR(lo, 1.0, 1e-8);
   EXPECT_NEAR(hi, 1.0, 1e-8);
}
