// Unit test for src/post/pressure_mean: a field with a known offset has that
// mass-weighted mean, and SubtractMean drives the mean to zero. Exact (to
// quadrature precision) for a constant shift and for a polynomial field that
// lives in the discrete space. Runs on a partitioned ParMesh so np {2,4}
// exercises the parallel integral reductions.

#include <gtest/gtest.h>

#include "post/pressure_mean.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MassWeightedMean;
using incns::RuleBook;
using incns::SubtractMean;

namespace
{
ParMesh MakeParBox(const BoxSpec& s)
{
   Mesh serial = MakeBoxMesh(s);
   return ParMesh(MPI_COMM_WORLD, serial);
}
} // namespace

// Constant field on a fully periodic box: the mean is exactly the constant, and
// shifting zeroes it.
TEST(PressureMean, ConstantOffsetPeriodic2D)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   ParMesh mesh = MakeParBox(s);
   H1_FECollection fec(2, mesh.Dimension());
   ParFiniteElementSpace pfes(&mesh, &fec);
   RuleBook rules;

   ParGridFunction p(&pfes);
   p = 3.75;
   EXPECT_NEAR(MassWeightedMean(p, rules), 3.75, 1e-12);
   SubtractMean(p, rules);
   EXPECT_NEAR(MassWeightedMean(p, rules), 0.0, 1e-12);
}

TEST(PressureMean, ConstantOffset3D)
{
   BoxSpec s;
   s.dim = 3;
   s.num_elems = {3, 3, 3};
   ParMesh mesh = MakeParBox(s);
   H1_FECollection fec(2, mesh.Dimension());
   ParFiniteElementSpace pfes(&mesh, &fec);
   RuleBook rules;

   ParGridFunction p(&pfes);
   p = -2.0;
   EXPECT_NEAR(MassWeightedMean(p, rules), -2.0, 1e-12);
   SubtractMean(p, rules);
   EXPECT_NEAR(MassWeightedMean(p, rules), 0.0, 1e-12);
}

// Linear field p = x on an open [0,2*pi]^2 box: x lies in the Q2 space, so the
// mass-weighted mean is exactly the domain-centroid value pi. (Open, not
// periodic, so there is no seam discontinuity in x.)
TEST(PressureMean, LinearFieldMeanIsCentroid2D)
{
   BoxSpec s;
   s.dim = 2;
   s.num_elems = {4, 4, 0};
   s.periodic = {false, false, false};
   ParMesh mesh = MakeParBox(s);
   H1_FECollection fec(2, mesh.Dimension());
   ParFiniteElementSpace pfes(&mesh, &fec);
   RuleBook rules;

   ParGridFunction p(&pfes);
   FunctionCoefficient fx([](const Vector & x) { return x[0]; });
   p.ProjectCoefficient(fx);

   EXPECT_NEAR(MassWeightedMean(p, rules), M_PI, 1e-10);
   SubtractMean(p, rules);
   EXPECT_NEAR(MassWeightedMean(p, rules), 0.0, 1e-12);
}
