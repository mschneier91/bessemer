// Unit tests for src/operators/stokes_operator -- two of 1.4's green criteria:
//
//  * Collocated-mass diagonality (2D and 3D): with the GLL mass option, applying
//    M to unit true-dof basis vectors produces zero off-diagonal response, and
//    the diagonal matches the assembled (Jacobi) diagonal. The default GL mass
//    is NOT diagonal (guards a silent basis/rule mismatch), and the lumped rule
//    preserves total mass onesT M ones = dim*|Omega|.
//  * Divergence operator: B applied to a known field matches the closed form
//    (u=(x,y[,z]) => div u = dim => (B u)_i = dim * integral of q_i), and B^T is
//    structurally B's transpose (<Bu,q> == <u,BTq>).
//
// Pure algebra: no solver, no stepping. All checks are exact to roundoff, with
// relative tolerances so np {2,4} reductions pass.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "operators/stokes_operator.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesOperator;
using incns::StokesOperatorOptions;

namespace
{

// Unit box (|Omega| = 1) so closed-form checks are simple.
BoxSpec UnitBox(int dim, int n)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// Max |off-diagonal| response and max |diag - AssembleDiagonal| of the mass
// operator, probed column-by-column with unit true-dof vectors (global
// reductions, identical on every rank).
void ProbeMassDiagonality(StokesOperator& op, ParFiniteElementSpace& vfes,
                          double& max_offdiag, double& max_diag_err)
{
   Operator& M = op.Mass();
   const Vector& diag = op.MassDiagonal();
   const HYPRE_BigInt n_global = vfes.GlobalTrueVSize();
   const HYPRE_BigInt offset = vfes.GetMyTDofOffset();
   const int n_local = vfes.GetTrueVSize();

   Vector e(n_local), y(n_local);
   max_offdiag = 0.0;
   max_diag_err = 0.0;
   for (HYPRE_BigInt j = 0; j < n_global; ++j)
   {
      e = 0.0;
      const bool mine = (j >= offset && j < offset + n_local);
      if (mine) { e(static_cast<int>(j - offset)) = 1.0; }
      M.Mult(e, y);
      if (mine)
      {
         const int jl = static_cast<int>(j - offset);
         max_diag_err = std::max(max_diag_err, std::abs(y(jl) - diag(jl)));
         y(jl) = 0.0; // remove the diagonal entry; the rest must vanish
      }
      max_offdiag = std::max(max_offdiag, y.Normlinf());
   }
   MPI_Allreduce(MPI_IN_PLACE, &max_offdiag, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &max_diag_err, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
}

// onesT M ones == dim * |Omega| (= dim on the unit box): total mass preserved.
double TotalMass(StokesOperator& op, ParFiniteElementSpace& vfes)
{
   Vector ones(vfes.GetTrueVSize()), y(vfes.GetTrueVSize());
   ones = 1.0;
   op.Mass().Mult(ones, y);
   return InnerProduct(vfes.GetComm(), ones, y);
}

void CheckCollocatedDiagonality(int dim)
{
   Mesh serial = MakeBoxMesh(UnitBox(dim, 2));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   const int ku = (dim == 3) ? 2 : 3; // keep the 3D probe cheap
   MixedSpaces spaces(mesh, ku, ku - 1);
   RuleBook rules;

   StokesOperatorOptions gll;
   gll.collocated_mass = true;
   StokesOperator op_gll(spaces, rules, gll);

   double offdiag = 0.0, diag_err = 0.0;
   ProbeMassDiagonality(op_gll, spaces.Velocity(), offdiag, diag_err);
   const double scale = op_gll.MassDiagonal().Normlinf();
   EXPECT_LE(offdiag, 1e-14 * scale) << "collocated mass is not diagonal";
   EXPECT_LE(diag_err, 1e-13 * scale) << "Jacobi diagonal mismatch";
   EXPECT_NEAR(TotalMass(op_gll, spaces.Velocity()), static_cast<double>(dim),
               1e-12 * dim);

   // The default GL mass must NOT be diagonal (family/basis fingerprint).
   StokesOperator op_gl(spaces, rules, StokesOperatorOptions());
   ProbeMassDiagonality(op_gl, spaces.Velocity(), offdiag, diag_err);
   EXPECT_GT(offdiag, 1e-6 * scale) << "GL mass unexpectedly diagonal";
   EXPECT_NEAR(TotalMass(op_gl, spaces.Velocity()), static_cast<double>(dim),
               1e-12 * dim);
}

} // namespace

TEST(StokesOperator, CollocatedMassDiagonal2D) { CheckCollocatedDiagonality(2); }

TEST(StokesOperator, CollocatedMassDiagonal3D) { CheckCollocatedDiagonality(3); }

// The viscous operator annihilates constant fields (its null space contains
// rigid translations): K * const = 0 to roundoff.
TEST(StokesOperator, MomentumAnnihilatesConstantsWhenSteady)
{
   Mesh serial = MakeBoxMesh(UnitBox(2, 3));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   StokesOperatorOptions opts;
   opts.nu = 2.5;
   StokesOperator op(spaces, rules, opts);

   Vector c(spaces.Velocity().GetTrueVSize()), y(c.Size());
   c = 1.0;
   op.Momentum().Mult(c, y);
   const double err = std::sqrt(InnerProduct(MPI_COMM_WORLD, y, y));
   EXPECT_LE(err, 1e-12);
}

// B applied to u = (x, y[, z]) equals dim * (integral of q_i): div u = dim
// exactly, and u lies in the velocity space, so the identity is algebraic.
TEST(StokesOperator, DivergenceMatchesWeakFormOfKnownField)
{
   for (int dim : {2, 3})
   {
      Mesh serial = MakeBoxMesh(UnitBox(dim, 2));
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int ku = (dim == 3) ? 2 : 3;
      MixedSpaces spaces(mesh, ku, ku - 1);
      RuleBook rules;
      StokesOperator op(spaces, rules);

      ParGridFunction u(&spaces.Velocity());
      VectorFunctionCoefficient identity(dim, [](const Vector & x, Vector & v)
      { v = x; });
      u.ProjectCoefficient(identity);
      Vector u_true(spaces.Velocity().GetTrueVSize());
      u.GetTrueDofs(u_true);

      Vector Bu(spaces.Pressure().GetTrueVSize());
      op.Divergence().Mult(u_true, Bu);

      // dim * integral of each pressure basis function, on true dofs.
      ParLinearForm ones(&spaces.Pressure());
      ConstantCoefficient one(1.0);
      auto* integ = new DomainLFIntegrator(one);
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      integ->SetIntRule(&rules.Get(geom, 2 * spaces.OrderP()));
      ones.AddDomainIntegrator(integ);
      ones.Assemble();
      std::unique_ptr<HypreParVector> ones_true(ones.ParallelAssemble());

      Vector expected(*ones_true);
      expected *= static_cast<double>(dim);
      expected -= Bu;
      const double err =
         std::sqrt(InnerProduct(MPI_COMM_WORLD, expected, expected));
      const double ref =
         std::sqrt(InnerProduct(MPI_COMM_WORLD, Bu, Bu));
      EXPECT_LE(err, 1e-13 * ref) << "dim=" << dim;
   }
}

// Structural transpose: <B u, q> == <u, B^T q> for deterministic test vectors.
TEST(StokesOperator, DivergenceTransposeIsConsistent)
{
   Mesh serial = MakeBoxMesh(UnitBox(2, 3));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   StokesOperator op(spaces, rules);

   ParFiniteElementSpace& vfes = spaces.Velocity();
   ParFiniteElementSpace& pfes = spaces.Pressure();
   Vector u(vfes.GetTrueVSize()), q(pfes.GetTrueVSize());
   const HYPRE_BigInt uoff = vfes.GetMyTDofOffset();
   const HYPRE_BigInt poff = pfes.GetMyTDofOffset();
   for (int i = 0; i < u.Size(); ++i) { u(i) = std::sin(0.7 * (uoff + i) + 0.3); }
   for (int i = 0; i < q.Size(); ++i) { q(i) = std::cos(0.4 * (poff + i)); }

   Vector Bu(pfes.GetTrueVSize()), BTq(vfes.GetTrueVSize());
   op.Divergence().Mult(u, Bu);
   op.Divergence().MultTranspose(q, BTq);

   const double a = InnerProduct(MPI_COMM_WORLD, Bu, q);
   const double b = InnerProduct(MPI_COMM_WORLD, u, BTq);
   EXPECT_NEAR(a, b, 1e-13 * std::abs(a));
}
