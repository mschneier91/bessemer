// CC.3 -- the consistent mixed Poisson operator B M_v^-1 B^T (SPEC T1a/b/c):
//
//  * T1a adjoint: <Bu, q> = <u, B^T q> to 1e-13 (random vectors), enclosed and
//    fully periodic.
//  * T1b nullspace: ||B^T 1|| ~ 0 enclosed and periodic; > 0 with an outflow
//    boundary. Symmetry of BM^-1B^T via random adjoint; positive
//    semi-definiteness (no negative Rayleigh quotients); S*1 ~ 0 when singular.
//  * T1c eliminated-B semantics: the eliminated B^T lands ZERO on essential
//    velocity dofs, the raw B^T does not -- pinning the requirement that the
//    preconditioner consume the ELIMINATED operator.
//
// T1e (never assemble BM^-1B^T) is structural: MixedPoissonOperator exposes
// only Mult; no assembly path exists to test.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/stokes_operator.hpp"
#include "precond/mass_inverse.hpp"
#include "precond/mixed_poisson_op.hpp"
#include "precond/nullspace.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::ConstantPressureProjector;
using incns::MakeBoxMesh;
using incns::MassInverse;
using incns::MassInvType;
using incns::MixedPoissonOperator;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesOperator;
using incns::StokesOperatorOptions;

namespace
{

enum class Domain { Enclosed, Periodic, Outflow };

// One assembled setting: spaces, BCs per the domain type, and the (eliminated)
// Stokes blocks with a collocated (diagonal) velocity mass.
struct SettingCase
{
   ParMesh mesh;
   std::unique_ptr<MixedSpaces> spaces;
   RuleBook rules;
   std::unique_ptr<BoundaryConditions> bc;
   std::unique_ptr<StokesOperator> op;
   std::unique_ptr<MassInverse> mv_inv;

   explicit SettingCase(Domain d)
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {3, 3, 0};
      if (d != Domain::Periodic)
      {
         s.lengths = {1.0, 1.0};
         s.periodic = {false, false, false};
      }
      Mesh serial = MakeBoxMesh(s);
      mesh = ParMesh(MPI_COMM_WORLD, serial);
      spaces = std::make_unique<MixedSpaces>(mesh, 3, 2);

      bc = std::make_unique<BoundaryConditions>(spaces->Velocity());
      if (d == Domain::Enclosed)
      {
         for (int a = 1; a <= 4; ++a) { bc->AddNoSlip(a); }
      }
      else if (d == Domain::Outflow)
      {
         for (int a = 1; a <= 3; ++a) { bc->AddNoSlip(a); }
         bc->AddOutflow(4);
      }

      StokesOperatorOptions oo;
      oo.nu = 1.0;
      oo.collocated_mass = true; // diagonal M_v => DiagDirect inside BM^-1B^T
      op = std::make_unique<StokesOperator>(*spaces, rules, oo,
                                            &bc->EssentialTrueDofs());
      mv_inv = std::make_unique<MassInverse>(op->Mass(), op->MassDiagonal(),
                                             MPI_COMM_WORLD, MassInvType::Auto);
      EXPECT_EQ(mv_inv->Resolved(), MassInvType::DiagDirect);
   }
};

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

// ||B^T 1|| normalized by ||B^T q_rand|| (a scale reference).
double RelBtOne(SettingCase& sc)
{
   const int nu = sc.spaces->Velocity().GetTrueVSize();
   const int np = sc.spaces->Pressure().GetTrueVSize();
   ConstantPressureProjector proj(sc.spaces->Pressure());
   Vector bt1(nu), btq(nu), q(np);
   sc.op->Divergence().MultTranspose(proj.One(), bt1);
   q.Randomize(31);
   sc.op->Divergence().MultTranspose(q, btq);
   return Norm(bt1) / Norm(btq);
}

} // namespace

TEST(MixedPoisson, AdjointOfB)
{
   for (Domain d : {Domain::Enclosed, Domain::Periodic})
   {
      SettingCase sc(d);
      const int nu = sc.spaces->Velocity().GetTrueVSize();
      const int np = sc.spaces->Pressure().GetTrueVSize();
      Vector u(nu), q(np), Bu(np), Btq(nu);
      u.Randomize(41);
      q.Randomize(43);
      sc.op->Divergence().Mult(u, Bu);
      sc.op->Divergence().MultTranspose(q, Btq);
      const double a = InnerProduct(MPI_COMM_WORLD, Bu, q);
      const double b = InnerProduct(MPI_COMM_WORLD, u, Btq);
      // Scale by ||Bu||*||q|| (the Cauchy-Schwarz magnitude of the pairing):
      // a itself can be small through random cancellation.
      const double scale = Norm(Bu) * Norm(q);
      EXPECT_NEAR(a, b, 1e-13 * scale) << "domain " << static_cast<int>(d);
   }
}

TEST(MixedPoisson, BtOneVanishesIffSingular)
{
   SettingCase enclosed(Domain::Enclosed);
   SettingCase periodic(Domain::Periodic);
   SettingCase outflow(Domain::Outflow);
   EXPECT_LE(RelBtOne(enclosed), 1e-13);
   EXPECT_LE(RelBtOne(periodic), 1e-13);
   EXPECT_GT(RelBtOne(outflow), 1e-3); // outflow boundary: constant NOT in kernel
}

TEST(MixedPoisson, SymmetricPositiveSemidefinite)
{
   for (Domain d : {Domain::Enclosed, Domain::Periodic})
   {
      SettingCase sc(d);
      MixedPoissonOperator S(sc.op->Divergence(), *sc.mv_inv);
      const int np = sc.spaces->Pressure().GetTrueVSize();

      // Symmetry: <Sx, y> = <x, Sy>.
      Vector x(np), y(np), Sx(np), Sy(np);
      x.Randomize(51);
      y.Randomize(53);
      S.Mult(x, Sx);
      S.Mult(y, Sy);
      const double a = InnerProduct(MPI_COMM_WORLD, Sx, y);
      const double b = InnerProduct(MPI_COMM_WORLD, x, Sy);
      EXPECT_NEAR(a, b, 1e-12 * Norm(Sx) * Norm(y));

      // Positive semi-definite: Rayleigh quotients never below -1e-12.
      for (int seed : {61, 67, 71})
      {
         Vector z(np), Sz(np);
         z.Randomize(seed);
         S.Mult(z, Sz);
         const double num = InnerProduct(MPI_COMM_WORLD, Sz, z);
         const double den = InnerProduct(MPI_COMM_WORLD, z, z);
         EXPECT_GE(num / den, -1e-12);
      }

      // Singular domains: the constant mode is (numerically) in the kernel.
      ConstantPressureProjector proj(sc.spaces->Pressure());
      Vector S1(np);
      S.Mult(proj.One(), S1);
      Vector Sxr(np);
      S.Mult(x, Sxr); // scale reference
      EXPECT_LE(Norm(S1) / Norm(Sxr), 1e-12);
   }
}

TEST(MixedPoisson, EliminatedBSemantics)
{
   SettingCase sc(Domain::Enclosed);
   const Array<int>& ess = sc.bc->EssentialTrueDofs();
   ASSERT_GT(ess.Size(), 0);

   // Raw (uneliminated) blocks on the same spaces for contrast.
   StokesOperatorOptions oo;
   oo.nu = 1.0;
   oo.collocated_mass = true;
   StokesOperator raw(*sc.spaces, sc.rules, oo, nullptr);

   const int nu = sc.spaces->Velocity().GetTrueVSize();
   const int np = sc.spaces->Pressure().GetTrueVSize();
   Vector q(np), bt_elim(nu), bt_raw(nu);
   q.Randomize(81);
   sc.op->Divergence().MultTranspose(q, bt_elim);
   raw.Divergence().MultTranspose(q, bt_raw);

   const double* e = bt_elim.HostRead();
   const double* r = bt_raw.HostRead();
   double raw_ess_max = 0.0;
   for (int i = 0; i < ess.Size(); ++i)
   {
      EXPECT_EQ(e[ess[i]], 0.0) << "eliminated B^T nonzero at essential dof";
      raw_ess_max = std::max(raw_ess_max, std::abs(r[ess[i]]));
   }
   // The raw operator genuinely differs there (global check: some rank owns
   // boundary dofs with nonzero raw B^T entries).
   MPI_Allreduce(MPI_IN_PLACE, &raw_ess_max, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   EXPECT_GT(raw_ess_max, 1e-8);
}
