// CC.4 -- inner Poisson PC (LOR-AMG on L_p, SPEC par.6.2):
//  * the application is a FIXED symmetric linear operator (CG legality):
//    deterministic across applications and adjoint-symmetric;
//  * BC policy: Auto puts Dirichlet on outflow attributes; inconsistent
//    singular/Dirichlet combinations throw at construction;
//  * effectiveness: CG on B M_v^-1 B^T preconditioned by ONE L_p V-cycle
//    converges to 1e-8 in a bounded iteration count on both the singular
//    (enclosed, projected RHS + Ortho wrap) and nonsingular (outflow) paths
//    -- an early end-to-end check of the whole inner stack of the CC PC.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/stokes_operator.hpp"
#include "precond/lp_surrogate.hpp"
#include "precond/mass_inverse.hpp"
#include "precond/mixed_poisson_op.hpp"
#include "precond/nullspace.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::ConstantPressureProjector;
using incns::LpBC;
using incns::LpSurrogate;
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

struct SettingCase
{
   ParMesh mesh;
   std::unique_ptr<MixedSpaces> spaces;
   RuleBook rules;
   std::unique_ptr<BoundaryConditions> bc;
   std::unique_ptr<StokesOperator> op;
   std::unique_ptr<MassInverse> mv_inv;
   Array<int> outflow_attrs;
   bool singular;

   explicit SettingCase(bool with_outflow, int n = 4)
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {n, n, 0};
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
      Mesh serial = MakeBoxMesh(s);
      mesh = ParMesh(MPI_COMM_WORLD, serial);
      spaces = std::make_unique<MixedSpaces>(mesh, 3, 2);

      bc = std::make_unique<BoundaryConditions>(spaces->Velocity());
      const int last = with_outflow ? 3 : 4;
      for (int a = 1; a <= last; ++a) { bc->AddNoSlip(a); }
      if (with_outflow)
      {
         bc->AddOutflow(4);
         outflow_attrs.Append(4);
      }
      singular = bc->PressureNullspaceExists();
      EXPECT_EQ(singular, !with_outflow);

      StokesOperatorOptions oo;
      oo.nu = 1.0;
      oo.collocated_mass = true;
      op = std::make_unique<StokesOperator>(*spaces, rules, oo,
                                            &bc->EssentialTrueDofs());
      mv_inv = std::make_unique<MassInverse>(op->Mass(), op->MassDiagonal(),
                                             MPI_COMM_WORLD, MassInvType::Auto);
   }
};

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

} // namespace

TEST(LpSurrogate, FixedAndSymmetricApplication)
{
   for (bool outflow : {false, true})
   {
      SettingCase sc(outflow);
      LpSurrogate lp(sc.spaces->Pressure(), sc.outflow_attrs, sc.singular);
      EXPECT_EQ(lp.Singular(), sc.singular);
      const int np = sc.spaces->Pressure().GetTrueVSize();

      // Fixed linear operator: identical output on repeated application.
      Vector x(np), y1(np), y2(np);
      x.Randomize(3);
      lp.Mult(x, y1);
      lp.Mult(x, y2);
      y2 -= y1;
      EXPECT_LE(Norm(y2), 1e-14 * Norm(y1));

      // Symmetric (CG-legal): <Px, y> = <x, Py>, Cauchy-Schwarz-scaled.
      Vector y(np), px(np), py(np);
      y.Randomize(5);
      lp.Mult(x, px);
      lp.Mult(y, py);
      const double a = InnerProduct(MPI_COMM_WORLD, px, y);
      const double b = InnerProduct(MPI_COMM_WORLD, x, py);
      EXPECT_NEAR(a, b, 1e-12 * Norm(px) * Norm(y)) << "outflow=" << outflow;
   }
}

TEST(LpSurrogate, ConfigValidation)
{
   SettingCase enclosed(false);
   SettingCase outflow(true);

   // Singular + Dirichlet-carrying BC mode disagree -> throw.
   EXPECT_THROW(LpSurrogate(outflow.spaces->Pressure(), outflow.outflow_attrs,
                            /*singular=*/true),
                std::invalid_argument);
   // DirichletOnAttrs with an empty list -> throw.
   EXPECT_THROW(LpSurrogate(enclosed.spaces->Pressure(),
                            enclosed.outflow_attrs, /*singular=*/false,
                            LpBC::DirichletOnAttrs),
                std::invalid_argument);
   // vcycles < 1 -> throw.
   EXPECT_THROW(LpSurrogate(enclosed.spaces->Pressure(),
                            enclosed.outflow_attrs, /*singular=*/true,
                            LpBC::AllNeumann, mfem::Array<int>(),
                            /*vcycles=*/0),
                std::invalid_argument);
}

TEST(LpSurrogate, PreconditionsInnerCg)
{
   for (bool outflow : {false, true})
   {
      SettingCase sc(outflow);
      MixedPoissonOperator S(sc.op->Divergence(), *sc.mv_inv);
      LpSurrogate lp(sc.spaces->Pressure(), sc.outflow_attrs, sc.singular);
      ConstantPressureProjector proj(sc.spaces->Pressure());
      const int np = sc.spaces->Pressure().GetTrueVSize();

      Vector rhs(np), w(np);
      rhs.Randomize(9);
      if (sc.singular) { proj.Project(rhs); } // consistent singular system
      w = 0.0;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(S);
      cg.SetPreconditioner(lp);
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(-1);
      cg.Mult(rhs, w);

      EXPECT_TRUE(cg.GetConverged()) << "outflow=" << outflow;
      if (Mpi::Root())
      {
         mfem::out << "[lp_surrogate] inner CG (outflow=" << outflow
                   << ") iters=" << cg.GetNumIterations() << std::endl;
      }
      EXPECT_LE(cg.GetNumIterations(), 60);

      // True residual of the projected/consistent system.
      Vector r(np);
      S.Mult(w, r);
      r -= rhs;
      if (sc.singular) { proj.Project(r); }
      EXPECT_LE(Norm(r), 1e-6 * Norm(rhs));
   }
}
