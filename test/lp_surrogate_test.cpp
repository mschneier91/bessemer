// CC.4 -- inner Poisson PC (LOR-AMG on L_p, SPEC par.6.2):
//  * the application is a FIXED symmetric linear operator (CG legality):
//    deterministic across applications and adjoint-symmetric;
//  * BC policy: Auto puts Dirichlet on outflow attributes; inconsistent
//    singular/Dirichlet combinations throw at construction;
//  * effectiveness: CG on B M_v^-1 B^T preconditioned by ONE L_p V-cycle
//    converges to 1e-8 in a bounded iteration count on both the singular
//    (enclosed, projected RHS + Ortho wrap) and nonsingular (outflow) paths
//    -- an early end-to-end check of the whole inner stack of the CC PC.
//  * folded LOR sub-elements (a moved curved mesh): counted, and L_p then
//    takes |det J| -- positive diagonal, its V-cycle preconditions CG on it;
//    without a fold the matrix is the batched LOR one bitwise.
//  * the same for the velocity LOR (MakeFoldSafeForm on c0 M + nu K +
//    gamma grad-div): positive diagonal, AMG preconditions CG; without a
//    fold the fold-safe form's LOR matrix is the plain one bitwise.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/grad_div_integrator.hpp"
#include "operators/stokes_operator.hpp"
#include "precond/lor_folds.hpp"
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
using incns::CountFoldedLorElements;
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

// A folded LOR sub-element: on a curved Q3 mesh (2 x 2, the pressure space's
// GLL nodes are the mesh nodes) the corner element's first interior node is
// pushed past the element's corner, so the corner sub-element folds. The
// surrogate counts it and assembles with |det J|: every diagonal entry is
// positive and CG with one V-cycle solves L_p x = b. Unfolded, nothing is
// counted and the matrix is the batched LOR assembly's, bitwise.
TEST(LpSurrogate, FoldedLorSubElementsStaySpd)
{
   for (int fold = 0; fold < 2; ++fold)
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {2, 2, 0};
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
      Mesh serial = MakeBoxMesh(s);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      mesh.SetCurvature(3);
      if (fold)
      {
         GridFunction& x = *mesh.GetNodes();
         const FiniteElementSpace& nf = *x.FESpace();
         const double xi = 0.25 * (1.0 - 1.0 / std::sqrt(5.0)); // h = 0.5
         int moved = 0;
         for (int i = 0; i < nf.GetNDofs(); ++i)
         {
            double& px = x(nf.DofToVDof(i, 0));
            double& py = x(nf.DofToVDof(i, 1));
            if (std::fabs(px - xi) < 1e-10 && std::fabs(py - xi) < 1e-10)
            {
               px = -0.02;
               py = -0.02;
               ++moved;
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, &moved, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         ASSERT_EQ(moved, 1);
         mesh.NodesUpdated();
      }
      H1_FECollection fec(3, 2);
      ParFiniteElementSpace pfes(&mesh, &fec);
      Array<int> outflow;
      outflow.Append(2);
      LpSurrogate lp(pfes, outflow, false);
      const HypreParMatrix& A = lp.Matrix();
      // The batched LOR assembly's matrix (the path without folds).
      ParBilinearForm f(&pfes);
      f.AddDomainIntegrator(new DiffusionIntegrator);
      ParLORDiscretization lor(f, lp.EssentialTrueDofs());
      HypreParMatrix& B = lor.GetAssembledMatrix();
      Vector da, db;
      A.GetDiag(da);
      B.GetDiag(db);
      double min_a = da.Size() ? da.Min() : 1.0, min_b = db.Size() ? db.Min() : 1.0;
      MPI_Allreduce(MPI_IN_PLACE, &min_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &min_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      EXPECT_EQ(CountFoldedLorElements(pfes), lp.FoldedLorElements());
      Vector b(A.Height()), xs(A.Height()), y(A.Height());
      b.Randomize(7 + Mpi::WorldRank());
      if (!fold)
      {
         EXPECT_EQ(lp.FoldedLorElements(), 0);
         A.Mult(b, xs);
         B.Mult(b, y);
         double scale[2] = {y.Size() ? y.Normlinf() : 0.0, 0.0};
         xs -= y;
         scale[1] = xs.Size() ? xs.Normlinf() : 0.0;
         MPI_Allreduce(MPI_IN_PLACE, scale, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         // Bitwise on a host; on a GPU the batched assembly orders each row's
         // entries by atomics, which moves the products' roundoff.
         const bool gpu = Device::Allows(Backend::DEVICE_MASK);
         EXPECT_LE(scale[1], gpu ? 1e-13 * scale[0] : 0.0)
               << "unfolded: the batched LOR matrix";
         continue;
      }
      EXPECT_GT(lp.FoldedLorElements(), 0);
      EXPECT_GT(min_a, 0.0) << "|det J|: every diagonal entry positive";
      for (int i : lp.EssentialTrueDofs()) { b(i) = 0.0; }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(A);
      cg.SetPreconditioner(lp);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(200);
      xs = 0.0;
      cg.Mult(b, xs);
      if (Mpi::Root())
      {
         std::cout << "[fold] " << lp.FoldedLorElements() << " folded; min diagonal "
                   << min_a << " (batched " << min_b << "); CG "
                   << cg.GetNumIterations() << " iterations" << std::endl;
      }
      EXPECT_TRUE(cg.GetConverged());
   }
}

// The velocity LOR's momentum-like form on the folded mesh of
// FoldedLorSubElementsStaySpd, through MakeFoldSafeForm: every diagonal entry
// positive and CG with AMG (systems mode) converges on its LOR matrix.
// Unfolded, the fold-safe form's LOR matrix is the plain form's, bitwise on
// a host.
TEST(LpSurrogate, FoldSafeVelocityLorStaysSpd)
{
   for (int fold = 0; fold < 2; ++fold)
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {2, 2, 0};
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
      Mesh serial = MakeBoxMesh(s);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      mesh.SetCurvature(3);
      if (fold)
      {
         GridFunction& x = *mesh.GetNodes();
         const FiniteElementSpace& nf = *x.FESpace();
         const double xi = 0.25 * (1.0 - 1.0 / std::sqrt(5.0)); // h = 0.5
         for (int i = 0; i < nf.GetNDofs(); ++i)
         {
            double& px = x(nf.DofToVDof(i, 0));
            double& py = x(nf.DofToVDof(i, 1));
            if (std::fabs(px - xi) < 1e-10 && std::fabs(py - xi) < 1e-10)
            {
               px = -0.02;
               py = -0.02;
            }
         }
         mesh.NodesUpdated();
      }
      H1_FECollection fec(3, 2);
      ParFiniteElementSpace vfes(&mesh, &fec, 2);
      EXPECT_EQ(CountFoldedLorElements(vfes) > 0, fold == 1);
      ConstantCoefficient c0(10.0), nu(1e-2), gamma(0.1);
      ParBilinearForm a(&vfes);
      a.AddDomainIntegrator(new VectorMassIntegrator(c0));
      a.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
      a.AddDomainIntegrator(new incns::GradDivIntegrator(gamma));
      Array<int> marker(mesh.bdr_attributes.Max()), ess;
      marker = 1;
      marker[1] = 0; // one side free
      vfes.GetEssentialTrueDofs(marker, ess);
      auto safe = incns::MakeFoldSafeForm(a);
      ParLORDiscretization plain(a, ess), lsafe(*safe, ess);
      HypreParMatrix& P = plain.GetAssembledMatrix();
      HypreParMatrix& S = lsafe.GetAssembledMatrix();
      Vector dp, ds;
      P.GetDiag(dp);
      S.GetDiag(ds);
      double mins[2] = {dp.Size() ? dp.Min() : 1.0, ds.Size() ? ds.Min() : 1.0};
      MPI_Allreduce(MPI_IN_PLACE, mins, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      Vector b(S.Height()), xs(S.Height()), y(S.Height());
      b.Randomize(11 + Mpi::WorldRank());
      if (!fold)
      {
         S.Mult(b, xs);
         P.Mult(b, y);
         double scale[2] = {y.Size() ? y.Normlinf() : 0.0, 0.0};
         xs -= y;
         scale[1] = xs.Size() ? xs.Normlinf() : 0.0;
         MPI_Allreduce(MPI_IN_PLACE, scale, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         const bool gpu = Device::Allows(Backend::DEVICE_MASK);
         EXPECT_LE(scale[1], gpu ? 1e-13 * scale[0] : 0.0)
               << "unfolded: the plain LOR matrix";
         continue;
      }
      EXPECT_GT(mins[1], 0.0) << "|det J|: every diagonal entry positive";
      for (int i : ess) { b(i) = 0.0; }
      HypreBoomerAMG amg(S);
      amg.SetSystemsOptions(2, true);
      amg.SetPrintLevel(0);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(S);
      cg.SetPreconditioner(amg);
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(300);
      xs = 0.0;
      cg.Mult(b, xs);
      if (Mpi::Root())
      {
         std::cout << "[fold-velocity] min diagonal " << mins[1] << " (plain "
                   << mins[0] << "); CG " << cg.GetNumIterations()
                   << " iterations" << std::endl;
      }
      EXPECT_TRUE(cg.GetConverged());
   }
}

