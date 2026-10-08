// Grad-div in the LOR-AMG velocity preconditioner (2026-10-07: it used to be
// omitted, which wrecked LOR-AMG once gamma = c h_K >> nu).
//  G1 the LOR-side gamma (MeshSizeCoefficient with children = k^dim) returns,
//     on every LOR sub-element, c h_K of the high-order element that
//     geometrically contains it -- on a NON-uniform mesh, so a wrong parent
//     (or the sub-element's own size) would show; 2D and 3D.
//  G2 the LOR operator MFEM assembles from StokesOperator::MomentumLORForm()
//     equals an independent reference on the LOR space: sigma M + nu K +
//     ElasticityIntegrator(lambda = gamma, mu = 0) at MFEM's LOR rule (GLL,
//     order 1), and differs from the operator without grad-div.
//  G3 regression guard: one generalized-Stokes solve with the LOR-AMG velocity
//     PC and grad-div needs at most 5x the outer iterations of the same solve
//     without grad-div. Measured 2026-10-07: 95 / 96 / 100 vs 26 at np 1/2/4
//     (AMG counts are rank-sensitive); before the fix 511 vs 26 (~20x).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/mesh_size_coefficient.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/stokes_operator.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace mfem;

namespace
{

Mesh Box(int dim, int n, bool stretched)
{
   incns::BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   Mesh m = incns::MakeBoxMesh(s);
   if (stretched)
   {
      // Separable, monotone: elements stay axis-aligned boxes of varying size.
      m.Transform([](const Vector & x, Vector & y)
      {
         y = x;
         y(0) = x(0) * x(0);
         y(1) = std::pow(x(1), 1.5);
      });
   }
   return m;
}

Vector Random(int n, int seed)
{
   Vector v(n);
   v.Randomize(seed);
   return v;
}

} // namespace

TEST(LorGradDiv, G1_ParentElementGamma)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      const int p = 3;
      Mesh serial = Box(dim, 3, true);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(p, dim);
      ParFiniteElementSpace fes(&mesh, &fec, dim);
      ParLORDiscretization lor(fes);
      ParMesh& lmesh = *lor.GetParFESpace().GetParMesh();
      int children = 1;
      for (int d = 0; d < dim; ++d) { children *= p; }
      ASSERT_EQ(lmesh.GetNE(), mesh.GetNE() * children);

      const double c = 0.7;
      incns::MeshSizeCoefficient lor_gamma(mesh, c, children);
      incns::MeshSizeCoefficient own_size(lmesh, c); // what NOT to get

      // The high-order element containing each LOR element's centre.
      DenseMatrix centres(dim, lmesh.GetNE());
      Vector x;
      for (int e = 0; e < lmesh.GetNE(); ++e)
      {
         lmesh.GetElementCenter(e, x);
         centres.SetCol(e, x);
      }
      Array<int> elems;
      Array<IntegrationPoint> ips;
      // Local search only (Mesh::FindPoints): ParMesh's version is collective
      // and expects the same points on every rank.
      mesh.Mesh::FindPoints(centres, elems, ips, false);
      const IntegrationPoint& ip = Geometries.GetCenter(lmesh.GetElementGeometry(0));
      double err = 0.0, differs = 0.0;
      for (int e = 0; e < lmesh.GetNE(); ++e)
      {
         ASSERT_GE(elems[e], 0) << "LOR element " << e << " not in this rank's "
                                "high-order elements";
         ElementTransformation& T = *lmesh.GetElementTransformation(e);
         const double got = lor_gamma.Eval(T, ip);
         const double want = c * mesh.GetElementSize(elems[e]);
         err = std::max(err, std::abs(got - want) / want);
         differs = std::max(differs, std::abs(own_size.Eval(T, ip) - want) / want);
      }
      EXPECT_LE(err, 1e-14);
      EXPECT_GT(differs, 0.5); // the sub-element's own size is ~h/p
   }
}

TEST(LorGradDiv, G2_LorOperatorIncludesGradDiv)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      const int ku = 3, n = dim == 2 ? 4 : 2;
      const double nu = 1e-3, sigma = 2.0, cgd = 1.0, h = 1.0 / n;
      incns::RuleBook rules; // before the mesh (rule-keyed caches)
      Mesh serial = Box(dim, n, false);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, ku, ku - 1);
      incns::BoundaryConditions bc(spaces.Velocity());
      for (int a : mesh.bdr_attributes) { bc.AddNoSlip(a); }

      auto lor_operator = [&](double gd)
      {
         incns::StokesOperatorOptions oo;
         oo.nu = nu;
         oo.mass_coeff = sigma;
         oo.grad_div = gd;
         oo.lor_momentum = true;
         incns::StokesOperator op(spaces, rules, oo, &bc.EssentialTrueDofs());
         auto lor = std::make_unique<ParLORDiscretization>(spaces.Velocity());
         lor->AssembleSystem(op.MomentumLORForm(), bc.EssentialTrueDofs());
         return lor;
      };
      auto with = lor_operator(cgd);
      auto without = lor_operator(0.0);

      // Reference on the LOR space at MFEM's LOR element rule (GLL, order 1):
      // uniform mesh, so gamma = c_gd h is a constant.
      ParFiniteElementSpace& lfes = with->GetParFESpace();
      IntegrationRules gll(0, Quadrature1D::GaussLobatto);
      const IntegrationRule& ir = gll.Get(dim == 3 ? Geometry::CUBE
                                          : Geometry::SQUARE, 1);
      ConstantCoefficient nu_c(nu), sigma_c(sigma), lambda(cgd * h), zero(0.0);
      ParBilinearForm ref(&lfes);
      auto* k = new VectorDiffusionIntegrator(nu_c);
      auto* m = new VectorMassIntegrator(sigma_c);
      auto* g = new ElasticityIntegrator(lambda, zero);
      k->SetIntRule(&ir);
      m->SetIntRule(&ir);
      g->SetIntRule(&ir);
      ref.AddDomainIntegrator(k);
      ref.AddDomainIntegrator(m);
      ref.AddDomainIntegrator(g);
      ref.Assemble();
      OperatorHandle R;
      ref.FormSystemMatrix(bc.EssentialTrueDofs(), R);

      const Vector xv = Random(R->Width(), 5);
      Vector y_ref(xv.Size()), y_with(xv.Size()), y_without(xv.Size());
      R->Mult(xv, y_ref);
      with->GetAssembledMatrix().Mult(xv, y_with);
      without->GetAssembledMatrix().Mult(xv, y_without);
      const double nref = std::sqrt(InnerProduct(MPI_COMM_WORLD, y_ref, y_ref));
      y_with -= y_ref;
      y_without -= y_ref;
      EXPECT_LE(std::sqrt(InnerProduct(MPI_COMM_WORLD, y_with, y_with)) / nref,
                1e-12);
      EXPECT_GT(std::sqrt(InnerProduct(MPI_COMM_WORLD, y_without, y_without)) /
                nref, 1e-2) << "grad-div should matter at gamma/nu = "
                            << cgd* h / nu;
   }
}

TEST(LorGradDiv, G3_LorAmgIterationsWithGradDiv)
{
   auto solve = [](double gd)
   {
      incns::RuleBook rules;
      Mesh serial = Box(2, 16, false);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      incns::MixedSpaces spaces(mesh, 3, 2);
      incns::BoundaryConditions bc(spaces.Velocity());
      for (int a : mesh.bdr_attributes) { bc.AddNoSlip(a); }
      incns::StokesSolverOptions so;
      so.nu = 1e-3;
      so.mass_coeff = 1.0;
      so.grad_div = gd;
      so.schur = incns::SchurBlockType::CahouetChabard;
      so.cc.a_pc = incns::APC::LORAMG;
      so.rtol = 1e-8;
      so.max_iter = 2000;
      so.kdim = 400;
      incns::StokesSolver solver(spaces, rules, bc, so);
      VectorFunctionCoefficient f(2, [](const Vector & x, Vector & v)
      {
         v(0) = std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]);
         v(1) = -std::cos(2.0 * M_PI * x[0]) * std::sin(2.0 * M_PI * x[1]);
      });
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      solver.Solve(f, u, p);
      EXPECT_TRUE(solver.Converged());
      return solver.Iterations();
   };
   const int plain = solve(0.0), with_gd = solve(1.0);
   if (Mpi::Root())
   {
      std::printf("G3 LOR-AMG outer iterations: no grad-div %d, gamma = h %d\n",
                  plain, with_gd);
   }
   EXPECT_LE(with_gd, 5 * plain);
}
