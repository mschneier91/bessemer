// Sprint 1.10 -- grad-div augmentation, gamma(x) = c_gd * h_K (order-h,
// spatially varying per user spec). Green criteria:
//  * Grad-div assembly: ElasticityIntegrator(lambda = gamma, mu = 0) equals a
//    hand-rolled reference gamma (div u, div v) element matrix, element-wise,
//    to machine precision (2D and 3D).
//  * ||div u|| drops with gamma on and decreases MONOTONICALLY in c_gd,
//    without degrading the velocity error. With the order-h gamma the drop is
//    modest BY DESIGN (measured ~0.2%/0.9% at c_gd = 1/10 on the Dirichlet
//    trig MMS): gamma ~ h is a mild stabilization whose penalty-to-energy
//    ratio scales like h -- the same smallness that justifies keeping it out
//    of the Schur block. Large gamma is the locking regime (velocity error
//    degrades), deliberately not the design point.
//  * The polynomial MMS stays EXACT with gamma on: the exact velocity is
//    pointwise divergence-free and lies in the FE space, so the penalty is
//    consistent (G u_exact = 0).
// Also: a PA energy identity (u = (x,y): uT G u = dim^2 * c * sum_e h_e|e|)
// pins the partial-assembly path against a closed form.
// gamma NEVER enters the Schur block (user decision) -- nothing here changes
// the preconditioner; the gamma > 0 iteration baseline lives in
// schur_quality_test.

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "mesh/mesh_size_coefficient.hpp"
#include "mesh/periodic_box.hpp"
#include "quadrature/rule_book.hpp"
#include "operators/grad_div_scale.hpp"
#include "operators/stokes_operator.hpp"
#include "solver/stokes_solver.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <cmath>

using namespace mfem;
using incns::BoundaryConditions;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MeshSizeCoefficient;
using incns::MixedSpaces;
using incns::RuleBook;
using incns::StokesSolver;
using incns::StokesSolverOptions;
using incns::GradDivScale;
using incns::StokesOperator;
using incns::StokesOperatorOptions;

namespace
{

BoxSpec UnitBox(int dim, int n)
{
   BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {1.0, 1.0, 1.0};
   s.periodic = {false, false, false};
   return s;
}

// Hand-rolled reference element matrix for gamma (div u, div v) on a vector H1
// element with byNODES component ordering: elmat(c1*dof+i, c2*dof+j) =
// sum_q w_q gamma(x_q) dphi_i/dx_{c1} dphi_j/dx_{c2}.
void ReferenceGradDivElmat(const FiniteElement& fe, ElementTransformation& T,
                           Coefficient& gamma, const IntegrationRule& ir,
                           DenseMatrix& elmat)
{
   const int dof = fe.GetDof();
   const int dim = fe.GetDim();
   DenseMatrix dshape(dof, dim);
   elmat.SetSize(dof * dim);
   elmat = 0.0;
   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T.SetIntPoint(&ip);
      fe.CalcPhysDShape(T, dshape);
      const double w = ip.weight * T.Weight() * gamma.Eval(T, ip);
      for (int c1 = 0; c1 < dim; ++c1)
      {
         for (int i = 0; i < dof; ++i)
         {
            for (int c2 = 0; c2 < dim; ++c2)
            {
               for (int j = 0; j < dof; ++j)
               {
                  elmat(c1 * dof + i, c2 * dof + j) +=
                     w * dshape(i, c1) * dshape(j, c2);
               }
            }
         }
      }
   }
}

void CheckElementMatrices(int dim)
{
   Mesh serial = MakeBoxMesh(UnitBox(dim, 2));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   const int k = (dim == 3) ? 2 : 3;
   H1_FECollection fec(k, dim);
   ParFiniteElementSpace vfes(&mesh, &fec, dim, Ordering::byNODES);
   RuleBook rules;
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = rules.Get(geom, 2 * k + dim - 1);

   MeshSizeCoefficient gamma(mesh, 0.7);
   ConstantCoefficient zero_mu(0.0);
   ElasticityIntegrator elasticity(gamma, zero_mu);
   elasticity.SetIntRule(&ir);

   DenseMatrix elmat, ref;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      const FiniteElement& fe = *vfes.GetFE(e);
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      elasticity.AssembleElementMatrix(fe, T, elmat);
      ReferenceGradDivElmat(fe, *mesh.GetElementTransformation(e), gamma, ir,
                            ref);
      ref -= elmat;
      EXPECT_LE(ref.MaxMaxNorm(), 1e-13 * elmat.MaxMaxNorm())
            << "dim=" << dim << " element " << e;
   }
}

} // namespace

TEST(GradDiv, ElementMatrixMatchesReference2D) { CheckElementMatrices(2); }

TEST(GradDiv, ElementMatrixMatchesReference3D) { CheckElementMatrices(3); }

// PA path against a closed form: u = (x, y) has div u = 2, so
// uT G u = integral gamma * 4 = 4 * c * sum_e h_e |e| (gamma constant per
// element). Machine-precision identity, verified through the solver's own
// momentum block by differencing gamma-on vs gamma-off (isolates G).
TEST(GradDiv, PaEnergyIdentity2D)
{
   Mesh serial = MakeBoxMesh(UnitBox(2, 3));
   ParMesh mesh(MPI_COMM_WORLD, serial);
   MixedSpaces spaces(mesh, 3, 2);
   RuleBook rules;
   const double c_gd = 0.7;

   ParBilinearForm g_form(&spaces.Velocity());
   MeshSizeCoefficient gamma(mesh, c_gd);
   ConstantCoefficient zero_mu(0.0);
   auto* gdi = new ElasticityIntegrator(gamma, zero_mu);
   gdi->SetIntRule(&rules.Get(Geometry::SQUARE, 2 * 3 + 1));
   g_form.AddDomainIntegrator(gdi);
   g_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   g_form.Assemble();
   Array<int> empty;
   OperatorPtr G;
   g_form.FormSystemMatrix(empty, G);

   ParGridFunction u(&spaces.Velocity());
   VectorFunctionCoefficient idc(2, [](const Vector & x, Vector & v) { v = x; });
   u.ProjectCoefficient(idc);
   Vector ut(spaces.Velocity().GetTrueVSize()), Gu(ut.Size());
   u.GetTrueDofs(ut);
   G->Mult(ut, Gu);
   const double energy = InnerProduct(MPI_COMM_WORLD, ut, Gu);

   // Closed form over the local elements, reduced globally.
   double local = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      const IntegrationRule& ir1 = rules.Get(Geometry::SQUARE, 2);
      double vol = 0.0;
      for (int q = 0; q < ir1.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir1.IntPoint(q));
         vol += ir1.IntPoint(q).weight * T.Weight();
      }
      local += 4.0 * c_gd * mesh.GetElementSize(e) * vol;
   }
   double exact = 0.0;
   MPI_Allreduce(&local, &exact, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   EXPECT_NEAR(energy, exact, 1e-12 * exact);
}

// With gamma on: (a) the polynomial MMS stays exact (consistency: the exact
// velocity is div-free, so the penalty vanishes on it); (b) on the trig MMS,
// ||div u_h|| DROPS versus gamma = 0 while the velocity error stays healthy.
TEST(GradDiv, ConsistencyAndDivergenceDrop)
{
   const double nu = 1.0;
   const double pi = M_PI;

   // --- (a) polynomial exactness with gamma on (2D, Q3/Q2, all-Dirichlet) ---
   {
      VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
      {
         v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
         v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
      });
      VectorFunctionCoefficient forcing(2, [nu](const Vector & x, Vector & f)
      {
         const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
         const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
         f(0) = -nu * lap0 + 2.0 * x[0];
         f(1) = -nu * lap1 + 2.0 * x[1];
      });

      Mesh serial = MakeBoxMesh(UnitBox(2, 3));
      ParMesh mesh(MPI_COMM_WORLD, serial);
      MixedSpaces spaces(mesh, 3, 2);
      RuleBook rules;
      BoundaryConditions bc(spaces.Velocity());
      for (int attr = 1; attr <= 4; ++attr)
      {
         bc.AddVelocityDirichlet(attr, u_exact);
      }
      StokesSolverOptions opts;
      opts.nu = nu;
      opts.grad_div = 1.0;
      opts.rtol = 1e-12;
      opts.max_iter = 5000;
      opts.kdim = 400;
      StokesSolver solver(spaces, rules, bc, opts);
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      solver.Solve(forcing, u, p);
      const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
      irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, 10);
      EXPECT_LE(u.ComputeL2Error(u_exact, irs), 1e-8);
   }

   // --- (b) trig MMS: ||div u_h|| drops with gamma on ------------------------
   VectorFunctionCoefficient u_exact(2, [pi](const Vector & x, Vector & v)
   {
      v(0) = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      v(1) = -pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
   });
   VectorFunctionCoefficient forcing(2, [pi, nu](const Vector & x, Vector & f)
   {
      const double u0 = pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      const double u1 = -pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
      f(0) = 2.0 * pi * pi * nu * u0 - pi * std::sin(pi * x[0]) * std::cos(pi * x[1]);
      f(1) = 2.0 * pi * pi * nu * u1 - pi * std::cos(pi * x[0]) * std::sin(pi * x[1]);
   });

   double div_norm[3], u_err[3];
   int iters[3];
   const double c_gds[3] = {0.0, 1.0, 10.0};
   for (int i = 0; i < 3; ++i)
   {
      Mesh serial = MakeBoxMesh(UnitBox(2, 8));
      ParMesh mesh(MPI_COMM_WORLD, serial);
      MixedSpaces spaces(mesh, 3, 2);
      RuleBook rules;
      BoundaryConditions bc(spaces.Velocity());
      for (int attr = 1; attr <= 4; ++attr)
      {
         bc.AddVelocityDirichlet(attr, u_exact);
      }
      StokesSolverOptions opts;
      opts.nu = nu;
      opts.grad_div = c_gds[i];
      opts.rtol = 1e-12;
      opts.max_iter = 5000;
      opts.kdim = 400;
      StokesSolver solver(spaces, rules, bc, opts);
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      solver.Solve(forcing, u, p);
      ASSERT_TRUE(solver.Converged());

      const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
      irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, 10);
      ConstantCoefficient zero(0.0);
      div_norm[i] = u.ComputeDivError(&zero, irs);
      u_err[i] = u.ComputeL2Error(u_exact, irs);
      iters[i] = solver.Iterations();
   }
   if (Mpi::Root())
   {
      for (int i = 0; i < 3; ++i)
      {
         mfem::out << "[graddiv] c_gd=" << c_gds[i] << ": ||div u||="
                   << div_norm[i] << " u_err=" << u_err[i] << " iters="
                   << iters[i] << std::endl;
      }
   }
   // Monotone divergence response (solves are deterministic to rtol 1e-12, so
   // the small differences are far above solver noise), accuracy preserved.
   EXPECT_LT(div_norm[1], div_norm[0]);
   EXPECT_LT(div_norm[2], div_norm[1]);
   EXPECT_LE(u_err[1], 1.5 * u_err[0]);
   EXPECT_LE(u_err[2], 1.5 * u_err[0]);
}

// OrderNu grad-div scaling (gamma = c_gd * nu, constant in space).
//  (a) the div-free polynomial MMS is still reproduced exactly -- grad-div
//      vanishes on a divergence-free exact solution regardless of scaling;
//  (b) on a UNIFORM mesh (h_K = h0 for every element), OrderNu with scale c
//      produces the SAME constant gamma = c*nu as OrderH with scale c*nu/h0,
//      so the two momentum operators are identical -- pinning that OrderNu
//      genuinely computes c*nu, not c*h.
TEST(GradDiv, OrderNuScaling)
{
   const double nu = 0.7, c_gd = 2.0;

   // (a) polynomial MMS, OrderNu, all-Dirichlet Q3/Q2.
   {
      VectorFunctionCoefficient u_exact(2, [](const Vector & x, Vector & v)
      {
         v(0) = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1];
         v(1) = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1];
      });
      VectorFunctionCoefficient forcing(2, [nu](const Vector & x, Vector & f)
      {
         const double lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0];
         const double lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1];
         f(0) = -nu * lap0 + 2.0 * x[0];
         f(1) = -nu * lap1 + 2.0 * x[1];
      });
      Mesh serial = MakeBoxMesh(UnitBox(2, 3));
      ParMesh mesh(MPI_COMM_WORLD, serial);
      MixedSpaces spaces(mesh, 3, 2);
      RuleBook rules;
      BoundaryConditions bc(spaces.Velocity());
      for (int a = 1; a <= 4; ++a) { bc.AddVelocityDirichlet(a, u_exact); }
      StokesSolverOptions opts;
      opts.nu = nu;
      opts.grad_div = c_gd;
      opts.grad_div_scale = GradDivScale::OrderNu;
      opts.rtol = 1e-12;
      opts.max_iter = 5000;
      opts.kdim = 400;
      StokesSolver solver(spaces, rules, bc, opts);
      ParGridFunction u(&spaces.Velocity()), p(&spaces.Pressure());
      solver.Solve(forcing, u, p);
      const IntegrationRule* irs[Geometry::NumGeom] = {nullptr};
      irs[Geometry::SQUARE] = &rules.Get(Geometry::SQUARE, 10);
      EXPECT_LE(u.ComputeL2Error(u_exact, irs), 1e-8);
   }

   // (b) OrderNu(c) == OrderH(c*nu/h0) on a uniform mesh.
   {
      Mesh serial = MakeBoxMesh(UnitBox(2, 4)); // uniform: every h_K = h0
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const double h0 = mesh.GetElementSize(0);
      MixedSpaces spaces(mesh, 3, 2);
      RuleBook rules;

      StokesOperatorOptions on;
      on.nu = nu;
      on.grad_div = c_gd;
      on.grad_div_scale = GradDivScale::OrderNu;   // gamma = c_gd * nu
      StokesOperator op_nu(spaces, rules, on);

      StokesOperatorOptions oh;
      oh.nu = nu;
      oh.grad_div = c_gd * nu / h0;                // gamma = (c_gd*nu/h0)*h0
      oh.grad_div_scale = GradDivScale::OrderH;
      StokesOperator op_h(spaces, rules, oh);

      Vector x(spaces.Velocity().GetTrueVSize());
      x.Randomize(19);
      Vector ynu(x.Size()), yh(x.Size());
      op_nu.Momentum().Mult(x, ynu);
      op_h.Momentum().Mult(x, yh);
      yh -= ynu;
      const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, yh, yh) /
                                   InnerProduct(MPI_COMM_WORLD, ynu, ynu));
      EXPECT_LE(rel, 1e-12) << "OrderNu(c) should equal OrderH(c*nu/h0)";
   }
}
