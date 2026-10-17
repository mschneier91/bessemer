// The subgrid stress of large-eddy simulation (operators/subgrid_stress,
// operators/subgrid_model; docs/les.md).
//
//  S1 the pointwise models: WALE and sigma vanish in pure shear (both),
//     sigma also in solid rotation, axisymmetric and isotropic expansion and
//     for a two-component flow (WALE does not: its known flaws); against
//     an independent evaluation on a random gradient (the singular values
//     from MFEM's eigenvalue routine on g^T g; WALE from DenseMatrix algebra);
//     the y^3 wall behaviour of both down to y = 1e-4 (where Cardano's
//     formula had already returned 0, see SingularValues3).
//  S2 the operator with a CONSTANT nu_t equals MFEM's ElasticityIntegrator
//     (lambda = 0, mu = nu_t), fully assembled on the same rule, to 1e-12:
//     the stress form with the symmetric gradient, on a curved partitioned
//     mesh in 2D and 3D at p = 2, 3 (the kernels' interpolation, Jacobians,
//     weights, pull-back and assembly, with hanging-node-free P).
//  S3 the filter width is h_K / k_u; the dissipation statistic equals
//     <Mult(u), u> (the term does nonnegative work: energy stable);
//     statistics agree across ranks.
//  S4 marching: a constant nu_t carried EXPLICITLY (the model's path) on the
//     2D Taylor-Green vortex converges to the IMPLICIT run at viscosity
//     nu + nu_t at second order in dt (the exact solution is the TGV at the
//     total viscosity because it is divergence-free, so the stress term IS
//     nu_t times the Laplacian there), under IMEX and under OIFS.
//  S5 through Case and Parameters: the sigma model on the 3D Taylor-Green
//     vortex steps, writes finite statistics, and nu_t stays small against
//     nu (the initial field is two-component, where sigma is zero).

#include <gtest/gtest.h>

#include "bc/boundary_conditions.hpp"
#include "config/parameters.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/periodic_box.hpp"
#include "operators/subgrid_model.hpp"
#include "operators/subgrid_stress.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/case.hpp"
#include "spaces/mixed_spaces.hpp"
#include "time/time_integrator.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

using namespace mfem;
using incns::SubgridModel;
using incns::SubgridOptions;
using incns::SubgridStress;

namespace
{

double GlobalNorm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

double GlobalDot(const Vector& a, const Vector& b)
{
   return InnerProduct(MPI_COMM_WORLD, a, b);
}

// A curved (non-affine) partitioned box of the given order: every Jacobian
// varies within the element, so the pull-back is exercised.
std::unique_ptr<ParMesh> CurvedBox(int dim, int n, int p)
{
   Mesh serial = (dim == 2)
                 ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL)
                 : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);
   serial.SetCurvature(p);
   serial.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      y(0) += 0.08 * std::sin(M_PI * x(1));
      y(1) += 0.08 * std::sin(M_PI * x(0));
      if (x.Size() == 3) { y(2) += 0.08 * std::sin(M_PI * x(0)) * std::cos(M_PI * x(1)); }
   });
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

// WALE and sigma from independent algebra (DenseMatrix, MFEM's symmetric
// eigenvalue routine): the reference for S1.
double WaleReference(const DenseMatrix& g)
{
   DenseMatrix g2(3), S(3), Sd(3);
   Mult(g, g, g2);
   const double tr = g2.Trace() / 3.0;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         S(i, j) = 0.5 * (g(i, j) + g(j, i));
         Sd(i, j) = 0.5 * (g2(i, j) + g2(j, i)) - (i == j ? tr : 0.0);
      }
   const double ss = S.FNorm2(), sdsd = Sd.FNorm2();
   const double den = std::pow(ss, 2.5) + std::pow(sdsd, 1.25);
   return den > 0.0 ? std::pow(sdsd, 1.5) / den : 0.0;
}

double SigmaReference(const DenseMatrix& g)
{
   DenseMatrix G(3);
   MultAtB(g, g, G);
   double lambda[3], vec[9];
   G.CalcEigenvalues(lambda, vec);
   std::sort(lambda, lambda + 3, std::greater<double>());
   double s[3];
   for (int i = 0; i < 3; ++i) { s[i] = std::sqrt(std::max(lambda[i], 0.0)); }
   if (s[0] <= 0.0) { return 0.0; }
   return s[2] * (s[0] - s[1]) * (s[1] - s[2]) / (s[0] * s[0]);
}

void ToArray(const DenseMatrix& m, double g[3][3])
{
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) { g[i][j] = m(i, j); }
}

} // namespace

TEST(Subgrid, S1_PointwiseModels)
{
   using incns::subgrid::Sigma;
   using incns::subgrid::Wale;
   double g[3][3];
   // Pure shear: both vanish (the near-wall requirement).
   double shear[3][3] = {{0.0, 1.3, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   EXPECT_EQ(Wale(shear, 1.0), 0.0);
   EXPECT_EQ(Sigma(shear, 1.0), 0.0);
   // Solid rotation: sigma vanishes, WALE does not (its known flaw).
   double rot[3][3] = {{0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   EXPECT_NEAR(Sigma(rot, 1.0), 0.0, 1e-14);
   EXPECT_GT(Wale(rot, 1.0), 0.1);
   // Axisymmetric expansion diag(a, a, -2a): sigma vanishes (sigma_2 =
   // sigma_3), WALE does not. Isotropic expansion: both vanish.
   double axi[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, -2.0}};
   EXPECT_NEAR(Sigma(axi, 1.0), 0.0, 1e-14);
   EXPECT_GT(Wale(axi, 1.0), 0.1);
   double iso[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
   EXPECT_NEAR(Sigma(iso, 1.0), 0.0, 1e-14);
   EXPECT_NEAR(Wale(iso, 1.0), 0.0, 1e-14);
   // A two-component (2D) flow: sigma_3 = 0, so sigma vanishes identically.
   double two[3][3] = {{0.7, -0.4, 0.0}, {1.1, -0.7, 0.0}, {0.0, 0.0, 0.0}};
   EXPECT_NEAR(Sigma(two, 1.0), 0.0, 1e-14);
   // Random (well-conditioned) gradients against the independent
   // evaluations, with the scaling (C Delta)^2 applied. The reference's
   // closed-form eigenvalues are only trusted away from the ill-conditioned
   // regime; the wall case below covers that one by hand.
   std::mt19937 rng(7);
   std::normal_distribution<double> nd(0.0, 1.0);
   for (int k = 0; k < 50; ++k)
   {
      DenseMatrix m(3);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j) { m(i, j) = nd(rng); }
      ToArray(m, g);
      const double cd2 = 0.3;
      const double sr = SigmaReference(m), wr = WaleReference(m);
      EXPECT_NEAR(Sigma(g, cd2), cd2 * sr, 1e-9 * cd2 * sr + 1e-15);
      EXPECT_NEAR(Wale(g, cd2), cd2 * wr, 1e-11 * cd2 * wr + 1e-15);
   }
   // The y^3 wall behaviour: the gradient of u = (y + x y, -y^2, y z) (a
   // divergence-free wall flow) at x = z = 0 is [[y, 1, 0], [0, -2y, 0],
   // [0, 0, y]]; both models give nu_t = 2 y^3 (1 + O(y)) (the sigma
   // model's relative correction is about -3y: 1.939e-6 at y = 1e-2 against
   // 2e-6, checked with LAPACK's SVD). Cardano's formula for sigma_3
   // returned 0 at y = 1e-3 here.
   for (double y : {1e-2, 1e-3, 1e-4})
   {
      double wall[3][3] = {{y, 1.0, 0.0}, {0.0, -2.0 * y, 0.0}, {0.0, 0.0, y}};
      const double expect = 2.0 * y * y * y;
      EXPECT_NEAR(Sigma(wall, 1.0), expect, 4.0 * y * expect) << "y = " << y;
      EXPECT_NEAR(Wale(wall, 1.0), expect, 4.0 * y * expect) << "y = " << y;
   }
}

// S2: with nu_t constant the term is (2 nu_t S(u), S(v)) exactly, which is
// MFEM's elasticity form with lambda = 0, mu = nu_t.
TEST(Subgrid, S2_ConstantModelMatchesElasticity)
{
   for (int dim : {2, 3})
   {
      for (int p : {2, 3})
      {
         incns::RuleBook rules; // outlives the mesh and every integrator
         std::unique_ptr<ParMesh> mesh = CurvedBox(dim, dim == 2 ? 4 : 2, p);
         incns::MixedSpaces spaces(*mesh, p, p - 1);
         const double nu_t = 0.37;
         SubgridOptions opts;
         opts.model = SubgridModel::Constant;
         opts.coefficient = nu_t;
         SubgridStress sgs(spaces, rules, opts);

         ConstantCoefficient lambda(0.0), mu(nu_t);
         ParBilinearForm el(&spaces.Velocity());
         auto* integ = new ElasticityIntegrator(lambda, mu);
         integ->SetIntRule(&sgs.Rule());
         el.AddDomainIntegrator(integ);
         el.Assemble();
         el.Finalize();
         Array<int> empty;
         OperatorPtr A;
         el.FormSystemMatrix(empty, A);

         Vector x(spaces.Velocity().GetTrueVSize()), y_sgs, y_el(x.Size());
         x.Randomize(11);
         sgs.Mult(x, y_sgs);
         A->Mult(x, y_el);
         ASSERT_EQ(y_sgs.Size(), y_el.Size());
         Vector diff(y_el);
         diff -= y_sgs;
         const double rel = GlobalNorm(diff) / GlobalNorm(y_el);
         if (Mpi::Root())
         {
            mfem::out << "[subgrid S2] dim " << dim << " p " << p
                      << " |sgs - elasticity| / |elasticity| = " << rel << "\n";
         }
         EXPECT_LT(rel, 1e-12) << "dim " << dim << " p " << p;
      }
   }
}

// S3: filter width and statistics.
TEST(Subgrid, S3_FilterWidthAndDissipation)
{
   for (int dim : {2, 3})
   {
      incns::RuleBook rules;
      incns::BoxSpec spec;
      spec.dim = dim;
      spec.num_elems = {4, 4, 4};
      spec.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
      spec.periodic = {true, true, true};
      Mesh serial = incns::MakeBoxMesh(spec);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      const int ku = 3;
      incns::MixedSpaces spaces(mesh, ku, ku - 1);
      SubgridOptions opts;
      opts.model = SubgridModel::Wale;
      opts.filter_width = 1.5;
      SubgridStress sgs(spaces, rules, opts);
      // Delta_K = filter_width * h_K / k_u, h_K the element edge on a box.
      const double h = 2.0 * M_PI / 4.0;
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         EXPECT_NEAR(sgs.FilterWidth(e), opts.filter_width * h / ku, 1e-12 * h);
      }
      EXPECT_EQ(opts.Coefficient(), 0.5); // WALE's published constant
      // A field with a nontrivial gradient everywhere (the 2D TGV, extended
      // with a swirl in 3D so sigma_3 != 0 there too).
      ParGridFunction u(&spaces.Velocity());
      VectorFunctionCoefficient uc(dim, [dim](const Vector & x, Vector & v)
      {
         v(0) = std::sin(x(0)) * std::cos(x(1));
         v(1) = -std::cos(x(0)) * std::sin(x(1));
         if (dim == 3)
         {
            v(0) *= std::cos(x(2));
            v(1) *= std::cos(x(2));
            v(2) = 0.3 * std::sin(x(2)) * std::cos(x(0));
         }
      });
      u.ProjectCoefficient(uc);
      Vector ut, y;
      u.GetTrueDofs(ut);
      sgs.Mult(ut, y);
      const SubgridStress::Values v = sgs.Statistics(ut);
      const double work = GlobalDot(y, ut); // = int 2 nu_t S(u):grad u
      if (Mpi::Root())
      {
         mfem::out << "[subgrid S3] dim " << dim << " nu_t max " << v.nu_t_max
                   << " mean " << v.nu_t_mean << " dissipation " << v.dissipation
                   << " <y, u> " << work << " volume " << v.volume << "\n";
      }
      EXPECT_GT(v.nu_t_max, 0.0);
      EXPECT_GT(v.nu_t_mean, 0.0);
      EXPECT_LE(v.nu_t_mean, v.nu_t_max);
      EXPECT_NEAR(v.volume, std::pow(2.0 * M_PI, dim), 1e-10 * v.volume);
      EXPECT_GT(v.dissipation, 0.0);
      EXPECT_NEAR(work, v.dissipation, 1e-11 * v.dissipation);
      // Every rank holds the same (reduced) numbers.
      double mx = v.nu_t_max, mn = v.nu_t_max;
      MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      EXPECT_EQ(mx, mn);
   }
}

namespace
{

constexpr double kNuTotal = 0.5;  ///< S4: nu + nu_t of every run.
constexpr double kNuT = 0.05;     ///< S4: the explicit part.
constexpr double kTFinalS4 = 0.2; ///< S4: short, the difference is in dt.

// S4: the 2D TGV on an 8x8 Q3/Q2 periodic box, fixed-step BDF2, to t = 0.2;
// returns the velocity true dofs. explicit_part = 0: the whole viscosity
// implicit; > 0: that part carried by the constant subgrid model.
Vector MarchTgv(incns::ConvectionTreatment conv, double dt,
                double explicit_part)
{
   incns::BoxSpec spec;
   spec.dim = 2;
   spec.num_elems = {8, 8, 8};
   spec.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   spec.periodic = {true, true, true};
   Mesh serial = incns::MakeBoxMesh(spec);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   incns::MixedSpaces spaces(mesh, 3, 2);
   incns::RuleBook rules;
   incns::BoundaryConditions bc(spaces.Velocity()); // periodic
   Vector zero_vec(2);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_forcing(zero_vec);
   incns::TimeIntegratorOptions opts;
   opts.nu = kNuTotal - explicit_part;
   opts.dt = dt;
   opts.t_final = kTFinalS4;
   opts.order = 2;
   opts.convection = true;
   opts.convection_treatment = conv;
   opts.collocated_mass = (conv == incns::ConvectionTreatment::Oifs);
   opts.rtol = 1e-12;
   opts.max_iter = 5000;
   opts.kdim = 400;
   if (explicit_part > 0.0)
   {
      opts.subgrid.model = SubgridModel::Constant;
      opts.subgrid.coefficient = explicit_part;
   }
   incns::StokesTimeIntegrator stepper(spaces, rules, bc, zero_forcing, opts);
   VectorFunctionCoefficient u0(2, [](const Vector & x, double t, Vector & u)
   { incns::tgv2d::Velocity(x, t, kNuTotal, u); });
   stepper.SetInitialVelocity(u0);
   stepper.Run();
   EXPECT_NEAR(stepper.Time(), kTFinalS4, 1e-10);
   Vector ut;
   stepper.Velocity().GetTrueDofs(ut);
   return ut;
}

} // namespace

TEST(Subgrid, S4_ExplicitConstantModelConvergesToImplicitViscosity)
{
   using incns::ConvectionTreatment;
   const ConvectionTreatment convs[2] = {ConvectionTreatment::Imex,
                                         ConvectionTreatment::Oifs
                                        };
   for (ConvectionTreatment conv : convs)
   {
      const bool oifs = (conv == ConvectionTreatment::Oifs);
      // The split error is O(dt^2) (BDF2/EXT2 on the explicit part), so the
      // distance to the implicit run shrinks 4x per halving; require 3x.
      // Three steps are enough to see it; the explicit diffusion is far
      // inside its stability limit (nu_t dt / h_node^2 <= 0.02).
      const double dts[3] = {0.02, 0.01, 0.005};
      double dist[3];
      for (int i = 0; i < 3; ++i)
      {
         Vector implicit_run = MarchTgv(conv, dts[i], 0.0);
         Vector split_run = MarchTgv(conv, dts[i], kNuT);
         split_run -= implicit_run;
         dist[i] = GlobalNorm(split_run) / GlobalNorm(implicit_run);
      }
      if (Mpi::Root())
      {
         mfem::out << "[subgrid S4] " << (oifs ? "OIFS" : "IMEX")
                   << " |explicit - implicit| / |u| at dt 0.02, 0.01, 0.005: "
                   << dist[0] << " " << dist[1] << " " << dist[2] << "\n";
      }
      EXPECT_LT(dist[2], 1e-4) << (oifs ? "OIFS" : "IMEX");
      EXPECT_GT(dist[0] / dist[1], 3.0) << (oifs ? "OIFS" : "IMEX");
      EXPECT_GT(dist[1] / dist[2], 3.0) << (oifs ? "OIFS" : "IMEX");
   }
}

// S5: the deck path. The sigma model on the 3D Taylor-Green vortex (4^3
// Q3/Q2, nu = 0.1) through Case: it steps, the statistics are finite and
// nonnegative, and nu_t stays a small fraction of nu: the initial vortex is
// a two-component field (u_z = 0), where the sigma model is zero by
// construction, and three steps develop little u_z.
TEST(Subgrid, S5_SigmaModelThroughCase)
{
   incns::Parameters p;
   p.equation = incns::Equation::NavierStokes;
   p.nu = 0.1;
   p.order_u = 3;
   p.order_p = 2;
   p.mesh.dim = 3;
   p.mesh.num_elems = {4, 4, 4};
   p.mesh.lengths = {2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI};
   p.mesh.periodic = {true, true, true};
   p.initial_velocity = "taylor_green_3d";
   p.step_control = incns::StepControl::Fixed;
   p.dt = 0.02;
   p.t_final = 0.06;
   p.subgrid.model = SubgridModel::Sigma;
   p.Normalize();
   EXPECT_EQ(p.subgrid.Coefficient(), 1.35);
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   int steps = 0;
   while (!flow.Done()) { flow.Step(); ++steps; }
   EXPECT_EQ(steps, 3);
   ASSERT_NE(flow.Integrator().Subgrid(), nullptr);
   const SubgridStress::Values v = flow.Integrator().SubgridStatistics();
   if (Mpi::Root())
   {
      mfem::out << "[subgrid S5] sigma on TGV 3D 4^3: nu_t max " << v.nu_t_max
                << " (" << v.nu_t_max / p.nu << " nu), mean " << v.nu_t_mean
                << ", dissipation " << v.dissipation << ", KE "
                << flow.KineticEnergy() << "\n";
   }
   EXPECT_TRUE(std::isfinite(v.nu_t_max) && std::isfinite(v.dissipation));
   EXPECT_GE(v.nu_t_max, 0.0);
   EXPECT_GE(v.dissipation, 0.0);
   EXPECT_LT(v.nu_t_max, 0.1 * p.nu);
   EXPECT_TRUE(std::isfinite(flow.KineticEnergy()));
}
