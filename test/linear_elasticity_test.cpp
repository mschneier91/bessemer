// Linear elasticity (src/solid): exactness and consistency at np 1/2/4.
//  E1 the operator and its diagonal equal MFEM's at orders 1 to 7 in 2D
//     and 3D (curved, distorted elements, per-element stiffness factors):
//     its assembled ElasticityIntegrator matrix up to 3D Q3, its partial
//     assembly above (where it launches; on a GPU to Q5) and St.
//     Venant-Kirchhoff's tangent at u = 0 with a probed diagonal; on a
//     nonconforming mesh too (the diagonal there |P|^T d, as MFEM's);
//  E2 a quadratic displacement is reproduced exactly by Q2 (2D, 3D), with a
//     Dirichlet face and the exact traction on every other face;
//  E3 the Timoshenko-Goodier cantilever (plane stress, an exact cubic) is
//     reproduced by Q3, tip deflection P L^3 / (3 E I) included;
//  E4 energy identity 2 W = (f, u) + (t, u) with zero Dirichlet data, and the
//     LOR-AMG and Jacobi preconditioners reaching the same displacement;
//  E5 a deck (equation: linear_elasticity) through ElasticityCase, summary
//     written;
//  E6 an elastic support (AddSupport) on a bar with lambda = 0, exact at Q2
//     (u = a x, a = t / (E + k L)) with both preconditioners, and the
//     clamp's reaction (TrackReaction) balancing the loaded end.
// These pin their own orders and tolerances (unit tests); the convergence
// rates at the deck defaults are the verify tier's (verify_elasticity_test).

#include <gtest/gtest.h>

#include "elasticity_test_util.hpp"

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/element_scaled_coefficient.hpp"
#include "solid/hyperelastic_operator.hpp"
#include "solid/linear_elasticity.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace mfem;
using namespace elasticity_test;
using incns::LinearElasticity;

namespace
{

std::unique_ptr<ParMesh> Partition(Mesh& serial)
{
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

std::unique_ptr<ParMesh> UnitBox(int dim, int n)
{
   Mesh m = dim == 2 ?
            Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0) :
            Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   return Partition(m);
}

LinearElasticity::Options Opts(int order, double lambda, double mu)
{
   LinearElasticity::Options o;
   o.order = order;
   o.lambda = lambda;
   o.mu = mu;
   o.rtol = 1e-12;
   o.atol = 0.0;
   o.max_iter = 5000;
   return o;
}

/// Dirichlet (exact data) on @p held, the exact traction on every other face,
/// the exact body force when there is one.
void ApplyExact(LinearElasticity& s, ParMesh& mesh, Exact& ex,
                const std::vector<int>& held)
{
   for (int a : held) { s.AddDisplacement(a, *ex.u); }
   for (int a : OtherAttributes(mesh, held)) { s.AddTraction(a, *ex.traction); }
   if (ex.f) { s.SetBodyForce(*ex.f); }
}

/// A full quadratic u_i = a_i + B_ij x_j + 1/2 x^T C_i x (C_i symmetric):
/// grad u_i = B_i + C_i x, f_i = -(mu tr C_i + (lambda + mu) sum_k (C_k)_ki).
std::unique_ptr<Exact> Quadratic(int dim, double lambda, double mu)
{
   // Fixed, arbitrary coefficients.
   auto A = [](int i) { return 0.1 * (i + 1); };
   auto B = [](int i, int j) { return 0.2 + 0.1 * i - 0.15 * j + 0.05 * i * j; };
   auto C = [](int i, int j, int k)
   {
      const int a = std::min(j, k), b = std::max(j, k);
      return 0.3 - 0.1 * i + 0.07 * a - 0.05 * b + 0.02 * i * a * b;
   };
   VecFn uf = [ = ](const Vector & x, Vector & v)
   {
      v.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         double s = A(i);
         for (int j = 0; j < dim; ++j)
         {
            s += B(i, j) * x(j);
            for (int k = 0; k < dim; ++k) { s += 0.5 * C(i, j, k) * x(j) * x(k); }
         }
         v(i) = s;
      }
   };
   MatFn gf = [ = ](const Vector & x, DenseMatrix & g)
   {
      g.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            double s = B(i, j);
            for (int k = 0; k < dim; ++k) { s += C(i, j, k) * x(k); }
            g(i, j) = s;
         }
      }
   };
   VecFn ff = [ = ](const Vector&, Vector & v)
   {
      v.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         double tr = 0.0, gd = 0.0;
         for (int k = 0; k < dim; ++k)
         {
            tr += C(i, k, k);
            gd += C(k, k, i);
         }
         v(i) = -(mu * tr + (lambda + mu) * gd);
      }
   };
   return std::make_unique<Exact>(dim, uf, gf, ff, lambda, mu);
}

/// Non-affine elements: a smooth distortion of the nodes.
void Distort(Mesh& m, int dim)
{
   m.Transform([dim](const Vector & x, Vector & y)
   {
      y = x;
      y(0) += 0.1 * std::sin(M_PI * x(1)) * x(0);
      y(1) += 0.07 * std::sin(M_PI * x(0));
      if (dim == 3) { y(2) += 0.05 * x(0) * x(1) + 0.04 * std::sin(M_PI * x(1)); }
   });
}

/// Per-element stiffness factors in [0.5, 1.5], from the element centres
/// (no communication: a rank without elements takes no part).
Vector Factors(ParMesh& mesh)
{
   Vector f(mesh.GetNE()), c;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      const double z = c.Size() == 3 ? c(2) : 0.0;
      f(e) = 1.0 + 0.5 * std::sin(3.1 * c(0) + 1.7 * c(1) + 1.3 * z);
   }
   return f;
}

/// The largest |a - b| over the largest |b|, across ranks.
double RelMax(const Vector& a, const Vector& b)
{
   const real_t* pa = a.HostRead();
   const real_t* pb = b.HostRead();
   double m[2] = {0.0, 0.0};
   for (int i = 0; i < a.Size(); ++i)
   {
      m[0] = std::max(m[0], std::abs(pa[i] - pb[i]));
      m[1] = std::max(m[1], std::abs(pb[i]));
   }
   MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m[0] / m[1];
}

/// What LinearElasticity's K and diagonal are checked against.
enum class Ref
{
   /// MFEM's assembled ElasticityIntegrator matrix and its diagonal.
   Assembled,
   /// MFEM's partial assembly: its operator and AssembleDiagonal (|P|^T d
   /// on a nonconforming mesh).
   Partial,
   /// St. Venant-Kirchhoff's tangent at u = 0 (the same operator through
   /// the other material), the diagonal probed by unit vectors.
   Svk
};

/// The relative errors of K x and of the diagonal against @p ref (the
/// element factors in both).
std::pair<double, double> Compare(ParMesh& mesh, int order, Ref ref)
{
   const double lambda = 1.7, mu = 0.6;
   const Vector f = Factors(mesh);
   LinearElasticity::Options o = Opts(order, lambda, mu);
   o.element_stiffness = &f;
   LinearElasticity s(mesh, o);
   ParFiniteElementSpace& fes = s.Space();
   const int n = fes.GetTrueVSize();
   Vector x(n), y1(n), y2(n), d1(n), d2(n);
   x.Randomize(1 + Mpi::WorldRank());
   s.ApplyStiffness(x, y1);
   s.StiffnessDiagonal(d1);

   incns::ElementScaledCoefficient lam(lambda, f), mc(mu, f);
   ParBilinearForm form(&fes);
   std::unique_ptr<HypreParMatrix> M;
   std::unique_ptr<incns::HyperelasticOperator> svk;
   Vector le(mesh.GetNE()), me(mesh.GetNE());
   if (ref == Ref::Svk)
   {
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         le(e) = lambda * f(e);
         me(e) = mu * f(e);
      }
      svk = std::make_unique<incns::HyperelasticOperator>(
               fes, s.DomainRule(), incns::HyperelasticMaterial::StVenantKirchhoff, le,
               me);
      Vector zero(n);
      zero.UseDevice(true);
      zero = 0.0;
      svk->GetGradient(zero).Mult(x, y2);
      // The diagonal at a few dofs: e_g^T K e_g.
      const HYPRE_BigInt ng = fes.GlobalTrueVSize(), off = fes.GetMyTDofOffset();
      double m[2] = {0.0, 0.0};
      for (int j = 0; j < 6; ++j)
      {
         const HYPRE_BigInt g = (2 * j + 1) * ng / 12;
         const bool mine = g >= off && g < off + n;
         Vector e(n), ke(n);
         e.UseDevice(true);
         e = 0.0;
         if (mine) { e.HostReadWrite()[g - off] = 1.0; }
         s.ApplyStiffness(e, ke);
         if (mine)
         {
            const double r = ke.HostRead()[g - off], v = d1.HostRead()[g - off];
            m[0] = std::max(m[0], std::abs(r - v));
            m[1] = std::max(m[1], std::abs(r));
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, m, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      y1 -= y2;
      return {std::sqrt(InnerProduct(MPI_COMM_WORLD, y1, y1) /
                        InnerProduct(MPI_COMM_WORLD, y2, y2)), m[0] / m[1]};
   }
   auto* integ = new ElasticityIntegrator(lam, mc);
   integ->SetIntRule(&s.DomainRule());
   form.AddDomainIntegrator(integ);
   if (ref == Ref::Assembled)
   {
      form.Assemble();
      form.Finalize();
      M.reset(form.ParallelAssemble());
      M->Mult(x, y2);
      M->GetDiag(d2);
   }
   else
   {
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form.Assemble();
      OperatorPtr A;
      Array<int> none;
      form.FormSystemMatrix(none, A);
      A->Mult(x, y2);
      d2.UseDevice(true);
      form.AssembleDiagonal(d2);
   }
   y1 -= y2;
   const double kdiff = std::sqrt(InnerProduct(MPI_COMM_WORLD, y1, y1) /
                                  InnerProduct(MPI_COMM_WORLD, y2, y2));
   return {kdiff, RelMax(d1, d2)};
}

} // namespace

TEST(LinearElasticity, E1_OperatorAndDiagonalMatchTheReferences)
{
   // MFEM's partial assembly cannot launch its 3D kernel on a GPU past
   // 1024 threads per block, 3 (k + 1)^3.
   const bool gpu = Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK);
   for (int dim : {2, 3})
   {
      for (int order = 1; order <= 7; ++order)
      {
         // Curved (quadratic geometry), distorted elements.
         Mesh serial = dim == 2 ?
                       Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true, 1.0, 1.0) :
                       Mesh::MakeCartesian3D(2, 2, 1, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
         serial.SetCurvature(2);
         Distort(serial, dim);
         auto mesh = Partition(serial);
         // The assembled matrix takes tens of seconds from 3D Q4 (MFEM's
         // element matrices, then the sparse insertion).
         std::vector<Ref> refs;
         if (dim == 2 || order <= 3) { refs.push_back(Ref::Assembled); }
         else
         {
            const int threads = 3 * (order + 1) * (order + 1) * (order + 1);
            if (!gpu || threads <= 1024) { refs.push_back(Ref::Partial); }
            refs.push_back(Ref::Svk);
         }
         for (Ref r : refs)
         {
            const auto [kdiff, ddiff] = Compare(*mesh, order, r);
            EXPECT_LT(kdiff, 1e-12) << "dim " << dim << " order " << order << " ref "
                                    << static_cast<int>(r);
            EXPECT_LT(ddiff, 1e-12) << "dim " << dim << " order " << order << " ref "
                                    << static_cast<int>(r);
         }
      }
   }
   // Hanging nodes (one element refined): K against the assembled matrix,
   // the diagonal against MFEM's partial assembly (|P|^T d there).
   for (int dim : {2, 3})
   {
      Mesh serial = dim == 2 ?
                    Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true, 1.0, 1.0) :
                    Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
      serial.EnsureNCMesh();
      Array<Refinement> refs;
      refs.Append(Refinement(0));
      serial.GeneralRefinement(refs);
      Distort(serial, dim);
      auto mesh = Partition(serial);
      const int order = dim == 2 ? 3 : 2;
      EXPECT_LT(Compare(*mesh, order, Ref::Assembled).first, 1e-12)
            << "nonconforming, dim " << dim;
      const auto [kdiff, ddiff] = Compare(*mesh, order, Ref::Partial);
      EXPECT_LT(kdiff, 1e-12) << "nonconforming, dim " << dim;
      EXPECT_LT(ddiff, 1e-12) << "nonconforming, dim " << dim;
   }
}

TEST(LinearElasticity, E2_QuadraticDisplacementIsExact)
{
   for (int dim : {2, 3})
   {
      auto mesh = UnitBox(dim, dim == 2 ? 3 : 2);
      const double lam = 1.3, mu = 0.8;
      LinearElasticity s(*mesh, Opts(2, lam, mu));
      auto ex = Quadratic(dim, lam, mu);
      const std::vector<int> held =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; });
      ASSERT_EQ(held.size(), 1u);
      ApplyExact(s, *mesh, *ex, held);
      ASSERT_TRUE(s.Solve());
      const Errors e = ComputeErrors(s.Displacement(), *ex, s.Rules(), 8);
      EXPECT_LT(e.l2, 1e-8 * e.u_norm) << "dim " << dim;
      EXPECT_LT(e.h1, 1e-7 * e.u_norm) << "dim " << dim;
   }
}

TEST(LinearElasticity, E3_TimoshenkoCantileverIsExactAtQ3)
{
   const double E = 1000.0, nu = 0.3, P = 2.0, L = 8.0, D = 1.0;
   Mesh serial = Mesh::MakeCartesian2D(8, 2, Element::QUADRILATERAL, true, L, D);
   serial.Transform([D](const Vector & x, Vector & y) { y = x; y(1) -= 0.5 * D; });
   auto mesh = Partition(serial);
   const double lam = E * nu / (1.0 - nu * nu), mu = E / (2.0 * (1.0 + nu));
   LinearElasticity s(*mesh, Opts(3, lam, mu));
   auto ex = TimoshenkoCantilever(E, nu, P, L, D);
   const std::vector<int> held = AttributesWhere(*mesh, [L](const Vector & c)
   {
      return std::abs(c(0) - L) < 1e-12;
   });
   ASSERT_EQ(held.size(), 1u);
   ApplyExact(s, *mesh, *ex, held);
   ASSERT_TRUE(s.Solve());
   const Errors e = ComputeErrors(s.Displacement(), *ex, s.Rules(), 10);
   // Exact to the solver tolerance (Q2 is off by ~1e-3 here).
   EXPECT_LT(e.l2, 1e-7 * e.u_norm);
   Vector tip(2), u;
   tip = 0.0;
   ASSERT_TRUE(s.DisplacementAt(tip, u));
   const double I = D * D * D / 12.0;
   const double deflection = P * L * L * L / (3.0 * E * I);
   EXPECT_NEAR(u(1), deflection, 1e-7 * deflection);
   EXPECT_NEAR(u(0), 0.0, 1e-7 * deflection);
}

TEST(LinearElasticity, E4_EnergyIdentityAndBothPreconditioners)
{
   for (int dim : {2, 3})
   {
      auto mesh = UnitBox(dim, dim == 2 ? 8 : 3);
      const double lam = 1.5, mu = 1.0;
      Vector t(dim), f(dim);
      t = 0.0;
      t(1) = -0.2;
      f = 0.05;
      VectorConstantCoefficient tc(t), fc(f);
      const std::vector<int> held =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; });
      const std::vector<int> loaded =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) > 1.0 - 1e-12; });
      ASSERT_EQ(held.size(), 1u);
      ASSERT_EQ(loaded.size(), 1u);

      Vector sol[2];
      int its[2] = {0, 0};
      for (int k = 0; k < 2; ++k)
      {
         LinearElasticity::Options o = Opts(3, lam, mu);
         o.preconditioner = k == 0 ? incns::ElasticityPC::LorAmg :
                            incns::ElasticityPC::Jacobi;
         LinearElasticity s(*mesh, o);
         s.AddFixed(held[0]);
         s.AddTraction(loaded[0], tc);
         s.SetBodyForce(fc);
         ASSERT_TRUE(s.Solve()) << "dim " << dim << " pc " << k;
         its[k] = s.Iterations();
         const double W = s.StrainEnergy(), work = s.ExternalWork();
         EXPECT_GT(W, 0.0);
         EXPECT_NEAR(2.0 * W, work, 1e-8 * work) << "dim " << dim << " pc " << k;
         s.Displacement().GetTrueDofs(sol[k]);
      }
      if (Mpi::Root())
      {
         std::printf("[ info     ] E4 dim %d: CG iterations LOR-AMG %d, Jacobi %d\n",
                     dim, its[0], its[1]);
      }
      EXPECT_LT(its[0], its[1]) << "dim " << dim;
      const double ref = std::sqrt(InnerProduct(MPI_COMM_WORLD, sol[0], sol[0]));
      sol[1] -= sol[0];
      EXPECT_LT(std::sqrt(InnerProduct(MPI_COMM_WORLD, sol[1], sol[1])), 1e-8 * ref);
   }
}

TEST(LinearElasticity, E5_DeckThroughElasticityCase)
{
   const std::string deck =
      "equation: linear_elasticity\n"
      "mesh:\n"
      "  dim: 2\n"
      "  elements: [8, 2]\n"
      "  lengths: [4.0, 1.0]\n"
      "  periodic: [false, false]\n"
      "solid:\n"
      "  youngs_modulus: 100.0\n"
      "  poisson_ratio: 0.25\n"
      "  plane: stress\n"
      "  order: 2\n"
      "  rtol: 1.0e-12\n"
      "  probe: [4.0, 0.5]\n"
      "boundary_conditions:\n"
      "  - {select: [xmin], type: fixed}\n"
      "  - {select: [xmax], type: traction, value: [0.0, -0.1]}\n"
      "  - {select: [ymin, ymax], type: free}\n";
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   EXPECT_EQ(p.equation, incns::Equation::LinearElasticity);
   EXPECT_EQ(p.solid.plane, incns::PlaneModel::Stress);
   EXPECT_TRUE(p.solid.has_probe);
   const incns::LinearElasticity::Options o = incns::ElasticityOptions(p);
   EXPECT_EQ(o.order, 2);
   EXPECT_NEAR(o.mu, 100.0 / 2.5, 1e-12);
   EXPECT_NEAR(o.lambda, 100.0 * 0.25 / (1.0 - 0.0625), 1e-12);

   auto mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   ASSERT_TRUE(c.Solve());
   // Bends down, and the energy identity holds through the deck's groups.
   Vector x(2), u;
   x(0) = 4.0;
   x(1) = 0.5;
   ASSERT_TRUE(c.Solver().DisplacementAt(x, u));
   EXPECT_LT(u(1), 0.0);
   const double W = c.Solver().StrainEnergy(), work = c.Solver().ExternalWork();
   EXPECT_NEAR(2.0 * W, work, 1e-8 * work);

   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   const std::string path = "linear_elasticity_E5_np" + std::to_string(np) +
                            "_summary.json";
   c.WriteSummary(path, "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::ifstream f(path);
      ASSERT_TRUE(f.good());
      std::stringstream ss;
      ss << f.rdbuf();
      EXPECT_NE(ss.str().find("\"equation\": \"linear_elasticity\""),
                std::string::npos) << ss.str();
      EXPECT_NE(ss.str().find("\"strain_energy\""), std::string::npos);
      std::filesystem::remove(path);
   }
}

TEST(LinearElasticity, E6_ElasticSupportAndReaction)
{
   for (int dim : {2, 3})
   {
      auto mesh = UnitBox(dim, dim == 2 ? 4 : 2);
      // lambda = 0: uniaxial tension without lateral strain, E = 2 mu.
      const double mu = 1.0, E = 2.0 * mu, k = 3.0, tx = 0.4;
      Vector t(dim);
      t = 0.0;
      t(0) = tx;
      VectorConstantCoefficient tc(t);
      const std::vector<int> held =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) < 1e-12; });
      const std::vector<int> loaded =
      AttributesWhere(*mesh, [](const Vector & c) { return c(0) > 1.0 - 1e-12; });
      ASSERT_EQ(held.size(), 1u);
      ASSERT_EQ(loaded.size(), 1u);
      // E u' = t - k u at x = 1, u(0) = 0: u = a x.
      const double a = tx / (E + k);
      for (int pc = 0; pc < 2; ++pc)
      {
         LinearElasticity::Options o = Opts(2, 0.0, mu);
         o.preconditioner = pc == 0 ? incns::ElasticityPC::LorAmg :
                            incns::ElasticityPC::Jacobi;
         LinearElasticity s(*mesh, o);
         s.AddFixed(held[0]);
         s.AddTraction(loaded[0], tc);
         s.AddSupport(loaded[0], k);
         s.TrackReaction();
         ASSERT_TRUE(s.Solve()) << "dim " << dim << " pc " << pc;
         VectorFunctionCoefficient exact(dim, [a](const Vector & x, Vector & u)
         {
            u = 0.0;
            u(0) = a * x(0);
         });
         EXPECT_LT(s.Displacement().ComputeMaxError(exact), 1e-9 * a)
               << "dim " << dim << " pc " << pc;
         // The clamp's reaction balances the end's net load (t - k u) A.
         const Vector& R = s.Reaction();
         const int n = R.Size() / dim; // byNODES
         double rx = 0.0;
         const double* rr = R.HostRead();
         for (int i = 0; i < n; ++i) { rx += rr[i]; }
         MPI_Allreduce(MPI_IN_PLACE, &rx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         EXPECT_NEAR(rx, -(tx - k * a), 1e-9) << "dim " << dim << " pc " << pc;
      }
   }
}
