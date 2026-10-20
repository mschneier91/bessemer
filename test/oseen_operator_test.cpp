// The Oseen convection of the semi-implicit and implicit BDF2 schemes
// (oseen/oseen_convection.hpp, docs/design/oseen_bdf2.md):
//  O1 the true dofs of a byNODES vector space are the scalar space's, block
//     by component (what ComponentwiseOperator relies on);
//  O2 the partially assembled operator, OSS included, equals the fully
//     assembled one built from the same quadrature data (curved meshes);
//  O3 the convection C(w) + 1/2 (div w) M is skew-symmetric on fields that
//     vanish on the boundary (affine mesh, exact quadrature) -- the test that
//     catches a wrong sign or a wrong layout of 1/2 div w;
//  O4 the OSS term S is symmetric, and positive semidefinite on affine
//     meshes: there the GLL-lumped projection mass bounds the Gauss one from
//     above (GLL quadrature overestimates u^2 for u in Q_k), which is what
//     makes K_tau - C_tau^T W_tau^-1 C_tau >= 0. On curved elements the
//     varying Jacobian breaks that bound: this strongly curved test mesh
//     gives eigenvalues down to -0.3 % of the largest (2026-10-20), bounded
//     here at 1 %;
//  O5 two applies agree bitwise (no atomics on a device).
// Runs on whatever device the test main configures (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "oseen/oseen_convection.hpp"
#include "oseen/oseen_wind.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace mfem;
using incns::ComponentwiseOperator;
using incns::OseenWindData;
using incns::RuleBook;
using incns::ScalarOseenOperator;

namespace
{

// A wind with nonzero divergence in every direction.
void WindFn(const Vector& x, Vector& w)
{
   const real_t X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   w(0) = 1.0 + 0.5 * std::sin(M_PI * Y) + 0.3 * X * Z;
   w(1) = 0.7 * std::cos(M_PI * X) + 0.2 * Y * Y;
   if (w.Size() == 3) { w(2) = 0.4 + 0.3 * std::sin(M_PI * X * Y) + 0.1 * Z; }
}

// An n^dim box, curved (order k nodes moved by a smooth map) unless affine.
Mesh BoxMesh(int dim, int n, int k, bool curved)
{
   Mesh m = (dim == 3) ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
            : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   if (curved)
   {
      m.SetCurvature(k);
      m.Transform([](const Vector & x, Vector & y)
      {
         y = x;
         const real_t s = 0.06 * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
         y(0) += s;
         y(1) -= 0.5 * s;
         if (x.Size() == 3) { y(2) += 0.04 * std::sin(M_PI * x(2)) * std::cos(M_PI * x(0)); }
      });
   }
   return m;
}

struct Problem
{
   Problem(MPI_Comm comm, int dim, int n, int k, bool curved, double nu)
      : serial(BoxMesh(dim, n, k, curved)), mesh(comm, serial), fec(k, dim),
        vfes(&mesh, &fec, dim, Ordering::byNODES), sfes(&mesh, &fec),
        wind_gf(&vfes), data(vfes, rules, nu), op(sfes, data, true)
   {
      VectorFunctionCoefficient wc(dim, WindFn);
      wind_gf.ProjectCoefficient(wc);
      Vector W;
      wind_gf.GetTrueDofs(W);
      wind_gf.SetFromTrueDofs(W);
      data.Update(wind_gf);
      op.Assemble();
   }
   Mesh serial;
   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace vfes, sfes;
   ParGridFunction wind_gf;
   RuleBook rules;
   OseenWindData data;
   ScalarOseenOperator op;
};

Vector Random(int n, int seed)
{
   Vector x(n);
   x.Randomize(seed + Mpi::WorldRank());
   x.UseDevice(true);
   return x;
}

real_t Dot(const Vector& a, const Vector& b)
{
   return InnerProduct(MPI_COMM_WORLD, a, b);
}

} // namespace

// O1
TEST(OseenOperator, O1_TrueDofsByComponent)
{
   for (int dim : {2, 3})
   {
      Problem p(MPI_COMM_WORLD, dim, 3, 2, true, 1e-3);
      const int n = p.sfes.GetTrueVSize();
      ASSERT_EQ(p.vfes.GetTrueVSize(), dim * n);
      Vector W;
      p.wind_gf.GetTrueDofs(W);
      W.HostRead();
      for (int c = 0; c < dim; c++)
      {
         // Component c of the wind through the scalar space.
         ParGridFunction s(&p.sfes);
         auto f = [c](const Vector & x)
         {
            Vector w(x.Size());
            WindFn(x, w);
            return w(c);
         };
         FunctionCoefficient fc(f);
         s.ProjectCoefficient(fc);
         Vector S;
         s.GetTrueDofs(S);
         S.HostRead();
         for (int i = 0; i < n; i++) { ASSERT_EQ(W(c * n + i), S(i)) << dim << " " << c; }
      }
      // The componentwise apply equals the scalar apply per block.
      ComponentwiseOperator V(p.op, dim);
      Vector x = Random(dim * n, 3), y(dim * n), yc(n), xc(n);
      y.UseDevice(true); yc.UseDevice(true); xc.UseDevice(true);
      V.Mult(x, y);
      x.HostRead();
      y.HostRead();
      for (int c = 0; c < dim; c++)
      {
         // Through HostWrite: operator() writes the host copy without
         // invalidating the device one, which a GPU apply would then read.
         real_t* xh = xc.HostWrite();
         for (int i = 0; i < n; i++) { xh[i] = x(c * n + i); }
         p.op.Mult(xc, yc);
         yc.HostRead();
         for (int i = 0; i < n; i++) { ASSERT_EQ(y(c * n + i), yc(i)); }
      }
   }
}

// O2
TEST(OseenOperator, O2_PartialAssemblyMatchesFullAssembly)
{
   for (int dim : {2, 3})
   {
      const int k = (dim == 2) ? 4 : 3;
      Problem p(MPI_COMM_WORLD, dim, 3, k, true, 1e-3);
      OseenWindData& d = p.data;
      // Full assembly evaluates the coefficients at ip.index: make sure the
      // rules carry their point indices.
      for (const IntegrationRule* ir : {&d.ConvRule(), &d.OssRule(), &d.GllRule()})
      {
         const_cast<IntegrationRule*>(ir)->SetPointIndices();
      }
      auto assemble = [&](BilinearFormIntegrator * a, BilinearFormIntegrator * b)
      {
         auto f = std::make_unique<ParBilinearForm>(&p.sfes);
         f->AddDomainIntegrator(a);
         if (b) { f->AddDomainIntegrator(b); }
         f->Assemble();
         f->Finalize();
         return std::unique_ptr<HypreParMatrix>(f->ParallelAssemble());
      };
      auto* ci = new ConvectionIntegrator(d.Wind(), 1.0);
      ci->SetIntRule(&d.ConvRule());
      auto* mi = new MassIntegrator(d.HalfDivWind());
      mi->SetIntRule(&d.ConvRule());
      auto Conv = assemble(ci, mi);
      auto* ki = new DiffusionIntegrator(d.TauWW());
      ki->SetIntRule(&d.OssRule());
      auto Ktau = assemble(ki, nullptr);
      auto* cti = new ConvectionIntegrator(d.TauW(), 1.0);
      cti->SetIntRule(&d.OssRule());
      auto Ctau = assemble(cti, nullptr);
      auto* wi = new MassIntegrator(d.TauGll());
      wi->SetIntRule(&d.GllRule());
      auto Wt = assemble(wi, nullptr);
      Vector wdiag;
      Wt->GetDiag(wdiag);

      const int n = p.sfes.GetTrueVSize();
      Vector x = Random(n, 7), y(n), r(n), t(n), t2(n);
      y.UseDevice(true);
      p.op.Mult(x, y);
      // Reference on the host: r = (Conv + Ktau) x - Ctau^T W^-1 Ctau x.
      x.HostRead();
      Conv->Mult(x, r);
      Ktau->Mult(1.0, x, 1.0, r);
      Ctau->Mult(x, t);
      t.HostReadWrite();
      wdiag.HostRead();
      for (int i = 0; i < n; i++) { t(i) = (wdiag(i) > 0.0) ? t(i) / wdiag(i) : 0.0; }
      Ctau->MultTranspose(t, t2);
      r -= t2;
      y.HostRead();
      Vector e(r);
      e -= y;
      const real_t rel = std::sqrt(Dot(e, e) / Dot(r, r));
      EXPECT_LT(rel, 1e-12) << "dim " << dim;
   }
}

// O3
TEST(OseenOperator, O3_ConvectionIsSkewSymmetric)
{
   for (int dim : {2, 3})
   {
      const int k = (dim == 2) ? 4 : 3;
      Problem p(MPI_COMM_WORLD, dim, 3, k, false, 1e-3);
      Array<int> ess_bdr(p.mesh.bdr_attributes.Max()), ess;
      ess_bdr = 1;
      p.sfes.GetEssentialTrueDofs(ess_bdr, ess);
      const int n = p.sfes.GetTrueVSize();
      Vector x = Random(n, 11), z = Random(n, 13), cx(n), cz(n);
      cx.UseDevice(true); cz.UseDevice(true);
      x.HostReadWrite();
      z.HostReadWrite();
      for (int i : ess) { x(i) = 0.0; z(i) = 0.0; }
      p.op.MultConvection(x, cx);
      p.op.MultConvection(z, cz);
      const real_t zcx = Dot(z, cx), xcz = Dot(x, cz);
      // c(w; x, z) = -c(w; z, x), and c(w; x, x) = 0.
      EXPECT_LT(std::fabs(zcx + xcz), 1e-12 * std::fabs(zcx)) << "dim " << dim;
      EXPECT_LT(std::fabs(Dot(x, cx)), 1e-12 * std::fabs(zcx)) << "dim " << dim;
      EXPECT_GT(std::fabs(zcx), 1e-3); // not trivially zero
   }
}

// O4: on each rank alone (MPI_COMM_SELF), a dense S from unit vectors.
TEST(OseenOperator, O4_StabilizationIsSymmetricPositiveSemidefinite)
{
   for (int dim : {2, 3})
   {
      const int k = (dim == 2) ? 3 : 2;
      for (bool curved : {false, true})
         for (double nu : {1e-1, 1e-3, 1e-6})
         {
            Problem p(MPI_COMM_SELF, dim, (dim == 2) ? 3 : 2, k, curved, nu);
            const int n = p.sfes.GetTrueVSize();
            DenseMatrix S(n);
            Vector e(n), s(n);
            e.UseDevice(true);
            s.UseDevice(true);
            for (int j = 0; j < n; j++)
            {
               e = 0.0;
               e.HostReadWrite()[j] = 1.0;
               p.op.MultStabilization(e, s);
               s.HostRead();
               for (int i = 0; i < n; i++) { S(i, j) = s(i); }
            }
            real_t smax = 0.0, asym = 0.0;
            for (int j = 0; j < n; j++)
            {
               for (int i = 0; i < n; i++)
               {
                  smax = std::max(smax, std::fabs(S(i, j)));
                  asym = std::max(asym, std::fabs(S(i, j) - S(j, i)));
               }
            }
            ASSERT_GT(smax, 0.0);
            EXPECT_LT(asym, 1e-12 * smax) << "dim " << dim << " nu " << nu;
            S.Symmetrize();
            Vector ev;
            S.Eigenvalues(ev);
            const real_t lmax = ev.Max(), lmin = ev.Min();
            EXPECT_GE(lmin, (curved ? -1e-2 : -1e-12) * lmax)
                  << "dim " << dim << " nu " << nu << (curved ? " curved" : " affine")
                  << ": eigenvalues " << lmin << " .. " << lmax;
            if (curved && Mpi::Root())
            {
               std::cout << "[ oss      ] dim " << dim << " nu " << nu
                         << " curved: lambda_min / lambda_max = " << lmin / lmax << std::endl;
            }
         }
   }
}

// O5
TEST(OseenOperator, O5_ApplyIsDeterministic)
{
   Problem p(MPI_COMM_WORLD, 3, 3, 4, true, 1e-4);
   const int n = p.sfes.GetTrueVSize();
   Vector x = Random(n, 17), y1(n), y2(n);
   y1.UseDevice(true);
   y2.UseDevice(true);
   p.op.Mult(x, y1);
   p.op.Mult(x, y2);
   y1.HostRead();
   y2.HostRead();
   for (int i = 0; i < n; i++) { ASSERT_EQ(y1(i), y2(i)); }
}
