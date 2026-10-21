// The hyperelastic operators (solid/hyperelastic_operator.hpp,
// docs/design/hyperelasticity_spec.md), milestone M1:
//  H1 the St. Venant-Kirchhoff material at a point, 2D and 3D: S and P against
//     an independent DenseMatrix evaluation; the tangent against central
//     differences of P; frame indifference P(QF) = Q P(F); no stress under a
//     rotation; the small-strain limit P(I + eps H) / eps -> lambda tr(H) I +
//     mu (H + H^T), the linear model's stress;
//  H2 the partially assembled residual and tangent equal MFEM's
//     HyperelasticNLFIntegrator with an independent SVK model (full assembly,
//     its unknown the deformed position X + u) at a finite random displacement
//     on curved meshes: 2D Q3, 3D Q2, 2D Q6; at 3D Q6 the residual only (that
//     full-assembly tangent is too slow for a test; H3 covers it);
//  H3 the tangent is the residual's derivative (the Taylor remainder falls as
//     eps^2) and the residual the energy's (a central difference), 2D Q3, 3D Q2,
//     3D Q6;
//  HD two applies agree bitwise (no atomics on a device).
//  HM the three tangent storages give the same tangent (to roundoff), the
//     displacement storage keeps the least memory, and borrowed work buffers
//     (ElementScratch) give bitwise the results of owned ones.
// Runs on whatever device the test main configures (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "quadrature/rule_book.hpp"
#include "solid/hyperelastic_operator.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::HyperelasticMaterial;
using incns::HyperelasticOperator;
using incns::RuleBook;
using incns::TangentStorage;
using incns::ElementScratch;
using incns::StVenantKirchhoff;

namespace
{

constexpr double kLambda = 1.7, kMu = 0.9;

/// E = (F^T F - I) / 2.
void Strain(const DenseMatrix& F, DenseMatrix& E)
{
   const int dim = F.Height();
   E.SetSize(dim);
   MultAtB(F, F, E);
   E *= 0.5;
   for (int i = 0; i < dim; i++) { E(i, i) -= 0.5; }
}

/// S = lambda tr(E) I + 2 mu E.
void SecondPiola(const DenseMatrix& E, double lambda, double mu, DenseMatrix& S)
{
   S = E;
   S *= 2.0 * mu;
   const double tr = E.Trace();
   for (int i = 0; i < E.Height(); i++) { S(i, i) += lambda * tr; }
}

/// The SVK model written independently of solid/hyperelastic_material.hpp,
/// with DenseMatrix operations, for MFEM's HyperelasticNLFIntegrator.
class SvkReference : public HyperelasticModel
{
public:
   SvkReference(double lambda, double mu) : lambda_(lambda), mu_(mu) {}

   real_t EvalW(const DenseMatrix& F) const override
   {
      DenseMatrix E;
      Strain(F, E);
      const double tr = E.Trace();
      return 0.5 * lambda_ * tr * tr + mu_ * E.FNorm2();
   }

   void EvalP(const DenseMatrix& F, DenseMatrix& P) const override
   {
      DenseMatrix E, S;
      Strain(F, E);
      SecondPiola(E, lambda_, mu_, S);
      P.SetSize(F.Height());
      Mult(F, S, P);
   }

   void AssembleH(const DenseMatrix& F, const DenseMatrix& DS, const real_t w,
                  DenseMatrix& A) const override
   {
      const int dof = DS.Height(), dim = DS.Width();
      DenseMatrix E, S, dF(dim), dE(dim), dS, dP(dim);
      Strain(F, E);
      SecondPiola(E, lambda_, mu_, S);
      for (int j = 0; j < dof; j++)
      {
         for (int b = 0; b < dim; b++)
         {
            // dF = e_b grad(phi_j)^T, dE = sym(F^T dF), dP = dF S + F dS.
            dF = 0.0;
            for (int l = 0; l < dim; l++) { dF(b, l) = DS(j, l); }
            MultAtB(F, dF, dE);
            dE.Symmetrize();
            SecondPiola(dE, lambda_, mu_, dS);
            Mult(dF, S, dP);
            AddMult(F, dS, dP);
            for (int i = 0; i < dof; i++)
            {
               for (int a = 0; a < dim; a++)
               {
                  double s = 0.0;
                  for (int k = 0; k < dim; k++) { s += dP(a, k) * DS(i, k); }
                  A(i + a * dof, j + b * dof) += w * s;
               }
            }
         }
      }
   }

private:
   double lambda_, mu_;
};

// An n^dim box with order-k nodes moved by a smooth map (curved elements).
Mesh BoxMesh(int dim, int n, int k)
{
   Mesh m = (dim == 3) ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
            : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   m.SetCurvature(k);
   m.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      const real_t s = 0.06 * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
      y(0) += s;
      y(1) -= 0.5 * s;
      if (x.Size() == 3) { y(2) += 0.04 * std::sin(M_PI * x(2)) * std::cos(M_PI * x(0)); }
   });
   return m;
}

struct Problem
{
   Problem(int dim, int n, int k)
      : serial(BoxMesh(dim, n, k)), mesh(MPI_COMM_WORLD, serial), fec(k, dim),
        fes(&mesh, &fec, dim, Ordering::byNODES)
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      ir = &rules.Get(geom, 2 * k + 1);
      lambda.SetSize(mesh.GetNE());
      mu.SetSize(mesh.GetNE());
      lambda = kLambda;
      mu = kMu;
      op = std::make_unique<HyperelasticOperator>(
              fes, *ir, HyperelasticMaterial::StVenantKirchhoff, lambda, mu);
   }
   ~Problem() { mesh.DeleteGeometricFactors(); } // keyed by rules' rules

   /// The true dofs of the identity map X (exact: the nodes are order k).
   Vector Identity()
   {
      ParGridFunction X(&fes);
      VectorFunctionCoefficient id(fes.GetVDim(), [](const Vector & x, Vector & y)
      {
         y = x;
      });
      X.ProjectCoefficient(id);
      Vector t;
      X.GetTrueDofs(t);
      return t;
   }

   Mesh serial;
   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   RuleBook rules;
   const IntegrationRule* ir = nullptr;
   Vector lambda, mu;
   std::unique_ptr<HyperelasticOperator> op;
};

Vector Random(int n, int seed, double scale)
{
   Vector x(n);
   x.Randomize(seed + 7 * Mpi::WorldRank());
   x -= 0.5;
   x *= scale;
   x.UseDevice(true);
   return x;
}

real_t Dot(const Vector& a, const Vector& b)
{
   return InnerProduct(MPI_COMM_WORLD, a, b);
}

real_t Norm(const Vector& a) { return std::sqrt(Dot(a, a)); }

/// A random F = I + s R as DenseMatrix and column-major array.
DenseMatrix RandomF(int dim, int seed, double s)
{
   DenseMatrix F(dim);
   Vector r(dim * dim);
   r.Randomize(seed);
   for (int j = 0; j < dim; j++)
   {
      for (int i = 0; i < dim; i++) { F(i, j) = (i == j ? 1.0 : 0.0) + s * (r(i + dim * j) - 0.5); }
   }
   return F;
}

/// A rotation (Rodrigues) about a fixed axis; in 2D the angle's rotation.
DenseMatrix Rotation(int dim, double angle)
{
   DenseMatrix Q(dim);
   if (dim == 2)
   {
      Q(0, 0) = std::cos(angle); Q(0, 1) = -std::sin(angle);
      Q(1, 0) = std::sin(angle); Q(1, 1) = std::cos(angle);
      return Q;
   }
   double n[3] = {0.36, -0.48, 0.8}; // unit
   const double c = std::cos(angle), s = std::sin(angle);
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         Q(i, j) = (1.0 - c) * n[i] * n[j] + (i == j ? c : 0.0);
      }
   }
   Q(0, 1) -= s * n[2]; Q(1, 0) += s * n[2];
   Q(0, 2) += s * n[1]; Q(2, 0) -= s * n[1];
   Q(1, 2) -= s * n[0]; Q(2, 1) += s * n[0];
   return Q;
}

template <int DIM>
void MaterialChecks()
{
   constexpr int D2 = DIM * DIM;
   SvkReference ref(kLambda, kMu);
   for (int trial = 0; trial < 5; trial++)
   {
      const DenseMatrix F = RandomF(DIM, 11 + trial, 0.6);
      real_t S[D2], P[D2];
      StVenantKirchhoff::Stress<DIM>(F.GetData(), kLambda, kMu, S, P);
      DenseMatrix Pref;
      ref.EvalP(F, Pref);
      double scale = Pref.MaxMaxNorm();
      for (int i = 0; i < D2; i++)
      {
         EXPECT_NEAR(P[i], Pref.GetData()[i], 1e-14 * scale) << "P, dim " << DIM;
      }
      EXPECT_NEAR(StVenantKirchhoff::Energy<DIM>(F.GetData(), kLambda, kMu),
                  ref.EvalW(F), 1e-14 * std::max(1.0, ref.EvalW(F)));

      // The tangent against central differences of P.
      const DenseMatrix dF = RandomF(DIM, 31 + trial, 2.0);
      DenseMatrix dFm(dF);
      for (int i = 0; i < DIM; i++) { dFm(i, i) -= 1.0; } // drop RandomF's I
      real_t dP[D2], Pp[D2], Pm[D2], Fp[D2], Fm[D2], Sx[D2];
      StVenantKirchhoff::Tangent<DIM>(F.GetData(), S, dFm.GetData(), kLambda, kMu,
                                      dP);
      const double eps = 1e-5;
      for (int i = 0; i < D2; i++)
      {
         Fp[i] = F.GetData()[i] + eps * dFm.GetData()[i];
         Fm[i] = F.GetData()[i] - eps * dFm.GetData()[i];
      }
      StVenantKirchhoff::Stress<DIM>(Fp, kLambda, kMu, Sx, Pp);
      StVenantKirchhoff::Stress<DIM>(Fm, kLambda, kMu, Sx, Pm);
      double dmax = 0.0, err = 0.0;
      for (int i = 0; i < D2; i++)
      {
         const double fd = (Pp[i] - Pm[i]) / (2.0 * eps);
         dmax = std::max(dmax, std::fabs(dP[i]));
         err = std::max(err, std::fabs(fd - dP[i]));
      }
      EXPECT_LT(err, 1e-8 * dmax) << "tangent, dim " << DIM;

      // Frame indifference: P(QF) = Q P(F).
      const DenseMatrix Q = Rotation(DIM, 0.7 + trial);
      DenseMatrix QF(DIM), QP(DIM);
      Mult(Q, F, QF);
      DenseMatrix Pm_(P, DIM, DIM);
      Mult(Q, Pm_, QP);
      real_t Sq[D2], Pq[D2];
      StVenantKirchhoff::Stress<DIM>(QF.GetData(), kLambda, kMu, Sq, Pq);
      for (int i = 0; i < D2; i++)
      {
         EXPECT_NEAR(Pq[i], QP.GetData()[i], 1e-13 * scale) << "frame, dim " << DIM;
      }
   }
   // No stress under a rotation, however large.
   for (double angle : {0.3, 1.5707963267948966, 3.0})
   {
      const DenseMatrix Q = Rotation(DIM, angle);
      real_t S[D2], P[D2];
      StVenantKirchhoff::Stress<DIM>(Q.GetData(), kLambda, kMu, S, P);
      for (int i = 0; i < D2; i++) { EXPECT_LT(std::fabs(P[i]), 1e-14); }
      EXPECT_LT(StVenantKirchhoff::Energy<DIM>(Q.GetData(), kLambda, kMu), 1e-28);
   }
   // The small-strain limit: P(I + eps H) / eps -> lambda tr(H) I + mu (H + H^T).
   const DenseMatrix H = RandomF(DIM, 51, 2.0);
   DenseMatrix Hm(H);
   for (int i = 0; i < DIM; i++) { Hm(i, i) -= 1.0; }
   DenseMatrix sigma(DIM);
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         sigma(i, j) = kMu * (Hm(i, j) + Hm(j,
                                            i)) + (i == j ? kLambda * Hm.Trace() : 0.0);
      }
   }
   double errs[2];
   int n = 0;
   for (double eps : {1e-3, 1e-4})
   {
      real_t F[D2], S[D2], P[D2];
      for (int i = 0; i < D2; i++) { F[i] = (i % (DIM + 1) == 0 ? 1.0 : 0.0) + eps * Hm.GetData()[i]; }
      StVenantKirchhoff::Stress<DIM>(F, kLambda, kMu, S, P);
      double e = 0.0;
      for (int i = 0; i < D2; i++) { e = std::max(e, std::fabs(P[i] / eps - sigma.GetData()[i])); }
      errs[n++] = e / sigma.MaxMaxNorm();
   }
   EXPECT_LT(errs[0], 1e-2) << "dim " << DIM;
   EXPECT_GT(errs[0] / errs[1],
             8.0) << "linear limit not first order, dim " << DIM;
   EXPECT_LT(errs[0] / errs[1], 12.0) << "dim " << DIM;
}

} // namespace

// H1
TEST(Hyperelasticity, H1_StVenantKirchhoffAtAPoint)
{
   MaterialChecks<2>();
   MaterialChecks<3>();
}

// H2
TEST(Hyperelasticity, H2_OperatorsMatchMfemHyperelasticIntegrator)
{
   // dim, elements per direction, order, check the tangent.
   const int cases[][4] = {{2, 3, 3, 1}, {3, 2, 2, 1}, {2, 2, 6, 1}, {3, 2, 6, 0}};
   for (const auto& c : cases)
   {
      const int dim = c[0], k = c[2];
      Problem p(dim, c[1], k);
      SvkReference model(kLambda, kMu);
      ParNonlinearForm nlf(&p.fes);
      auto* integ = new HyperelasticNLFIntegrator(&model);
      integ->SetIntRule(p.ir);
      nlf.AddDomainIntegrator(integ);

      const int n = p.fes.GetTrueVSize();
      Vector u = Random(n, 3, 0.1), x = p.Identity();
      x.UseDevice(true);
      x += u;
      Vector r(n), rref(n);
      r.UseDevice(true);
      p.op->Mult(u, r);
      x.HostRead();
      nlf.Mult(x, rref);
      rref.HostRead();
      r.HostRead();
      Vector e(r);
      e -= rref;
      ASSERT_GT(Norm(rref), 1e-3);
      EXPECT_LT(Norm(e), 1e-12 * Norm(rref)) << "residual, dim " << dim << " k " << k;

      if (c[3])
      {
         Vector w = Random(n, 5, 1.0), kw(n), kwref(n);
         kw.UseDevice(true);
         p.op->GetGradient(u).Mult(w, kw);
         Operator& K = nlf.GetGradient(x);
         w.HostRead();
         K.Mult(w, kwref);
         kwref.HostRead();
         kw.HostRead();
         Vector ek(kw);
         ek -= kwref;
         EXPECT_LT(Norm(ek), 1e-12 * Norm(kwref)) << "tangent, dim " << dim << " k " <<
               k;
      }
   }
}

// H3
TEST(Hyperelasticity, H3_TangentAndResidualAreDerivatives)
{
   const int cases[][3] = {{2, 3, 3}, {3, 2, 2}, {3, 2, 6}}; // dim, n, order
   for (const auto& c : cases)
   {
      const int dim = c[0], k = c[2];
      Problem p(dim, c[1], k);
      const int n = p.fes.GetTrueVSize();
      Vector u = Random(n, 13, 0.1), w = Random(n, 17, 0.1);
      Vector r0(n), kw(n), r1(n), t(n), d(n);
      for (Vector* v : {&r0, &kw, &r1, &t, &d}) { v->UseDevice(true); }
      p.op->Mult(u, r0);
      p.op->GetGradient(u).Mult(w, kw);

      // Taylor: |R(u + eps w) - R(u) - eps K w| falls as eps^2.
      double prev = 0.0;
      for (double eps : {4e-3, 2e-3, 1e-3})
      {
         add(u, eps, w, t);
         p.op->Mult(t, r1);
         add(r1, -1.0, r0, d);
         d.Add(-eps, kw);
         const double rem = Norm(d);
         if (prev > 0.0)
         {
            EXPECT_GT(prev / rem, 3.5) << "dim " << dim << " k " << k << " eps " << eps;
            EXPECT_LT(prev / rem, 4.6) << "dim " << dim << " k " << k << " eps " << eps;
         }
         prev = rem;
      }

      // R(u) . w against a central difference of the energy.
      const double eps = 1e-4;
      add(u, eps, w, t);
      const double wp = p.op->Energy(t);
      add(u, -eps, w, t);
      const double wm = p.op->Energy(t);
      const double dw = (wp - wm) / (2.0 * eps), rw = Dot(r0, w);
      EXPECT_NEAR(dw, rw, 1e-7 * std::fabs(rw)) << "energy, dim " << dim << " k " <<
            k;
      EXPECT_GT(std::fabs(rw), 1e-8);
   }
}

// HD
TEST(Hyperelasticity, HD_ApplyIsDeterministic)
{
   Problem p(3, 2, 4);
   const int n = p.fes.GetTrueVSize();
   Vector u = Random(n, 19, 0.1), w = Random(n, 23, 1.0);
   Vector r1(n), r2(n), k1(n), k2(n);
   for (Vector* v : {&r1, &r2, &k1, &k2}) { v->UseDevice(true); }
   p.op->Mult(u, r1);
   p.op->Mult(u, r2);
   Operator& K = p.op->GetGradient(u);
   K.Mult(w, k1);
   K.Mult(w, k2);
   r1.HostRead(); r2.HostRead(); k1.HostRead(); k2.HostRead();
   for (int i = 0; i < n; i++)
   {
      ASSERT_EQ(r1(i), r2(i));
      ASSERT_EQ(k1(i), k2(i));
   }
}

// HM
TEST(Hyperelasticity, HM_TangentStoragesAgreeAndScratchIsTransparent)
{
   const int cases[][3] = {{2, 3, 4}, {3, 2, 3}}; // dim, n, order
   for (const auto& c : cases)
   {
      const int dim = c[0], k = c[2];
      Problem p(dim, c[1], k);
      const int n = p.fes.GetTrueVSize();
      Vector u = Random(n, 29, 0.1), w = Random(n, 31, 1.0);
      ElementScratch scratch(p.fes);
      Vector kref(n);
      kref.UseDevice(true);
      p.op->GetGradient(u).Mult(w, kref);
      size_t prev_bytes = p.op->MemoryBytes();
      for (TangentStorage s : {TangentStorage::Deformation, TangentStorage::Displacement})
      {
         HyperelasticOperator op(p.fes, *p.ir, HyperelasticMaterial::StVenantKirchhoff,
                                 p.lambda, p.mu, s);
         Vector kw(n), e(n);
         kw.UseDevice(true);
         op.GetGradient(u).Mult(w, kw);
         add(kw, -1.0, kref, e);
         EXPECT_LT(Norm(e), 1e-13 * Norm(kref)) << "dim " << dim << " storage "
                                                << static_cast<int>(s);
         EXPECT_LT(op.MemoryBytes(), prev_bytes) << "dim " << dim;
         prev_bytes = op.MemoryBytes();
      }
      // Borrowed buffers: the same bits, and less memory owned.
      HyperelasticOperator lent(p.fes, *p.ir, HyperelasticMaterial::StVenantKirchhoff,
                                p.lambda, p.mu, TangentStorage::StressAndDeformation,
                                &scratch);
      Vector r1(n), r2(n), k2(n);
      for (Vector* v : {&r1, &r2, &k2}) { v->UseDevice(true); }
      p.op->Mult(u, r1);
      lent.Mult(u, r2);
      lent.GetGradient(u).Mult(w, k2);
      EXPECT_LT(lent.MemoryBytes(), p.op->MemoryBytes());
      r1.HostRead(); r2.HostRead(); k2.HostRead(); kref.HostRead();
      for (int i = 0; i < n; i++)
      {
         ASSERT_EQ(r1(i), r2(i));
         ASSERT_EQ(k2(i), kref(i));
      }
   }
}
