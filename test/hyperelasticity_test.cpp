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
// The static solver (solid/hyperelasticity.hpp), milestone M2:
//  H4 a homogeneous finite deformation (stretch and shear, det F != 1),
//     displacement on the faces through the origin and the exact traction P N
//     on the others, is reproduced to roundoff, Newton converging in a few
//     steps: the full load at once, and in three load steps with the line
//     search;
//  H5 a large rigid rotation (90 and 180 degrees, prescribed on the whole
//     boundary in steps of its angle) rotates the interior exactly with no
//     strain energy, where the linear model stores a lot;
//  H6 small loads give the linear solution: the difference to
//     LinearElasticity falls as the load squared;
//  H7 generalized-alpha (rho_inf 0.8) and BDF2 at second order in time on a
//     separable finite motion (2D and 3D, Q_2, whose space holds it exactly);
//  H8 a deck (equation: hyperelasticity) through ElasticityCase: a cantilever
//     bent to ~20 % of its length, whose tip on the mid-line moves back, where
//     the linear model's stays (by symmetry); the summary written. Dynamic: a
//     load applied suddenly at rest, the midpoint rule's energy balance, the
//     history CSV with the Newton iterations.
// Runs on whatever device the test main configures (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "quadrature/rule_book.hpp"
#include "solid/elasticity_case.hpp"
#include "solid/hyperelastic_operator.hpp"
#include "solid/hyperelasticity.hpp"
#include "solid/linear_elasticity.hpp"
#include "mfem.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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
      // The reference keeps F and S (the most memory), whatever the default.
      HyperelasticOperator ref(p.fes, *p.ir, HyperelasticMaterial::StVenantKirchhoff,
                               p.lambda, p.mu, TangentStorage::StressAndDeformation);
      Vector kref(n);
      kref.UseDevice(true);
      ref.GetGradient(u).Mult(w, kref);
      size_t prev_bytes = ref.MemoryBytes();
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
      ref.Mult(u, r1);
      lent.Mult(u, r2);
      lent.GetGradient(u).Mult(w, k2);
      EXPECT_LT(lent.MemoryBytes(), ref.MemoryBytes());
      r1.HostRead(); r2.HostRead(); k2.HostRead(); kref.HostRead();
      for (int i = 0; i < n; i++)
      {
         ASSERT_EQ(r1(i), r2(i));
         ASSERT_EQ(k2(i), kref(i));
      }
   }
}

namespace
{

/// An n^dim unit box (straight elements; attributes as MFEM's Cartesian
/// meshes: 2D bottom 1, right 2, top 3, left 4; 3D z = 0 1, y = 0 2, x = 1 3,
/// y = 1 4, x = 0 5, z = 1 6).
std::unique_ptr<ParMesh> UnitBox(int dim, int n)
{
   Mesh m = (dim == 3) ? Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)
            : Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, m);
}

/// max |a - b| over the true dofs (global).
double MaxDiff(const Vector& a, const Vector& b)
{
   const real_t* pa = a.HostRead();
   const real_t* pb = b.HostRead();
   double m = 0.0;
   for (int i = 0; i < a.Size(); ++i) { m = std::max(m, std::abs(pa[i] - pb[i])); }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

/// The true dofs of @p g's interpolant on @p fes.
Vector Interpolant(ParFiniteElementSpace& fes, VectorCoefficient& g)
{
   ParGridFunction x(&fes);
   x.ProjectCoefficient(g);
   Vector t(fes.GetTrueVSize());
   t.UseDevice(true);
   x.GetTrueDofs(t);
   return t;
}

Vector TrueDofs(const ParGridFunction& u)
{
   Vector t(u.ParFESpace()->GetTrueVSize());
   t.UseDevice(true);
   u.GetTrueDofs(t);
   return t;
}

/// The solver's residual history, for messages.
std::string History(const incns::Hyperelasticity& s)
{
   std::ostringstream o;
   o << "Newton residuals:";
   for (double r : s.ResidualHistory()) { o << " " << r; }
   return o.str();
}

incns::Hyperelasticity::Options SolverOptions(int order)
{
   incns::Hyperelasticity::Options o;
   o.order = order;
   o.lambda = kLambda;
   o.mu = kMu;
   o.newton_rtol = 1e-12;
   o.rtol = 1e-12;
   return o;
}

} // namespace

// H4
TEST(Hyperelasticity, H4_HomogeneousFiniteDeformationIsExact)
{
   for (int dim = 2; dim <= 3; ++dim)
   {
      auto mesh = UnitBox(dim, dim == 2 ? 3 : 2);
      DenseMatrix F(dim);
      if (dim == 2)
      {
         F(0, 0) = 1.2; F(0, 1) = 0.3;
         F(1, 0) = 0.1; F(1, 1) = 0.9;
      }
      else
      {
         F(0, 0) = 1.15; F(0, 1) = 0.2;   F(0, 2) = -0.1;
         F(1, 0) = 0.05; F(1, 1) = 0.9;   F(1, 2) = 0.1;
         F(2, 0) = 0.1;  F(2, 1) = -0.05; F(2, 2) = 1.1;
      }
      ASSERT_GT(std::abs(F.Det() - 1.0), 0.03);
      DenseMatrix E, S, P(dim);
      Strain(F, E);
      SecondPiola(E, kLambda, kMu, S);
      Mult(F, S, P);
      VectorFunctionCoefficient exact(dim, [F, dim](const Vector & X, Vector & u)
      {
         for (int i = 0; i < dim; i++)
         {
            u(i) = -X(i);
            for (int j = 0; j < dim; j++) { u(i) += F(i, j) * X(j); }
         }
      });
      // The faces through the origin prescribed; the others carry P N.
      const std::vector<int> held = (dim == 2) ? std::vector<int> {1, 4} :
                                    std::vector<int> {1, 2, 5};
      const std::vector<std::pair<int, int>> loaded = (dim == 2) ?
std::vector<std::pair<int, int>> {{2, 0}, {3, 1}} :
      std::vector<std::pair<int, int>> {{3, 0}, {4, 1}, {6, 2}};
      std::vector<std::unique_ptr<VectorConstantCoefficient>> t;
      for (const auto& l : loaded)
      {
         Vector pn(dim);
         P.GetColumn(l.second, pn); // P N, N the face's unit normal e_c
         t.push_back(std::make_unique<VectorConstantCoefficient>(pn));
      }
      // The full load at once; then three load steps with the line search.
      for (int steps : {1, 3})
      {
         incns::Hyperelasticity::Options o = SolverOptions(2);
         o.load_steps = steps;
         o.line_search = steps > 1;
         incns::Hyperelasticity solid(*mesh, o);
         for (int a : held) { solid.AddDisplacement(a, exact); }
         for (size_t i = 0; i < loaded.size(); i++)
         {
            solid.AddTraction(loaded[i].first, *t[i]);
         }
         ASSERT_TRUE(solid.Solve()) << "dim " << dim << " steps " << steps << " "
                                    << History(solid);
         const Vector ue = Interpolant(solid.Space(), exact);
         const Vector uh = TrueDofs(solid.Displacement());
         EXPECT_LT(MaxDiff(uh, ue), 1e-10) << "dim " << dim << " steps " << steps;
         EXPECT_LE(solid.MaxNewtonIterations(), 8) << "dim " << dim << " steps "
               << steps << " " << History(solid);
         EXPECT_EQ(solid.CorrectionHistory().size(),
                   static_cast<size_t>(solid.NewtonIterations()));
         if (Mpi::Root())
         {
            std::cout << "[H4] dim " << dim << " steps " << steps << " "
                      << History(solid) << std::endl;
         }
      }
   }
}

// H5
TEST(Hyperelasticity, H5_RigidRotationIsStrainFree)
{
   for (int dim = 2; dim <= 3; ++dim)
   {
      for (double degrees : {90.0, 180.0})
      {
         auto mesh = UnitBox(dim, 2);
         const double angle = degrees * M_PI / 180.0;
         // (Q(angle t) - I)(X - c), c the box's centre.
         VectorFunctionCoefficient g(dim, [dim, angle](const Vector & X, double t,
                                     Vector & u)
         {
            const DenseMatrix Q = Rotation(dim, angle * t);
            Vector Y(dim), QY(dim);
            for (int i = 0; i < dim; i++) { Y(i) = X(i) - 0.5; }
            Q.Mult(Y, QY);
            for (int i = 0; i < dim; i++) { u(i) = QY(i) - Y(i); }
         });
         incns::Hyperelasticity solid(*mesh, SolverOptions(2));
         for (int a = 1; a <= mesh->bdr_attributes.Max(); a++)
         {
            solid.AddDisplacement(a, g);
         }
         const int steps = 8;
         for (int k = 1; k <= steps; k++)
         {
            ASSERT_TRUE(solid.Solve(static_cast<double>(k) / steps))
                  << "dim " << dim << " degrees " << degrees << " step " << k;
         }
         g.SetTime(1.0);
         const Vector ue = Interpolant(solid.Space(), g);
         const Vector uh = TrueDofs(solid.Displacement());
         EXPECT_LT(MaxDiff(uh, ue), 1e-10) << "dim " << dim << " degrees " << degrees;
         EXPECT_LT(std::abs(solid.StrainEnergy()), 1e-16)
               << "dim " << dim << " degrees " << degrees;
         // The linear model on the same data stores spurious strain energy.
         incns::LinearElasticity::Options lo;
         lo.order = 2;
         lo.lambda = kLambda;
         lo.mu = kMu;
         lo.rtol = 1e-12;
         incns::LinearElasticity linear(*mesh, lo);
         for (int a = 1; a <= mesh->bdr_attributes.Max(); a++)
         {
            linear.AddDisplacement(a, g);
         }
         ASSERT_TRUE(linear.Solve(1.0));
         EXPECT_GT(linear.StrainEnergy(),
                   0.1) << "dim " << dim << " degrees " << degrees;
      }
   }
}

// H6
TEST(Hyperelasticity, H6_SmallLoadsGiveTheLinearSolution)
{
   // A 4 x 1 cantilever clamped on the left (attribute 4), under its weight.
   Mesh serial = Mesh::MakeCartesian2D(8, 2, Element::QUADRILATERAL, true, 4.0,
                                       1.0);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   serial.Clear();
   double diff[2], scale = 0.0;
   const double loads[2] = {1e-4, 5e-5};
   for (int i = 0; i < 2; i++)
   {
      Vector fv(2);
      fv(0) = 0.0;
      fv(1) = -loads[i];
      VectorConstantCoefficient f(fv);
      // Newton at its default tolerance: the residual's roundoff floor here is
      // ~3e-11 of the load, above SolverOptions' 1e-12.
      incns::Hyperelasticity::Options o = SolverOptions(3);
      o.newton_rtol = incns::Hyperelasticity::Options().newton_rtol;
      incns::Hyperelasticity solid(mesh, o);
      solid.AddFixed(4);
      solid.SetBodyForce(f);
      ASSERT_TRUE(solid.Solve()) << "load " << loads[i] << " " << History(solid);
      if (Mpi::Root())
      {
         std::cout << "[H6] load " << loads[i] << " " << History(solid) << std::endl;
      }
      incns::LinearElasticity::Options lo;
      lo.order = 3;
      lo.lambda = kLambda;
      lo.mu = kMu;
      lo.rtol = 1e-13;
      incns::LinearElasticity linear(mesh, lo);
      linear.AddFixed(4);
      linear.SetBodyForce(f);
      ASSERT_TRUE(linear.Solve());
      const Vector uh = TrueDofs(solid.Displacement());
      const Vector ul = TrueDofs(linear.Displacement());
      diff[i] = MaxDiff(uh, ul);
      if (i == 0) { scale = linear.MaxDisplacement(); }
   }
   if (Mpi::Root())
   {
      std::cout << "[H6] max |u - u_linear|: " << diff[0] << " " << diff[1]
                << ", ratio " << diff[0] / diff[1] << ", max |u_linear| " << scale
                << std::endl;
   }
   EXPECT_LT(diff[0], 1e-2 * scale);
   EXPECT_GT(diff[0] / diff[1], 3.5);
   EXPECT_LT(diff[0] / diff[1], 4.5);
}

namespace
{

/// P = F S(F), St. Venant-Kirchhoff at the tests' material.
DenseMatrix Piola(const DenseMatrix& F)
{
   DenseMatrix E, S, P(F.Height());
   Strain(F, E);
   SecondPiola(E, kLambda, kMu, S);
   Mult(F, S, P);
   return P;
}

/**
 * A separable finite motion u = eps s(X) T(t): s_i = x (a_i + b_i x + c_i y +
 * d_i z) (zero on x = 0; in Q_2), T = cos 4t + sin 3t / 2, eps = 0.1. The
 * displacement gradient reaches 0.14 and J stays >= 0.87; in 2D the acoustic
 * tensor's least eigenvalue stays >= 0.46 (mu 0.9), far from SVK's loss of
 * ellipticity. With the exact body force rho u_tt - Div P and traction P N,
 * the semi-discrete solution is u itself (P is cubic in X, which the Q_2
 * rules integrate exactly): every error is the time integrator's.
 */
struct FiniteSeparableMotion
{
   static constexpr double eps = 0.1;
   static constexpr double rho = 1.3;
   static constexpr double a[3] = {0.3, -0.4, 0.2};
   static constexpr double b[3] = {0.5, 0.1, -0.3}; ///< x coefficients.
   static constexpr double c[3] = {-0.2, 0.6, 0.1}; ///< y coefficients.
   static constexpr double d[3] = {0.1, -0.3, 0.4}; ///< z coefficients.

   static double T0(double t) { return std::cos(4 * t) + 0.5 * std::sin(3 * t); }
   static double T1(double t) { return -4 * std::sin(4 * t) + 1.5 * std::cos(3 * t); }
   static double T2(double t) { return -16 * std::cos(4 * t) - 4.5 * std::sin(3 * t); }

   static void S(int dim, const Vector& x, Vector& s)
   {
      s.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         double lin = a[i] + b[i] * x(0) + c[i] * x(1);
         if (dim == 3) { lin += d[i] * x(2); }
         s(i) = x(0) * lin;
      }
   }

   /// d s_i / d x_j.
   static void GradS(int dim, const Vector& x, DenseMatrix& g)
   {
      g.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         g(i, 0) = a[i] + 2 * b[i] * x(0) + c[i] * x(1) + (dim == 3 ? d[i] * x(2) : 0.0);
         g(i, 1) = c[i] * x(0);
         if (dim == 3) { g(i, 2) = d[i] * x(0); }
      }
   }

   /// d^2 s_i / d x_j d x_k.
   static double H(int i, int j, int k)
   {
      if (j > k) { std::swap(j, k); }
      if (j != 0) { return 0.0; }
      return k == 0 ? 2 * b[i] : (k == 1 ? c[i] : d[i]);
   }

   /// F = I + eps T(t) grad s.
   static DenseMatrix Deformation(int dim, const Vector& x, double t)
   {
      DenseMatrix F;
      GradS(dim, x, F);
      F *= eps * T0(t);
      for (int i = 0; i < dim; ++i) { F(i, i) += 1.0; }
      return F;
   }

   /// The body force rho u_tt - Div P(F): Div P_i = sum_J dP(F; d_J F)_iJ.
   static void Force(int dim, const Vector& x, double t, Vector& f)
   {
      const DenseMatrix F = Deformation(dim, x, t);
      DenseMatrix E, Sm, dF(dim), FtdF(dim), dE(dim), dS, dP(dim), FdS(dim);
      Strain(F, E);
      SecondPiola(E, kLambda, kMu, Sm);
      S(dim, x, f);
      f *= rho * eps * T2(t);
      for (int J = 0; J < dim; ++J)
      {
         for (int i = 0; i < dim; ++i)
         {
            for (int K = 0; K < dim; ++K) { dF(i, K) = eps * T0(t) * H(i, J, K); }
         }
         // dP = dF S + F dS, dS = S(dE) (S is linear in E), dE = sym(F^T dF).
         MultAtB(F, dF, FtdF);
         for (int i = 0; i < dim; ++i)
         {
            for (int k = 0; k < dim; ++k) { dE(i, k) = 0.5 * (FtdF(i, k) + FtdF(k, i)); }
         }
         SecondPiola(dE, kLambda, kMu, dS);
         Mult(dF, Sm, dP);
         Mult(F, dS, FdS);
         dP += FdS;
         for (int i = 0; i < dim; ++i) { f(i) -= dP(i, J); }
      }
   }
};

/// The traction P(F(X, t)) N of FiniteSeparableMotion (reference normal N).
class PiolaTraction : public VectorCoefficient
{
public:
   explicit PiolaTraction(int dim) : VectorCoefficient(dim) {}

   using VectorCoefficient::Eval;
   void Eval(Vector& V, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      Vector x(vdim), n(vdim);
      T.SetIntPoint(&ip);
      T.Transform(ip, x);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();
      const DenseMatrix P = Piola(FiniteSeparableMotion::Deformation(vdim, x,
                                  GetTime()));
      V.SetSize(vdim);
      P.Mult(n, V);
   }
};

} // namespace

// H7
TEST(Hyperelasticity, H7_TimeSchemesAreSecondOrder)
{
   using M = FiniteSeparableMotion;
   const std::vector<int> steps = {20, 40, 80};
   for (int dim = 2; dim <= 3; ++dim)
   {
      for (incns::SolidTimeScheme scheme :
           {incns::SolidTimeScheme::GeneralizedAlpha, incns::SolidTimeScheme::Bdf})
      {
         const bool bdf = scheme == incns::SolidTimeScheme::Bdf;
         VectorFunctionCoefficient u(dim, [dim](const Vector & x, double t, Vector & v)
         {
            M::S(dim, x, v);
            v *= M::eps * M::T0(t);
         });
         VectorFunctionCoefficient v(dim, [dim](const Vector & x, double t, Vector & w)
         {
            M::S(dim, x, w);
            w *= M::eps * M::T1(t);
         });
         VectorFunctionCoefficient f(dim, [dim](const Vector & x, double t, Vector & w)
         {
            M::Force(dim, x, t, w);
         });
         PiolaTraction traction(dim);
         std::vector<double> eu, ev;
         std::vector<int> newton;
         for (int n : steps)
         {
            auto mesh = UnitBox(dim, 2);
            incns::Hyperelasticity::Options o = SolverOptions(2);
            o.density = M::rho;
            o.scheme = scheme;
            incns::Hyperelasticity solid(*mesh, o);
            // x = 0 held (s vanishes there): 2D attribute 4, 3D 5.
            const int held = dim == 2 ? 4 : 5;
            solid.AddFixed(held);
            for (int a = 1; a <= mesh->bdr_attributes.Max(); ++a)
            {
               if (a != held) { solid.AddTraction(a, traction); }
            }
            solid.SetBodyForce(f);
            solid.Initialize(0.0, &u, &v);
            int most = 0;
            for (int k = 0; k < n; ++k)
            {
               ASSERT_TRUE(solid.Step(1.0 / n)) << "dim " << dim << " bdf " << bdf
                                                << " n " << n << " step " << k << " "
                                                << History(solid);
               most = std::max(most, solid.NewtonIterations());
            }
            ASSERT_NEAR(solid.Time(), 1.0, 1e-12);
            u.SetTime(1.0);
            v.SetTime(1.0);
            eu.push_back(solid.Displacement().ComputeL2Error(u));
            ev.push_back(solid.Velocity().ComputeL2Error(v));
            newton.push_back(most);
         }
         if (Mpi::Root())
         {
            std::cout << "[H7] dim " << dim << (bdf ? " bdf2" : " generalized-alpha")
                      << ": u errors";
            for (double e : eu) { std::cout << " " << e; }
            std::cout << ", v errors";
            for (double e : ev) { std::cout << " " << e; }
            std::cout << ", most Newton per step";
            for (int m : newton) { std::cout << " " << m; }
            std::cout << std::endl;
         }
         for (std::size_t i = 0; i + 1 < steps.size(); ++i)
         {
            EXPECT_GT(std::log2(eu[i] / eu[i + 1]), 1.8) << "dim " << dim << " bdf "
               << bdf << " level " << i;
            EXPECT_GT(std::log2(ev[i] / ev[i + 1]), 1.8) << "dim " << dim << " bdf "
                  << bdf << " level " << i;
         }
      }
   }
}

// H8
TEST(Hyperelasticity, H8_DeckThroughElasticityCase)
{
   const std::string deck =
      "equation: hyperelasticity\n"
      "mesh:\n"
      "  dim: 2\n"
      "  elements: [8, 2]\n"
      "  lengths: [4.0, 1.0]\n"
      "  periodic: [false, false]\n"
      "solid:\n"
      "  model: st_venant_kirchhoff\n"
      "  youngs_modulus: 100.0\n"
      "  poisson_ratio: 0.25\n"
      "  order: 2\n"
      "  body_force: [0.0, -0.2]\n"
      "  load_steps: 2\n"
      "  rtol: 1.0e-12\n"
      "  probe: [4.0, 0.5]\n"
      "boundary_conditions:\n"
      "  - {select: [xmin], type: fixed}\n"
      "  - {select: [xmax], type: traction, value: [0.0, -0.05]}\n"
      "  - {select: [ymin, ymax], type: free}\n";
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   EXPECT_EQ(p.equation, incns::Equation::Hyperelasticity);
   EXPECT_EQ(p.solid.model, incns::SolidModel::StVenantKirchhoff);
   EXPECT_EQ(p.solid.load_steps, 2);
   const incns::Hyperelasticity::Options o = incns::HyperelasticOptions(p);
   EXPECT_EQ(o.order, 2);
   EXPECT_EQ(o.load_steps, 2);
   EXPECT_NEAR(o.mu, 100.0 / 2.5, 1e-12);
   EXPECT_NEAR(o.lambda, 100.0 * 0.25 / (1.25 * 0.5), 1e-12); // plane strain

   auto mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   ASSERT_TRUE(c.Solve()) << History(c.HyperelasticSolver());
   Vector x(2), uh, ul;
   x(0) = 4.0;
   x(1) = 0.5;
   ASSERT_TRUE(c.HyperelasticSolver().DisplacementAt(x, uh));
   EXPECT_LE(c.HyperelasticSolver().MaxNewtonIterations(), 8)
         << History(c.HyperelasticSolver());

   // The same deck, linear: its tip on the mid-line does not move along the
   // beam (the problem is symmetric about it up to the loads' sign).
   std::string linear_deck = deck;
   linear_deck.replace(linear_deck.find("hyperelasticity"),
                       std::string("hyperelasticity").size(), "linear_elasticity");
   const incns::Parameters pl = incns::Parameters::LoadYAMLString(linear_deck);
   auto mesh_l = incns::MakeCaseMesh(pl);
   incns::ElasticityCase cl(*mesh_l, pl);
   ASSERT_TRUE(cl.Solve());
   ASSERT_TRUE(cl.Solver().DisplacementAt(x, ul));
   if (Mpi::Root())
   {
      std::cout << "[H8] tip u: hyperelastic (" << uh(0) << ", " << uh(1)
                << "), linear (" << ul(0) << ", " << ul(1) << "); "
                << History(c.HyperelasticSolver()) << std::endl;
   }
   EXPECT_LT(ul(1), -0.15 * 4.0); // bent by more than 15 % of its length
   EXPECT_LT(uh(1), 0.0);
   EXPECT_LT(std::abs(ul(0)), 1e-6 * std::abs(ul(1)));
   EXPECT_LT(uh(0), -0.01 * std::abs(uh(1))); // the length is kept: it moves back
   EXPECT_LT(std::abs(uh(1)), std::abs(ul(1)));

   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   const std::string path = "hyperelasticity_H8_np" + std::to_string(np) +
                            "_summary.json";
   c.WriteSummary(path, "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::ifstream f(path);
      ASSERT_TRUE(f.good());
      std::stringstream ss;
      ss << f.rdbuf();
      for (const char* key :
           {"\"equation\": \"hyperelasticity\"",
            "\"model\": \"st_venant_kirchhoff\"", "\"newton\"",
            "\"max_iterations_per_load_step\"", "\"strain_energy\"",
            "\"found\": true"
           })
      {
         EXPECT_NE(ss.str().find(key), std::string::npos) << key << "\n" << ss.str();
      }
      std::filesystem::remove(path);
   }
}

// H8, dynamic
TEST(Hyperelasticity, H8_DynamicDeckThroughElasticityCase)
{
   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   const std::string dir = "hyperelasticity_test_out_np" + std::to_string(np);
   std::ostringstream deck;
   deck << "equation: hyperelasticity\n"
        << "mesh:\n  dim: 2\n  elements: [6, 2]\n  lengths: [3.0, 0.5]\n"
        << "  periodic: [false, false]\n"
        << "solid:\n  youngs_modulus: 100.0\n  poisson_ratio: 0.3\n"
        << "  order: 3\n  rtol: 1.0e-13\n"
        << "  analysis: dynamic\n  density: 2.0\n  rho_inf: 1.0\n"
        << "  probe: [3.0, 0.25]\n"
        << "time:\n  dt: 0.02\n  t_final: 0.25\n"
        << "boundary_conditions:\n"
        << "  - {select: [xmin], type: fixed}\n"
        << "  - {select: [xmax], type: traction, value: [0.0, -0.5]}\n"
        << "  - {select: [ymin, ymax], type: free}\n"
        << "output:\n  path: " << dir << "\n  name: beam\n  history: true\n";
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck.str());
   ASSERT_EQ(p.solid.analysis, incns::SolidAnalysis::Dynamic);
   std::unique_ptr<ParMesh> mesh = incns::MakeCaseMesh(p);
   incns::ElasticityCase c(*mesh, p);
   ASSERT_TRUE(c.Run());
   incns::Hyperelasticity& s = c.HyperelasticSolver();
   // 0.25 / 0.02: 12 steps of 0.02 and one of 0.01.
   EXPECT_EQ(s.StepCount(), 13);
   EXPECT_NEAR(s.Time(), 0.25, 1e-14);
   // At rho_inf 1 (the midpoint rule) a constant load from rest keeps E(t) =
   // f . u(t) up to the energy's cubic and quartic terms: SVK is not
   // quadratic, so not to the solver tolerance as in the linear D5.
   const double e = s.KineticEnergy() + s.StrainEnergy();
   const double w = s.ExternalWork();
   EXPECT_GT(w, 0.0);
   EXPECT_NEAR(e, w, 1e-3 * w);
   Vector x(2), u;
   x(0) = 3.0;
   x(1) = 0.25;
   ASSERT_TRUE(s.DisplacementAt(x, u));
   EXPECT_LT(u(1), 0.0); // the tip moves with the load
   if (Mpi::Root())
   {
      std::cout << "[H8] dynamic: E - f.u = " << e - w << " of f.u " << w
                << ", tip u_y " << u(1) << std::endl;
   }

   const std::string summary = dir + "/beam_summary.json";
   c.WriteSummary(summary, "ok");
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
   {
      std::ifstream f(summary);
      std::stringstream ss;
      ss << f.rdbuf();
      const std::string j = ss.str();
      for (const char* key :
           {"\"dynamics\"", "\"generalized_alpha\"", "\"total_energy\"",
            "\"steps\": 13", "\"newton_iterations_total\"", "\"newton_iterations_max\""
           })
      {
         EXPECT_NE(j.find(key), std::string::npos) << key << " in\n" << j;
      }
      std::ifstream h(dir + "/beam_history.csv");
      std::string line;
      int rows = 0;
      std::getline(h, line);
      EXPECT_EQ(line.rfind("step,t,dt,cg_iterations,newton_iterations,"
                           "kinetic_energy", 0), 0u) << line;
      while (std::getline(h, line)) { ++rows; }
      EXPECT_EQ(rows, 14); // the initial state and 13 steps
      std::filesystem::remove_all(dir);
   }
}
