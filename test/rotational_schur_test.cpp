// precond/rotational_schur -- docs/design/rotational_schur_velocity_mg_spec.md Part C,
// tests S1 and S2 (S3, the iteration study, runs through StokesSolver in
// rotational_schur_solver_test.cpp). Every test runs in 2D and 3D (the spec
// is 3D only; bessemer's tensor mode also covers 2D).
//  S1 the tensor operator: partial assembly (device Project) vs legacy
//     assembly (host Eval) with the same coefficient -- this also catches a
//     mishandled transpose flag; null spaces; nonsymmetry; Fill inverts
//     sigma I + [o]x; omega = 0 gives L_p / sigma; re-assembly after changing
//     sigma and w* in place acts like a freshly built form.
//  S2 modes and plumbing: the Auto switch with hysteresis on both criteria;
//     CahouetChabard mode is the CC solver bitwise; Tensor mode with omega = 0
//     and an exact Laplacian reproduces CC (the sigma-scaling contract);
//     remove_mean; SetOperator is never forwarded to the Laplacian solver.
// Stand-ins for bessemer's solvers (spec 6): CG to 1e-14 with Jacobi on the
// partially assembled pressure Laplacian and mass. Mesh (spec App. B): curved
// 3^dim box, mesh order 4, velocity Q4, pressure Q3, pressure Dirichlet on
// boundary attribute 1; sigma = 1.5, alpha = 1, vel_fn with amplitude 5.

#include <gtest/gtest.h>

#include "precond/rotational_schur.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <string>

using namespace mfem;
using incns::RotatingDarcyTensor;
using incns::RotationalSchurPreconditioner;
using incns::RotationNumberStats;

namespace
{

double g_amp = 5.0; // amplitude of vel_fn

// Spec App. B (3D); a 2D field of the same kind. Every vorticity component
// is nonzero and varies in space.
void vel_fn(const Vector& x, Vector& v)
{
   if (x.Size() == 3)
   {
      v(0) = g_amp * (std::sin(2 * x(1)) + 0.3 * std::cos(3 * x(2)));
      v(1) = g_amp * (std::sin(2 * x(2)) + 0.5 * x(0) * x(0));
      v(2) = g_amp * (std::sin(2 * x(0)) + 0.2 * x(1));
   }
   else
   {
      v(0) = g_amp * (std::sin(2 * x(1)) + 0.3 * std::cos(3 * x(0)));
      v(1) = g_amp * (std::sin(2 * x(0)) + 0.5 * x(0) * x(1));
   }
}

// Curved 3^dim mesh (spec App. B): nodes of order 4 stored byNODES, then
// x += 0.04 sin(pi y), y += 0.04 sin(pi x).
Mesh CurvedMesh(int dim)
{
   Mesh m = (dim == 3) ? Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON,
            1.0, 1.0, 1.0)
            : Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL, true, 1.0,
                                    1.0);
   m.SetCurvature(4, false, -1, Ordering::byNODES);
   m.Transform([](const Vector & x, Vector & y)
   {
      y = x;
      y(0) += 0.04 * std::sin(M_PI * x(1));
      y(1) += 0.04 * std::sin(M_PI * x(0));
   });
   return m;
}

// Interpolate through the true dofs so shared dofs agree across ranks.
void SetField(ParGridFunction& w)
{
   VectorFunctionCoefficient c(w.ParFESpace()->GetVDim(), vel_fn);
   w.ProjectCoefficient(c);
   Vector t;
   w.GetTrueDofs(t);
   w.SetFromTrueDofs(t);
}

const Geometry::Type kGeom2 = Geometry::SQUARE;
const Geometry::Type kGeom3 = Geometry::CUBE;

// The rule of L_T and the stand-in forms: Gauss-Legendre of order 2 k_p + 2
// from IntRules (global, so it outlives every interpolator cache).
const IntegrationRule& PressureRule(int dim, int kp)
{
   return IntRules.Get(dim == 3 ? kGeom3 : kGeom2, 2 * kp + 2);
}

struct Problem
{
   explicit Problem(int dim, bool dirichlet = true)
      : serial(CurvedMesh(dim)), mesh(MPI_COMM_WORLD, serial),
        vfec(4, dim), pfec(3, dim), vfes(&mesh, &vfec, dim, Ordering::byNODES),
        pfes(&mesh, &pfec), w(&vfes), ir(PressureRule(dim, 3))
   {
      SetField(w);
      if (dirichlet)
      {
         Array<int> bdr(mesh.bdr_attributes.Max());
         bdr = 0;
         bdr[0] = 1;
         pfes.GetEssentialTrueDofs(bdr, ess);
      }
   }
   Mesh serial;
   ParMesh mesh;
   H1_FECollection vfec, pfec;
   ParFiniteElementSpace vfes, pfes;
   ParGridFunction w;
   Array<int> ess;
   const IntegrationRule& ir;
};

// Partially assembled form with one integrator; true-dof operator.
struct PaForm
{
   PaForm(ParFiniteElementSpace& fes, BilinearFormIntegrator* integ,
          const Array<int>& ess)
      : form(&fes)
   {
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form.AddDomainIntegrator(integ);
      form.Assemble();
      form.FormSystemMatrix(ess, op);
   }
   ParBilinearForm form;
   OperatorHandle op;
};

// CG to 1e-14 with Jacobi on a partially assembled form (spec 6 stand-in);
// `iters` > 0 runs that fixed count instead.
// It owns its essential list: MFEM's Jacobi smoother and constrained operators
// keep pointers to the list they were given.
struct CgStandIn
{
   CgStandIn(ParFiniteElementSpace& fes, BilinearFormIntegrator* integ,
             const Array<int>& ess_in, int iters = 0)
      : ess(ess_in), pa(fes, integ, ess), diag(fes.GetTrueVSize()),
        cg(fes.GetComm())
   {
      pa.form.AssembleDiagonal(diag);
      jacobi = std::make_unique<OperatorJacobiSmoother>(diag, ess);
      cg.SetOperator(*pa.op);
      cg.SetPreconditioner(*jacobi);
      cg.SetRelTol(iters > 0 ? 0.0 : 1e-14);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(iters > 0 ? iters : 5000);
      cg.SetPrintLevel(0);
      cg.iterative_mode = false;
   }
   Array<int> ess;
   PaForm pa;
   Vector diag;
   std::unique_ptr<OperatorJacobiSmoother> jacobi;
   CGSolver cg;
};

// sigma * lap(r) + nu * mass(r): the CC stand-in.
class CcStandIn : public Solver
{
public:
   CcStandIn(const Solver& lap, const Solver& mass, double sigma, double nu)
      : Solver(lap.Height()), lap_(lap), mass_(mass), sigma_(sigma), nu_(nu),
        t_(lap.Height()) { t_.UseDevice(true); }
   void Mult(const Vector& r, Vector& z) const override
   {
      lap_.Mult(r, z);
      z *= sigma_;
      mass_.Mult(r, t_);
      z.Add(nu_, t_);
   }
   void SetOperator(const Operator&) override {}
private:
   const Solver& lap_;
   const Solver& mass_;
   double sigma_, nu_;
   mutable Vector t_;
};

// A Laplacian solver whose SetOperator must never be called.
class NoSetOperator : public Solver
{
public:
   explicit NoSetOperator(const Solver& s) : Solver(s.Height()), s_(s) {}
   void Mult(const Vector& x, Vector& y) const override { s_.Mult(x, y); }
   void SetOperator(const Operator&) override
   {
      MFEM_ABORT("rotational_schur_test: SetOperator was forwarded");
   }
private:
   const Solver& s_;
};

double GlobalNormInf(const Vector& v)
{
   double n = v.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return n;
}

double RelDiff(const Vector& a, const Vector& b)
{
   Vector d(a);
   d -= b;
   return GlobalNormInf(d) / GlobalNormInf(b);
}

Vector RandomTrue(ParFiniteElementSpace& fes, int seed)
{
   // Same global vector on any partition: built from the dof coordinates.
   ParGridFunction g(&fes);
   FunctionCoefficient c([seed](const Vector & x)
   {
      double h = 12.9898 * x(0) + 78.233 * x(1) + 7.0 * seed;
      if (x.Size() == 3) { h += 37.719 * x(2); }
      const double s = std::sin(h) * 43758.5453;
      return s - std::floor(s) - 0.5;
   });
   g.ProjectCoefficient(c);
   Vector t(fes.GetTrueVSize());
   g.GetTrueDofs(t);
   return t;
}

// The pressure Laplacian at the test's pressure rule (the rule of L_T), so
// that omega = 0 makes L_T exactly this operator over sigma.
DiffusionIntegrator* Laplacian(const IntegrationRule& ir)
{
   auto* d = new DiffusionIntegrator;
   d->SetIntRule(&ir);
   return d;
}

const double kSigma = 1.5;
const double kAlpha = 1.0;

} // namespace

// S1 -- the tensor operator L_T = -div(T grad).
TEST(RotationalSchur, S1_TensorOperator)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Problem P(dim);
      RotatingDarcyTensor T(P.w, kSigma, kAlpha);

      // Partial assembly (device Project, transposed by MFEM) vs legacy
      // assembly (host Eval), same coefficient and rule.
      auto* di = new DiffusionIntegrator(T);
      di->SetIntRule(&P.ir);
      Array<int> none;
      PaForm pa(P.pfes, di, none);
      ParBilinearForm legacy(&P.pfes);
      auto* dl = new DiffusionIntegrator(T);
      dl->SetIntRule(&P.ir);
      legacy.AddDomainIntegrator(dl);
      legacy.Assemble();
      legacy.Finalize();
      std::unique_ptr<HypreParMatrix> L(legacy.ParallelAssemble());
      const Vector x = RandomTrue(P.pfes, 1), v = RandomTrue(P.pfes, 2);
      Vector y_pa(x.Size()), y_lg(x.Size());
      pa.op->Mult(x, y_pa);
      L->Mult(x, y_lg);
      EXPECT_LE(RelDiff(y_pa, y_lg), 1e-12) << "PA vs legacy";

      // Null spaces: L_T 1 = 0 and 1^T L_T x = 0 (grad 1 = 0 on both sides).
      Vector one(x.Size()), y1(x.Size());
      one = 1.0;
      pa.op->Mult(one, y1);
      const double nLx = std::sqrt(InnerProduct(MPI_COMM_WORLD, y_pa, y_pa));
      EXPECT_LE(std::sqrt(InnerProduct(MPI_COMM_WORLD, y1, y1)) / nLx, 1e-13);
      const double n1 = std::sqrt(InnerProduct(MPI_COMM_WORLD, one, one));
      EXPECT_LE(std::abs(InnerProduct(MPI_COMM_WORLD, one, y_pa)) / (n1 * nLx),
                1e-13);

      // Nonsymmetry is present (the skew part of T).
      Vector Lv(x.Size());
      pa.op->Mult(v, Lv);
      const double xLv = InnerProduct(MPI_COMM_WORLD, x, Lv);
      const double vLx = InnerProduct(MPI_COMM_WORLD, v, y_pa);
      EXPECT_GT(std::abs(xLv - vLx) / std::abs(xLv), 1e-2);

      // omega = 0: L_T = L_p / sigma.
      ParGridFunction zero(&P.vfes);
      zero = 0.0;
      RotatingDarcyTensor T0(zero, kSigma, kAlpha);
      auto* d0 = new DiffusionIntegrator(T0);
      d0->SetIntRule(&P.ir);
      PaForm pa0(P.pfes, d0, none);
      auto* dp = new DiffusionIntegrator;
      dp->SetIntRule(&P.ir);
      PaForm lap(P.pfes, dp, none);
      Vector y0(x.Size()), yl(x.Size());
      pa0.op->Mult(x, y0);
      lap.op->Mult(x, yl);
      yl /= kSigma;
      EXPECT_LE(RelDiff(y0, yl), 1e-13) << "omega = 0";
   }
}

// S1 (cont.) -- Fill inverts sigma I + [o]x, for random sigma and o.
TEST(RotationalSchur, S1_FillIsTheInverse)
{
   std::mt19937 gen(42);
   std::uniform_real_distribution<double> U(-3.0, 3.0), S(0.1, 5.0);
   double err3 = 0.0, err2 = 0.0;
   for (int trial = 0; trial < 100; ++trial)
   {
      const double s = S(gen), o0 = U(gen), o1 = U(gen), o2 = U(gen);
      real_t T[9];
      RotatingDarcyTensor::Fill3(s, o0, o1, o2, T);
      // A = s I + [o]x, column-major like T.
      const double A[9] = {s, o2, -o1, -o2, s, o0, o1, -o0, s};
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c)
         {
            double prod = 0.0;
            for (int k = 0; k < 3; ++k) { prod += T[r + 3 * k] * A[k + 3 * c]; }
            err3 = std::max(err3, std::abs(prod - (r == c ? 1.0 : 0.0)));
         }
      real_t T2[4];
      RotatingDarcyTensor::Fill2(s, o2, T2);
      const double A2[4] = {s, o2, -o2, s}; // [[s, -o], [o, s]] column-major
      for (int r = 0; r < 2; ++r)
         for (int c = 0; c < 2; ++c)
         {
            double prod = 0.0;
            for (int k = 0; k < 2; ++k) { prod += T2[r + 2 * k] * A2[k + 2 * c]; }
            err2 = std::max(err2, std::abs(prod - (r == c ? 1.0 : 0.0)));
         }
   }
   EXPECT_LE(err3, 1e-14);
   EXPECT_LE(err2, 1e-14);
}

// S1 (cont.) -- update semantics: in Tensor mode, Update(sigma1), change w*
// in place, Update(sigma2); the operator (sigma L_T) acts like a freshly
// built and constrained form with sigma2 and the new w*.
TEST(RotationalSchur, S1_UpdateReassemblesInPlace)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Problem P(dim);
      CgStandIn lap(P.pfes, Laplacian(P.ir), P.ess);
      CgStandIn mass(P.pfes, new MassIntegrator, Array<int>());
      CcStandIn cc(lap.cg, mass.cg, kSigma, 0.1);
      RotationalSchurPreconditioner::Options opt;
      opt.mode = RotationalSchurPreconditioner::Mode::Tensor;
      RotationalSchurPreconditioner S(P.pfes, P.ess, P.ir, P.w, kAlpha, 0.1,
                                      cc, mass.cg, lap.cg, opt);
      RotationNumberStats st;
      S.Update(kSigma, st);
      g_amp = 8.0;
      SetField(P.w); // in place
      g_amp = 5.0;
      S.Update(2.0 * kSigma, st);

      RotatingDarcyTensor T(P.w, 2.0 * kSigma, kAlpha);
      T.SetScale(2.0 * kSigma); // the preconditioner assembles sigma L_T
      auto* di = new DiffusionIntegrator(T);
      di->SetIntRule(&P.ir);
      PaForm fresh(P.pfes, di, P.ess);
      const Vector x = RandomTrue(P.pfes, 3);
      Vector y1(x.Size()), y2(x.Size());
      ASSERT_NE(S.GetTensorOperator(), nullptr);
      S.GetTensorOperator()->Mult(x, y1);
      fresh.op->Mult(x, y2);
      EXPECT_LE(RelDiff(y1, y2), 1e-14);
   }
}

// S2 -- the Auto switch with hysteresis (defaults: on above 20, off below 10;
// volume fraction on above 1e-3, off below 2.5e-4).
TEST(RotationalSchur, S2_AutoSwitch)
{
   Problem P(2);
   CgStandIn lap(P.pfes, Laplacian(P.ir), P.ess);
   CgStandIn mass(P.pfes, new MassIntegrator, Array<int>());
   CcStandIn cc(lap.cg, mass.cg, kSigma, 0.1);
   const bool expect[6] = {false, false, true, true, false, true};

   RotationalSchurPreconditioner::Options opt; // MaxMu
   opt.mode = RotationalSchurPreconditioner::Mode::Auto;
   RotationalSchurPreconditioner S(P.pfes, P.ess, P.ir, P.w, kAlpha, 0.1, cc,
                                   mass.cg, lap.cg, opt);
   const double mu[6] = {5, 15, 25, 15, 8, 25};
   for (int k = 0; k < 6; ++k)
   {
      RotationNumberStats st;
      st.max_mu = mu[k];
      S.Update(kSigma, st);
      EXPECT_EQ(S.TensorActive(), expect[k]) << "MaxMu step " << k;
   }
   EXPECT_EQ(S.NumSwitches(), 3);

   opt.criterion = RotationalSchurPreconditioner::Criterion::VolumeFraction;
   RotationalSchurPreconditioner V(P.pfes, P.ess, P.ir, P.w, kAlpha, 0.1, cc,
                                   mass.cg, lap.cg, opt);
   const double vf[6] = {1e-4, 5e-4, 2e-3, 5e-4, 1e-4, 2e-3};
   for (int k = 0; k < 6; ++k)
   {
      RotationNumberStats st;
      st.vol_fraction = vf[k];
      st.threshold = opt.mu_on;
      V.Update(kSigma, st);
      EXPECT_EQ(V.TensorActive(), expect[k]) << "VolumeFraction step " << k;
   }
   EXPECT_EQ(V.NumSwitches(), 3);
}

// S2 (cont.) -- CahouetChabard mode is the CC solver bitwise, whatever mu;
// Tensor mode with omega = 0 and an exact Laplacian reproduces it (the
// sigma-scaling contract of the tensor path).
TEST(RotationalSchur, S2_ModesAndScaling)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Problem P(dim);
      CgStandIn lap(P.pfes, Laplacian(P.ir), P.ess);
      CgStandIn mass(P.pfes, new MassIntegrator, Array<int>());
      const double nu = 0.1;
      CcStandIn cc(lap.cg, mass.cg, kSigma, nu);
      const Vector r = RandomTrue(P.pfes, 4);
      Vector z_cc(r.Size()), z(r.Size());
      cc.Mult(r, z_cc);

      RotationalSchurPreconditioner::Options opt;
      opt.mode = RotationalSchurPreconditioner::Mode::CahouetChabard;
      RotationalSchurPreconditioner C(P.pfes, P.ess, P.ir, P.w, kAlpha, nu, cc,
                                      mass.cg, lap.cg, opt);
      RotationNumberStats huge;
      huge.max_mu = 1e6;
      C.Update(kSigma, huge);
      EXPECT_FALSE(C.TensorActive());
      C.Mult(r, z);
      Vector d(z);
      d -= z_cc;
      EXPECT_EQ(GlobalNormInf(d), 0.0) << "CC mode must be bitwise CC";

      ParGridFunction zero(&P.vfes);
      zero = 0.0;
      opt.mode = RotationalSchurPreconditioner::Mode::Tensor;
      RotationalSchurPreconditioner T(P.pfes, P.ess, P.ir, zero, kAlpha, nu,
                                      cc, mass.cg, lap.cg, opt);
      T.Update(kSigma, huge);
      T.Mult(r, z);
      EXPECT_LE(RelDiff(z, z_cc), 1e-11) << "tensor(omega = 0) vs CC";
   }
}

// S2 (cont.) -- remove_mean on a pure-Neumann pressure space: the tensor
// part of the output has zero mean (nu = 0 isolates it).
TEST(RotationalSchur, S2_RemoveMean)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Problem P(dim, /*dirichlet=*/false);
      CgStandIn lap(P.pfes, Laplacian(P.ir), P.ess, 5);
      CgStandIn mass(P.pfes, new MassIntegrator, P.ess);
      CcStandIn cc(lap.cg, mass.cg, kSigma, 0.0);
      RotationalSchurPreconditioner::Options opt;
      opt.mode = RotationalSchurPreconditioner::Mode::Tensor;
      opt.remove_mean = true;
      RotationalSchurPreconditioner T(P.pfes, P.ess, P.ir, P.w, kAlpha, 0.0,
                                      cc, mass.cg, lap.cg, opt);
      T.Update(kSigma, RotationNumberStats());
      const Vector r = RandomTrue(P.pfes, 5);
      Vector z(r.Size());
      T.Mult(r, z);
      double loc[2] = {z.Sum(), static_cast<double>(z.Size())};
      MPI_Allreduce(MPI_IN_PLACE, loc, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      EXPECT_LE(std::abs(loc[0] / loc[1]), 1e-14 * GlobalNormInf(z));
   }
}

// S2 (cont.) -- SetOperator is never forwarded to the Laplacian solver (MFEM's
// Krylov solvers forward SetOperator to their preconditioner).
TEST(RotationalSchur, S2_SetOperatorNotForwarded)
{
   Problem P(3);
   CgStandIn lap(P.pfes, Laplacian(P.ir), P.ess);
   CgStandIn mass(P.pfes, new MassIntegrator, Array<int>());
   CcStandIn cc(lap.cg, mass.cg, kSigma, 0.1);
   NoSetOperator guarded(lap.cg);
   RotationalSchurPreconditioner::Options opt;
   opt.mode = RotationalSchurPreconditioner::Mode::Tensor;
   RotationalSchurPreconditioner T(P.pfes, P.ess, P.ir, P.w, kAlpha, 0.1, cc,
                                   mass.cg, guarded, opt);
   T.Update(kSigma, RotationNumberStats());
   T.Update(2.0 * kSigma, RotationNumberStats());
   const Vector r = RandomTrue(P.pfes, 6);
   Vector z(r.Size());
   T.Mult(r, z);
   EXPECT_GT(GlobalNormInf(z), 0.0);
}

// RotationNumber -- max mu and the volume fraction against a host reference
// (MFEM's GridFunction::GetCurl at the same points).
TEST(RotationalSchur, RotationNumberStats)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE("dim=" + std::to_string(dim));
      Problem P(dim);
      const IntegrationRule& ir = IntRules.Get(dim == 3 ? kGeom3 : kGeom2, 8);
      incns::RotationNumber rn(P.vfes, ir);
      const double sigma = 2.0, alpha = 0.5;
      // Host reference.
      double mx = 0.0, total = 0.0;
      for (int e = 0; e < P.mesh.GetNE(); ++e)
      {
         ElementTransformation& T = *P.mesh.GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            T.SetIntPoint(&ir.IntPoint(q));
            Vector o;
            P.w.GetCurl(T, o);
            mx = std::max(mx, std::abs(alpha) * o.Norml2() / sigma);
            total += ir.IntPoint(q).weight * T.Weight();
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      const double thr = 0.5 * mx;
      double above = 0.0;
      for (int e = 0; e < P.mesh.GetNE(); ++e)
      {
         ElementTransformation& T = *P.mesh.GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            T.SetIntPoint(&ir.IntPoint(q));
            Vector o;
            P.w.GetCurl(T, o);
            if (std::abs(alpha) * o.Norml2() / sigma > thr)
            {
               above += ir.IntPoint(q).weight * T.Weight();
            }
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &above, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      const RotationNumberStats s = rn.Compute(P.w, alpha, sigma, thr);
      EXPECT_NEAR(s.max_mu, mx, 1e-12 * mx);
      EXPECT_NEAR(s.vol_fraction, above / total, 1e-12);
      EXPECT_GT(s.vol_fraction, 0.0);
      EXPECT_LT(s.vol_fraction, 1.0);
      EXPECT_EQ(s.threshold, thr);
   }
}
