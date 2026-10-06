// VectorRotationalConvectionIntegrator -- rotational_convection_pa_spec.md
// Part A tests (spec par.6):
//  A1 vector identity C(u)u - N(u)u - G(u) = 0 against MFEM's
//     VectorConvectionNLFIntegrator (shares no code with the new kernels),
//     plus the sign-flipped residual proving u x omega would be caught;
//  A2 operator properties: skew-symmetry/energy, transpose, diagonal, update
//     semantics on a prebuilt operator, rotation-number diagnostic, GPU
//     determinism;
//  A3 serial versus parallel (y^T N x, x^T N x = 0, diagnostics).
// Sweep: dim {2,3} x p {1..4} x rule {R1 default, R2 GLL collocated, R3 Gauss
// 3/2, R4 Gauss p+1} x mesh {M2 curved, M3 periodic}. Every case runs on a
// partitioned ParMesh, so np = 2, 4 also exercise shared dofs. Repo
// conventions replace the spec's Catch2 GENERATE/CAPTURE with loops and
// SCOPED_TRACE; the debug device comes from scripts/debug_device.sh.

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "operators/convection.hpp"
#include "operators/rotational_convection.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using incns::Rule1D;
using incns::RuleBook;
using incns::RotationNumberStats;
using incns::VectorRotationalConvectionIntegrator;

namespace
{

int g_dim = 2; // mesh dimension for the field callbacks (spec App. B)
// On M3 the fields must be periodic: a non-periodic field projected onto a
// periodic space takes, at each seam dof, whichever side wrote last -- and
// serial and partitioned runs pick differently. So on M3 the periodic
// coordinates are remapped through sin(2 pi .) before evaluation.
bool g_periodic = false;

// The evaluation point: x itself, or its periodic remap on M3.
void FieldPoint(const Vector& x, real_t& X, real_t& Y, real_t& Z)
{
   X = x(0);
   Y = x(1);
   Z = (g_dim == 3) ? x(2) : 0.0;
   if (g_periodic)
   {
      X = 0.5 + 0.3 * sin(2 * M_PI * X);
      if (g_dim == 3) { Z = 0.5 + 0.3 * sin(2 * M_PI * Z); }
   }
}

// Smooth, non-polynomial; every curl component nonzero (spec App. B).
void vel_fn(const Vector& x, Vector& u)
{
   real_t X, Y, Z;
   FieldPoint(x, X, Y, Z);
   if (g_dim == 2)
   {
      u(0) = sin(M_PI * X) * cos(1.3 * M_PI * Y) + 0.3 * Y;
      u(1) = cos(0.7 * M_PI * X) * sin(M_PI * Y) - 0.2 * X * X;
   }
   else
   {
      u(0) = sin(M_PI * X) * cos(1.3 * M_PI * Y) * cos(0.6 * Z) + 0.3 * Y;
      u(1) = cos(0.7 * M_PI * X) * sin(M_PI * Y) * sin(0.9 * Z) - 0.2 * X * X;
      u(2) = sin(0.8 * X + 1.1 * Y) * cos(M_PI * Z) + 0.1 * Z * Y;
   }
}

// Two more smooth fields (update tests, A3's x and y).
void vel_fn2(const Vector& x, Vector& u)
{
   real_t X, Y, Z;
   FieldPoint(x, X, Y, Z);
   u(0) = cos(1.1 * M_PI * X) * sin(0.8 * M_PI * Y) + 0.1 + 0.2 * Z;
   u(1) = sin(0.9 * M_PI * X + 0.3) * cos(M_PI * Y) - 0.2 * Y;
   if (g_dim == 3) { u(2) = cos(0.5 * X - 0.7 * Y) * sin(1.2 * Z) + 0.4 * X; }
}

void vel_fn3(const Vector& x, Vector& u)
{
   real_t X, Y, Z;
   FieldPoint(x, X, Y, Z);
   u(0) = X * Y + 0.3 - Z * Z;
   u(1) = X - Y * Y + 0.5 * X * Z;
   if (g_dim == 3) { u(2) = exp(0.3 * X) * Y - 0.2 * Z; }
}

// Rigid rotation: 3D w = Omega x x, 2D w = Omega (-y, x); curl w = 2 Omega.
const real_t kOmega3[3] = {0.3, -0.5, 0.8};
const real_t kOmega2 = 0.7;
void rigid_fn(const Vector& x, Vector& u)
{
   if (g_dim == 2)
   {
      u(0) = -kOmega2 * x(1);
      u(1) = kOmega2 * x(0);
   }
   else
   {
      u(0) = kOmega3[1] * x(2) - kOmega3[2] * x(1);
      u(1) = kOmega3[2] * x(0) - kOmega3[0] * x(2);
      u(2) = kOmega3[0] * x(1) - kOmega3[1] * x(0);
   }
}

// grad(|u|^2/2) = (grad u)^T u from a GridFunction (spec App. B, test A1).
class KEGrad : public VectorCoefficient
{
   const GridFunction& u_;
   DenseMatrix G_;
   Vector U_;

public:
   KEGrad(const GridFunction& u, int d) : VectorCoefficient(d), u_(u) { }
   void Eval(Vector& V, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      T.SetIntPoint(&ip);
      u_.GetVectorGradient(T, G_); // G(c, k) = du_c/dx_k
      u_.GetVectorValue(T, ip, U_);
      V.SetSize(vdim);
      G_.MultTranspose(U_, V);
   }
};

// M2 (spec App. B): shear (non-symmetric J) plus smooth curvature.
Mesh MakeCurvedMesh(int dim, int n, int mesh_order)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0,
                                       1.0, 1.0);
   mesh.SetCurvature(mesh_order, false, dim, Ordering::byNODES);
   mesh.Transform([ = ](const Vector & x, Vector & y)
   {
      y = x;
      y(1) += 0.2 * x(0);
      if (dim == 3) { y(2) += 0.3 * x(0); }
      y(0) += 0.04 * sin(M_PI * x(1)) * sin(M_PI * x(0));
      y(1) += 0.04 * sin(M_PI * x(0)) * sin(2 * M_PI * x(1));
      if (dim == 3) { y(2) += 0.03 * sin(M_PI * x(0)) * sin(M_PI * x(2)); }
   });
   return mesh;
}

// M3: periodic in x (and z in 3D), 3 elements per direction (MakeBoxMesh's
// minimum for periodicity), curved by a displacement that is periodic across
// the seams and vanishes on the y walls.
Mesh MakePeriodicMesh(int dim, int mesh_order)
{
   incns::BoxSpec spec;
   spec.dim = dim;
   spec.num_elems = {3, 3, 3};
   spec.lengths = {1.0, 1.0, 1.0};
   spec.periodic = {true, false, dim == 3};
   Mesh mesh = incns::MakeBoxMesh(spec);
   mesh.SetCurvature(mesh_order, true, dim, Ordering::byNODES);
   mesh.Transform([ = ](const Vector & x, Vector & y)
   {
      y = x;
      const real_t s = sin(M_PI * x(1));
      y(0) += 0.03 * sin(2 * M_PI * x(0)) * s;
      y(1) += 0.04 * sin(2 * M_PI * x(0)) * s;
      if (dim == 3) { y(2) += 0.03 * sin(2 * M_PI * x(2)) * s; }
   });
   return mesh;
}

enum class Rule { R1, R2, R3, R4 };
enum class MeshKind { M2, M3 };

struct Case
{
   int dim, p;
   Rule rule;
   MeshKind mesh;
   std::string Label() const
   {
      static const char* rn[] = {"R1 default", "R2 GLL", "R3 3/2", "R4 Gauss p+1"};
      std::ostringstream s;
      s << "dim=" << dim << " p=" << p << " rule=" << rn[static_cast<int>(rule)]
        << " mesh=" << (mesh == MeshKind::M2 ? "M2 curved" : "M3 periodic");
      return s.str();
   }
};

std::vector<Case> Sweep(bool with_periodic = true)
{
   std::vector<Case> cases;
   for (int dim : {2, 3})
      for (int p = 1; p <= 4; ++p)
         for (Rule r : {Rule::R1, Rule::R2, Rule::R3, Rule::R4})
         {
            cases.push_back({dim, p, r, MeshKind::M2});
            if (with_periodic) { cases.push_back({dim, p, r, MeshKind::M3}); }
         }
   return cases;
}

Mesh MakeMesh(const Case& c)
{
   return c.mesh == MeshKind::M2 ? MakeCurvedMesh(c.dim, c.dim == 2 ? 3 : 2, c.p)
          : MakePeriodicMesh(c.dim, c.p);
}

// The four rules of the sweep; RuleBook-owned (stable addresses) except R1,
// the integrator's own default (an MFEM IntRules entry, also stable).
const IntegrationRule& PickRule(RuleBook& rules, const Case& c, Mesh& mesh,
                                const FiniteElement& fe)
{
   const Geometry::Type geom = (c.dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   switch (c.rule)
   {
      case Rule::R1:
         return VectorRotationalConvectionIntegrator::GetRule(
                   fe, *mesh.GetTypicalElementTransformation());
      case Rule::R2: // p + 1 GLL points, collocated with the nodes
         return rules.Get(geom, 2 * c.p - 1, Rule1D::GaussLobatto);
      case Rule::R3: // ceil(3(p+1)/2) Gauss points
      {
         const int q = (3 * (c.p + 1) + 1) / 2;
         return rules.Get(geom, 2 * q - 1);
      }
      case Rule::R4: // p + 1 Gauss points
      default:
         return rules.Get(geom, 2 * c.p + 1);
   }
}

real_t Dot(const Vector& a, const Vector& b)
{
   return InnerProduct(MPI_COMM_WORLD, a, b);
}

real_t Norm(const Vector& v) { return std::sqrt(Dot(v, v)); }

// True-dof interpolant of fn, with the GridFunction left distributed.
void Interpolate(ParGridFunction& gf, void (*fn)(const Vector&, Vector&),
                 Vector& tdofs)
{
   VectorFunctionCoefficient vc(gf.ParFESpace()->GetVDim(), fn);
   gf.ProjectCoefficient(vc);
   tdofs.SetSize(gf.ParFESpace()->GetTrueVSize());
   gf.GetTrueDofs(tdofs);
   gf.SetFromTrueDofs(tdofs);
}

// One-integrator PA form for N, its operator on true dofs (P^T N P).
struct NForm
{
   std::unique_ptr<ParBilinearForm> form;
   VectorRotationalConvectionIntegrator* rot = nullptr; // owned by form
   OperatorPtr op;
   NForm(ParFiniteElementSpace& fes, const GridFunction& w, real_t alpha,
         const IntegrationRule& ir, const Array<int>& ess)
   {
      form = std::make_unique<ParBilinearForm>(&fes);
      rot = new VectorRotationalConvectionIntegrator(w, alpha);
      rot->SetIntRule(&ir);
      form->AddDomainIntegrator(rot);
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form->Assemble();
      form->FormSystemMatrix(ess, op);
   }
};

} // namespace

// A1 -- C(u)u - N(u)u - G(u) = 0 to roundoff, for every rule (including the
// under-integrated R2/R4: the identity is pointwise), and the sign-flipped
// residual is O(1), so omega x u vs u x omega is caught.
TEST(RotConv, A1_VectorIdentity)
{
   RuleBook rules;
   for (const Case& c : Sweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());

      ParGridFunction u(&fes);
      Vector U;
      Interpolate(u, vel_fn, U);

      // C(u)u by MFEM's convection integrator, PA as the spec says -- except
      // where the installed MFEM (17d1afc) has no 3D PA kernel: measured
      // q1d <= 8 only, and R1 on an order-4 mesh needs 10. There the legacy
      // element-vector path stands in (still MFEM's code, still independent).
      const int q1d = static_cast<int>(
                         std::lround(std::pow(ir.GetNPoints(), 1.0 / c.dim)));
      const bool conv_pa = !(c.dim == 3 && q1d > 8);
      ParNonlinearForm cform(&fes);
      auto* ci = new VectorConvectionNLFIntegrator();
      ci->SetIntRule(&ir);
      cform.AddDomainIntegrator(ci);
      if (conv_pa) { cform.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      cform.Setup();
      Vector C(U.Size());
      cform.Mult(U, C);

      Array<int> empty;
      NForm n(fes, u, 1.0, ir, empty);
      Vector N(U.Size());
      n.op->Mult(U, N);

      KEGrad ke(u, c.dim);
      ParLinearForm gform(&fes);
      auto* gi = new VectorDomainLFIntegrator(ke);
      gi->SetIntRule(&ir);
      gform.AddDomainIntegrator(gi);
      gform.Assemble();
      Vector G(U.Size());
      gform.ParallelAssemble(G);

      Vector r(C), rflip(C);
      r -= N;
      r -= G;
      rflip += N;
      rflip -= G;
      const real_t nc = Norm(C);
      EXPECT_LE(Norm(r), 1e-12 * nc);
      EXPECT_GT(Norm(rflip), 0.1 * nc);
   }
}

// A2 -- skew-symmetry and energy neutrality on random vectors (also catches
// a kernel that reads w instead of x, which A1 cannot: there x = w), and the
// transpose is -N.
TEST(RotConv, A2_SkewAndTranspose)
{
   RuleBook rules;
   for (const Case& c : Sweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());
      ParGridFunction w(&fes);
      Vector W;
      Interpolate(w, vel_fn, W);
      Array<int> empty;
      NForm n(fes, w, 1.7, ir, empty);

      const int ts = fes.GetTrueVSize();
      Vector x(ts), y(ts), Nx(ts), Ny(ts), NTx(ts);
      x.Randomize(11);
      y.Randomize(23);
      n.op->Mult(x, Nx);
      n.op->Mult(y, Ny);
      EXPECT_LE(std::fabs(Dot(y, Nx) + Dot(x, Ny)), 1e-12 * Norm(x) * Norm(Ny));
      EXPECT_LE(std::fabs(Dot(x, Nx)), 1e-12 * Norm(x) * Norm(Nx));

      n.op->MultTranspose(x, NTx);
      NTx += Nx;
      EXPECT_LE(Norm(NTx), 1e-15 * Norm(Nx));
   }
}

// A2 -- the rotation term adds exact zeros to the diagonal: sigma M + nu K + N
// has the bitwise diagonal of sigma M + nu K, and N alone has a zero one.
TEST(RotConv, A2_Diagonal)
{
   RuleBook rules;
   for (const Case& c : Sweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());
      ParGridFunction w(&fes);
      Vector W;
      Interpolate(w, vel_fn, W);
      ConstantCoefficient sigma(3.0), nu(0.05);

      auto build = [&](bool with_n, bool with_sym, Vector & diag)
      {
         ParBilinearForm a(&fes);
         if (with_sym)
         {
            a.AddDomainIntegrator(new VectorMassIntegrator(sigma));
            a.AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
         }
         if (with_n)
         {
            auto* rot = new VectorRotationalConvectionIntegrator(w, 1.7);
            rot->SetIntRule(&ir);
            a.AddDomainIntegrator(rot);
         }
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.Assemble();
         diag.SetSize(fes.GetTrueVSize());
         a.AssembleDiagonal(diag);
      };
      Vector d_sym, d_full, d_n;
      build(false, true, d_sym);
      build(true, true, d_full);
      build(true, false, d_n);
      d_full -= d_sym;
      EXPECT_EQ(d_full.Normlinf(), 0.0);
      EXPECT_EQ(d_n.Normlinf(), 0.0);
   }
}

// A2 -- update semantics on a PREBUILT constrained operator: changing w in
// place, swapping in a w on a different (byVDIM) space, and changing alpha
// all reach the old operator through UpdateVorticity() and match a freshly
// built one. Catches stale buffers and copies of the quadrature data.
TEST(RotConv, A2_UpdateSemantics)
{
   RuleBook rules;
   for (const Case& c : Sweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      ParFiniteElementSpace fes_vdim(&mesh, &fec, c.dim, Ordering::byVDIM);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());
      Array<int> ess;
      if (mesh.bdr_attributes.Size())
      {
         Array<int> bdr(mesh.bdr_attributes.Max());
         bdr = 1;
         fes.GetEssentialTrueDofs(bdr, ess);
      }

      ParGridFunction w(&fes), w2(&fes_vdim);
      Vector T;
      Interpolate(w, vel_fn, T);
      NForm kept(fes, w, 1.7, ir, ess);

      const int ts = fes.GetTrueVSize();
      Vector x(ts), y_kept(ts), y_fresh(ts);
      x.Randomize(5);
      auto check = [&](const GridFunction & wnow, real_t alpha, const char* what)
      {
         SCOPED_TRACE(what);
         kept.rot->UpdateVorticity();
         NForm fresh(fes, wnow, alpha, ir, ess);
         kept.op->Mult(x, y_kept);
         fresh.op->Mult(x, y_fresh);
         y_kept -= y_fresh;
         EXPECT_LE(Norm(y_kept), 1e-12 * Norm(y_fresh));
      };

      Interpolate(w, vel_fn2, T); // in place
      check(w, 1.7, "w changed in place");
      Interpolate(w2, vel_fn3, T);
      kept.rot->SetLaggedVelocity(w2); // different space object and ordering
      check(w2, 1.7, "SetLaggedVelocity(byVDIM w2)");
      kept.rot->SetAlpha(2.0);
      check(w2, 2.0, "SetAlpha(2)");
   }
}

// A2 -- rotation-number diagnostic. A rigid rotation has curl w_h = 2 Omega
// exactly on any mesh of order <= p (P4), so with sigma = 2: max mu = |Omega|
// and the volume fraction is 1 below |Omega| and 0 above. M2 only: the
// rigid rotation is not periodic, so M3 would not represent it exactly.
TEST(RotConv, A2_RotationNumber)
{
   RuleBook rules;
   for (const Case& c : Sweep(false))
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());
      ParGridFunction w(&fes);
      Vector W;
      Interpolate(w, rigid_fn, W);
      Array<int> empty;
      NForm n(fes, w, 1.7, ir, empty);

      const real_t om = (c.dim == 2)
                        ? kOmega2
                        : std::sqrt(kOmega3[0] * kOmega3[0] + kOmega3[1] * kOmega3[1] +
                                    kOmega3[2] * kOmega3[2]);
      const RotationNumberStats lo = n.rot->GetRotationNumberStats(2.0, 0.9 * om);
      const RotationNumberStats hi = n.rot->GetRotationNumberStats(2.0, 1.1 * om);
      EXPECT_NEAR(lo.max_mu, om, 1e-12 * om);
      EXPECT_NEAR(lo.vol_fraction, 1.0, 1e-12);
      EXPECT_EQ(hi.vol_fraction, 0.0);
      EXPECT_EQ(lo.threshold, 0.9 * om);
   }
}

// A2 -- determinism (GPU only): 100 applies of the same input are bitwise
// identical; a missing MFEM_SYNC_THREAD or a shared-memory race shows up here.
TEST(RotConv, A2_DeterminismGpu)
{
   if (!Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
   {
      GTEST_SKIP() << "GPU only";
   }
   RuleBook rules;
   for (const Case& c : Sweep(false))
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *fes.GetTypicalFE());
      ParGridFunction w(&fes);
      Vector W;
      Interpolate(w, vel_fn, W);
      Array<int> empty;
      NForm n(fes, w, 1.7, ir, empty);
      Vector x(fes.GetTrueVSize()), y0(x.Size()), y(x.Size());
      x.Randomize(3);
      n.op->Mult(x, y0);
      for (int k = 0; k < 100; ++k)
      {
         n.op->Mult(x, y);
         y -= y0;
         ASSERT_EQ(y.Normlinf(), 0.0) << "apply " << k;
      }
   }
}

// A3 -- serial versus parallel: y^T N x and the diagnostics on a partitioned
// copy of the same global mesh equal the serial values (stale shared dofs in
// w, wrong reductions), and P^T N P stays skew across ranks.
TEST(RotConv, A3_SerialVersusParallel)
{
   RuleBook rules;
   for (const Case& c : Sweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = (c.mesh == MeshKind::M3);
      Mesh serial = MakeMesh(c);
      H1_FECollection fec(c.p, c.dim);

      // Serial reference, computed identically on every rank.
      FiniteElementSpace sfes(&serial, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, serial, *sfes.GetTypicalFE());
      GridFunction sw(&sfes), sx(&sfes), sy(&sfes);
      VectorFunctionCoefficient fw(c.dim, vel_fn), fx(c.dim, vel_fn2),
                                fy(c.dim, vel_fn3);
      sw.ProjectCoefficient(fw);
      sx.ProjectCoefficient(fx);
      sy.ProjectCoefficient(fy);
      BilinearForm sform(&sfes);
      auto* srot = new VectorRotationalConvectionIntegrator(sw, 1.7);
      srot->SetIntRule(&ir);
      sform.AddDomainIntegrator(srot);
      sform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      sform.Assemble();
      Array<int> empty;
      OperatorPtr sop;
      sform.FormSystemMatrix(empty, sop);
      Vector sNx(sx.Size());
      sop->Mult(sx, sNx);
      const real_t yNx_ser = sy * sNx;
      const RotationNumberStats st_ser0 = srot->GetRotationNumberStats(2.0);
      const RotationNumberStats st_ser = srot->GetRotationNumberStats(
                                            2.0, 0.5 * st_ser0.max_mu);

      // Parallel.
      ParMesh mesh(MPI_COMM_WORLD, serial);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      ParGridFunction w(&fes), xg(&fes), yg(&fes);
      Vector W, X, Y;
      Interpolate(w, vel_fn, W);
      Interpolate(xg, vel_fn2, X);
      Interpolate(yg, vel_fn3, Y);
      NForm n(fes, w, 1.7, ir, empty);
      Vector NX(X.Size());
      n.op->Mult(X, NX);
      EXPECT_NEAR(Dot(Y, NX), yNx_ser, 1e-12 * std::fabs(yNx_ser));
      EXPECT_LE(std::fabs(Dot(X, NX)), 1e-12 * Norm(X) * Norm(NX));

      const RotationNumberStats st = n.rot->GetRotationNumberStats(
                                        2.0, 0.5 * st_ser0.max_mu);
      EXPECT_NEAR(st.max_mu, st_ser.max_mu, 1e-12 * st_ser.max_mu);
      EXPECT_NEAR(st.vol_fraction, st_ser.vol_fraction, 1e-12);
      EXPECT_GT(st_ser.vol_fraction, 0.0); // threshold actually splits the domain
      EXPECT_LT(st_ser.vol_fraction, 1.0);
   }
}

// ===========================================================================
// Full / element assembly (added after the spec, for LOR use): legacy
// AssembleElementMatrix, the per-component EA integrator, and the LOR path.
// ===========================================================================
namespace
{
// alpha [omega]_x from a vector coefficient omega (vdim 1 in 2D, 3 in 3D);
// with MFEM's VectorMassIntegrator(MatrixCoefficient&) it is an MFEM-only
// reference for N (spec App. B).
class SkewFromOmega : public MatrixCoefficient
{
   VectorCoefficient& om_;
   real_t alpha_;
   Vector o_;

public:
   SkewFromOmega(int dim, VectorCoefficient& om, real_t a)
      : MatrixCoefficient(dim), om_(om), alpha_(a), o_(om.GetVDim()) { }
   void Eval(DenseMatrix& K, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      om_.Eval(o_, T, ip);
      const int d = GetHeight();
      K.SetSize(d);
      K = 0.0;
      if (d == 2)
      {
         K(0, 1) = -alpha_ * o_(0);
         K(1, 0) = alpha_ * o_(0);
      }
      else
      {
         K(0, 1) = -alpha_ * o_(2);
         K(0, 2) = alpha_ * o_(1);
         K(1, 0) = alpha_ * o_(2);
         K(1, 2) = -alpha_ * o_(0);
         K(2, 0) = -alpha_ * o_(1);
         K(2, 1) = alpha_ * o_(0);
      }
   }
};

std::vector<Case> AssemblySweep()
{
   std::vector<Case> cases;
   for (int dim : {2, 3})
      for (int p = 1; p <= 3; ++p)
         for (Rule r : {Rule::R1, Rule::R2, Rule::R4})
         {
            cases.push_back({dim, p, r, MeshKind::M2});
         }
   return cases;
}
} // namespace

// Legacy element matrices equal an MFEM-only reference (VectorMassIntegrator
// with alpha [curl w]_x from CurlGridFunctionCoefficient) element by element,
// and the legacy-assembled operator equals the PA one on a partitioned mesh.
TEST(RotConv, FullAssemblyMatchesMfemReference)
{
   RuleBook rules;
   for (const Case& c : AssemblySweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = false;
      Mesh serial = MakeMesh(c);
      H1_FECollection fec(c.p, c.dim);

      // Element by element, serial.
      FiniteElementSpace sfes(&serial, &fec, c.dim, Ordering::byNODES);
      const IntegrationRule& ir = PickRule(rules, c, serial, *sfes.GetTypicalFE());
      GridFunction sw(&sfes);
      VectorFunctionCoefficient fw(c.dim, vel_fn);
      sw.ProjectCoefficient(fw);
      VectorRotationalConvectionIntegrator rot(sw, 1.7);
      rot.SetIntRule(&ir);
      CurlGridFunctionCoefficient curl(&sw);
      SkewFromOmega skew(c.dim, curl, 1.7);
      VectorMassIntegrator ref(skew);
      ref.SetIntRule(&ir);
      for (int e = 0; e < serial.GetNE(); ++e)
      {
         DenseMatrix A, B;
         rot.AssembleElementMatrix(*sfes.GetFE(e),
                                   *serial.GetElementTransformation(e), A);
         ref.AssembleElementMatrix(*sfes.GetFE(e),
                                   *serial.GetElementTransformation(e), B);
         B -= A;
         EXPECT_LE(B.MaxMaxNorm(), 1e-12 * A.MaxMaxNorm()) << "element " << e;
      }

      // Legacy-assembled operator vs PA, partitioned.
      ParMesh mesh(MPI_COMM_WORLD, serial);
      ParFiniteElementSpace fes(&mesh, &fec, c.dim, Ordering::byNODES);
      ParGridFunction w(&fes);
      Vector W;
      Interpolate(w, vel_fn, W);
      Array<int> empty;
      NForm pa(fes, w, 1.7, ir, empty);
      ParBilinearForm fa(&fes);
      auto* fi = new VectorRotationalConvectionIntegrator(w, 1.7);
      fi->SetIntRule(&ir);
      fa.AddDomainIntegrator(fi);
      fa.Assemble();
      fa.Finalize();
      OperatorPtr Afa;
      fa.FormSystemMatrix(empty, Afa);
      Vector x(fes.GetTrueVSize()), ypa(x.Size()), yfa(x.Size());
      x.Randomize(19);
      pa.op->Mult(x, ypa);
      Afa->Mult(x, yfa);
      yfa -= ypa;
      EXPECT_LE(Norm(yfa), 1e-12 * Norm(ypa));
   }
}

// The dim x dim component blocks, each element-assembled on a SCALAR space
// (AssemblyLevel::FULL -> AssembleEA -> sparse matrix) and merged with
// HypreParMatrixFromBlocks, equal the legacy-assembled vector operator (the
// off-diagonal blocks are not symmetric: pins the EA layout); each block's
// ELEMENT-level action equals its FULL matrix. At np > 1 this also pins the
// merged block ordering against the byNODES true-dof ordering.
TEST(RotConv, ComponentEaMatchesVectorAssembly)
{
   RuleBook rules;
   for (const Case& c : AssemblySweep())
   {
      SCOPED_TRACE(c.Label());
      g_dim = c.dim;
      g_periodic = false;
      Mesh serial = MakeMesh(c);
      ParMesh mesh(MPI_COMM_WORLD, serial);
      H1_FECollection fec(c.p, c.dim);
      ParFiniteElementSpace vfes(&mesh, &fec, c.dim, Ordering::byNODES);
      ParFiniteElementSpace sfes(&mesh, &fec);
      const IntegrationRule& ir = PickRule(rules, c, mesh, *vfes.GetTypicalFE());
      ParGridFunction w(&vfes);
      Vector W;
      Interpolate(w, vel_fn, W);
      Array<int> empty;

      ParBilinearForm ref(&vfes);
      auto* ri = new VectorRotationalConvectionIntegrator(w, 1.7);
      ri->SetIntRule(&ir);
      ref.AddDomainIntegrator(ri);
      ref.Assemble();
      ref.Finalize();
      OperatorPtr K_ref;
      ref.FormSystemMatrix(empty, K_ref);

      // Lifetimes (learned on the grad-div blocks): OperatorHandle copies do
      // not own, so the handle vector must never reallocate; and at np = 1 a
      // FULL matrix borrows its form's storage, so the forms outlive the merge.
      std::vector<std::unique_ptr<ParBilinearForm>> forms;
      std::vector<OperatorPtr> K_blocks;
      K_blocks.reserve(c.dim * c.dim);
      Array2D<const HypreParMatrix*> blocks(c.dim, c.dim);
      for (int i = 0; i < c.dim; ++i)
         for (int j = 0; j < c.dim; ++j)
         {
            forms.push_back(std::make_unique<ParBilinearForm>(&sfes));
            ParBilinearForm& fa = *forms.back();
            ParBilinearForm ea(&sfes);
            fa.SetAssemblyLevel(AssemblyLevel::FULL);
            ea.SetAssemblyLevel(AssemblyLevel::ELEMENT);
            auto make = [&]()
            {
               auto* bi = new incns::VectorRotationalConvectionComponentIntegrator(
                  w, 1.7, i, j);
               bi->SetIntRule(&ir);
               return bi;
            };
            fa.AddDomainIntegrator(make());
            ea.AddDomainIntegrator(make());
            fa.Assemble();
            ea.Assemble();
            K_blocks.emplace_back(Operator::Hypre_ParCSR);
            fa.FormSystemMatrix(empty, K_blocks.back());
            const HypreParMatrix* Kij = K_blocks.back().As<HypreParMatrix>();
            ASSERT_NE(Kij, nullptr);
            blocks(i, j) = Kij;

            OperatorPtr A_ea;
            ea.FormSystemMatrix(empty, A_ea);
            Vector x(sfes.GetTrueVSize()), y_ea(x.Size()), y_fa(x.Size());
            x.Randomize(31 + 3 * i + j);
            A_ea->Mult(x, y_ea);
            Kij->Mult(x, y_fa);
            y_ea -= y_fa;
            EXPECT_LE(Norm(y_ea), 1e-12 * std::max(Norm(y_fa), 1e-300))
                  << "ELEMENT vs FULL, block (" << i << "," << j << ")";
         }
      std::unique_ptr<HypreParMatrix> K(HypreParMatrixFromBlocks(blocks));
      ASSERT_EQ(K->Height(), K_ref->Height());
      Vector x(vfes.GetTrueVSize()), y(x.Size()), y_ref(x.Size());
      x.Randomize(7);
      K->Mult(x, y);
      K_ref->Mult(x, y_ref);
      y -= y_ref;
      EXPECT_LE(Norm(y), 1e-12 * Norm(y_ref)) << "merged blocks vs vector";
   }
}

// LOR: w copied onto the LOR space by TRUE DOFS (H1 LOR shares them) and the
// rotation term assembled through MFEM's ParLORDiscretization (legacy element
// matrices on the LOR elements). A rigid rotation has curl = 2 Omega exactly
// on the straight-sided LOR elements, so the LOR operator must be exactly
// alpha [2 Omega]_x (x) M_LOR, with M_LOR the LOR vector mass assembled the
// same way: N_LOR x = M_LOR (K (x) I) x. Pins the w-on-LOR handling, the
// mesh match, the sign and alpha on the LOR path.
TEST(RotConv, LorRigidRotation)
{
   for (int dim : {2, 3})
      for (int p = 2; p <= 3; ++p)
      {
         SCOPED_TRACE("dim=" + std::to_string(dim) + " p=" + std::to_string(p));
         g_dim = dim;
         g_periodic = false;
         Mesh serial = MakeCurvedMesh(dim, dim == 2 ? 3 : 2, p);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
         ParGridFunction w_ho(&fes);
         Vector W;
         Interpolate(w_ho, rigid_fn, W);
         const real_t alpha = 1.7;

         ParLORDiscretization lor_n(fes), lor_m(fes);
         ParGridFunction w_lor(&lor_n.GetParFESpace());
         w_lor.SetFromTrueDofs(W); // the solver's copy path
         Array<int> empty;
         ParBilinearForm an(&fes), am(&fes);
         an.AddDomainIntegrator(
              new VectorRotationalConvectionIntegrator(w_lor, alpha));
         am.AddDomainIntegrator(new VectorMassIntegrator());
         lor_n.AssembleSystem(an, empty);
         lor_m.AssembleSystem(am, empty);
         const HypreParMatrix& N = lor_n.GetAssembledMatrix();
         const HypreParMatrix& M = lor_m.GetAssembledMatrix();

         // K = alpha [2 Omega]_x.
         DenseMatrix K(dim);
         K = 0.0;
         if (dim == 2)
         {
            K(0, 1) = -alpha * 2 * kOmega2;
            K(1, 0) = alpha * 2 * kOmega2;
         }
         else
         {
            const real_t o0 = 2 * kOmega3[0], o1 = 2 * kOmega3[1],
                         o2 = 2 * kOmega3[2];
            K(0, 1) = -alpha * o2; K(0, 2) = alpha * o1;
            K(1, 0) = alpha * o2;  K(1, 2) = -alpha * o0;
            K(2, 0) = -alpha * o1; K(2, 1) = alpha * o0;
         }
         const int n = fes.GetTrueVSize() / dim;
         Vector x(fes.GetTrueVSize()), z(x.Size()), y(x.Size()), y_ref(x.Size());
         x.Randomize(13);
         const real_t* X = x.HostRead();
         real_t* Z = z.HostWrite();
         for (int a = 0; a < n; ++a)
            for (int i = 0; i < dim; ++i)
            {
               real_t s = 0.0;
               for (int j = 0; j < dim; ++j) { s += K(i, j) * X[j * n + a]; }
               Z[i * n + a] = s;
            }
         N.Mult(x, y);
         M.Mult(z, y_ref);
         y -= y_ref;
         EXPECT_LE(Norm(y), 1e-12 * Norm(y_ref));
      }
}

// The solver integrates the rotation term at the RuleBook's dealiased
// Gauss-Legendre rule (Convection::DealiasedOrder, see StokesOperator). Every
// such size for p = 1..5 must hit a compile-time-specialized kernel, so a
// change to the dealiasing order (or to the specialization list) cannot
// silently drop production runs onto the slower generic fallback.
TEST(RotConv, DealiasedRuleIsSpecialized)
{
   RuleBook rules;
   for (int dim : {2, 3})
      for (int p = 1; p <= 5; ++p)
      {
         const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
         const IntegrationRule& ir =
            rules.Get(geom, incns::Convection::DealiasedOrder(p));
         const int q1d = static_cast<int>(
                            std::lround(std::pow(ir.GetNPoints(), 1.0 / dim)));
         EXPECT_TRUE(VectorRotationalConvectionIntegrator::HasSpecialization(
                        dim, p + 1, q1d))
               << "dim=" << dim << " p=" << p << " q1d=" << q1d;
      }
}
