// PointBlockJacobi -- docs/design/rotational_convection_pa_spec.md Part B tests (par.7.6):
//  B1 PBJ applies the exact inverse of every nodal block of a legacy-assembled
//     reference, with no / all-component / component-0 essential dofs,
//     byNODES + byVDIM;
//  B2 the nodal blocks diag(d) + [s]_x themselves vs that reference;
//  B3 GMRES iterations at max|omega| dt = 100 vs 0, after changing w, and the
//     partitioned count vs a serial (MPI_COMM_SELF) solve of the same system.
// Every operator is integrated at the solver's rules (SolverRules below), so
// the rotation term always runs its compile-time-specialized kernels -- the
// configuration that runs in production, on a GPU too. (The spec's B1 used
// GLL collocation, where PBJ is exact; the solver never uses that rule for N.)
// Mesh M2 (spec App. B), alpha = 1.7, sigma = 3 unless stated; gtest loops +
// SCOPED_TRACE replace the spec's Catch2 GENERATE/CAPTURE.
// Self-contained on purpose: MFEM, the rotational-form integrator and the
// preconditioner under test only (references and rules come from MFEM).

#include <gtest/gtest.h>

#include "operators/rotational_convection.hpp"
#include "precond/point_block_jacobi.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using incns::PointBlockJacobi;
using incns::VectorRotationalConvectionIntegrator;

namespace
{

int g_dim = 2; // mesh dimension for the field callbacks

void vel_fn(const Vector& x, Vector& u) // spec App. B
{
   const real_t X = x(0), Y = x(1), Z = (g_dim == 3) ? x(2) : 0.0;
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

void vel_fn2(const Vector& x, Vector& u) // a different smooth field (B3)
{
   const real_t X = x(0), Y = x(1), Z = (g_dim == 3) ? x(2) : 0.0;
   u(0) = cos(1.1 * M_PI * X) * sin(0.8 * M_PI * Y) + 0.1 + 0.2 * Z;
   u(1) = sin(0.9 * M_PI * X + 0.3) * cos(M_PI * Y) - 0.2 * Y;
   if (g_dim == 3) { u(2) = cos(0.5 * X - 0.7 * Y) * sin(1.2 * Z) + 0.4 * X; }
}

// Position-hashed noise in [-1/2, 1/2): a "random" right-hand side that is
// the same global vector in serial and partitioned runs (B3 compares them).
void noise_fn(const Vector& x, Vector& u)
{
   for (int c = 0; c < u.Size(); ++c)
   {
      real_t h = 12.9898 * x(0) + 78.233 * x(1) + 19.17 * c;
      if (x.Size() == 3) { h += 37.719 * x(2); }
      const real_t s = sin(h) * 43758.5453;
      u(c) = s - std::floor(s) - 0.5;
   }
}

// alpha [omega]_x from a vector coefficient omega (vdim 1 in 2D, 3 in 3D);
// with MFEM's VectorMassIntegrator(MatrixCoefficient&) it is the B2 reference
// for N (spec App. B).
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

void Interpolate(ParGridFunction& gf, void (*fn)(const Vector&, Vector&))
{
   VectorFunctionCoefficient vc(gf.ParFESpace()->GetVDim(), fn);
   gf.ProjectCoefficient(vc);
   Vector t(gf.ParFESpace()->GetTrueVSize());
   gf.GetTrueDofs(t);
   gf.SetFromTrueDofs(t);
}

enum class Ess { None, All, Component0 };

const char* EssName(Ess e)
{
   return e == Ess::None ? "none" : e == Ess::All ? "all components" :
          "component 0";
}

void BoundaryDofs(ParFiniteElementSpace& fes, Ess mode, Array<int>& ess)
{
   ess.SetSize(0);
   if (mode == Ess::None || fes.GetParMesh()->bdr_attributes.Size() == 0)
   {
      return;
   }
   Array<int> bdr(fes.GetParMesh()->bdr_attributes.Max());
   bdr = 1;
   if (mode == Ess::All) { fes.GetEssentialTrueDofs(bdr, ess); }
   else { fes.GetEssentialTrueDofs(bdr, ess, 0); }
}

int TrueIdx(int c, int a, int n, int dim, bool by_vdim)
{
   return by_vdim ? a * dim + c : c * n + a;
}

// Entry (i, j) of a finalized SparseMatrix, 0 if absent.
real_t Entry(const SparseMatrix& S, int i, int j)
{
   const int* I = S.HostReadI();
   const int* J = S.HostReadJ();
   const real_t* V = S.HostReadData();
   for (int k = I[i]; k < I[i + 1]; ++k)
   {
      if (J[k] == j) { return V[k]; }
   }
   return 0.0;
}

const real_t kAlpha = 1.7;

// The rules the solver integrates each velocity-block term with, all
// Gauss-Legendre (MFEM's IntRules, the same rules the solver's RuleBook hands
// out): order 2p for the mass, 2p + dim - 1 for the diffusion and 3p for the
// rotation term -- the dealiased rule its kernels are specialized for, p <= 5.
// Hard-coded like rotational_convection_test: if the solver's dealiasing order
// changes, update both tests and the kernel specialization list together.
struct SolverRules
{
   const IntegrationRule* mass;
   const IntegrationRule* diffusion;
   const IntegrationRule* rotation;
};

SolverRules GetSolverRules(int dim, int p)
{
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   return {&IntRules.Get(geom, 2 * p), &IntRules.Get(geom, 2 * p + dim - 1),
           &IntRules.Get(geom, 3 * p)};
}

// sigma M + nu K + N assembled the legacy way from MFEM classes only, N as
// VectorMassIntegrator(alpha [curl w]_x), each term at its solver rule. Its
// GetDiag block (owned rows and columns) holds every nodal block.
std::unique_ptr<HypreParMatrix> Reference(ParFiniteElementSpace& fes,
      ParGridFunction& w, Coefficient& sigma,
      Coefficient& nu, const SolverRules& r)
{
   w.HostRead();
   CurlGridFunctionCoefficient curl(&w);
   SkewFromOmega skew(fes.GetMesh()->Dimension(), curl, kAlpha);
   ParBilinearForm legacy(&fes);
   auto* m = new VectorMassIntegrator(sigma);
   auto* k = new VectorDiffusionIntegrator(nu);
   auto* n = new VectorMassIntegrator(skew);
   m->SetIntRule(r.mass);
   k->SetIntRule(r.diffusion);
   n->SetIntRule(r.rotation);
   legacy.AddDomainIntegrator(m);
   legacy.AddDomainIntegrator(k);
   legacy.AddDomainIntegrator(n);
   legacy.Assemble();
   legacy.Finalize();
   return std::unique_ptr<HypreParMatrix>(legacy.ParallelAssemble());
}

// What the solver builds: the symmetric PA form sigma M + nu K (its diagonal
// is what PBJ gets), the PA form of N, and PBJ on N's integrator.
struct PbjSetup
{
   PbjSetup(ParFiniteElementSpace& fes, ParGridFunction& w, Coefficient& sigma,
            Coefficient& nu, const SolverRules& r, const Array<int>& ess)
      : sym(&fes), nform(&fes)
   {
      auto* m = new VectorMassIntegrator(sigma);
      auto* k = new VectorDiffusionIntegrator(nu);
      m->SetIntRule(r.mass);
      k->SetIntRule(r.diffusion);
      sym.AddDomainIntegrator(m);
      sym.AddDomainIntegrator(k);
      rot = new VectorRotationalConvectionIntegrator(w, kAlpha);
      rot->SetIntRule(r.rotation);
      nform.AddDomainIntegrator(rot);
      for (ParBilinearForm* f : {&sym, &nform})
      {
         f->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         f->Assemble();
      }
      pbj = std::make_unique<PointBlockJacobi>(fes, *rot, ess);
      Vector diag(fes.GetTrueVSize());
      sym.AssembleDiagonal(diag);
      pbj->SetDiagonal(diag);
      pbj->UpdateSkew();
   }

   ParBilinearForm sym;
   ParBilinearForm nform;
   VectorRotationalConvectionIntegrator* rot = nullptr; // owned by nform
   std::unique_ptr<PointBlockJacobi> pbj;
};

} // namespace

// B1 -- PBJ applies the exact inverse of every nodal block of the solver's
// operator sigma M + nu K + N, at the solver's rules. For a random r,
// z = PBJ r; then at every node a, B_a z_a = r_a, where B_a is the dim x dim
// block of the legacy reference (MFEM classes only) with, for an essential
// component c, row and column c replaced by the identity (the constrained
// operator's rows). Catches a wrong d or s, a transposed block, byNODES /
// byVDIM indexing errors and the essential-dof rule.
TEST(PointBlockJacobi, B1_InvertsReferenceBlocks)
{
   ConstantCoefficient sigma(3.0), nu(0.05);
   for (int dim : {2, 3})
      for (int p = 1; p <= 4; ++p)
         for (Ordering::Type ord : {Ordering::byNODES, Ordering::byVDIM})
            for (Ess em : {Ess::None, Ess::All, Ess::Component0})
            {
               std::ostringstream lbl;
               lbl << "dim=" << dim << " p=" << p << " ordering="
                   << (ord == Ordering::byNODES ? "byNODES" : "byVDIM")
                   << " essential=" << EssName(em);
               SCOPED_TRACE(lbl.str());
               g_dim = dim;
               Mesh serial = MakeCurvedMesh(dim, dim == 2 ? 3 : 2, p);
               ParMesh mesh(MPI_COMM_WORLD, serial);
               H1_FECollection fec(p, dim);
               ParFiniteElementSpace fes(&mesh, &fec, dim, ord);
               const SolverRules sr = GetSolverRules(dim, p);
               Array<int> ess;
               BoundaryDofs(fes, em, ess);
               ParGridFunction w(&fes);
               Interpolate(w, vel_fn);

               PbjSetup setup(fes, w, sigma, nu, sr, ess);
               Vector r(fes.GetTrueVSize()), z(r.Size());
               r.Randomize(17);
               setup.pbj->Mult(r, z);

               std::unique_ptr<HypreParMatrix> L = Reference(fes, w, sigma, nu, sr);
               SparseMatrix Ld;
               L->GetDiag(Ld);
               std::vector<bool> is_ess(r.Size(), false);
               const int* E = ess.HostRead(); // PBJ read it on the device
               for (int k = 0; k < ess.Size(); ++k) { is_ess[E[k]] = true; }
               const int n = r.Size() / dim;
               const bool bv = (ord == Ordering::byVDIM);
               const real_t* R = r.HostRead();
               const real_t* Z = z.HostRead();
               real_t err = 0.0, scale = 0.0;
               for (int a = 0; a < n; ++a)
                  for (int i = 0; i < dim; ++i)
                  {
                     const int row = TrueIdx(i, a, n, dim, bv);
                     real_t Bz = 0.0;
                     for (int j = 0; j < dim; ++j)
                     {
                        const int col = TrueIdx(j, a, n, dim, bv);
                        const real_t Bij = (is_ess[row] || is_ess[col])
                                           ? (i == j ? 1.0 : 0.0)
                                           : Entry(Ld, row, col);
                        Bz += Bij * Z[col];
                     }
                     err = std::max(err, std::fabs(Bz - R[row]));
                     scale = std::max(scale, std::fabs(R[row]));
                  }
               MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPITypeMap<real_t>::mpi_type,
                             MPI_MAX, MPI_COMM_WORLD);
               MPI_Allreduce(MPI_IN_PLACE, &scale, 1,
                             MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
               EXPECT_LE(err, 1e-12 * scale);
            }
}

// B2 -- the nodal blocks themselves: PBJ's diag(d) + [s]_x equals the
// legacy reference's dim x dim block at every node and component pair, at the
// solver's rules (the operator is not block diagonal there, so this checks
// the AddNodalSkewPA contraction directly).
TEST(PointBlockJacobi, B2_BlocksMatchLegacyReference)
{
   ConstantCoefficient sigma(3.0), nu(0.05);
   for (int dim : {2, 3})
      for (int p = 1; p <= 4; ++p)
         for (Ordering::Type ord : {Ordering::byNODES, Ordering::byVDIM})
         {
            std::ostringstream lbl;
            lbl << "dim=" << dim << " p=" << p << " ordering="
                << (ord == Ordering::byNODES ? "byNODES" : "byVDIM");
            SCOPED_TRACE(lbl.str());
            g_dim = dim;
            Mesh serial = MakeCurvedMesh(dim, 2, p);
            ParMesh mesh(MPI_COMM_WORLD, serial);
            H1_FECollection fec(p, dim);
            ParFiniteElementSpace fes(&mesh, &fec, dim, ord);
            const SolverRules sr = GetSolverRules(dim, p);
            ParGridFunction w(&fes);
            Interpolate(w, vel_fn);

            std::unique_ptr<HypreParMatrix> L = Reference(fes, w, sigma, nu, sr);
            SparseMatrix Ld;
            L->GetDiag(Ld);
            Array<int> empty;
            PbjSetup setup(fes, w, sigma, nu, sr, empty);

            const int n = fes.GetTrueVSize() / dim;
            const bool bv = (ord == Ordering::byVDIM);
            const real_t* d = setup.pbj->GetDiagonal().HostRead();
            const real_t* s = setup.pbj->GetSkew().HostRead();
            auto block = [&](int i, int j, int a) -> real_t
            {
               if (i == j) { return d[TrueIdx(i, a, n, dim, bv)]; }
               auto S = [&](int c) { return s[TrueIdx(c, a, n, dim, bv)]; };
               if (dim == 2) { return (i == 0) ? -S(0) : S(0); }
               // [s]_x: (0,1) = -s2, (0,2) = s1, (1,0) = s2, (1,2) = -s0,
               // (2,0) = -s1, (2,1) = s0.
               const int kk = 3 - i - j; // the missing index
               const bool cyclic = (j == (i + 1) % 3);
               return cyclic ? -S(kk) : S(kk);
            };
            real_t err = 0.0, scale = 0.0;
            for (int a = 0; a < n; ++a)
               for (int i = 0; i < dim; ++i)
                  for (int j = 0; j < dim; ++j)
                  {
                     const real_t ref = Entry(Ld, TrueIdx(i, a, n, dim, bv),
                                              TrueIdx(j, a, n, dim, bv));
                     err = std::max(err, std::fabs(ref - block(i, j, a)));
                     scale = std::max(scale, std::fabs(ref));
                  }
            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPITypeMap<real_t>::mpi_type,
                          MPI_MAX, MPI_COMM_WORLD);
            EXPECT_LE(err, 1e-12 * scale);
            EXPECT_GT(scale, 0.0);
         }
}

namespace
{

// max |curl w| over the points of ir in every element, global over ranks,
// from MFEM's GridFunction::GetCurl (2D: the scalar vorticity).
real_t MaxVorticity(const ParGridFunction& w, const IntegrationRule& ir)
{
   ParMesh& mesh = *w.ParFESpace()->GetParMesh();
   w.HostRead();
   Vector curl;
   real_t mx = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         w.GetCurl(T, curl);
         mx = std::max(mx, curl.Norml2());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 mesh.GetComm());
   return mx;
}

// B3's system on a given communicator: A = sigma M + nu K + N at the solver's
// rules, Dirichlet on the whole boundary, PA, preconditioned GMRES(50) to 1e-10. Returns the
// iteration counts at max|alpha omega| dt = 0, 100 (field 1) and 100 after
// switching w to field 2; -1 for a solve that did not converge.
struct B3Counts { int zero, rot1, rot2; };

B3Counts RunB3(MPI_Comm comm, int dim)
{
   g_dim = dim;
   const int p = 4, n = 3;
   Mesh serial = MakeCurvedMesh(dim, n, p);
   ParMesh mesh(comm, serial);
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   const SolverRules sr = GetSolverRules(dim, p);

   // dt = 1, sigma = 1/dt (as Table A2), nu dt / h^2 = 0.01 with h = 1/n.
   const real_t dt = 1.0, h = 1.0 / n;
   ConstantCoefficient sigma(1.0 / dt), nu(0.01 * h * h / dt);
   Array<int> ess, bdr(mesh.bdr_attributes.Max());
   bdr = 1;
   fes.GetEssentialTrueDofs(bdr, ess);

   ParGridFunction w1(&fes), w2(&fes), bgf(&fes);
   Interpolate(w1, vel_fn);
   Interpolate(w2, vel_fn2);
   Interpolate(bgf, noise_fn);
   Vector b(fes.GetTrueVSize());
   bgf.GetTrueDofs(b);

   ParBilinearForm a(&fes);
   auto* m = new VectorMassIntegrator(sigma);
   auto* k = new VectorDiffusionIntegrator(nu);
   auto* rot = new VectorRotationalConvectionIntegrator(w1, 1.0);
   m->SetIntRule(sr.mass);
   k->SetIntRule(sr.diffusion);
   rot->SetIntRule(sr.rotation);
   a.AddDomainIntegrator(m);
   a.AddDomainIntegrator(k);
   a.AddDomainIntegrator(rot);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   OperatorPtr A;
   a.FormSystemMatrix(ess, A);

   PointBlockJacobi pbj(fes, *rot, ess);
   Vector diag(fes.GetTrueVSize());
   a.AssembleDiagonal(diag); // N adds zeros: this is sigma M + nu K
   pbj.SetDiagonal(diag);

   GMRESSolver gmres(comm);
   gmres.SetKDim(50);
   gmres.SetRelTol(1e-10);
   gmres.SetAbsTol(0.0);
   gmres.SetMaxIter(3000);
   gmres.SetPrintLevel(0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(pbj);

   auto solve = [&](real_t alpha)
   {
      rot->SetAlpha(alpha);
      rot->UpdateVorticity();
      pbj.UpdateSkew();
      Vector x(b.Size());
      x = 0.0;
      gmres.Mult(b, x);
      return gmres.GetConverged() ? gmres.GetNumIterations() : -1;
   };
   // alpha giving max|alpha omega| dt = 100 for the lagged field w.
   auto alpha100 = [&](const ParGridFunction & w)
   {
      return 100.0 / (MaxVorticity(w, *sr.rotation) * dt);
   };

   B3Counts c;
   c.zero = solve(0.0);
   c.rot1 = solve(alpha100(w1));
   rot->SetLaggedVelocity(w2);
   c.rot2 = solve(alpha100(w2));
   return c;
}

} // namespace

// B3 -- mass-dominated velocity block at max|omega| dt = 100: point-block
// Jacobi keeps GMRES within 3x of the rotation-free count (scalar Jacobi
// fails outright here, Table A2), still after w changes (stale skew data),
// and the partitioned count matches a serial solve of the same system.
TEST(PointBlockJacobi, B3_IterationsAtLargeRotation)
{
   for (int dim : {2, 3})
   {
      SCOPED_TRACE(dim == 2 ? "dim=2" : "dim=3");
      const B3Counts par = RunB3(MPI_COMM_WORLD, dim);
      const B3Counts ser = RunB3(MPI_COMM_SELF, dim);
      if (Mpi::Root())
      {
         std::printf("B3 dim=%d iterations: |omega|dt=0: %d, 100: %d, 100 "
                     "after w change: %d (serial %d/%d/%d)\n", dim, par.zero,
                     par.rot1, par.rot2, ser.zero, ser.rot1, ser.rot2);
      }
      ASSERT_GT(par.zero, 0) << "no convergence at |omega| dt = 0";
      ASSERT_GT(par.rot1, 0) << "no convergence at |omega| dt = 100";
      ASSERT_GT(par.rot2, 0) << "no convergence after changing w";
      EXPECT_LE(par.rot1, 3 * par.zero);
      EXPECT_LE(par.rot2, 3 * par.zero);
      EXPECT_LE(std::abs(par.zero - ser.zero), 2);
      EXPECT_LE(std::abs(par.rot1 - ser.rot1), 2);
      EXPECT_LE(std::abs(par.rot2 - ser.rot2), 2);
   }
}
