// The velocity preconditioner of the Oseen schemes (oseen/oseen_lor_pc.hpp,
// docs/design/oseen_bdf2.md):
//  L1 its grad-div blocks -- the diagonal ones and the upper coupling blocks
//     of the modified-AL sweep, the lower ones their transposes -- add up to
//     the vector grad-div form gamma (div u, div v) assembled on the same
//     LOR space by GradDivIntegrator (the in-repo integrator pinned against
//     MFEM's elasticity): the test that catches a transposed or misplaced
//     block;
//  L2 with each block solved tightly (GMRES), block Jacobi inverts the
//     diagonal blocks and upper Gauss-Seidel the block upper triangle
//     F_i z_i + sum_{j > i} G_ij z_j = r_i, with Dirichlet dofs;
//  L3 two applies agree bitwise (no atomics on a device), and the coupling
//     changes the result.
// Runs on whatever device the test main configures (INCNS_DEVICE=cuda).

#include <gtest/gtest.h>

#include "operators/grad_div_integrator.hpp"
#include "oseen/oseen_lor_pc.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::OseenLorPreconditioner;
using incns::OseenVelocityCoupling;
using incns::RuleBook;

namespace
{

void WindFn(const Vector& x, Vector& w)
{
   const real_t X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   w(0) = 1.0 + 0.5 * std::sin(M_PI * Y) + 0.3 * X * Z;
   w(1) = 0.7 * std::cos(M_PI * X) + 0.2 * Y * Y;
   if (w.Size() == 3) { w(2) = 0.4 + 0.3 * std::sin(M_PI * X * Y) + 0.1 * Z; }
}

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
   Problem(int dim, int n, int k, bool dirichlet)
      : serial(BoxMesh(dim, n, k)), mesh(MPI_COMM_WORLD, serial), fec(k, dim),
        vfes(&mesh, &fec, dim, Ordering::byNODES), sfes(&mesh, &fec),
        wind_gf(&vfes)
   {
      VectorFunctionCoefficient wc(dim, WindFn);
      wind_gf.ProjectCoefficient(wc);
      wind_gf.GetTrueDofs(W);
      if (dirichlet)
      {
         Array<int> ess_bdr(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         vfes.GetEssentialTrueDofs(ess_bdr, ess);
      }
      // gamma varying over the elements: each LOR element must take its
      // parent's.
      gamma.SetSize(mesh.GetNE());
      for (int e = 0; e < mesh.GetNE(); e++) { gamma(e) = 0.2 + 0.1 * ((e + Mpi::WorldRank()) % 3); }
   }
   std::unique_ptr<OseenLorPreconditioner> MakePC(OseenVelocityCoupling coupling,
         double block_rtol)
   {
      OseenLorPreconditioner::Options o;
      o.nu = 1e-2;
      o.coupling = coupling;
      o.block_rtol = block_rtol;
      o.block_maxit = 400;
      auto pc = std::make_unique<OseenLorPreconditioner>(vfes, rules, ess, o);
      pc->SetGradDiv(gamma);
      pc->Update(W, 20.0);
      return pc;
   }
   Mesh serial;
   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace vfes, sfes;
   ParGridFunction wind_gf;
   RuleBook rules;
   Vector W, gamma;
   Array<int> ess;
};

Vector Random(int n, int seed)
{
   Vector x(n);
   x.Randomize(seed + Mpi::WorldRank());
   x.UseDevice(true);
   return x;
}

real_t Norm(const Vector& a)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, a, a));
}

/// The parent element's gamma on the LOR mesh (the test's own map).
class ParentCoefficient : public Coefficient
{
public:
   ParentCoefficient(const Vector& gamma, const Array<int>& parent)
      : gamma_(gamma), parent_(parent) {}
   real_t Eval(ElementTransformation& T, const IntegrationPoint&) override
   {
      return gamma_(parent_[T.ElementNo]);
   }

private:
   const Vector& gamma_;
   const Array<int>& parent_;
};

} // namespace

// L1
TEST(OseenLorPC, L1_GradDivBlocksAssembleTheVectorForm)
{
   for (int dim : {2, 3})
   {
      const int k = (dim == 2) ? 3 : 2;
      Problem p(dim, 3, k, false);
      auto pc = p.MakePC(OseenVelocityCoupling::UpperGaussSeidel, 0.0);
      const int n = p.sfes.GetTrueVSize();

      // The reference: GradDivIntegrator on the vector LOR space, the same
      // vertex rule and the parents' gamma.
      ParLORDiscretization lor(p.sfes, BasisType::GaussLobatto);
      ParFiniteElementSpace& lfes = lor.GetParFESpace();
      ParMesh& lmesh = *lfes.GetParMesh();
      ParFiniteElementSpace lvfes(&lmesh, lfes.FEColl(), dim, Ordering::byNODES);
      const CoarseFineTransformations& cf = lmesh.GetRefinementTransforms();
      Array<int> parent(lmesh.GetNE());
      for (int l = 0; l < lmesh.GetNE(); l++) { parent[l] = cf.embeddings[l].parent; }
      Vector gamma(p.gamma);
      gamma.HostRead();
      ParentCoefficient gc(gamma, parent);
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      ParBilinearForm gv(&lvfes);
      gv.AddDomainIntegrator(new incns::GradDivIntegrator(
                                gc, &p.rules.Get(geom, 1, incns::Rule1D::GaussLobatto)));
      gv.Assemble();
      gv.Finalize();
      std::unique_ptr<HypreParMatrix> Gv(gv.ParallelAssemble());

      Vector x = Random(dim * n, 5), yref(dim * n), y(dim * n);
      yref.UseDevice(true);
      y.UseDevice(true);
      Gv->Mult(x, yref);
      y.Write();
      Vector yi, xj;
      for (int i = 0; i < dim; i++)
      {
         yi.MakeRef(y, i * n, n);
         yi = 0.0;
         for (int j = 0; j < dim; j++)
         {
            xj.MakeRef(x, j * n, n);
            if (i <= j)
            {
               const HypreParMatrix* B = pc->GradDivBlock(i, j);
               ASSERT_NE(B, nullptr) << i << " " << j;
               B->Mult(1.0, xj, 1.0, yi);
            }
            else { pc->GradDivBlock(j, i)->MultTranspose(1.0, xj, 1.0, yi); }
         }
      }
      Vector e(y);
      e -= yref;
      const real_t rel = Norm(e) / Norm(yref);
      EXPECT_LT(rel, 1e-12) << "dim " << dim;
      EXPECT_GT(Norm(yref), 0.0);
   }
}

// L2
TEST(OseenLorPC, L2_TightBlockSolvesInvertTheBlockTriangle)
{
   for (int dim : {2, 3})
   {
      const int k = (dim == 2) ? 3 : 2;
      Problem p(dim, 3, k, true);
      const int n = p.sfes.GetTrueVSize();
      for (auto coupling :
           {
              OseenVelocityCoupling::BlockJacobi,
              OseenVelocityCoupling::UpperGaussSeidel
           })
      {
         const bool gs = coupling == OseenVelocityCoupling::UpperGaussSeidel;
         auto pc = p.MakePC(coupling, 1e-11);
         Vector r = Random(dim * n, 9), z(dim * n), u(dim * n);
         z.UseDevice(true);
         u.UseDevice(true);
         r.HostReadWrite();
         for (int t : p.ess) { r(t) = 0.0; }
         pc->ResetVCycles();
         pc->Mult(r, z);
         EXPECT_GT(pc->VCycles(), dim);
         // u = U z: the diagonal blocks, plus the upper ones for the sweep.
         u.Write();
         Vector ui, zi, zj;
         for (int i = 0; i < dim; i++)
         {
            ui.MakeRef(u, i * n, n);
            zi.MakeRef(z, i * n, n);
            pc->DiagonalBlock(i).Mult(zi, ui);
            for (int j = i + 1; gs && j < dim; j++)
            {
               zj.MakeRef(z, j * n, n);
               pc->GradDivBlock(i, j)->Mult(1.0, zj, 1.0, ui);
            }
         }
         u -= r;
         EXPECT_LT(Norm(u), 1e-8 * Norm(r)) << "dim " << dim << (gs ? " ugs" :
                                            " jacobi");
      }
   }
}

// L3
TEST(OseenLorPC, L3_ApplyIsDeterministic)
{
   Problem p(3, 3, 3, true);
   const int n = p.vfes.GetTrueVSize();
   auto ugs = p.MakePC(OseenVelocityCoupling::UpperGaussSeidel, 0.0);
   auto jac = p.MakePC(OseenVelocityCoupling::BlockJacobi, 0.0);
   Vector r = Random(n, 21), z1(n), z2(n), zj(n);
   z1.UseDevice(true);
   z2.UseDevice(true);
   zj.UseDevice(true);
   ugs->Mult(r, z1);
   ugs->Mult(r, z2);
   jac->Mult(r, zj);
   z1.HostRead();
   z2.HostRead();
   for (int i = 0; i < n; i++) { ASSERT_EQ(z1(i), z2(i)); }
   Vector d(z1);
   d -= zj;
   EXPECT_GT(Norm(d), 1e-6 * Norm(z1)); // the coupling does something
}
