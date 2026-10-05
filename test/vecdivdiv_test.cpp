// VDD -- VectorDivDivIntegrator validation (vecdivdiv_spec.md par.7):
//  7.2 full-assembly element matrices equal ElasticityIntegrator(lambda, mu=0)
//      on CURVED meshes, 2D+3D (locks the math before any PA work);
//  7.1 PA operator action equals the fully assembled action, dim x order x
//      coefficient sweep, curved meshes, <= 1e-12 relative;
//  7.3 the collocated-GLL rule (Q1D == D1D, the SEM configuration);
//  7.4 PA diagonal (sum-factorized) equals the assembled matrix diagonal,
//      and equals the direct reference/fallback form element by element;
//  7.5 transpose apply equals primal apply (symmetric forwarding);
//  EA  component blocks, element-assembled on a scalar space and merged,
//      equal the assembled vector operator.

#include <gtest/gtest.h>

#include "operators/vecdivdiv_integrator.hpp"
#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace mfem;
using incns::Rule1D;
using incns::RuleBook;
using incns::VectorDivDivComponentIntegrator;
using incns::VectorDivDivIntegrator;

namespace
{

// Curved, non-affine tensor mesh: Cartesian box, high-order nodes, then a
// smooth invertible perturbation so the Jacobian genuinely varies per point
// (the adjugate/detJ path is exercised, not just the affine special case).
Mesh MakeCurvedMesh(int dim, int n, int geom_order)
{
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                       1.0, 1.0)
               : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON,
                                       1.0, 1.0, 1.0);
   mesh.SetCurvature(geom_order);
   mesh.Transform([](const Vector & x, Vector & p)
   {
      p = x;
      p(0) += 0.05 * std::sin(M_PI * x(0)) * std::cos(M_PI * x(1));
      p(1) += 0.04 * std::cos(M_PI * x(0)) * std::sin(M_PI * x(1));
      if (x.Size() == 3) { p(2) += 0.03 * std::sin(M_PI * x(2)); }
   });
   return mesh;
}

double Coef(const Vector& x)
{
   double v = 1.0 + 0.5 * std::sin(2.0 * x(0)) * std::cos(x(1));
   if (x.Size() == 3) { v += 0.25 * x(2); }
   return v;
}

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

} // namespace

// 7.2 -- element matrices equal the lambda-part of elasticity, forced same
// rule, curved 2D and 3D. Pins conventions (byNODES blocks, weights, adjugate).
TEST(VecDivDiv, ElmatMatchesElasticity)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const int p = (dim == 2) ? 3 : 2;
      Mesh mesh = MakeCurvedMesh(dim, 2, p);
      H1_FECollection fec(p, dim);
      FiniteElementSpace fes(&mesh, &fec); // scalar; elmat is per-element
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const IntegrationRule& ir = rules.Get(geom, 2 * p + dim - 1);

      FunctionCoefficient q(Coef);
      ConstantCoefficient zero(0.0);
      VectorDivDivIntegrator vdd(q);
      ElasticityIntegrator ela(q, zero); // lambda = q, mu = 0
      vdd.SetIntRule(&ir);
      ela.SetIntRule(&ir);

      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         DenseMatrix m_vdd, m_ela;
         vdd.AssembleElementMatrix(*fes.GetFE(e),
                                   *mesh.GetElementTransformation(e), m_vdd);
         ela.AssembleElementMatrix(*fes.GetFE(e),
                                   *mesh.GetElementTransformation(e), m_ela);
         m_ela -= m_vdd;
         EXPECT_LE(m_ela.MaxMaxNorm(), 1e-12 * m_vdd.MaxMaxNorm())
               << "dim=" << dim << " e=" << e;
      }
   }
}

// 7.1 + 7.3 + 7.5 -- PA action vs fully assembled action on curved parallel
// meshes: dim {2,3} x order sweep x coefficient {none, constant, function},
// with both the standard GL rule and the collocated GLL rule (Q1D == D1D).
TEST(VecDivDiv, PaMatchesFullAssembly)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int pmax = (dim == 2) ? 4 : 3;
      for (int p = 1; p <= pmax; ++p)
      {
         Mesh serial = MakeCurvedMesh(dim, (dim == 2) ? 3 : 2, p);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);

         FunctionCoefficient qfun(Coef);
         ConstantCoefficient qconst(0.7);
         for (int cv = 0; cv < 3; ++cv)
         {
            for (int gll = 0; gll < 2; ++gll)
            {
               // GL default rule vs collocated GLL (2p-1 -> p+1 = D1D points).
               const IntegrationRule& ir =
                  gll ? rules.Get(geom, 2 * p - 1, Rule1D::GaussLobatto)
                  : rules.Get(geom, 2 * p + dim - 1);
               auto make = [&]() -> VectorDivDivIntegrator *
               {
                  VectorDivDivIntegrator* i =
                  (cv == 0) ? new VectorDivDivIntegrator()
                  : (cv == 1) ? new VectorDivDivIntegrator(qconst)
                  : new VectorDivDivIntegrator(qfun);
                  i->SetIntRule(&ir);
                  return i;
               };

               ParBilinearForm pa(&fes), fa(&fes);
               pa.AddDomainIntegrator(make());
               fa.AddDomainIntegrator(make());
               pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
               pa.Assemble();
               fa.Assemble();
               fa.Finalize();
               Array<int> empty;
               OperatorPtr Apa, Afa;
               pa.FormSystemMatrix(empty, Apa);
               fa.FormSystemMatrix(empty, Afa);

               Vector x(fes.GetTrueVSize()), ypa(x.Size()), yfa(x.Size()),
                      yt(x.Size());
               x.Randomize(7 + p + cv);
               Apa->Mult(x, ypa);
               Afa->Mult(x, yfa);
               yfa -= ypa;
               EXPECT_LE(Norm(yfa), 1e-12 * Norm(ypa))
                     << "dim=" << dim << " p=" << p << " coeff=" << cv
                     << " gll=" << gll;

               // 7.5: symmetric transpose forwarding through the PA operator.
               Apa->MultTranspose(x, yt);
               yt -= ypa;
               EXPECT_LE(Norm(yt), 1e-13 * Norm(ypa));
            }
         }
      }
   }
}

// 7.4 -- PA diagonal (sum-factorized path) equals the assembled matrix
// diagonal: curved meshes, function coefficient, GL and collocated GLL rules,
// dim {2,3} x p sweep.
TEST(VecDivDiv, DiagonalMatchesAssembled)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int pmax = (dim == 2) ? 4 : 3;
      for (int p = 1; p <= pmax; ++p)
         for (int gll = 0; gll < 2; ++gll)
         {
            Mesh serial = MakeCurvedMesh(dim, 2, p);
            ParMesh mesh(MPI_COMM_WORLD, serial);
            H1_FECollection fec(p, dim);
            ParFiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
            const IntegrationRule& ir =
               gll ? rules.Get(geom, 2 * p - 1, Rule1D::GaussLobatto)
               : rules.Get(geom, 2 * p + dim - 1);
            FunctionCoefficient q(Coef);

            ParBilinearForm pa(&fes), fa(&fes);
            auto* ipa = new VectorDivDivIntegrator(q);
            auto* ifa = new VectorDivDivIntegrator(q);
            ipa->SetIntRule(&ir);
            ifa->SetIntRule(&ir);
            pa.AddDomainIntegrator(ipa);
            fa.AddDomainIntegrator(ifa);
            pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            pa.Assemble();
            fa.Assemble();
            fa.Finalize();

            Vector d_pa(fes.GetTrueVSize());
            pa.AssembleDiagonal(d_pa);

            Array<int> empty;
            OperatorPtr Afa;
            fa.FormSystemMatrix(empty, Afa);
            Vector d_fa(fes.GetTrueVSize());
            Afa.As<HypreParMatrix>()->GetDiag(d_fa);

            d_fa -= d_pa;
            EXPECT_LE(Norm(d_fa), 1e-12 * Norm(d_pa))
                  << "dim=" << dim << " p=" << p << " gll=" << gll;
         }
   }
}

// The sum-factorized diagonal equals the direct one (the fallback beyond the
// shared-tile limits, kept as the reference) element by element, on the
// E-vector, before any assembly to true dofs can average a defect away.
namespace
{
struct DirectDiagonalProbe : VectorDivDivIntegrator
{
   using VectorDivDivIntegrator::VectorDivDivIntegrator;
   using VectorDivDivIntegrator::AssembleDiagonalPADirect;
};
} // namespace

TEST(VecDivDiv, SumFactorizedDiagonalMatchesDirect)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int pmax = (dim == 2) ? 5 : 4;
      for (int p = 1; p <= pmax; ++p)
         for (int gll = 0; gll < 2; ++gll)
         {
            Mesh mesh = MakeCurvedMesh(dim, 2, p);
            H1_FECollection fec(p, dim);
            FiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
            const IntegrationRule& ir =
               gll ? rules.Get(geom, 2 * p - 1, Rule1D::GaussLobatto)
               : rules.Get(geom, 2 * p + dim - 1);
            FunctionCoefficient q(Coef);
            DirectDiagonalProbe vdd(q);
            vdd.SetIntRule(&ir);
            vdd.AssemblePA(fes);

            const int esize =
               fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)->Height();
            Vector d_sf(esize), d_dir(esize);
            d_sf = 0.0;
            d_dir = 0.0;
            vdd.AssembleDiagonalPA(d_sf);
            vdd.AssembleDiagonalPADirect(d_dir);
            d_dir -= d_sf;
            EXPECT_LE(d_dir.Normlinf(), 1e-13 * d_sf.Normlinf())
                  << "dim=" << dim << " p=" << p << " gll=" << gll;
         }
   }
}

// EA -- the dim x dim component blocks, each element-assembled on a SCALAR
// space (AssemblyLevel::FULL, i.e. AssembleEA -> sparse matrix) and merged
// with HypreParMatrixFromBlocks, equal the legacy-assembled vector operator.
// The off-diagonal blocks are not symmetric, so this pins the EA row/column
// convention and lexicographic dof order; at np > 1 it pins that the merged
// block ordering is the byNODES vector true-dof ordering the LOR-AMG merge
// relies on. Also checks each block's AssemblyLevel::ELEMENT action against
// its FULL matrix. Q1 is the LOR use; p = 2, 3 guard the tensor indexing.
TEST(VecDivDiv, ComponentEaMatchesVectorAssembly)
{
   RuleBook rules;
   for (int dim : {2, 3})
   {
      const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
      const int pmax = (dim == 2) ? 3 : 2;
      for (int p = 1; p <= pmax; ++p)
      {
         Mesh serial = MakeCurvedMesh(dim, (dim == 2) ? 3 : 2, p);
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace vfes(&mesh, &fec, dim, Ordering::byNODES);
         ParFiniteElementSpace sfes(&mesh, &fec);
         const IntegrationRule& ir = rules.Get(geom, 2 * p + dim - 1);
         FunctionCoefficient q(Coef);
         Array<int> empty;

         // Reference: the vector operator, legacy full assembly.
         auto* ref_integ = new VectorDivDivIntegrator(q);
         ref_integ->SetIntRule(&ir);
         ParBilinearForm ref(&vfes);
         ref.AddDomainIntegrator(ref_integ);
         ref.Assemble();
         ref.Finalize();
         OperatorPtr K_ref;
         ref.FormSystemMatrix(empty, K_ref);

         // Blocks: the parent supplies the coefficient and the rule.
         VectorDivDivIntegrator parent(q);
         parent.SetIntRule(&ir);
         // Lifetimes (both bit here): the handle from FormSystemMatrix owns
         // each block matrix and OperatorHandle copies do NOT, so the handle
         // vector must never reallocate; and at np = 1 a FULL-assembled matrix
         // borrows its form's storage, so the forms must outlive the merge.
         std::vector<std::unique_ptr<ParBilinearForm>> fa_forms;
         std::vector<OperatorPtr> K_blocks;
         K_blocks.reserve(dim * dim);
         Array2D<const HypreParMatrix*> blocks(dim, dim);
         for (int i = 0; i < dim; ++i)
         {
            for (int j = 0; j < dim; ++j)
            {
               fa_forms.push_back(std::make_unique<ParBilinearForm>(&sfes));
               ParBilinearForm& fa = *fa_forms.back();
               ParBilinearForm ea(&sfes);
               fa.SetAssemblyLevel(AssemblyLevel::FULL);
               ea.SetAssemblyLevel(AssemblyLevel::ELEMENT);
               fa.AddDomainIntegrator(
                    new VectorDivDivComponentIntegrator(parent, i, j));
               ea.AddDomainIntegrator(
                    new VectorDivDivComponentIntegrator(parent, i, j));
               fa.Assemble();
               ea.Assemble();
               K_blocks.emplace_back(Operator::Hypre_ParCSR);
               fa.FormSystemMatrix(empty, K_blocks.back());
               const HypreParMatrix* Kij = K_blocks.back().As<HypreParMatrix>();
               ASSERT_NE(Kij, nullptr) << "FULL did not yield a HypreParMatrix";
               blocks(i, j) = Kij;

               // ELEMENT-level action (EA data applied per element) vs the
               // FULL matrix built from the same EA data.
               OperatorPtr A_ea;
               ea.FormSystemMatrix(empty, A_ea);
               Vector x(sfes.GetTrueVSize()), y_ea(x.Size()), y_fa(x.Size());
               x.Randomize(11 + 3 * i + j);
               A_ea->Mult(x, y_ea);
               Kij->Mult(x, y_fa);
               y_ea -= y_fa;
               EXPECT_LE(Norm(y_ea), 1e-12 * Norm(y_fa))
                     << "ELEMENT vs FULL: dim=" << dim << " p=" << p
                     << " block=(" << i << "," << j << ")";
            }
         }
         std::unique_ptr<HypreParMatrix> K(HypreParMatrixFromBlocks(blocks));
         ASSERT_EQ(K->Height(), K_ref->Height());

         Vector x(vfes.GetTrueVSize()), y(x.Size()), y_ref(x.Size());
         x.Randomize(5 + p);
         K->Mult(x, y);
         K_ref->Mult(x, y_ref);
         y -= y_ref;
         EXPECT_LE(Norm(y), 1e-12 * Norm(y_ref))
               << "merged blocks vs vector: dim=" << dim << " p=" << p;
      }
   }
}
