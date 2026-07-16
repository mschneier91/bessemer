// CC.1 -- mass-inverse strategies (T1f + validation, SPEC par.6.1/par.10):
//
//  * Collocated GLL vector mass: Auto resolves DiagDirect; the application is
//    bitwise the fused reciprocal-diagonal multiply; NO solver object exists on
//    that path; M o M^{-1} = identity to machine precision (diagonal => exact).
//  * Gauss-Legendre (non-collocated) pressure mass: Auto resolves Chebyshev;
//    the application is symmetric (CG legality) and a genuine approximate
//    inverse whose quality improves with the polynomial order.
//  * Config validation: DiagDirect on a non-diagonal mass throws; AbsLumped
//    throws (simplex fallback with no callers -- quad/hex only).

#include <gtest/gtest.h>

#include "mesh/periodic_box.hpp"
#include "precond/mass_inverse.hpp"
#include "quadrature/rule_book.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <memory>
#include <stdexcept>

using namespace mfem;
using incns::BoxSpec;
using incns::MakeBoxMesh;
using incns::MassInverse;
using incns::MassInvType;
using incns::MixedSpaces;
using incns::RuleBook;

namespace
{

// Small non-periodic unit-square Q3/Q2 setup shared by the tests.
struct SmallCase
{
   ParMesh mesh;
   std::unique_ptr<MixedSpaces> spaces;
   RuleBook rules;

   SmallCase()
   {
      BoxSpec s;
      s.dim = 2;
      s.num_elems = {3, 3, 0};
      s.lengths = {1.0, 1.0};
      s.periodic = {false, false, false};
      Mesh serial = MakeBoxMesh(s);
      mesh = ParMesh(MPI_COMM_WORLD, serial);
      spaces = std::make_unique<MixedSpaces>(mesh, 3, 2);
   }
};

// PA-assemble a mass form with the given rule; returns the true-dof operator
// and its diagonal through the out-params (forms kept alive by the caller).
void AssembleMass(ParBilinearForm& form, BilinearFormIntegrator* integ,
                  OperatorPtr& op, Vector& diag)
{
   form.AddDomainIntegrator(integ);
   form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form.Assemble();
   Array<int> empty;
   form.FormSystemMatrix(empty, op);
   diag.SetSize(form.ParFESpace()->GetTrueVSize());
   form.AssembleDiagonal(diag);
}

} // namespace

TEST(MassInverse, CollocatedGllResolvesDiagDirect)
{
   SmallCase su;
   ParBilinearForm m(&su.spaces->Velocity());
   auto* mi = new VectorMassIntegrator;
   mi->SetIntRule(&su.rules.CollocatedMass(Geometry::SQUARE, 3));
   OperatorPtr M;
   Vector diag;
   AssembleMass(m, mi, M, diag);

   MassInverse minv(*M.Ptr(), diag, MPI_COMM_WORLD, MassInvType::Auto);
   EXPECT_EQ(minv.Resolved(), MassInvType::DiagDirect);
   EXPECT_FALSE(minv.HasSolverObject()); // T1f: bare vector multiply, no solver

   // Bitwise: application == the explicit fused multiply y_i = (1/d_i) * x_i.
   Vector x(diag.Size()), y(diag.Size());
   x.Randomize(7);
   minv.Mult(x, y);
   const double* xd = x.HostRead();
   const double* dd = diag.HostRead();
   const double* yd = y.HostRead();
   for (int i = 0; i < diag.Size(); ++i)
   {
      const double dinv = 1.0 / dd[i];
      EXPECT_EQ(yd[i], dinv * xd[i]) << "not the bare multiply at i=" << i;
   }

   // Exact inverse: M(Minv x) = x to machine precision (diagonal mass).
   Vector z(diag.Size());
   M->Mult(y, z);
   z -= x;
   const double rel = std::sqrt(InnerProduct(MPI_COMM_WORLD, z, z) /
                                InnerProduct(MPI_COMM_WORLD, x, x));
   EXPECT_LE(rel, 1e-13);
}

TEST(MassInverse, GaussLegendreResolvesChebyshev)
{
   SmallCase su;
   ParBilinearForm m(&su.spaces->Pressure());
   auto* mi = new MassIntegrator;
   mi->SetIntRule(&su.rules.Get(Geometry::SQUARE, 4)); // 2k, k_p = 2: GL rule
   OperatorPtr M;
   Vector diag;
   AssembleMass(m, mi, M, diag);

   MassInverse minv(*M.Ptr(), diag, MPI_COMM_WORLD, MassInvType::Auto,
                    /*cheb_order=*/4);
   EXPECT_EQ(minv.Resolved(), MassInvType::Chebyshev);
   EXPECT_TRUE(minv.HasSolverObject());

   // Symmetry (CG legality): <Minv x, y> == <x, Minv y> to tight relative tol.
   Vector x(diag.Size()), y(diag.Size()), mx(diag.Size()), my(diag.Size());
   x.Randomize(11);
   y.Randomize(13);
   minv.Mult(x, mx);
   minv.Mult(y, my);
   const double a = InnerProduct(MPI_COMM_WORLD, mx, y);
   const double b = InnerProduct(MPI_COMM_WORLD, x, my);
   EXPECT_NEAR(a, b, 1e-12 * std::abs(a));

   // Genuine approximate inverse: ||M(Minv x) - x|| well below ||x||, and the
   // order-5 application strictly beats order-2 (fixed-order polynomial; MFEM
   // implements Chebyshev orders 1-5).
   auto residual = [&](const MassInverse & inv)
   {
      Vector w(diag.Size()), r(diag.Size());
      inv.Mult(x, w);
      M->Mult(w, r);
      r -= x;
      return std::sqrt(InnerProduct(MPI_COMM_WORLD, r, r) /
                       InnerProduct(MPI_COMM_WORLD, x, x));
   };
   MassInverse minv2(*M.Ptr(), diag, MPI_COMM_WORLD, MassInvType::Chebyshev,
                     /*cheb_order=*/2);
   MassInverse minv5(*M.Ptr(), diag, MPI_COMM_WORLD, MassInvType::Chebyshev,
                     /*cheb_order=*/5);
   const double r2 = residual(minv2);
   const double r4 = residual(minv);
   const double r5 = residual(minv5);
   if (Mpi::Root())
   {
      mfem::out << "[mass_inverse] chebyshev residual: k=2 " << r2 << "  k=4 "
                << r4 << "  k=5 " << r5 << std::endl;
   }
   EXPECT_LT(r4, 0.6);
   EXPECT_LT(r5, r2);

   // Out-of-range Chebyshev order is a config error, not an MFEM abort.
   EXPECT_THROW(MassInverse(*M.Ptr(), diag, MPI_COMM_WORLD,
                            MassInvType::Chebyshev, /*cheb_order=*/8),
                std::invalid_argument);
}

TEST(MassInverse, DiagDirectOnNonDiagonalThrows)
{
   SmallCase su;
   ParBilinearForm m(&su.spaces->Pressure());
   auto* mi = new MassIntegrator;
   mi->SetIntRule(&su.rules.Get(Geometry::SQUARE, 4));
   OperatorPtr M;
   Vector diag;
   AssembleMass(m, mi, M, diag);

   EXPECT_THROW(MassInverse(*M.Ptr(), diag, MPI_COMM_WORLD,
                            MassInvType::DiagDirect),
                std::invalid_argument);
}

TEST(MassInverse, AbsLumpedRejected)
{
   SmallCase su;
   ParBilinearForm m(&su.spaces->Pressure());
   auto* mi = new MassIntegrator;
   mi->SetIntRule(&su.rules.Get(Geometry::SQUARE, 4));
   OperatorPtr M;
   Vector diag;
   AssembleMass(m, mi, M, diag);

   EXPECT_THROW(MassInverse(*M.Ptr(), diag, MPI_COMM_WORLD,
                            MassInvType::AbsLumped),
                std::invalid_argument);
}
