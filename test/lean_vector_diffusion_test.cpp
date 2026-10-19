// LeanVectorDiffusionIntegrator (operators/lean_vector_diffusion): compact
// partial-assembly storage and a stack-free diagonal, against MFEM's
// VectorDiffusionIntegrator. 2D and 3D, orders 2, 3, 5, 6 and 7, a deformed mesh,
// byNODES and byVDIM, a variable and a constant coefficient.
//  L1 Storage::Mfem: the apply is MFEM's, bit for bit; the diagonal equals
//     MFEM's to roundoff.
//  L2 Storage::Compact: the apply and the diagonal equal MFEM's to roundoff,
//     and the stored data is dim (dim + 1) / 2 numbers per point against
//     MFEM's dim^3 (6 against 27 in 3D).
// The measured differences are printed. Self-contained: MFEM and the
// integrator.

#include <gtest/gtest.h>

#include "operators/grad_div_integrator.hpp"
#include "operators/lean_vector_diffusion.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>

using namespace mfem;
using incns::LeanVectorDiffusionIntegrator;
using Storage = LeanVectorDiffusionIntegrator::Storage;

namespace
{

void Deform(const Vector& x, Vector& y)
{
   y = x;
   y(0) += 0.08 * std::sin(2.3 * x(1));
   y(1) += 0.06 * x(0) * x(0);
   if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
}

double NuFn(const Vector& x)
{
   return 0.7 + 0.5 * x(0) + 0.3 * x(1) * x(1);
}

double Norm(const Vector& v)
{
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
}

double MaxAbs(const Vector& v)
{
   double m = v.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

void Check(int dim, int order, Ordering::Type ordering, Storage storage,
           bool constant_coeff)
{
   Mesh serial = (dim == 2)
                 ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                 : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
   serial.Transform(Deform);
   ParMesh mesh(MPI_COMM_WORLD, serial);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim, ordering);
   FunctionCoefficient nu_var(NuFn);
   ConstantCoefficient nu_const(0.37);
   Coefficient& nu = constant_coeff ? static_cast<Coefficient&>(nu_const)
                     : static_cast<Coefficient&>(nu_var);
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = IntRules.Get(geom, 2 * order + dim - 1);

   ParBilinearForm lean(&fes), ref(&fes);
   auto* li = new LeanVectorDiffusionIntegrator(nu, storage);
   li->SetIntRule(&ir);
   lean.AddDomainIntegrator(li);
   auto* ri = new VectorDiffusionIntegrator(nu);
   ri->SetIntRule(&ir);
   ref.AddDomainIntegrator(ri);
   lean.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   ref.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   lean.Assemble();
   ref.Assemble();

   EXPECT_EQ(li->Compact(), storage == Storage::Compact);
   if (storage == Storage::Compact)
   {
      const long long nq = ir.GetNPoints(), ne = mesh.GetNE();
      EXPECT_EQ(li->PADataSize(), nq * (dim * (dim + 1) / 2) * ne);
   }

   const int n = fes.GetTrueVSize();
   Vector d_lean(n), d_ref(n);
   lean.AssembleDiagonal(d_lean);
   ref.AssembleDiagonal(d_ref);
   const double d_scale = MaxAbs(d_ref);
   EXPECT_GT(d_scale, 0.0);
   d_lean -= d_ref;
   const double d_diff = MaxAbs(d_lean) / d_scale;

   Array<int> no_ess;
   OperatorPtr A_lean, A_ref;
   lean.FormSystemMatrix(no_ess, A_lean);
   ref.FormSystemMatrix(no_ess, A_ref);
   Vector x(n), y_lean(n), y_ref(n);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   x.Randomize(3 + rank);
   A_lean->Mult(x, y_lean);
   A_ref->Mult(x, y_ref);
   const double y_scale = Norm(y_ref);
   y_lean -= y_ref;
   const double y_diff = Norm(y_lean) / y_scale;

   if (Mpi::Root())
   {
      mfem::out << "[lean diffusion] " << (storage == Storage::Compact ?
                                           "compact" : "mfem   ")
                << " dim " << dim << " order " << order
                << (ordering == Ordering::byVDIM ? " byVDIM " : " byNODES")
                << (constant_coeff ? " const" : " var  ")
                << ": diagonal max diff / max entry " << d_diff
                << ", apply relative l2 diff " << y_diff << std::endl;
   }
   // The diagonal: the same sum in another order (direct against
   // sum-factorized), measured 3e-16 to 8e-16 (H100, 2026-10-13).
   EXPECT_LT(d_diff, 5e-15);
   if (storage == Storage::Mfem)
   {
      EXPECT_EQ(y_diff, 0.0); // MFEM's kernel on MFEM's data
   }
   else
   {
      // MFEM's kernel arithmetic on MFEM's values, stored once: equal up to
      // the compiler's contraction choices.
      EXPECT_LT(y_diff, 1e-14);
   }
}

} // namespace

TEST(LeanVectorDiffusion, L1_MfemStorageIsMfems)
{
   for (const int dim : {2, 3})
      for (const int order : {2, 3})
         for (const Ordering::Type ordering : {Ordering::byNODES, Ordering::byVDIM})
         {
            Check(dim, order, ordering, Storage::Mfem, false);
         }
}

TEST(LeanVectorDiffusion, L2_CompactStorageEqualsMfem)
{
   for (const int dim : {2, 3})
      for (const int order : {2, 3, 5, 6, 7})
         for (const Ordering::Type ordering : {Ordering::byNODES, Ordering::byVDIM})
            for (const bool constant : {false, true})
            {
               Check(dim, order, ordering, Storage::Compact, constant);
            }
}

// L3 FuseGradDiv: nu K + gamma (div u, div v) applied by one kernel equals the
//    two integrators applied separately, to roundoff (the sums are formed at
//    the quadrature points instead of after the contraction), and the
//    grad-div integrator's own apply is switched off. 2D and 3D, orders 2, 3,
//    5, 6 and 7, deformed mesh, variable gamma; nu variable (L3) or constant
//    (L4: on a device at orders 6 and 7 the by-lines kernel then forms the
//    compact data from J).
void CheckFusedGradDiv(const bool constant_nu)
{
   for (const int dim : {2, 3})
   {
      for (const int k : {2, 3, 5, 6, 7})
      {
         SCOPED_TRACE(testing::Message() << "dim " << dim << " order " << k
                      << (constant_nu ? " constant nu" : " variable nu"));
         Mesh serial = (dim == 2)
                       ? Mesh::MakeCartesian2D(4, 3, Element::QUADRILATERAL, true)
                       : Mesh::MakeCartesian3D(3, 2, 2, Element::HEXAHEDRON);
         serial.Transform([](const Vector & x, Vector & y)
         {
            y = x;
            y(0) += 0.08 * std::sin(2.3 * x(1));
            y(1) += 0.06 * x(0) * x(0);
            if (x.Size() == 3) { y(2) += 0.05 * x(0) * x(1); }
         });
         ParMesh mesh(MPI_COMM_WORLD, serial);
         H1_FECollection fec(k, dim);
         ParFiniteElementSpace fes(&mesh, &fec, dim);
         const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
         const IntegrationRule& ir = IntRules.Get(geom, 2 * k + dim - 1);
         ConstantCoefficient nu_c(0.2);
         FunctionCoefficient nu_f([](const Vector & x) { return 0.2 + 0.1 * x(0); });
         Coefficient& nu = constant_nu ? static_cast<Coefficient&>(nu_c)
                           : static_cast<Coefficient&>(nu_f);
         FunctionCoefficient gamma([](const Vector & x)
         {
            return 0.7 + 0.5 * x(0) + 0.3 * x(1) * x(1);
         });
         int rank = 0;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
         auto apply = [&](bool fuse, Vector & y)
         {
            ParBilinearForm a(&fes);
            auto* kd = new incns::LeanVectorDiffusionIntegrator(nu);
            kd->SetIntRule(&ir);
            auto* gd = new incns::GradDivIntegrator(gamma, &ir);
            a.AddDomainIntegrator(kd);
            a.AddDomainIntegrator(gd);
            a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            a.Assemble();
            if (fuse)
            {
               EXPECT_TRUE(kd->FuseGradDiv(*gd));
               EXPECT_TRUE(kd->FusedGradDiv());
               EXPECT_TRUE(gd->ApplyFused());
            }
            Vector x(fes.GetVSize());
            x.Randomize(11 + rank);
            x.UseDevice(true);
            y.SetSize(fes.GetVSize());
            y.UseDevice(true);
            a.Mult(x, y);
            y.HostRead();
         };
         Vector separate, fused;
         apply(false, separate);
         apply(true, fused);
         Vector d(fused);
         d -= separate;
         double diff = d.Normlinf(), size = separate.Normlinf();
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         if (Mpi::Root())
         {
            mfem::out << "[fused grad-div] dim " << dim << " order " << k
                      << (constant_nu ? ", constant nu" : ", variable nu")
                      << ": max difference / max entry " << diff / size << std::endl;
         }
         EXPECT_GT(size, 0.0);
         EXPECT_LE(diff, 1e-13 * size);
      }
   }
}

TEST(LeanVectorDiffusion, L3_FusedGradDivMatchesSeparateApplies)
{
   CheckFusedGradDiv(false);
}

TEST(LeanVectorDiffusion, L4_FusedGradDivConstantNuMatchesSeparateApplies)
{
   CheckFusedGradDiv(true);
}
