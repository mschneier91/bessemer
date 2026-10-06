// VectorRotationalConvectionIntegrator, N(w): (u, v) -> alpha ((curl w) x u, v).
// Two tests, described above each TEST: right values (MatchesMfemReference)
// and correct GPU execution (SpecializedAndDeterministic). Both run on
// whatever device the test main configures, so INCNS_DEVICE=cuda puts the
// partial-assembly kernels on a GPU. Self-contained on purpose -- MFEM and
// the integrator header only.

#include <gtest/gtest.h>

#include "operators/rotational_convection.hpp"
#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <string>

using namespace mfem;
using incns::VectorRotationalConvectionIntegrator;

namespace
{

const real_t kAlpha = 1.7;

// Two smooth lagged velocities whose curl is nonzero in every component.
void w1_fn(const Vector& x, Vector& u)
{
   const real_t X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   u(0) = std::sin(M_PI * Y) + 0.5 * Z * Z;
   u(1) = std::cos(M_PI * X) * (1.0 + Z);
   if (u.Size() == 3) { u(2) = std::sin(M_PI * X * Y); }
}

void w2_fn(const Vector& x, Vector& u)
{
   const real_t X = x(0), Y = x(1), Z = (x.Size() == 3) ? x(2) : 0.0;
   u(0) = X * Y * Y - Z;
   u(1) = std::exp(X) * std::cos(Z);
   if (u.Size() == 3) { u(2) = Y * Z + std::sin(X); }
}

// Interpolates f into w through the true dofs, so shared dofs agree across
// ranks.
void SetVelocity(ParGridFunction& w, void (*f)(const Vector&, Vector&))
{
   VectorFunctionCoefficient c(w.ParFESpace()->GetVDim(), f);
   w.ProjectCoefficient(c);
   Vector W;
   w.GetTrueDofs(W);
   w.SetFromTrueDofs(W);
}

// A 3x3 (2D) or 2x2x2 (3D) box with order-p curved elements (the Jacobian
// varies from point to point), partitioned, with the order-p vector H1 space,
// the lagged velocity w = w1 and the order-3p Gauss-Legendre rule.
struct Problem
{
   Problem(int dim, int p)
      : serial(CurvedMesh(dim, p)), mesh(MPI_COMM_WORLD, serial), fec(p, dim),
        fes(&mesh, &fec, dim, Ordering::byNODES), w(&fes),
        ir(IntRules.Get(dim == 3 ? Geometry::CUBE : Geometry::SQUARE, 3 * p))
   {
      SetVelocity(w, w1_fn);
   }

   static Mesh CurvedMesh(int dim, int p)
   {
      Mesh m = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
      m.SetCurvature(p);
      m.Transform([](const Vector & x, Vector & y)
      {
         y = x;
         y(0) += 0.1 * std::sin(M_PI * x(1));
         y(1) += 0.1 * std::sin(M_PI * x(0));
         if (x.Size() == 3) { y(2) += 0.1 * std::sin(M_PI * x(0)); }
      });
      return m;
   }

   Mesh serial;
   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   ParGridFunction w;
   const IntegrationRule& ir;
};

// alpha [curl w]_x as a matrix coefficient, so that VectorMassIntegrator gives
// alpha ((curl w) x u, v) with MFEM code only. [omega]_x u = omega x u; in 2D
// omega is the scalar vorticity and omega x u = omega (-u_1, u_0). curl w comes
// from MFEM's GridFunction::GetCurl.
class AlphaCurlCross : public MatrixCoefficient
{
   const GridFunction& w_;
   Vector o_;

public:
   AlphaCurlCross(int dim, const GridFunction& w)
      : MatrixCoefficient(dim), w_(w) { }

   void Eval(DenseMatrix& K, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      T.SetIntPoint(&ip);
      w_.GetCurl(T, o_);
      K.SetSize(GetHeight());
      K = 0.0;
      if (GetHeight() == 2)
      {
         K(0, 1) = -kAlpha * o_(0);
         K(1, 0) = kAlpha * o_(0);
      }
      else
      {
         K(0, 1) = -kAlpha * o_(2);
         K(0, 2) = kAlpha * o_(1);
         K(1, 0) = kAlpha * o_(2);
         K(1, 2) = -kAlpha * o_(0);
         K(2, 0) = -kAlpha * o_(1);
         K(2, 1) = kAlpha * o_(0);
      }
   }
};

// The PA form with this integrator as its only term (the device kernels).
std::unique_ptr<ParBilinearForm> PaForm(Problem& pb,
                                        VectorRotationalConvectionIntegrator*& rot)
{
   auto form = std::make_unique<ParBilinearForm>(&pb.fes);
   rot = new VectorRotationalConvectionIntegrator(pb.w, kAlpha);
   rot->SetIntRule(&pb.ir);
   form->AddDomainIntegrator(rot);
   form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form->Assemble();
   return form;
}

// y = A x with A the form's true-dof operator (no essential dofs).
Vector Apply(ParBilinearForm& form, const Vector& x)
{
   Array<int> empty;
   OperatorPtr A;
   form.FormSystemMatrix(empty, A);
   Vector y(x.Size());
   A->Mult(x, y);
   return y;
}

// ||a - b|| / ||b||, global over ranks.
real_t RelDiff(Vector a, const Vector& b)
{
   a -= b;
   return std::sqrt(InnerProduct(MPI_COMM_WORLD, a, a) /
                    InnerProduct(MPI_COMM_WORLD, b, b));
}

std::string Label(int dim, int p)
{
   return "dim=" + std::to_string(dim) + " p=" + std::to_string(p);
}

} // namespace

// Right values: the integrator computes alpha ((curl w) x u, v).
//
// For dim = 2, 3 and p = 1..3, at the order-3p Gauss-Legendre rule the solver
// uses, on a curved 3x3 (2D) or 2x2x2 (3D) mesh split across the ranks, with
// x a random true-dof vector:
//  1. Reference: y_ref = R x, with R assembled by MFEM's own
//     VectorMassIntegrator using the matrix coefficient alpha [curl w]_x
//     (curl w from GridFunction::GetCurl). It shares no code with the
//     integrator.
//  2. Partial assembly: N x from the integrator's PA kernels -- the code that
//     runs on the GPU -- equals y_ref to 1e-12 (relative, global norm).
//  3. Full assembly: the integrator's legacy element matrices (the route
//     MFEM's LOR assembly takes), assembled into a matrix, also give y_ref.
//  4. Update: w is replaced by a second field and UpdateVorticity() is called
//     on the already-assembled PA operator, as every time step does; N x
//     equals the reference rebuilt for the new w.
// Why p <= 3: steps 1 and 3 assemble dense element matrices, whose cost grows
// like p^9 in 3D (p = 4 and 5 took 0.5 s and 2.3 s on a desktop CPU, and far
// longer on a GPU node), while the PA applies take under a millisecond. The
// p = 4, 5 kernels still run in SpecializedAndDeterministic.
TEST(RotationalConvection, MatchesMfemReference)
{
   for (int dim : {2, 3})
   {
      for (int p = 1; p <= 3; ++p)
      {
         SCOPED_TRACE(Label(dim, p));
         Problem pb(dim, p);
         VectorRotationalConvectionIntegrator* rot = nullptr;
         std::unique_ptr<ParBilinearForm> pa = PaForm(pb, rot);
         Vector x(pb.fes.GetTrueVSize());
         x.Randomize(7);

         for (int field = 1; field <= 2; ++field)
         {
            SCOPED_TRACE("w" + std::to_string(field));
            if (field == 2)
            {
               // A new lagged velocity, refreshed in place as every step does.
               SetVelocity(pb.w, w2_fn);
               rot->UpdateVorticity();
            }
            pb.w.HostRead(); // the reference reads w on the host

            AlphaCurlCross K(dim, pb.w);
            ParBilinearForm ref(&pb.fes);
            auto* mass = new VectorMassIntegrator(K);
            mass->SetIntRule(&pb.ir);
            ref.AddDomainIntegrator(mass);
            ref.Assemble();
            ref.Finalize();
            const Vector y_ref = Apply(ref, x);

            EXPECT_LE(RelDiff(Apply(*pa, x), y_ref), 1e-12) << "partial assembly";

            if (field == 1)
            {
               ParBilinearForm fa(&pb.fes);
               auto* full = new VectorRotationalConvectionIntegrator(pb.w, kAlpha);
               full->SetIntRule(&pb.ir);
               fa.AddDomainIntegrator(full);
               fa.Assemble();
               fa.Finalize();
               EXPECT_LE(RelDiff(Apply(fa, x), y_ref), 1e-12) << "full assembly";
            }
         }
      }
   }
}

// Correct GPU execution. For dim = 2, 3 and every specialized order p = 1..5,
// with the same rule and mesh as above:
//  1. Specialized: the kernel dispatch table has a compile-time
//     specialization for (dim, D1D = p + 1, Q1D = the order-3p rule's points
//     per direction). Without one the kernels run a generic version sized for
//     the largest supported order: still correct, so no value check notices,
//     but slower (~2.5x on a CPU, worse on a GPU).
//  2. Deterministic: 100 repeats of UpdateVorticity() + apply, with the same w
//     and x, are bitwise identical to the first apply. On a GPU, a missing
//     MFEM_SYNC_THREAD or a shared-memory race makes results vary from run to
//     run, which a single comparison at 1e-12 can miss. On a CPU this always
//     holds.
TEST(RotationalConvection, SpecializedAndDeterministic)
{
   for (int dim : {2, 3})
   {
      for (int p = 1; p <= 5; ++p)
      {
         SCOPED_TRACE(Label(dim, p));
         const int q1d = IntRules.Get(Geometry::SEGMENT, 3 * p).GetNPoints();
         EXPECT_TRUE(VectorRotationalConvectionIntegrator::HasSpecialization(
                        dim, p + 1, q1d)) << "q1d=" << q1d;

         Problem pb(dim, p);
         VectorRotationalConvectionIntegrator* rot = nullptr;
         std::unique_ptr<ParBilinearForm> pa = PaForm(pb, rot);
         Vector x(pb.fes.GetTrueVSize());
         x.Randomize(3);
         const Vector y0 = Apply(*pa, x);
         for (int k = 0; k < 100; ++k)
         {
            rot->UpdateVorticity();
            Vector y = Apply(*pa, x);
            y -= y0;
            ASSERT_EQ(y.Normlinf(), 0.0) << "repeat " << k;
         }
      }
   }
}
