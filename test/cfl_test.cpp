// Convective CFL ceiling (time/cfl, amr_spec.md D6).
//  C1 constant velocity on an affine sheared mesh: the rate equals
//     k^2 max_K sum_d |(J_K^{-1} u)_d| exactly, conforming and refined NC
//     (refined cells halve J's split column, doubling that entry);
//  C2 adaptive Navier-Stokes TGV with a binding time.cfl_max: the controller
//     keeps c * dt within the limit, and its steps are smaller than without.

#include <gtest/gtest.h>

#include "amr_test_util.hpp"
#include "exact/tgv2d.hpp"
#include "mesh/case_mesh.hpp"
#include "quadrature/rule_book.hpp"
#include "solver/case.hpp"
#include "time/cfl.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>

using namespace mfem;

namespace
{

const double kShear[3][3] = {{1.0, 0.3, 0.1}, {0.2, 1.1, 0.0}, {0.1, -0.2, 0.9}};
const double kU[3] = {2.0, -0.7, 0.4};

} // namespace

TEST(Cfl, C1_ConstantVelocityOnAffineMeshIsExact)
{
   incns::RuleBook rules;
   for (int dim : {2, 3})
      for (bool refine : {false, true})
      {
         SCOPED_TRACE("dim=" + std::to_string(dim) + (refine ? " refined" : ""));
         Mesh serial = incns::MakeBoxMesh(amr_test::Box(dim, dim == 2 ? 4 : 2, false));
         serial.Transform([dim](const Vector & x, Vector & y)
         {
            y.SetSize(dim);
            for (int i = 0; i < dim; ++i)
            {
               y(i) = 0.0;
               for (int j = 0; j < dim; ++j) { y(i) += kShear[i][j] * x(j); }
            }
         });
         auto mesh = incns::PartitionMesh(serial, true);
         if (refine) { amr_test::RefineBands(*mesh, true); }
         const int k = 3;
         H1_FECollection fec(k, dim);
         ParFiniteElementSpace fes(mesh.get(), &fec, dim, Ordering::byNODES);
         ParGridFunction u(&fes);
         Vector uv(dim);
         for (int i = 0; i < dim; ++i) { uv(i) = kU[i]; }
         VectorConstantCoefficient uc(uv);
         u.ProjectCoefficient(uc);

         incns::ConvectiveCfl cfl(fes, rules);
         const double rate = cfl.Rate(u);

         double ref = 0.0;
         IntegrationPoint c;
         c.Set3(0.5, 0.5, 0.5);
         for (int e = 0; e < mesh->GetNE(); ++e)
         {
            ElementTransformation& T = *mesh->GetElementTransformation(e);
            T.SetIntPoint(&c);
            DenseMatrix Jinv(T.InverseJacobian());
            Vector xi(dim);
            Jinv.Mult(uv, xi);
            double s = 0.0;
            for (int d = 0; d < dim; ++d) { s += std::fabs(xi(d)); }
            ref = std::max(ref, s);
         }
         MPI_Allreduce(MPI_IN_PLACE, &ref, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         ref *= k * k;
         EXPECT_NEAR(rate, ref, 1e-12 * ref);
      }
}

TEST(Cfl, C2_AdaptiveStepRespectsTheCeiling)
{
   auto run = [](double cfl_max, double & final_cfl, double & max_dt)
   {
      incns::Parameters p;
      p.equation = incns::Equation::NavierStokes;
      p.nu = 0.01;
      p.order_u = 3;
      p.order_p = 2;
      p.mesh = amr_test::Box(2, 4, true, 2.0 * M_PI);
      p.dt = 0.02;
      p.t_final = 0.6;
      p.adaptive = true;
      p.controller.atol = 1e-3; // loose: accuracy alone would take big steps
      p.controller.rtol = 1e-3;
      p.cfl_max = cfl_max;
      p.Normalize();
      auto mesh = incns::MakeCaseMesh(p);
      incns::Case flow(*mesh, p);
      VectorFunctionCoefficient u0 = incns::tgv2d::VelocityCoefficient(p.nu);
      flow.SetInitialVelocity(u0);
      flow.Run();
      final_cfl = flow.ConvectiveCflNumber();
      max_dt = 0.0;
      for (const incns::StepAttempt& a : flow.Integrator().Controller()->History())
      {
         if (a.accepted) { max_dt = std::max(max_dt, a.dt); }
      }
   };
   double cfl_free = 0.0, dt_free = 0.0, cfl_cap = 0.0, dt_cap = 0.0;
   run(0.0, cfl_free, dt_free);
   const double limit = 0.5 * cfl_free; // binds: the free run went past it
   run(limit, cfl_cap, dt_cap);
   if (Mpi::Root())
   {
      mfem::out << "[cfl] free: cfl=" << cfl_free << " max dt=" << dt_free
                << "   capped (" << limit << "): cfl=" << cfl_cap << " max dt="
                << dt_cap << std::endl;
   }
   EXPECT_LE(cfl_cap, 1.05 * limit);
   EXPECT_LT(dt_cap, dt_free);
}
