// Fluid-structure interaction (docs/fsi.md): the partitioned coupling of the
// ALE flow and the linear solid on the Turek-Hron geometry, coarse and at
// Q2/Q1, at np 1/2/4:
//  F1 the interface maps: a linear field crosses between the solid mesh and
//     the fluid's reference mesh exactly, both ways (the two meshes share the
//     interface's nodes);
//  F2 a fluid trial step undone and taken again reproduces itself, and a
//     committed trial step equals a plain Step();
//  F3 the traction handed to the solid carries the fluid's force: its
//     integral equals rho_f times the force on the flag (BodyForce);
//  F4 coupled steps converge (IQN-ILS with and without reuse, Aitken), the
//     fluid's interface displacement is the solid's at the interface's
//     nodes, and IQN-ILS needs fewer subiterations than Aitken;
//  F5 a static solid couples too (on a ramped start: under the impulsive
//     start's pressure pulse a static flag swings far out).
// These are unit tests: they pin their own small orders and tolerances.

#include <gtest/gtest.h>

#include "config/parameters.hpp"
#include "fsi/fsi_case.hpp"
#include "fsi/interface_transfer.hpp"
#include "mesh/case_mesh.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;
using incns::Case;
using incns::FsiCase;
using incns::InterfaceTransfer;
using incns::Parameters;

namespace
{

/// A coarse Turek-Hron FSI1-like deck: Q2/Q1, the solid of FSI1 (E 1.4e6,
/// nu 0.4, rho_s = rho_f = 1000), an impulsive start for visible motion.
std::string Deck(const std::string& fsi_extra, const std::string& solid_extra,
                 int np, bool ramp = false)
{
   return std::string(
             "equation: navier_stokes\n"
             "physics:\n  nu: 1.0e-3\n"
             "discretization:\n  order_u: 2\n  order_p: 1\n"
             "mesh:\n  dim: 2\n  geometry: cylinder_channel\n"
             "  cylinder_channel:\n    length: 2.5\n    flag_length: 0.35\n"
             "    flag_thickness: 0.02\n    n_flag: 8\n    n_flag_y: 2\n"
             "    n_down: 16\n"
             "time:\n  dt: 0.005\n  t_final: 1.0\n  step_control: fixed\n"
             "  convection: imex\n"
             "solver:\n  rtol: 1.0e-10\n"
             "boundary_conditions:\n"
             "  - {select: [inflow], type: velocity, profile: parabolic, "
             "u_max: 0.3, height: 0.41" +
             std::string(ramp ? ", time_profile: ramp, time_scale: 0.05" : "") +
             "}\n"
             "  - {select: [walls, cylinder], type: no_slip}\n"
             "  - {select: [flag], type: interface}\n"
             "  - {select: [outflow], type: outflow}\n"
             "forces:\n  enabled: true\n  boundaries: [flag]\n  interval: 0\n"
             "solid:\n  youngs_modulus: 1.4e6\n  poisson_ratio: 0.4\n"
             "  density: 1000\n  probe: [0.6, 0.2]\n") + solid_extra +
          "fsi:\n  enabled: true\n  solid_fixed: [cylinder]\n"
          "  fluid_density: 1000\n  rtol: 1.0e-8\n" + fsi_extra +
          "output:\n  path: fsi_test_out_np" + std::to_string(np) +
          "\n  name: fsi\n  progress: 0\n";
}

int Np()
{
   int np = 1;
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   return np;
}

double MaxDiff(const Vector& a, const Vector& b)
{
   double m = 0.0;
   for (int i = 0; i < a.Size(); ++i) { m = std::max(m, std::abs(a(i) - b(i))); }
   MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return m;
}

} // namespace

// F1: a linear field crosses exactly between the matching meshes.
TEST(Fsi, F1_InterfaceMapsAreExactOnLinearFields)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("", "  analysis: dynamic\n", Np()));
   auto fluid_mesh = incns::MakeCaseMesh(p);
   FsiCase fsi(*fluid_mesh, p);
   ParMesh& solid_mesh = fsi.SolidMesh();
   VectorFunctionCoefficient lin(2, [](const Vector & x, Vector & v)
   {
      v(0) = 1.0 + 2.0 * x(0) - 3.0 * x(1);
      v(1) = -0.5 + 0.25 * x(0) + x(1);
   });
   // Solid -> fluid (the displacement's way) onto the motion space.
   ParFiniteElementSpace& mfes = fsi.Fluid().Motion().Model().Space();
   ParGridFunction src(&fsi.Solid().Space());
   src.ProjectCoefficient(lin);
   ParGridFunction dst(&mfes), exact(&mfes);
   dst = 0.0;
   exact.ProjectCoefficient(lin);
   const int flag = 5;
   InterfaceTransfer to_fluid(solid_mesh, mfes, {flag});
   to_fluid.Transfer(src, dst);
   EXPECT_GT(to_fluid.GlobalPoints(), 0);
   double err = 0.0;
   for (int i : to_fluid.Vdofs()) { err = std::max(err, std::abs(dst(i) - exact(i))); }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LT(err, 1e-11);

   // Fluid -> solid (the traction's way) from the fluid's reference mesh.
   ParMesh& ref = fsi.Fluid().Motion().ReferenceMesh();
   ParFiniteElementSpace& vfes = fsi.Fluid().Spaces().Velocity();
   ParFiniteElementSpace ref_vfes(&ref, vfes.FEColl(), 2, vfes.GetOrdering());
   ParGridFunction fsrc(&ref_vfes);
   fsrc.ProjectCoefficient(lin);
   ParGridFunction sdst(&fsi.Solid().Space()), sexact(&fsi.Solid().Space());
   sdst = 0.0;
   sexact.ProjectCoefficient(lin);
   InterfaceTransfer to_solid(ref, fsi.Solid().Space(), {flag});
   to_solid.Transfer(fsrc, sdst);
   err = 0.0;
   for (int i : to_solid.Vdofs()) { err = std::max(err, std::abs(sdst(i) - sexact(i))); }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   EXPECT_LT(err, 1e-11);
   EXPECT_EQ(to_solid.GlobalPoints() > 0, true);
}

// F2: trial, undo, trial again; commit = Step().
TEST(Fsi, F2_TrialStepUndoesAndCommits)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("", "  analysis: dynamic\n", Np()));
   Vector u_step;
   {
      auto mesh = incns::MakeCaseMesh(p);
      FsiCase fsi(*mesh, p);
      Case& f = fsi.Fluid();
      ParGridFunction& d = *f.InterfaceDisplacement();
      // A made-up interface displacement: the flag bends a little.
      VectorFunctionCoefficient bend(2, [](const Vector & x, Vector & v)
      {
         v(0) = 0.0;
         v(1) = 1e-3 * std::max(0.0, x(0) - 0.25) * (x(0) - 0.25);
      });
      d.ProjectCoefficient(bend);
      f.Step();
      f.Step();
      u_step.SetSize(f.Spaces().Velocity().GetTrueVSize());
      f.Velocity().GetTrueDofs(u_step);
   }
   auto mesh = incns::MakeCaseMesh(p);
   FsiCase fsi(*mesh, p);
   Case& f = fsi.Fluid();
   ParGridFunction& d = *f.InterfaceDisplacement();
   VectorFunctionCoefficient bend(2, [](const Vector & x, Vector & v)
   {
      v(0) = 0.0;
      v(1) = 1e-3 * std::max(0.0, x(0) - 0.25) * (x(0) - 0.25);
   });
   d.ProjectCoefficient(bend);
   f.Step();
   const int n = f.Spaces().Velocity().GetTrueVSize();
   Vector u1(n), u2(n), u3(n);
   f.TrialStep();
   f.Velocity().GetTrueDofs(u1);
   f.UndoTrialStep();
   EXPECT_EQ(f.Motion().MoveCount(), 1);
   // A different interface position in between, then back.
   ParGridFunction keep(d);
   d *= 3.0;
   f.TrialStep();
   f.Velocity().GetTrueDofs(u3);
   f.UndoTrialStep();
   d = keep;
   f.TrialStep();
   f.Velocity().GetTrueDofs(u2);
   f.CommitTrialStep();
   const double scale = u1.Normlinf();
   EXPECT_GT(MaxDiff(u1, u3), 1e-8 * scale); // the interface mattered
   EXPECT_LT(MaxDiff(u1, u2), 1e-9 * scale);
   EXPECT_LT(MaxDiff(u2, u_step), 1e-9 * scale);
   EXPECT_EQ(f.Motion().MoveCount(), 2);
}

// F3: the traction's integral is rho_f times the force on the flag.
TEST(Fsi, F3_TractionCarriesTheForce)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("", "  analysis: dynamic\n", Np()));
   auto mesh = incns::MakeCaseMesh(p);
   FsiCase fsi(*mesh, p);
   fsi.Fluid().Step();
   fsi.Fluid().Step();
   fsi.UpdateTraction();
   const Vector F = fsi.Fluid().BodyForceVector();
   const Vector T = fsi.InterfaceForce();
   for (int c = 0; c < 2; ++c)
   {
      EXPECT_NEAR(T(c), 1000.0 * F(c), 1e-9 * std::abs(1000.0 * F(0)));
   }
   // The traction field integrates to it on the fluid's interface: through
   // the transfer to the solid (the same nodes, the reference interface).
   ParFiniteElementSpace& sfes = fsi.Solid().Space();
   ParLinearForm l(&sfes);
   VectorGridFunctionCoefficient tc(const_cast<ParGridFunction*>(
                                       &fsi.SolidTraction()));
   Array<int> marker(fsi.SolidMesh().bdr_attributes.Max());
   marker = 0;
   marker[4] = 1; // flag
   l.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tc), marker);
   l.Assemble();
   ParGridFunction one(&sfes);
   for (int c = 0; c < 2; ++c)
   {
      VectorFunctionCoefficient e(2, [c](const Vector&, Vector & v)
      {
         v = 0.0;
         v(c) = 1.0;
      });
      one.ProjectCoefficient(e);
      double s = l * one;
      MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      // Shared dofs are counted on each rank in a local dot product: use the
      // true-dof form instead.
      Vector lt(sfes.GetTrueVSize()), ot(sfes.GetTrueVSize());
      l.ParallelAssemble(lt);
      one.GetTrueDofs(ot);
      const double st = InnerProduct(MPI_COMM_WORLD, lt, ot);
      // The reference and the moved interface differ by the (small) motion.
      EXPECT_NEAR(st, T(c), 1e-6 * std::abs(T(0))) << c;
      (void)s;
   }
}

// F4: coupled steps converge with each accelerator; IQN-ILS needs fewer
// subiterations than Aitken.
TEST(Fsi, F4_CoupledStepsConverge)
{
   struct Variant
   {
      const char* name;
      std::string keys;
   };
   const Variant variants[] =
   {
      {"iqn_ils", "  acceleration: iqn_ils\n"},
      {"iqn_ils reuse 2", "  acceleration: iqn_ils\n  reuse: 2\n"},
      {"aitken", "  acceleration: aitken\n"},
   };
   std::vector<long long> its;
   for (const Variant& v : variants)
   {
      const Parameters p = Parameters::LoadYAMLString(
                              Deck(v.keys + "  max_iterations: 300\n",
                                   "  analysis: dynamic\n", Np()));
      auto mesh = incns::MakeCaseMesh(p);
      FsiCase fsi(*mesh, p);
      for (int s = 0; s < 2; ++s) { ASSERT_TRUE(fsi.Step()) << v.name << " " << s; }
      long long n = 0;
      for (const FsiCase::StepRecord& r : fsi.History())
      {
         n += r.iterations;
         EXPECT_LE(r.residual, 1.01e-8); // rtol, + atol / |d|
      }
      its.push_back(n);
      // The fluid's interface sits where the solid's is.
      ParGridFunction& d = *fsi.Fluid().InterfaceDisplacement();
      ParFiniteElementSpace& mfes = *d.ParFESpace();
      ParGridFunction sd(&mfes);
      sd = 0.0;
      InterfaceTransfer map(fsi.SolidMesh(), mfes, {5});
      map.Transfer(fsi.Solid().Displacement(), sd);
      double dmax = 0.0, err = 0.0;
      for (int i : map.Vdofs())
      {
         dmax = std::max(dmax, std::abs(sd(i)));
         err = std::max(err, std::abs(sd(i) - d(i)));
      }
      MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_GT(dmax, 0.0);
      EXPECT_LT(err, 1e-6 * dmax) << v.name;
      if (Mpi::Root())
      {
         std::cout << "[fsi] " << v.name << ": subiterations " << n
                   << " in 2 steps, max |d| " << dmax << std::endl;
      }
   }
   // On this impulsive start IQN-ILS needs half of Aitken's (constant
   // relaxation at 0.1: 4.7x Aitken's over 4 steps; not run here).
   EXPECT_LT(its[0], its[2]);
}

// F5: a static solid (quasi-static coupling) converges too.
TEST(Fsi, F5_StaticSolid)
{
   const Parameters p = Parameters::LoadYAMLString(
                           Deck("  acceleration: iqn_ils\n",
                                "  analysis: static\n", Np(), true));
   auto mesh = incns::MakeCaseMesh(p);
   FsiCase fsi(*mesh, p);
   for (int s = 0; s < 2; ++s) { ASSERT_TRUE(fsi.Step()) << s; }
   EXPECT_EQ(static_cast<int>(fsi.History().size()), 2);
   EXPECT_GT(fsi.Solid().MaxDisplacement(), 0.0);
}
