// Deck boundary conditions and geometries (bc/boundary_names,
// bc/deck_boundary, MakeCaseMesh's geometry switch):
//  B1 box faces resolve by name; a walled-box deck that leaves a face
//     uncovered, or selects one twice, is reported (the face named); a
//     complete one is clean;
//  B2 the cylinder geometries: MakeCaseMesh builds them, their boundary names
//     map to the documented attributes (body / cylinder = 4), every attribute
//     is real;
//  B3 a deck-given velocity (type velocity, value) is imposed: a lid-driven
//     cavity's lid velocity after a step is the deck's value;
//  B4 a deck-given velocity is separable in time, u_D = s(t) g(x): on the DFG
//     2D channel (the 2D-3 sine inflow, walls and cylinder no-slip) the time
//     factors times the projected shapes reproduce the projected data on
//     every Dirichlet dof -- bit for bit where s = 1, to roundoff during the
//     sine (the projection averages s g where shapes are averaged) -- and so
//     does the cached path the Stokes step takes.

#include <gtest/gtest.h>

#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "spaces/mixed_spaces.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace mfem;

namespace
{
const char* kWalledBox =
   "equation: navier_stokes\n"
   "physics:\n  Re: 10\n"
   "mesh:\n  dim: 2\n  elements: [3, 3]\n  lengths: [1, 1]\n  periodic: [false, false]\n"
   "time:\n  dt: 0.01\n  t_final: 0.02\n  step_control: fixed\n";

bool AnyContains(const std::vector<std::string>& v, const std::string& s)
{
   for (const std::string& x : v)
   {
      if (x.find(s) != std::string::npos) { return true; }
   }
   return false;
}
} // namespace

TEST(DeckBoundary, B1_BoxCoverage)
{
   const incns::Parameters missing = incns::Parameters::LoadYAMLString(
                                        std::string(kWalledBox) +
                                        "boundary_conditions:\n"
                                        "  - {select: [xmin, xmax, ymin], type: no_slip}\n");
   auto mesh = incns::MakeCaseMesh(missing);
   EXPECT_EQ(incns::AllBoundaryAttributes(missing, *mesh).size(), 4u);
   {
      incns::DeckBoundaryConditions d(missing, *mesh);
      const std::vector<std::string> pr = d.CoverageProblems();
      EXPECT_TRUE(AnyContains(pr, "'ymax'")
                  && AnyContains(pr, "no boundary condition"))
            << (pr.empty() ? std::string("(no problems)") : pr[0]);
   }
   const incns::Parameters twice = incns::Parameters::LoadYAMLString(
                                      std::string(kWalledBox) +
                                      "boundary_conditions:\n"
                                      "  - {select: all, type: no_slip}\n"
                                      "  - {select: [ymax], type: velocity, value: [1, 0]}\n");
   {
      incns::DeckBoundaryConditions d(twice, *mesh);
      EXPECT_TRUE(AnyContains(d.CoverageProblems(), "selected by 2 groups"));
   }
   const incns::Parameters ok = incns::Parameters::LoadYAMLString(
                                   std::string(kWalledBox) +
                                   "boundary_conditions:\n"
                                   "  - {select: [ymax], type: velocity, value: [1, 0]}\n"
                                   "  - {select: [xmin, xmax, ymin], type: no_slip}\n");
   incns::DeckBoundaryConditions d(ok, *mesh);
   EXPECT_TRUE(d.CoverageProblems().empty());
}

TEST(DeckBoundary, B2_CylinderGeometries)
{
   {
      const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                     "equation: navier_stokes\nphysics:\n  Re: 100\n"
                                     "mesh:\n  dim: 2\n  geometry: square_cylinder\n"
                                     "  square_cylinder: {n_face: 2, far_ratio: 1.5, "
                                     "wake_h: 1.0, wake_end: 25.0}\n");
      auto mesh = incns::MakeCaseMesh(p);
      EXPECT_EQ(mesh->bdr_attributes.Max(), 4);
      EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "inflow"), 1);
      EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "body"), 4);
      EXPECT_EQ(incns::AllBoundaryAttributes(p, *mesh),
                (std::vector<int> {1, 2, 3, 4}));
   }
   {
      const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                     "equation: navier_stokes\nphysics:\n  nu: 0.001\n"
                                     "mesh:\n  dim: 2\n  geometry: cylinder_channel\n");
      auto mesh = incns::MakeCaseMesh(p);
      EXPECT_EQ(mesh->GetGlobalNE(), 208); // the DFG base mesh
      EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "cylinder"), 4);
      EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "body"), 4);
      EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "walls"), 3);
   }
}

TEST(DeckBoundary, B3_DeckVelocityIsImposed)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  std::string(kWalledBox) +
                                  "boundary_conditions:\n"
                                  "  - {select: [ymax], type: velocity, value: [0.7, 0]}\n"
                                  "  - {select: [xmin, xmax, ymin], type: no_slip}\n");
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   flow.Step();
   // The velocity at the middle of the lid (y = 1) and of the floor (y = 0).
   ParGridFunction& u = flow.Velocity();
   DenseMatrix pts(2, 2);
   pts(0, 0) = 0.5;
   pts(1, 0) = 1.0;
   pts(0, 1) = 0.5;
   pts(1, 1) = 0.0;
   Array<int> elems;
   Array<IntegrationPoint> ips;
   mesh->FindPoints(pts, elems, ips, false);
   double v[4] = {0.0, 0.0, 0.0, 0.0}; // lid u, floor u, found lid, found floor
   Vector uv(2);
   for (int k = 0; k < 2; ++k)
   {
      if (elems[k] < 0) { continue; }
      u.GetVectorValue(elems[k], ips[k], uv);
      v[k] = uv(0);
      v[2 + k] = 1.0;
   }
   MPI_Allreduce(MPI_IN_PLACE, v, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   ASSERT_GT(v[2], 0.0);
   ASSERT_GT(v[3], 0.0);
   EXPECT_NEAR(v[0] / v[2], 0.7, 1e-12);
   EXPECT_NEAR(v[1] / v[3], 0.0, 1e-12);
}

TEST(DeckBoundary, B4_DeckVelocityIsSeparable)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "equation: navier_stokes\nphysics:\n  nu: 0.001\n"
                                  "mesh:\n  dim: 2\n  geometry: cylinder_channel\n"
                                  "boundary_conditions:\n"
                                  "  - {select: [inflow], type: velocity, profile: parabolic, "
                                  "u_max: 1.5, height: 0.41, time_profile: sine, time_scale: 8.0}\n"
                                  "  - {select: [walls, cylinder], type: no_slip}\n"
                                  "  - {select: [outflow], type: outflow}\n");
   auto mesh = incns::MakeCaseMesh(p);
   incns::MixedSpaces spaces(*mesh, 3, 2);
   ParFiniteElementSpace& V = spaces.Velocity();
   incns::BoundaryConditions bc(V);
   incns::DeckBoundaryConditions deck(p, *mesh);
   deck.Apply(bc, nullptr, true);
   ASSERT_TRUE(bc.DirichletIsSeparable());
   EXPECT_FALSE(bc.DirichletIsConstant());
   EXPECT_EQ(bc.NumDirichlet(), 3); // inflow, walls, cylinder

   ParGridFunction g(&V), entry(&V), u(&V);
   g = 0.0;
   entry = -1.0;
   bc.ProjectDirichletShapes(g, entry);
   Vector gt, et, ut;
   g.GetTrueDofs(gt);
   entry.GetTrueDofs(et);
   const Array<int>& ess = bc.EssentialTrueDofs();
   // t = 4: s = sin(pi / 2) = 1 exactly; t = 1.3: s = 0.49.
   for (double t : {4.0, 1.3})
   {
      bc.SetTime(t);
      u = 0.0;
      bc.ProjectDirichlet(u);
      u.GetTrueDofs(ut);
      const double* G = gt.HostRead();
      const double* E = et.HostRead();
      const double* U = ut.HostRead();
      double v[3] = {0.0, 0.0, 0.0}; // max |u|, max |s g - u|, dofs differing
      int bad_entry = 0;
      for (int i : ess)
      {
         const int e = static_cast<int>(E[i]);
         if (e < 0 || e >= bc.NumDirichlet() || E[i] != e) { ++bad_entry; continue; }
         const double d = std::abs(bc.DirichletFactor(e, t) * G[i] - U[i]);
         v[0] = std::max(v[0], std::abs(U[i]));
         v[1] = std::max(v[1], d);
         if (d != 0.0) { v[2] += 1.0; }
      }
      MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &v[2], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &bad_entry, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         mfem::out << "[deck B4] t " << t << ": max |u_D| " << v[0]
                   << ", max |s g - u_D| " << v[1] << " (" << v[2]
                   << " dofs differ)\n";
      }
      EXPECT_EQ(bad_entry, 0);
      EXPECT_GT(v[0], 0.5);
      if (t == 4.0) { EXPECT_EQ(v[1], 0.0); }
      else { EXPECT_LE(v[1], 2e-15 * v[0]); }
      // The cached path the Stokes step takes (SeparableEssentialValues):
      // the same values.
      Vector sv;
      bc.SeparableEssentialValues(sv);
      const double* SV = sv.HostRead();
      const int* ED = ess.HostRead();
      double dmax = 0.0;
      for (int k = 0; k < ess.Size(); ++k)
      {
         dmax = std::max(dmax, std::abs(SV[k] - U[ED[k]]));
      }
      MPI_Allreduce(MPI_IN_PLACE, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (t == 4.0) { EXPECT_EQ(dmax, 0.0); }
      else { EXPECT_LE(dmax, 2e-15 * v[0]); }
   }
}
