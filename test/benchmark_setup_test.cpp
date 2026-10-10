// The benchmark setup features (decks, no benchmark runs):
//  F1 forcing.body_force: plane Poiseuille flow u = f y (2 - y) / (2 nu)
//     between walls at y = 0 and 2 is steady under the deck's forcing (to
//     the solver tolerance) and decays without it;
//  F2 forcing in dimensional mode is an acceleration: scaled by L_ref /
//     U_ref^2;
//  I1 initial_velocity taylor_green_3d: the standard field and its energy;
//  I2 initial_velocity channel: Reichardt's mean profile for Re_tau = u_tau
//     delta / nu (u_tau from the forcing), zero at the walls, periodic, and
//     divergence-free (finite differences), the perturbation bounded by
//     initial.perturbation x the centreline velocity;
//  M1 mesh.geometry file: a Gmsh 2.2 file's physical groups become boundary
//     names (MFEM's attribute = the physical tag), mesh.boundary_names adds
//     names, and a run with boundary conditions by those names starts;
//  M2 the DFG channel in 3D: Extrude2D of the curved 2D mesh, nz layers;
//     volume and the boundary areas (inflow, cylinder, walls with the end
//     faces) exact to the geometry order, and the 3D parabolic inflow
//     carries u_max (4/9) H W;
//  B1 every deck in benchmarks/ loads (the parser rejects unknown keys), and
//     the benchmark settings are what their headers say.

#include <gtest/gtest.h>

#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "mesh/cylinder_channel.hpp"
#include "mesh/mesh_file.hpp"
#include "solver/case.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mfem;

namespace
{
double PoiseuilleError(bool forced)
{
   const std::string deck = std::string(
                               "equation: navier_stokes\n"
                               "physics:\n  nu: 0.1\n"
                               "mesh:\n  dim: 2\n  elements: [3, 2]\n  lengths: [1.0, 2.0]\n"
                               "  periodic: [true, false]\n"
                               "time:\n  dt: 0.01\n  t_final: 0.05\n  step_control: fixed\n"
                               // The start is the exact steady state: its
                               // residual cannot drop by rtol (atol is 0 by
                               // default).
                               "solver:\n  atol: 1.0e-12\n"
                               "boundary_conditions:\n  - {select: [ymin, ymax], type: no_slip}\n") +
                            (forced ? "forcing:\n  body_force: [0.2, 0.0]\n" : "");
   const incns::Parameters p = incns::Parameters::LoadYAMLString(deck);
   auto mesh = incns::MakeCaseMesh(p);
   incns::Case flow(*mesh, p);
   // u = f y (2 - y) / (2 nu) = y (2 - y) for f = 0.2, nu = 0.1.
   VectorFunctionCoefficient exact(2, [](const Vector & x, Vector & u)
   {
      u(0) = x(1) * (2.0 - x(1));
      u(1) = 0.0;
   });
   flow.SetInitialVelocity(exact);
   flow.Run();
   return flow.Velocity().ComputeL2Error(exact);
}
} // namespace

TEST(BenchmarkSetup, F1_ForcingDrivesPoiseuille)
{
   const double forced = PoiseuilleError(true), unforced = PoiseuilleError(false);
   if (Mpi::Root())
   {
      mfem::out << "[setup F1] Poiseuille after t = 0.05: forced " << forced
                << ", unforced " << unforced << "\n";
   }
   EXPECT_LT(forced, 1e-6);   // steady: the forcing balances nu u''
   EXPECT_GT(unforced, 1e-3); // decays without it (~1% by t = 0.05)
}

TEST(BenchmarkSetup, F2_ForcingIsAnAcceleration)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "physics:\n  nu: 0.5\n"
                                  "nondimensionalization:\n  mode: dimensional\n"
                                  "  L_ref: 2.0\n  U_ref: 3.0\n"
                                  "mesh:\n  dim: 2\n  elements: [2, 2]\n  lengths: [4.0, 4.0]\n"
                                  "forcing:\n  body_force: [9.0, -4.5]\n");
   EXPECT_NEAR(p.forcing.body_force[0], 9.0 * 2.0 / 9.0, 1e-14);
   EXPECT_NEAR(p.forcing.body_force[1], -4.5 * 2.0 / 9.0, 1e-14);
}

TEST(BenchmarkSetup, I1_TaylorGreen3D)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "mesh:\n  dim: 3\n  elements: [3, 3, 3]\n"
                                  "  lengths: [6.283185307179586, 6.283185307179586, "
                                  "6.283185307179586]\n"
                                  "initial_velocity: taylor_green_3d\n");
   auto u0 = incns::MakeInitialVelocity(p);
   // The field at a point, through a mesh that contains it.
   Mesh unit = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 3.0, 3.0, 3.0);
   ElementTransformation* T = unit.GetElementTransformation(0);
   IntegrationPoint ip;
   ip.Set3(0.3 / 3.0, 1.1 / 3.0, 2.5 / 3.0);
   T->SetIntPoint(&ip);
   Vector u(3);
   u0->Eval(u, *T, ip);
   EXPECT_NEAR(u(0), std::sin(0.3) * std::cos(1.1) * std::cos(2.5), 1e-14);
   EXPECT_NEAR(u(1), -std::cos(0.3) * std::sin(1.1) * std::cos(2.5), 1e-14);
   EXPECT_NEAR(u(2), 0.0, 1e-15);
   // The energy (1 / |Omega|) int |u|^2 / 2 = 1/8 (the reference's E(0)),
   // up to the interpolation error of this coarse mesh.
   auto mesh = incns::MakeCaseMesh(p);
   H1_FECollection fec(3, 3);
   ParFiniteElementSpace V(mesh.get(), &fec, 3);
   ParGridFunction g(&V);
   g.ProjectCoefficient(*u0);
   Vector zero(3);
   zero = 0.0;
   VectorConstantCoefficient z(zero);
   const double l2 = g.ComputeL2Error(z);
   const double vol = std::pow(2.0 * M_PI, 3);
   EXPECT_NEAR(0.5 * l2 * l2 / vol, 0.125, 2e-3);
}

namespace
{
const char* kChannel3D =
   "physics:\n  nu: 0.005\n"
   "mesh:\n  dim: 3\n  elements: [2, 2, 2]\n"
   "  lengths: [6.283185307179586, 2.0, 3.141592653589793]\n"
   "  periodic: [true, false, true]\n"
   "forcing:\n  body_force: [1.0, 0.0, 0.0]\n"
   "initial_velocity: channel\n";

// The coefficient at a point (through a mesh that contains it).
void EvalAt(VectorCoefficient& c, double x, double y, double z, Vector& u)
{
   static Mesh unit = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                      7.0, 2.0, 4.0);
   ElementTransformation* T = unit.GetElementTransformation(0);
   IntegrationPoint ip;
   ip.Set3(x / 7.0, y / 2.0, z / 4.0);
   T->SetIntPoint(&ip);
   u.SetSize(3);
   c.Eval(u, *T, ip);
}
} // namespace

TEST(BenchmarkSetup, I2_ChannelInitialCondition)
{
   const double eps = 0.1;
   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  std::string(kChannel3D) +
                                  "initial:\n  perturbation: 0.1\n  seed: 7\n");
   const incns::Parameters mean_only = incns::Parameters::LoadYAMLString(
                                          kChannel3D);
   auto u0 = incns::MakeInitialVelocity(p);
   auto um = incns::MakeInitialVelocity(mean_only);
   const double nu = 0.005, u_tau = 1.0, re_tau = 200.0;
   const double u_c = u_tau * incns::ReichardtUPlus(re_tau);
   Vector u, v;
   // The mean: Reichardt at the centreline, symmetric, zero at the walls.
   EvalAt(*um, 1.0, 1.0, 1.0, u);
   EXPECT_NEAR(u(0), u_c, 1e-12);
   EvalAt(*um, 1.0, 0.1, 1.0, u);
   EvalAt(*um, 1.0, 1.9, 1.0, v);
   EXPECT_NEAR(u(0), v(0), 1e-12);
   EXPECT_NEAR(u(0), u_tau * incns::ReichardtUPlus(0.1 * u_tau / nu), 1e-12);
   // Walls: zero, perturbation included.
   for (double x : {0.3, 2.0, 5.1})
   {
      for (double y : {0.0, 2.0})
      {
         EvalAt(*u0, x, y, 1.7, u);
         EXPECT_NEAR(u.Normlinf(), 0.0, 1e-13) << x << " " << y;
      }
   }
   // Periodic in x (2 pi) and z (pi).
   EvalAt(*u0, 0.4, 0.7, 0.9, u);
   EvalAt(*u0, 0.4 + 2.0 * M_PI, 0.7, 0.9 + M_PI, v);
   v -= u;
   EXPECT_LT(v.Normlinf(), 1e-11);
   // Divergence-free (central differences, h^2 error ~ 1e-8 here) and the
   // perturbation bounded by eps u_c in the streamwise component.
   const double h = 1e-4;
   double max_div = 0.0, max_up = 0.0;
   for (int i = 0; i < 40; ++i)
   {
      const double x = 0.37 + 0.149 * i, y = 0.05 + 0.0475 * i,
                   z = 0.11 + 0.071 * i;
      Vector a, b;
      double div = 0.0;
      EvalAt(*u0, x + h, y, z, a);
      EvalAt(*u0, x - h, y, z, b);
      div += (a(0) - b(0)) / (2 * h);
      EvalAt(*u0, x, y + h, z, a);
      EvalAt(*u0, x, y - h, z, b);
      div += (a(1) - b(1)) / (2 * h);
      EvalAt(*u0, x, y, z + h, a);
      EvalAt(*u0, x, y, z - h, b);
      div += (a(2) - b(2)) / (2 * h);
      max_div = std::max(max_div, std::abs(div));
      EvalAt(*u0, x, y, z, a);
      EvalAt(*um, x, y, z, b);
      max_up = std::max(max_up, std::abs(a(0) - b(0)));
   }
   EXPECT_LT(max_div, 1e-6 * u_c);
   EXPECT_GT(max_up, 0.05 * eps * u_c); // present
   EXPECT_LE(max_up, eps * u_c);        // bounded
}

namespace
{
// Three quads [0,3] x [0,1]: inlet x = 0, outlet x = 3, walls top and bottom.
const char* kGmsh =
   "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
   "$PhysicalNames\n4\n1 1 \"inlet\"\n1 2 \"outlet\"\n1 3 \"walls\"\n"
   "2 4 \"fluid\"\n$EndPhysicalNames\n"
   "$Nodes\n8\n1 0 0 0\n2 1 0 0\n3 2 0 0\n4 3 0 0\n"
   "5 0 1 0\n6 1 1 0\n7 2 1 0\n8 3 1 0\n$EndNodes\n"
   "$Elements\n11\n"
   "1 1 2 1 1 1 5\n2 1 2 2 2 4 8\n"
   "3 1 2 3 3 1 2\n4 1 2 3 3 2 3\n5 1 2 3 3 3 4\n"
   "6 1 2 3 3 5 6\n7 1 2 3 3 6 7\n8 1 2 3 3 7 8\n"
   "9 3 2 4 4 1 2 6 5\n10 3 2 4 4 2 3 7 6\n11 3 2 4 4 3 4 8 7\n"
   "$EndElements\n";
} // namespace

TEST(BenchmarkSetup, M1_GmshNamedBoundaries)
{
   const std::string path =
      "setup_gmsh_np" + std::to_string(Mpi::WorldSize()) + ".msh";
   if (Mpi::Root())
   {
      std::ofstream f(path);
      f << kGmsh;
   }
   MPI_Barrier(MPI_COMM_WORLD);
   const auto names = incns::GmshPhysicalNames(path, 1);
   ASSERT_EQ(names.size(), 3u);
   EXPECT_EQ(names[0].first, "inlet");
   EXPECT_EQ(names[2].second, 3);

   const incns::Parameters p = incns::Parameters::LoadYAMLString(
                                  "equation: stokes\n"
                                  "physics:\n  nu: 1.0\n"
                                  "mesh:\n  dim: 2\n  geometry: file\n  file: " + path + "\n"
                                  "  boundary_names: {lid: 3}\n"
                                  "time:\n  dt: 0.05\n  t_final: 0.05\n  step_control: fixed\n"
                                  "boundary_conditions:\n"
                                  "  - {select: [inlet], type: velocity, value: [1.0, 0.0]}\n"
                                  "  - {select: [outlet], type: outflow}\n"
                                  "  - {select: [walls], type: no_slip}\n");
   auto mesh = incns::MakeCaseMesh(p);
   EXPECT_EQ(mesh->GetGlobalNE(), 3);
   EXPECT_EQ(mesh->bdr_attributes.Size(), 3); // physical tags 1, 2, 3
   EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "inlet"), 1);
   EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "outlet"), 2);
   EXPECT_EQ(incns::NamedBoundaryAttribute(p, *mesh, "lid"),
             3); // added by the deck
   incns::Case flow(*mesh, p); // coverage: every boundary in exactly one group
   flow.Run();
   EXPECT_TRUE(flow.Done());
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root()) { std::remove(path.c_str()); }
}

namespace
{
const char* kDfg3D =
   "mesh:\n  dim: 3\n  geometry: cylinder_channel\n"
   "  cylinder_channel:\n    length: 2.5\n    cx: 0.5\n    n_up: 4\n"
   "    nz: 2\n    depth: 0.41\n";

// Sum of the areas of the boundary elements with attribute @p attr.
double BoundaryArea(ParMesh& mesh, int attr)
{
   double a = 0.0;
   for (int b = 0; b < mesh.GetNBE(); ++b)
   {
      if (mesh.GetBdrAttribute(b) != attr) { continue; }
      ElementTransformation* T = mesh.GetBdrElementTransformation(b);
      const IntegrationRule& ir = IntRules.Get(T->GetGeometryType(), 8);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         T->SetIntPoint(&ir.IntPoint(q));
         a += ir.IntPoint(q).weight * T->Weight();
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &a, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   return a;
}
} // namespace

TEST(BenchmarkSetup, M2_ExtrudedDfgChannel)
{
   const incns::Parameters p = incns::Parameters::LoadYAMLString(kDfg3D);
   auto mesh = incns::MakeCaseMesh(p);
   std::string flat = kDfg3D;
   flat.replace(flat.find("dim: 3"), 6, "dim: 2");
   auto mesh2 = incns::MakeCaseMesh(incns::Parameters::LoadYAMLString(flat));
   EXPECT_EQ(mesh->GetGlobalNE(), 2 * mesh2->GetGlobalNE());
   ASSERT_NE(mesh->GetNodes(), nullptr); // curved (the extruded nodes)

   double vol = 0.0;
   for (int e = 0; e < mesh->GetNE(); ++e) { vol += mesh->GetElementVolume(e); }
   MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double H = 0.41, r = 0.05;
   EXPECT_NEAR(vol, (2.5 * H - M_PI * r * r) * H, 1e-6 * vol);
   EXPECT_NEAR(BoundaryArea(*mesh, incns::kCylinderInflow), H * H, 1e-12);
   EXPECT_NEAR(BoundaryArea(*mesh, incns::kCylinderBody), 2.0 * M_PI * r * H,
               1e-6);
   EXPECT_NEAR(BoundaryArea(*mesh, incns::kCylinderWalls),
               2.0 * 2.5 * H + 2.0 * (2.5 * H - M_PI * r * r), 1e-6);
   const std::vector<int> attrs = incns::AllBoundaryAttributes(p, *mesh);
   EXPECT_EQ(attrs.size(), 4u); // the end faces joined the walls

   // The 3D parabolic inflow: flux u_max (4/9) H W (exact: the profile is
   // quadratic in y and z).
   const incns::Parameters pb = incns::Parameters::LoadYAMLString(
                                   std::string(kDfg3D) +
                                   "boundary_conditions:\n"
                                   "  - {select: [inflow], type: velocity, profile: parabolic, "
                                   "u_max: 0.45, height: 0.41}\n"
                                   "  - {select: [outflow], type: outflow}\n"
                                   "  - {select: [walls, cylinder], type: no_slip}\n");
   H1_FECollection fec(3, 3);
   ParFiniteElementSpace V(mesh.get(), &fec, 3);
   incns::BoundaryConditions bc(V);
   incns::DeckBoundaryConditions deck(pb, *mesh);
   deck.Apply(bc, nullptr, true);
   bc.SetTime(0.0);
   ParGridFunction g(&V);
   g = 0.0;
   bc.ProjectDirichlet(g);
   double flux = 0.0;
   Vector val(3);
   for (int b = 0; b < mesh->GetNBE(); ++b)
   {
      if (mesh->GetBdrAttribute(b) != incns::kCylinderInflow) { continue; }
      FaceElementTransformations* ft = mesh->GetBdrFaceTransformations(b);
      const IntegrationRule& ir = IntRules.Get(ft->GetGeometryType(), 8);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         ft->SetAllIntPoints(&ir.IntPoint(q));
         g.GetVectorValue(*ft->Elem1, ft->GetElement1IntPoint(), val);
         flux += ir.IntPoint(q).weight * ft->Weight() * val(0);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &flux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   EXPECT_NEAR(flux, 0.45 * 4.0 / 9.0 * H * H, 1e-12);
}

TEST(BenchmarkSetup, B1_EveryBenchmarkDeckLoads)
{
   namespace fs = std::filesystem;
   const fs::path root = fs::path(INCNS_SOURCE_DIR) / "benchmarks";
   int decks = 0;
   for (const auto& e : fs::recursive_directory_iterator(root))
   {
      if (e.path().extension() != ".yaml") { continue; }
      const incns::Parameters p = incns::Parameters::LoadYAML(e.path().string());
      EXPECT_EQ(p.equation, incns::Equation::NavierStokes) << e.path();
      ++decks;
   }
   EXPECT_EQ(decks, 7); // tgv n16/n32/n64, channel 180/395, dfg 3d-1z/3z
   const incns::Parameters ch = incns::Parameters::LoadYAML(
                                   (root / "channel" / "channel_retau180.yaml").string());
   EXPECT_TRUE(ch.channel_statistics.enabled);
   EXPECT_NEAR(1.0 / ch.nu, 178.12, 1e-6); // Re_tau = 1 / nu at u_tau = delta = 1
   EXPECT_NEAR(ch.reference.re_tau, 178.12, 1e-12);
   const incns::Parameters tg = incns::Parameters::LoadYAML(
                                   (root / "tgv_re1600" / "tgv_re1600_n16.yaml").string());
   EXPECT_NEAR(1.0 / tg.nu, 1600.0, 1e-9);
   EXPECT_TRUE(tg.output.diagnostics);
   EXPECT_FALSE(tg.output.enabled); // no field output for the energy history
}
