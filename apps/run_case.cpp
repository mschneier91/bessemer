// Generic YAML-driven driver: `run_case <deck.yaml>` -- no recompilation per
// case. The deck supplies parameters, mesh, initial condition (by name), and
// output settings; the case goes through the same Case surface as any
// in-code driver, equation: linear_elasticity or hyperelasticity runs a solid
// instead (ElasticityCase), and equation: mesh_motion deforms the mesh
// (MeshMotionCase). `run_case --deck-reference` prints every deck key with its
// default (docs/deck_reference.md is that output). Every run writes a
// machine-readable summary (<output.path>/<output.name>_summary.json, or
// --summary <file>): "running" at the start, then "ok" or "diverged" (exit
// code 2). `run_case <deck.yaml> --check` loads the deck, builds the mesh and
// checks the boundary groups without stepping, and prints a JSON report (the
// DOF counts, the settings in effect, or the deck's error); `run_case
// --deck-schema` prints every deck key as JSON. Both serve tools that write
// and vet decks (mcp_server/, docs/assistant.md); neither changes a run.

#include "ale/mesh_motion_case.hpp"
#include "fsi/fsi_case.hpp"
#include "bc/boundary_names.hpp"
#include "bc/deck_boundary.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solid/elasticity_case.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "util/json.hpp"
#include "mfem.hpp"

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/// @return The equation's deck name.
const char* EquationName(incns::Equation e)
{
   switch (e)
   {
      case incns::Equation::Stokes: return "stokes";
      case incns::Equation::NavierStokes: return "navier_stokes";
      case incns::Equation::LinearElasticity: return "linear_elasticity";
      case incns::Equation::MeshMotion: return "mesh_motion";
      case incns::Equation::Hyperelasticity: return "hyperelasticity";
   }
   return "unknown";
}

/// @return The step control's deck name.
const char* StepControlName(incns::StepControl s)
{
   switch (s)
   {
      case incns::StepControl::Fixed: return "fixed";
      case incns::StepControl::Error: return "error";
      case incns::StepControl::Cfl: return "cfl";
   }
   return "unknown";
}

/// @return The global true size of an H1 space of @p order, @p vdim components.
long long H1Size(ParMesh& mesh, int order, int vdim)
{
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec, vdim);
   return static_cast<long long>(fes.GlobalTrueVSize());
}

/**
 * @brief `run_case <deck> --check`: load the deck, build its mesh, check the
 *        boundary groups and count the DOFs, without building a solver or
 *        stepping. Prints one JSON object on rank 0 (status "ok" or "error"
 *        with the loader's message). Collective.
 * @param deck The deck's path.
 * @return 0 when the deck is ready to run, 3 otherwise.
 */
int CheckDeck(const char* deck)
{
#ifdef MFEM_USE_EXCEPTIONS
   // Errors become exceptions here, so the report carries their message.
   // Without exceptions an MFEM error prints its message and aborts, and the
   // report is missing: callers read the message from stderr then.
   mfem::set_error_action(mfem::MFEM_ERROR_THROW);
#endif
   incns::Json out;
   out["deck"].Set(deck);
   bool ok = true;
   try
   {
      const incns::Parameters p = incns::Parameters::LoadYAML(deck);
      out["equation"].Set(EquationName(p.equation));
      std::unique_ptr<ParMesh> pmesh = incns::MakeCaseMesh(p);
      ParMesh& mesh = *pmesh;
      const int dim = mesh.Dimension();
      out["mesh"]["dimension"].Set(dim);
      out["mesh"]["elements"].Set(static_cast<long long>(mesh.GetGlobalNE()));
      std::string names;
      for (const std::string& n : incns::BoundaryNames(p))
      {
         names += (names.empty() ? "" : ", ") + n;
      }
      out["mesh"]["boundary_names"].Set(names);

      // The checks run_case makes before building a case.
      std::string problems;
      if (p.boundary_conditions.empty() && !names.empty())
      {
         problems = "the mesh has boundaries (" + names + ") but the deck "
                    "declares no boundary_conditions";
      }
      else if (!p.boundary_conditions.empty())
      {
         incns::DeckBoundaryConditions bc(p, mesh);
         for (const std::string& s : bc.CoverageProblems())
         {
            problems += (problems.empty() ? "" : "; ") + s;
         }
      }
      if (!problems.empty())
      {
         ok = false;
         out["error"].Set("boundary_conditions: every real boundary needs "
                          "exactly one group: " + problems);
      }

      if (incns::IsSolid(p.equation) || p.equation == incns::Equation::MeshMotion)
      {
         const int k = incns::IsSolid(p.equation) ? p.solid.Order(p.order_u) :
                       p.mesh_motion.Order(p.order_u);
         out["orders"]["displacement"].Set(k);
         out["dofs"]["displacement"].Set(H1Size(mesh, k, dim));
      }
      else
      {
         const long long nu = H1Size(mesh, p.order_u, dim);
         const long long np = H1Size(mesh, p.order_p, 1);
         out["orders"]["velocity"].Set(p.order_u);
         out["orders"]["pressure"].Set(p.order_p);
         out["dofs"]["velocity"].Set(nu);
         out["dofs"]["pressure"].Set(np);
         out["dofs"]["total"].Set(nu + np);
         incns::Json& e = out["in_effect"];
         e["convection"].Set(p.Convection() == incns::ConvectionTreatment::Oifs ?
                             "oifs" : "imex");
         e["bdf_order"].Set(p.BdfOrder());
         e["collocated_mass"].Set(p.CollocatedMass());
         e["grad_div_h"].Set(p.GradDivH());
         e["nu"].Set(p.nu);
         if (p.CflSteps()) { e["cfl_target"].Set(p.CflTarget()); }
      }
      incns::Json& t = out["time"];
      t["step_control"].Set(StepControlName(p.step_control));
      t["dt"].Set(p.dt);
      t["t_final"].Set(p.t_final);
      // The step count is known only when dt does not adapt.
      const bool fixed_dt = p.step_control == incns::StepControl::Fixed ||
                            (p.step_control == incns::StepControl::Cfl &&
                             !p.CflSteps());
      if (p.equation == incns::Equation::MeshMotion)
      {
         t["steps"].Set(p.mesh_motion.steps); // moves
      }
      else if (incns::IsSolid(p.equation) && !p.fsi.enabled &&
               p.solid.analysis == incns::SolidAnalysis::Static)
      {
         t["steps"].Set(0); // one static solve
      }
      else if (fixed_dt && p.dt > 0.0 && !p.amr.enabled)
      {
         t["steps"].Set(static_cast<long long>(std::ceil(p.t_final / p.dt - 1e-9)));
      }
      out["amr"].Set(p.amr.enabled);
      out["fsi"].Set(p.fsi.enabled);
      out["output_summary"].Set(p.output.path + "/" + p.output.name +
                                "_summary.json");
   }
   catch (const std::exception& ex)
   {
      ok = false;
      out["error"].Set(ex.what());
   }
   out["status"].Set(ok ? "ok" : "error");
   if (Mpi::Root()) { std::cout << out.Dump(); }
   return ok ? 0 : 3;
}

/// `run_case --deck-schema`: every deck key as JSON, keyed by its path.
void PrintDeckSchema()
{
   incns::Json out;
   for (const incns::DeckKey& k : incns::Parameters::DeckSchema())
   {
      incns::Json& e = out[k.path];
      e["type"].Set(k.type);
      e["default"].Set(k.default_value);
      if (!k.values.empty()) { e["values"].Set(k.values); }
      e["description"].Set(k.description);
   }
   std::cout << out.Dump();
}

} // namespace

int main(int argc, char* argv[])
{
   incns::ConfigureMpiTransport(); // before MPI_Init
   Mpi::Init(argc, argv);
   Hypre::Init();

   if (argc < 2 || !std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h"))
   {
      if (Mpi::Root())
      {
         std::cerr << "usage: run_case <deck.yaml>\n"
                   "       run_case <deck.yaml> --summary <file.json>\n"
                   "       run_case <deck.yaml> --check   (vet the deck, JSON; no run)\n"
                   "       run_case --deck-reference   (every deck key, Markdown)\n"
                   "       run_case --deck-schema      (every deck key, JSON)"
                   << std::endl;
      }
      return argc < 2 ? 1 : 0;
   }
   if (!std::strcmp(argv[1], "--deck-reference"))
   {
      if (Mpi::Root()) { std::cout << incns::Parameters::DeckReferenceMarkdown(); }
      return 0;
   }
   if (!std::strcmp(argv[1], "--deck-schema"))
   {
      if (Mpi::Root()) { PrintDeckSchema(); }
      return 0;
   }
   for (int i = 2; i < argc; ++i)
   {
      if (!std::strcmp(argv[i], "--check")) { return CheckDeck(argv[1]); }
   }

   std::string summary;
   for (int i = 2; i + 1 < argc; ++i)
   {
      if (!std::strcmp(argv[i], "--summary")) { summary = argv[i + 1]; }
   }
   const incns::Parameters params = incns::Parameters::LoadYAML(argv[1]);
   if (summary.empty())
   {
      summary = params.output.path + "/" + params.output.name + "_summary.json";
   }
   incns::ConfigureDevice(params.device);

   // The one mesh factory: nonconforming-ready when params.amr is enabled.
   std::unique_ptr<ParMesh> pmesh = incns::MakeCaseMesh(params);
   ParMesh& mesh = *pmesh;

   // A deck must say what happens on every real boundary: with no
   // boundary_conditions at all, walls would silently become do-nothing.
   if (params.boundary_conditions.empty())
   {
      const std::vector<std::string> names = incns::BoundaryNames(params);
      if (!names.empty())
      {
         std::string list;
         for (const std::string& n : names) { list += (list.empty() ? "" : ", ") + n; }
         MFEM_ABORT("run_case: the mesh has boundaries (" << list << ") but the "
                    "deck declares no boundary_conditions");
      }
   }

   // A solid: a static solve or a dynamic march (solid/elasticity_case).
   if (incns::IsSolid(params.equation))
   {
      incns::ElasticityCase solid(mesh, params);
      solid.WriteSummary(summary, "running");
      const bool ok = solid.Run();
      solid.WriteSummary(summary, ok ? "ok" : "diverged");
      if (Mpi::Root())
      {
         std::cout << "run_case: "
                   << (params.equation == incns::Equation::Hyperelasticity ?
                       "hyperelasticity" : "linear_elasticity") << "  status="
                   << (ok ? "ok" : "diverged") << "  summary: " << summary
                   << std::endl;
      }
      return ok ? 0 : 2;
   }

   // A mesh deformation: prescribed boundary displacements extended into the
   // domain (ale/mesh_motion_case).
   if (params.equation == incns::Equation::MeshMotion)
   {
      incns::MeshMotionCase motion(mesh, params);
      motion.WriteSummary(summary, "running");
      const bool ok = motion.Run();
      motion.WriteSummary(summary, ok ? "ok" : "diverged");
      if (Mpi::Root())
      {
         std::cout << "run_case: mesh_motion  status="
                   << (ok ? "ok" : "diverged") << "  summary: " << summary
                   << std::endl;
      }
      return ok ? 0 : 2;
   }

   // Fluid-structure interaction: the flow and the solid of section solid,
   // coupled in every step (fsi/fsi_case).
   if (params.fsi.enabled)
   {
      incns::FsiCase fsi(mesh, params);
      fsi.WriteSummary(summary, "running");
      const bool ok = fsi.Run();
      fsi.WriteSummary(summary, ok ? "ok" : "diverged");
      if (Mpi::Root())
      {
         std::cout << "run_case: fsi  t=" << fsi.Fluid().Time() << "  status="
                   << (ok ? "ok" : "diverged") << "  summary: " << summary
                   << std::endl;
      }
      return ok ? 0 : 2;
   }

   incns::Case flow_case(mesh, params);
   auto u0 = incns::MakeInitialVelocity(params);
   flow_case.SetInitialVelocity(*u0);
   // The summary exists from the start: a run that dies leaves "running".
   flow_case.WriteSummary(summary, "running");
   flow_case.Run();
   const bool diverged = flow_case.Diverged();
   flow_case.WriteSummary(summary, diverged ? "diverged" : "ok");

   // The profile goes next to the summary: <...>_summary.json ->
   // <...>_profile.csv (any other summary name gets ".profile.csv" appended).
   std::string profile;
   if (params.output.profile != incns::ProfileOutput::Off)
   {
      const std::string tail = "_summary.json";
      const bool named = summary.size() >= tail.size() &&
                         summary.compare(summary.size() - tail.size(),
                                         tail.size(), tail) == 0;
      profile = named ? summary.substr(0, summary.size() - tail.size()) +
                "_profile.csv" : summary + ".profile.csv";
      flow_case.WriteProfile(profile);
   }

   if (Mpi::Root())
   {
      std::cout << "run_case: t=" << flow_case.Time() << "  status="
                << (diverged ? "diverged" : "ok") << "  summary: " << summary;
      if (!profile.empty()) { std::cout << "  profile: " << profile; }
      std::cout << std::endl;
   }
   return diverged ? 2 : 0;
}
