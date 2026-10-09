// Generic YAML-driven driver: `run_case <deck.yaml>` -- no recompilation per
// case. The deck supplies parameters, mesh, initial condition (by name), and
// output settings; the case goes through the same Case surface as any
// in-code driver. `run_case --deck-reference` prints every deck key with its
// default (docs/deck_reference.md is that output). Every run writes a
// machine-readable summary (<output.path>/<output.name>_summary.json, or
// --summary <file>): "running" at the start, then "ok" or "diverged" (exit
// code 2).

#include "bc/boundary_names.hpp"
#include "config/initial_conditions.hpp"
#include "config/parameters.hpp"
#include "mesh/case_mesh.hpp"
#include "solver/case.hpp"
#include "util/device.hpp"
#include "mfem.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if (argc < 2 || !std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h"))
   {
      if (Mpi::Root())
      {
         std::cerr << "usage: run_case <deck.yaml>\n"
                   "       run_case <deck.yaml> --summary <file.json>\n"
                   "       run_case --deck-reference   (every deck key, Markdown)"
                   << std::endl;
      }
      return argc < 2 ? 1 : 0;
   }
   if (!std::strcmp(argv[1], "--deck-reference"))
   {
      if (Mpi::Root()) { std::cout << incns::Parameters::DeckReferenceMarkdown(); }
      return 0;
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

   incns::Case flow_case(mesh, params);
   auto u0 = incns::MakeInitialVelocity(params);
   flow_case.SetInitialVelocity(*u0);
   // The summary exists from the start: a run that dies leaves "running".
   flow_case.WriteSummary(summary, "running");
   flow_case.Run();
   const bool diverged = flow_case.Diverged();
   flow_case.WriteSummary(summary, diverged ? "diverged" : "ok");

   if (Mpi::Root())
   {
      std::cout << "run_case: t=" << flow_case.Time() << "  status="
                << (diverged ? "diverged" : "ok") << "  summary: " << summary
                << std::endl;
   }
   return diverged ? 2 : 0;
}
