// mesh_census: where a checkpointed AMR run put its elements, pass by pass.
//
//   mpirun -np <the run's ranks> build/cpu/apps/mesh_census -d <deck.yaml>
//       -c <checkpoint dir> [-H <run>_history.csv] [-b -2,2,5,10,20,50] [-e 5]
//       [-o census.csv]                        (one command line)
//
// Rebuilds the run's mesh from its checkpoint's refinement record, one pass at
// a time (exactly as a restart does), and after each pass counts the elements
// per band of the streamwise coordinate x (from -x0; the square cylinder sits at
// the origin). With the run's history file, each pass gets the time from which
// the run had that many elements. Last, per band: elements, how many are
// refined, the largest |y| of a refined element, the smallest element and the
// deepest refinement.
//
// What it is for: checking an `amr.end_time` freeze (refinement-only AMR keeps
// growing behind the vortices). The freeze is safe for the forces if the bands
// near the body stopped growing well before it and the growth is downstream:
// benchmarks/square_cylinder_re1000/README.md, "Is the mesh freeze safe?".
// It reads the checkpoint only; it never writes or runs anything.

#include "amr/mesh_census.hpp"
#include "config/parameters.hpp"
#include "mfem.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

namespace
{
std::vector<double> ParseEdges(const std::string& s)
{
   std::vector<double> e;
   std::stringstream ss(s);
   std::string tok;
   while (std::getline(ss, tok, ',')) { e.push_back(std::stod(tok)); }
   return e;
}

std::string BandLabel(const std::vector<double>& e, std::size_t b)
{
   char buf[64];
   if (b == 0) { std::snprintf(buf, sizeof buf, "<%g", e[0]); }
   else if (b == e.size()) { std::snprintf(buf, sizeof buf, ">=%g", e.back()); }
   else { std::snprintf(buf, sizeof buf, "%g..%g", e[b - 1], e[b]); }
   return buf;
}

// (t, elements) of every row of a run's history file.
std::vector<std::pair<double, long long>> ReadHistory(const std::string& path)
{
   std::vector<std::pair<double, long long>> rows;
   std::ifstream f(path);
   MFEM_VERIFY(f, "mesh_census: cannot read " << path);
   std::string line;
   std::getline(f, line);
   std::vector<std::string> head;
   {
      std::stringstream ss(line);
      std::string h;
      while (std::getline(ss, h, ',')) { head.push_back(h); }
   }
   int it = -1, ie = -1;
   for (int i = 0; i < static_cast<int>(head.size()); ++i)
   {
      if (head[i] == "t") { it = i; }
      if (head[i] == "elements") { ie = i; }
   }
   MFEM_VERIFY(it >= 0 && ie >= 0, "mesh_census: " << path << " has no t and "
               "elements columns (a run's <name>_history.csv)");
   while (std::getline(f, line))
   {
      std::stringstream ss(line);
      std::string v;
      std::vector<std::string> cols;
      while (std::getline(ss, v, ',')) { cols.push_back(v); }
      if (static_cast<int>(cols.size()) <= std::max(it, ie)) { continue; }
      rows.emplace_back(std::stod(cols[it]), std::stoll(cols[ie]));
   }
   return rows;
}
} // namespace

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const char* deck = "";
   const char* chk = "";
   const char* history = "";
   const char* bands = "-2,2,5,10,20,50";
   const char* csv = "";
   double x0 = 0.0, y0 = 0.0;
   int every = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&deck, "-d", "--deck",
                  "The run's deck (its base mesh and AMR settings).");
   args.AddOption(&chk, "-c", "--checkpoint", "The run's checkpoint directory.");
   args.AddOption(&history, "-H", "--history",
                  "The run's <name>_history.csv: gives each pass its time.");
   args.AddOption(&bands, "-b", "--bands", "Band edges of x - x0, increasing.");
   args.AddOption(&x0, "-x0", "--x0", "Origin of x (the body's centre).");
   args.AddOption(&y0, "-y0", "--y0", "Origin of y.");
   args.AddOption(&every, "-e", "--every",
                  "Print every this many passes (and the last).");
   args.AddOption(&csv, "-o", "--csv", "Also write every pass to this CSV.");
   args.ParseCheck();
   MFEM_VERIFY(*deck && *chk, "mesh_census: give -d <deck> and -c <checkpoint>");

   const incns::Parameters p = incns::Parameters::LoadYAML(deck);
   const std::vector<double> edges = ParseEdges(bands);
   const std::vector<incns::MeshCensus> census =
      incns::ReplayMeshCensus(p, chk, edges, x0, y0);
   std::vector<std::pair<double, long long>> hist;
   if (*history && Mpi::Root()) { hist = ReadHistory(history); }
   if (!Mpi::Root()) { return 0; }

   auto time_of = [&](long long total) -> std::string
   {
      for (const auto& r : hist)
      {
         if (r.second == total)
         {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.2f", r.first);
            return buf;
         }
      }
      return "-";
   };
   const std::size_t nb = edges.size() + 1;
   std::printf("mesh census of %s (%zu refinement passes); bands of x - %g\n", chk,
               census.size() - 1, x0);
   std::printf("%5s %8s %9s", "pass", "t", "elements");
   for (std::size_t b = 0; b < nb; ++b) { std::printf(" %9s", BandLabel(edges, b).c_str()); }
   std::printf("\n");
   for (std::size_t i = 0; i < census.size(); ++i)
   {
      if (i % every != 0 && i + 1 != census.size()) { continue; }
      const incns::MeshCensus& c = census[i];
      std::printf("%5zu %8s %9lld", i, i == 0 ? "base" : time_of(c.total).c_str(),
                  c.total);
      for (std::size_t b = 0; b < nb; ++b) { std::printf(" %9lld", c.count[b]); }
      std::printf("\n");
   }
   const incns::MeshCensus& last = census.back();
   std::printf("\nper band, after the last pass:\n%12s %9s %9s %16s %10s %6s\n",
               "band",
               "elements", "refined", "max |y| refined", "min size", "depth");
   for (std::size_t b = 0; b < nb; ++b)
   {
      std::printf("%12s %9lld %9lld %16.2f %10.4f %6d\n", BandLabel(edges, b).c_str(),
                  last.count[b], last.refined[b], last.max_abs_y_refined[b],
                  last.min_size[b], last.max_depth[b]);
   }
   if (*csv)
   {
      std::ofstream f(csv);
      f << "pass,t,elements";
      for (std::size_t b = 0; b < nb; ++b) { f << "," << BandLabel(edges, b); }
      f << "\n";
      for (std::size_t i = 0; i < census.size(); ++i)
      {
         f << i << "," << (i == 0 ? "base" : time_of(census[i].total)) << ","
           << census[i].total;
         for (std::size_t b = 0; b < nb; ++b) { f << "," << census[i].count[b]; }
         f << "\n";
      }
      std::printf("wrote %s\n", csv);
   }
   return 0;
}
