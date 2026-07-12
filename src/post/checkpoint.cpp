#include "post/checkpoint.hpp"

#include "util/profiler.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

namespace incns
{

using namespace mfem;

namespace
{

std::string BinPath(const std::string& dir, const std::string& base, int rank)
{
   return dir + "/" + base + ".r" + std::to_string(rank) + ".bin";
}

// Raw binary vector dump: int64 size, then the doubles. Bitwise-exact, and the
// size prefix doubles as the per-rank partition check on read.
void WriteVector(const std::string& path, const Vector& v)
{
   std::ofstream f(path, std::ios::binary);
   MFEM_VERIFY(f.good(), "checkpoint: cannot open " << path << " for writing");
   const std::int64_t n = v.Size();
   f.write(reinterpret_cast<const char*>(&n), sizeof(n));
   f.write(reinterpret_cast<const char*>(v.GetData()),
           static_cast<std::streamsize>(n * sizeof(double)));
   MFEM_VERIFY(f.good(), "checkpoint: write failed for " << path);
}

void ReadVector(const std::string& path, int expected_size, Vector& v)
{
   std::ifstream f(path, std::ios::binary);
   MFEM_VERIFY(f.good(), "checkpoint: cannot open " << path
               << " (missing checkpoint or wrong rank count?)");
   std::int64_t n = 0;
   f.read(reinterpret_cast<char*>(&n), sizeof(n));
   MFEM_VERIFY(n == expected_size,
               "checkpoint: " << path << " holds " << n << " dofs but this "
               "rank owns " << expected_size
               << " -- rank count or partition mismatch");
   v.SetSize(static_cast<int>(n));
   f.read(reinterpret_cast<char*>(v.GetData()),
          static_cast<std::streamsize>(n * sizeof(double)));
   MFEM_VERIFY(f.good(), "checkpoint: short read from " << path);
}

} // namespace

void Checkpoint::Write(const std::string& dir,
                       StokesTimeIntegrator& integrator,
                       const Parameters& params)
{
   INCNS_PROFILE("checkpoint::write");

   const MPI_Comm comm = MPI_COMM_WORLD;
   int rank = 0, nranks = 0;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);

   if (rank == 0) { std::filesystem::create_directories(dir); }
   MPI_Barrier(comm); // directory exists before any rank writes

   const auto& hist = integrator.History();
   const auto& times = integrator.HistoryTimes();
   MFEM_VERIFY(!hist.empty(), "checkpoint: nothing to write (no history)");

   for (std::size_t j = 0; j < hist.size(); ++j)
   {
      WriteVector(BinPath(dir, "u_hist" + std::to_string(j), rank), hist[j]);
   }
   Vector p_true(integrator.Pressure().ParFESpace()->GetTrueVSize());
   integrator.Pressure().GetTrueDofs(p_true);
   WriteVector(BinPath(dir, "pressure", rank), p_true);

   if (rank == 0)
   {
      std::ofstream meta(dir + "/meta.yaml");
      MFEM_VERIFY(meta.good(), "checkpoint: cannot write metadata");
      meta << std::setprecision(17);
      meta << "# incns rolling checkpoint (same-np restart contract)\n"
           << "np: " << nranks << "\n"
           << "t: " << integrator.Time() << "\n"
           << "next_dt: " << integrator.CurrentDt() << "\n"
           << "completed_steps: " << integrator.StepCount() << "\n"
           << "history_times: [";
      for (std::size_t j = 0; j < times.size(); ++j)
      {
         meta << (j ? ", " : "") << times[j];
      }
      meta << "]\n";
      if (integrator.Controller())
      {
         meta << "adaptive_prev_es: "
              << integrator.Controller()->PrevScaledError() << "\n";
      }
      // Reference scales, so a consumer can re-dimensionalize the state.
      meta << "Re: " << params.nondim.Re << "\nL_ref: " << params.nondim.L_ref
           << "\nU_ref: " << params.nondim.U_ref << "\n";
   }
   MPI_Barrier(comm); // checkpoint complete on return, on every rank
}

void Checkpoint::Read(const std::string& dir, StokesTimeIntegrator& integrator)
{
   INCNS_PROFILE("checkpoint::read");

   const MPI_Comm comm = MPI_COMM_WORLD;
   int rank = 0, nranks = 0;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);

   const YAML::Node meta = YAML::LoadFile(dir + "/meta.yaml");
   MFEM_VERIFY(meta["np"].as<int>() == nranks,
               "checkpoint: written at np = " << meta["np"].as<int>()
               << " but restarting at np = " << nranks
               << " -- same-np restart only");

   std::vector<double> times;
   for (const auto& tn : meta["history_times"])
   {
      times.push_back(tn.as<double>());
   }
   const int completed = meta["completed_steps"].as<int>();
   const double next_dt = meta["next_dt"].as<double>();

   const int n_u = integrator.Velocity().ParFESpace()->GetTrueVSize();
   const int n_p = integrator.Pressure().ParFESpace()->GetTrueVSize();
   std::vector<Vector> states(times.size());
   for (std::size_t j = 0; j < times.size(); ++j)
   {
      ReadVector(BinPath(dir, "u_hist" + std::to_string(j), rank), n_u,
                 states[j]);
   }
   Vector p_true;
   ReadVector(BinPath(dir, "pressure", rank), n_p, p_true);

   integrator.SetHistory(states, times, completed, next_dt, &p_true);
   if (integrator.Controller() && meta["adaptive_prev_es"])
   {
      integrator.Controller()->RestorePrevScaledError(
                   meta["adaptive_prev_es"].as<double>());
   }
}

} // namespace incns
