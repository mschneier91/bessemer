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
   // HostRead(), not GetData(): on a device backend the state lives in device
   // memory and the host buffer is never populated, so GetData() would dump
   // zeros to the file.
   f.write(reinterpret_cast<const char*>(v.HostRead()),
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
   // The restored vector is handed to the integrator and prolongated on the
   // device, so it must be device-aware; HostWrite() then declares the host
   // copy authoritative, and the host->device copy happens on first use.
   v.UseDevice(true);
   f.read(reinterpret_cast<char*>(v.HostWrite()),
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

   // One definition of the marching state (shared with AMR events). Its
   // pressure is the solver's own variable -- the Bernoulli head in the
   // rotational form, the physical pressure otherwise -- so a restart's warm
   // start is exactly the uninterrupted run's.
   const IntegratorState state = integrator.ExportState();
   const auto& hist = state.u_hist;
   const auto& times = state.times;
   MFEM_VERIFY(!hist.empty(), "checkpoint: nothing to write (no history)");

   for (std::size_t j = 0; j < hist.size(); ++j)
   {
      WriteVector(BinPath(dir, "u_hist" + std::to_string(j), rank), hist[j]);
   }
   WriteVector(BinPath(dir, "pressure", rank), state.pressure);

   if (rank == 0)
   {
      std::ofstream meta(dir + "/meta.yaml");
      MFEM_VERIFY(meta.good(), "checkpoint: cannot write metadata");
      meta << std::setprecision(17);
      meta << "# incns rolling checkpoint (same-np restart contract)\n"
           << "np: " << nranks << "\n"
           << "t: " << state.times[0] << "\n"
           << "next_dt: " << state.next_dt << "\n"
           << "completed_steps: " << state.completed_steps << "\n"
           << "history_times: [";
      for (std::size_t j = 0; j < times.size(); ++j)
      {
         meta << (j ? ", " : "") << times[j];
      }
      meta << "]\n";
      if (state.has_controller)
      {
         meta << "adaptive_prev_es: " << state.controller.prev_es << "\n";
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

   IntegratorState state;
   for (const auto& tn : meta["history_times"])
   {
      state.times.push_back(tn.as<double>());
   }
   state.completed_steps = meta["completed_steps"].as<int>();
   state.next_dt = meta["next_dt"].as<double>();

   const int n_u = integrator.Velocity().ParFESpace()->GetTrueVSize();
   const int n_p = integrator.Pressure().ParFESpace()->GetTrueVSize();
   state.u_hist.resize(state.times.size());
   for (std::size_t j = 0; j < state.times.size(); ++j)
   {
      ReadVector(BinPath(dir, "u_hist" + std::to_string(j), rank), n_u,
                 state.u_hist[j]);
   }
   ReadVector(BinPath(dir, "pressure", rank), n_p, state.pressure);

   // The controller's step record is not persisted: a restart starts a fresh
   // record, with the PI memory restored so the next step matches.
   state.has_controller = (integrator.Controller() != nullptr);
   if (state.has_controller)
   {
      state.controller = integrator.Controller()->ExportState();
      if (meta["adaptive_prev_es"])
      {
         state.controller.prev_es = meta["adaptive_prev_es"].as<double>();
      }
   }
   integrator.ImportState(state);
}

} // namespace incns
