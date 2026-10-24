#include "solid/solid_checkpoint.hpp"

#include "post/checkpoint.hpp"
#include "solid/hyperelasticity.hpp"
#include "solid/linear_elasticity.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iomanip>

namespace incns
{

using namespace mfem;

namespace
{

/// @return Whether any rank holds entries of a vector of size @p n (a rank
///         without dofs holds none of a vector that is there).
bool Present(int n, MPI_Comm comm)
{
   int any = n > 0 ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &any, 1, MPI_INT, MPI_MAX, comm);
   return any != 0;
}

} // namespace

template <class Solver>
void WriteSolidState(const std::string& dir, const std::string& name,
                     Solver& solver)
{
   const MPI_Comm comm = solver.Space().GetComm();
   int rank = 0, np = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &np);
   const typename Solver::State s = solver.ExportState();
   auto file = [&](const std::string & what)
   {
      return Checkpoint::RankFile(dir, name + "_" + what, rank);
   };
   for (std::size_t j = 0; j < s.u_hist.size(); ++j)
   {
      Checkpoint::WriteVector(file("u" + std::to_string(j)), s.u_hist[j]);
   }
   for (std::size_t j = 0; j < s.v_hist.size(); ++j)
   {
      Checkpoint::WriteVector(file("v" + std::to_string(j)), s.v_hist[j]);
   }
   // The parts a scheme may not have: every rank writes its file, also an
   // empty one, when any rank holds entries.
   const bool has_a = Present(s.a_n.Size(), comm);
   const bool has_f = Present(s.f_n.Size(), comm);
   const bool has_load = Present(s.load.Size(), comm);
   const bool has_reaction = Present(s.reaction.Size(), comm);
   if (has_a) { Checkpoint::WriteVector(file("a"), s.a_n); }
   if (has_f) { Checkpoint::WriteVector(file("f"), s.f_n); }
   if (has_load) { Checkpoint::WriteVector(file("load"), s.load); }
   if (has_reaction) { Checkpoint::WriteVector(file("reaction"), s.reaction); }
   Vector x;
   solver.Displacement().GetTrueDofs(x);
   Checkpoint::WriteVector(file("x"), x);
   if (rank == 0)
   {
      std::ofstream m(dir + "/" + name + ".yaml");
      MFEM_VERIFY(m.good(), "solid_checkpoint: cannot write " << dir << "/"
                  << name << ".yaml");
      m << std::setprecision(17)
        << "# incns solid state (same-np restart contract)\n"
        << "np: " << np << "\n"
        << "initialized: " << (s.initialized ? 1 : 0) << "\n"
        << "time: " << s.time << "\n"
        << "steps: " << s.steps << "\n"
        << "displacement_levels: " << s.u_hist.size() << "\n"
        << "velocity_levels: " << s.v_hist.size() << "\n"
        << "times: [";
      for (std::size_t j = 0; j < s.t_hist.size(); ++j)
      {
         m << (j ? ", " : "") << s.t_hist[j];
      }
      m << "]\n"
        << "acceleration: " << (has_a ? 1 : 0) << "\n"
        << "load_now: " << (has_f ? 1 : 0) << "\n"
        << "load: " << (has_load ? 1 : 0) << "\n"
        << "reaction: " << (has_reaction ? 1 : 0) << "\n";
      MFEM_VERIFY(m.good(), "solid_checkpoint: writing " << name << ".yaml failed");
   }
}

template <class Solver>
void ReadSolidState(const std::string& dir, const std::string& name,
                    Solver& solver)
{
   const MPI_Comm comm = solver.Space().GetComm();
   int rank = 0, np = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &np);
   const YAML::Node m = YAML::LoadFile(dir + "/" + name + ".yaml");
   MFEM_VERIFY(m["np"].as<int>() == np, "solid_checkpoint: written at np = "
               << m["np"].as<int>() << " but restarting at np = " << np
               << " -- same-np restart only");
   // The solver's own state gives the layout; the checkpoint's replaces it.
   typename Solver::State s = solver.ExportState();
   const bool dynamic = m["initialized"].as<int>() != 0;
   MFEM_VERIFY(dynamic == s.initialized, "solid_checkpoint: the checkpoint's "
               "solid is " << (dynamic ? "dynamic" : "static") << ", this "
               "run's is not");
   const int n = solver.Space().GetTrueVSize();
   auto file = [&](const std::string & what)
   {
      return Checkpoint::RankFile(dir, name + "_" + what, rank);
   };
   Vector x;
   Checkpoint::ReadVector(file("x"), n, x);
   if (!dynamic)
   {
      solver.SetSolution(x); // where the next static solve starts
      return;
   }
   s.time = m["time"].as<double>();
   s.steps = m["steps"].as<int>();
   s.t_hist.clear();
   for (const auto& t : m["times"]) { s.t_hist.push_back(t.as<double>()); }
   s.u_hist.resize(m["displacement_levels"].as<std::size_t>());
   for (std::size_t j = 0; j < s.u_hist.size(); ++j)
   {
      Checkpoint::ReadVector(file("u" + std::to_string(j)), n, s.u_hist[j]);
   }
   s.v_hist.resize(m["velocity_levels"].as<std::size_t>());
   for (std::size_t j = 0; j < s.v_hist.size(); ++j)
   {
      Checkpoint::ReadVector(file("v" + std::to_string(j)), n, s.v_hist[j]);
   }
   auto optional = [&](const char* key, const char* what, Vector & v)
   {
      if (m[key].as<int>() != 0) { Checkpoint::ReadVector(file(what), n, v); }
      else { v.SetSize(0); }
   };
   optional("acceleration", "a", s.a_n);
   optional("load_now", "f", s.f_n);
   optional("load", "load", s.load);
   optional("reaction", "reaction", s.reaction);
   solver.ImportState(s);
}

template void WriteSolidState<LinearElasticity>(const std::string&,
      const std::string&,
      LinearElasticity&);
template void WriteSolidState<Hyperelasticity>(const std::string&,
      const std::string&,
      Hyperelasticity&);
template void ReadSolidState<LinearElasticity>(const std::string&,
      const std::string&,
      LinearElasticity&);
template void ReadSolidState<Hyperelasticity>(const std::string&,
      const std::string&,
      Hyperelasticity&);

} // namespace incns
