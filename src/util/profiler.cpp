#include "util/profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace incns
{

Profiler::Node* Profiler::Node::FindOrAdd(const char* child_name)
{
   for (auto& c : children)
   {
      if (c->name == child_name) { return c.get(); }
   }
   auto node = std::make_unique<Node>();
   node->name = child_name;
   node->parent = this;
   children.push_back(std::move(node));
   return children.back().get();
}

Profiler::Profiler()
{
   Reset();
}

Profiler& Profiler::Instance()
{
   static Profiler instance;
   return instance;
}

void Profiler::SetClock(ClockFn clock)
{
   clock_ = std::move(clock);
}

void Profiler::SetSync(SyncFn sync)
{
   sync_ = std::move(sync);
}

void Profiler::Reset()
{
   root_ = std::make_unique<Node>();
   current_ = root_.get();
   clock_ = [] { return MPI_Wtime(); };
   sync_ = nullptr;
}

double Profiler::Enter(const char* name)
{
   current_ = current_->FindOrAdd(name);
   current_->calls += 1;
   // Work launched before this region belongs to its parent.
   if (sync_) { sync_(); }
   return clock_();
}

void Profiler::Exit(double start)
{
   // Work launched inside this region belongs to it.
   if (sync_) { sync_(); }
   current_->inclusive += clock_() - start;
   current_ = current_->parent;
}

// Flatten the tree depth-first, recording each node, its depth, and the index of
// its parent in the flat list (-1 for a top-level region). Order is identical on
// every rank because all ranks walk the same SPMD region tree.
void Profiler::Flatten(const Node* node, int depth, int parent_index,
                       std::vector<const Node*>& nodes,
                       std::vector<int>& depths, std::vector<int>& parents)
{
   for (const auto& child : node->children)
   {
      const int my_index = static_cast<int>(nodes.size());
      nodes.push_back(child.get());
      depths.push_back(depth);
      parents.push_back(parent_index);
      Flatten(child.get(), depth + 1, my_index, nodes, depths, parents);
   }
}

std::vector<RegionStats> Profiler::Report(MPI_Comm comm) const
{
   std::vector<const Node*> nodes;
   std::vector<int> depths;
   std::vector<int> parents;
   Flatten(root_.get(), 0, -1, nodes, depths, parents);

   const int n = static_cast<int>(nodes.size());
   std::vector<double> incl(n), excl(n);
   for (int i = 0; i < n; ++i)
   {
      double child_sum = 0.0;
      for (const auto& c : nodes[i]->children) { child_sum += c->inclusive; }
      incl[i] = nodes[i]->inclusive;
      excl[i] = nodes[i]->inclusive - child_sum;
   }

   std::vector<double> incl_max(incl), incl_min(incl);
   std::vector<double> excl_max(excl), excl_min(excl);
   if (n > 0)
   {
      MPI_Allreduce(MPI_IN_PLACE, incl_max.data(), n, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, incl_min.data(), n, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(MPI_IN_PLACE, excl_max.data(), n, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, excl_min.data(), n, MPI_DOUBLE, MPI_MIN, comm);
   }

   double top_total = 0.0;
   for (int i = 0; i < n; ++i)
   {
      if (depths[i] == 0) { top_total += incl_max[i]; }
   }

   std::vector<RegionStats> out(n);
   for (int i = 0; i < n; ++i)
   {
      out[i].name = nodes[i]->name;
      out[i].depth = depths[i];
      out[i].calls = nodes[i]->calls;
      out[i].incl_max = incl_max[i];
      out[i].incl_min = incl_min[i];
      out[i].excl_max = excl_max[i];
      out[i].excl_min = excl_min[i];

      const int p = parents[i];
      const double ref = (p >= 0) ? incl_max[p] : top_total;
      out[i].pct_of_parent = (ref > 0.0) ? 100.0 * incl_max[i] / ref : 0.0;
   }
   return out;
}

void Profiler::Print(std::ostream& os, MPI_Comm comm) const
{
   const std::vector<RegionStats> stats = Report(comm); // collective

   int rank = 0;
   MPI_Comm_rank(comm, &rank);
   if (rank != 0) { return; }

   os << "Profiler (wall seconds; incl/excl reduced max over ranks, "
      "[min] in brackets):\n";
   os << std::left << std::setw(36) << "region"
      << std::right << std::setw(8) << "calls"
      << std::setw(12) << "incl_max"
      << std::setw(12) << "incl_min"
      << std::setw(12) << "excl_max"
      << std::setw(9) << "%parent" << "\n";
   for (const RegionStats& s : stats)
   {
      std::string label(2 * s.depth, ' ');
      label += s.name;
      os << std::left << std::setw(36) << label
         << std::right << std::setw(8) << s.calls
         << std::setw(12) << std::fixed << std::setprecision(4) << s.incl_max
         << std::setw(12) << s.incl_min
         << std::setw(12) << s.excl_max
         << std::setw(8) << std::setprecision(1) << s.pct_of_parent << "%"
         << "\n";
   }
}

void Profiler::WriteCsv(std::ostream& os, MPI_Comm comm) const
{
   const std::vector<RegionStats> stats = Report(comm); // collective

   int rank = 0;
   MPI_Comm_rank(comm, &rank);
   if (rank != 0) { return; }

   os << "# bessemer profile: wall seconds per region, reduced over the MPI "
      "ranks (max, min)\n";
   os << "# mode: "
      << (sync_ ? "sync (the device is synchronized at every region boundary)"
          : "wall (on a GPU, asynchronous kernels are charged to the region "
          "that next waits for the device)")
      << "\n";
   os << "depth,region,path,calls,incl_max,incl_min,excl_max,excl_min,"
      "pct_of_parent\n";
   std::vector<std::string> ancestors; // ancestors[d]: the open region at depth d
   os << std::setprecision(9);
   for (const RegionStats& s : stats)
   {
      ancestors.resize(s.depth);
      std::string path;
      for (const std::string& a : ancestors) { path += a + "/"; }
      path += s.name;
      ancestors.push_back(s.name);
      os << s.depth << "," << s.name << "," << path << "," << s.calls << ","
         << s.incl_max << "," << s.incl_min << "," << s.excl_max << ","
         << s.excl_min << "," << s.pct_of_parent << "\n";
   }
}

} // namespace incns
