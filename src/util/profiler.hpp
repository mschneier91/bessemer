/**
 * @file profiler.hpp
 * @brief Nested scoped wall-time profiler with MPI max/min reduction.
 */
#ifndef INCNS_UTIL_PROFILER_HPP
#define INCNS_UTIL_PROFILER_HPP

#include <mpi.h>

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @brief One region's reduced timing, as returned by Profiler::Report.
 *
 * Times are wall-clock seconds reduced across ranks (max and min expose load
 * imbalance). Inclusive counts time spent anywhere inside the region; exclusive
 * subtracts the time attributed to child regions.
 */
struct RegionStats
{
   std::string name;      ///< Region label.
   int depth = 0;         ///< Nesting depth (top-level regions are depth 0).
   long long calls = 0;   ///< Number of times the region was entered (local rank).
   double incl_max = 0.0; ///< Inclusive time, max over ranks.
   double incl_min = 0.0; ///< Inclusive time, min over ranks.
   double excl_max = 0.0; ///< Exclusive time, max over ranks.
   double excl_min = 0.0; ///< Exclusive time, min over ranks.
   double pct_of_parent = 0.0; ///< Inclusive (max) as a percent of the parent's.
};

/**
 * @brief Process-wide nested wall-time profiler.
 *
 * Regions are opened with the INCNS_PROFILE() macro, which creates a
 * ScopedRegion; entering pushes a child of the current region and exiting pops
 * back, so the dynamic call nesting forms a tree. Each node accumulates
 * inclusive time and an entry count. Report() folds the tree across ranks with
 * MPI max/min reductions; Print() renders it on rank 0.
 *
 * Not thread-safe: intended for the single-threaded-per-rank MPI model. The
 * clock is injectable (SetClock) so tests are deterministic rather than
 * dependent on wall-clock timing.
 */
class Profiler
{
public:
   /// Clock signature: returns a monotonically increasing time in seconds.
   using ClockFn = std::function<double()>;

   /// @return The process-wide profiler instance.
   static Profiler& Instance();

   /**
    * @brief Override the time source (test hook).
    * @param clock Callable returning seconds; e.g. a controlled fake in tests.
    */
   void SetClock(ClockFn clock);

   /// Clear all recorded regions and restore the default (MPI_Wtime) clock.
   void Reset();

   /**
    * @brief Open a region as a child of the current one (used by ScopedRegion).
    * @param name Region label (borrowed for the duration of the call).
    * @return The clock reading at entry, to be passed back to Exit().
    */
   double Enter(const char* name);

   /**
    * @brief Close the current region and accumulate its elapsed time.
    * @param start The value returned by the matching Enter().
    */
   void Exit(double start);

   /**
    * @brief Reduce the region tree across ranks. Collective over @p comm.
    * @param comm MPI communicator (defaults to MPI_COMM_WORLD).
    * @return Regions in depth-first order; identical on every rank.
    */
   std::vector<RegionStats> Report(MPI_Comm comm = MPI_COMM_WORLD) const;

   /**
    * @brief Print the reduced region tree. Collective; output only on rank 0.
    * @param os   Output stream (written on rank 0 only).
    * @param comm MPI communicator (defaults to MPI_COMM_WORLD).
    */
   void Print(std::ostream& os, MPI_Comm comm = MPI_COMM_WORLD) const;

private:
   Profiler();

   /// A single region in the tree.
   struct Node
   {
      std::string name;                            ///< Region label.
      Node* parent = nullptr;                      ///< Owning parent (null at root).
      std::vector<std::unique_ptr<Node>> children; ///< Child regions.
      double inclusive = 0.0;                      ///< Accumulated inclusive time.
      long long calls = 0;                         ///< Entry count.

      /**
       * @brief Find a child region by name, creating it if absent.
       * @param child_name Region label.
       * @return The child with @p child_name.
       */
      Node* FindOrAdd(const char* child_name);
   };

   /**
    * @brief Depth-first flatten of the tree into parallel arrays (helper for
    *        Report): each node's pointer, nesting depth, and the flat index of
    *        its parent (-1 for a top-level region).
    * @param node         Subtree root to flatten (its children are visited).
    * @param depth        Nesting depth of @p node's children.
    * @param parent_index Flat index of @p node (-1 for the sentinel root).
    * @param nodes        Output: node pointers, depth-first.
    * @param depths       Output: matching nesting depths.
    * @param parents      Output: matching parent flat indices.
    */
   static void Flatten(const Node* node, int depth, int parent_index,
                       std::vector<const Node*>& nodes,
                       std::vector<int>& depths, std::vector<int>& parents);

   std::unique_ptr<Node> root_; ///< Sentinel root; real regions are its subtree.
   Node* current_;              ///< Currently open region.
   ClockFn clock_;             ///< Time source.
};

/**
 * @brief RAII guard that times the enclosing scope. Prefer the INCNS_PROFILE
 *        macro over constructing this directly.
 */
class ScopedRegion
{
public:
   /**
    * @brief Open a region named @p name on the process-wide Profiler.
    * @param name Region label (borrowed for the ScopedRegion's lifetime).
    */
   explicit ScopedRegion(const char* name)
      : start_(Profiler::Instance().Enter(name)) {}

   /// Close the region.
   ~ScopedRegion() { Profiler::Instance().Exit(start_); }

   ScopedRegion(const ScopedRegion&) = delete;
   ScopedRegion& operator=(const ScopedRegion&) = delete;

private:
   double start_; ///< Clock reading at construction.
};

} // namespace incns

// --- INCNS_PROFILE toggle ---------------------------------------------------
/// @cond INCNS_INTERNAL
#define INCNS_PROFILE_CONCAT2_(a, b) a##b
#define INCNS_PROFILE_CONCAT_(a, b) INCNS_PROFILE_CONCAT2_(a, b)
/// @endcond

#ifdef INCNS_ENABLE_PROFILING
/**
 * @brief Time the enclosing scope under region @p name via a ScopedRegion.
 *
 * Active only when INCNS_ENABLE_PROFILING is defined for the translation unit;
 * otherwise it expands to a no-op that still validates @p name, so code compiles
 * both ways.
 * @param name String-literal region label.
 */
#define INCNS_PROFILE(name) \
   ::incns::ScopedRegion INCNS_PROFILE_CONCAT_(incns_prof_, __LINE__)(name)
#else
#define INCNS_PROFILE(name) ((void)sizeof(name))
#endif

#endif // INCNS_UTIL_PROFILER_HPP
