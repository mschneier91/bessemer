/**
 * @file output.hpp
 * @brief ParaView output writer (high-order + levels of detail).
 */
#pragma once

#include "config/parameters.hpp"
#include "mfem.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Writes velocity/pressure snapshots as a ParaViewDataCollection.
 *
 * High-order output is ALWAYS enabled (SetHighOrderOutput(true)) with a
 * levels-of-detail refinement >= k_u -- without both, Q4/Q5 fields render as
 * low-order mush that looks exactly like a solver bug and is not one. Do not
 * "fix" the solver over a visualization artifact; check these settings first.
 */
class OutputWriter
{
public:
   /**
    * @brief Create the collection and register the fields.
    * @param mesh    The mesh (borrowed).
    * @param u       Velocity field to write (borrowed).
    * @param p       Pressure field to write (borrowed).
    * @param out     Path/name/interval settings.
    * @param order_u Velocity order k_u; the levels of detail are >= this.
    * @param nd      Reference scales; recorded in a scales.yaml sidecar next
    *                to the collection (fields are written nondimensional).
    * @param restart Continue an existing collection (keep its earlier time
    *                steps): used when the writer is rebuilt after an AMR
    *                event changed the mesh and the fields.
    */
   OutputWriter(mfem::ParMesh& mesh, mfem::ParGridFunction& u,
                mfem::ParGridFunction& p, const OutputParameters& out,
                int order_u, const Nondimensionalization& nd,
                bool restart = false);

   /**
    * @brief Also write @p gf (e.g. AMR cell data) at every save.
    * @param name Field name in the collection.
    * @param gf   Field (borrowed; must outlive this writer).
    */
   void RegisterExtra(const std::string& name, mfem::ParGridFunction* gf);

   /**
    * @brief Save a snapshot if the cycle is on the configured interval
    *        (cycle 0 -- the initial state -- always saves).
    * @param cycle Step counter.
    * @param time  Physical time of the snapshot.
    */
   void MaybeSave(int cycle, double time);

private:
   OutputParameters out_;                              ///< Settings.
   Nondimensionalization nd_;                          ///< Scales for the sidecar.
   bool wrote_scales_ = false;                         ///< Sidecar written yet?
   int rank_ = 0;                                      ///< For rank-0-only I/O.
   std::unique_ptr<mfem::ParaViewDataCollection> pv_;  ///< The collection.
};

} // namespace incns
