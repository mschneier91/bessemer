#include "amr/refinement_marker.hpp"

#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace incns
{

using namespace mfem;

void RefinementMarker::ElementExtents(Mesh& mesh, Vector& h)
{
   const int dim = mesh.Dimension(), ne = mesh.GetNE();
   h.SetSize(dim * ne);
   for (int e = 0; e < ne; ++e)
   {
      ElementTransformation& T = *mesh.GetElementTransformation(e);
      const IntegrationPoint& c = Geometries.GetCenter(mesh.GetElementGeometry(e));
      T.SetIntPoint(&c);
      const DenseMatrix& J = T.Jacobian();
      for (int d = 0; d < dim; ++d)
      {
         double s = 0.0;
         for (int i = 0; i < J.Height(); ++i) { s += J(i, d) * J(i, d); }
         h(d + dim * e) = std::sqrt(s);
      }
   }
}

int RefinementMarker::ResolveAnisotropicConflicts(ParMesh& mesh,
      Array<Refinement>& refs)
{
   if (mesh.Dimension() < 3 || mesh.GetNRanks() == 1) { return 0; }
   int upgraded = 0;
   for (int round = 0; round < 5; ++round)
   {
      std::set<int> conflicts;
      if (!mesh.AnisotropicConflict(refs, conflicts)) { return upgraded; }
      for (int i : conflicts)
      {
         if (refs[i].GetType() != Refinement::XYZ)
         {
            refs[i].SetType(Refinement::XYZ);
            ++upgraded;
         }
      }
   }
   // Still conflicting after 5 rounds: give up on anisotropy for this pass.
   std::set<int> conflicts;
   if (mesh.AnisotropicConflict(refs, conflicts))
   {
      for (int i = 0; i < refs.Size(); ++i)
      {
         if (refs[i].GetType() != Refinement::XYZ)
         {
            refs[i].SetType(Refinement::XYZ);
            ++upgraded;
         }
      }
      if (Mpi::Root())
      {
         mfem::out << "[amr] WARNING: anisotropic conflicts persisted; this "
                   "pass refines isotropically" << std::endl;
      }
   }
   return upgraded;
}

// Relative width within which two indicators count as equal for the element
// cap: far above the reproducibility floor of a GPU run (~1.5e-11, see
// test/repro_tolerance.hpp), far below any difference the indicator resolves.
static constexpr double kTieBand = 1e-8;
// How far past max_elements a group of equal indicators may take the mesh.
static constexpr double kCapSlack = 0.1;

void RefinementMarker::Mark(ParMesh& mesh, const Vector& g,
                            Array<Refinement>& refs, MarkStats* stats) const
{
   INCNS_PROFILE("amr::mark");
   const int dim = mesh.Dimension(), ne = mesh.GetNE();
   const MPI_Comm comm = mesh.GetComm();
   MFEM_VERIFY(g.Size() == dim * ne, "refinement_marker: indicator size "
               << g.Size() << " does not match dim * NE = " << dim * ne);
   refs.SetSize(0);
   const double* G = g.HostRead();

   // eta and the per-element direction mask (anisotropy, then min size).
   std::vector<double> eta(ne);
   std::vector<int> mask(ne, 0);
   Vector h;
   if (opts_.min_size > 0.0) { ElementExtents(mesh, h); }
   double max_eta = 0.0;
   for (int e = 0; e < ne; ++e)
   {
      double s = 0.0, gmax = 0.0;
      for (int d = 0; d < dim; ++d)
      {
         s += G[d + dim * e] * G[d + dim * e];
         gmax = std::max(gmax, G[d + dim * e]);
      }
      eta[e] = std::sqrt(s);
      max_eta = std::max(max_eta, eta[e]);
      for (int d = 0; d < dim; ++d)
      {
         const bool wanted = !opts_.anisotropic ||
                             G[d + dim * e] >= opts_.aniso_ratio * gmax;
         const bool allowed = opts_.min_size <= 0.0 ||
                              0.5 * h(d + dim * e) >= opts_.min_size;
         if (wanted && allowed) { mask[e] |= (1 << d); }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &max_eta, 1, MPI_DOUBLE, MPI_MAX, comm);

   long long ne_global = ne;
   MPI_Allreduce(MPI_IN_PLACE, &ne_global, 1, MPI_LONG_LONG, MPI_SUM, comm);

   double threshold = (opts_.threshold_mode == AmrThreshold::Relative)
                      ? opts_.theta * max_eta : opts_.tolerance;

   // Global projected element count if every element with eta >= t (and a
   // non-empty mask) is split.
   auto projected = [&](double t)
   {
      long long add = 0;
      for (int e = 0; e < ne; ++e)
      {
         if (mask[e] != 0 && eta[e] > 0.0 && eta[e] >= t)
         {
            int ndir = 0;
            for (int d = 0; d < dim; ++d) { ndir += (mask[e] >> d) & 1; }
            add += (1LL << ndir) - 1;
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &add, 1, MPI_LONG_LONG, MPI_SUM, comm);
      return ne_global + add;
   };

   if (max_eta > 0.0 && opts_.max_elements > 0 &&
       projected(threshold) > opts_.max_elements)
   {
      // Raise the threshold: largest indicators first. lo marks too many,
      // hi marks nothing (or fits); 60 bisections pin it to roundoff.
      double lo = threshold, hi = max_eta * (1.0 + 1e-12) + 1e-300;
      for (int it = 0; it < 60; ++it)
      {
         const double mid = 0.5 * (lo + hi);
         if (projected(mid) > opts_.max_elements) { lo = mid; }
         else { hi = mid; }
      }
      threshold = hi;
      // Indicators that agree to roundoff (a symmetric flow has many) are
      // marked together. The bisection can end inside such a group, and which
      // of its members lie above hi is then decided by their last bits: bits
      // that differ on a restart or on another backend, where the state is
      // reproduced to ~1e-11 and not bitwise. When hi splits a group, the
      // whole group goes one way: marked, if that exceeds the cap by at most
      // kCapSlack (human 2026-10-10: a small overshoot is fine), and left out
      // otherwise (in a symmetric flow one group can be a third of the mesh).
      // Either way the marked set is a function of the flow, not of roundoff.
      long long near[2] = {0, 0}; // within the band: [0] at or above hi, [1] below
      for (int e = 0; e < ne; ++e)
      {
         if (mask[e] == 0 || eta[e] <= 0.0) { continue; }
         if (eta[e] >= hi && eta[e] < hi * (1.0 + kTieBand)) { ++near[0]; }
         else if (eta[e] < hi && eta[e] >= hi * (1.0 - kTieBand)) { ++near[1]; }
      }
      MPI_Allreduce(MPI_IN_PLACE, near, 2, MPI_LONG_LONG, MPI_SUM, comm);
      if (near[0] > 0 && near[1] > 0)
      {
         const double with_group = hi * (1.0 - kTieBand);
         const bool fits = static_cast<double>(projected(with_group)) <=
                           (1.0 + kCapSlack) * opts_.max_elements;
         threshold = fits ? with_group : hi * (1.0 + kTieBand);
      }
   }

   MarkStats st;
   st.max_eta = max_eta;
   st.threshold = threshold;
   st.ne_before = ne_global;
   if (max_eta > 0.0)
   {
      for (int e = 0; e < ne; ++e)
      {
         if (mask[e] != 0 && eta[e] > 0.0 && eta[e] >= threshold)
         {
            refs.Append(Refinement(e, static_cast<char>(mask[e])));
         }
      }
   }
   st.conflict_upgrades = ResolveAnisotropicConflicts(mesh, refs);

   long long local[5] = {refs.Size(), 0, 0, 0, 0};
   for (int i = 0; i < refs.Size(); ++i)
   {
      const int t = refs[i].GetType();
      int ndir = 0;
      for (int d = 0; d < dim; ++d)
      {
         if ((t >> d) & 1) { ++local[1 + d]; ++ndir; }
      }
      local[4] += (1LL << ndir) - 1;
   }
   MPI_Allreduce(MPI_IN_PLACE, local, 5, MPI_LONG_LONG, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &st.conflict_upgrades, 1, MPI_INT, MPI_SUM,
                 comm);
   st.marked = local[0];
   for (int d = 0; d < 3; ++d) { st.per_dir[d] = local[1 + d]; }
   st.projected_ne = ne_global + local[4];
   if (stats) { *stats = st; }
}

} // namespace incns
