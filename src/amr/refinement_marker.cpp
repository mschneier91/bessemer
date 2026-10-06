#include "amr/refinement_marker.hpp"

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

void RefinementMarker::Mark(ParMesh& mesh, const Vector& g,
                            Array<Refinement>& refs, MarkStats* stats) const
{
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
