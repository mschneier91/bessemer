#include "amr/mesh_adapter.hpp"

#include "util/profiler.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <utility>

namespace incns
{

using namespace mfem;

namespace
{

void VerifyOnSpaces(MixedSpaces& spaces,
                    const std::vector<ParGridFunction*>& fields,
                    bool velocity_only)
{
   for (const ParGridFunction* gf : fields)
   {
      MFEM_VERIFY(gf->ParFESpace() == &spaces.Velocity() ||
                  (!velocity_only && gf->ParFESpace() == &spaces.Pressure()),
                  "mesh_adapter: every field must live on the mixed spaces "
                  "(a projected one on the velocity space)");
   }
}

// Each local element's moments M_K u_K of a velocity field, in its vdof order
// (entries offsets[e] .. offsets[e + 1]). The element mass is at MFEM's
// default rule, the one MassSolve's partially assembled mass uses.
void ElementMoments(ParGridFunction& u, std::vector<double>& m,
                    std::vector<int>& offsets)
{
   ParFiniteElementSpace& fes = *u.ParFESpace();
   Mesh& mesh = *fes.GetMesh();
   u.HostRead();
   VectorMassIntegrator mass;
   DenseMatrix elmat;
   Array<int> vdofs;
   Vector ue, me;
   m.clear();
   offsets.assign(1, 0);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      fes.GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, ue);
      mass.AssembleElementMatrix(*fes.GetFE(e),
                                 *mesh.GetElementTransformation(e), elmat);
      me.SetSize(ue.Size());
      elmat.Mult(ue, me);
      m.insert(m.end(), me.GetData(), me.GetData() + me.Size());
      offsets.push_back(static_cast<int>(m.size()));
   }
}

// After a merge: the coarse moments of the fine field whose element moments
// ElementMoments took. Each coarse element K gets, from each old element c it
// covers, (I_vdim (x) P_c^T) m_c -- P_c the local interpolation from K to c,
// the identity for an element that was kept -- assembled and restricted to
// the true dofs. Every old element must be on its coarse element's rank
// (Derefine's gather).
void CoarseMoments(ParFiniteElementSpace& fes, const std::vector<double>& m,
                   const std::vector<int>& offsets, Vector& b)
{
   ParMesh& mesh = *fes.GetParMesh();
   ParNCMesh& pnc = *mesh.pncmesh;
   const CoarseFineTransformations& dtr = pnc.GetDerefinementTransforms();
   const Array<int>& old_rank = pnc.GetDerefineOldRanks();
   const int me = mesh.GetMyRank(), vdim = fes.GetVDim();
   const int n_old = static_cast<int>(offsets.size()) - 1;
   std::map<std::pair<int, int>, DenseMatrix> interp; // (geometry, matrix)
   IsoparametricTransformation isotr;
   Vector bl(fes.GetVSize());
   bl.UseDevice(true);
   {
      double* h = bl.HostWrite();
      std::fill(h, h + bl.Size(), 0.0);
   }
   Array<int> vdofs;
   Vector bk;
   for (int k = 0; k < dtr.embeddings.Size(); ++k)
   {
      const Embedding& emb = dtr.embeddings[k];
      const int coarse_rank = (emb.parent < 0) ? (-1 - emb.parent)
                              : pnc.ElementRank(emb.parent);
      if (old_rank[k] != me)
      {
         MFEM_VERIFY(coarse_rank != me, "mesh_adapter: a merged family was "
                     "split across ranks");
         continue;
      }
      MFEM_VERIFY(coarse_rank == me && k < n_old, "mesh_adapter: a merged "
                  "family was split across ranks");
      const int K = emb.parent;
      const FiniteElement& fe = *fes.GetFE(K);
      const int nd = fe.GetDof();
      fes.GetElementVDofs(K, vdofs);
      const int len = offsets[k + 1] - offsets[k];
      MFEM_VERIFY(len == nd * vdim && vdofs.Size() == len,
                  "mesh_adapter: the fine and coarse elements differ in size");
      const double* mc = m.data() + offsets[k];
      bk.SetSize(len);
      if (emb.matrix == 0) // kept: the identity embedding
      {
         for (int i = 0; i < len; ++i) { bk(i) = mc[i]; }
      }
      else
      {
         const Geometry::Type geom = static_cast<Geometry::Type>(emb.geom);
         DenseMatrix& P = interp[ {static_cast<int>(geom),
                                   static_cast<int>(emb.matrix)
                                  }];
         if (P.Height() == 0)
         {
            isotr.SetIdentityTransformation(geom);
            isotr.SetPointMat(dtr.point_matrices[geom](emb.matrix));
            fe.GetLocalInterpolation(isotr, P); // child dofs from parent dofs
         }
         for (int c = 0; c < vdim; ++c)
         {
            const Vector mcc(const_cast<double*>(mc) + c * nd, nd);
            Vector bkc(bk.GetData() + c * nd, nd);
            P.MultTranspose(mcc, bkc);
         }
      }
      bl.AddElementVector(vdofs, bk);
   }
   b.SetSize(fes.GetTrueVSize());
   b.UseDevice(true);
   fes.GetProlongationMatrix()->MultTranspose(bl, b);
}

// x = M^{-1} b on the true dofs, no essential dofs: the plain L2 projection.
void MassSolve(ParFiniteElementSpace& fes, const Vector& b, Vector& x)
{
   ParBilinearForm form(&fes);
   form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form.AddDomainIntegrator(new VectorMassIntegrator());
   form.Assemble();
   Array<int> none;
   OperatorPtr M;
   form.FormSystemMatrix(none, M);
   OperatorJacobiSmoother jacobi(form, none);
   CGSolver cg(fes.GetComm());
   cg.SetRelTol(1e-12);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(0);
   cg.SetOperator(*M);
   cg.SetPreconditioner(jacobi);
   x.SetSize(fes.GetTrueVSize());
   x.UseDevice(true);
   x = 0.0;
   cg.Mult(b, x);
   MFEM_VERIFY(cg.GetConverged(), "mesh_adapter: the L2 projection's mass "
               "solve did not converge in " << cg.GetNumIterations()
               << " iterations");
}

} // namespace

void MeshAdapter::Refine(ParMesh& mesh, MixedSpaces& spaces,
                         BoundaryConditions* bc,
                         const Array<Refinement>& refs, int nc_limit,
                         bool rebalance,
                         const std::vector<ParGridFunction*>& fields)
{
   INCNS_PROFILE("amr::refine");
   MFEM_VERIFY(mesh.Nonconforming(), "mesh_adapter: the mesh is conforming -- "
               "build it with MakeCaseMesh/PartitionMesh(nonconforming) so it "
               "can be refined");
   MFEM_VERIFY(nc_limit >= 1, "mesh_adapter: nc_limit must be >= 1");
   VerifyOnSpaces(spaces, fields, false);

   auto update_all = [&]()
   {
      spaces.Update();
      for (ParGridFunction* gf : fields) { gf->Update(); }
   };

   mesh.GeneralRefinement(refs, /*nonconforming=*/1, nc_limit);
   update_all();
   if (rebalance && mesh.GetNRanks() > 1)
   {
      INCNS_PROFILE("amr::rebalance");
      mesh.Rebalance();
      update_all();
   }
   spaces.UpdatesFinished();
   if (bc) { bc->Update(); }
}

long long MeshAdapter::SelectDerefinements(ParMesh& mesh, const Vector& values,
      double threshold, int nc_limit,
      Array<int>& gather,
      Array<int>& merge)
{
   INCNS_PROFILE("amr::select_derefinements");
   MFEM_VERIFY(mesh.Nonconforming() && mesh.pncmesh, "mesh_adapter: the mesh "
               "is conforming (nothing to coarsen)");
   MFEM_VERIFY(nc_limit >= 1, "mesh_adapter: nc_limit must be >= 1");
   const int ne = mesh.GetNE(), me = mesh.GetMyRank();
   MFEM_VERIFY(values.Size() == ne, "mesh_adapter: one value per element");
   ParNCMesh& pnc = *mesh.pncmesh;
   const Table& dt = pnc.GetDerefinementTable();
   // The values and ranks of ghost children too: a family can straddle ranks.
   Array<double> val(ne);
   const double* v = values.HostRead();
   for (int e = 0; e < ne; ++e) { val[e] = v[e]; }
   pnc.SynchronizeDerefinementData(val, dt);
   Array<int> rank(ne);
   rank = me;
   pnc.SynchronizeDerefinementData(rank, dt);
   Array<int> level_ok;
   pnc.CheckDerefinementNCLevel(dt, level_ok, nc_limit);

   gather.SetSize(ne);
   gather = me;
   merge.SetSize(ne);
   merge = 0;
   long long families = 0;
   for (int i = 0; i < dt.Size(); ++i)
   {
      if (!level_ok[i]) { continue; }
      const int* fine = dt.GetRow(i);
      const int n = dt.RowSize(i);
      double vmax = -std::numeric_limits<double>::infinity();
      int owner = INT_MAX;
      for (int j = 0; j < n; ++j)
      {
         vmax = std::max(vmax, val[fine[j]]);
         owner = std::min(owner, rank[fine[j]]);
      }
      if (!(vmax < threshold)) { continue; }
      if (owner == me) { ++families; } // once, by the rank that gets it
      for (int j = 0; j < n; ++j)
      {
         if (fine[j] < ne)
         {
            gather[fine[j]] = owner;
            merge[fine[j]] = 1;
         }
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &families, 1, MPI_LONG_LONG, MPI_SUM,
                 mesh.GetComm());
   return families;
}

bool MeshAdapter::Derefine(ParMesh& mesh, MixedSpaces& spaces,
                           BoundaryConditions* bc,
                           const Array<int>& gather, const Array<int>& merge,
                           int nc_limit, bool rebalance,
                           const std::vector<ParGridFunction*>& fields,
                           const std::vector<ParGridFunction*>& projected)
{
   INCNS_PROFILE("amr::derefine");
   MFEM_VERIFY(mesh.Nonconforming() && mesh.pncmesh, "mesh_adapter: the mesh "
               "is conforming (nothing to coarsen)");
   MFEM_VERIFY(nc_limit >= 1, "mesh_adapter: nc_limit must be >= 1");
   const int ne = mesh.GetNE(), me = mesh.GetMyRank();
   MFEM_VERIFY(gather.Size() == ne && merge.Size() == ne, "mesh_adapter: "
               "gather and merge need one entry per local element");
   VerifyOnSpaces(spaces, fields, false);
   VerifyOnSpaces(spaces, projected, true);
   std::vector<ParGridFunction*> all(fields);
   all.insert(all.end(), projected.begin(), projected.end());
   auto update_all = [&]()
   {
      spaces.Update();
      for (ParGridFunction* gf : all) { gf->Update(); }
   };

   // The merge flags travel with the elements as a piecewise constant: 0 in a
   // family to merge, 1 elsewhere (DerefineByError's "error").
   L2_FECollection fec0(0, mesh.Dimension());
   ParFiniteElementSpace fes0(&mesh, &fec0);
   ParGridFunction keep(&fes0);
   {
      double* k = keep.HostWrite();
      for (int e = 0; e < ne; ++e) { k[e] = merge[e] ? 0.0 : 1.0; }
   }

   // 1. Each family onto the rank of its parent-to-be.
   int moves = 0;
   for (int e = 0; e < ne; ++e) { moves += (gather[e] != me); }
   MPI_Allreduce(MPI_IN_PLACE, &moves, 1, MPI_INT, MPI_SUM, mesh.GetComm());
   if (moves > 0)
   {
      INCNS_PROFILE("amr::gather");
      mesh.Rebalance(gather);
      update_all();
      fes0.Update();
      keep.Update();
   }

   // 2. The projected fields' element moments, on the fine mesh.
   std::vector<std::vector<double>> moments(projected.size());
   std::vector<int> offsets;
   for (std::size_t f = 0; f < projected.size(); ++f)
   {
      ElementMoments(*projected[f], moments[f], offsets);
   }

   // 3. Merge, then the projected fields from their coarse moments.
   Vector err(mesh.GetNE());
   {
      const double* k = keep.HostRead();
      double* x = err.HostWrite();
      for (int e = 0; e < mesh.GetNE(); ++e) { x[e] = k[e]; }
   }
   const bool merged = mesh.DerefineByError(err, 0.5, nc_limit, /*max*/2);
   if (merged)
   {
      update_all(); // MFEM's interpolation; the projected ones are replaced
      INCNS_PROFILE("amr::l2_projection");
      for (std::size_t f = 0; f < projected.size(); ++f)
      {
         Vector b, x;
         CoarseMoments(spaces.Velocity(), moments[f], offsets, b);
         MassSolve(spaces.Velocity(), b, x);
         projected[f]->SetFromTrueDofs(x);
      }
   }

   // 4. Rebalance.
   if (rebalance && mesh.GetNRanks() > 1)
   {
      INCNS_PROFILE("amr::rebalance");
      mesh.Rebalance();
      update_all();
   }
   spaces.UpdatesFinished();
   if (bc) { bc->Update(); }
   return merged;
}

void MeshAdapter::Replay(ParMesh& mesh, MixedSpaces& spaces,
                         BoundaryConditions* bc, const RefinementRecord& rec)
{
   if (rec.derefine)
   {
      Derefine(mesh, spaces, bc, rec.gather, rec.merge, rec.nc_limit,
               rec.rebalance, {});
   }
   else
   {
      Refine(mesh, spaces, bc, rec.refs, rec.nc_limit, rec.rebalance, {});
   }
}

} // namespace incns
