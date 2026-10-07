// Shared helpers for the AMR tests: boxes and a fixed, deterministic
// refinement pattern that leaves hanging nodes in the interior AND on the
// boundary (refined bands touch the walls next to unrefined elements).

#pragma once

#include "mesh/case_mesh.hpp"
#include "mesh/periodic_box.hpp"
#include "mfem.hpp"

#include <memory>
#include <set>

namespace amr_test
{

/**
 * MFEM debug-device false positive (pure-MFEM reproduction in CLAUDE.md): a
 * host-valid BlockVector whose block spans >= 1 page, read on the device
 * through the block alias and then read whole on the device, faults in
 * MemoryManager::GetDevicePtr. Solves on REFINED PERIODIC meshes hit it: with
 * no essential dofs the constrained operators hand BlockVector blocks straight
 * to the NC prolongation (a CPU-hypre matrix that reads them on the host).
 * ASan is clean and other backends are unaffected, so those cases skip on the
 * debug device only.
 */
inline bool DebugDeviceSkipsPeriodicNcSolves()
{
   return mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE);
}

inline incns::BoxSpec Box(int dim, int n, bool periodic, double length = 1.0)
{
   incns::BoxSpec s;
   s.dim = dim;
   s.num_elems = {n, n, n};
   s.lengths = {length, length, length};
   s.periodic = {periodic, periodic, periodic};
   return s;
}

/// A partitioned nonconforming-ready box (no refinement yet).
inline std::unique_ptr<mfem::ParMesh> NcBox(const incns::BoxSpec& spec)
{
   mfem::Mesh serial = incns::MakeBoxMesh(spec);
   return incns::PartitionMesh(serial, true);
}

/// Refine every element whose center satisfies @p pick with @p type; in 3D
/// parallel, entries in anisotropic conflict are upgraded to XYZ first (the
/// production marker does the same).
template <typename Pick>
void RefineWhere(mfem::ParMesh& mesh, Pick pick, char type)
{
   const int dim = mesh.Dimension();
   mfem::Array<mfem::Refinement> refs;
   mfem::Vector c(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mesh.GetElementCenter(e, c);
      if (pick(c)) { refs.Append(mfem::Refinement(e, type)); }
   }
   if (dim == 3)
   {
      for (int it = 0; it < 5; ++it)
      {
         std::set<int> conflicts;
         if (!mesh.AnisotropicConflict(refs, conflicts)) { break; }
         for (int i : conflicts) { refs[i].SetType(mfem::Refinement::XYZ); }
      }
   }
   mesh.GeneralRefinement(refs, 1, 1);
}

/// The standard two/three-pass pattern on a box of edge @p L: x < L/2, then
/// y < 0.4 L (then z > 0.6 L in 3D). Anisotropic: X, then Y, then Z splits;
/// isotropic: XYZ every pass.
inline void RefineBands(mfem::ParMesh& mesh, bool aniso, double L = 1.0)
{
   const int dim = mesh.Dimension();
   const char iso = static_cast<char>((dim == 3) ? mfem::Refinement::XYZ
                                      : mfem::Refinement::XY);
   const char X = mfem::Refinement::X, Y = mfem::Refinement::Y,
         Z = mfem::Refinement::Z;
   RefineWhere(mesh, [L](const mfem::Vector & c) { return c(0) < 0.5 * L; },
   aniso ? X : iso);
   RefineWhere(mesh, [L](const mfem::Vector & c) { return c(1) < 0.4 * L; },
   aniso ? Y : iso);
   if (dim == 3)
   {
      RefineWhere(mesh, [L](const mfem::Vector & c) { return c(2) > 0.6 * L; },
      aniso ? Z : iso);
   }
}

} // namespace amr_test
