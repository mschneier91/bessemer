#include "mesh/periodic_box.hpp"

#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

/// Map a normalized coordinate xi in [0,1] to a clustered coordinate in [0,1].
/// Symmetric tanh clustering: nodes pull toward both ends. The endpoints are
/// fixed (0 -> 0, 1 -> 1), so a stretched direction can still be made periodic.
static double TwoSidedTanhMap(double xi, double beta)
{
   return 0.5 * (1.0 + std::tanh(beta * (xi - 0.5)) / std::tanh(0.5 * beta));
}

/// Remap vertex coordinates in place for the stretched directions. Runs on the
/// serial base mesh before periodic identification; leaves uniform directions
/// untouched, so a fully-uniform spec is bitwise unchanged.
static void ApplyStretching(Mesh& mesh, const BoxSpec& s)
{
   bool any = false;
   for (int d = 0; d < s.dim; ++d)
   {
      if (s.stretch[d] != Stretch::None) { any = true; }
   }
   if (!any) { return; }

   for (int d = 0; d < s.dim; ++d)
   {
      if (s.stretch[d] != Stretch::None)
      {
         MFEM_VERIFY(s.stretch_beta[d] > 0.0,
                     "periodic_box: stretch_beta must be > 0 on a stretched "
                     "direction");
      }
   }

   for (int v = 0; v < mesh.GetNV(); ++v)
   {
      double* x = mesh.GetVertex(v);
      for (int d = 0; d < s.dim; ++d)
      {
         if (s.stretch[d] == Stretch::None) { continue; }
         const double L = s.lengths[d];
         const double xi = x[d] / L;   // uniform mesh: xi in [0,1]
         x[d] = L * TwoSidedTanhMap(xi, s.stretch_beta[d]);
      }
   }
}

void AssertTensorProductGeometry(const Mesh& mesh)
{
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "periodic_box: only 2D/3D meshes are supported");
   const Geometry::Type want = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      MFEM_VERIFY(mesh.GetElementBaseGeometry(e) == want,
                  "periodic_box: non-tensor-product element found "
                  "(quad/hex only; simplices are rejected)");
   }
}

static Mesh MakeCartesian(const BoxSpec& s)
{
   if (s.dim == 2)
   {
      return Mesh::MakeCartesian2D(s.num_elems[0], s.num_elems[1],
                                   Element::QUADRILATERAL, false,
                                   s.lengths[0], s.lengths[1]);
   }
   return Mesh::MakeCartesian3D(s.num_elems[0], s.num_elems[1], s.num_elems[2],
                                Element::HEXAHEDRON,
                                s.lengths[0], s.lengths[1], s.lengths[2]);
}

Mesh MakeBoxMesh(const BoxSpec& s)
{
   MFEM_VERIFY(s.dim == 2 || s.dim == 3, "periodic_box: dim must be 2 or 3");
   for (int d = 0; d < s.dim; ++d)
   {
      MFEM_VERIFY(s.num_elems[d] >= 1,
                  "periodic_box: need >= 1 element per direction");
      MFEM_VERIFY(s.lengths[d] > 0.0,
                  "periodic_box: edge lengths must be positive");
      if (s.periodic[d])
      {
         MFEM_VERIFY(s.num_elems[d] >= 3,
                     "periodic_box: a periodic direction needs >= 3 elements "
                     "(MFEM periodic topology); got fewer");
      }
   }

   Mesh base = MakeCartesian(s);
   ApplyStretching(base, s);
   AssertTensorProductGeometry(base);

   // One translation vector per periodic direction: length along that axis.
   std::vector<Vector> translations;
   for (int d = 0; d < s.dim; ++d)
   {
      if (!s.periodic[d]) { continue; }
      Vector t(s.dim);
      t = 0.0;
      t(d) = s.lengths[d];
      translations.push_back(t);
   }

   if (translations.empty()) { return base; }

   std::vector<int> v2v = base.CreatePeriodicVertexMapping(translations);
   Mesh periodic = Mesh::MakePeriodic(base, v2v);
   AssertTensorProductGeometry(periodic);
   return periodic;
}

} // namespace incns
