#include "bc/boundary_names.hpp"

#include <algorithm>
#include <cmath>

namespace incns
{

using namespace mfem;

namespace
{
// Box face xmin ... zmax: the attribute of the boundary elements lying on it.
int BoxFaceAttribute(const Parameters& p, ParMesh& mesh,
                     const std::string& name)
{
   const int ci = (name[0] == 'x') ? 0 : (name[0] == 'y') ? 1 : 2;
   const bool is_max = (name.compare(1, 3, "max") == 0);
   const double val = is_max ? p.mesh.lengths[ci] : 0.0;
   int attr = 0;
   Vector c;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      T->Transform(Geometries.GetCenter(T->GetGeometryType()), c);
      if (std::abs(c(ci) - val) < 1e-9 * std::max(1.0, std::abs(val)))
      {
         attr = mesh.GetBdrAttribute(be);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &attr, 1, MPI_INT, MPI_MAX, mesh.GetComm());
   MFEM_VERIFY(attr > 0, "boundary_names: face '" << name << "' has no real "
               "boundary (periodic?)");
   return attr;
}
} // namespace

std::vector<std::string> BoundaryNames(const Parameters& p)
{
   switch (p.geometry)
   {
      case MeshGeometry::SquareCylinder:
         return {"inflow", "outflow", "sides", "body"};
      case MeshGeometry::CylinderChannel:
         return {"inflow", "outflow", "walls", "cylinder"};
      case MeshGeometry::File:
      {
         std::vector<std::string> out;
         for (const auto& nb : p.boundary_names) { out.push_back(nb.first); }
         return out;
      }
      case MeshGeometry::Box:
      {
         std::vector<std::string> out;
         const char axes[3] = {'x', 'y', 'z'};
         for (int d = 0; d < p.mesh.dim; ++d)
         {
            if (p.mesh.periodic[d]) { continue; }
            out.push_back(std::string(1, axes[d]) + "min");
            out.push_back(std::string(1, axes[d]) + "max");
         }
         return out;
      }
   }
   return {};
}

int NamedBoundaryAttribute(const Parameters& p, ParMesh& mesh,
                           const std::string& name)
{
   const std::vector<std::string> names = BoundaryNames(p);
   if (std::find(names.begin(), names.end(), name) == names.end() &&
       !(p.geometry == MeshGeometry::CylinderChannel && name == "body"))
   {
      std::string list;
      for (const std::string& n : names) { list += (list.empty() ? "" : ", ") + n; }
      MFEM_ABORT("boundary_names: unknown boundary '" << name << "' for this "
                 "geometry (" << list << ", all, or an attribute number)");
   }
   switch (p.geometry)
   {
      case MeshGeometry::SquareCylinder:
         if (name == "inflow") { return kSquareInflow; }
         if (name == "outflow") { return kSquareOutflow; }
         if (name == "sides") { return kSquareSides; }
         return kSquareBody;
      case MeshGeometry::CylinderChannel:
         if (name == "inflow") { return kCylinderInflow; }
         if (name == "outflow") { return kCylinderOutflow; }
         if (name == "walls") { return kCylinderWalls; }
         return kCylinderBody; // "cylinder" or "body"
      case MeshGeometry::File:
      {
         for (const auto& nb : p.boundary_names)
         {
            if (nb.first != name) { continue; }
            MFEM_VERIFY(mesh.bdr_attributes.Find(nb.second) >= 0,
                        "boundary_names: '" << name << "' is attribute "
                        << nb.second << ", which the mesh does not have");
            return nb.second;
         }
         return 0; // unreachable: the name was checked above
      }
      case MeshGeometry::Box:
         return BoxFaceAttribute(p, mesh, name);
   }
   return 0;
}

std::vector<int> AllBoundaryAttributes(const Parameters& p, ParMesh& mesh)
{
   std::vector<int> out;
   if (p.geometry == MeshGeometry::File)
   {
      // Every boundary of a file mesh is real, named or not (the coverage
      // check then reports an unnamed one by its attribute).
      for (int a : mesh.bdr_attributes) { out.push_back(a); }
      return out;
   }
   for (const std::string& n : BoundaryNames(p))
   {
      out.push_back(NamedBoundaryAttribute(p, mesh, n));
   }
   std::sort(out.begin(), out.end());
   out.erase(std::unique(out.begin(), out.end()), out.end());
   return out;
}

} // namespace incns
