#include "mesh/mesh_file.hpp"

#include <fstream>
#include <sstream>

namespace incns
{

using namespace mfem;

std::vector<std::pair<std::string, int>>
GmshPhysicalNames(const std::string& path, int dim)
{
   std::ifstream in(path);
   MFEM_VERIFY(in, "mesh_file: cannot open '" << path << "'");
   std::vector<std::pair<std::string, int>> out;
   std::string line;
   while (std::getline(in, line))
   {
      if (line.rfind("$PhysicalNames", 0) != 0) { continue; }
      int n = 0;
      MFEM_VERIFY(std::getline(in, line) && (std::istringstream(line) >> n),
                  "mesh_file: bad $PhysicalNames in '" << path << "'");
      for (int i = 0; i < n; ++i)
      {
         MFEM_VERIFY(std::getline(in, line), "mesh_file: $PhysicalNames in '"
                     << path << "' ends early");
         std::istringstream ls(line);
         int d = -1, tag = 0;
         ls >> d >> tag;
         const std::size_t q0 = line.find('"');
         const std::size_t q1 = line.rfind('"');
         MFEM_VERIFY(ls && q0 != std::string::npos && q1 > q0,
                     "mesh_file: bad $PhysicalNames line '" << line << "'");
         if (d == dim) { out.emplace_back(line.substr(q0 + 1, q1 - q0 - 1), tag); }
      }
      break;
   }
   return out;
}

Mesh LoadMeshFile(const std::string& path, int dim)
{
   {
      std::ifstream probe(path);
      MFEM_VERIFY(probe, "mesh_file: cannot open '" << path << "'");
   }
   Mesh mesh(path.c_str(), 1, 1);
   MFEM_VERIFY(mesh.Dimension() == dim, "mesh_file: '" << path << "' is "
               << mesh.Dimension() << "D but mesh.dim is " << dim);
   const Geometry::Type want = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      MFEM_VERIFY(mesh.GetElementBaseGeometry(e) == want, "mesh_file: '" << path
                  << "' has an element that is not a "
                  << (dim == 3 ? "hexahedron" : "quadrilateral")
                  << " (the solver needs tensor-product elements)");
   }
   for (int b = 0; b < mesh.GetNBE(); ++b)
   {
      MFEM_VERIFY(mesh.GetBdrAttribute(b) > 0, "mesh_file: '" << path
                  << "' has a boundary element without a physical group");
   }
   return mesh;
}

} // namespace incns
