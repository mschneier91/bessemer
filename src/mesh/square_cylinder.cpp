#include "mesh/square_cylinder.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

namespace
{

// Widths w0, w0 r, w0 r^2, ... covering `length`: the count that first
// reaches it, all rescaled to sum to exactly `length`.
std::vector<double> GrowWidths(double w0, double ratio, double length)
{
   std::vector<double> w;
   double sum = 0.0, cur = w0;
   while (sum < length * (1.0 - 1e-12))
   {
      w.push_back(cur);
      sum += cur;
      cur *= ratio;
   }
   for (double& v : w) { v *= length / sum; }
   return w;
}

// Widths growing from w0 by `ratio` until they reach `w_max` (that segment
// rescaled to end at a whole cell), then w_max cells to cover `length`.
std::vector<double> GrowThenUniform(double w0, double ratio, double w_max,
                                    double length)
{
   std::vector<double> w;
   double sum = 0.0, cur = w0;
   while (cur < w_max && sum + cur < length)
   {
      w.push_back(cur);
      sum += cur;
      cur *= ratio;
   }
   const int n_uni = std::max(1,
                              static_cast<int>(std::lround((length - sum) / w_max)));
   const double uni = (length - sum) / n_uni;
   for (int k = 0; k < n_uni; ++k) { w.push_back(uni); }
   return w;
}

} // namespace

std::vector<double> SquareCylinderNodes(const SquareCylinderSpec& s, bool x)
{
   MFEM_VERIFY(s.n_face >= 2 && s.n_face % 2 == 0, "square_cylinder: n_face "
               "must be even and >= 2");
   MFEM_VERIFY(s.corner_ratio >= 1.0 && s.far_ratio > 1.0 && s.wake_ratio >= 1.0
               && s.wake_h > 0.0, "square_cylinder: bad grading");
   const double a = 0.5 * s.side;
   // Along a face: corner-graded halves, mirrored.
   const int half = s.n_face / 2;
   std::vector<double> face_half(half);
   {
      double sum = 0.0, cur = 1.0;
      for (int k = 0; k < half; ++k) { face_half[k] = cur; sum += cur; cur *= s.corner_ratio; }
      for (double& v : face_half) { v *= a / sum; }
   }
   const double w0 = face_half[0];
   std::vector<double> face(face_half.begin(), face_half.end());
   face.insert(face.end(), face_half.rbegin(), face_half.rend());

   std::vector<double> before, after; // widths outward from the square
   if (x)
   {
      MFEM_VERIFY(s.upstream > a && s.downstream > s.wake_end && s.wake_end > a,
                  "square_cylinder: need upstream > D/2 and downstream > "
                  "wake_end > D/2");
      before = GrowWidths(w0, s.far_ratio, s.upstream - a);
      after = GrowThenUniform(w0, s.wake_ratio, s.wake_h, s.wake_end - a);
      const auto far = GrowWidths(s.wake_h, s.far_ratio, s.downstream - s.wake_end);
      after.insert(after.end(), far.begin(), far.end());
   }
   else
   {
      MFEM_VERIFY(s.half_height > a, "square_cylinder: half_height <= D/2");
      before = GrowWidths(w0, s.far_ratio, s.half_height - a);
      after = before;
   }
   std::vector<double> nodes;
   double pos = x ? -s.upstream : -s.half_height;
   nodes.push_back(pos);
   for (auto it = before.rbegin(); it != before.rend(); ++it) { pos += *it; nodes.push_back(pos); }
   nodes.back() = -a; // exact
   for (double w : face) { pos += w; nodes.push_back(pos); }
   nodes.back() = a;
   for (double w : after) { pos += w; nodes.push_back(pos); }
   nodes.back() = x ? s.downstream : s.half_height;
   return nodes;
}

Mesh MakeSquareCylinderMesh(const SquareCylinderSpec& s)
{
   const std::vector<double> X = SquareCylinderNodes(s, true);
   const std::vector<double> Y = SquareCylinderNodes(s, false);
   const int nx = static_cast<int>(X.size()) - 1,
             ny = static_cast<int>(Y.size()) - 1;
   const double a = 0.5 * s.side;
   // Node indices of the square's faces.
   auto index_of = [](const std::vector<double>& v, double val)
   {
      const auto it = std::min_element(v.begin(), v.end(), [val](double p, double q)
      { return std::abs(p - val) < std::abs(q - val); });
      return static_cast<int>(it - v.begin());
   };
   const int iA = index_of(X, -a), iB = index_of(X, a);
   const int jA = index_of(Y, -a), jB = index_of(Y, a);
   auto inside_open = [&](int i, int j) { return i > iA && i < iB && j > jA && j < jB; };
   auto inside_cell = [&](int i, int j) { return i >= iA && i < iB && j >= jA && j < jB; };

   int nv = 0, ne = 0;
   for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i) { nv += inside_open(i, j) ? 0 : 1; }
   for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) { ne += inside_cell(i, j) ? 0 : 1; }
   const int nbe = 2 * nx + 2 * ny + 2 * (iB - iA) + 2 * (jB - jA);

   Mesh mesh(2, nv, ne, nbe, 2);
   std::vector<std::vector<int>> vid(nx + 1, std::vector<int>(ny + 1, -1));
   int v = 0;
   for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i)
      {
         if (inside_open(i, j)) { continue; }
         mesh.AddVertex(X[i], Y[j]);
         vid[i][j] = v++;
      }
   for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
      {
         if (inside_cell(i, j)) { continue; }
         mesh.AddQuad(vid[i][j], vid[i + 1][j], vid[i + 1][j + 1], vid[i][j + 1], 1);
      }
   // Outer boundary counter-clockwise, the square clockwise (fluid on the left).
   for (int j = 0; j < ny; ++j)
   {
      mesh.AddBdrSegment(vid[0][j + 1], vid[0][j], kSquareInflow);
      mesh.AddBdrSegment(vid[nx][j], vid[nx][j + 1], kSquareOutflow);
   }
   for (int i = 0; i < nx; ++i)
   {
      mesh.AddBdrSegment(vid[i][0], vid[i + 1][0], kSquareSides);
      mesh.AddBdrSegment(vid[i + 1][ny], vid[i][ny], kSquareSides);
   }
   for (int j = jA; j < jB; ++j)
   {
      mesh.AddBdrSegment(vid[iA][j], vid[iA][j + 1],
                         kSquareBody);     // left face, up
      mesh.AddBdrSegment(vid[iB][j + 1], vid[iB][j],
                         kSquareBody);     // right face, down
   }
   for (int i = iA; i < iB; ++i)
   {
      mesh.AddBdrSegment(vid[i][jB], vid[i + 1][jB],
                         kSquareBody);     // top face, right
      mesh.AddBdrSegment(vid[i + 1][jA], vid[i][jA],
                         kSquareBody);     // bottom face, left
   }
   mesh.FinalizeQuadMesh(1, 1, true);
   return mesh;
}

} // namespace incns
