#include "mesh/cylinder_channel.hpp"

#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

namespace
{

// n cell widths on [a, b] growing by `ratio` from a, as n+1 node positions.
std::vector<double> Graded(double a, double b, int n, double ratio)
{
   std::vector<double> x(n + 1);
   double w = 1.0, sum = 0.0;
   std::vector<double> widths(n);
   for (int k = 0; k < n; ++k) { widths[k] = w; sum += w; w *= ratio; }
   x[0] = a;
   for (int k = 0; k < n; ++k) { x[k + 1] = x[k] + (b - a) * widths[k] / sum; }
   x[n] = b;
   return x;
}

struct Point { double x, y; };

} // namespace

Mesh MakeCylinderChannelMesh(const CylinderChannelSpec& s)
{
   const double a = s.box_half;
   MFEM_VERIFY(s.radius > 0.0 && a > s.radius, "cylinder_channel: need "
               "0 < radius < box_half");
   MFEM_VERIFY(s.cx - a > 0.0 && s.cx + a < s.length && s.cy - a > 0.0 &&
               s.cy + a < s.height, "cylinder_channel: the O-grid square must "
               "lie strictly inside the channel");
   MFEM_VERIFY(s.n_side >= 1 && s.n_ring >= 1 && s.n_up >= 1 && s.n_down >= 1
               && s.n_below >= 1 && s.n_above >= 1 && s.order >= 1,
               "cylinder_channel: bad resolution");

   // Grid lines of the eight Cartesian blocks.
   std::vector<double> X, Y;
   {
      const auto up = Graded(0.0, s.cx - a, s.n_up, 1.0);
      const auto mid = Graded(s.cx - a, s.cx + a, s.n_side, 1.0);
      const auto down = Graded(s.cx + a, s.length, s.n_down, s.down_grading);
      X = up;
      X.insert(X.end(), mid.begin() + 1, mid.end());
      X.insert(X.end(), down.begin() + 1, down.end());
      const auto below = Graded(0.0, s.cy - a, s.n_below, 1.0);
      const auto ymid = Graded(s.cy - a, s.cy + a, s.n_side, 1.0);
      const auto above = Graded(s.cy + a, s.height, s.n_above, 1.0);
      Y = below;
      Y.insert(Y.end(), ymid.begin() + 1, ymid.end());
      Y.insert(Y.end(), above.begin() + 1, above.end());
   }
   const int nx = static_cast<int>(X.size()) - 1,
             ny = static_cast<int>(Y.size()) - 1;
   const int iA = s.n_up, iB = s.n_up + s.n_side;
   const int jA = s.n_below, jB = s.n_below + s.n_side;
   auto inside_open = [&](int i, int j) { return i > iA && i < iB && j > jA && j < jB; };
   auto inside_cell = [&](int i, int j) { return i >= iA && i < iB && j >= jA && j < jB; };

   // Ring radial parameters t_k (0 = circle, 1 = square).
   std::vector<double> t(s.n_ring + 1);
   {
      const auto r = Graded(0.0, 1.0, s.n_ring, s.ring_grading);
      for (int k = 0; k <= s.n_ring; ++k) { t[k] = r[k]; }
   }
   const int n_sq = 4 * s.n_side;

   // Counting.
   int nv_cart = 0;
   for (int i = 0; i <= nx; ++i)
      for (int j = 0; j <= ny; ++j) { if (!inside_open(i, j)) { ++nv_cart; } }
   const int nv = nv_cart + n_sq * s.n_ring;
   const int ne = nx * ny - s.n_side * s.n_side + n_sq * s.n_ring;
   const int nbe = 2 * ny + 2 * nx + n_sq;
   Mesh mesh(2, nv, ne, nbe, 2);

   // Cartesian vertices.
   std::vector<std::vector<int>> vid(nx + 1, std::vector<int>(ny + 1, -1));
   int v = 0;
   for (int i = 0; i <= nx; ++i)
      for (int j = 0; j <= ny; ++j)
      {
         if (inside_open(i, j)) { continue; }
         mesh.AddVertex(X[i], Y[j]);
         vid[i][j] = v++;
      }

   // The square boundary, counterclockwise from its bottom-left corner.
   std::vector<int> sq_i(n_sq), sq_j(n_sq);
   {
      int q = 0;
      for (int i = iA; i < iB; ++i) { sq_i[q] = i; sq_j[q] = jA; ++q; }
      for (int j = jA; j < jB; ++j) { sq_i[q] = iB; sq_j[q] = j; ++q; }
      for (int i = iB; i > iA; --i) { sq_i[q] = i; sq_j[q] = jB; ++q; }
      for (int j = jB; j > jA; --j) { sq_i[q] = iA; sq_j[q] = j; ++q; }
   }
   auto square_pt = [&](int q) { return Point{X[sq_i[q]], Y[sq_j[q]]}; };
   // Circle point on the ray from the centre through square point S.
   auto circle_of = [&](const Point & S)
   {
      const double dx = S.x - s.cx, dy = S.y - s.cy;
      const double r = std::sqrt(dx * dx + dy * dy);
      return Point{s.cx + s.radius* dx / r, s.cy + s.radius* dy / r};
   };

   // Ring vertices for k < n_ring (k = n_ring is the square itself).
   std::vector<std::vector<int>> ring(n_sq, std::vector<int>(s.n_ring + 1, -1));
   for (int q = 0; q < n_sq; ++q)
   {
      const Point S = square_pt(q), C = circle_of(S);
      for (int k = 0; k < s.n_ring; ++k)
      {
         mesh.AddVertex(C.x + t[k] * (S.x - C.x), C.y + t[k] * (S.y - C.y));
         ring[q][k] = v++;
      }
      ring[q][s.n_ring] = vid[sq_i[q]][sq_j[q]];
   }

   // Elements: Cartesian blocks, then the ring (counterclockwise quads; ring
   // quads go radially out along xi_x and around along xi_y).
   for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
         if (inside_cell(i, j)) { continue; }
         mesh.AddQuad(vid[i][j], vid[i + 1][j], vid[i + 1][j + 1], vid[i][j + 1], 1);
      }
   const int first_ring_elem = mesh.GetNE();
   for (int q = 0; q < n_sq; ++q)
   {
      const int q1 = (q + 1) % n_sq;
      for (int k = 0; k < s.n_ring; ++k)
      {
         mesh.AddQuad(ring[q][k], ring[q][k + 1], ring[q1][k + 1], ring[q1][k], 1);
      }
   }

   // Boundary.
   for (int j = 0; j < ny; ++j)
   {
      mesh.AddBdrSegment(vid[0][j + 1], vid[0][j], kCylinderInflow);
      mesh.AddBdrSegment(vid[nx][j], vid[nx][j + 1], kCylinderOutflow);
   }
   for (int i = 0; i < nx; ++i)
   {
      mesh.AddBdrSegment(vid[i][0], vid[i + 1][0], kCylinderWalls);
      mesh.AddBdrSegment(vid[i + 1][ny], vid[i][ny], kCylinderWalls);
   }
   for (int q = 0; q < n_sq; ++q)
   {
      mesh.AddBdrSegment(ring[(q + 1) % n_sq][0], ring[q][0], kCylinderBody);
   }
   mesh.FinalizeQuadMesh(1, 1, true);

   // Exact curved geometry: high-order nodes of the ring elements from the
   // transfinite ring map x(sigma, t) = C(sigma) + t (S(sigma) - C(sigma)),
   // with S piecewise linear along the square and C its ray projection on the
   // circle. Cartesian elements keep the (exact) bilinear interpolation.
   mesh.SetCurvature(s.order, false, 2, Ordering::byVDIM);
   GridFunction& nodes = *mesh.GetNodes();
   const FiniteElementSpace& nfes = *nodes.FESpace();
   Array<int> vdofs;
   for (int q = 0; q < n_sq; ++q)
   {
      const int q1 = (q + 1) % n_sq;
      const Point S0 = square_pt(q), S1 = square_pt(q1);
      for (int k = 0; k < s.n_ring; ++k)
      {
         const int e = first_ring_elem + q * s.n_ring + k;
         const FiniteElement& fe = *nfes.GetFE(e);
         const IntegrationRule& nir = fe.GetNodes();
         nfes.GetElementVDofs(e, vdofs);
         const int nd = fe.GetDof();
         for (int j = 0; j < nd; ++j)
         {
            const IntegrationPoint& ip = nir.IntPoint(j);
            const Point S{S0.x + ip.y* (S1.x - S0.x), S0.y + ip.y* (S1.y - S0.y)};
            const Point C = circle_of(S);
            const double tt = t[k] + ip.x * (t[k + 1] - t[k]);
            nodes(vdofs[j]) = C.x + tt * (S.x - C.x);
            nodes(vdofs[j + nd]) = C.y + tt * (S.y - C.y);
         }
      }
   }
   return mesh;
}

} // namespace incns
