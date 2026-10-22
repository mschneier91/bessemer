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
   const bool flag = s.flag_length > 0.0;
   const double b = 0.5 * s.flag_thickness;
   const double tip = s.cx + s.radius + s.flag_length;
   MFEM_VERIFY(s.radius > 0.0 && a > s.radius, "cylinder_channel: need "
               "0 < radius < box_half");
   MFEM_VERIFY(s.cx - a > 0.0 && s.cx + a < s.length && s.cy - a > 0.0 &&
               s.cy + a < s.height, "cylinder_channel: the O-grid square must "
               "lie strictly inside the channel");
   MFEM_VERIFY(s.n_side >= 1 && s.n_ring >= 1 && s.n_up >= 1 && s.n_down >= 1
               && s.n_below >= 1 && s.n_above >= 1 && s.order >= 1 &&
               s.outer_grading > 0.0 && s.down_grading > 0.0 &&
               s.ring_grading > 0.0,
               "cylinder_channel: bad resolution");
   MFEM_VERIFY(flag || !s.solid, "cylinder_channel: the solid mesh needs a "
               "flag (flag_length > 0)");
   if (flag)
   {
      MFEM_VERIFY(b > 0.0 && b < s.radius, "cylinder_channel: need 0 < "
                  "flag_thickness < 2 radius");
      MFEM_VERIFY(tip > s.cx + a && tip < s.length, "cylinder_channel: the "
                  "flag's tip must lie between the O-grid square and the "
                  "outflow");
      MFEM_VERIFY(s.n_side >= 2 && s.n_side % 2 == 0 && s.n_flag >= 1 &&
                  s.n_flag_y >= 1, "cylinder_channel: a flag needs an even "
                  "n_side and n_flag, n_flag_y >= 1");
   }

   // Grid lines of the Cartesian blocks.
   std::vector<double> X, Y;
   auto append = [](std::vector<double>& to, const std::vector<double>& from)
   {
      to.insert(to.end(), from.begin() + (to.empty() ? 0 : 1), from.end());
   };
   {
      // Upstream and below the widths shrink toward the square (ratio 1 /
      // outer_grading from the far end); above they grow away from it.
      const double g = s.outer_grading;
      append(X, Graded(0.0, s.cx - a, s.n_up, 1.0 / g));
      append(X, Graded(s.cx - a, s.cx + a, s.n_side, 1.0));
      if (flag) { append(X, Graded(s.cx + a, tip, s.n_flag, 1.0)); }
      append(X, Graded(flag ? tip : s.cx + a, s.length, s.n_down, s.down_grading));
      append(Y, Graded(0.0, s.cy - a, s.n_below, 1.0 / g));
      if (flag)
      {
         const int h = s.n_side / 2;
         append(Y, Graded(s.cy - a, s.cy - b, h, 1.0));
         append(Y, Graded(s.cy - b, s.cy + b, s.n_flag_y, 1.0));
         append(Y, Graded(s.cy + b, s.cy + a, h, 1.0));
      }
      else { append(Y, Graded(s.cy - a, s.cy + a, s.n_side, 1.0)); }
      append(Y, Graded(s.cy + a, s.height, s.n_above, g));
   }
   const int nx = static_cast<int>(X.size()) - 1,
             ny = static_cast<int>(Y.size()) - 1;
   const int n_sq_y = flag ? s.n_side + s.n_flag_y : s.n_side;
   const int iA = s.n_up, iB = s.n_up + s.n_side;
   const int jA = s.n_below, jB = s.n_below + n_sq_y;
   // The flag's Cartesian block: i in [iB, iT), j in [jF0, jF1).
   const int iT = flag ? iB + s.n_flag : iB;
   const int jF0 = jA + s.n_side / 2, jF1 = jF0 + (flag ? s.n_flag_y : 0);
   auto inside_open = [&](int i, int j) { return i > iA && i < iB && j > jA && j < jB; };
   auto inside_cell = [&](int i, int j) { return i >= iA && i < iB && j >= jA && j < jB; };
   auto flag_cell = [&](int i, int j) { return flag && i >= iB && i < iT && j >= jF0 && j < jF1; };

   // Ring radial parameters t_k (0 = circle, 1 = square).
   std::vector<double> t(s.n_ring + 1);
   {
      const auto r = Graded(0.0, 1.0, s.n_ring, s.ring_grading);
      for (int k = 0; k <= s.n_ring; ++k) { t[k] = r[k]; }
   }
   const int n_sq = 2 * s.n_side + 2 * n_sq_y;

   // The square boundary, counterclockwise from its bottom-left corner.
   std::vector<int> sq_i(n_sq), sq_j(n_sq);
   {
      int q = 0;
      for (int i = iA; i < iB; ++i) { sq_i[q] = i; sq_j[q] = jA; ++q; }
      for (int j = jA; j < jB; ++j) { sq_i[q] = iB; sq_j[q] = j; ++q; }
      for (int i = iB; i > iA; --i) { sq_i[q] = i; sq_j[q] = jB; ++q; }
      for (int j = jB; j > jA; --j) { sq_i[q] = iA; sq_j[q] = j; ++q; }
   }
   // The right side's nodes are q = n_side + (j - jA); its elements in the
   // flag's sector (from node j to j + 1, j in [jF0, jF1)) are the flag's.
   const int qF0 = s.n_side + (jF0 - jA), qF1 = s.n_side + (jF1 - jA);
   auto sector = [&](int q) { return flag && q >= qF0 && q < qF1; };
   auto right = [&](int q) { return sq_i[q] == iB; };
   auto square_pt = [&](int q) { return Point{X[sq_i[q]], Y[sq_j[q]]}; };
   // Circle point on the ray from the centre through square point S.
   auto circle_of = [&](const Point & S)
   {
      const double dx = S.x - s.cx, dy = S.y - s.cy;
      const double r = std::sqrt(dx * dx + dy * dy);
      return Point{s.cx + s.radius* dx / r, s.cy + s.radius* dy / r};
   };
   // With a flag the right side pairs by angle, piecewise linear in y, so
   // that y = cy -+ b meets the circle at y = cy -+ b: the flag's edges are
   // straight through the ring.
   const double th_c = 0.25 * M_PI, th_f = std::asin(b / s.radius);
   auto right_angle = [&](double y)
   {
      if (y <= s.cy - b) { return -th_c + (th_c - th_f) * (y - (s.cy - a)) / (a - b); }
      if (y >= s.cy + b) { return th_f + (th_c - th_f) * (y - (s.cy + b)) / (a - b); }
      return th_f * (y - s.cy) / b;
   };
   auto circle_at = [&](double th)
   {
      return Point{s.cx + s.radius* std::cos(th), s.cy + s.radius* std::sin(th)};
   };
   auto circle_q = [&](int q)
   {
      return flag && right(q) ? circle_at(right_angle(Y[sq_j[q]]))
             : circle_of(square_pt(q));
   };

   // Vertices (every grid and ring node; the unused ones are removed below).
   // (No preallocation: the counts depend on the part, and MFEM checks a
   // preallocated count against what was added.)
   Mesh mesh(2, 0, 0, 0, 2);
   std::vector<std::vector<int>> vid(nx + 1, std::vector<int>(ny + 1, -1));
   int v = 0;
   for (int i = 0; i <= nx; ++i)
      for (int j = 0; j <= ny; ++j)
      {
         if (inside_open(i, j)) { continue; }
         mesh.AddVertex(X[i], Y[j]);
         vid[i][j] = v++;
      }
   // Ring vertices for k < n_ring (k = n_ring is the square itself).
   std::vector<std::vector<int>> ring(n_sq, std::vector<int>(s.n_ring + 1, -1));
   for (int q = 0; q < n_sq; ++q)
   {
      const Point S = square_pt(q), C = circle_q(q);
      for (int k = 0; k < s.n_ring; ++k)
      {
         mesh.AddVertex(C.x + t[k] * (S.x - C.x), C.y + t[k] * (S.y - C.y));
         ring[q][k] = v++;
      }
      ring[q][s.n_ring] = vid[sq_i[q]][sq_j[q]];
   }

   // Elements: Cartesian blocks, then the ring (counterclockwise quads; ring
   // quads go radially out along xi_x and around along xi_y). The fluid mesh
   // leaves out the flag's cells, the solid mesh keeps only them.
   for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
         if (inside_cell(i, j) || flag_cell(i, j) != s.solid) { continue; }
         mesh.AddQuad(vid[i][j], vid[i + 1][j], vid[i + 1][j + 1], vid[i][j + 1], 1);
      }
   std::vector<std::vector<int>> ring_elem(n_sq, std::vector<int>(s.n_ring, -1));
   for (int q = 0; q < n_sq; ++q)
   {
      if (sector(q) != s.solid) { continue; }
      const int q1 = (q + 1) % n_sq;
      for (int k = 0; k < s.n_ring; ++k)
      {
         ring_elem[q][k] = mesh.GetNE();
         mesh.AddQuad(ring[q][k], ring[q][k + 1], ring[q1][k + 1], ring[q1][k], 1);
      }
   }

   // Boundary.
   if (!s.solid)
   {
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
   }
   for (int q = 0; q < n_sq; ++q)
   {
      if (sector(q) != s.solid) { continue; }
      mesh.AddBdrSegment(ring[(q + 1) % n_sq][0], ring[q][0], kCylinderBody);
   }
   if (flag)
   {
      // The flag's surface: its lower and upper edges through the ring and
      // the Cartesian block, and its tip.
      for (int k = 0; k < s.n_ring; ++k)
      {
         mesh.AddBdrSegment(ring[qF0][k], ring[qF0][k + 1], kCylinderFlag);
         mesh.AddBdrSegment(ring[qF1][k + 1], ring[qF1][k], kCylinderFlag);
      }
      for (int i = iB; i < iT; ++i)
      {
         mesh.AddBdrSegment(vid[i][jF0], vid[i + 1][jF0], kCylinderFlag);
         mesh.AddBdrSegment(vid[i + 1][jF1], vid[i][jF1], kCylinderFlag);
      }
      for (int j = jF0; j < jF1; ++j)
      {
         mesh.AddBdrSegment(vid[iT][j], vid[iT][j + 1], kCylinderFlag);
      }
   }
   mesh.FinalizeQuadMesh(1, 1, true);
   if (flag) { mesh.RemoveUnusedVertices(); }

   // Exact curved geometry: high-order nodes of the ring elements from the
   // transfinite ring map x(sigma, t) = C(sigma) + t (S(sigma) - C(sigma)),
   // with S piecewise linear along the square and C its ray projection on the
   // circle (with a flag, on the right side, C at the angle interpolated
   // between the element's ends). Cartesian elements keep the (exact)
   // bilinear interpolation.
   mesh.SetCurvature(s.order, false, 2, Ordering::byVDIM);
   GridFunction& nodes = *mesh.GetNodes();
   const FiniteElementSpace& nfes = *nodes.FESpace();
   Array<int> vdofs;
   for (int q = 0; q < n_sq; ++q)
   {
      if (ring_elem[q][0] < 0) { continue; }
      const int q1 = (q + 1) % n_sq;
      const Point S0 = square_pt(q), S1 = square_pt(q1);
      const bool by_angle = flag && right(q) && right(q1);
      const double th0 = by_angle ? right_angle(S0.y) : 0.0;
      const double th1 = by_angle ? right_angle(S1.y) : 0.0;
      for (int k = 0; k < s.n_ring; ++k)
      {
         const int e = ring_elem[q][k];
         const FiniteElement& fe = *nfes.GetFE(e);
         const IntegrationRule& nir = fe.GetNodes();
         nfes.GetElementVDofs(e, vdofs);
         const int nd = fe.GetDof();
         for (int j = 0; j < nd; ++j)
         {
            const IntegrationPoint& ip = nir.IntPoint(j);
            const Point S{S0.x + ip.y* (S1.x - S0.x), S0.y + ip.y* (S1.y - S0.y)};
            const Point C = by_angle ? circle_at(th0 + ip.y * (th1 - th0))
                            : circle_of(S);
            const double tt = t[k] + ip.x * (t[k + 1] - t[k]);
            nodes(vdofs[j]) = C.x + tt * (S.x - C.x);
            nodes(vdofs[j + nd]) = C.y + tt * (S.y - C.y);
         }
      }
   }
   return mesh;
}

} // namespace incns
