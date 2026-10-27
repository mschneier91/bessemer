#include "post/flow_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

namespace incns
{

using namespace mfem;

namespace
{

const char* const kSplitFile = "flow_statistics_split.txt";
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kTwoPi = 6.283185307179586476925286766559;

// The unit vector along @p a (its first dim components).
void Unit(const std::array<double, 3>& a, int dim, double e[3])
{
   double n = 0.0;
   for (int d = 0; d < dim; ++d) { n += a[d] * a[d]; }
   n = std::sqrt(n);
   for (int d = 0; d < 3; ++d) { e[d] = (d < dim) ? a[d] / n : 0.0; }
}

double Dot(const double* a, const double* b, int dim)
{
   double s = 0.0;
   for (int d = 0; d < dim; ++d) { s += a[d] * b[d]; }
   return s;
}

// c = a x b (3D).
void Cross(const double* a, const double* b, double* c)
{
   c[0] = a[1] * b[2] - a[2] * b[1];
   c[1] = a[2] * b[0] - a[0] * b[2];
   c[2] = a[0] * b[1] - a[1] * b[0];
}

// Symmetric tensor component names in the slot order xx, yy(, zz), xy(, xz,
// yz), with the axis letters @p ax.
std::vector<std::string> SymNames(const std::string& prefix, const char* ax,
                                  int dim)
{
   std::vector<std::string> n;
   for (int a = 0; a < dim; ++a) { n.push_back(prefix + ax[a] + ax[a]); }
   n.push_back(prefix + ax[0] + ax[1]);
   if (dim == 3)
   {
      n.push_back(prefix + ax[0] + ax[2]);
      n.push_back(prefix + ax[1] + ax[2]);
   }
   return n;
}

std::vector<int> Range(int first, int count)
{
   std::vector<int> r(count);
   for (int i = 0; i < count; ++i) { r[i] = first + i; }
   return r;
}

std::string Vec(const std::array<double, 3>& a, int dim)
{
   std::ostringstream s;
   s << std::setprecision(12);
   for (int d = 0; d < dim; ++d) { s << (d ? " " : "") << a[d]; }
   return s.str();
}

} // namespace

FlowSampler::FlowSampler(ParMesh& mesh, const FlowStatisticsParameters& params,
                         double nu, int order_u)
   : mesh_(mesh), params_(params), nu_(nu), order_(order_u)
{
   dim_ = mesh.Dimension();
   nsym_ = (dim_ == 3) ? 6 : 3;
   root_ = (mesh.GetMyRank() == 0);
   for (std::size_t i = 0; i < params_.lines.size(); ++i)
   {
      const StatisticsLine& l = params_.lines[i];
      MFEM_VERIFY(l.points >= 2, "flow_statistics: line '" << l.name
                  << "' needs points >= 2");
      sets_.push_back({l.name, "line", static_cast<int>(i), total_, l.points});
      total_ += l.points;
   }
   for (std::size_t i = 0; i < params_.profiles.size(); ++i)
   {
      const StatisticsProfiles& s = params_.profiles[i];
      MFEM_VERIFY(dim_ == 3, "flow_statistics: profiles are for 3D (use lines "
                  "in 2D)");
      const int n = static_cast<int>(s.stations.size()) * s.points * s.azimuths;
      sets_.push_back({s.name, "profile", static_cast<int>(i), total_, n});
      total_ += n;
   }
   for (const StatisticsWall& w : params_.walls)
   {
      MFEM_VERIFY(!w.attributes.empty(), "flow_statistics: wall '" << w.name
                  << "' has no boundary attributes");
   }
   if (total_ > 0)
   {
      if (root_) { Points(); }
      Locate();
   }
}

int FlowSampler::Sym(int a, int b) const
{
   if (a > b) { std::swap(a, b); }
   if (a == b) { return a; }
   if (dim_ == 2) { return 2; }
   return (a == 0) ? (b == 1 ? 3 : 4) : 5;
}

void FlowSampler::Frame(const StatisticsProfiles& s, double theta,
                        double e[3][3]) const
{
   // e_a, b1 = the coordinate axis least aligned with e_a made perpendicular,
   // b2 = e_a x b1; then e_r = cos b1 + sin b2, e_t = e_a x e_r.
   double a[3];
   Unit(s.axis, 3, a);
   int c = 0;
   for (int d = 1; d < 3; ++d)
   {
      if (std::abs(a[d]) < std::abs(a[c])) { c = d; }
   }
   double b1[3] = {0.0, 0.0, 0.0};
   b1[c] = 1.0;
   for (int d = 0; d < 3; ++d) { b1[d] -= a[c] * a[d]; }
   const double nb = std::sqrt(Dot(b1, b1, 3));
   for (double& v : b1) { v /= nb; }
   double b2[3];
   Cross(a, b1, b2);
   const double ct = std::cos(theta), st = std::sin(theta);
   for (int d = 0; d < 3; ++d)
   {
      e[0][d] = a[d];
      e[1][d] = ct * b1[d] + st * b2[d];
      e[2][d] = -st * b1[d] + ct * b2[d];
   }
}

void FlowSampler::Points()
{
   coords_.assign(static_cast<std::size_t>(total_) * dim_, 0.0);
   for (const PointSet& ps : sets_)
   {
      double* x = &coords_[static_cast<std::size_t>(ps.first) * dim_];
      if (ps.kind == "line")
      {
         const StatisticsLine& l = params_.lines[ps.index];
         for (int i = 0; i < l.points; ++i)
         {
            const double s = static_cast<double>(i) / (l.points - 1);
            for (int d = 0; d < dim_; ++d)
            {
               x[i * dim_ + d] = l.from[d] + s * (l.to[d] - l.from[d]);
            }
         }
         continue;
      }
      const StatisticsProfiles& s = params_.profiles[ps.index];
      int i = 0;
      for (double station : s.stations)
      {
         for (int j = 0; j < s.points; ++j)
         {
            const double r = s.radius * j / (s.points - 1);
            for (int k = 0; k < s.azimuths; ++k, ++i)
            {
               double e[3][3];
               Frame(s, kTwoPi * k / s.azimuths, e);
               for (int d = 0; d < 3; ++d)
               {
                  x[i * 3 + d] = s.origin[d] + station * e[0][d] + r * e[1][d];
               }
            }
         }
      }
   }
}

void FlowSampler::Locate()
{
#ifdef MFEM_USE_GSLIB
   const MPI_Comm comm = mesh_.GetComm();
   int rank = 0;
   MPI_Comm_rank(comm, &rank);
   // Rank 0 asks for every point; gslib needs mesh nodes, so a mesh without
   // them is searched through a copy (the case's mesh is left as it is).
   Vector pts(root_ ? total_ * dim_ : 0);
   for (int i = 0; i < pts.Size(); ++i) { pts(i) = coords_[i]; }
   std::unique_ptr<ParMesh> copy;
   ParMesh* m = &mesh_;
   if (!mesh_.GetNodes())
   {
      copy = std::make_unique<ParMesh>(mesh_, true);
      copy->EnsureNodes();
      m = copy.get();
   }
   // A point gslib finds on an element's border but away from it lies just
   // outside the domain: not found either (tolerance: 1e-9 of the domain).
   Vector lo, hi;
   mesh_.GetBoundingBox(lo, hi);
   MPI_Allreduce(MPI_IN_PLACE, lo.GetData(), dim_, MPI_DOUBLE, MPI_MIN, comm);
   MPI_Allreduce(MPI_IN_PLACE, hi.GetData(), dim_, MPI_DOUBLE, MPI_MAX, comm);
   double diag = 0.0;
   for (int d = 0; d < dim_; ++d) { diag += (hi(d) - lo(d)) * (hi(d) - lo(d)); }
   const double dist_tol = 1e-9 * std::sqrt(diag);
   std::vector<int> info(3 * static_cast<std::size_t>(total_), 0);
   std::vector<double> ref(static_cast<std::size_t>(total_) * dim_, 0.0);
   {
      FindPointsGSLIB finder(comm);
      finder.Setup(*m);
      finder.FindPoints(pts, Ordering::byVDIM);
      if (root_)
      {
         const Array<unsigned int>& code = finder.GetCode();
         const Array<unsigned int>& proc = finder.GetProc();
         const Array<unsigned int>& elem = finder.GetElem();
         const double* r = finder.GetReferencePosition().HostRead();
         const double* dist = finder.GetDist().HostRead();
         for (int i = 0; i < total_; ++i)
         {
            const bool outside = code[i] == 2 || (code[i] == 1 && dist[i] > dist_tol);
            info[3 * i] = outside ? 2 : static_cast<int>(code[i]);
            info[3 * i + 1] = static_cast<int>(proc[i]);
            info[3 * i + 2] = static_cast<int>(elem[i]);
         }
         std::copy(r, r + ref.size(), ref.begin());
      }
      finder.FreeData();
   }
   MPI_Bcast(info.data(), static_cast<int>(info.size()), MPI_INT, 0, comm);
   MPI_Bcast(ref.data(), static_cast<int>(ref.size()), MPI_DOUBLE, 0, comm);
   if (root_) { found_.assign(total_, 0); }
   for (int i = 0; i < total_; ++i)
   {
      const bool found = info[3 * i] != 2;
      if (root_)
      {
         found_[i] = found ? 1 : 0;
         if (!found) { ++not_found_; }
      }
      if (!found || info[3 * i + 1] != rank) { continue; }
      OwnedPoint op;
      op.index = i;
      op.element = info[3 * i + 2];
      for (int d = 0; d < dim_; ++d) { op.ref[d] = ref[i * dim_ + d]; }
      owned_.push_back(op);
   }
#else
   MFEM_ABORT("flow_statistics: lines and profiles need MFEM built with GSLIB");
#endif
}

void FlowSampler::WallSums(const FlowStatistics& stats,
                           const StatisticsWall& set,
                           std::vector<double>& sums) const
{
   sums.assign(static_cast<std::size_t>(set.bins) * kWallWidth + 1, 0.0);
   const ParGridFunction& U = stats.MeanVelocity();
   const ParGridFunction& P = stats.MeanPressure();
   double e[3];
   Unit(set.axis, dim_, e);
   const double span = set.to - set.from;
   DenseMatrix grad(dim_);
   Vector nor(dim_), x(dim_);
   for (int be = 0; be < mesh_.GetNBE(); ++be)
   {
      const int attr = mesh_.GetBdrAttribute(be);
      if (std::find(set.attributes.begin(), set.attributes.end(), attr) ==
          set.attributes.end()) { continue; }
      FaceElementTransformations* ft = mesh_.GetBdrFaceTransformations(be);
      if (!ft) { continue; }
      const int order = 2 * order_ + 2;
      const IntegrationRule& ir = IntRules.Get(ft->GetGeometryType(), order);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         ft->SetAllIntPoints(&ip);
         ft->Transform(ip, x);
         double xi = 0.0;
         for (int d = 0; d < dim_; ++d) { xi += (x(d) - set.origin[d]) * e[d]; }
         if (xi < set.from || xi > set.to) { continue; }
         // The unit normal into the fluid (CalcOrtho: out of element 1).
         CalcOrtho(ft->Jacobian(), nor);
         const double area = nor.Norml2();
         double n[3] = {0.0, 0.0, 0.0};
         for (int d = 0; d < dim_; ++d) { n[d] = -nor(d) / area; }
         // The axis's direction along the wall.
         double ea[3] = {0.0, 0.0, 0.0};
         const double en = Dot(e, n, dim_);
         for (int d = 0; d < dim_; ++d) { ea[d] = e[d] - en * n[d]; }
         const double na = std::sqrt(Dot(ea, ea, dim_));
         if (na < 1e-3)
         {
            sums.back() += 1.0;
            continue;
         }
         for (int d = 0; d < dim_; ++d) { ea[d] /= na; }
         ElementTransformation& T1 = ft->GetElement1Transformation();
         U.GetVectorGradient(T1, grad);
         const double p = P.GetValue(T1, ft->GetElement1IntPoint());
         // The traction of the mean flow's viscous stress and its tangential part.
         double t[3] = {0.0, 0.0, 0.0};
         for (int a = 0; a < dim_; ++a)
         {
            for (int b = 0; b < dim_; ++b)
            {
               t[a] += nu_ * (grad(a, b) + grad(b, a)) * n[b];
            }
         }
         const double tn = Dot(t, n, dim_);
         for (int d = 0; d < dim_; ++d) { t[d] -= tn * n[d]; }
         double cross = 0.0;
         if (dim_ == 3)
         {
            double ec[3];
            Cross(n, ea, ec);
            cross = Dot(t, ec, 3);
         }
         const int bin = std::min(static_cast<int>((xi - set.from) / span * set.bins),
                                  set.bins - 1);
         const double w = ip.weight * area;
         double* s = &sums[static_cast<std::size_t>(bin) * kWallWidth];
         s[0] += w;
         s[1] += w * p;
         s[2] += w * Dot(t, ea, dim_);
         s[3] += w * cross;
      }
   }
}

SampleValues FlowSampler::Evaluate(const FlowStatistics& stats) const
{
   const MPI_Comm comm = mesh_.GetComm();
   SampleValues out;
   if (total_ > 0)
   {
      const int nv = PointWidth();
      std::vector<double> buf(static_cast<std::size_t>(total_) * nv, 0.0);
      const ParGridFunction& U = stats.MeanVelocity();
      const ParGridFunction& P = stats.MeanPressure();
      const ParGridFunction& RN = stats.NormalStress();
      const ParGridFunction& RS = stats.ShearStress();
      Vector u(dim_), rn(dim_), rs(dim_);
      DenseMatrix grad(dim_);
      for (const OwnedPoint& op : owned_)
      {
         IntegrationPoint ip;
         ip.Set(op.ref, dim_);
         ElementTransformation* T = mesh_.GetElementTransformation(op.element);
         T->SetIntPoint(&ip);
         U.GetVectorValue(*T, ip, u);
         U.GetVectorGradient(*T, grad);
         RN.GetVectorValue(*T, ip, rn);
         RS.GetVectorValue(*T, ip, rs);
         double* o = &buf[static_cast<std::size_t>(op.index) * nv];
         for (int a = 0; a < dim_; ++a) { o[a] = u(a); }
         o[dim_] = P.GetValue(*T, ip);
         double* R = o + dim_ + 1;
         double* tau = R + nsym_;
         for (int a = 0; a < dim_; ++a) { R[a] = rn(a); }
         R[Sym(0, 1)] = rs(0);
         if (dim_ == 3)
         {
            R[Sym(0, 2)] = rs(1);
            R[Sym(1, 2)] = rs(2);
         }
         for (int a = 0; a < dim_; ++a)
         {
            for (int b = a; b < dim_; ++b)
            {
               tau[Sym(a, b)] = nu_ * (grad(a, b) + grad(b, a));
            }
         }
      }
      // Each point has one owner: the sum is its values, exactly.
      MPI_Reduce(root_ ? MPI_IN_PLACE : buf.data(), root_ ? buf.data() : nullptr,
                 static_cast<int>(buf.size()), MPI_DOUBLE, MPI_SUM, 0, comm);
      if (root_)
      {
         for (const PointSet& ps : sets_)
         {
            std::vector<double> v(buf.begin() + static_cast<std::ptrdiff_t>(ps.first) * nv,
                                  buf.begin() + static_cast<std::ptrdiff_t>(ps.first + ps.count) * nv);
            for (int i = 0; i < ps.count; ++i)
            {
               if (found_[ps.first + i]) { continue; }
               std::fill(v.begin() + static_cast<std::ptrdiff_t>(i) * nv,
                         v.begin() + static_cast<std::ptrdiff_t>(i + 1) * nv, kNaN);
            }
            out.points.push_back(std::move(v));
         }
      }
   }
   for (const StatisticsWall& w : params_.walls)
   {
      std::vector<double> sums;
      WallSums(stats, w, sums);
      MPI_Reduce(root_ ? MPI_IN_PLACE : sums.data(), root_ ? sums.data() : nullptr,
                 static_cast<int>(sums.size()), MPI_DOUBLE, MPI_SUM, 0, comm);
      if (!root_) { continue; }
      for (int b = 0; b < w.bins; ++b)
      {
         double* s = &sums[static_cast<std::size_t>(b) * kWallWidth];
         for (int k = 1; k < kWallWidth; ++k) { s[k] = (s[0] > 0.0) ? s[k] / s[0] : kNaN; }
         if (dim_ == 2 && s[0] > 0.0) { s[3] = 0.0; }
      }
      out.wall_skipped.push_back(static_cast<long long>(sums.back()));
      sums.pop_back();
      out.walls.push_back(std::move(sums));
   }
   return out;
}

void FlowSampler::LineTable(const std::vector<double>& v,
                            const StatisticsLine& set,
                            SampleTable& t) const
{
   const char* ax = "xyz";
   const int nv = PointWidth();
   t.columns = {"s"};
   for (int d = 0; d < dim_; ++d) { t.columns.push_back(std::string(1, ax[d])); }
   const int c_u = static_cast<int>(t.columns.size());
   for (int d = 0; d < dim_; ++d) { t.columns.push_back(std::string(1, "uvw"[d])); }
   t.columns.push_back("p");
   for (const std::string& n : SymNames("R_", ax, dim_)) { t.columns.push_back(n); }
   for (const std::string& n : SymNames("tau_", ax, dim_)) { t.columns.push_back(n); }
   t.groups = {{"velocity", Range(c_u, dim_)}, {"pressure", Range(c_u + dim_, 1)},
      {"reynolds_stress", Range(c_u + dim_ + 1, nsym_)},
      {"viscous_stress", Range(c_u + dim_ + 1 + nsym_, nsym_)}
   };
   t.header = {"from " + Vec(set.from, dim_), "to " + Vec(set.to, dim_)};
   double len = 0.0;
   for (int d = 0; d < dim_; ++d) { len += (set.to[d] - set.from[d]) * (set.to[d] - set.from[d]); }
   len = std::sqrt(len);
   const int first = [&]
   {
      for (const PointSet& ps : sets_) { if (ps.kind == "line" && ps.name == set.name) { return ps.first; } }
      return 0;
   }();
   for (int i = 0; i < set.points; ++i)
   {
      t.values.push_back(len * i / (set.points - 1));
      for (int d = 0; d < dim_; ++d)
      {
         t.values.push_back(coords_[static_cast<std::size_t>(first + i) * dim_ + d]);
      }
      for (int k = 0; k < nv; ++k) { t.values.push_back(v[static_cast<std::size_t>(i) * nv + k]); }
   }
}

void FlowSampler::ProfileTable(const std::vector<double>& v,
                               const StatisticsProfiles& set, SampleTable& t) const
{
   const char* ax = "art"; // axial, radial, azimuthal
   const int nv = PointWidth();
   t.columns = {"station", "r", "n", "u_a", "u_r", "u_t", "p"};
   for (const std::string& n : SymNames("R_", ax, 3)) { t.columns.push_back(n); }
   for (const std::string& n : SymNames("tau_", ax, 3)) { t.columns.push_back(n); }
   t.groups = {{"velocity", Range(3, 3)}, {"pressure", Range(6, 1)},
      {"reynolds_stress", Range(7, 6)}, {"viscous_stress", Range(13, 6)}
   };
   std::ostringstream rad;
   rad << std::setprecision(12) << "radius " << set.radius;
   t.header = {"origin " + Vec(set.origin, 3), "axis " + Vec(set.axis, 3),
               rad.str(), "azimuths " + std::to_string(set.azimuths)
              };
   const int width = 1 + 3 + 1 + 2 * 6; // n, u, p, R, tau
   int i = 0;
   for (double station : set.stations)
   {
      for (int j = 0; j < set.points; ++j)
      {
         std::vector<double> acc(width, 0.0);
         for (int k = 0; k < set.azimuths; ++k, ++i)
         {
            const double* o = &v[static_cast<std::size_t>(i) * nv];
            if (std::isnan(o[0])) { continue; }
            double e[3][3];
            Frame(set, kTwoPi * k / set.azimuths, e);
            acc[0] += 1.0;
            for (int a = 0; a < 3; ++a) { acc[1 + a] += Dot(o, e[a], 3); }
            acc[4] += o[3];
            for (int part = 0; part < 2; ++part)
            {
               // The Cartesian tensor, then e_a^T T e_b in the slot order.
               const double* S = o + 4 + part * 6;
               double T[3][3];
               for (int a = 0; a < 3; ++a)
               {
                  for (int b = 0; b < 3; ++b) { T[a][b] = S[Sym(a, b)]; }
               }
               for (int a = 0; a < 3; ++a)
               {
                  for (int b = a; b < 3; ++b)
                  {
                     double s = 0.0;
                     for (int c = 0; c < 3; ++c)
                     {
                        for (int d = 0; d < 3; ++d) { s += e[a][c] * T[c][d] * e[b][d]; }
                     }
                     acc[5 + part * 6 + Sym(a, b)] += s;
                  }
               }
            }
         }
         t.values.push_back(station);
         t.values.push_back(set.radius * j / (set.points - 1));
         t.values.push_back(acc[0]);
         for (int k = 1; k < width; ++k)
         {
            t.values.push_back(acc[0] > 0.0 ? acc[k] / acc[0] : kNaN);
         }
      }
   }
}

void FlowSampler::WallTable(const std::vector<double>& v,
                            const StatisticsWall& set,
                            SampleTable& t) const
{
   t.columns = {"xi", "xi_from", "xi_to", "area", "p", "wss_axial"};
   if (dim_ == 3) { t.columns.push_back("wss_cross"); }
   t.groups = {{"pressure", {4}}, {"wss", Range(5, dim_ == 3 ? 2 : 1)}};
   std::string names;
   for (const std::string& b : set.boundaries) { names += (names.empty() ? "" : " ") + b; }
   t.header = {"boundaries " + names, "origin " + Vec(set.origin, dim_),
               "axis " + Vec(set.axis, dim_)
              };
   const double h = (set.to - set.from) / set.bins;
   for (int b = 0; b < set.bins; ++b)
   {
      const double* s = &v[static_cast<std::size_t>(b) * kWallWidth];
      t.values.push_back(set.from + (b + 0.5) * h);
      t.values.push_back(set.from + b * h);
      t.values.push_back(set.from + (b + 1) * h);
      for (int k = 0; k < (dim_ == 3 ? 4 : 3); ++k) { t.values.push_back(s[k]); }
   }
}

std::vector<SampleTable> FlowSampler::Tabulate(const SampleValues& v) const
{
   std::vector<SampleTable> tables;
   if (!root_) { return tables; }
   for (std::size_t i = 0; i < sets_.size(); ++i)
   {
      SampleTable t;
      t.name = sets_[i].name;
      t.kind = sets_[i].kind;
      if (t.kind == "line") { LineTable(v.points[i], params_.lines[sets_[i].index], t); }
      else { ProfileTable(v.points[i], params_.profiles[sets_[i].index], t); }
      tables.push_back(std::move(t));
   }
   for (std::size_t i = 0; i < params_.walls.size(); ++i)
   {
      SampleTable t;
      t.name = params_.walls[i].name;
      t.kind = "wall";
      WallTable(v.walls[i], params_.walls[i], t);
      if (i < v.wall_skipped.size())
      {
         t.header.push_back("skipped " + std::to_string(v.wall_skipped[i]));
      }
      tables.push_back(std::move(t));
   }
   return tables;
}

void FlowSampler::Snapshot(const FlowStatistics& stats)
{
   split_ = Evaluate(stats);
   split_weight_ = stats.AveragingTime();
   split_time_ = stats.LastTime();
   split_taken_ = true;
}

SampleValues FlowSampler::SecondPart(const SampleValues& all,
                                     double weight) const
{
   SampleValues out = all;
   if (!root_) { return out; }
   const double W1 = split_weight_, W2 = weight - W1;
   MFEM_VERIFY(split_taken_ && W2 > 0.0, "flow_statistics: no second part yet");
   // x2 = (W x - W1 x1) / W2 for the means; the stresses by the parallel-axis
   // rule W R = W1 R1 + W2 R2 + (W1 W2 / W) d d^T, d = U2 - U1.
   const double a = weight / W2, b = W1 / W2, c = W1 / weight;
   const int nv = PointWidth();
   for (std::size_t s = 0; s < out.points.size(); ++s)
   {
      std::vector<double>& v = out.points[s];
      const std::vector<double>& v1 = split_.points[s];
      for (std::size_t i = 0; i < v.size() / nv; ++i)
      {
         double* x = &v[i * nv];
         const double* x1 = &v1[i * nv];
         double d[3] = {0.0, 0.0, 0.0};
         for (int k = 0; k < nv; ++k)
         {
            if (k >= dim_ + 1 && k < dim_ + 1 + nsym_) { continue; } // R below
            x[k] = a * x[k] - b * x1[k];
            if (k < dim_) { d[k] = x[k] - x1[k]; }
         }
         double* R = x + dim_ + 1;
         const double* R1 = x1 + dim_ + 1;
         for (int p = 0; p < dim_; ++p)
         {
            for (int q = p; q < dim_; ++q)
            {
               const int k = Sym(p, q);
               R[k] = a * R[k] - b * R1[k] - c * d[p] * d[q];
            }
         }
      }
   }
   for (std::size_t s = 0; s < out.walls.size(); ++s)
   {
      std::vector<double>& v = out.walls[s];
      const std::vector<double>& v1 = split_.walls[s];
      for (std::size_t i = 0; i < v.size(); ++i)
      {
         if (i % kWallWidth == 0) { continue; } // the area
         v[i] = a * v[i] - b * v1[i];
      }
   }
   return out;
}

std::vector<HalfWindowDifference>
FlowSampler::Compare(const SampleValues& all, double weight) const
{
   std::vector<HalfWindowDifference> out;
   if (!root_ || !split_taken_ || weight <= split_weight_) { return out; }
   const std::vector<SampleTable> ta = Tabulate(all);
   const std::vector<SampleTable> t1 = Tabulate(split_);
   const std::vector<SampleTable> t2 = Tabulate(SecondPart(all, weight));
   for (std::size_t s = 0; s < ta.size(); ++s)
   {
      const int nc = static_cast<int>(ta[s].columns.size());
      for (const auto& [group, cols] : ta[s].groups)
      {
         const bool range = (group == "pressure");
         double diff = 0.0, scale = 0.0;
         double lo = std::numeric_limits<double>::max(), hi = -lo;
         bool any = false;
         for (int r = 0; r < ta[s].Rows(); ++r)
         {
            for (int c : cols)
            {
               const double v = ta[s].values[r * nc + c];
               const double a = t1[s].values[r * nc + c];
               const double b = t2[s].values[r * nc + c];
               if (std::isnan(v) || std::isnan(a) || std::isnan(b)) { continue; }
               any = true;
               diff = std::max(diff, std::abs(a - b));
               lo = std::min(lo, v);
               hi = std::max(hi, v);
               scale = std::max(scale, std::abs(v));
            }
         }
         if (range && any && hi > lo) { scale = hi - lo; }
         HalfWindowDifference d;
         d.set = ta[s].name;
         d.group = group;
         d.relative = (any && scale > 0.0) ? diff / scale : kNaN;
         out.push_back(d);
      }
   }
   return out;
}

void FlowSampler::WriteTables(const std::vector<SampleTable>& tables,
                              const std::string& prefix, const std::string& suffix,
                              const std::vector<std::string>& header) const
{
   for (const SampleTable& t : tables)
   {
      const std::string path = prefix + "_" + t.name + suffix + ".csv";
      std::ofstream f(path);
      MFEM_VERIFY(f, "flow_statistics: cannot write " << path);
      f << std::setprecision(12);
      f << "# flow statistics (incns post/flow_sampling): " << t.kind << " "
        << t.name << "\n";
      for (const std::string& h : header) { f << "# " << h << "\n"; }
      for (const std::string& h : t.header) { f << "# " << h << "\n"; }
      for (std::size_t c = 0; c < t.columns.size(); ++c)
      {
         f << (c ? "," : "") << t.columns[c];
      }
      f << "\n";
      const std::size_t nc = t.columns.size();
      for (int r = 0; r < t.Rows(); ++r)
      {
         for (std::size_t c = 0; c < nc; ++c)
         {
            f << (c ? "," : "") << t.values[r * nc + c];
         }
         f << "\n";
      }
   }
}

void FlowSampler::WriteCsv(const SampleValues& all, const FlowStatistics& stats,
                           const std::string& prefix) const
{
   if (!root_) { return; }
   auto header = [&](const std::string & part, double weight, double t0, double t1)
   {
      std::ostringstream s;
      s << std::setprecision(12);
      std::vector<std::string> h;
      s << "part " << part;
      h.push_back(s.str());
      s.str("");
      s << "nu " << nu_;
      h.push_back(s.str());
      s.str("");
      s << "averaging_time " << weight;
      h.push_back(s.str());
      s.str("");
      s << "window_start " << t0;
      h.push_back(s.str());
      s.str("");
      s << "window_end " << t1;
      h.push_back(s.str());
      s.str("");
      s << "samples " << stats.Samples();
      if (part == "whole") { h.push_back(s.str()); }
      return h;
   };
   WriteTables(Tabulate(all), prefix, "",
               header("whole", stats.AveragingTime(), stats.FirstTime(),
                      stats.LastTime()));
   if (!split_taken_ || stats.AveragingTime() <= split_weight_) { return; }
   WriteTables(Tabulate(split_), prefix, "_half1",
               header("first", split_weight_, stats.FirstTime(), split_time_));
   WriteTables(Tabulate(SecondPart(all, stats.AveragingTime())), prefix, "_half2",
               header("second", stats.AveragingTime() - split_weight_, split_time_,
                      stats.LastTime()));
}

void FlowSampler::Save(const std::string& dir) const
{
   if (!root_) { return; }
   std::ofstream f(dir + "/" + kSplitFile);
   MFEM_VERIFY(f, "flow_statistics: cannot write " << dir << "/" << kSplitFile);
   f << std::setprecision(17); // round-trips every double exactly
   f << "incns_flow_statistics_split 1\n"
     << "taken " << (split_taken_ ? 1 : 0) << " time " << split_time_
     << " weight " << split_weight_ << "\n";
   if (!split_taken_) { return; }
   f << "sets " << split_.points.size() + split_.walls.size() << "\n";
   for (std::size_t i = 0; i < split_.points.size(); ++i)
   {
      f << sets_[i].kind << " " << sets_[i].name << " " << split_.points[i].size()
        << "\n";
      for (double v : split_.points[i]) { f << v << "\n"; }
   }
   for (std::size_t i = 0; i < split_.walls.size(); ++i)
   {
      f << "wall " << params_.walls[i].name << " " << split_.walls[i].size() << "\n";
      for (double v : split_.walls[i]) { f << v << "\n"; }
   }
}

void FlowSampler::Load(const std::string& dir)
{
   const std::string path = dir + "/" + kSplitFile;
   int taken = 0, lost = 0;
   double head[2] = {0.0, 0.0};
   if (root_ && std::filesystem::exists(path))
   {
      std::ifstream f(path);
      std::string tag;
      int version = 0;
      f >> tag >> version;
      MFEM_VERIFY(tag == "incns_flow_statistics_split" && version == 1,
                  "flow_statistics: " << path << " is not a statistics split");
      f >> tag >> taken >> tag >> head[0] >> tag >> head[1];
      if (taken)
      {
         // Every current set starts without a split; a saved set with the
         // same kind, name and size gives it its values.
         const int nv = PointWidth();
         split_.points.clear();
         split_.walls.clear();
         for (const PointSet& ps : sets_)
         {
            split_.points.push_back(std::vector<double>(
                                       static_cast<std::size_t>(ps.count) * nv, kNaN));
         }
         for (const StatisticsWall& w : params_.walls)
         {
            split_.walls.push_back(std::vector<double>(
                                      static_cast<std::size_t>(w.bins) * kWallWidth, kNaN));
         }
         std::vector<char> matched(sets_.size() + params_.walls.size(), 0);
         std::size_t nsets = 0;
         f >> tag >> nsets;
         for (std::size_t s = 0; s < nsets; ++s)
         {
            std::string kind, name;
            std::size_t n = 0;
            f >> kind >> name >> n;
            std::vector<double> v(n);
            for (double& x : v)
            {
               // "nan" does not read back through operator>>.
               std::string word;
               f >> word;
               x = (word == "nan" || word == "-nan") ? kNaN : std::stod(word);
            }
            std::vector<double>* slot = nullptr;
            std::size_t idx = 0;
            for (std::size_t i = 0; i < sets_.size() && !slot; ++i)
            {
               if (sets_[i].kind == kind && sets_[i].name == name) { slot = &split_.points[i]; idx = i; }
            }
            for (std::size_t i = 0; i < params_.walls.size() && !slot; ++i)
            {
               if (kind == "wall" && params_.walls[i].name == name)
               {
                  slot = &split_.walls[i];
                  idx = sets_.size() + i;
               }
            }
            if (slot && slot->size() == n)
            {
               *slot = std::move(v);
               matched[idx] = 1;
            }
         }
         MFEM_VERIFY(f, "flow_statistics: " << path << " ends early");
         for (char m : matched) { if (!m) { ++lost; } }
      }
   }
   int flags[2] = {taken, lost};
   MPI_Bcast(flags, 2, MPI_INT, 0, mesh_.GetComm());
   MPI_Bcast(head, 2, MPI_DOUBLE, 0, mesh_.GetComm());
   split_taken_ = (flags[0] != 0);
   split_time_ = head[0];
   split_weight_ = head[1];
   if (root_ && flags[1] > 0)
   {
      mfem::out << "[incns] flow_statistics: " << flags[1] << " sample set(s) "
                "differ from the checkpoint's (new, renamed or resized): no "
                "half-window comparison for them" << std::endl;
   }
}

} // namespace incns
