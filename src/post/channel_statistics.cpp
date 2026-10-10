#include "post/channel_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace incns
{

using namespace mfem;

ChannelStatistics::ChannelStatistics(ParFiniteElementSpace& V, double nu,
                                     double start_time)
   : V_(V), nu_(nu), start_(start_time)
{
   ParMesh& mesh = *V.GetParMesh();
   const MPI_Comm comm = V.GetComm();
   dim_ = mesh.Dimension();
   MFEM_VERIFY(dim_ == 2 || dim_ == 3, "channel_statistics: 2D or 3D");
   MFEM_VERIFY(V.GetVDim() == dim_, "channel_statistics: needs the velocity "
               "space (vdim = dim)");
   MFEM_VERIFY(nu > 0.0, "channel_statistics: nu must be positive");
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const FiniteElement* fe = V.FEColl()->FiniteElementForGeometry(geom);
   MFEM_VERIFY(fe, "channel_statistics: needs quadrilaterals / hexahedra");
   const int k = V.FEColl()->GetOrder();
   ndof_ = fe->GetDof();

   // Node lines eta_j (Gauss-Lobatto, the H1 nodes along y) and the face
   // quadrature (Gauss, exact for products of the velocity's polynomials).
   IntegrationRule gll;
   QuadratureFunctions1D::GaussLobatto(k + 1, &gll);
   const IntegrationRule& g1 = IntRules.Get(Geometry::SEGMENT, 2 * k + 2);
   const int nq1 = g1.GetNPoints();
   const int npts = (dim_ == 3) ? nq1 * nq1 : nq1;
   lines_.resize(k + 1);
   shape_.resize(k + 1);
   Vector sh(ndof_);
   for (int j = 0; j <= k; ++j)
   {
      lines_[j].SetSize(npts);
      for (int a = 0; a < nq1; ++a)
      {
         for (int b = 0; b < ((dim_ == 3) ? nq1 : 1); ++b)
         {
            IntegrationPoint& ip = lines_[j].IntPoint(a * ((dim_ == 3) ? nq1 : 1) + b);
            if (dim_ == 3)
            {
               ip.Set(g1.IntPoint(a).x, gll.IntPoint(j).x, g1.IntPoint(b).x,
                      g1.IntPoint(a).weight * g1.IntPoint(b).weight);
            }
            else
            {
               ip.Set2w(g1.IntPoint(a).x, gll.IntPoint(j).x, g1.IntPoint(a).weight);
            }
         }
      }
      shape_[j].SetSize(npts, ndof_);
      for (int q = 0; q < npts; ++q)
      {
         fe->CalcShape(lines_[j].IntPoint(q), sh);
         for (int i = 0; i < ndof_; ++i) { shape_[j](q, i) = sh(i); }
      }
   }
   DenseMatrix ds(ndof_, dim_);
   dshape_bottom_.SetSize(npts, ndof_);
   dshape_top_.SetSize(npts, ndof_);
   for (int q = 0; q < npts; ++q)
   {
      fe->CalcDShape(lines_[0].IntPoint(q), ds);
      for (int i = 0; i < ndof_; ++i) { dshape_bottom_(q, i) = ds(i, 1); }
      fe->CalcDShape(lines_[k].IntPoint(q), ds);
      for (int i = 0; i < ndof_; ++i) { dshape_top_(q, i) = ds(i, 1); }
   }

   // Element geometry: axis-aligned affine boxes only (checked), and the
   // global extent.
   struct Box { double lo[3], hi[3]; };
   std::vector<Box> boxes(mesh.GetNE());
   double lo[3], hi[3];
   for (int d = 0; d < 3; ++d)
   {
      lo[d] = std::numeric_limits<double>::max();
      hi[d] = -std::numeric_limits<double>::max();
   }
   IntegrationPoint c0, c1;
   c0.Set3(0.5, 0.5, 0.5);
   c1.Set3(0.2, 0.7, 0.4);
   Vector x0, x1;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation* T = mesh.GetElementTransformation(e);
      T->SetIntPoint(&c0);
      const DenseMatrix J0 = T->Jacobian();
      T->SetIntPoint(&c1);
      const DenseMatrix& J1 = T->Jacobian();
      const double jmax = J0.MaxMaxNorm();
      for (int a = 0; a < dim_; ++a)
      {
         for (int b = 0; b < dim_; ++b)
         {
            MFEM_VERIFY(std::abs(J1(a, b) - J0(a, b)) <= 1e-10 * jmax &&
                        (a == b ? J0(a, b) > 0.0 : std::abs(J0(a, b)) <= 1e-10 * jmax),
                        "channel_statistics: every element must be an "
                        "axis-aligned box (affine, reference axes along x, y, z)");
         }
      }
      IntegrationPoint p0, p1;
      p0.Set3(0.0, 0.0, 0.0);
      p1.Set3(1.0, 1.0, 1.0);
      T->Transform(p0, x0);
      T->Transform(p1, x1);
      for (int d = 0; d < dim_; ++d)
      {
         boxes[e].lo[d] = x0(d);
         boxes[e].hi[d] = x1(d);
         lo[d] = std::min(lo[d], x0(d));
         hi[d] = std::max(hi[d], x1(d));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, lo, 3, MPI_DOUBLE, MPI_MIN, comm);
   MPI_Allreduce(MPI_IN_PLACE, hi, 3, MPI_DOUBLE, MPI_MAX, comm);
   y_min_ = lo[1];
   y_max_ = hi[1];
   delta_ = 0.5 * (y_max_ - y_min_);
   double area = hi[0] - lo[0];
   if (dim_ == 3) { area *= hi[2] - lo[2]; }
   const double tol = 1e-9 * (y_max_ - y_min_);

   // The stations: every element's node heights, merged across ranks.
   std::vector<double> local;
   for (const Box& b : boxes)
   {
      for (int j = 0; j <= k; ++j)
      {
         local.push_back(b.lo[1] + gll.IntPoint(j).x * (b.hi[1] - b.lo[1]));
      }
   }
   auto unique_sorted = [tol](std::vector<double>& v)
   {
      std::sort(v.begin(), v.end());
      std::vector<double> out;
      for (double y : v)
      {
         if (out.empty() || y - out.back() > tol) { out.push_back(y); }
      }
      v = std::move(out);
   };
   unique_sorted(local);
   int np = 1;
   MPI_Comm_size(comm, &np);
   int n_local = static_cast<int>(local.size());
   std::vector<int> counts(np), displs(np, 0);
   MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
   for (int r = 1; r < np; ++r) { displs[r] = displs[r - 1] + counts[r - 1]; }
   std::vector<double> all(displs[np - 1] + counts[np - 1]);
   MPI_Allgatherv(local.data(), n_local, MPI_DOUBLE, all.data(), counts.data(),
                  displs.data(), MPI_DOUBLE, comm);
   unique_sorted(all);
   y_ = all;
   const int ns = static_cast<int>(y_.size());

   auto station = [&](double y)
   {
      const auto it = std::lower_bound(y_.begin(), y_.end(), y - tol);
      MFEM_VERIFY(it != y_.end() && std::abs(*it - y) <= tol,
                  "channel_statistics: internal station lookup failed");
      return static_cast<int>(it - y_.begin());
   };

   // Per element: the stations of its node lines. A line shared by two
   // layers is averaged once (by the upper layer); the y-integration weights
   // take every line.
   std::vector<double> covered(ns, 0.0);
   y_weight_.assign(ns, 0.0);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      const Box& b = boxes[e];
      ElementLines el;
      el.element = e;
      el.dy = b.hi[1] - b.lo[1];
      double face = b.hi[0] - b.lo[0];
      if (dim_ == 3) { face *= b.hi[2] - b.lo[2]; }
      el.face_weight = face / area;
      el.bottom = std::abs(b.lo[1] - y_min_) <= tol;
      el.top = std::abs(b.hi[1] - y_max_) <= tol;
      el.station.resize(k + 1);
      for (int j = 0; j <= k; ++j)
      {
         const int s = station(b.lo[1] + gll.IntPoint(j).x * el.dy);
         el.station[j] = s;
         y_weight_[s] += gll.IntPoint(j).weight * el.dy * el.face_weight;
         if (j < k || el.top) { covered[s] += el.face_weight; }
      }
      elements_.push_back(std::move(el));
   }
   MPI_Allreduce(MPI_IN_PLACE, covered.data(), ns, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, y_weight_.data(), ns, MPI_DOUBLE, MPI_SUM, comm);
   for (int s = 0; s < ns; ++s)
   {
      MFEM_VERIFY(std::abs(covered[s] - 1.0) <= 1e-9, "channel_statistics: the "
                  "elements at y = " << y_[s] << " cover " << covered[s]
                  << " of the x-z plane; every node height must span the whole "
                  "plane (a box mesh without refinement)");
   }

   const std::size_t nval = static_cast<std::size_t>(ns) * kQuantities + 2;
   prev_.assign(nval, 0.0);
   sums_.assign(nval, 0.0);
}

std::vector<double>
ChannelStatistics::PlaneAverages(const ParGridFunction& u) const
{
   const int ns = static_cast<int>(y_.size());
   const int k = static_cast<int>(lines_.size()) - 1;
   std::vector<double> r(static_cast<std::size_t>(ns) * kQuantities + 2, 0.0);
   u.HostRead();
   Array<int> vdofs;
   Vector vals;
   const int npts = lines_[0].GetNPoints();
   Vector ua(npts), ub(npts), uc(npts);
   uc = 0.0;
   for (const ElementLines& el : elements_)
   {
      V_.GetElementVDofs(el.element, vdofs);
      u.GetSubVector(vdofs, vals);
      const double* vx = vals.GetData();
      const double* vy = vx + ndof_;
      const double* vz = (dim_ == 3) ? vx + 2 * ndof_ : nullptr;
      for (int j = 0; j <= k; ++j)
      {
         if (j == k && !el.top) { continue; } // the next layer's line 0
         const DenseMatrix& N = shape_[j];
         N.Mult(vx, ua.GetData());
         N.Mult(vy, ub.GetData());
         if (vz) { N.Mult(vz, uc.GetData()); }
         double acc[kQuantities] = {0, 0, 0, 0, 0, 0, 0};
         for (int q = 0; q < npts; ++q)
         {
            const double w = lines_[j].IntPoint(q).weight * el.face_weight;
            const double a = ua(q), b = ub(q), c = uc(q);
            acc[0] += w * a;
            acc[1] += w * b;
            acc[2] += w * c;
            acc[3] += w * a * a;
            acc[4] += w * b * b;
            acc[5] += w * c * c;
            acc[6] += w * a * b;
         }
         double* out = &r[static_cast<std::size_t>(el.station[j]) * kQuantities];
         for (int m = 0; m < kQuantities; ++m) { out[m] += acc[m]; }
      }
      for (int wall = 0; wall < 2; ++wall)
      {
         if (!(wall == 0 ? el.bottom : el.top)) { continue; }
         const DenseMatrix& D = (wall == 0) ? dshape_bottom_ : dshape_top_;
         const IntegrationRule& ir = lines_[wall == 0 ? 0 : k];
         D.Mult(vx, ua.GetData());
         double acc = 0.0;
         for (int q = 0; q < npts; ++q) { acc += ir.IntPoint(q).weight * ua(q); }
         r[static_cast<std::size_t>(ns) * kQuantities + wall] +=
            acc * el.face_weight / el.dy;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, r.data(), static_cast<int>(r.size()), MPI_DOUBLE,
                 MPI_SUM, V_.GetComm());
   return r;
}

void ChannelStatistics::Sample(const ParGridFunction& u, double t)
{
   if (t < start_ - 1e-12 * std::max(1.0, std::abs(start_))) { return; }
   const std::vector<double> q = PlaneAverages(u);
   if (have_prev_)
   {
      const double dt = t - t_prev_;
      MFEM_VERIFY(dt >= 0.0, "channel_statistics: samples must not go back in "
                  "time");
      for (std::size_t i = 0; i < q.size(); ++i)
      {
         sums_[i] += 0.5 * (prev_[i] + q[i]) * dt;
      }
      time_ += dt;
   }
   prev_ = q;
   t_prev_ = t;
   have_prev_ = true;
   ++samples_;
}

namespace
{
// The time averages, or the last sample's plane averages before any
// averaging time has accumulated.
std::vector<double> MeanOf(const std::vector<double>& sums,
                           const std::vector<double>& prev, double time)
{
   if (time <= 0.0) { return prev; }
   std::vector<double> m(sums.size());
   for (std::size_t i = 0; i < sums.size(); ++i) { m[i] = sums[i] / time; }
   return m;
}
} // namespace

ChannelProfiles ChannelStatistics::Averages() const
{
   MFEM_VERIFY(samples_ > 0, "channel_statistics: no samples yet");
   const std::vector<double> m = MeanOf(sums_, prev_, time_);
   ChannelProfiles p;
   p.y = y_;
   const std::size_t ns = y_.size();
   for (std::size_t s = 0; s < ns; ++s)
   {
      const double* q = &m[s * kQuantities];
      p.U.push_back(q[0]);
      p.V.push_back(q[1]);
      p.W.push_back(q[2]);
      p.uu.push_back(q[3] - q[0] * q[0]);
      p.vv.push_back(q[4] - q[1] * q[1]);
      p.ww.push_back(q[5] - q[2] * q[2]);
      p.uv.push_back(q[6] - q[0] * q[1]);
   }
   return p;
}

double ChannelStatistics::TauWall() const
{
   MFEM_VERIFY(samples_ > 0, "channel_statistics: no samples yet");
   const std::vector<double> m = MeanOf(sums_, prev_, time_);
   const std::size_t w = y_.size() * kQuantities;
   return 0.5 * nu_ * (m[w] - m[w + 1]); // dU/dy < 0 at the top wall
}

double ChannelStatistics::TauWallNow() const
{
   MFEM_VERIFY(samples_ > 0, "channel_statistics: no samples yet");
   const std::size_t w = y_.size() * kQuantities;
   return 0.5 * nu_ * (prev_[w] - prev_[w + 1]);
}

double ChannelStatistics::UTau() const
{
   return std::sqrt(std::max(TauWall(), 0.0));
}

double ChannelStatistics::ReTau() const { return UTau() * delta_ / nu_; }

double ChannelStatistics::BulkVelocity() const
{
   const ChannelProfiles p = Averages();
   double ub = 0.0;
   for (std::size_t s = 0; s < y_.size(); ++s) { ub += y_weight_[s] * p.U[s]; }
   return ub / (y_max_ - y_min_);
}

void ChannelStatistics::WriteCsv(const std::string& path) const
{
   if (V_.GetMyRank() != 0) { return; }
   std::ofstream f(path);
   MFEM_VERIFY(f, "channel_statistics: cannot write " << path);
   f << std::setprecision(10);
   f << "# Channel statistics (incns post/channel_statistics): x-z plane and "
     "time averages\n";
   f << "# nu " << nu_ << "\n# delta " << delta_ << "\n";
   if (samples_ == 0)
   {
      f << "# no samples yet\n";
      return;
   }
   const double u_tau = UTau();
   const double nan = std::numeric_limits<double>::quiet_NaN();
   f << "# averaging_time " << time_ << "\n# samples " << samples_
     << "\n# tau_wall " << TauWall() << "\n# u_tau " << u_tau
     << "\n# re_tau " << ReTau() << "\n# u_bulk " << BulkVelocity() << "\n";
   f << "y,y_plus,U,V,W,uu,vv,ww,uv,U_plus,urms_plus,vrms_plus,wrms_plus,"
     "uv_plus\n";
   const ChannelProfiles p = Averages();
   for (std::size_t s = 0; s < p.y.size(); ++s)
   {
      const double wall = std::min(p.y[s] - y_min_, y_max_ - p.y[s]);
      const bool unit = u_tau > 0.0;
      const double u2 = u_tau * u_tau;
      f << p.y[s] << "," << (unit ? wall * u_tau / nu_ : nan) << "," << p.U[s]
        << "," << p.V[s] << "," << p.W[s] << "," << p.uu[s] << "," << p.vv[s]
        << "," << p.ww[s] << "," << p.uv[s] << ","
        << (unit ? p.U[s] / u_tau : nan) << ","
        << (unit ? std::sqrt(std::max(p.uu[s], 0.0)) / u_tau : nan) << ","
        << (unit ? std::sqrt(std::max(p.vv[s], 0.0)) / u_tau : nan) << ","
        << (unit ? std::sqrt(std::max(p.ww[s], 0.0)) / u_tau : nan) << ","
        << (unit ? p.uv[s] / u2 : nan) << "\n";
   }
}

void ChannelStatistics::Save(const std::string& path) const
{
   if (V_.GetMyRank() != 0) { return; }
   std::ofstream f(path);
   MFEM_VERIFY(f, "channel_statistics: cannot write " << path);
   f << std::setprecision(17); // round-trips every double exactly
   f << "incns_channel_statistics 1\n";
   f << "nu " << nu_ << "\n";
   f << "stations " << y_.size() << "\n";
   for (double y : y_) { f << y << "\n"; }
   f << "values " << prev_.size() << "\n";
   f << "have_prev " << (have_prev_ ? 1 : 0) << " t_prev " << t_prev_ << "\n";
   f << "time " << time_ << " samples " << samples_ << "\n";
   for (double v : prev_) { f << v << "\n"; }
   for (double v : sums_) { f << v << "\n"; }
}

void ChannelStatistics::Load(const std::string& path)
{
   std::ifstream f(path);
   MFEM_VERIFY(f, "channel_statistics: cannot read " << path);
   std::string tag;
   int version = 0;
   f >> tag >> version;
   MFEM_VERIFY(tag == "incns_channel_statistics" && version == 1,
               "channel_statistics: " << path << " is not a statistics state");
   double nu = 0.0;
   std::size_t ns = 0, nval = 0;
   f >> tag >> nu >> tag >> ns;
   MFEM_VERIFY(ns == y_.size(), "channel_statistics: the state in " << path
               << " has " << ns << " stations, this mesh " << y_.size());
   const double tol = 1e-9 * (y_max_ - y_min_);
   for (std::size_t s = 0; s < ns; ++s)
   {
      double y = 0.0;
      f >> y;
      MFEM_VERIFY(std::abs(y - y_[s]) <= tol, "channel_statistics: the state "
                  "in " << path << " belongs to a different mesh (station "
                  << s << " at y = " << y << ", here " << y_[s] << ")");
   }
   MFEM_VERIFY(std::abs(nu - nu_) <= 1e-12 * nu_, "channel_statistics: the "
               "state in " << path << " has nu = " << nu << ", the case " << nu_);
   int have = 0;
   f >> tag >> nval >> tag >> have >> tag >> t_prev_ >> tag >> time_ >> tag
     >> samples_;
   MFEM_VERIFY(nval == prev_.size(), "channel_statistics: bad state in " << path);
   have_prev_ = (have != 0);
   for (double& v : prev_) { f >> v; }
   for (double& v : sums_) { f >> v; }
   MFEM_VERIFY(f, "channel_statistics: " << path << " ends early");
}

} // namespace incns
