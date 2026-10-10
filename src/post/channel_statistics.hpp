/**
 * @file channel_statistics.hpp
 * @brief Turbulent-channel statistics: time averages of x-z plane-averaged
 *        velocity moments at every velocity-node height, the wall shear
 *        stress, u_tau and Re_tau (the channel flow benchmarks; Moser, Kim &
 *        Mansour 1999 report the same quantities).
 */
#pragma once

#include "mfem.hpp"

#include <string>
#include <vector>

namespace incns
{

/// Time-averaged channel profiles, one entry per station (node height).
struct ChannelProfiles
{
   std::vector<double> y;  ///< Station heights, increasing.
   std::vector<double> U;  ///< Mean streamwise velocity avg(u).
   std::vector<double> V;  ///< Mean wall-normal velocity avg(v).
   std::vector<double> W;  ///< Mean spanwise velocity avg(w) (0 in 2D).
   std::vector<double> uu; ///< Reynolds stress avg(u'u') = avg(uu) - avg(u)^2.
   std::vector<double> vv; ///< avg(v'v').
   std::vector<double> ww; ///< avg(w'w') (0 in 2D).
   std::vector<double> uv; ///< avg(u'v') = avg(uv) - avg(u) avg(v).
};

/**
 * @brief Plane and time averages for a channel with walls normal to y (at the
 *        mesh's y_min and y_max) and homogeneous x (and z).
 *
 * Stations: the heights of the velocity nodes -- the k + 1 Gauss-Lobatto
 * lines of every element layer (k the velocity order), shared lines once.
 * Plane averages at a station: Gauss quadrature (order 2k + 2, exact for the
 * products of the velocity's polynomials) over the x-z face of every element
 * crossing it, summed over ranks, divided by the plane's area. Time averages:
 * the trapezoid rule between consecutive samples with t >= start_time.
 * Second moments are averaged as totals and the fluctuations formed from the
 * combined plane-time averages (avg(u'u') = avg(uu) - avg(u)^2). The wall shear stress
 * is nu times the plane-averaged dU/dy at each wall (sign toward the
 * interior), averaged over the two walls; u_tau = sqrt(tau_w), Re_tau =
 * u_tau delta / nu with delta the half-height.
 *
 * Requires every element to be an axis-aligned box (affine, reference axes
 * along x, y, z) and every station's elements to cover the whole x-z plane:
 * a box mesh, stretched or not, without refinement. The constructor aborts
 * otherwise. The state round-trips exactly through Save()/Load(), so a
 * restarted run continues the same averages.
 */
class ChannelStatistics
{
public:
   /// Plane-averaged quantities per station: u, v, w, uu, vv, ww, uv.
   static constexpr int kQuantities = 7;

   /**
    * @param V          Velocity space (H1, vdim = dim, 2D or 3D); borrowed.
    * @param nu         Kinematic viscosity.
    * @param start_time Samples before it are ignored.
    * @pre Collective; aborts on a mesh it cannot average over (see above).
    */
   ChannelStatistics(mfem::ParFiniteElementSpace& V, double nu,
                     double start_time);

   /**
    * @brief Take a sample (collective); ignored when t < start_time.
    * @param u The velocity.
    * @param t Its time (non-decreasing from sample to sample).
    */
   void Sample(const mfem::ParGridFunction& u, double t);

   /// @return The station heights (increasing).
   const std::vector<double>& Stations() const { return y_; }
   /// @return Time covered by the averages (last sample - first).
   double AveragingTime() const { return time_; }
   /// @return Samples taken (at t >= start_time).
   long Samples() const { return samples_; }
   /// @return The channel's half-height delta.
   double Delta() const { return delta_; }

   /// @return Time-averaged profiles; the last sample's plane averages while
   ///         no averaging time has accumulated. @pre Samples() > 0.
   ChannelProfiles Averages() const;
   /// @return Time-averaged wall shear stress (mean of both walls).
   double TauWall() const;
   /// @return The wall shear stress of the last sample.
   double TauWallNow() const;
   /// @return sqrt(TauWall()) (unit density).
   double UTau() const;
   /// @return UTau() Delta() / nu.
   double ReTau() const;
   /// @return The bulk velocity (1 / 2 delta) int avg(U) dy of the averages.
   double BulkVelocity() const;

   /**
    * @brief Write the profiles as CSV (root): y, y+, U, V, W, uu, vv, ww,
    *        uv, then U+, the rms velocities and uv in wall units (y+ from
    *        the nearer wall); a commented header carries nu, delta, the
    *        averaging time, samples, tau_w, u_tau, Re_tau and U_bulk.
    * @param path Output file.
    */
   void WriteCsv(const std::string& path) const;

   /**
    * @brief Write the averaging state (root, full precision: exact restart).
    * @param path Output file.
    */
   void Save(const std::string& path) const;

   /**
    * @brief Restore a state written by Save() (every rank reads). Aborts if
    *        its stations differ from this mesh's.
    * @param path Input file.
    */
   void Load(const std::string& path);

private:
   /**
    * @brief The plane averages of one velocity field (collective).
    * @param u The velocity.
    * @return kQuantities per station, then the plane-averaged dU/dy at
    *         y_min and at y_max.
    */
   std::vector<double> PlaneAverages(const mfem::ParGridFunction& u) const;

   /// Per local element: geometry and the stations its node lines feed.
   struct ElementLines
   {
      int element = 0;          ///< Local element index.
      double dy = 0.0;          ///< Element height.
      double face_weight = 0.0; ///< x-z face area / plane area.
      bool bottom = false;      ///< On the wall y = y_min.
      bool top = false;         ///< On the wall y = y_max.
      std::vector<int> station; ///< Per node line j: station index, or -1.
   };

   mfem::ParFiniteElementSpace& V_; ///< Velocity space (borrowed).
   double nu_;                      ///< Kinematic viscosity.
   double start_;                   ///< Averaging start time.
   int dim_ = 0;                    ///< Spatial dimension.
   int ndof_ = 0;                   ///< Scalar dofs per element.
   double y_min_ = 0.0;             ///< The lower wall.
   double y_max_ = 0.0;             ///< The upper wall.
   double delta_ = 0.0;             ///< Half-height.
   std::vector<double> y_;          ///< Station heights.
   std::vector<double> y_weight_;   ///< Quadrature weights of int dy.
   std::vector<ElementLines> elements_; ///< Local elements.
   /// Per node line j: the face quadrature points at eta_j.
   std::vector<mfem::IntegrationRule> lines_;
   /// Per node line j: shape values (points x dofs).
   std::vector<mfem::DenseMatrix> shape_;
   mfem::DenseMatrix dshape_bottom_; ///< d/deta of the shapes at eta = 0.
   mfem::DenseMatrix dshape_top_;    ///< d/deta of the shapes at eta = 1.

   bool have_prev_ = false;         ///< A sample was taken.
   double t_prev_ = 0.0;            ///< Its time.
   std::vector<double> prev_;       ///< Its plane averages.
   std::vector<double> sums_;       ///< Time integrals of the plane averages.
   double time_ = 0.0;              ///< Averaging time.
   long samples_ = 0;               ///< Samples taken.
};

} // namespace incns
