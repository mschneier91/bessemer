/**
 * @file flow_sampling.hpp
 * @brief The sample sets of the running flow statistics: lines, azimuthally
 *        averaged radial profiles and wall bins, evaluated from the nodal
 *        averages at output time only, written as CSV, and the half-window
 *        comparison that says whether the averages have converged.
 */
#pragma once

#include "post/flow_statistics.hpp"
#include "post/flow_statistics_parameters.hpp"
#include "mfem.hpp"

#include <string>
#include <vector>

namespace incns
{

/**
 * @brief The raw values of every sample set (rank 0; empty elsewhere).
 *
 * Per point of a line or profile set, Cartesian and in this order: the mean
 * velocity (dim), the mean pressure, the Reynolds stresses R_ij = <u_i' u_j'>
 * and the mean flow's viscous stress tau_ij = nu (dU_i/dx_j + dU_j/dx_i),
 * each symmetric tensor as xx, yy(, zz), xy(, xz, yz). NaN at a point outside
 * the mesh. Per bin of a wall set: its area, then the area averages of the
 * mean pressure, the axial and (3D) cross wall shear stress; NaN but the
 * area for an empty bin.
 */
struct SampleValues
{
   std::vector<std::vector<double>> points; ///< Per line, then profile set.
   std::vector<std::vector<double>> walls;  ///< Per wall set.
   /// Per wall set: wall points skipped (normal along the axis).
   std::vector<long long> wall_skipped;
};

/// A sample set as written: named columns, rows of values (rank 0).
struct SampleTable
{
   std::string name;                  ///< The set's name.
   std::string kind;                  ///< "line", "profile" or "wall".
   std::vector<std::string> columns;  ///< Column names.
   std::vector<double> values;        ///< Rows x columns, row-major.
   std::vector<std::string> header;   ///< The set's "# key value" lines.
   /// Column groups the half-window comparison reports: name, columns.
   std::vector<std::pair<std::string, std::vector<int>>> groups;
   /// @return Rows.
   int Rows() const
   {
      return columns.empty() ? 0 : static_cast<int>(values.size() / columns.size());
   }
};

/// One entry of the half-window comparison.
struct HalfWindowDifference
{
   std::string set;   ///< Sample set.
   /// velocity, pressure, reynolds_stress, viscous_stress or wss.
   std::string group;
   /// max |first half - second half| over the set's rows and the group's
   /// columns, over the group's scale in the whole window (max |value|; the
   /// pressure's range for the pressure). NaN without a scale or values.
   double relative = 0.0;
};

/**
 * @brief Evaluates the sample sets of FlowStatisticsParameters from a
 *        FlowStatistics state, writes them as CSV and compares the two halves
 *        of the averaging window.
 *
 * Construction locates the line and profile points once (MFEM's
 * FindPointsGSLIB, on a copy of the mesh when it has no nodes) and keeps only
 * each rank's own points (element, reference coordinates); gslib's data is
 * freed right after. Evaluate() is then local work on the owning ranks (the
 * fields and the mean velocity gradient at each point; the wall faces'
 * quadrature points binned by their axial coordinate) and one reduction to
 * rank 0. Nothing is done between outputs: the per-step cost of the
 * statistics is FlowStatistics::Sample() alone.
 *
 * Profiles: at station s, radius r and azimuth theta the point
 * origin + s e_a + r (cos theta b1 + sin theta b2), with e_a the unit axis,
 * b1 the coordinate axis least aligned with it made perpendicular, and b2 =
 * e_a x b1; the vectors and tensors in the (a, r, t) frame of each point,
 * then averaged over the found angles (column n counts them).
 *
 * Walls: each quadrature point of the wall's faces (order 2 k_u + 2) goes to
 * the bin of its axial coordinate, so a bin is an exact area integral when
 * its edges lie on element edges, and an approximate one otherwise (bins
 * narrower than the wall elements get few points). The wall shear stress is
 * the tangential part of tau n, n the unit normal into the fluid; wss_axial
 * is its component along the axis's projection on the wall (positive where
 * the near-wall flow goes along the axis), wss_cross (3D) along n x that
 * direction. Wall points whose normal is
 * within 0.06 degrees of the axis (a step facing the flow) have no axial
 * direction and are skipped; the CSV header counts them.
 *
 * Half-window comparison: Snapshot() keeps the raw values of every set at the
 * split (rank 0; a few numbers per point). With W1 the weight there and W at
 * the end, the second part's means are (W x - W1 x1) / (W - W1), exactly
 * (point values and bin averages are linear in the nodal means), and its
 * Reynolds stresses follow from the parallel-axis rule at each point,
 * W R = W1 R1 + W2 R2 + (W1 W2 / W) d d^T with d = U2 - U1 (exact up to the
 * interpolation of d d^T between nodes, a term ~ 1/N of the stresses).
 */
class FlowSampler
{
public:
   /**
    * @param mesh    The mesh (borrowed; fixed).
    * @param params  The sample sets (walls with their attributes resolved).
    * @param nu      Kinematic viscosity (the viscous stress).
    * @param order_u Velocity order (the wall quadrature).
    * @pre Collective.
    */
   FlowSampler(mfem::ParMesh& mesh, const FlowStatisticsParameters& params,
               double nu, int order_u);

   /**
    * @brief The raw values of every set from @p stats (collective).
    * @param stats The averages.
    * @return The values on rank 0; empty vectors elsewhere.
    */
   SampleValues Evaluate(const FlowStatistics& stats) const;

   /**
    * @brief The tables of @p v (rank 0): lines Cartesian, profiles in
    *        cylindrical components averaged over the azimuth, walls per bin.
    * @param v Values from Evaluate() or SecondPart().
    * @return One table per set, lines then profiles then walls.
    */
   std::vector<SampleTable> Tabulate(const SampleValues& v) const;

   /**
    * @brief Take the half-window split from @p stats (collective).
    * @param stats The averages at the split.
    */
   void Snapshot(const FlowStatistics& stats);
   /// @return Whether the split has been taken (the same on every rank).
   bool HasSplit() const { return split_taken_; }
   /// @return The split's time (its last sample).
   double SplitTime() const { return split_time_; }
   /// @return The averaging time up to the split, W1.
   double SplitWeight() const { return split_weight_; }

   /**
    * @brief The values of the averages after the split (rank 0).
    * @param all    Values over the whole window (Evaluate()).
    * @param weight The whole window's averaging time W (> SplitWeight()).
    * @return The second part's values; sets without a split value NaN.
    */
   SampleValues SecondPart(const SampleValues& all, double weight) const;
   /// @return The values at the split (rank 0; NaN for sets without one).
   const SampleValues& FirstPart() const { return split_; }

   /**
    * @brief Compare the halves (rank 0).
    * @param all    Values over the whole window.
    * @param weight Its averaging time.
    * @return Per set and group the relative difference; empty without a
    *         split or a second part.
    */
   std::vector<HalfWindowDifference>
   Compare(const SampleValues& all, double weight) const;

   /**
    * @brief Write every set as `<prefix>_<set>.csv` (rank 0), and with a
    *        split also `_half1.csv` and `_half2.csv`.
    * @param all   Values over the whole window (Evaluate()).
    * @param stats The averages (window and weight in the header).
    * @param prefix Path prefix, e.g. <path>/<name>_stats.
    */
   void WriteCsv(const SampleValues& all, const FlowStatistics& stats,
                 const std::string& prefix) const;

   /**
    * @brief Write the split to `<dir>/flow_statistics_split.txt` (rank 0).
    * @param dir Checkpoint directory.
    */
   void Save(const std::string& dir) const;
   /**
    * @brief Restore a split written by Save() (collective): sets whose name,
    *        kind and size still match keep their split values; others lose
    *        them (said once on rank 0). No file: no split.
    * @param dir Checkpoint directory.
    */
   void Load(const std::string& dir);

   /// @return Line and profile points outside the mesh (rank 0's count).
   int PointsNotFound() const { return not_found_; }
   /// @return Values per point (dim + 1 + 2 x stress components).
   int PointWidth() const { return dim_ + 1 + 2 * nsym_; }
   /// Values per wall bin (area, p, wss_axial, wss_cross).
   static constexpr int kWallWidth = 4;

private:
   /// A located point this rank evaluates.
   struct OwnedPoint
   {
      int index = 0;                 ///< Global point index.
      int element = 0;               ///< Local element.
      double ref[3] = {0.0, 0.0, 0.0}; ///< Reference coordinates in [0, 1].
   };
   /// A line or profile set's place in the global point list.
   struct PointSet
   {
      std::string name;   ///< Set name.
      std::string kind;   ///< "line" or "profile".
      int index = 0;      ///< Position in params_.lines or params_.profiles.
      int first = 0;      ///< First global point.
      int count = 0;      ///< Points.
   };

   /**
    * @param a A tensor index.
    * @param b A tensor index.
    * @return The symmetric component's slot: xx, yy(, zz), xy(, xz, yz).
    */
   int Sym(int a, int b) const;
   /// @brief The points of every line and profile set into coords_ (rank 0).
   void Points();
   /// @brief Locate the points and keep this rank's (collective).
   void Locate();
   /**
    * @brief The profile frame at an azimuth.
    * @param s     Profile set.
    * @param theta Azimuth.
    * @param e     Out: e[0] axial, e[1] radial, e[2] azimuthal unit vectors.
    */
   void Frame(const StatisticsProfiles& s, double theta, double e[3][3]) const;
   /**
    * @brief Fill a line set's table.
    * @param v   The set's point values.
    * @param set The line.
    * @param t   The table.
    */
   void LineTable(const std::vector<double>& v, const StatisticsLine& set,
                  SampleTable& t) const;
   /**
    * @brief Fill a profile set's table (cylindrical, azimuthally averaged).
    * @param v   The set's point values.
    * @param set The profiles.
    * @param t   The table.
    */
   void ProfileTable(const std::vector<double>& v,
                     const StatisticsProfiles& set, SampleTable& t) const;
   /**
    * @brief Fill a wall set's table.
    * @param v   The set's bin values.
    * @param set The wall.
    * @param t   The table.
    */
   void WallTable(const std::vector<double>& v, const StatisticsWall& set,
                  SampleTable& t) const;
   /**
    * @brief The wall bins' sums of one wall set on this rank.
    * @param stats The averages.
    * @param set   Wall set.
    * @param sums  Out: bins x (area, p, wss_axial, wss_cross) area-weighted,
    *              then the skipped points' count.
    */
   void WallSums(const FlowStatistics& stats, const StatisticsWall& set,
                 std::vector<double>& sums) const;
   /**
    * @brief Write tables as CSV files (rank 0).
    * @param tables The tables.
    * @param prefix Path prefix.
    * @param suffix File name suffix ("" or "_half1" / "_half2").
    * @param header Comment lines common to all files.
    */
   void WriteTables(const std::vector<SampleTable>& tables,
                    const std::string& prefix, const std::string& suffix,
                    const std::vector<std::string>& header) const;

   mfem::ParMesh& mesh_;                  ///< The mesh (borrowed).
   FlowStatisticsParameters params_;      ///< The sets.
   double nu_;                            ///< Kinematic viscosity.
   int order_;                            ///< Velocity order.
   int dim_ = 0;                          ///< Spatial dimension.
   int nsym_ = 0;                         ///< Tensor components (3, 6).
   bool root_ = false;                    ///< This is rank 0.
   std::vector<PointSet> sets_;           ///< Line, then profile sets.
   int total_ = 0;                        ///< Points of all sets.
   std::vector<OwnedPoint> owned_;        ///< This rank's points.
   std::vector<char> found_;              ///< Per point: inside (rank 0).
   int not_found_ = 0;                    ///< Points outside (rank 0).
   std::vector<double> coords_;           ///< The points, by vdim (rank 0).
   bool split_taken_ = false;             ///< The split exists.
   double split_time_ = 0.0;              ///< Its time.
   double split_weight_ = 0.0;            ///< W1.
   SampleValues split_;                   ///< Values at the split (rank 0).
};

} // namespace incns
