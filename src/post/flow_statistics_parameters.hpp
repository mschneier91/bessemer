/**
 * @file flow_statistics_parameters.hpp
 * @brief Settings of the running flow statistics (deck section
 *        `flow_statistics`): when to average, and the sample sets written
 *        to CSV (lines, azimuthally averaged radial profiles, wall bins).
 */
#pragma once

#include <array>
#include <string>
#include <vector>

namespace incns
{

/// A straight line of evenly spaced sample points (flow_statistics.lines).
struct StatisticsLine
{
   /// Set name (in the file name).
   std::string name;
   std::array<double, 3> from = {0.0, 0.0, 0.0}; ///< First point.
   std::array<double, 3> to = {0.0, 0.0, 0.0};   ///< Last point.
   int points = 101; ///< Points, both ends included.
};

/// Radial profiles at axial stations, averaged over the azimuth
/// (flow_statistics.profiles; 3D).
struct StatisticsProfiles
{
   /// Set name (in the file name).
   std::string name;
   /// A point on the axis.
   std::array<double, 3> origin = {0.0, 0.0, 0.0};
   /// Axis direction (any length).
   std::array<double, 3> axis = {0.0, 0.0, 0.0};
   /// Axial coordinates of the stations, (x - origin) . axis / |axis|.
   std::vector<double> stations;
   double radius = 0.0; ///< Outermost radius.
   int points = 21;     ///< Radii 0 .. radius, both included.
   int azimuths = 8;    ///< Equally spaced angles averaged over.
};

/// Area averages over axial bins of a wall (flow_statistics.walls).
struct StatisticsWall
{
   /// Set name (in the file name).
   std::string name;
   /// Boundary names of the wall (box faces, geometry or mesh.boundary_names).
   std::vector<std::string> boundaries;
   /// Their attributes (resolved from the names by Case).
   std::vector<int> attributes;
   /// Where the axial coordinate is 0.
   std::array<double, 3> origin = {0.0, 0.0, 0.0};
   /// Axis direction (any length).
   std::array<double, 3> axis = {0.0, 0.0, 0.0};
   double from = 0.0; ///< Start of the binned range.
   double to = 0.0;   ///< End of the binned range.
   int bins = 100;    ///< Equal bins over [from, to].
};

/// Running time averages at every velocity node (post/flow_statistics) and
/// the sample sets taken from them (post/flow_sampling).
struct FlowStatisticsParameters
{
   bool enabled = false;     ///< Collect them.
   double start_time = 0.0;  ///< Average over t >= start_time (skip transients).
   int interval = 1;         ///< Sample every this many steps.
   /// Write mean_velocity, mean_pressure, reynolds_normal and reynolds_shear
   /// with the ParaView output.
   bool fields = true;
   std::vector<StatisticsLine> lines;        ///< Line sample sets.
   std::vector<StatisticsProfiles> profiles; ///< Radial profile sets (3D).
   std::vector<StatisticsWall> walls;        ///< Wall bin sets.
};

} // namespace incns
