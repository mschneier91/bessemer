/**
 * @file flow_statistics.hpp
 * @brief Running time averages at every velocity and pressure node: the mean
 *        velocity, the mean pressure and the Reynolds stresses, updated in
 *        place by one device pass per sample (no time history is kept).
 */
#pragma once

#include "mfem.hpp"

#include <string>

namespace incns
{

/**
 * @brief Time-weighted running statistics of the velocity and the pressure
 *        at their nodes.
 *
 * A sample at time t has the weight w = t - t_prev, the time since the
 * previous sample: the right-endpoint rule in time, equal weights at a fixed
 * dt. The first sample (weight 0) starts the averages at its values; a
 * sample at the previous sample's time, or before start_time, is ignored.
 * With W the weight so far and W' = W + w, every node updates in place by the
 * weighted Welford update (West 1979; UpdateRunningMoments):
 *   d = u - U,   U <- U + (w / W') d,   R <- (W / W') (R + (w / W') d d^T),
 *   P <- P + (w / W') (p - P),
 * so U, P and R = <u'u'> are at every moment the averages over the samples
 * so far, without a sum of squares to cancel. Cost per sample: one pass over
 * the velocity nodes (read u, update 3 + 6 values per 3D node) and one over
 * the pressure nodes; nothing is reduced or communicated, so the result is
 * bitwise deterministic. Shared nodes are updated from the same values on
 * every rank that holds them, so the fields stay consistent L-vectors that
 * ParaView and point evaluation read directly.
 *
 * Memory: three velocity-space fields (the mean velocity and the Reynolds
 * stresses as two vectors: the normal stresses and the shear stresses) and
 * one pressure-space field. The normal-stress field holds (u'u', v'v', w'w'),
 * the shear-stress field (u'v', u'w', v'w') in 3D and (u'v', 0) in 2D.
 *
 * Needs a fixed mesh: the averages are per node, so moving or adapting the
 * mesh would mix different points. Save()/Load() round-trip exactly
 * (per-rank binary files and a metadata file), so a restarted run continues
 * the same averages bit for bit.
 */
class FlowStatistics
{
public:
   /**
    * @param V          Velocity space (vdim = dim); borrowed.
    * @param Q          Pressure space; borrowed.
    * @param start_time Samples before it are ignored.
    */
   FlowStatistics(mfem::ParFiniteElementSpace& V,
                  mfem::ParFiniteElementSpace& Q, double start_time);

   /**
    * @brief Take a sample (no communication).
    * @param u The velocity (on V).
    * @param p The pressure (on Q).
    * @param t Their time (non-decreasing from sample to sample).
    * @return Whether the sample was taken (false before start_time or at
    *         the previous sample's time).
    */
   bool Sample(const mfem::ParGridFunction& u, const mfem::ParGridFunction& p,
               double t);

   /// @return The mean velocity.
   mfem::ParGridFunction& MeanVelocity() { return mean_u_; }
   /// @return The mean velocity.
   const mfem::ParGridFunction& MeanVelocity() const { return mean_u_; }
   /// @return The mean pressure.
   mfem::ParGridFunction& MeanPressure() { return mean_p_; }
   /// @return The mean pressure.
   const mfem::ParGridFunction& MeanPressure() const { return mean_p_; }
   /// @return The normal Reynolds stresses (u'u', v'v'(, w'w')) on V.
   mfem::ParGridFunction& NormalStress() { return normal_; }
   /// @return The normal Reynolds stresses (u'u', v'v'(, w'w')) on V.
   const mfem::ParGridFunction& NormalStress() const { return normal_; }
   /// @return The shear Reynolds stresses (u'v', u'w', v'w'; 2D: u'v', 0) on V.
   mfem::ParGridFunction& ShearStress() { return shear_; }
   /// @return The shear Reynolds stresses (u'v', u'w', v'w'; 2D: u'v', 0) on V.
   const mfem::ParGridFunction& ShearStress() const { return shear_; }

   /// @return Samples taken (at t >= start_time).
   long Samples() const { return samples_; }
   /// @return The averaging time W: the last sample's time - the first's.
   double AveragingTime() const { return weight_; }
   /// @return The first sample's time (0 before any).
   double FirstTime() const { return t_first_; }
   /// @return The last sample's time (0 before any).
   double LastTime() const { return t_prev_; }
   /// @return The averaging start time.
   double StartTime() const { return start_; }

   /**
    * @brief Write the state to @p dir (collective; the directory must
    *        exist): per-rank binary fields and rank 0's flow_statistics.txt.
    * @param dir Checkpoint directory.
    */
   void Save(const std::string& dir) const;

   /**
    * @brief Restore a state written by Save() at the same rank count
    *        (collective). Aborts on a size mismatch.
    * @param dir Checkpoint directory.
    */
   void Load(const std::string& dir);

   /**
    * @param dir A checkpoint directory.
    * @return Whether it holds a statistics state.
    */
   static bool Saved(const std::string& dir);

private:
   mfem::ParGridFunction mean_u_; ///< Mean velocity.
   mfem::ParGridFunction mean_p_; ///< Mean pressure.
   mfem::ParGridFunction normal_; ///< Normal Reynolds stresses.
   mfem::ParGridFunction shear_;  ///< Shear Reynolds stresses.
   double start_;                 ///< Averaging start time.
   int dim_ = 0;                  ///< Spatial dimension.
   long samples_ = 0;             ///< Samples taken.
   double weight_ = 0.0;          ///< Accumulated weight (time).
   double t_first_ = 0.0;         ///< First sample's time.
   double t_prev_ = 0.0;          ///< Last sample's time.
};

} // namespace incns
