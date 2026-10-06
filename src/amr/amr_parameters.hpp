/**
 * @file amr_parameters.hpp
 * @brief Settings for refinement-only adaptive mesh refinement (deck section
 *        `amr:`); see amr_spec.md.
 */
#pragma once

namespace incns
{

/// How elements are selected for refinement.
enum class AmrThreshold
{
   /// Mark K when eta_K >= theta * max_K eta_K (global max). Always marks
   /// something unless the velocity is constant: suits tests and studies.
   Relative,
   /// Mark K when eta_K >= tolerance (nondimensional velocity units). Stops
   /// refining once the flow is resolved: suits production runs.
   Absolute
};

/**
 * @brief Adaptive mesh refinement settings. Refinement only: elements are never
 *        coarsened, so the mesh grows monotonically (bounded by max_elements).
 *
 * The indicator is the directional velocity gradient G_{K,d}: the RMS change of
 * the velocity across element K along its d-th reference direction, and
 * eta_K = |G_K| (see amr/gradient_indicator.hpp).
 */
struct AmrParameters
{
   /// Turn AMR on. Makes the mesh nonconforming-ready at construction (no
   /// numerical change until the first refinement).
   bool enabled = false;
   /// Accepted steps between adaptation events; 0 = no events during the run.
   int interval = 50;
   /// Refinement passes on the initial condition before the first step (the
   /// analytic initial condition is re-projected after each pass).
   int initial_passes = 0;
   /// Refinement passes per adaptation event during the run.
   int passes_per_event = 1;
   /// Split each marked element only in the directions the velocity varies
   /// in (true), or in every direction (false).
   bool anisotropic = false;
   /// Anisotropic mode: split direction d of a marked element when
   /// G_{K,d} >= aniso_ratio * max_d' G_{K,d'}. 1 = dominant direction only.
   double aniso_ratio = 0.5;
   /// Marking rule (relative to the maximum, or absolute).
   AmrThreshold threshold_mode = AmrThreshold::Relative;
   /// Relative mode: mark eta_K >= theta * max eta.
   double theta = 0.5;
   /// Absolute mode: mark eta_K >= tolerance (nondimensional velocity units).
   double tolerance = 0.05;
   /// Do not split a direction whose element extent is below this
   /// (nondimensional length); 0 = no limit.
   double min_size = 0.0;
   /// Global element cap; marking keeps the largest indicators so the
   /// refined mesh stays within it. 0 = no cap.
   long long max_elements = 0;
   /// Maximum refinement-level difference across a face (1 = 2:1 balance).
   int nc_limit = 1;
   /// Re-partition after each event (np > 1).
   bool rebalance = true;
   /// Replace each transferred velocity history level by its M-orthogonal
   /// projection onto the discretely divergence-free subspace of the refined
   /// mesh (avoids a pressure transient after an event).
   bool project_history = true;
   /// Also write the indicator and refinement levels as ParaView cell data.
   bool write_indicator = false;

   /// Abort (MFEM_VERIFY) on out-of-range settings.
   void Validate() const;
};

} // namespace incns
