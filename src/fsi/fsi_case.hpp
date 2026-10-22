/**
 * @file fsi_case.hpp
 * @brief A deck-driven fluid-structure interaction run (section `fsi`): the
 *        ALE flow and the linear solid, coupled by partitioned
 *        Dirichlet-Neumann subiterations with dynamic Aitken relaxation in
 *        every time step (docs/fsi.md).
 */
#pragma once

#include "fsi/interface_accelerator.hpp"
#include "fsi/interface_transfer.hpp"
#include "solid/linear_elasticity.hpp"
#include "solver/case.hpp"

#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @brief The flow of the deck (Case, on a moving mesh whose `interface`
 *        groups follow the solid) and a linear solid on the solid part of
 *        the deck's geometry (`mesh.cylinder_channel.part: solid`),
 *        coupled in every time step.
 *
 * A step: predict the interface displacement d from the committed steps
 * (fsi.predictor), then iterate: the fluid takes a trial step with its
 * interface at d (u = w there); the fluid's traction (the momentum residual
 * at the interface dofs, recovered as a traction field by the interface's
 * mass matrix and scaled by fsi.fluid_density) loads the solid, which steps
 * from the start of the step (or solves statically); its interface
 * displacement d~ gives the residual r = d~ - d. Converged when |r| <=
 * rtol |d~| + atol: the fluid's trial step is committed. Otherwise the fluid
 * step is undone and d <- d + omega r with Aitken's omega.
 *
 * The solid is the small-strain one (the traction, a Cauchy traction on the
 * moved interface, is applied on the reference interface); St.
 * Venant-Kirchhoff takes its place when it lands.
 */
class FsiCase
{
public:
   /// One coupled step.
   struct StepRecord
   {
      double time = 0.0;     ///< Time at the end of the step.
      int iterations = 0;    ///< Subiterations.
      double residual = 0.0; ///< |r| / max(|d~|, tiny) at the last one.
      double omega = 0.0;    ///< The last relaxation factor used.
   };

   /**
    * @param fluid_mesh The deck's (fluid) mesh (borrowed; must outlive the
    *                   case).
    * @param params     Normalized parameters with fsi.enabled.
    */
   FsiCase(mfem::ParMesh& fluid_mesh, const Parameters& params);

   ~FsiCase();

   /// One coupled step. @return False when the subiterations did not
   /// converge in fsi.max_iterations or the fluid diverged.
   bool Step();

   /// March to time.t_final. @return Whether every step converged.
   bool Run();

   /**
    * @brief Write the run summary (JSON) on rank 0. Collective.
    * @param path   File.
    * @param status running, ok or diverged.
    */
   void WriteSummary(const std::string& path, const std::string& status);

   /// @return The flow.
   Case& Fluid() { return *fluid_; }

   /// @return The solid.
   LinearElasticity& Solid() { return *solid_; }

   /// @return The solid's mesh.
   mfem::ParMesh& SolidMesh() { return *solid_mesh_; }

   /// @return The committed steps.
   const std::vector<StepRecord>& History() const { return history_; }

   /// @return The fluid's traction on the interface of the fluid's last
   ///         step, interpolated to the solid's space (rho_f times the
   ///         kinematic traction; UpdateTraction()).
   const mfem::ParGridFunction& SolidTraction() const { return *solid_traction_; }

   /// Recompute SolidTraction() from the fluid's last step. Collective.
   void UpdateTraction();

   /// @return The total force of SolidTraction() over the fluid's moved
   ///         interface (rho_f times the fluid's force on the interface).
   mfem::Vector InterfaceForce() const { return interface_force_; }

private:
   /// @param d The solid's displacement interpolated to the fluid
   ///          interface's nodes (the motion space on the reference mesh).
   void SolidInterfaceDisplacement(mfem::ParGridFunction& d);

   /// @param r The step to append to the coupling log. Collective (the
   ///          probe); rank 0 writes.
   void LogStep(const StepRecord& r);

   /// @param d The predicted interface displacement (true dofs of the
   ///          motion space).
   void Predict(mfem::Vector& d) const;

   mfem::ParMesh& fluid_mesh_; ///< The fluid mesh (borrowed).
   Parameters params_;         ///< Parameters (copied).
   Parameters solid_params_;   ///< The solid's: the geometry's solid part.
   std::unique_ptr<mfem::ParMesh> solid_mesh_; ///< The solid's mesh.
   std::unique_ptr<Case> fluid_;               ///< The flow.
   std::unique_ptr<mfem::VectorCoefficient> u0_; ///< Its initial velocity.
   std::unique_ptr<LinearElasticity> solid_;   ///< The solid.
   bool dynamic_ = false;      ///< Whether the solid is dynamic.
   std::vector<int> fluid_attrs_; ///< Fluid mesh interface attributes.
   std::vector<int> solid_attrs_; ///< Solid mesh interface attributes.

   /// The fluid's velocity space on the motion's reference mesh (the
   /// traction's data, interpolated in the reference configuration).
   std::unique_ptr<mfem::ParFiniteElementSpace> ref_vfes_;
   std::unique_ptr<mfem::ParGridFunction> ref_traction_; ///< On ref_vfes_.
   std::unique_ptr<mfem::ParGridFunction> solid_traction_; ///< On the solid.
   /// The solid's traction load (reads solid_traction_).
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> traction_coeff_;
   std::unique_ptr<InterfaceTransfer> to_solid_; ///< Fluid -> solid map.
   std::unique_ptr<InterfaceTransfer> to_fluid_; ///< Solid -> fluid map.
   mfem::Array<int> fluid_marker_; ///< Interface marker of the fluid mesh.
   mfem::Vector interface_force_;  ///< See InterfaceForce().

   std::unique_ptr<InterfaceAccelerator> accel_; ///< The relaxation.
   /// Committed interface displacements (motion true dofs), newest first.
   std::vector<mfem::Vector> committed_;
   std::vector<StepRecord> history_; ///< See History().
   long long total_iterations_ = 0;  ///< Subiterations over the run.
   int max_iterations_ = 0;          ///< Most subiterations in a step.
   double wall_ = 0.0;               ///< Wall time of the march.
   bool log_started_ = false;        ///< Coupling log header written.
   std::string failure_;             ///< Why a step failed (empty if none).
};

} // namespace incns
